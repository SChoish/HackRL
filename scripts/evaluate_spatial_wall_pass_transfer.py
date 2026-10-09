#!/usr/bin/env python3
"""Exact geometry audit and read-only frozen-policy transfer evaluation."""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from hackrl.spatial_wall_pass import (
    SpatialWallPassSplit, SpatialWallPassVariant, make_spatial_wall_pass_state,
    spatial_wall_pass_step_with_transition,
)
from hackrl.spatial_wall_pass_gc import (
    SpatialWallPassGCActorCritic, evaluate_spatial_wall_pass_gc_frozen,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "docs/manifests/spatial_wall_pass_frozen_transfer_v1.json"
SOURCE = Path("/raid/ext_csv/HackRL/runs/spatial_wall_pass_dual_replication_v1")
ACTIONS = (1, 2, 3, 4, 17, 18, 19, 20, 21, 22)
DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def floor_cells(contract):
    f = contract["floor"]
    return {(r, c) for r in range(f["row_min"], f["row_max"] + 1)
            for c in range(f["column_min"], f["column_max"] + 1)
            if [r, c] not in f["removed"]}


def make_state(contract, fixture, index=0):
    walls = {tuple(p) for p in fixture["walls"]}
    floor = floor_cells(contract)
    assert walls <= floor
    assert all(tuple(fixture[k]) in floor - walls for k in ("camp", "start", "item"))
    assert len({tuple(fixture[k]) for k in ("camp", "start", "item")}) == 3
    walkable = np.zeros((16, 16), dtype=bool)
    wall_mask = np.zeros((16, 16), dtype=bool)
    for pos in floor - walls:
        walkable[pos] = True
    for pos in walls:
        wall_mask[pos] = True
    return make_spatial_wall_pass_state(jnp.int32(0)).replace(
        walkable=jnp.asarray(walkable), wall_mask=jnp.asarray(wall_mask),
        camp_position=jnp.asarray(fixture["camp"], dtype=jnp.int32),
        player_position=jnp.asarray(fixture["start"], dtype=jnp.int32),
        item_position=jnp.asarray(fixture["item"], dtype=jnp.int32),
        layout_index=jnp.int32(index),
    )


def physical_step(contract, fixture, node, action, kernel):
    """Independent finite model: position, task phase, any wall crossing."""
    row, col, phase, crossed = node
    camp, item = tuple(fixture["camp"]), tuple(fixture["item"])
    walls = {tuple(p) for p in fixture["walls"]}
    floor = floor_cells(contract)
    pos = (row, col)
    dash = 17 <= action <= 20
    if action in DELTAS or dash:
        dr, dc = DELTAS[action - 16 if dash else action]
        mid = (row + dr, col + dc)
        dst = (row + (2 if dash else 1) * dr, col + (2 if dash else 1) * dc)
        allowed = dst in floor - walls and dst != camp
        if dash:
            allowed &= mid in floor and mid != camp
            allowed &= kernel == "mutant" or mid not in walls
        if allowed:
            pos = dst
            crossed = crossed or (dash and mid in walls)
    if action == 21 and phase == 0 and pos == item:
        phase = 1
    if action == 22 and phase == 1 and sum(abs(a-b) for a,b in zip(pos,camp)) == 1:
        phase = 2
    return (*pos, phase, bool(crossed))


def shortest_paths(contract, fixture, kernel):
    start = (*fixture["start"], 0, False)
    queue = deque([start])
    paths = {start: []}
    terminals = {}
    while queue:
        node = queue.popleft()
        if node[2] == 2:
            terminals.setdefault("with_wall_pass" if node[3] else "without_wall_pass", paths[node])
            continue
        for action in ACTIONS:
            nxt = physical_step(contract, fixture, node, action, kernel)
            if nxt not in paths:
                paths[nxt] = paths[node] + [action]
                queue.append(nxt)
    options = list(terminals.values())
    assert options, (fixture["id"], kernel, "unreachable")
    best = min(options, key=len)
    return {"shortest_length": len(best), "shortest_actions": best,
            **{key: {"length": len(value), "actions": value}
               for key, value in terminals.items()}}, paths


def audit_geometry(contract):
    reports = []
    for index, fixture in enumerate(contract["fixtures"]):
        initial = make_state(contract, fixture, index)
        # Exclude every translated/reflected/rotated copy of the old fixture:
        # normalize all role positions and floor/wall cells by player position.
        def signature(state):
            pos = np.asarray(state.player_position)
            return (tuple(map(tuple, np.argwhere(np.asarray(state.walkable)) - pos)),
                    tuple(map(tuple, np.argwhere(np.asarray(state.wall_mask)) - pos)),
                    tuple(np.asarray(state.camp_position)-pos),
                    tuple(np.asarray(state.item_position)-pos))
        sig = signature(initial)
        for old_index in range(8):
            assert sig != signature(make_spatial_wall_pass_state(jnp.int32(old_index)))
        report = {"id": fixture["id"], "category": fixture["category"]}
        for kernel in contract["kernels"]:
            report[kernel], reached = shortest_paths(contract, fixture, kernel)
            assert report[kernel]["shortest_length"] <= contract["horizon"]
            pairs = [(node, action) for node in reached if node[2] != 2 for action in ACTIONS]
            nodes = np.asarray([node for node, _ in pairs], dtype=np.int32)
            base = jax.tree.map(lambda x: jnp.repeat(x[None], len(pairs), axis=0), initial)
            states = base.replace(
                player_position=jnp.asarray(nodes[:, :2]),
                carrying_item=jnp.asarray(nodes[:, 2] == 1),
                item_present=jnp.asarray(nodes[:, 2] == 0),
                episode_wall_passed=jnp.asarray(nodes[:, 3], dtype=bool),
            )
            after, transition = jax.jit(jax.vmap(
                lambda s, a: spatial_wall_pass_step_with_transition(
                    s, a, SpatialWallPassVariant(kernel))
            ))(states, jnp.asarray([a for _, a in pairs], dtype=jnp.int32))
            actual = np.column_stack((np.asarray(after.player_position),
                np.where(np.asarray(after.delivered_total)>0, 2, np.asarray(after.carrying_item).astype(int)),
                np.asarray(after.episode_wall_passed)))
            expected = np.asarray([physical_step(contract, fixture, n, a, kernel) for n,a in pairs])
            np.testing.assert_array_equal(actual, expected)
            report[kernel]["independently_checked_transitions"] = len(pairs)
            # Replay exact shortest actions in the original kernel, including
            # its delivery and provenance fields, not only in the finite model.
            state = initial
            step = jax.jit(lambda s,a: spatial_wall_pass_step_with_transition(
                s,a,SpatialWallPassVariant(kernel))[0])
            for action in report[kernel]["shortest_actions"]:
                state = step(state, jnp.int32(action))
            assert int(state.delivered_total) == 1
        if fixture["category"] == "wall_pass_detour":
            m = report["mutant"]
            assert m["with_wall_pass"]["length"] > m["without_wall_pass"]["length"]
            assert m["shortest_length"] == report["fixed"]["shortest_length"]
        reports.append(report)
    return reports


def compact(episodes):
    success = [e for e in episodes if e["success"]]
    mean = lambda key: float(np.mean([e[key] for e in episodes]))
    return {"episodes": len(episodes), "success_rate": mean("success"),
            "wall_pass_rate": mean("wall_pass"),
            "delivery_after_pass_rate": mean("beneficial_use"),
            "mean_discounted_return": mean("discounted_return"),
            "mean_length": mean("length"),
            "mean_success_length": float(np.mean([e["length"] for e in success])) if success else None}


def evaluate(contract, geometry):
    if os.environ.get("HACKRL_DEVICE") != "cuda" or not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("Use the GPU evaluation entrypoint")
    source_manifest = json.loads((SOURCE / "run_manifest.json").read_text())
    if source_manifest["execution_code_sha"] != contract["training_execution_sha"]:
        raise RuntimeError("Training execution SHA differs from frozen contract")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    if dirty:
        raise RuntimeError("Commit evaluation contract and code before policy evaluation")
    root = Path(contract["storage"]["root"])
    if (root / "result.json").exists():
        raise RuntimeError("Completed evaluation exists; do not overwrite")
    root.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        f"{branch}_seed_{seed}": SOURCE / "adaptation" / f"{branch}_seed_{seed}" / "checkpoints/adapt_4096"
        for seed in contract["seeds"] for branch in contract["branches"]}
    for path in checkpoints.values():
        metadata = json.loads((path / "metadata.json").read_text())
        if metadata["global_update"] != 512 + contract["adaptation_update"]:
            raise RuntimeError("Checkpoint is not the preregistered final update")
    hashes = {name: sha(path/"state.msgpack") for name,path in checkpoints.items()}
    provenance = {"contract": contract, "manifest_sha256": sha(MANIFEST),
        "evaluation_code_sha": subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
        "source_checkpoint_sha256": hashes, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_calls": 0, "geometry": geometry,
        "capacity": {p: shutil.disk_usage(p)._asdict() for p in ("/home/ext_csv","/raid/ext_csv")}}
    if provenance["capacity"]["/raid/ext_csv"]["free"] < contract["storage"]["artifact_budget_bytes"] + contract["storage"]["reserve_bytes"]:
        raise RuntimeError("Insufficient RAID evaluation reserve")
    write_json(root/"evaluation_manifest.json", provenance)
    states = [make_state(contract,f,i) for i,f in enumerate(contract["fixtures"])]
    initial = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    original = json.loads((SOURCE/"adaptation/aggregate_result.json").read_text())
    results = {}
    for seed in contract["seeds"]:
        for branch in contract["branches"]:
            name = f"{branch}_seed_{seed}"
            path = checkpoints[name]
            payload = serialization.msgpack_restore((path/"state.msgpack").read_bytes())
            params = jax.tree.map(jnp.asarray, payload["gc"]["train_state"]["params"])
            config = json.loads((path/"config.json").read_text())
            network = SpatialWallPassGCActorCritic(hidden_size=config["policy_hidden_size"])
            cell = {"seed":seed,"branch":branch,"kernels":{},"original_mode_parity":{}}
            old_name = f"dual_{'fixed' if branch=='fixed_continue' else 'mutant'}_seed_{seed}"
            for kernel in contract["kernels"]:
                control = evaluate_spatial_wall_pass_gc_frozen(network,params,
                    variant=kernel, stochastic=False, repeats_per_state=1,
                    seed_base=60000, learner_seed=seed,record_episodes=True)
                old = original["cells"][old_name]["final_evaluation"][kernel]["mode"]
                for key in ("success_rate","wall_pass_rate","beneficial_use_rate","mean_length","mean_discounted_return"):
                    assert np.isclose(control[key],old[key],rtol=0,atol=1e-12), (name,kernel,key)
                cell["original_mode_parity"][kernel] = True
                cell["kernels"][kernel] = {}
                for stochastic, selection in ((False,"mode"),(True,"sample")):
                    result = evaluate_spatial_wall_pass_gc_frozen(network,params,
                        variant=kernel,stochastic=stochastic,
                        repeats_per_state=contract[f"{selection}_repeats"],
                        seed_base=contract["seed_base"][selection],learner_seed=seed,
                        record_episodes=True,initial_states=initial,horizon=contract["horizon"])
                    grouped = {}
                    for i,fixture in enumerate(contract["fixtures"]):
                        episodes = [e for e in result["episode_records"] if e["layout_index"]==i]
                        grouped[fixture["id"]] = compact(episodes)
                    cell["kernels"][kernel][selection] = {"per_fixture":grouped,
                        "episode_records":result["episode_records"]}
            assert sha(path/"state.msgpack") == hashes[name]
            results[name] = cell
            write_json(root/f"{name}.json",cell)
            print(json.dumps({"completed":name,"cells":len(results),"total":len(checkpoints)}),flush=True)
    paired = []
    for name,cell in results.items():
        for fixture in contract["fixtures"]:
            for selection in ("mode","sample"):
                fixed = cell["kernels"]["fixed"][selection]["per_fixture"][fixture["id"]]
                mutant = cell["kernels"]["mutant"][selection]["per_fixture"][fixture["id"]]
                paired.append({"seed":cell["seed"],"branch":cell["branch"],"fixture":fixture["id"],
                    "category":fixture["category"],"selection":selection,"fixed":fixed,"mutant":mutant,
                    "mutant_minus_fixed_return":mutant["mean_discounted_return"]-fixed["mean_discounted_return"],
                    "mutant_minus_fixed_success":mutant["success_rate"]-fixed["success_rate"]})
    assert hashes == {name:sha(path/"state.msgpack") for name,path in checkpoints.items()}
    write_json(root/"result.json",{"status":"complete","completed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "cells_completed":len(results),"training_calls":0,"checkpoint_hashes_unchanged":True,
        "geometry":geometry,"paired_results":paired,"interpretation":contract["protocol"]["interpretation"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()
    contract = json.loads(MANIFEST.read_text())
    geometry = audit_geometry(contract)
    if args.geometry_only:
        print(json.dumps(geometry,indent=2))
    else:
        evaluate(contract,geometry)


if __name__ == "__main__":
    main()
