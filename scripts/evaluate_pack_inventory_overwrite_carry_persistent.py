#!/usr/bin/env python3
"""Evaluate one persistent handoff at the first positive PACK inventory state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

import evaluate_pack_inventory_overwrite_policy_switch as prior  # noqa: E402


SPEC_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_carry_persistent_v1.json"
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
RUN_ROOT = Path(SPEC["run_root"])
SEEDS = tuple(int(seed) for seed in SPEC["seeds"])
PROBABILITIES = tuple(float(value) for value in SPEC["evaluation"]["exact_overwrite_probabilities"])
HORIZON = int(SPEC["evaluation"]["horizon_ticks"])
MODE = "carry_persistent"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPOSITORY, text=True).strip()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def capacity() -> dict:
    home = shutil.disk_usage("/home/ext_csv")
    raid = shutil.disk_usage("/raid/ext_csv")
    required = int(SPEC["storage"]["minimum_required_free_bytes"])
    if raid.free <= required:
        raise RuntimeError(f"RAID free bytes {raid.free} do not preserve required reserve {required}")
    existing = sum(path.stat().st_size for path in RUN_ROOT.rglob("*") if path.is_file()) if RUN_ROOT.exists() else 0
    return {
        "measured_unix": time.time(),
        "home": home._asdict(),
        "raid": raid._asdict(),
        "chosen_filesystem": "/raid/ext_csv",
        "existing_run_bytes": existing,
        "projected_peak_writes_bytes": int(SPEC["storage"]["projected_peak_writes_bytes"]),
        "safety_reserve_bytes": int(SPEC["storage"]["safety_reserve_bytes"]),
        "minimum_required_free_bytes": required,
    }


def prerequisite_paths() -> dict[str, Path]:
    return {
        key: (Path(value) if Path(value).is_absolute() else REPOSITORY / value)
        for key, value in SPEC["prerequisites"].items()
        if not key.endswith("_sha256")
    }


def validate_prerequisites() -> dict:
    records = {}
    for key, path in prerequisite_paths().items():
        actual = sha256(path)
        expected = SPEC["prerequisites"][f"{key}_sha256"]
        if actual != expected:
            raise RuntimeError(f"pinned prerequisite changed: {path} expected={expected} actual={actual}")
        records[key] = {"path": str(path), "sha256": actual}
    complete = json.loads(Path(records["carry_gated_complete"]["path"]).read_text(encoding="utf-8"))
    if complete["result_sha256"] != records["carry_gated_result"]["sha256"]:
        raise RuntimeError("carry-gated completion marker does not match result")
    late = json.loads(Path(records["late_result"]["path"]).read_text(encoding="utf-8"))
    gated = json.loads(Path(records["carry_gated_result"]["path"]).read_text(encoding="utf-8"))
    if not all(late["gates"].values()) or not all(gated["gates"].values()):
        raise RuntimeError("pinned prerequisite result has a failed gate")
    prior.validate_hashes()
    return records


def validate_contract() -> dict:
    if SPEC["status"] != "frozen_before_execution":
        raise RuntimeError("manifest is not frozen for execution")
    prerequisites = validate_prerequisites()
    configs = prior.validate_configs()
    episodes = len(SEEDS) * int(SPEC["evaluation"]["states_per_seed"])
    if episodes != int(SPEC["budget"]["episodes"]):
        raise AssertionError("episode budget arithmetic changed")
    if episodes * HORIZON != int(SPEC["budget"]["maximum_environment_steps"]):
        raise AssertionError("maximum step budget arithmetic changed")
    return {
        "status": "contract_valid",
        "seeds": list(SEEDS),
        "new_mode": MODE,
        "episodes": episodes,
        "maximum_environment_steps": episodes * HORIZON,
        "policy_hidden_size": configs[SEEDS[0]]["source"].policy_hidden_size,
        "prerequisites": prerequisites,
        "capacity": capacity(),
    }


def make_rollout(network, source_growth_period: int):
    initial, labels, state_indices, repeat_indices = prior._evaluation_states(
        prior.PackRestoreSplit.VALIDATION, 1, source_growth_period, natural_only=True
    )
    episode_count = int(labels.shape[0])
    if episode_count != int(SPEC["evaluation"]["states_per_seed"]):
        raise AssertionError("natural start panel changed")

    def step(parameters, carry, step_index):
        del step_index
        source_params, adapted_params = parameters
        (
            state,
            done,
            success,
            length,
            lost_grain,
            loss_seen,
            executed_risk_count,
            source_risk_count,
            switch_count,
            adapted_steps,
            persistent_active,
        ) = carry
        goals = jnp.full((episode_count,), prior.DELIVER_3_GOAL_INDEX, dtype=jnp.int32)
        inputs = prior._batch_inputs(state, goals)
        source_policy, _ = network.apply(source_params, *inputs)
        adapted_policy, _ = network.apply(adapted_params, *inputs)
        source_actions = jnp.argmax(source_policy.logits, axis=-1)
        adapted_actions = jnp.argmax(adapted_policy.logits, axis=-1)
        active = jnp.logical_not(done)
        source_actions = jnp.where(active, source_actions, int(prior.PackRestoreAction.NOOP))
        adapted_actions = jnp.where(active, adapted_actions, int(prior.PackRestoreAction.NOOP))
        trigger = jnp.logical_and(active, state.carried_grain > 0)
        switch_event = jnp.logical_and(jnp.logical_not(persistent_active), trigger)
        use_adapted = jnp.logical_or(persistent_active, switch_event)
        use_adapted = jnp.logical_and(active, use_adapted)
        chosen = jnp.where(use_adapted, adapted_actions, source_actions)
        chosen = jnp.where(active, chosen, int(prior.PackRestoreAction.NOOP))
        source_risk = jnp.logical_and(
            active, jax.vmap(prior.inventory_overwrite_trigger)(state, source_actions)
        )
        executed_risk = jnp.logical_and(
            active, jax.vmap(prior.inventory_overwrite_trigger)(state, chosen)
        )
        stepped = jax.vmap(
            lambda item, action: prior.pack_restore_step(item, action, prior.PackRestoreVariant.FIXED)
        )(state, chosen)
        achieved = jax.vmap(prior.pack_restore_goal_vector)(stepped)[:, prior.DELIVER_3_GOAL_INDEX]
        world_done = jax.vmap(prior.pack_restore_world_done)(stepped)
        just_success = jnp.logical_and(active, achieved)
        just_done = jnp.logical_or(just_success, jnp.logical_and(active, world_done))
        next_done = jnp.logical_or(done, just_done)
        zeros = jnp.zeros((episode_count,), dtype=jnp.int32)
        trace = {
            "pre_signature": prior.compact_signature(state),
            "source_action": source_actions,
            "adapted_action": adapted_actions,
            "chosen_action": chosen,
            "use_adapted": use_adapted,
            "switch_event": switch_event,
            "switch_exit": jnp.zeros((episode_count,), dtype=jnp.bool_),
            "source_risk": source_risk,
            "executed_risk": executed_risk,
            "overwrite_fired": jnp.zeros((episode_count,), dtype=jnp.bool_),
            "lost_grain": zeros,
            "active": active,
        }
        return (
            prior._tree_where(active, stepped, state),
            next_done,
            jnp.logical_or(success, just_success),
            length + active.astype(jnp.int32),
            lost_grain,
            loss_seen,
            executed_risk_count + executed_risk.astype(jnp.int32),
            source_risk_count + source_risk.astype(jnp.int32),
            switch_count + switch_event.astype(jnp.int32),
            adapted_steps + use_adapted.astype(jnp.int32),
            jnp.logical_and(use_adapted, jnp.logical_not(next_done)),
        ), trace

    zeros_bool = jnp.zeros((episode_count,), dtype=jnp.bool_)
    zeros_int = jnp.zeros((episode_count,), dtype=jnp.int32)
    initial_carry = (
        initial,
        zeros_bool,
        zeros_bool,
        zeros_int,
        zeros_int,
        zeros_bool,
        zeros_int,
        zeros_int,
        zeros_int,
        zeros_int,
        zeros_bool,
    )

    def rollout(source_params, adapted_params):
        return jax.lax.scan(
            lambda carry, index: step((source_params, adapted_params), carry, index),
            initial_carry,
            jnp.arange(HORIZON, dtype=jnp.int32),
        )

    return jax.jit(rollout), np.asarray(state_indices), np.asarray(repeat_indices)


def mode_summary(rows: list[dict], mode: str) -> dict:
    fixed = [episode for row in rows for episode in row["fixed"]["episodes"]]
    total_steps = sum(episode["length"] for episode in fixed)
    return {
        "exact": {
            label: prior.evaluate_profile(prior.polynomial_profile(fixed), probability)
            for label, probability in (("p00", 0.0), ("p05", 0.05))
        },
        "fixed_success_rate": math.fsum(float(episode["success"]) for episode in fixed) / len(fixed),
        "fixed_mean_discounted_return": math.fsum(episode["discounted_return"] for episode in fixed) / len(fixed),
        "fixed_mean_length": math.fsum(episode["length"] for episode in fixed) / len(fixed),
        "fixed_risky_start_count": sum(episode["executed_risk_count"] > 0 for episode in fixed),
        "adapted_policy_steps": sum(episode["adapted_policy_steps"] for episode in fixed),
        "total_active_steps": total_steps,
        "adapted_step_fraction": sum(episode["adapted_policy_steps"] for episode in fixed) / total_steps,
        "per_seed": [
            {
                "seed": row["seed"],
                "p00_expected_discounted_return": row["exact"]["p00"]["expected_discounted_return"],
                "p05_expected_discounted_return": row["exact"]["p05"]["expected_discounted_return"],
                "fixed_success_rate": row["fixed"]["aggregate"]["success_rate"],
                "fixed_risky_start_count": sum(
                    episode["executed_risk_count"] > 0 for episode in row["fixed"]["episodes"]
                ),
                "adapted_policy_steps": sum(
                    episode["adapted_policy_steps"] for episode in row["fixed"]["episodes"]
                ),
                "total_active_steps": sum(episode["length"] for episode in row["fixed"]["episodes"]),
            }
            for row in sorted(rows, key=lambda item: item["seed"])
        ],
    }


def prior_rows(raw: dict, mode: str) -> list[dict]:
    return [row for row in raw["rows"] if row["mode"] == mode]


def execution_manifest() -> dict:
    relative = list(SPEC["execution_sources"])
    dirty = git("status", "--porcelain", "--", *relative)
    if dirty:
        raise RuntimeError("execution sources must be committed before launch:\n" + dirty)
    tracked = set(git("ls-files", "--", *relative).splitlines())
    if set(relative) - tracked:
        raise RuntimeError(f"execution sources are untracked: {sorted(set(relative) - tracked)}")
    return {
        "schema_version": "hackrl_pack_inventory_overwrite_carry_persistent_run_v1",
        "manifest_id": SPEC["manifest_id"],
        "execution_head": git("rev-parse", "HEAD"),
        "manifest_sha256": sha256(SPEC_PATH),
        "execution_source_sha256": {path: sha256(REPOSITORY / path) for path in relative},
        "capacity": capacity(),
        "learning_updates": 0,
        "planned_episodes": int(SPEC["budget"]["episodes"]),
    }


def run() -> dict:
    validation = validate_contract()
    manifest = execution_manifest()
    configs = prior.validate_configs()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    run_manifest_path = RUN_ROOT / "run_manifest.json"
    if run_manifest_path.exists() and json.loads(run_manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("run manifest differs from existing execution")
    write_json(run_manifest_path, manifest)
    rows = []
    evaluator = None
    state_indices = None
    for seed in SEEDS:
        source_network, source_params, adapted_network, adapted_params = prior.load_policy_pair(seed, configs)
        if type(source_network) is not type(adapted_network):
            raise AssertionError("source and adapted network types differ")
        if evaluator is None:
            evaluator, state_indices, repeats = make_rollout(
                source_network, configs[seed]["source"].source_growth_period
            )
            if np.any(repeats != 0):
                raise AssertionError("mode evaluation repeat index changed")
        final, traces = evaluator(source_params, adapted_params)
        fixed = prior.summarize_rollout(final, traces, state_indices, MODE, "fixed")
        profile = prior.polynomial_profile(fixed["episodes"])
        exact = {
            f"p{int(round(probability * 100)):02d}": prior.evaluate_profile(profile, probability)
            for probability in PROBABILITIES
        }
        for episode in fixed["episodes"]:
            switches = [step for step in episode["steps"] if step["switch_event"]]
            if len(switches) != 1 or switches[0]["carried_grain"] <= 0:
                raise AssertionError("carry-persistent handoff contract failed")
            used = [step["used_adapted"] for step in episode["steps"]]
            first = used.index(True)
            if not all(used[first:]) or any(step["switch_exit"] for step in episode["steps"]):
                raise AssertionError("carry-persistent policy returned to source")
        rows.append({"seed": seed, "mode": MODE, "fixed": fixed, "risk_profile": profile, "exact": exact})
        print(f"[seed] {seed} complete", flush=True)

    late = json.loads(Path(SPEC["prerequisites"]["late_result"]).read_text(encoding="utf-8"))
    gated = json.loads(Path(SPEC["prerequisites"]["carry_gated_result"]).read_text(encoding="utf-8"))
    comparison = {
        "A_late_risk_persistent": mode_summary(prior_rows(late, "persistent"), "persistent"),
        "B_carry_persistent": mode_summary(rows, MODE),
        "C_carry_gated": mode_summary(gated["rows"], "carry_gated"),
    }
    gates = {
        "all_160_fixed_episodes_recorded": sum(row["fixed"]["aggregate"]["episodes"] for row in rows)
        == int(SPEC["budget"]["episodes"]),
        "every_episode_has_one_positive_inventory_handoff": all(
            sum(step["switch_event"] for step in episode["steps"]) == 1
            for row in rows
            for episode in row["fixed"]["episodes"]
        ),
        "no_episode_returns_to_source_after_handoff": all(
            not any(step["switch_exit"] for step in episode["steps"])
            for row in rows
            for episode in row["fixed"]["episodes"]
        ),
        "prior_endpoint_validation_remains_pinned": all(late["gates"].values()) and all(gated["gates"].values()),
        "no_learning_updates": int(SPEC["budget"]["learning_updates"]) == 0,
    }
    if not all(gates.values()):
        raise AssertionError(f"result gates failed: {gates}")
    result = {
        "schema_version": "hackrl_pack_inventory_overwrite_carry_persistent_result_v1",
        "status": "complete_no_learning",
        "generated_unix": time.time(),
        "execution_head": manifest["execution_head"],
        "manifest_sha256": manifest["manifest_sha256"],
        "design": validation,
        "gates": gates,
        "comparison": comparison,
        "rows": rows,
        "interpretation_guard": SPEC["analysis"]["non_claims"],
    }
    result_path = RUN_ROOT / "result.json"
    write_json(result_path, result)
    write_json(
        RUN_ROOT / "complete.json",
        {"status": "complete_no_learning", "execution_head": manifest["execution_head"], "result_sha256": sha256(result_path)},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-contract", action="store_true")
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if arguments.validate_contract == arguments.run:
        raise SystemExit("choose exactly one of --validate-contract or --run")
    output = validate_contract() if arguments.validate_contract else run()
    print(
        json.dumps(
            output if arguments.validate_contract else {"status": output["status"], "run_root": str(RUN_ROOT)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
