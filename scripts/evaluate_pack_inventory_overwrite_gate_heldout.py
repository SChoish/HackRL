#!/usr/bin/env python3
"""Evaluate the frozen PACK carry gate on the predefined held-out TEST panel."""

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


SPEC_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_gate_heldout_v1.json"
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
RUN_ROOT = Path(SPEC["run_root"])
SEEDS = tuple(int(seed) for seed in SPEC["seeds"])
MODES = tuple(SPEC["conditions"])
KERNELS = tuple(SPEC["evaluation"]["direct_kernels"])
PROBABILITIES = tuple(float(value) for value in SPEC["evaluation"]["exact_overwrite_probabilities"])
HORIZON = int(SPEC["evaluation"]["horizon_ticks"])


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
    existing = (
        sum(path.stat().st_size for path in RUN_ROOT.rglob("*") if path.is_file())
        if RUN_ROOT.exists()
        else 0
    )
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


def state_hashes(split, source_growth_period: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    states, _, state_indices, repeat_indices = prior._evaluation_states(
        split, 1, source_growth_period, natural_only=True
    )
    leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(states)]
    hashes = []
    for state_offset in range(len(state_indices)):
        digest = hashlib.sha256()
        for leaf_index, batched_leaf in enumerate(leaves):
            value = np.ascontiguousarray(batched_leaf[state_offset])
            digest.update(f"{leaf_index}|{value.dtype.str}|{value.shape}|".encode("utf-8"))
            digest.update(value.tobytes(order="C"))
        hashes.append(digest.hexdigest())
    return hashes, np.asarray(state_indices), np.asarray(repeat_indices)


def panel_hash(hashes: list[str]) -> str:
    return hashlib.sha256("".join(hashes).encode("ascii")).hexdigest()


def state_panel_contract(source_growth_period: int) -> dict:
    test, indices, repeats = state_hashes(prior.PackRestoreSplit.TEST, source_growth_period)
    train, _, _ = state_hashes(prior.PackRestoreSplit.TRAIN, source_growth_period)
    validation, _, _ = state_hashes(prior.PackRestoreSplit.VALIDATION, source_growth_period)
    return {
        "split": "test",
        "ordered_full_state_sha256": panel_hash(test),
        "state_count": len(test),
        "unique_state_count": len(set(test)),
        "exact_overlap_with_train_natural_states": len(set(test) & set(train)),
        "exact_overlap_with_validation_natural_states": len(set(test) & set(validation)),
        "state_indices": indices.tolist(),
        "all_repeat_indices_zero": bool(np.all(repeats == 0)),
    }


def choose_actions(mode: str, source_actions, adapted_actions, carried_grain, active):
    """Pure frozen-policy selection rule used by the held-out evaluator."""
    if mode == "source":
        use_adapted = jnp.zeros_like(active)
    elif mode == "adapted":
        use_adapted = active
    elif mode == "carry_gated":
        use_adapted = jnp.logical_and(active, carried_grain > 0)
    else:
        raise ValueError(f"unknown held-out mode: {mode}")
    chosen = jnp.where(use_adapted, adapted_actions, source_actions)
    return chosen, use_adapted


def validate_prerequisites() -> dict:
    path = REPOSITORY / SPEC["prerequisites"]["policy_pair_manifest"]
    actual = sha256(path)
    expected = SPEC["prerequisites"]["policy_pair_manifest_sha256"]
    if actual != expected:
        raise RuntimeError(f"pinned prerequisite changed: {path} expected={expected} actual={actual}")
    prior.validate_hashes()
    return {"policy_pair_manifest": {"path": str(path), "sha256": actual}}


def validate_contract() -> dict:
    if SPEC["status"] != "frozen_before_execution":
        raise RuntimeError("manifest is not frozen for execution")
    prerequisites = validate_prerequisites()
    configs = prior.validate_configs()
    episodes = len(SEEDS) * len(MODES) * len(KERNELS) * int(SPEC["evaluation"]["states_per_seed"])
    if episodes != int(SPEC["budget"]["direct_episodes"]):
        raise AssertionError("episode budget arithmetic changed")
    if episodes * HORIZON != int(SPEC["budget"]["maximum_environment_steps"]):
        raise AssertionError("maximum step budget arithmetic changed")
    panel = state_panel_contract(configs[SEEDS[0]]["source"].source_growth_period)
    expected_panel = SPEC["heldout_panel"]
    if panel["ordered_full_state_sha256"] != expected_panel["ordered_full_state_sha256"]:
        raise AssertionError(f"TEST panel hash changed: {panel['ordered_full_state_sha256']}")
    expected_unique = expected_panel["preflight_uniqueness"]
    if (
        panel["state_count"] != int(SPEC["evaluation"]["states_per_seed"])
        or panel["unique_state_count"] != expected_unique["test_unique_states"]
        or panel["exact_overlap_with_train_natural_states"]
        != expected_unique["exact_overlap_with_train_natural_states"]
        or panel["exact_overlap_with_validation_natural_states"]
        != expected_unique["exact_overlap_with_validation_natural_states"]
        or panel["state_indices"] != expected_panel["state_indices"]
        or not panel["all_repeat_indices_zero"]
    ):
        raise AssertionError(f"held-out state-panel contract failed: {panel}")
    return {
        "status": "contract_valid",
        "seeds": list(SEEDS),
        "modes": list(MODES),
        "episodes": episodes,
        "maximum_environment_steps": episodes * HORIZON,
        "policy_hidden_size": configs[SEEDS[0]]["source"].policy_hidden_size,
        "panel": panel,
        "prerequisites": prerequisites,
        "capacity": capacity(),
    }


def make_rollout(network, mode: str, kernel: str, source_growth_period: int):
    initial, labels, state_indices, repeat_indices = prior._evaluation_states(
        prior.PackRestoreSplit.TEST, 1, source_growth_period, natural_only=True
    )
    episode_count = int(labels.shape[0])
    if episode_count != int(SPEC["evaluation"]["states_per_seed"]):
        raise AssertionError("TEST natural-start panel changed")

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
            was_adapted,
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
        chosen, use_adapted = choose_actions(
            mode, source_actions, adapted_actions, state.carried_grain, active
        )
        chosen = jnp.where(active, chosen, int(prior.PackRestoreAction.NOOP))
        switch_event = jnp.logical_and(use_adapted, jnp.logical_not(was_adapted))
        switch_exit = jnp.logical_and(active, jnp.logical_and(was_adapted, jnp.logical_not(use_adapted)))
        source_risk = jnp.logical_and(
            active, jax.vmap(prior.inventory_overwrite_trigger)(state, source_actions)
        )
        executed_risk = jnp.logical_and(
            active, jax.vmap(prior.inventory_overwrite_trigger)(state, chosen)
        )
        before_total = jax.vmap(prior.physical_grain_total)(state)
        if kernel == "fixed":
            stepped = jax.vmap(
                lambda item, action: prior.pack_restore_step(
                    item, action, prior.PackRestoreVariant.FIXED
                )
            )(state, chosen)
            fired = jnp.zeros((episode_count,), dtype=jnp.bool_)
        elif kernel == "inventory_overwrite_p1":
            stepped, fired = jax.vmap(
                lambda item, action: prior.pack_restore_step_inventory_overwrite_event(
                    item, action, True
                )
            )(state, chosen)
            fired = jnp.logical_and(active, fired)
        else:
            raise ValueError(f"unknown direct kernel: {kernel}")
        after_total = jax.vmap(prior.physical_grain_total)(stepped)
        lost = jnp.where(active, jnp.maximum(before_total - after_total, 0), 0)
        achieved = jax.vmap(prior.pack_restore_goal_vector)(stepped)[:, prior.DELIVER_3_GOAL_INDEX]
        world_done = jax.vmap(prior.pack_restore_world_done)(stepped)
        just_success = jnp.logical_and(active, achieved)
        just_done = jnp.logical_or(just_success, jnp.logical_and(active, world_done))
        next_done = jnp.logical_or(done, just_done)
        trace = {
            "pre_signature": prior.compact_signature(state),
            "source_action": source_actions,
            "adapted_action": adapted_actions,
            "chosen_action": chosen,
            "use_adapted": use_adapted,
            "switch_event": switch_event,
            "switch_exit": switch_exit,
            "source_risk": source_risk,
            "executed_risk": executed_risk,
            "overwrite_fired": fired,
            "lost_grain": lost,
            "active": active,
        }
        return (
            prior._tree_where(active, stepped, state),
            next_done,
            jnp.logical_or(success, just_success),
            length + active.astype(jnp.int32),
            lost_grain + lost,
            jnp.logical_or(loss_seen, lost > 0),
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


def execution_manifest() -> dict:
    relative = list(SPEC["execution_sources"])
    dirty = git("status", "--porcelain", "--", *relative)
    if dirty:
        raise RuntimeError("execution sources must be committed before launch:\n" + dirty)
    tracked = set(git("ls-files", "--", *relative).splitlines())
    if set(relative) - tracked:
        raise RuntimeError(f"execution sources are untracked: {sorted(set(relative) - tracked)}")
    return {
        "schema_version": "hackrl_pack_inventory_overwrite_gate_heldout_run_v1",
        "manifest_id": SPEC["manifest_id"],
        "execution_head": git("rev-parse", "HEAD"),
        "manifest_sha256": sha256(SPEC_PATH),
        "execution_source_sha256": {path: sha256(REPOSITORY / path) for path in relative},
        "capacity": capacity(),
        "learning_updates": 0,
        "planned_direct_episodes": int(SPEC["budget"]["direct_episodes"]),
    }


def mode_summary(rows: list[dict], mode: str) -> dict:
    members = sorted((row for row in rows if row["mode"] == mode), key=lambda row: row["seed"])
    fixed = [episode for row in members for episode in row["fixed"]["episodes"]]
    p1 = [episode for row in members for episode in row["inventory_overwrite_p1"]["episodes"]]
    profile = prior.polynomial_profile(fixed)
    return {
        "seeds": len(members),
        "fixed": prior.summarize_episode_pool(fixed),
        "inventory_overwrite_p1": prior.summarize_episode_pool(p1),
        "exact": {
            f"p{int(round(probability * 100)):02d}": prior.evaluate_profile(profile, probability)
            for probability in PROBABILITIES
        },
        "risk_count_distribution": profile["risk_count_distribution"],
        "fixed_risky_start_count": sum(episode["executed_risk_count"] > 0 for episode in fixed),
        "per_seed": [
            {
                "seed": row["seed"],
                "p00_expected_discounted_return": row["exact"]["p00"]["expected_discounted_return"],
                "p05_expected_discounted_return": row["exact"]["p05"]["expected_discounted_return"],
                "p100_expected_discounted_return": row["exact"]["p100"]["expected_discounted_return"],
                "fixed_success_rate": row["fixed"]["aggregate"]["success_rate"],
                "fixed_risky_start_count": sum(
                    episode["executed_risk_count"] > 0 for episode in row["fixed"]["episodes"]
                ),
            }
            for row in members
        ],
    }


def paired_contrast(comparison: dict, left: str, right: str) -> dict:
    left_rows = {row["seed"]: row for row in comparison[left]["per_seed"]}
    right_rows = {row["seed"]: row for row in comparison[right]["per_seed"]}
    rows = []
    for seed in SEEDS:
        difference = (
            left_rows[seed]["p05_expected_discounted_return"]
            - right_rows[seed]["p05_expected_discounted_return"]
        )
        rows.append({"seed": seed, "p05_expected_discounted_return_difference": difference})
    return {
        "left": left,
        "right": right,
        "metric": "p05_expected_discounted_return",
        "mean_difference": math.fsum(
            row["p05_expected_discounted_return_difference"] for row in rows
        ) / len(rows),
        "positive_seed_count": sum(
            row["p05_expected_discounted_return_difference"] > 0 for row in rows
        ),
        "zero_seed_count": sum(
            math.isclose(row["p05_expected_discounted_return_difference"], 0.0, abs_tol=1e-15)
            for row in rows
        ),
        "paired_seed_differences": rows,
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
    evaluators = None
    state_indices = None
    for seed in SEEDS:
        source_network, source_params, adapted_network, adapted_params = prior.load_policy_pair(
            seed, configs
        )
        if type(source_network) is not type(adapted_network):
            raise AssertionError("source and adapted network types differ")
        if evaluators is None:
            evaluators = {}
            for mode in MODES:
                for kernel in KERNELS:
                    evaluator, indices, repeats = make_rollout(
                        source_network,
                        mode,
                        kernel,
                        configs[seed]["source"].source_growth_period,
                    )
                    evaluators[(mode, kernel)] = evaluator
                    state_indices = indices
                    if np.any(repeats != 0):
                        raise AssertionError("TEST evaluation repeat index changed")
        for mode in MODES:
            direct = {}
            for kernel in KERNELS:
                final, traces = evaluators[(mode, kernel)](source_params, adapted_params)
                direct[kernel] = prior.summarize_rollout(
                    final, traces, state_indices, mode, kernel
                )
            profile = prior.polynomial_profile(direct["fixed"]["episodes"])
            exact = {
                f"p{int(round(probability * 100)):02d}": prior.evaluate_profile(
                    profile, probability
                )
                for probability in PROBABILITIES
            }
            checks = {
                "p0_return_matches_direct_fixed": math.isclose(
                    exact["p00"]["expected_discounted_return"],
                    direct["fixed"]["aggregate"]["mean_discounted_return"],
                    abs_tol=1e-12,
                ),
                "p0_success_matches_direct_fixed": math.isclose(
                    exact["p00"]["success_rate"],
                    direct["fixed"]["aggregate"]["success_rate"],
                    abs_tol=1e-12,
                ),
                "p1_return_matches_direct_always_fire": math.isclose(
                    exact["p100"]["expected_discounted_return"],
                    direct["inventory_overwrite_p1"]["aggregate"]["mean_discounted_return"],
                    abs_tol=1e-12,
                ),
                "p1_success_matches_direct_always_fire": math.isclose(
                    exact["p100"]["success_rate"],
                    direct["inventory_overwrite_p1"]["aggregate"]["success_rate"],
                    abs_tol=1e-12,
                ),
                "all_first_overwrite_prehistories_match": all(
                    prior.prehistory_matches(left, right)
                    for left, right in zip(
                        direct["fixed"]["episodes"],
                        direct["inventory_overwrite_p1"]["episodes"],
                        strict=True,
                    )
                ),
                "every_p1_success_has_zero_fixed_path_risk": all(
                    not right["success"] or left["executed_risk_count"] == 0
                    for left, right in zip(
                        direct["fixed"]["episodes"],
                        direct["inventory_overwrite_p1"]["episodes"],
                        strict=True,
                    )
                ),
            }
            if not all(checks.values()):
                raise AssertionError(f"endpoint validation failed: seed={seed} mode={mode} {checks}")
            rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "fixed": direct["fixed"],
                    "inventory_overwrite_p1": direct["inventory_overwrite_p1"],
                    "risk_profile": profile,
                    "exact": exact,
                    "checks": checks,
                }
            )
        print(f"[seed] {seed} complete", flush=True)

    comparison = {mode: mode_summary(rows, mode) for mode in MODES}
    primary = paired_contrast(comparison, "carry_gated", "source")
    adapted_contrast = paired_contrast(comparison, "carry_gated", "adapted")
    gates = {
        "all_960_direct_episodes_recorded": sum(
            row[kernel]["aggregate"]["episodes"] for row in rows for kernel in KERNELS
        )
        == int(SPEC["budget"]["direct_episodes"]),
        "all_endpoint_checks_pass": all(all(row["checks"].values()) for row in rows),
        "test_panel_hash_matches_frozen_manifest": validation["panel"]["ordered_full_state_sha256"]
        == SPEC["heldout_panel"]["ordered_full_state_sha256"],
        "all_test_states_unique": validation["panel"]["unique_state_count"]
        == int(SPEC["evaluation"]["states_per_seed"]),
        "test_has_no_exact_train_or_validation_state_overlap": validation["panel"][
            "exact_overlap_with_train_natural_states"
        ]
        == 0
        and validation["panel"]["exact_overlap_with_validation_natural_states"] == 0,
        "no_learning_updates": int(SPEC["budget"]["learning_updates"]) == 0,
    }
    if not all(gates.values()):
        raise AssertionError(f"result gates failed: {gates}")
    result = {
        "schema_version": "hackrl_pack_inventory_overwrite_gate_heldout_result_v1",
        "status": "complete_no_learning",
        "generated_unix": time.time(),
        "execution_head": manifest["execution_head"],
        "manifest_sha256": manifest["manifest_sha256"],
        "gates": gates,
        "design": validation,
        "comparison": comparison,
        "primary_contrast": primary,
        "carry_gated_minus_adapted": adapted_contrast,
        "rows": rows,
        "interpretation_guard": SPEC["analysis"]["non_claims"],
    }
    write_json(RUN_ROOT / "result.json", result)
    write_json(
        RUN_ROOT / "complete.json",
        {
            "status": "complete_no_learning",
            "execution_head": manifest["execution_head"],
            "result_sha256": sha256(RUN_ROOT / "result.json"),
        },
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
            output
            if arguments.validate_contract
            else {"status": output["status"], "run_root": str(RUN_ROOT)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
