#!/usr/bin/env python3
"""Evaluate one source-actor action at the first post-handoff empty-state disagreement."""

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

import evaluate_pack_inventory_overwrite_carry_persistent as carry  # noqa: E402


prior = carry.prior
SPEC_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_one_action_swap_v1.json"
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
RUN_ROOT = Path(SPEC["run_root"])
SEEDS = tuple(int(seed) for seed in SPEC["seeds"])
PROBABILITIES = tuple(float(value) for value in SPEC["evaluation"]["exact_overwrite_probabilities"])
HORIZON = int(SPEC["evaluation"]["horizon_ticks"])
MODE = "one_action_swap"
STEP_COMPARISON_KEYS = (
    "position",
    "carried_grain",
    "source_grain",
    "delivered_total",
    "source_action",
    "adapted_action",
)


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
        key: Path(value)
        for key, value in SPEC["prerequisites"].items()
        if not key.endswith("_sha256")
    }


def mode_episode_map(raw: dict, mode: str) -> dict[tuple[int, int], dict]:
    return {
        (int(row["seed"]), int(episode["state_index"])): episode
        for row in raw["rows"]
        if row["mode"] == mode
        for episode in row["fixed"]["episodes"]
    }


def pinned_inputs() -> tuple[dict, dict]:
    paths = prerequisite_paths()
    for key, path in paths.items():
        expected = SPEC["prerequisites"][f"{key}_sha256"]
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"pinned prerequisite changed: {path} expected={expected} actual={actual}")
    b_complete = json.loads(paths["B_complete"].read_text(encoding="utf-8"))
    c_complete = json.loads(paths["C_complete"].read_text(encoding="utf-8"))
    if b_complete["result_sha256"] != SPEC["prerequisites"]["B_result_sha256"]:
        raise RuntimeError("B completion marker does not match B result")
    if c_complete["result_sha256"] != SPEC["prerequisites"]["C_result_sha256"]:
        raise RuntimeError("C completion marker does not match C result")
    b_raw = json.loads(paths["B_result"].read_text(encoding="utf-8"))
    c_raw = json.loads(paths["C_result"].read_text(encoding="utf-8"))
    if not all(b_raw["gates"].values()) or not all(c_raw["gates"].values()):
        raise RuntimeError("a pinned prerequisite has a failed gate")
    return b_raw, c_raw


def diagnostic_keys(b_raw: dict, c_raw: dict) -> list[tuple[int, int]]:
    b = mode_episode_map(b_raw, "carry_persistent")
    c = mode_episode_map(c_raw, "carry_gated")
    if set(b) != set(c):
        raise AssertionError("B/C natural-start panels differ")
    keys = sorted(key for key in b if not b[key]["success"] and c[key]["success"])
    expected = sorted(tuple(item) for item in SPEC["diagnostic_panel"]["expected_seed_state_indices"])
    if keys != expected or len(keys) != int(SPEC["diagnostic_panel"]["expected_count"]):
        raise AssertionError(f"pinned diagnostic panel changed: actual={keys} expected={expected}")
    return keys


def validate_contract() -> dict:
    if SPEC["status"] != "frozen_before_execution":
        raise RuntimeError("manifest is not frozen for execution")
    b_raw, c_raw = pinned_inputs()
    targets = diagnostic_keys(b_raw, c_raw)
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
        "diagnostic_keys": [list(key) for key in targets],
        "capacity": capacity(),
    }


def choose_one_action_swap(
    source_actions,
    adapted_actions,
    carried_grain,
    active,
    control_phase,
):
    first_positive_handoff = jnp.logical_and(
        active, jnp.logical_and(control_phase == 0, carried_grain > 0)
    )
    handed_off = jnp.logical_or(control_phase > 0, first_positive_handoff)
    one_action_swap = jnp.logical_and(
        active,
        jnp.logical_and(
            handed_off,
            jnp.logical_and(
                control_phase < 2,
                jnp.logical_and(carried_grain == 0, source_actions != adapted_actions),
            ),
        ),
    )
    use_adapted = jnp.logical_and(handed_off, jnp.logical_not(one_action_swap))
    use_adapted = jnp.logical_and(active, use_adapted)
    chosen = jnp.where(use_adapted, adapted_actions, source_actions)
    next_phase = jnp.where(
        one_action_swap,
        jnp.asarray(2, dtype=control_phase.dtype),
        jnp.where(
            first_positive_handoff,
            jnp.asarray(1, dtype=control_phase.dtype),
            control_phase,
        ),
    )
    return chosen, use_adapted, first_positive_handoff, one_action_swap, next_phase


def make_rollout(network, source_growth_period: int):
    initial, labels, state_indices, repeat_indices = prior._evaluation_states(
        prior.PackRestoreSplit.VALIDATION, 1, source_growth_period, natural_only=True
    )
    episode_count = int(labels.shape[0])
    if episode_count != int(SPEC["evaluation"]["states_per_seed"]):
        raise AssertionError("natural start panel changed")

    def step(parameters, carry_state, step_index):
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
            control_phase,
        ) = carry_state
        goals = jnp.full((episode_count,), prior.DELIVER_3_GOAL_INDEX, dtype=jnp.int32)
        inputs = prior._batch_inputs(state, goals)
        source_policy, _ = network.apply(source_params, *inputs)
        adapted_policy, _ = network.apply(adapted_params, *inputs)
        source_actions = jnp.argmax(source_policy.logits, axis=-1)
        adapted_actions = jnp.argmax(adapted_policy.logits, axis=-1)
        active = jnp.logical_not(done)
        source_actions = jnp.where(active, source_actions, int(prior.PackRestoreAction.NOOP))
        adapted_actions = jnp.where(active, adapted_actions, int(prior.PackRestoreAction.NOOP))
        chosen, use_adapted, handoff, one_swap, next_phase = choose_one_action_swap(
            source_actions,
            adapted_actions,
            state.carried_grain,
            active,
            control_phase,
        )
        chosen = jnp.where(active, chosen, int(prior.PackRestoreAction.NOOP))
        source_risk = jnp.logical_and(
            active, jax.vmap(prior.inventory_overwrite_trigger)(state, source_actions)
        )
        executed_risk = jnp.logical_and(
            active, jax.vmap(prior.inventory_overwrite_trigger)(state, chosen)
        )
        stepped = jax.vmap(
            lambda item, action: prior.pack_restore_step(
                item, action, prior.PackRestoreVariant.FIXED
            )
        )(state, chosen)
        achieved = jax.vmap(prior.pack_restore_goal_vector)(stepped)[
            :, prior.DELIVER_3_GOAL_INDEX
        ]
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
            "switch_event": handoff,
            "switch_exit": one_swap,
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
            switch_count + handoff.astype(jnp.int32),
            adapted_steps + use_adapted.astype(jnp.int32),
            jnp.where(next_done, control_phase, next_phase),
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
        zeros_int,
    )

    def rollout(source_params, adapted_params):
        return jax.lax.scan(
            lambda carry_state, index: step(
                (source_params, adapted_params), carry_state, index
            ),
            initial_carry,
            jnp.arange(HORIZON, dtype=jnp.int32),
        )

    return jax.jit(rollout), np.asarray(state_indices), np.asarray(repeat_indices)


def decorate_episode(episode: dict) -> dict:
    swap_steps = [step for step in episode["steps"] if step["switch_exit"]]
    if len(swap_steps) > 1:
        raise AssertionError("D used more than one source-action swap")
    swap = swap_steps[0] if swap_steps else None
    episode["one_action_swap_count"] = len(swap_steps)
    episode["one_action_swap_step"] = None if swap is None else int(swap["step"])
    episode["one_action_source_action"] = None if swap is None else int(swap["source_action"])
    episode["one_action_adapted_action"] = None if swap is None else int(swap["adapted_action"])
    episode["risk_count_before_swap"] = (
        None
        if swap is None
        else sum(
            int(step["executed_risky_harvest"])
            for step in episode["steps"]
            if int(step["step"]) < int(swap["step"])
        )
    )
    return episode


def step_key(step: dict) -> tuple:
    values = []
    for key in STEP_COMPARISON_KEYS:
        value = step[key]
        values.append(tuple(value) if isinstance(value, list) else value)
    return tuple(values)


def first_common_disagreement(b_episode: dict, c_episode: dict) -> int | None:
    limit = min(len(b_episode["steps"]), len(c_episode["steps"]))
    for index in range(limit):
        b_step = b_episode["steps"][index]
        c_step = c_episode["steps"][index]
        if step_key(b_step) != step_key(c_step):
            raise AssertionError("B/C policy inputs or argmax actions differ before chosen-action divergence")
        if b_step["chosen_action"] != c_step["chosen_action"]:
            if b_step["carried_grain"] != 0:
                raise AssertionError("first B/C divergence is not at empty inventory")
            if b_step["chosen_action"] != b_step["adapted_action"]:
                raise AssertionError("B did not choose the adapted action at divergence")
            if c_step["chosen_action"] != c_step["source_action"]:
                raise AssertionError("C did not choose the source action at divergence")
            return index + 1
    if len(b_episode["steps"]) != len(c_episode["steps"]):
        raise AssertionError("B/C lengths differ without an earlier chosen-action divergence")
    return None


def validate_path(
    key: tuple[int, int],
    b_episode: dict,
    c_episode: dict,
    d_episode: dict,
) -> dict:
    branch_step = first_common_disagreement(b_episode, c_episode)
    d_swap_step = d_episode["one_action_swap_step"]
    if d_swap_step != branch_step:
        raise AssertionError(f"D swap does not match pinned first disagreement: key={key}")
    if branch_step is None:
        if d_episode != b_episode:
            comparable = {field: d_episode[field] for field in d_episode if not field.startswith("one_action") and field != "risk_count_before_swap"}
            if comparable != b_episode:
                raise AssertionError(f"no-disagreement D path differs from B: key={key}")
        return {"seed": key[0], "state_index": key[1], "branch_step": None, "passed": True}

    index = branch_step - 1
    if len(d_episode["steps"]) <= index:
        raise AssertionError(f"D ended before its branch: key={key}")
    for prefix_index in range(index):
        d_step = d_episode["steps"][prefix_index]
        b_step = b_episode["steps"][prefix_index]
        c_step = c_episode["steps"][prefix_index]
        if step_key(d_step) != step_key(b_step) or step_key(d_step) != step_key(c_step):
            raise AssertionError(f"D state/action prefix differs: key={key} step={prefix_index + 1}")
        if d_step["chosen_action"] != b_step["chosen_action"] or d_step["chosen_action"] != c_step["chosen_action"]:
            raise AssertionError(f"D chosen-action prefix differs: key={key} step={prefix_index + 1}")

    d_branch = d_episode["steps"][index]
    b_branch = b_episode["steps"][index]
    c_branch = c_episode["steps"][index]
    if step_key(d_branch) != step_key(b_branch) or step_key(d_branch) != step_key(c_branch):
        raise AssertionError(f"D branch state differs from pinned B/C branch: key={key}")
    if not d_branch["switch_exit"] or d_branch["chosen_action"] != d_branch["source_action"]:
        raise AssertionError(f"D did not use the one source action at branch: key={key}")
    if any(not step["used_adapted"] for step in d_episode["steps"][index + 1 :]):
        raise AssertionError(f"D relinquished adapted control after its one swap: key={key}")
    return {"seed": key[0], "state_index": key[1], "branch_step": branch_step, "passed": True}


def execution_manifest() -> dict:
    relative = list(SPEC["execution_sources"])
    dirty = git("status", "--porcelain", "--", *relative)
    if dirty:
        raise RuntimeError("execution sources must be committed before launch:\n" + dirty)
    tracked = set(git("ls-files", "--", *relative).splitlines())
    if set(relative) - tracked:
        raise RuntimeError(f"execution sources are untracked: {sorted(set(relative) - tracked)}")
    return {
        "schema_version": "hackrl_pack_inventory_overwrite_one_action_swap_run_v1",
        "manifest_id": SPEC["manifest_id"],
        "execution_head": git("rev-parse", "HEAD"),
        "manifest_sha256": sha256(SPEC_PATH),
        "execution_source_sha256": {
            path: sha256(REPOSITORY / path) for path in relative
        },
        "capacity": capacity(),
        "learning_updates": 0,
        "planned_episodes": int(SPEC["budget"]["episodes"]),
    }


def run() -> dict:
    validation = validate_contract()
    manifest = execution_manifest()
    b_raw, c_raw = pinned_inputs()
    targets = diagnostic_keys(b_raw, c_raw)
    b_map = mode_episode_map(b_raw, "carry_persistent")
    c_map = mode_episode_map(c_raw, "carry_gated")
    configs = prior.validate_configs()

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    run_manifest_path = RUN_ROOT / "run_manifest.json"
    if run_manifest_path.exists() and json.loads(run_manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("run manifest differs from existing execution")
    write_json(run_manifest_path, manifest)

    rows = []
    path_checks = []
    evaluator = None
    state_indices = None
    for seed in SEEDS:
        source_network, source_params, adapted_network, adapted_params = prior.load_policy_pair(
            seed, configs
        )
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
        fixed["episodes"] = [decorate_episode(episode) for episode in fixed["episodes"]]
        fixed["aggregate"]["mean_one_action_swap_count"] = float(
            np.mean([episode["one_action_swap_count"] for episode in fixed["episodes"]])
        )
        profile = prior.polynomial_profile(fixed["episodes"])
        exact = {
            f"p{int(round(probability * 100)):02d}": prior.evaluate_profile(
                profile, probability
            )
            for probability in PROBABILITIES
        }
        for episode in fixed["episodes"]:
            key = (seed, int(episode["state_index"]))
            path_checks.append(validate_path(key, b_map[key], c_map[key], episode))
            handoffs = [step for step in episode["steps"] if step["switch_event"]]
            if len(handoffs) != 1 or handoffs[0]["carried_grain"] <= 0:
                raise AssertionError(f"D handoff contract failed: key={key}")
            swaps = [step for step in episode["steps"] if step["switch_exit"]]
            if any(
                step["carried_grain"] != 0
                or step["source_action"] == step["adapted_action"]
                or step["chosen_action"] != step["source_action"]
                for step in swaps
            ):
                raise AssertionError(f"D swap eligibility failed: key={key}")
        rows.append(
            {
                "seed": seed,
                "mode": MODE,
                "fixed": fixed,
                "risk_profile": profile,
                "exact": exact,
            }
        )
        print(f"[seed] {seed} complete", flush=True)

    d_map = {
        (row["seed"], episode["state_index"]): episode
        for row in rows
        for episode in row["fixed"]["episodes"]
    }
    fixed_pool = [episode for row in rows for episode in row["fixed"]["episodes"]]
    comparison = {
        "B_carry_persistent": carry.mode_summary(
            [row for row in b_raw["rows"] if row["mode"] == "carry_persistent"],
            "carry_persistent",
        ),
        "C_carry_gated": carry.mode_summary(
            [row for row in c_raw["rows"] if row["mode"] == "carry_gated"],
            "carry_gated",
        ),
        "D_one_action_swap": carry.mode_summary(rows, MODE),
    }
    target_rows = [
        {
            "seed": key[0],
            "state_index": key[1],
            "B_success": bool(b_map[key]["success"]),
            "C_success": bool(c_map[key]["success"]),
            "D_success": bool(d_map[key]["success"]),
            "B_length": int(b_map[key]["length"]),
            "C_length": int(c_map[key]["length"]),
            "D_length": int(d_map[key]["length"]),
            "branch_step": d_map[key]["one_action_swap_step"],
            "source_action": d_map[key]["one_action_source_action"],
            "adapted_action": d_map[key]["one_action_adapted_action"],
            "D_executed_risk_count": int(d_map[key]["executed_risk_count"]),
        }
        for key in targets
    ]
    recovered = [row for row in target_rows if row["D_success"]]
    new_failures_from_B_success = [
        {"seed": key[0], "state_index": key[1]}
        for key in sorted(b_map)
        if b_map[key]["success"] and not d_map[key]["success"]
    ]
    diagnostic = {
        "targets": target_rows,
        "recovered_count": len(recovered),
        "recovered_seed_state_indices": [
            [row["seed"], row["state_index"]] for row in recovered
        ],
        "all_four_recovered": len(recovered) == len(target_rows) == 4,
        "new_D_failures_among_B_successes": new_failures_from_B_success,
    }
    gates = {
        "all_160_D_fixed_episodes_recorded": len(fixed_pool)
        == int(SPEC["budget"]["episodes"]),
        "pinned_B_C_inputs_unchanged": all(
            sha256(path) == SPEC["prerequisites"][f"{key}_sha256"]
            for key, path in prerequisite_paths().items()
        ),
        "diagnostic_panel_matches_four_predeclared_cases": targets
        == sorted(tuple(item) for item in SPEC["diagnostic_panel"]["expected_seed_state_indices"]),
        "all_D_prefix_and_branch_checks_pass": len(path_checks) == len(fixed_pool)
        and all(check["passed"] for check in path_checks),
        "each_D_path_uses_at_most_one_source_action_swap": all(
            episode["one_action_swap_count"] <= 1 for episode in fixed_pool
        ),
        "each_swap_is_at_first_pinned_B_C_empty_action_disagreement": all(
            check["branch_step"] == d_map[(check["seed"], check["state_index"])]["one_action_swap_step"]
            for check in path_checks
        ),
        "after_swap_D_uses_adapted_actor_until_termination": all(
            all(step["used_adapted"] for step in episode["steps"][episode["one_action_swap_step"] :])
            for episode in fixed_pool
            if episode["one_action_swap_step"] is not None
        ),
        "no_learning_updates": int(SPEC["budget"]["learning_updates"]) == 0,
    }
    if not all(gates.values()):
        raise AssertionError(f"result gates failed: {gates}")

    result = {
        "schema_version": "hackrl_pack_inventory_overwrite_one_action_swap_result_v1",
        "status": "complete_no_learning",
        "generated_unix": time.time(),
        "execution_head": manifest["execution_head"],
        "manifest_sha256": manifest["manifest_sha256"],
        "design": validation,
        "gates": gates,
        "diagnostic": diagnostic,
        "aggregate": {
            "swap_path_count": sum(
                episode["one_action_swap_count"] == 1 for episode in fixed_pool
            ),
            "no_swap_path_count": sum(
                episode["one_action_swap_count"] == 0 for episode in fixed_pool
            ),
            "fixed_success_rate": math.fsum(
                float(episode["success"]) for episode in fixed_pool
            )
            / len(fixed_pool),
            "fixed_mean_length": math.fsum(
                episode["length"] for episode in fixed_pool
            )
            / len(fixed_pool),
            "fixed_risky_path_count": sum(
                episode["executed_risk_count"] > 0 for episode in fixed_pool
            ),
            "fixed_total_executed_risky_harvests": sum(
                episode["executed_risk_count"] for episode in fixed_pool
            ),
            "exact": {
                label: prior.evaluate_profile(
                    prior.polynomial_profile(fixed_pool), probability
                )
                for label, probability in (("p00", 0.0), ("p05", 0.05))
            },
            "failed_paths": [
                {"seed": row["seed"], **episode}
                for row in rows
                for episode in row["fixed"]["episodes"]
                if not episode["success"]
            ],
        },
        "comparison": comparison,
        "path_checks": path_checks,
        "rows": rows,
        "interpretation_guard": SPEC["analysis"]["non_claims"],
    }
    result_path = RUN_ROOT / "result.json"
    write_json(result_path, result)
    write_json(
        RUN_ROOT / "complete.json",
        {
            "status": "complete_no_learning",
            "execution_head": manifest["execution_head"],
            "result_sha256": sha256(result_path),
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
            else {
                "status": output["status"],
                "run_root": str(RUN_ROOT),
                "diagnostic": output["diagnostic"],
                "aggregate": output["aggregate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
