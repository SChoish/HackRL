#!/usr/bin/env python3
"""Evaluate frozen source/adapted PACK policies under oracle-gated switching."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from hackrl.dual_leo import init_dual_leo_teacher, load_dual_checkpoint
from hackrl.pack_restore import (
    PackRestoreAction,
    PackRestoreSplit,
    PackRestoreVariant,
    inventory_overwrite_trigger,
    pack_restore_goal_vector,
    pack_restore_step,
    pack_restore_step_inventory_overwrite_event,
    pack_restore_world_done,
    physical_grain_total,
)
from hackrl.pack_restore_gc import (
    DELIVER_3_GOAL_INDEX,
    DISCOUNT,
    NUM_ACTIONS,
    NUM_GOALS,
    _batch_inputs,
    _evaluation_states,
    _tree_where,
    config_from_pack_restore_gc_payload,
    initialize_pack_restore_gc,
    load_pack_restore_gc_checkpoint,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_policy_switch_v1.json"
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
RUN_ROOT = Path(SPEC["run_root"])
SEEDS = tuple(int(seed) for seed in SPEC["seeds"])
MODES = tuple(SPEC["execution_modes"])
DIRECT_KERNELS = tuple(SPEC["evaluation"]["direct_kernels"])
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


def source_checkpoint(seed: int) -> Path:
    return Path(SPEC["source_root"]) / f"seed{seed}" / "checkpoints" / SPEC["source_checkpoint"]


def adapted_checkpoint(seed: int) -> Path:
    return Path(SPEC["adapted_root"]) / f"seed{seed}" / "checkpoints" / SPEC["adapted_checkpoint"]


def validate_hashes() -> None:
    prerequisites = SPEC["prerequisites"]
    for path_key, hash_key in (
        ("transfer_compact_result", "transfer_compact_result_sha256"),
        ("transfer_raw_result", "transfer_raw_result_sha256"),
        ("reference_paths", "reference_paths_sha256"),
    ):
        path = Path(prerequisites[path_key])
        if not path.is_absolute():
            path = REPOSITORY / path
        actual = sha256(path)
        if actual != prerequisites[hash_key]:
            raise RuntimeError(f"pinned prerequisite changed: {path}")
    for kind, resolver in (("source", source_checkpoint), ("adapted", adapted_checkpoint)):
        for seed in SEEDS:
            root = resolver(seed)
            for name, expected in SPEC["checkpoint_sha256"][kind][str(seed)].items():
                path = root / name
                actual = sha256(path)
                if actual != expected:
                    raise RuntimeError(f"checkpoint input changed: {path}")


def validate_configs() -> dict[int, dict]:
    records = {}
    for seed in SEEDS:
        source_path = source_checkpoint(seed)
        adapted_path = adapted_checkpoint(seed)
        source_config = config_from_pack_restore_gc_payload(
            json.loads((source_path / "config.json").read_text(encoding="utf-8"))
        )
        adapted_config = config_from_pack_restore_gc_payload(
            json.loads((adapted_path / "config.json").read_text(encoding="utf-8"))
        )
        source_meta = json.loads((source_path / "metadata.json").read_text(encoding="utf-8"))
        adapted_meta = json.loads((adapted_path / "metadata.json").read_text(encoding="utf-8"))
        source_arm = json.loads((source_path / "arm.json").read_text(encoding="utf-8"))
        receipt = json.loads((adapted_path / "receipt.json").read_text(encoding="utf-8"))
        locked = (
            "seed",
            "num_envs",
            "num_steps",
            "policy_hidden_size",
            "source_growth_period",
            "evaluation_split",
            "goal_mode",
        )
        if any(getattr(source_config, name) != getattr(adapted_config, name) for name in locked):
            raise RuntimeError(f"source/adapted locked config differs for seed {seed}")
        if (
            source_config.variant != PackRestoreVariant.FIXED.value
            or adapted_config.variant != PackRestoreVariant.INVENTORY_OVERWRITE.value
            or float(adapted_config.inventory_overwrite_probability) != 1.0
            or source_config.goal_mode != "deliver_3"
            or source_arm != {"imitate_teacher": True, "learn_teacher": True}
            or int(source_meta["global_update"]) != 4608
            or int(adapted_meta["global_update"]) != 4640
            or int(receipt["adaptation_update"]) != 32
        ):
            raise RuntimeError(f"checkpoint role contract failed for seed {seed}")
        records[seed] = {"source": source_config, "adapted": adapted_config}
    return records


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


def validate_contract() -> dict:
    if SPEC["status"] != "frozen_before_execution":
        raise RuntimeError("manifest is not frozen for execution")
    validate_hashes()
    configs = validate_configs()
    episodes = len(SEEDS) * len(MODES) * len(DIRECT_KERNELS) * int(SPEC["evaluation"]["states_per_seed"])
    if episodes != int(SPEC["budget"]["episodes"]):
        raise AssertionError("episode budget arithmetic changed")
    if episodes * HORIZON != int(SPEC["budget"]["maximum_environment_steps"]):
        raise AssertionError("maximum step budget arithmetic changed")
    return {
        "status": "contract_valid",
        "seeds": list(SEEDS),
        "modes": list(MODES),
        "episodes": episodes,
        "maximum_environment_steps": episodes * HORIZON,
        "policy_hidden_size": configs[SEEDS[0]]["source"].policy_hidden_size,
        "capacity": capacity(),
    }


def load_policy_pair(seed: int, configs: dict[int, dict]):
    source_config = configs[seed]["source"]
    source_network, fresh_source = initialize_pack_restore_gc(source_config)
    inputs = _batch_inputs(fresh_source.env_state, fresh_source.current_goal)
    _, teacher_template, _ = init_dual_leo_teacher(source_config, inputs[0], inputs[1], NUM_GOALS, NUM_ACTIONS)
    source_runner, _ = load_dual_checkpoint(source_checkpoint(seed), fresh_source, teacher_template)

    adapted_config = configs[seed]["adapted"]
    adapted_network, fresh_adapted = initialize_pack_restore_gc(adapted_config)
    adapted_runner = load_pack_restore_gc_checkpoint(adapted_checkpoint(seed), fresh_adapted, adapted_config)
    if source_config.policy_hidden_size != adapted_config.policy_hidden_size:
        raise AssertionError("policy architectures differ")
    return source_network, source_runner.train_state.params, adapted_network, adapted_runner.train_state.params


def choose_actions(mode, source_actions, adapted_actions, source_risk, persistent_active):
    """Pure switching rule, vectorized over episodes."""
    if mode == "source":
        use_adapted = jnp.zeros_like(source_risk)
        switch_event = jnp.zeros_like(source_risk)
    elif mode == "adapted":
        use_adapted = jnp.ones_like(source_risk)
        switch_event = jnp.zeros_like(source_risk)
    elif mode == "one_step":
        use_adapted = source_risk
        switch_event = source_risk
    elif mode == "persistent":
        switch_event = jnp.logical_and(jnp.logical_not(persistent_active), source_risk)
        use_adapted = jnp.logical_or(persistent_active, switch_event)
    else:
        raise ValueError(f"unknown execution mode: {mode}")
    chosen = jnp.where(use_adapted, adapted_actions, source_actions)
    return chosen, use_adapted, switch_event


def compact_signature(state):
    return jnp.stack(
        (
            state.player_position[:, 0],
            state.player_position[:, 1],
            state.player_direction,
            state.carried_grain,
            state.source_grain,
            state.source_age,
            state.anchor_grain,
            state.unpack_grain,
            state.packed_grain,
            state.delivered_total,
            state.anchor_present.astype(jnp.int32),
            state.unpack_present.astype(jnp.int32),
            state.packed_present.astype(jnp.int32),
        ),
        axis=-1,
    ).astype(jnp.int32)


def make_rollout(network, mode: str, kernel: str, source_growth_period: int):
    initial, labels, state_indices, repeat_indices = _evaluation_states(
        PackRestoreSplit.VALIDATION, 1, source_growth_period, natural_only=True
    )
    episode_count = int(labels.shape[0])
    if episode_count != int(SPEC["evaluation"]["states_per_seed"]):
        raise AssertionError("natural start panel changed")

    def step(parameters, carry, step_index):
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
        goals = jnp.full((episode_count,), DELIVER_3_GOAL_INDEX, dtype=jnp.int32)
        inputs = _batch_inputs(state, goals)
        source_policy, _ = network.apply(source_params, *inputs)
        adapted_policy, _ = network.apply(adapted_params, *inputs)
        source_actions = jnp.argmax(source_policy.logits, axis=-1)
        adapted_actions = jnp.argmax(adapted_policy.logits, axis=-1)
        active = jnp.logical_not(done)
        source_actions = jnp.where(active, source_actions, int(PackRestoreAction.NOOP))
        adapted_actions = jnp.where(active, adapted_actions, int(PackRestoreAction.NOOP))
        source_risk = jnp.logical_and(active, jax.vmap(inventory_overwrite_trigger)(state, source_actions))
        chosen, use_adapted, switch_event = choose_actions(
            mode, source_actions, adapted_actions, source_risk, persistent_active
        )
        chosen = jnp.where(active, chosen, int(PackRestoreAction.NOOP))
        use_adapted = jnp.logical_and(active, use_adapted)
        switch_event = jnp.logical_and(active, switch_event)
        executed_risk = jnp.logical_and(active, jax.vmap(inventory_overwrite_trigger)(state, chosen))
        before_total = jax.vmap(physical_grain_total)(state)
        if kernel == "fixed":
            stepped = jax.vmap(lambda item, action: pack_restore_step(item, action, PackRestoreVariant.FIXED))(state, chosen)
            fired = jnp.zeros((episode_count,), dtype=jnp.bool_)
        elif kernel == "inventory_overwrite_p1":
            stepped, fired = jax.vmap(
                lambda item, action: pack_restore_step_inventory_overwrite_event(item, action, True)
            )(state, chosen)
            fired = jnp.logical_and(active, fired)
        else:
            raise ValueError(f"unknown direct kernel: {kernel}")
        after_total = jax.vmap(physical_grain_total)(stepped)
        lost = jnp.where(active, jnp.maximum(before_total - after_total, 0), 0)
        delivered = jnp.where(active, stepped.delivered_total - state.delivered_total, 0)
        stored = jnp.logical_and(
            active,
            jnp.logical_and(
                chosen == int(PackRestoreAction.STORE_ONE),
                stepped.carried_grain < state.carried_grain,
            ),
        )
        emptied_by_protection = jnp.logical_and(
            jnp.logical_or(delivered > 0, stored),
            stepped.carried_grain == 0,
        )
        current_persistent = jnp.logical_or(persistent_active, switch_event) if mode == "persistent" else jnp.zeros_like(active)
        next_persistent = jnp.logical_and(current_persistent, jnp.logical_not(emptied_by_protection))
        achieved = jax.vmap(pack_restore_goal_vector)(stepped)[:, DELIVER_3_GOAL_INDEX]
        world_done = jax.vmap(pack_restore_world_done)(stepped)
        just_success = jnp.logical_and(active, achieved)
        just_done = jnp.logical_or(just_success, jnp.logical_and(active, world_done))
        next_done = jnp.logical_or(done, just_done)
        trace = {
            "pre_signature": compact_signature(state),
            "source_action": source_actions,
            "adapted_action": adapted_actions,
            "chosen_action": chosen,
            "use_adapted": use_adapted,
            "switch_event": switch_event,
            "switch_exit": jnp.logical_and(current_persistent, emptied_by_protection),
            "source_risk": source_risk,
            "executed_risk": executed_risk,
            "overwrite_fired": fired,
            "lost_grain": lost,
            "active": active,
        }
        return (
            _tree_where(active, stepped, state),
            next_done,
            jnp.logical_or(success, just_success),
            length + active.astype(jnp.int32),
            lost_grain + lost,
            jnp.logical_or(loss_seen, lost > 0),
            executed_risk_count + executed_risk.astype(jnp.int32),
            source_risk_count + source_risk.astype(jnp.int32),
            switch_count + switch_event.astype(jnp.int32),
            adapted_steps + use_adapted.astype(jnp.int32),
            jnp.logical_and(next_persistent, jnp.logical_not(next_done)),
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
        final, traces = jax.lax.scan(
            lambda carry, step_index: step((source_params, adapted_params), carry, step_index),
            initial_carry,
            jnp.arange(HORIZON, dtype=jnp.int32),
        )
        return final, traces

    return jax.jit(rollout), np.asarray(state_indices), np.asarray(repeat_indices)


def summarize_rollout(final, traces, state_indices, mode: str, kernel: str) -> dict:
    host_final = jax.device_get(final)
    host_trace = jax.device_get(traces)
    (
        _,
        done,
        success,
        length,
        lost_grain,
        loss_seen,
        executed_risk_count,
        source_risk_count,
        switch_count,
        adapted_steps,
        _,
    ) = host_final
    returns = np.asarray(success, dtype=np.float64) * np.power(DISCOUNT, np.maximum(np.asarray(length) - 1, 0))
    episodes = []
    for index, state_index in enumerate(state_indices):
        active_steps = int(length[index])
        signatures = np.asarray(host_trace["pre_signature"][:active_steps, index], dtype=np.int32)
        chosen = np.asarray(host_trace["chosen_action"][:active_steps, index], dtype=np.int32)
        use_adapted = np.asarray(host_trace["use_adapted"][:active_steps, index], dtype=bool)
        switch_events = np.asarray(host_trace["switch_event"][:active_steps, index], dtype=bool)
        switch_exits = np.asarray(host_trace["switch_exit"][:active_steps, index], dtype=bool)
        source_risks = np.asarray(host_trace["source_risk"][:active_steps, index], dtype=bool)
        executed_risks = np.asarray(host_trace["executed_risk"][:active_steps, index], dtype=bool)
        fired = np.asarray(host_trace["overwrite_fired"][:active_steps, index], dtype=bool)
        step_loss = np.asarray(host_trace["lost_grain"][:active_steps, index], dtype=np.int32)
        compact_pairs = [tuple(signature.tolist()) + (int(action), int(adapted)) for signature, action, adapted in zip(signatures, chosen, use_adapted, strict=True)]
        repeated = len(compact_pairs) - len(set(compact_pairs))
        steps = [
            {
                "step": step + 1,
                "position": signatures[step, :2].tolist(),
                "carried_grain": int(signatures[step, 3]),
                "source_grain": int(signatures[step, 4]),
                "delivered_total": int(signatures[step, 9]),
                "source_action": int(host_trace["source_action"][step, index]),
                "adapted_action": int(host_trace["adapted_action"][step, index]),
                "chosen_action": int(chosen[step]),
                "used_adapted": bool(use_adapted[step]),
                "switch_event": bool(switch_events[step]),
                "switch_exit": bool(switch_exits[step]),
                "source_risk_proposal": bool(source_risks[step]),
                "executed_risky_harvest": bool(executed_risks[step]),
                "overwrite_fired": bool(fired[step]),
                "lost_grain": int(step_loss[step]),
            }
            for step in range(active_steps)
        ]
        episodes.append(
            {
                "state_index": int(state_index),
                "layout": int(state_index) % 16,
                "phase": int(state_index) // 16,
                "success": bool(success[index]),
                "length": active_steps,
                "discounted_return": float(returns[index]),
                "loss": bool(loss_seen[index]),
                "lost_grain": int(lost_grain[index]),
                "executed_risk_count": int(executed_risk_count[index]),
                "source_risk_proposal_count": int(source_risk_count[index]),
                "switch_count": int(switch_count[index]),
                "adapted_policy_steps": int(adapted_steps[index]),
                "repeated_compact_state_action_count": int(repeated),
                "steps": steps,
            }
        )
    return {
        "mode": mode,
        "kernel": kernel,
        "aggregate": {
            "episodes": len(episodes),
            "success_rate": float(np.mean(success)),
            "mean_discounted_return": float(np.mean(returns)),
            "mean_length": float(np.mean(length)),
            "loss_episode_rate": float(np.mean(loss_seen)),
            "mean_lost_grain": float(np.mean(lost_grain)),
            "mean_executed_risk_count": float(np.mean(executed_risk_count)),
            "mean_source_risk_proposal_count": float(np.mean(source_risk_count)),
            "mean_switch_count": float(np.mean(switch_count)),
            "mean_adapted_policy_steps": float(np.mean(adapted_steps)),
            "mean_repeated_compact_state_action_count": float(np.mean([row["repeated_compact_state_action_count"] for row in episodes])),
            "completed_rate": float(np.mean(done)),
        },
        "episodes": episodes,
    }


def polynomial_profile(episodes: list[dict]) -> dict:
    powers = sorted({int(row["executed_risk_count"]) for row in episodes})
    n = len(episodes)
    success = {}
    returns = {}
    no_fire = {}
    for power in powers:
        group = [row for row in episodes if int(row["executed_risk_count"]) == power]
        success[str(power)] = math.fsum(float(row["success"]) for row in group) / n
        returns[str(power)] = math.fsum(float(row["discounted_return"]) for row in group) / n
        no_fire[str(power)] = len(group) / n
    return {
        "episode_count": n,
        "risk_count_distribution": {str(power): sum(int(row["executed_risk_count"] == power) for row in episodes) for power in powers},
        "success_polynomial_by_survival_power": success,
        "return_polynomial_by_survival_power": returns,
        "no_overwrite_polynomial_by_survival_power": no_fire,
    }


def evaluate_profile(profile: dict, probability: float) -> dict:
    survival = 1.0 - probability
    def evaluate(coefficients):
        return math.fsum(float(value) * survival ** int(power) for power, value in coefficients.items())
    success = evaluate(profile["success_polynomial_by_survival_power"])
    expected_return = evaluate(profile["return_polynomial_by_survival_power"])
    no_fire = evaluate(profile["no_overwrite_polynomial_by_survival_power"])
    return {
        "overwrite_probability": probability,
        "success_rate": success,
        "expected_discounted_return": expected_return,
        "overwrite_loss_episode_probability": 1.0 - no_fire,
    }


def prehistory_matches(fixed: dict, p1: dict) -> bool:
    fixed_steps = fixed["steps"]
    p1_steps = p1["steps"]
    first = next((index for index, step in enumerate(fixed_steps) if step["executed_risky_harvest"]), None)
    limit = len(fixed_steps) if first is None else first + 1
    if len(p1_steps) < limit:
        return False
    keys = ("position", "carried_grain", "source_grain", "delivered_total", "source_action", "adapted_action", "chosen_action", "used_adapted", "source_risk_proposal", "executed_risky_harvest")
    return all(all(fixed_steps[index][key] == p1_steps[index][key] for key in keys) for index in range(limit))


def aggregate_seed_rows(rows: list[dict]) -> dict:
    output = {}
    for mode in MODES:
        members = [row for row in rows if row["mode"] == mode]
        fixed_episodes = [episode for row in members for episode in row["fixed"]["episodes"]]
        p1_episodes = [episode for row in members for episode in row["inventory_overwrite_p1"]["episodes"]]
        profile = polynomial_profile(fixed_episodes)
        output[mode] = {
            "seeds": len(members),
            "fixed": summarize_episode_pool(fixed_episodes),
            "inventory_overwrite_p1": summarize_episode_pool(p1_episodes),
            "exact": {f"p{int(round(probability * 100)):02d}": evaluate_profile(profile, probability) for probability in PROBABILITIES},
            "risk_count_distribution": profile["risk_count_distribution"],
            "failed_paths": [
                {"seed": row["seed"], **episode}
                for row in members
                for episode in row["fixed"]["episodes"]
                if not episode["success"]
            ],
        }
    return output


def summarize_episode_pool(episodes: list[dict]) -> dict:
    return {
        "episodes": len(episodes),
        "success_rate": math.fsum(float(row["success"]) for row in episodes) / len(episodes),
        "mean_discounted_return": math.fsum(float(row["discounted_return"]) for row in episodes) / len(episodes),
        "mean_length": math.fsum(float(row["length"]) for row in episodes) / len(episodes),
        "loss_episode_rate": math.fsum(float(row["loss"]) for row in episodes) / len(episodes),
        "mean_executed_risk_count": math.fsum(float(row["executed_risk_count"]) for row in episodes) / len(episodes),
        "mean_switch_count": math.fsum(float(row["switch_count"]) for row in episodes) / len(episodes),
        "mean_adapted_policy_steps": math.fsum(float(row["adapted_policy_steps"]) for row in episodes) / len(episodes),
        "mean_repeated_compact_state_action_count": math.fsum(float(row["repeated_compact_state_action_count"]) for row in episodes) / len(episodes),
    }


def execution_manifest() -> dict:
    relative = list(SPEC["execution_sources"])
    dirty = git("status", "--porcelain", "--", *relative)
    if dirty:
        raise RuntimeError("execution sources must be committed before launch:\n" + dirty)
    tracked = set(git("ls-files", "--", *relative).splitlines())
    if set(relative) - tracked:
        raise RuntimeError(f"execution sources are untracked: {sorted(set(relative) - tracked)}")
    return {
        "schema_version": "hackrl_pack_inventory_overwrite_policy_switch_run_v1",
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
    configs = validate_configs()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    run_manifest_path = RUN_ROOT / "run_manifest.json"
    if run_manifest_path.exists() and json.loads(run_manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("run manifest differs from existing execution")
    write_json(run_manifest_path, manifest)
    rows = []
    evaluators = None
    state_indices = None
    for seed in SEEDS:
        source_network, source_params, adapted_network, adapted_params = load_policy_pair(seed, configs)
        if type(source_network) is not type(adapted_network):
            raise AssertionError("source and adapted network types differ")
        if evaluators is None:
            evaluators = {}
            for mode in MODES:
                for kernel in DIRECT_KERNELS:
                    evaluator, indices, repeats = make_rollout(source_network, mode, kernel, configs[seed]["source"].source_growth_period)
                    evaluators[(mode, kernel)] = evaluator
                    state_indices = indices
                    if np.any(repeats != 0):
                        raise AssertionError("mode evaluation repeat index changed")
        for mode in MODES:
            direct = {}
            for kernel in DIRECT_KERNELS:
                final, traces = evaluators[(mode, kernel)](source_params, adapted_params)
                direct[kernel] = summarize_rollout(final, traces, state_indices, mode, kernel)
            profile = polynomial_profile(direct["fixed"]["episodes"])
            exact = {f"p{int(round(probability * 100)):02d}": evaluate_profile(profile, probability) for probability in PROBABILITIES}
            checks = {
                "p0_return_matches_direct_fixed": math.isclose(exact["p00"]["expected_discounted_return"], direct["fixed"]["aggregate"]["mean_discounted_return"], abs_tol=1e-12),
                "p0_success_matches_direct_fixed": math.isclose(exact["p00"]["success_rate"], direct["fixed"]["aggregate"]["success_rate"], abs_tol=1e-12),
                "p1_return_matches_direct_always_fire": math.isclose(exact["p100"]["expected_discounted_return"], direct["inventory_overwrite_p1"]["aggregate"]["mean_discounted_return"], abs_tol=1e-12),
                "p1_success_matches_direct_always_fire": math.isclose(exact["p100"]["success_rate"], direct["inventory_overwrite_p1"]["aggregate"]["success_rate"], abs_tol=1e-12),
                "all_first_overwrite_prehistories_match": all(prehistory_matches(left, right) for left, right in zip(direct["fixed"]["episodes"], direct["inventory_overwrite_p1"]["episodes"], strict=True)),
                "every_p1_success_has_zero_fixed_path_risk": all(not right["success"] or left["executed_risk_count"] == 0 for left, right in zip(direct["fixed"]["episodes"], direct["inventory_overwrite_p1"]["episodes"], strict=True)),
            }
            if not all(checks.values()):
                raise AssertionError(f"endpoint validation failed: seed={seed} mode={mode} {checks}")
            rows.append({"seed": seed, "mode": mode, "fixed": direct["fixed"], "inventory_overwrite_p1": direct["inventory_overwrite_p1"], "risk_profile": profile, "exact": exact, "checks": checks})
        print(f"[seed] {seed} complete", flush=True)

    aggregate = aggregate_seed_rows(rows)
    transfer = json.loads(Path(SPEC["prerequisites"]["transfer_raw_result"]).read_text(encoding="utf-8"))
    gates = {
        "all_1280_direct_episodes_recorded": sum(row[kernel]["aggregate"]["episodes"] for row in rows for kernel in DIRECT_KERNELS) == int(SPEC["budget"]["episodes"]),
        "all_endpoint_checks_pass": all(all(row["checks"].values()) for row in rows),
        "source_p05_matches_pinned_result": math.isclose(aggregate["source"]["exact"]["p05"]["expected_discounted_return"], transfer["source_baseline"]["means"]["cross_probability_exact"]["p05"]["expected_discounted_return"], abs_tol=1e-12),
        "adapted_p05_matches_pinned_result": math.isclose(aggregate["adapted"]["exact"]["p05"]["expected_discounted_return"], transfer["aggregates"]["p100_lr1"]["cross_probability_exact_mean"]["p05"]["expected_discounted_return"], abs_tol=1e-12),
        "no_learning_updates": int(SPEC["budget"]["learning_updates"]) == 0,
    }
    if not all(gates.values()):
        raise AssertionError(f"result gates failed: {gates}")
    result = {
        "schema_version": "hackrl_pack_inventory_overwrite_policy_switch_result_v1",
        "status": "complete_no_learning",
        "generated_unix": time.time(),
        "execution_head": manifest["execution_head"],
        "manifest_sha256": manifest["manifest_sha256"],
        "gates": gates,
        "design": validation,
        "aggregate": aggregate,
        "rows": rows,
        "interpretation_guard": SPEC["analysis"]["non_claims"],
    }
    write_json(RUN_ROOT / "result.json", result)
    write_json(RUN_ROOT / "complete.json", {"status": "complete_no_learning", "execution_head": manifest["execution_head"], "result_sha256": sha256(RUN_ROOT / "result.json")})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-contract", action="store_true")
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if arguments.validate_contract == arguments.run:
        raise SystemExit("choose exactly one of --validate-contract or --run")
    output = validate_contract() if arguments.validate_contract else run()
    print(json.dumps(output if arguments.validate_contract else {"status": output["status"], "run_root": str(RUN_ROOT)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
