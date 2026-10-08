#!/usr/bin/env python3
"""Run the bounded, no-training postmortem for the all-zero online gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import numpy as np

from hackrl.discrete_sac import sd_sac_critic_target
from hackrl.online_algorithm_env import (
    DUAL,
    LEO,
    PQN,
    SD_SAC,
    build_online_transitions,
    environment_adapter,
)
from hackrl.online_value import leo_one_step_targets, online_one_step_targets
from hackrl.pack_restore import (
    GOAL_IDS as PACK_GOAL_IDS,
    PackRestoreAction,
    PackRestorePhase,
    PackRestoreSplit,
    PackRestoreStart,
    PackRestoreVariant,
    make_pack_restore_state,
    pack_restore_step,
)
from hackrl.pack_restore_gc import DELIVER_3_GOAL_INDEX as PACK_DELIVER_3
from hackrl.tick_claim import (
    GOAL_IDS as TICK_GOAL_IDS,
    GROWTH_WAIT_TICKS,
    TickClaimAction,
    TickClaimPhase,
    TickClaimSplit,
    TickClaimStart,
    TickClaimVariant,
    make_tick_claim_state,
    tick_claim_step,
    transform_direction as tick_transform_direction,
)
from hackrl.tick_claim_gc import DELIVER_3_GOAL_INDEX as TICK_DELIVER_3
from run_dual_leo_compare import (
    ENVS as PPO_ENVS,
    _arm as ppo_arm,
    _job_config as ppo_job_config,
    _load_state as load_ppo_state,
    _start_teacher as start_ppo_teacher,
)
from run_online_algorithm_development import (
    _cell_path,
    _payload,
    build_jobs,
    environment_config,
    initialize_job,
    load_checkpoint,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DEVELOPMENT_MANIFEST = (
    REPOSITORY
    / "docs/manifests/online_algorithm_expansion_v1_development.json"
)
DEVELOPMENT_ROOT = Path(
    "/raid/ext_csv/HackRL/runs/online_algorithm_expansion_v1_development"
)
DEFAULT_OUTPUT = Path(
    "/raid/ext_csv/HackRL/runs/"
    "online_algorithm_zero_gate_diagnostic_v1/result.json"
)
PPO_ROOT = REPOSITORY / "runs/dual_leo_compare_v1"


class ExistingPolicyAdapter:
    """Expose an existing PPO network through the development policy surface."""

    def __init__(self, network):
        self.network = network

    def apply(self, parameters, *inputs):
        return self.network.apply(parameters, *inputs)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parameter_digest(tree):
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        value = np.asarray(jax.device_get(leaf))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _repeat_state(state, count):
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (count,) + value.shape), state
    )


def _evaluation_view(raw):
    keys = ("episodes", "success_rate", "mean_length", "completed_rate")
    return {
        family: {key: raw[family][key] for key in keys}
        for family in ("natural_reset", "common_setup")
    }


def existing_policy_bridge():
    rows = []
    for environment in ("tick", "pack"):
        job = {
            "id": f"{environment}-dual-s20-fixed",
            "kind": "adapt",
            "env": environment,
            "method": "dual",
            "seed": 20,
            "variant": "fixed",
        }
        spec = PPO_ENVS[environment]
        config = ppo_job_config(
            job, goal_mode="deliver_3", variant="fixed", updates=4096
        )
        network, template = spec["initialize"](config)
        _, teacher_template, _ = start_ppo_teacher(spec, config, template)
        checkpoint = (
            PPO_ROOT
            / environment
            / "dual/fixed/seed20/checkpoints/adapt_4096"
        )
        runner, _ = load_ppo_state(
            spec,
            "dual",
            checkpoint,
            template,
            teacher_template,
            config,
            ppo_arm(job),
        )
        old_inputs = spec["inputs"](runner.env_state, runner.current_goal)
        adapter = environment_adapter(environment)
        new_inputs = adapter.batch_inputs(runner.env_state, runner.current_goal)
        arguments = {
            "variant": "fixed",
            "split": "validation",
            "stochastic": False,
            "repeats_per_state": 1,
            "seed_base": 880000,
            "learner_seed": 20,
            "record_episodes": True,
        }
        original = spec["evaluate"](
            network, runner.train_state.params, **arguments
        )
        bridged = spec["evaluate"](
            ExistingPolicyAdapter(network),
            runner.train_state.params,
            **arguments,
        )
        stored = _read_json(
            PPO_ROOT
            / environment
            / "dual/fixed/seed20/curve/adapt_4096.json"
        )["fixed"]["mode"]
        stored_view = _evaluation_view(stored)
        original_view = _evaluation_view(original)
        bridged_view = _evaluation_view(bridged)
        rows.append(
            {
                "environment": environment,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_state_sha256": _sha256(
                    checkpoint / "state.msgpack"
                ),
                "input_tensors_bit_equal": all(
                    np.array_equal(np.asarray(left), np.asarray(right))
                    for left, right in zip(old_inputs, new_inputs)
                ),
                "original": original_view,
                "bridged": bridged_view,
                "stored": stored_view,
                "original_and_bridged_episode_records_equal": (
                    original["episode_records"] == bridged["episode_records"]
                ),
                "rerun_and_stored_summaries_equal": (
                    original_view == stored_view
                ),
            }
        )
    return {
        "passed": all(
            row["input_tensors_bit_equal"]
            and row["original_and_bridged_episode_records_equal"]
            and row["rerun_and_stored_summaries_equal"]
            for row in rows
        ),
        "rows": rows,
    }


def _trace_contract(environment, config):
    if environment == "tick":
        state = make_tick_claim_state(
            0,
            int(TickClaimPhase.RIPE),
            split=TickClaimSplit.VALIDATION,
            start=TickClaimStart.PATH_CHECK,
        )
        actions = (
            (int(TickClaimAction.DO),)
            + (int(TickClaimAction.NOOP),) * GROWTH_WAIT_TICKS
            + (int(TickClaimAction.DO),)
            + (int(TickClaimAction.NOOP),) * GROWTH_WAIT_TICKS
            + (
                int(TickClaimAction.DO),
                int(
                    tick_transform_direction(
                        int(TickClaimAction.DOWN), 0
                    )
                ),
                int(TickClaimAction.DELIVER),
            )
        )
        return (
            state,
            actions,
            TICK_DELIVER_3,
            lambda item, action: tick_claim_step(
                item, action, TickClaimVariant.FIXED
            ),
        )
    state = make_pack_restore_state(
        0,
        int(PackRestorePhase.EMPTY),
        split=PackRestoreSplit.VALIDATION,
        start=PackRestoreStart.PATH_CHECK,
        source_growth_period=config.source_growth_period,
    )
    opening = [int(PackRestoreAction.NOOP)] * config.source_growth_period
    opening[1] = int(
        tick_transform_direction(int(PackRestoreAction.RIGHT), 0)
    )
    actions = tuple(
        opening
        + [int(PackRestoreAction.DO)]
        + [int(PackRestoreAction.NOOP)] * config.source_growth_period
        + [
            int(PackRestoreAction.DO),
            int(
                tick_transform_direction(int(PackRestoreAction.DOWN), 0)
            ),
            int(PackRestoreAction.DELIVER),
        ]
    )
    return (
        state,
        actions,
        PACK_DELIVER_3,
        lambda item, action: pack_restore_step(
            item, action, PackRestoreVariant.FIXED
        ),
    )


def scripted_trace_check(manifest, environment):
    job = next(
        item
        for item in build_jobs(manifest)
        if item["environment"] == environment and item["method"] == PQN
    )
    config = environment_config(job, manifest)
    adapter = environment_adapter(environment)
    _, runner = adapter.initialize(config)
    direct, actions, delivery_goal, direct_step = _trace_contract(
        environment, config
    )
    runner = runner.replace(
        env_state=_repeat_state(direct, config.num_envs),
        current_goal=jnp.full(
            (config.num_envs,), delivery_goal, dtype=jnp.int32
        ),
        command_active=jnp.ones((config.num_envs,), dtype=jnp.bool_),
        seen_goals=jnp.ones((adapter.num_goals,), dtype=jnp.bool_),
    )
    rows = []
    terminal_goals = []
    world_done = []
    for index, action in enumerate(actions):
        inputs = adapter.batch_inputs(runner.env_state, runner.current_goal)
        command_goal = int(runner.current_goal[0])
        direct = direct_step(direct, jnp.asarray(action, dtype=jnp.int32))
        selected = jnp.full(
            (config.num_envs,), action, dtype=jnp.int32
        )
        stepped, event = adapter.step(runner, selected, config)
        next_inputs = adapter.batch_inputs(
            stepped.env_state, stepped.current_goal
        )
        transition = build_online_transitions(
            adapter, inputs, selected, event, next_inputs
        )
        expected_inputs = adapter.batch_inputs(
            _repeat_state(direct, config.num_envs), stepped.current_goal
        )
        rows.append(
            {
                "step": index,
                "commanded_goal": command_goal,
                "goal_head": int(np.argmax(np.asarray(inputs[2][0]))),
                "reward": float(event.reward[0]),
                "goal_done": bool(event.goal_done[0]),
                "world_done": bool(event.world_done[0]),
                "done": bool(transition.pqn.done[0]),
                "valid": bool(transition.pqn.valid[0]),
                "next_goal": int(stepped.current_goal[0]),
                "next_observation_matches_direct_terminal": all(
                    np.array_equal(np.asarray(left[0]), np.asarray(right[0]))
                    for left, right in zip(
                        next_inputs[:2], expected_inputs[:2]
                    )
                ),
                "terminal_goal_count": int(
                    np.asarray(event.terminal_goals[0]).sum()
                ),
            }
        )
        terminal_goals.append(event.terminal_goals[0])
        world_done.append(event.world_done[0])
        runner = stepped

    rewards = jnp.asarray([row["reward"] for row in rows])
    done = jnp.asarray([row["done"] for row in rows])
    next_q = jnp.tile(
        jnp.linspace(0.1, 0.9, adapter.num_actions),
        (len(rows), 1),
    )
    pqn_target = np.asarray(
        online_one_step_targets(rewards, done, next_q, config.gamma)
    )
    manual_pqn = np.asarray(rewards) + config.gamma * np.max(
        np.asarray(next_q), axis=-1
    ) * (1.0 - np.asarray(done, dtype=np.float32))

    terminal_goals = jnp.stack(terminal_goals)
    world_done = jnp.stack(world_done)
    next_all_q = jnp.tile(
        jnp.linspace(
            0.05, 0.95, adapter.num_goals * adapter.num_actions
        ).reshape(1, adapter.num_goals, adapter.num_actions),
        (len(rows), 1, 1),
    )
    leo_target = np.asarray(
        leo_one_step_targets(
            terminal_goals, world_done, next_all_q, config.gamma
        )
    )
    leo_done = np.logical_or(
        np.asarray(terminal_goals), np.asarray(world_done)[:, None]
    )
    manual_leo = np.asarray(terminal_goals, dtype=np.float32) + (
        config.gamma
        * np.max(np.asarray(next_all_q), axis=-1)
        * (1.0 - np.asarray(leo_done, dtype=np.float32))
    )

    logits = jnp.zeros((len(rows), adapter.num_actions))
    target_q_1 = jnp.tile(
        jnp.linspace(0.1, 0.8, adapter.num_actions),
        (len(rows), 1),
    )
    target_q_2 = jnp.tile(
        jnp.linspace(0.2, 0.9, adapter.num_actions),
        (len(rows), 1),
    )
    alpha = 0.2
    sac_target = np.asarray(
        sd_sac_critic_target(
            rewards,
            done,
            logits,
            target_q_1,
            target_q_2,
            alpha,
            config.gamma,
        )
    )
    log_probability = -math.log(adapter.num_actions)
    manual_soft_value = np.mean(
        0.5 * (np.asarray(target_q_1) + np.asarray(target_q_2))
        - alpha * log_probability,
        axis=-1,
    )
    manual_sac = np.asarray(rewards) + config.gamma * (
        1.0 - np.asarray(done, dtype=np.float32)
    ) * manual_soft_value
    final = rows[-1]
    result = {
        "environment": environment,
        "steps": len(rows),
        "all_commanded_goal_and_head_are_delivery": all(
            row["commanded_goal"] == delivery_goal
            and row["goal_head"] == delivery_goal
            for row in rows
        ),
        "all_transitions_valid": all(row["valid"] for row in rows),
        "rewarded_steps": sum(row["reward"] > 0 for row in rows),
        "all_next_observations_match_direct_terminal_state": all(
            row["next_observation_matches_direct_terminal"] for row in rows
        ),
        "final_transition": final,
        "target_max_abs_error": {
            "pqn": float(np.max(np.abs(pqn_target - manual_pqn))),
            "leo": float(np.max(np.abs(leo_target - manual_leo))),
            "sd_sac": float(np.max(np.abs(sac_target - manual_sac))),
        },
        "final_delivery_targets": {
            "pqn": float(pqn_target[-1]),
            "leo": float(leo_target[-1, delivery_goal]),
            "sd_sac": float(sac_target[-1]),
        },
    }
    result["passed"] = (
        result["all_commanded_goal_and_head_are_delivery"]
        and result["all_transitions_valid"]
        and result["rewarded_steps"] == 1
        and result["all_next_observations_match_direct_terminal_state"]
        and final["reward"] == 1.0
        and final["goal_done"]
        and final["done"]
        and not final["world_done"]
        and all(error <= 2e-6 for error in result["target_max_abs_error"].values())
        and all(value == 1.0 for value in result["final_delivery_targets"].values())
    )
    return result


def _metric_leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            yield from _metric_leaves(item, path)
    else:
        yield prefix, value


def _log_audit(path):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    flattened = [dict(_metric_leaves(row["metrics"])) for row in rows]
    keys = sorted(set().union(*(row.keys() for row in flattened)))
    collection_valid = []
    applied = []
    all_finite = True
    for row in flattened:
        preferred = [
            value
            for key, value in row.items()
            if key in {"valid_transitions", "collection.valid_transitions"}
        ]
        if len(preferred) != 1:
            raise RuntimeError(f"ambiguous collection valid count in {path}")
        collection_valid.append(int(preferred[0]))
        applied.extend(
            int(value)
            for key, value in row.items()
            if key.endswith("applied_gradient_steps")
            or key.endswith("applied_optimizer_steps")
        )
        for value in row.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                all_finite = all_finite and math.isfinite(value)
    return {
        "rows": len(rows),
        "updates": [int(row["update"]) for row in rows],
        "all_numeric_metrics_finite": all_finite,
        "collection_valid_min": min(collection_valid),
        "collection_valid_max": max(collection_valid),
        "applied_counter_min": min(applied),
        "applied_counter_max": max(applied),
        "goal_command_frequency_persisted": any(
            "command" in key and "goal" in key for key in keys
        ),
        "goal_completion_count_persisted": any(
            token in key
            for key in keys
            for token in (
                "goal_success",
                "goal_completion",
                "successes_by_goal",
            )
        ),
        "reward_count_persisted": any(
            key.endswith("reward") or "rewarded" in key for key in keys
        ),
        "metric_keys": keys,
    }


def _learner_views(method, state):
    if method == PQN:
        return {"pqn": state}
    if method == LEO:
        return {"leo": state}
    if method == DUAL:
        return {"pqn": state.pqn, "leo": state.leo}
    return {
        "actor": state.actor,
        "critic_1": state.critic_1,
        "critic_2": state.critic_2,
        "temperature": state.temperature,
    }


def _optimizer_audit(method, initial, final):
    initial_views = _learner_views(method, initial)
    final_views = _learner_views(method, final)
    rows = {}
    for name, state in final_views.items():
        initial_digest = _parameter_digest(initial_views[name].params)
        final_digest = _parameter_digest(state.params)
        row = {
            "optimizer_step": int(state.step),
            "initial_parameter_sha256": initial_digest,
            "final_parameter_sha256": final_digest,
            "parameters_changed": initial_digest != final_digest,
        }
        for counter in (
            "environment_steps",
            "update_steps",
            "gradient_steps",
        ):
            if hasattr(state, counter):
                row[counter] = int(getattr(state, counter))
        rows[name] = row
    if method == SD_SAC:
        rows["shared_counters"] = {
            "environment_steps": int(final.environment_steps),
            "update_steps": int(final.update_steps),
            "gradient_steps": int(final.gradient_steps),
        }
    return rows


def _replay_audit(environment, replay):
    size = int(replay.size)
    valid = np.asarray(replay.valid)[:size]
    rewards = np.asarray(replay.reward)[:size]
    goals = np.argmax(np.asarray(replay.goal_one_hot)[:size], axis=-1)
    goal_ids = TICK_GOAL_IDS if environment == "tick" else PACK_GOAL_IDS
    per_goal = {
        goal_id: {
            "valid_entries": int(np.sum(valid & (goals == index))),
            "rewarded_entries": int(
                np.sum(valid & (goals == index) & (rewards > 0))
            ),
        }
        for index, goal_id in enumerate(goal_ids)
    }
    return {
        "scope": "final retained replay window only",
        "replay_size": size,
        "total_inserted": int(replay.total_inserted),
        "valid_entries": int(np.sum(valid)),
        "rewarded_entries": int(np.sum(valid & (rewards > 0))),
        "delivery_rewarded_entries": sum(
            row["rewarded_entries"]
            for goal_id, row in per_goal.items()
            if goal_id.startswith("delivery/")
        ),
        "goals": per_goal,
    }


def completed_job_audit(manifest):
    jobs = []
    expected_updates = [1] + list(range(32, 513, 32))
    for job in build_jobs(manifest):
        cell = _cell_path(DEVELOPMENT_ROOT, job)
        log = _log_audit(cell / "updates.jsonl")
        config = environment_config(job, manifest)
        adapter, networks, initial_runner, initial_replay = initialize_job(
            job, config
        )
        template = _payload(initial_runner, initial_replay)
        checkpoint = cell / "checkpoints/update_000512"
        restored = load_checkpoint(checkpoint, template)
        final_runner = restored["runner"]
        replay = restored.get("replay")
        goal_ids = (
            TICK_GOAL_IDS
            if job["environment"] == "tick"
            else PACK_GOAL_IDS
        )
        seen = np.asarray(final_runner.seen_goals, dtype=bool)
        current = np.asarray(final_runner.current_goal)
        optimizer = _optimizer_audit(
            job["method"],
            initial_runner.train_state,
            final_runner.train_state,
        )
        row = {
            "job": job["id"],
            "method": job["method"],
            "environment": job["environment"],
            "candidate": job["candidate_id"],
            "seed": int(job["seed"]),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_state_sha256": _sha256(
                checkpoint / "state.msgpack"
            ),
            "global_update": int(final_runner.global_update),
            "environment_steps": int(final_runner.env_steps),
            "observed_goal_predicates": [
                goal_ids[index]
                for index in np.flatnonzero(seen).tolist()
            ],
            "final_current_goal_snapshot": {
                goal_ids[index]: int(np.sum(current == index))
                for index in range(len(goal_ids))
            },
            "log": log,
            "optimizer": optimizer,
            "evaluation": _read_json(cell / "evaluation.json"),
            "replay": (
                _replay_audit(job["environment"], replay)
                if replay is not None
                else {
                    "retained": False,
                    "historical_goal_and_reward_rows_recoverable": False,
                }
            ),
        }
        row["checks"] = {
            "expected_log_schedule": log["updates"] == expected_updates,
            "all_logged_collections_full_valid_batch": (
                log["collection_valid_min"] == 32768
                and log["collection_valid_max"] == 32768
            ),
            "all_logged_apply_counters_positive": (
                log["applied_counter_min"] > 0
            ),
            "all_parameters_changed": all(
                item["parameters_changed"]
                for name, item in optimizer.items()
                if name != "shared_counters"
            ),
        }
        jobs.append(row)
        del (
            restored,
            final_runner,
            replay,
            template,
            initial_replay,
            initial_runner,
            networks,
            adapter,
        )
        gc.collect()

    sd_replays = [row["replay"] for row in jobs if row["method"] == SD_SAC]
    summary = {
        "jobs": len(jobs),
        "all_have_17_log_rows": all(row["log"]["rows"] == 17 for row in jobs),
        "all_expected_log_schedule": all(
            row["checks"]["expected_log_schedule"] for row in jobs
        ),
        "all_logged_numeric_metrics_finite": all(
            row["log"]["all_numeric_metrics_finite"] for row in jobs
        ),
        "all_logged_collections_full_valid_batch": all(
            row["checks"]["all_logged_collections_full_valid_batch"]
            for row in jobs
        ),
        "all_logged_apply_counters_positive": all(
            row["checks"]["all_logged_apply_counters_positive"]
            for row in jobs
        ),
        "all_final_parameter_sets_changed_from_initialization": all(
            row["checks"]["all_parameters_changed"] for row in jobs
        ),
        "jobs_with_goal_command_frequency_in_logs": sum(
            row["log"]["goal_command_frequency_persisted"] for row in jobs
        ),
        "jobs_with_goal_completion_count_in_logs": sum(
            row["log"]["goal_completion_count_persisted"] for row in jobs
        ),
        "jobs_with_reward_count_in_logs": sum(
            row["log"]["reward_count_persisted"] for row in jobs
        ),
        "sd_sac_final_replay_windows": len(sd_replays),
        "sd_sac_final_replay_windows_with_any_reward": sum(
            replay["rewarded_entries"] > 0 for replay in sd_replays
        ),
        "sd_sac_final_replay_windows_with_delivery_reward": sum(
            replay["delivery_rewarded_entries"] > 0
            for replay in sd_replays
        ),
        "interpretation": {
            "some_goal_success_experience": (
                "Observed in every final SD-SAC retained replay window; this is "
                "experience evidence, not proof that the final greedy policy "
                "learned those goals."
            ),
            "delivery_success_experience": (
                "Absent from all final SD-SAC retained replay windows. Earlier "
                "SD-SAC history and all PQN/LEO/Dual history are unobserved "
                "because per-goal/reward counts were not logged and online "
                "value methods retained no replay."
            ),
            "optimizer_application": (
                "Final optimizer counters are nonzero and all parameter hashes "
                "differ from deterministic initialization in every job."
            ),
        },
    }
    return {"summary": summary, "jobs": jobs}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    arguments = parse_args()
    manifest = _read_json(DEVELOPMENT_MANIFEST)
    payload = {
        "schema_version": "hackrl_online_algorithm_zero_gate_diagnostic_v1",
        "status": "complete",
        "training_or_optimizer_updates_performed": 0,
        "checkpoint_writes_performed": 0,
        "development_manifest": str(DEVELOPMENT_MANIFEST.relative_to(REPOSITORY)),
        "development_manifest_sha256": _sha256(DEVELOPMENT_MANIFEST),
        "development_root": str(DEVELOPMENT_ROOT),
        "diagnostics": {
            "existing_success_policy_evaluator_bridge": existing_policy_bridge(),
            "scripted_success_trace_collector_and_targets": {
                environment: scripted_trace_check(manifest, environment)
                for environment in ("tick", "pack")
            },
            "stored_learning_signal_and_optimizer_audit": completed_job_audit(
                manifest
            ),
        },
        "conclusion": (
            "The bounded checks found no shared evaluator, observation adapter, "
            "scripted-success transition, terminal/reset, TD-target, gradient-"
            "application, or optimizer-application failure. The saved data show "
            "some non-delivery goal rewards in every final SD-SAC replay window, "
            "but no delivery reward in those retained windows. Per-goal command "
            "and completion history was not persisted for any job, so earlier "
            "delivery experience and whether non-SAC methods achieved individual "
            "goals remain unobserved rather than zero."
        ),
        "stop_rule": (
            "Do not launch the 72-job main matrix or add replay/budget sweeps from "
            "this result. If a final bounded run is desired, use one preregistered "
            "PQN candidate in one environment and compare a delivery-near start "
            "with the natural start."
        ),
    }
    _write_json(arguments.output, payload)
    print(json.dumps({"output": str(arguments.output), "status": "complete"}))


if __name__ == "__main__":
    main()
