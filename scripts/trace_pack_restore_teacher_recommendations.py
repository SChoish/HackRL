#!/usr/bin/env python3
"""Trace teacher recommendations on policy-visited PACK exploitation states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from hackrl.dual_leo import load_dual_checkpoint
from hackrl.pack_restore import (
    PackRestoreAction,
    PackRestoreSplit,
    PackRestoreVariant,
    pack_restore_goal_vector,
    pack_restore_step,
    pack_restore_world_done,
)
from hackrl.pack_restore_gc import (
    DELIVER_3_GOAL_INDEX,
    NUM_ACTIONS,
    NUM_GOALS,
    PackRestoreGCActorCritic,
    _batch_inputs,
    _evaluation_states,
    config_from_pack_restore_gc_payload,
    initialize_pack_restore_gc,
)
from hackrl.dual_leo import init_dual_leo_teacher

from run_pack_restore_pretrained_teacher_freeze import (
    ADAPT_UPDATES,
    SEEDS,
    SOURCE_RUN_ROOT,
    build_jobs,
)
from run_dual_leo_compare import _cell_dir


CONDITIONS = ("D-off", "D-frozen", "D-on")
TARGET_ACTIONS = (
    PackRestoreAction.MAKE_RECORD,
    PackRestoreAction.PACK_STORAGE,
    PackRestoreAction.REBUILD_EMPTY,
    PackRestoreAction.WITHDRAW_ONE,
)
ACTION_NAMES = {int(action): action.name for action in PackRestoreAction}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_for(run_root, condition, seed):
    if condition == "D-frozen":
        job = next(
            job
            for job in build_jobs()
            if job["seed"] == seed and job["variant"] == "mutant"
        )
        cell = _cell_dir(run_root, job)
    else:
        method = {"D-on": "dual", "D-off": "dual_bc_off"}[condition]
        cell = SOURCE_RUN_ROOT / "size_s" / "pack" / method / "mutant" / f"seed{seed}"
    return cell / "checkpoints" / f"adapt_{ADAPT_UPDATES}"


def _successful_stage(action, state, stepped):
    return jnp.logical_or(
        jnp.logical_and(
            action == int(PackRestoreAction.MAKE_RECORD),
            jnp.logical_and(jnp.logical_not(state.record_present), stepped.record_present),
        ),
        jnp.logical_or(
            jnp.logical_and(
                action == int(PackRestoreAction.PACK_STORAGE),
                jnp.logical_or(
                    stepped.packed_present,
                    stepped.empty_frames > state.empty_frames,
                ),
            ),
            jnp.logical_or(
                jnp.logical_and(
                    action == int(PackRestoreAction.REBUILD_EMPTY),
                    jnp.logical_and(
                        jnp.logical_not(state.anchor_present),
                        stepped.anchor_present,
                    ),
                ),
                jnp.logical_and(
                    action == int(PackRestoreAction.WITHDRAW_ONE),
                    jnp.logical_and(
                        state.record_present,
                        stepped.carried_grain > state.carried_grain,
                    ),
                ),
            ),
        ),
    )


def _make_rollout(network, teacher_network, initial):
    episode_count = initial.tick.shape[0]
    target_mask = jnp.zeros((NUM_ACTIONS,), dtype=jnp.bool_).at[
        jnp.asarray([int(action) for action in TARGET_ACTIONS])
    ].set(True)

    def rollout(policy_params, teacher_params, teacher_batch_stats):
        def step(carry, step_index):
            state, done = carry
            goals = jnp.full(
                (episode_count,), DELIVER_3_GOAL_INDEX, dtype=jnp.int32
            )
            inputs = _batch_inputs(state, goals)
            policy, _ = network.apply(policy_params, *inputs)
            actions = jnp.argmax(policy.logits, axis=-1)
            q_values = teacher_network.apply(
                {"params": teacher_params, "batch_stats": teacher_batch_stats},
                inputs[0],
                inputs[1],
                train=False,
            )[:, DELIVER_3_GOAL_INDEX, :]
            recommendations = jnp.argmax(q_values, axis=-1)
            active = jnp.logical_not(done)
            actions = jnp.where(active, actions, int(PackRestoreAction.NOOP))
            stepped = jax.vmap(
                lambda item, action: pack_restore_step(
                    item, action, PackRestoreVariant.MUTANT
                )
            )(state, actions)
            achieved = jax.vmap(pack_restore_goal_vector)(stepped)[
                :, DELIVER_3_GOAL_INDEX
            ]
            world_done = jax.vmap(pack_restore_world_done)(stepped)
            done_next = jnp.logical_or(done, jnp.logical_or(achieved, world_done))
            attempts = jnp.logical_and(active, target_mask[actions])
            successful = jnp.logical_and(
                attempts,
                jax.vmap(_successful_stage)(actions, state, stepped),
            )
            observation = {
                "step": jnp.full((episode_count,), step_index, dtype=jnp.int32),
                "active": active,
                "attempt": attempts,
                "successful": successful,
                "policy_action": actions,
                "teacher_action": recommendations,
                "teacher_q": q_values,
                "tick": state.tick,
                "player_position": state.player_position,
                "player_direction": state.player_direction,
                "carried_grain": state.carried_grain,
                "anchor_present": state.anchor_present,
                "anchor_grain": state.anchor_grain,
                "unpack_present": state.unpack_present,
                "unpack_grain": state.unpack_grain,
                "empty_frames": state.empty_frames,
                "packed_present": state.packed_present,
                "packed_grain": state.packed_grain,
                "record_present": state.record_present,
                "record_grain_preview": state.record_grain_preview,
                "delivered_total": state.delivered_total,
            }
            return (stepped, done_next), observation

        (_, _), observations = jax.lax.scan(
            step,
            (initial, jnp.zeros((episode_count,), dtype=jnp.bool_)),
            jnp.arange(128, dtype=jnp.int32),
        )
        return observations

    return jax.jit(rollout)


def _host_records(observations, labels, state_indices, condition, seed):
    values = jax.device_get(observations)
    attempts = np.asarray(values["attempt"])
    records = []
    for step, episode in np.argwhere(attempts):
        policy_action = int(values["policy_action"][step, episode])
        teacher_action = int(values["teacher_action"][step, episode])
        teacher_q = np.asarray(values["teacher_q"][step, episode])
        order = np.argsort(teacher_q)
        records.append(
            {
                "condition": condition,
                "seed": seed,
                "trained_variant": "mutant",
                "evaluation_kernel": "mutant",
                "policy": "mode",
                "family": (
                    "natural_reset" if int(labels[episode]) == 0 else "common_setup"
                ),
                "validation_state_index": int(state_indices[episode]),
                "episode_index": int(episode),
                "rollout_step": int(step),
                "world_tick_before": int(values["tick"][step, episode]),
                "policy_action": ACTION_NAMES[policy_action],
                "teacher_recommendation": ACTION_NAMES[teacher_action],
                "teacher_recommendation_matches_policy": teacher_action == policy_action,
                "teacher_q_for_policy_action": float(teacher_q[policy_action]),
                "teacher_q_for_recommendation": float(teacher_q[teacher_action]),
                "teacher_top2_gap": float(
                    teacher_q[order[-1]] - teacher_q[order[-2]]
                ),
                "teacher_q_by_action": {
                    ACTION_NAMES[index]: float(value)
                    for index, value in enumerate(teacher_q)
                },
                "stage_action_succeeded": bool(
                    values["successful"][step, episode]
                ),
                "pre_state": {
                    "player_position": [
                        int(value)
                        for value in values["player_position"][step, episode]
                    ],
                    "player_direction": int(
                        values["player_direction"][step, episode]
                    ),
                    "carried_grain": int(values["carried_grain"][step, episode]),
                    "anchor_present": bool(values["anchor_present"][step, episode]),
                    "anchor_grain": int(values["anchor_grain"][step, episode]),
                    "unpack_present": bool(values["unpack_present"][step, episode]),
                    "unpack_grain": int(values["unpack_grain"][step, episode]),
                    "empty_frames": int(values["empty_frames"][step, episode]),
                    "packed_present": bool(values["packed_present"][step, episode]),
                    "packed_grain": int(values["packed_grain"][step, episode]),
                    "record_present": bool(values["record_present"][step, episode]),
                    "record_grain_preview": int(
                        values["record_grain_preview"][step, episode]
                    ),
                    "delivered_total": int(values["delivered_total"][step, episode]),
                },
            }
        )
    return records


def trace(run_root):
    run_root = Path(run_root).resolve()
    first_checkpoint = checkpoint_for(run_root, "D-frozen", SEEDS[0])
    config = config_from_pack_restore_gc_payload(
        _read(first_checkpoint / "config.json")
    )
    network, template = initialize_pack_restore_gc(config)
    example_inputs = _batch_inputs(template.env_state, template.current_goal)
    teacher_network, leo_template, _ = init_dual_leo_teacher(
        config, example_inputs[0], example_inputs[1], NUM_GOALS, NUM_ACTIONS
    )
    initial, labels, state_indices, _ = _evaluation_states(
        PackRestoreSplit.VALIDATION,
        1,
        config.source_growth_period,
    )
    rollout = _make_rollout(network, teacher_network, initial)
    labels_host = np.asarray(labels)
    indices_host = np.asarray(state_indices)
    records = []
    checkpoints = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            checkpoint = checkpoint_for(run_root, condition, seed)
            recorded = config_from_pack_restore_gc_payload(
                _read(checkpoint / "config.json")
            )
            if any(
                getattr(recorded, field) != getattr(config, field)
                for field in (
                    "num_envs", "num_steps", "policy_hidden_size",
                    "teacher_hidden_size", "goal_mode", "source_growth_period",
                )
            ):
                raise RuntimeError(f"trace checkpoint config mismatch: {checkpoint}")
            runner, leo = load_dual_checkpoint(checkpoint, template, leo_template)
            observations = rollout(
                runner.train_state.params, leo.params, leo.batch_stats
            )
            jax.block_until_ready(observations["teacher_q"])
            condition_records = _host_records(
                observations, labels_host, indices_host, condition, seed
            )
            records.extend(condition_records)
            checkpoints.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "checkpoint": str(checkpoint.resolve()),
                    "state_sha256": _sha256(checkpoint / "state.msgpack"),
                    "trace_records": len(condition_records),
                }
            )
    counts = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in records if row["condition"] == condition]
        counts[condition] = {
            action.name: {
                "attempts": sum(
                    row["policy_action"] == action.name for row in condition_rows
                ),
                "succeeded": sum(
                    row["policy_action"] == action.name
                    and row["stage_action_succeeded"]
                    for row in condition_rows
                ),
                "teacher_agreements": sum(
                    row["policy_action"] == action.name
                    and row["teacher_recommendation_matches_policy"]
                    for row in condition_rows
                ),
            }
            for action in TARGET_ACTIONS
        }
    return {
        "schema_version": "hackrl_pack_restore_teacher_recommendation_trace_v1",
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "trained_variant": "mutant",
        "evaluation_kernel": "mutant",
        "evaluation_split": "validation",
        "policy": "mode",
        "start_families": ["natural_reset", "common_setup"],
        "checkpoint_update": ADAPT_UPDATES,
        "target_actions": [action.name for action in TARGET_ACTIONS],
        "checkpoints": checkpoints,
        "counts": counts,
        "records": records,
        "claim_limit": (
            "Recommendations are measured on states each learned policy actually "
            "visited. Differences can therefore reflect both teacher state and "
            "policy-induced state selection. Temporal order is descriptive and is "
            "not a causal intervention on teacher advice."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = trace(arguments.run_root)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps({
        "output": str(output.resolve()),
        "records": len(result["records"]),
        "counts": result["counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
