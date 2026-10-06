#!/usr/bin/env python3
"""Frozen-policy diagnosis of post-mine return behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

REQUESTED_DEVICE = os.environ.get("HACKRL_DEVICE", "cpu")
if REQUESTED_DEVICE not in {"cpu", "cuda"}:
    raise RuntimeError("HACKRL_DEVICE must be 'cpu' or 'cuda'")
if REQUESTED_DEVICE == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from hackrl.mine_expedition import (
    CAMP_POSITION,
    TARGET_POSE,
    WORLD_HORIZON,
    MineExpeditionAction,
    MineExpeditionVariant,
)
from hackrl.mine_expedition_env import (
    MAP_CHANNEL_NAMES,
    NUMERIC_FEATURE_NAMES,
    TASK_DISCOUNT,
    MineExpeditionStart,
    _DISTANCE_TO_CAMP,
    _task_distance,
    mine_expedition_potential,
    observe_mine_expedition,
    reset_mine_expedition,
    step_mine_expedition_env,
)
from hackrl.mine_expedition_ppo import (
    MineExpeditionPPOConfig,
    _batch_inputs,
    _tree_where,
    initialize_mine_expedition_ppo,
    load_mine_expedition_checkpoint,
)


REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST = (
    REPOSITORY
    / "docs/manifests/mine_expedition_post_mine_frozen_diagnostic_v1.json"
)
CORE_SOURCES = (
    "src/hackrl/mine_expedition.py",
    "src/hackrl/mine_expedition_env.py",
    "src/hackrl/mine_expedition_ppo.py",
)
EXECUTION_SOURCES = (
    *CORE_SOURCES,
    "scripts/diagnose_mine_expedition_post_mine.py",
    "docs/manifests/mine_expedition_post_mine_frozen_diagnostic_v1.json",
    "tests/test_mine_expedition_post_mine_diagnostic.py",
)
ACTION_NAMES = {int(action): action.name for action in MineExpeditionAction}
MOVEMENT_ACTIONS = {
    int(MineExpeditionAction.LEFT),
    int(MineExpeditionAction.RIGHT),
    int(MineExpeditionAction.UP),
    int(MineExpeditionAction.DOWN),
}
AGREEMENT_METRICS = (
    "success_state_mismatch_steps",
    "return_event_mismatch_steps",
    "terminal_classification_mismatch_steps",
)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _filesystem(path):
    usage = shutil.disk_usage(path)
    return {
        "path": str(Path(path).resolve()),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "utilization": usage.used / usage.total,
    }


def _broadcast_state(state, episodes):
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (episodes,) + value.shape), state
    )


def _camp_distances(states):
    positions = states.player_position
    return _DISTANCE_TO_CAMP[positions[:, 0], positions[:, 1]]


def _task_distances(states):
    return jax.vmap(_task_distance)(states)


def make_rollout(network, *, stochastic, episodes):
    def rollout(parameters, initial_state, action_keys):
        states = _broadcast_state(initial_state, episodes)
        done = jnp.zeros((episodes,), dtype=jnp.bool_)

        def step(carry, _):
            state, already_done, keys = carry
            policy, value = network.apply(parameters, *_batch_inputs(state))
            split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(keys)
            next_keys = split_keys[:, 0]
            draw_keys = split_keys[:, 1]
            if stochastic:
                actions = jax.vmap(
                    lambda logits, key: jax.random.categorical(key, logits)
                )(policy.logits, draw_keys)
            else:
                actions = jnp.argmax(policy.logits, axis=-1)
            active = jnp.logical_not(already_done)
            actions = jnp.where(active, actions, int(MineExpeditionAction.NOOP))
            _, stepped, events = jax.vmap(
                lambda item, action: step_mine_expedition_env(
                    item, action, MineExpeditionVariant.FIXED
                )
            )(state, actions)
            stepped = _tree_where(active, stepped, state)
            just_done = jnp.logical_and(active, events.done)
            record = {
                "active": active,
                "action": actions,
                "value": value,
                "reward": jnp.where(active, events.reward, 0.0),
                "done": just_done,
                "success": jnp.logical_and(active, events.success),
                "timeout": jnp.logical_and(active, events.timeout),
                "crafted": jnp.logical_and(active, events.crafted_pickaxe),
                "mined": jnp.logical_and(active, events.mined_target),
                "returned": jnp.logical_and(active, events.returned_target),
                "before_tick": state.tick,
                "after_tick": stepped.tick,
                "before_position": state.player_position,
                "after_position": stepped.player_position,
                "before_carried_target": state.carried_target,
                "after_carried_target": stepped.carried_target,
                "before_returned_target": state.returned_target,
                "after_returned_target": stepped.returned_target,
                "before_target_remaining": state.target_remaining,
                "after_target_remaining": stepped.target_remaining,
                "before_task_distance": _task_distances(state),
                "after_task_distance": _task_distances(stepped),
                "before_camp_distance": _camp_distances(state),
                "after_camp_distance": _camp_distances(stepped),
            }
            return (
                stepped,
                jnp.logical_or(already_done, just_done),
                next_keys,
            ), record

        final, trace = jax.lax.scan(
            step, (states, done, action_keys), None, length=WORLD_HORIZON
        )
        return final, trace

    return jax.jit(rollout)


def _first_indices(events):
    events = np.asarray(events, dtype=bool)
    any_event = events.any(axis=0)
    indices = np.argmax(events, axis=0)
    return np.where(any_event, indices, -1)


def _mean_or_none(values):
    values = np.asarray(values)
    return None if values.size == 0 else float(values.mean())


def _episode_metrics(trace, *, start_is_post_mine):
    trace = jax.device_get(trace)
    arrays = {name: np.asarray(value) for name, value in trace.items()}
    active = arrays["active"].astype(bool)
    mined = arrays["mined"].astype(bool)
    success = arrays["success"].astype(bool).any(axis=0)
    timeout = arrays["timeout"].astype(bool).any(axis=0)
    mined_episode = mined.any(axis=0)
    first_mine = _first_indices(mined)
    if start_is_post_mine:
        mined_episode = np.ones_like(success, dtype=bool)
        first_mine = np.full_like(first_mine, -1)
    post_mine = active & (arrays["before_carried_target"] > 0)
    camp_position = np.asarray(CAMP_POSITION, dtype=np.int32)
    before_return_adjacent = (
        np.abs(arrays["before_position"] - camp_position).sum(axis=-1) == 1
    )
    after_return_adjacent = (
        np.abs(arrays["after_position"] - camp_position).sum(axis=-1) == 1
    )
    reached_return_pose = (
        post_mine & (before_return_adjacent | after_return_adjacent)
    ).any(axis=0)
    return_attempt = (
        post_mine
        & (arrays["action"] == int(MineExpeditionAction.RETURN_TARGET))
    ).any(axis=0)
    legal_return_attempt = (
        post_mine
        & before_return_adjacent
        & (arrays["action"] == int(MineExpeditionAction.RETURN_TARGET))
    ).any(axis=0)
    returned_event = arrays["returned"].astype(bool).any(axis=0)

    episode_count = active.shape[1]
    minimum_camp_distance = np.full((episode_count,), WORLD_HORIZON, dtype=np.int32)
    post_mine_steps = post_mine.sum(axis=0)
    final_camp_distance = np.full((episode_count,), WORLD_HORIZON, dtype=np.int32)
    final_positions = np.zeros((episode_count, 2), dtype=np.int32)
    for episode in range(episode_count):
        mask = post_mine[:, episode]
        if mask.any():
            minimum_camp_distance[episode] = int(
                min(
                    arrays["before_camp_distance"][:, episode][mask].min(),
                    arrays["after_camp_distance"][:, episode][mask].min(),
                )
            )
        active_indices = np.flatnonzero(active[:, episode])
        if active_indices.size:
            final_index = int(active_indices[-1])
            final_camp_distance[episode] = int(
                arrays["after_camp_distance"][final_index, episode]
            )
            final_positions[episode] = arrays["after_position"][
                final_index, episode
            ]

    mine_ticks = []
    mine_rewards = []
    mine_nonterminal = []
    for episode, index in enumerate(first_mine):
        if index >= 0:
            mine_ticks.append(int(arrays["after_tick"][index, episode]))
            mine_rewards.append(float(arrays["reward"][index, episode]))
            mine_nonterminal.append(not bool(arrays["done"][index, episode]))

    after_mine_actions = Counter()
    for action in range(len(ACTION_NAMES)):
        after_mine_actions[ACTION_NAMES[action]] = int(
            np.sum(post_mine & (arrays["action"] == action))
        )
    after_mine_actions = {
        name: count for name, count in after_mine_actions.items() if count
    }

    success_event_mismatches = int(
        np.sum(
            active
            & (
                arrays["success"].astype(bool)
                != (arrays["after_returned_target"] > 0)
            )
        )
    )
    returned_event_mismatches = int(
        np.sum(
            active
            & (
                arrays["returned"].astype(bool)
                != (
                    arrays["after_returned_target"]
                    > arrays["before_returned_target"]
                )
            )
        )
    )
    terminal_classification_mismatches = int(
        np.sum(
            active
            & (
                arrays["done"].astype(bool)
                != np.logical_or(
                    arrays["success"].astype(bool),
                    arrays["timeout"].astype(bool),
                )
            )
        )
    )
    failed_after_mining = mined_episode & ~success
    timed_out_after_mining = mined_episode & timeout
    return {
        "episodes": int(episode_count),
        "mined_count": int(mined_episode.sum()),
        "success_count": int(success.sum()),
        "success_rate": float(success.mean()),
        "timeout_count": int(timeout.sum()),
        "mined_then_timeout_count": int((mined_episode & timeout).sum()),
        "reached_return_pose_count": int(reached_return_pose.sum()),
        "return_action_attempt_count": int(return_attempt.sum()),
        "legal_return_action_attempt_count": int(legal_return_attempt.sum()),
        "returned_event_count": int(returned_event.sum()),
        "failed_after_mining_count": int(failed_after_mining.sum()),
        "failed_after_mining_never_reached_return_pose_count": int(
            (failed_after_mining & ~reached_return_pose).sum()
        ),
        "failed_after_mining_reached_pose_without_legal_return_count": int(
            (failed_after_mining & reached_return_pose & ~legal_return_attempt).sum()
        ),
        "timed_out_after_mining_never_reached_return_pose_count": int(
            (timed_out_after_mining & ~reached_return_pose).sum()
        ),
        "timed_out_after_mining_reached_pose_without_legal_return_count": int(
            (
                timed_out_after_mining
                & reached_return_pose
                & ~legal_return_attempt
            ).sum()
        ),
        "mean_mine_tick": _mean_or_none(mine_ticks),
        "minimum_mine_tick": None if not mine_ticks else min(mine_ticks),
        "maximum_mine_tick": None if not mine_ticks else max(mine_ticks),
        "mean_remaining_ticks_after_mine": _mean_or_none(
            [WORLD_HORIZON - tick for tick in mine_ticks]
        ),
        "mean_mining_transition_reward": _mean_or_none(mine_rewards),
        "mining_transitions_nonterminal": int(sum(mine_nonterminal)),
        "mean_post_mine_steps": _mean_or_none(post_mine_steps[mined_episode]),
        "mean_minimum_camp_distance_after_mine": _mean_or_none(
            minimum_camp_distance[mined_episode]
        ),
        "mean_final_camp_distance": _mean_or_none(
            final_camp_distance[mined_episode]
        ),
        "after_mine_action_counts": after_mine_actions,
        "success_state_mismatch_steps": success_event_mismatches,
        "return_event_mismatch_steps": returned_event_mismatches,
        "terminal_classification_mismatch_steps": (
            terminal_classification_mismatches
        ),
        "episode_success": success.tolist(),
        "episode_timeout": timeout.tolist(),
        "episode_mined": mined_episode.tolist(),
        "episode_reached_return_pose": reached_return_pose.tolist(),
        "episode_legal_return_attempt": legal_return_attempt.tolist(),
        "episode_minimum_camp_distance": minimum_camp_distance.tolist(),
        "episode_final_camp_distance": final_camp_distance.tolist(),
        "episode_final_position": final_positions.tolist(),
    }


def _mode_steps(trace):
    arrays = {name: np.asarray(value) for name, value in jax.device_get(trace).items()}
    rows = []
    for index in range(WORLD_HORIZON):
        if not bool(arrays["active"][index, 0]):
            break
        rows.append(
            {
                "step": index + 1,
                "tick_before": int(arrays["before_tick"][index, 0]),
                "tick_after": int(arrays["after_tick"][index, 0]),
                "position_before": arrays["before_position"][index, 0].tolist(),
                "position_after": arrays["after_position"][index, 0].tolist(),
                "action": ACTION_NAMES[int(arrays["action"][index, 0])],
                "reward": float(arrays["reward"][index, 0]),
                "value": float(arrays["value"][index, 0]),
                "task_distance_before": int(
                    arrays["before_task_distance"][index, 0]
                ),
                "task_distance_after": int(
                    arrays["after_task_distance"][index, 0]
                ),
                "camp_distance_before": int(
                    arrays["before_camp_distance"][index, 0]
                ),
                "camp_distance_after": int(
                    arrays["after_camp_distance"][index, 0]
                ),
                "carried_target_before": int(
                    arrays["before_carried_target"][index, 0]
                ),
                "carried_target_after": int(
                    arrays["after_carried_target"][index, 0]
                ),
                "crafted": bool(arrays["crafted"][index, 0]),
                "mined": bool(arrays["mined"][index, 0]),
                "returned": bool(arrays["returned"][index, 0]),
                "success": bool(arrays["success"][index, 0]),
                "timeout": bool(arrays["timeout"][index, 0]),
            }
        )
    return rows


def _policy_profile(network, parameters, state):
    batched = _broadcast_state(state, 1)
    policy, value = network.apply(parameters, *_batch_inputs(batched))
    probabilities = np.asarray(jax.nn.softmax(policy.logits[0]))
    order = np.argsort(probabilities)[::-1]
    return {
        "critic_value": float(value[0]),
        "top_actions": [
            {
                "action": ACTION_NAMES[int(index)],
                "probability": float(probabilities[index]),
            }
            for index in order[:8]
        ],
        "movement_probability": float(
            sum(probabilities[index] for index in MOVEMENT_ACTIONS)
        ),
        "return_target_probability": float(
            probabilities[int(MineExpeditionAction.RETURN_TARGET)]
        ),
        "all_action_probabilities": {
            ACTION_NAMES[index]: float(probabilities[index])
            for index in range(len(probabilities))
        },
    }


def _canonical_transition():
    before = reset_mine_expedition(
        jax.random.PRNGKey(0), MineExpeditionStart.TARGET_READY
    )
    _, after, event = step_mine_expedition_env(
        before, MineExpeditionAction.DO, MineExpeditionVariant.FIXED
    )
    before_observation = observe_mine_expedition(before)
    after_observation = observe_mine_expedition(after)
    carried_index = NUMERIC_FEATURE_NAMES.index("carried_target_signed")
    target_channel = MAP_CHANNEL_NAMES.index("role/target_mineral")
    payload = {
        "action": MineExpeditionAction.DO.name,
        "position": np.asarray(after.player_position).tolist(),
        "expected_target_pose": list(TARGET_POSE),
        "tick_before": int(before.tick),
        "tick_after": int(after.tick),
        "target_remaining_before": int(before.target_remaining),
        "target_remaining_after": int(after.target_remaining),
        "carried_target_before": int(before.carried_target),
        "carried_target_after": int(after.carried_target),
        "returned_target_after": int(after.returned_target),
        "task_distance_before": int(_task_distance(before)),
        "task_distance_after": int(_task_distance(after)),
        "potential_before": float(mine_expedition_potential(before)),
        "potential_after": float(mine_expedition_potential(after)),
        "reward": float(event.reward),
        "done": bool(event.done),
        "success": bool(event.success),
        "timeout": bool(event.timeout),
        "mined_target": bool(event.mined_target),
        "gae_nonterminal_mask": float(1.0 - np.asarray(event.done, dtype=float)),
        "observation_carried_target_before": float(
            before_observation.numeric_features[carried_index]
        ),
        "observation_carried_target_after": float(
            after_observation.numeric_features[carried_index]
        ),
        "observation_target_channel_sum_before": float(
            before_observation.map_channels[..., target_channel].sum()
        ),
        "observation_target_channel_sum_after": float(
            after_observation.map_channels[..., target_channel].sum()
        ),
    }
    if any(
        (
            payload["target_remaining_after"] != 0,
            payload["carried_target_after"] != 1,
            payload["returned_target_after"] != 0,
            payload["done"],
            payload["success"],
            payload["timeout"],
            not payload["mined_target"],
            payload["tick_after"] != 1,
        )
    ):
        raise RuntimeError(f"canonical post-mine state is invalid: {payload}")
    return before, after, payload


def _checkpoint_specs(manifest):
    specs = []
    for group, block in manifest["checkpoint_groups"].items():
        qualification = {
            seed: seed in block["qualified_seeds"] for seed in (40, 41, 42)
        }
        for seed in (40, 41, 42):
            specs.append(
                {
                    "group": group,
                    "training_start": group,
                    "seed": seed,
                    "qualified": qualification[seed],
                    "checkpoint": str(
                        Path(block["directory"])
                        / f"seed{seed}"
                        / "checkpoints"
                        / "update_512"
                    ),
                }
            )
    return specs


def _evaluation_protocol(manifest):
    protocol = manifest["frozen_evaluation"]
    if protocol["horizon"] != WORLD_HORIZON:
        raise ValueError("manifest horizon differs from the environment horizon")
    mode_episodes = protocol["phase_start_mode_episodes_per_checkpoint"]
    sample_episodes = protocol["phase_start_sample_episodes_per_checkpoint"]
    if mode_episodes != protocol["canonical_post_mine_mode_episodes_per_checkpoint"]:
        raise ValueError("mode episode counts differ between declared starts")
    if sample_episodes != protocol["canonical_post_mine_sample_episodes_per_checkpoint"]:
        raise ValueError("sample episode counts differ between declared starts")
    if mode_episodes != 1 or sample_episodes <= 0:
        raise ValueError("diagnostic requires one mode and positive sample episodes")
    return {
        "mode_episodes": int(mode_episodes),
        "sample_episodes": int(sample_episodes),
        "mode_action_seed": int(protocol["mode_action_seed"]),
        "sample_action_seed": int(protocol["sample_action_seed"]),
        "horizon": int(protocol["horizon"]),
    }


def _validate_agreement(policy_results):
    mismatches = {
        metric: sum(
            int(evaluation[metric])
            for row in policy_results
            for evaluation in row["evaluations"].values()
        )
        for metric in AGREEMENT_METRICS
    }
    if any(mismatches.values()):
        raise RuntimeError(
            "success, return, or terminal classification mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return mismatches


def diagnose(output):
    manifest = _read_json(MANIFEST)
    protocol = _evaluation_protocol(manifest)
    expected_execution_sha = manifest["execution_code_sha_of_checkpoints"]
    dirty = _git("status", "--porcelain", "--", *EXECUTION_SOURCES)
    if dirty:
        raise RuntimeError("diagnostic execution sources must be committed:\n" + dirty)
    current_sha = _git("rev-parse", "HEAD")
    runtime_backend = jax.default_backend()
    expected_backend = "gpu" if REQUESTED_DEVICE == "cuda" else "cpu"
    if runtime_backend != expected_backend:
        raise RuntimeError(
            f"requested {REQUESTED_DEVICE}, but JAX selected {runtime_backend}"
        )

    _, post_mine_state, canonical = _canonical_transition()
    phase_starts = {
        "craft_ready": reset_mine_expedition(
            jax.random.PRNGKey(0), MineExpeditionStart.CRAFT_READY
        ),
        "target_ready": reset_mine_expedition(
            jax.random.PRNGKey(0), MineExpeditionStart.TARGET_READY
        ),
    }
    network = None
    rollouts = {}
    policy_results = []
    current_core_hashes = {
        name: _sha256(REPOSITORY / name) for name in CORE_SOURCES
    }

    for spec in _checkpoint_specs(manifest):
        checkpoint = Path(spec["checkpoint"])
        config = MineExpeditionPPOConfig(**_read_json(checkpoint / "config.json"))
        candidate_network, template = initialize_mine_expedition_ppo(config)
        if network is None:
            network = candidate_network
            rollouts = {
                (False, protocol["mode_episodes"]): make_rollout(
                    network,
                    stochastic=False,
                    episodes=protocol["mode_episodes"],
                ),
                (True, protocol["sample_episodes"]): make_rollout(
                    network,
                    stochastic=True,
                    episodes=protocol["sample_episodes"],
                ),
            }
        runner = load_mine_expedition_checkpoint(checkpoint, template, config)
        frozen_before = serialization.to_bytes(runner)
        metadata = _read_json(checkpoint / "metadata.json")
        provenance = _read_json(checkpoint.parent.parent / "run_manifest.json")
        if provenance.get("execution_code_sha") != expected_execution_sha:
            raise RuntimeError(f"unexpected execution SHA: {checkpoint}")
        recorded_sources = provenance.get("execution_source_sha256", {})
        mismatches = [
            name
            for name, digest in current_core_hashes.items()
            if recorded_sources.get(name) != digest
        ]
        if mismatches:
            raise RuntimeError(
                f"current core source differs from checkpoint provenance: {mismatches}"
            )
        if metadata.get("state_sha256") != _sha256(checkpoint / "state.msgpack"):
            raise RuntimeError(f"checkpoint digest mismatch: {checkpoint}")

        evaluations = {}
        for start_name, start_state in (
            (spec["group"], phase_starts[spec["group"]]),
            ("canonical_post_mine", post_mine_state),
        ):
            start_is_post_mine = start_name == "canonical_post_mine"
            for stochastic, episodes, label, seed in (
                (
                    False,
                    protocol["mode_episodes"],
                    "mode",
                    protocol["mode_action_seed"],
                ),
                (
                    True,
                    protocol["sample_episodes"],
                    "sample",
                    protocol["sample_action_seed"],
                ),
            ):
                action_keys = jax.random.split(jax.random.PRNGKey(seed), episodes)
                _, trace = rollouts[(stochastic, episodes)](
                    runner.train_state.params, start_state, action_keys
                )
                jax.block_until_ready(trace["action"])
                metrics = _episode_metrics(
                    trace, start_is_post_mine=start_is_post_mine
                )
                if not stochastic:
                    metrics["steps"] = _mode_steps(trace)
                evaluations[f"{start_name}_{label}"] = metrics

        profile_before = _policy_profile(
            network, runner.train_state.params, phase_starts["target_ready"]
        )
        profile_after = _policy_profile(
            network, runner.train_state.params, post_mine_state
        )
        one_step_td_residual = (
            canonical["reward"]
            + TASK_DISCOUNT * profile_after["critic_value"]
            - profile_before["critic_value"]
        )
        frozen_after = serialization.to_bytes(runner)
        policy_results.append(
            {
                **spec,
                "checkpoint_state_sha256": metadata["state_sha256"],
                "checkpoint_execution_sha": provenance["execution_code_sha"],
                "learner_state_immutable": frozen_before == frozen_after,
                "canonical_mining_transition": {
                    "critic_value_before": profile_before["critic_value"],
                    "critic_value_after": profile_after["critic_value"],
                    "one_step_td_residual": float(one_step_td_residual),
                    "gae_nonterminal_mask": canonical["gae_nonterminal_mask"],
                },
                "canonical_post_mine_policy": profile_after,
                "evaluations": evaluations,
            }
        )

    if not all(row["learner_state_immutable"] for row in policy_results):
        raise RuntimeError("frozen evaluation mutated a learner checkpoint")
    agreement_mismatches = _validate_agreement(policy_results)
    payload = {
        "schema_version": "hackrl_mine_expedition_post_mine_frozen_result_v1",
        "diagnostic_id": manifest["diagnostic_id"],
        "status": "complete",
        "execution_complete": True,
        "diagnostic_execution_sha": current_sha,
        "checkpoint_execution_sha": expected_execution_sha,
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime": {
            "requested_device": REQUESTED_DEVICE,
            "jax_backend": runtime_backend,
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "source_sha256": {
            name: _sha256(REPOSITORY / name) for name in EXECUTION_SOURCES
        },
        "storage": {
            "output": str(Path(output).resolve()),
            "home": _filesystem("/home/ext_csv"),
            "raid": _filesystem("/raid/ext_csv"),
        },
        "learning_contract": {
            "action_count": len(ACTION_NAMES),
            "action_mask_present": False,
            "training_transition_valid_mask_present": False,
            "bootstrap_mask": "1 - done",
            "canonical_post_mine_gae_nonterminal_mask": canonical[
                "gae_nonterminal_mask"
            ],
            "claim_limit": "This describes the fixed PPO implementation at the recorded source hashes; it is not an independent measurement from checkpoint tensors.",
        },
        "canonical_mining_transition": canonical,
        "evaluation_protocol": protocol,
        "agreement_mismatches": agreement_mismatches,
        "policy_results": policy_results,
        "claim_limits": manifest["claim_limits"],
    }
    _write_json(output, payload)
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=(
            "/raid/ext_csv/HackRL/runs/"
            "mine_expedition_post_mine_frozen_diagnostic_v1/result.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = diagnose(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
