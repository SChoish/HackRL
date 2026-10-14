"""Frozen normal-trajectory behavior anchors for PACK PPO adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from hackrl.pack_restore import (
    PackRestoreSplit,
    PackRestoreVariant,
    observe_pack_restore,
    pack_restore_goal_vector,
    pack_restore_step,
    pack_restore_world_done,
)
from hackrl.pack_restore_gc import (
    DELIVER_3_GOAL_INDEX,
    NUM_ACTIONS,
    _evaluation_states,
    categorical_source_kl,
    make_pack_restore_gc_update,
    pack_restore_gc_inputs,
)


@struct.dataclass
class PackBehaviorAnchor:
    """Fixed model inputs and soft source targets; no source parameters."""

    map_channels: jax.Array
    numeric_features: jax.Array
    goal_one_hot: jax.Array
    source_logits: jax.Array

    @property
    def state_count(self) -> int:
        return int(self.source_logits.shape[0])


@dataclass(frozen=True)
class PackBehaviorAnchorMetadata:
    split: str
    kernel: str
    start_count: int
    state_count: int
    successful_trajectories: int
    unsuccessful_trajectories: int
    horizon_ticks: int
    trajectory_lengths: tuple[int, ...]
    source_variable_collections: tuple[str, ...]
    source_has_mutable_normalization_statistics: bool
    source_parameters_retained: bool


def behavior_anchor_kl(network, parameters, anchor: PackBehaviorAnchor):
    """Mean full-categorical KL(source || current) over anchor states."""

    current_policy, _ = network.apply(
        parameters,
        anchor.map_channels,
        anchor.numeric_features,
        anchor.goal_one_hot,
    )
    return jnp.mean(
        categorical_source_kl(anchor.source_logits, current_policy.logits)
    )


def build_pack_behavior_anchor(
    network,
    source_parameters,
    *,
    source_growth_period: int,
    horizon_ticks: int = 128,
    split: PackRestoreSplit | str = PackRestoreSplit.TRAIN,
) -> tuple[PackBehaviorAnchor, PackBehaviorAnchorMetadata]:
    """Replay all 32 natural split starts with frozen greedy source actions."""

    if horizon_ticks <= 0:
        raise ValueError("horizon_ticks must be positive")
    split = PackRestoreSplit(split)
    starts, _, _, _ = _evaluation_states(
        split,
        1,
        source_growth_period,
        natural_only=True,
    )
    start_count = int(jax.tree.leaves(starts)[0].shape[0])
    if start_count != 32:
        raise AssertionError(f"expected 32 natural TRAIN starts, found {start_count}")

    maps = []
    numerics = []
    goals = []
    logits = []
    trajectory_lengths = []
    successful_trajectories = 0
    for start_index in range(start_count):
        state = jax.tree.map(lambda value: value[start_index], starts)
        success = False
        length = 0
        for _ in range(horizon_ticks):
            observation = observe_pack_restore(state)
            model_inputs = pack_restore_gc_inputs(
                observation,
                jnp.asarray(DELIVER_3_GOAL_INDEX, dtype=jnp.int32),
            )
            batched_inputs = tuple(value[None, ...] for value in model_inputs)
            source_policy, _ = network.apply(source_parameters, *batched_inputs)
            source_logits = source_policy.logits[0]
            maps.append(np.asarray(jax.device_get(model_inputs[0])))
            numerics.append(np.asarray(jax.device_get(model_inputs[1])))
            goals.append(np.asarray(jax.device_get(model_inputs[2])))
            logits.append(np.asarray(jax.device_get(source_logits)))
            action = int(np.asarray(jax.device_get(jnp.argmax(source_logits))))
            state = pack_restore_step(state, action, PackRestoreVariant.FIXED)
            length += 1
            success = bool(
                np.asarray(
                    jax.device_get(
                        pack_restore_goal_vector(state)[DELIVER_3_GOAL_INDEX]
                    )
                )
            )
            world_done = bool(np.asarray(jax.device_get(pack_restore_world_done(state))))
            if success or world_done:
                break
        trajectory_lengths.append(length)
        successful_trajectories += int(success)

    anchor = PackBehaviorAnchor(
        map_channels=jnp.asarray(np.stack(maps)),
        numeric_features=jnp.asarray(np.stack(numerics)),
        goal_one_hot=jnp.asarray(np.stack(goals)),
        source_logits=jax.lax.stop_gradient(jnp.asarray(np.stack(logits))),
    )
    if anchor.source_logits.shape[-1] != NUM_ACTIONS:
        raise AssertionError("anchor does not retain the complete action distribution")
    collections = (
        tuple(sorted(str(key) for key in source_parameters.keys()))
        if hasattr(source_parameters, "keys")
        else ()
    )
    metadata = PackBehaviorAnchorMetadata(
        split=split.value,
        kernel=PackRestoreVariant.FIXED.value,
        start_count=start_count,
        state_count=anchor.state_count,
        successful_trajectories=successful_trajectories,
        unsuccessful_trajectories=start_count - successful_trajectories,
        horizon_ticks=horizon_ticks,
        trajectory_lengths=tuple(trajectory_lengths),
        source_variable_collections=collections,
        source_has_mutable_normalization_statistics=collections != ("params",),
        source_parameters_retained=False,
    )
    return anchor, metadata


def make_behavior_anchored_pack_restore_update(
    network,
    config,
    anchor: PackBehaviorAnchor,
    coefficient: float,
):
    """Build the shared updater, preserving the exact legacy path at beta zero."""

    coefficient = float(coefficient)
    if coefficient < 0.0:
        raise ValueError("coefficient must be non-negative")
    if coefficient == 0.0:
        return make_pack_restore_gc_update(network, config)
    return make_pack_restore_gc_update(
        network,
        config,
        behavior_anchor=anchor,
        behavior_anchor_coefficient=coefficient,
    )
