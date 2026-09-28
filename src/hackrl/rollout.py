"""Batched HackRL rollouts with terminal-safe observations and metrics."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from hackrl.metrics import (
    EpisodeTracker,
    init_episode_tracker,
    update_episode_tracker,
)


@struct.dataclass
class VectorEnvState:
    """Environment and metric state for a batch of independent workers."""

    env_state: Any
    tracker: EpisodeTracker


@struct.dataclass
class EpisodeRecord:
    """Completed-episode values; use `completed` as the validity mask."""

    completed: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    goal_success: jax.Array
    violation_count: jax.Array
    harvest_count: jax.Array
    repeated_harvest_count: jax.Array
    ever_iron: jax.Array
    wood_exhausted_before_goal: jax.Array


@struct.dataclass
class BatchTransition:
    """One batched transition before any automatic reset is hidden.

    `terminal_observation` is always the direct result of the environment
    step. It is a true terminal observation where `done` is true.
    `reset_observation` is the independently sampled reset candidate, while
    the observation returned by :meth:`HackRLBatchEnv.step` is the observation
    the policy should consume next.
    """

    terminal_observation: jax.Array
    reset_observation: jax.Array
    reward: jax.Array
    done: jax.Array
    info: Any
    episode: EpisodeRecord
    repeated_harvest: jax.Array


def _batch_tracker(
    max_growing_plants: int,
    batch_size: int,
    inventory=None,
) -> EpisodeTracker:
    tracker = init_episode_tracker(max_growing_plants)
    batched = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape),
        tracker,
    )
    if inventory is None:
        return batched
    return batched.replace(ever_iron=inventory.iron >= 1)


def select_batched(mask, true_tree, false_tree):
    """Select matching batched pytrees with a scalar mask per worker."""

    def select(true_value, false_value):
        expanded = mask.reshape((mask.shape[0],) + (1,) * (true_value.ndim - 1))
        return jnp.where(expanded, true_value, false_value)

    return jax.tree.map(select, true_tree, false_tree)


class HackRLBatchEnv:
    """Vectorize a no-auto-reset HackRL env and reset each worker independently."""

    def __init__(self, env, num_envs: int):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.env = env
        self.num_envs = num_envs
        self.params = env.default_params
        self._reset = jax.vmap(lambda key: env.reset(key, self.params))
        self._step = jax.vmap(
            lambda key, state, action: env.step(
                key, state, action, self.params
            )
        )

    def reset(self, rng):
        """Reset every worker with a distinct PRNG key."""

        reset_keys = jax.random.split(rng, self.num_envs)
        observations, env_state = self._reset(reset_keys)
        trackers = _batch_tracker(
            self.env.static_env_params.max_growing_plants,
            self.num_envs,
            env_state.inventory,
        )
        return observations, VectorEnvState(env_state, trackers)

    def step(self, rng, state: VectorEnvState, actions):
        """Step workers, retain terminal observations, then auto-reset."""

        step_rng, reset_rng = jax.random.split(rng)
        step_keys = jax.random.split(step_rng, self.num_envs)
        (
            terminal_observations,
            stepped_states,
            rewards,
            dones,
            infos,
        ) = self._step(step_keys, state.env_state, actions)

        updated_trackers, repeated_harvests = jax.vmap(
            update_episode_tracker
        )(
            state.tracker,
            rewards,
            infos,
            stepped_states.timestep,
            stepped_states.inventory,
        )
        episode = EpisodeRecord(
            completed=dones,
            episode_return=updated_trackers.episode_return,
            episode_length=updated_trackers.episode_length,
            goal_success=updated_trackers.goal_success,
            violation_count=updated_trackers.violation_count,
            harvest_count=updated_trackers.harvest_count,
            repeated_harvest_count=updated_trackers.repeated_harvest_count,
            ever_iron=updated_trackers.ever_iron,
            wood_exhausted_before_goal=updated_trackers.wood_exhausted_before_goal,
        )

        # Reset candidates are intentionally independent per worker. The Easy
        # fixtures are cheap, so this favors simple and unambiguous semantics.
        reset_keys = jax.random.split(reset_rng, self.num_envs)
        reset_observations, reset_states = self._reset(reset_keys)
        empty_trackers = _batch_tracker(
            self.env.static_env_params.max_growing_plants,
            self.num_envs,
            reset_states.inventory,
        )
        policy_observations = jnp.where(
            dones[:, None], reset_observations, terminal_observations
        )
        next_state = VectorEnvState(
            env_state=select_batched(dones, reset_states, stepped_states),
            tracker=select_batched(
                dones, empty_trackers, updated_trackers
            ),
        )
        transition = BatchTransition(
            terminal_observation=terminal_observations,
            reset_observation=reset_observations,
            reward=rewards,
            done=dones,
            info=infos,
            episode=episode,
            repeated_harvest=repeated_harvests,
        )
        return policy_observations, next_state, transition
