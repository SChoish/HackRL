"""Craftax-Classic environments backed by HackRL mutation kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from craftax.craftax_classic.envs.common import compute_score
from craftax.craftax_classic.envs.craftax_symbolic_env import (
    CraftaxClassicSymbolicEnvNoAutoReset,
    get_flat_map_obs_shape,
    get_inventory_obs_shape,
)
from craftax.environment_base import spaces

from hackrl.mutations import RootMutation, craftax_step_with_violation


class HackRLClassicSymbolicEnvNoAutoReset(CraftaxClassicSymbolicEnvNoAutoReset):
    """Symbolic Craftax environment with a fixed per-instance mutation."""

    def __init__(self, mutation: RootMutation = RootMutation.FIXED, **kwargs):
        super().__init__(**kwargs)
        self.mutation = RootMutation(mutation)

    def step_env(self, rng, state, action, params):
        next_state, reward, violation = craftax_step_with_violation(
            rng,
            state,
            action,
            params,
            self.static_env_params,
            self.mutation,
        )
        done = self.is_terminal(next_state, params)
        info = compute_score(next_state, done)
        info["discount"] = self.discount(next_state, params)
        info["HackRL/violation"] = violation
        info["HackRL/goal_success"] = jnp.logical_and(
            state.inventory.diamond == 0,
            next_state.inventory.diamond > 0,
        )
        return (
            jax.lax.stop_gradient(self.get_obs(next_state)),
            jax.lax.stop_gradient(next_state),
            reward,
            done,
            info,
        )

    @property
    def name(self) -> str:
        return f"HackRL-Classic-Symbolic-{self.mutation.value}-NoAutoReset-v0"

    def observation_space(self, params=None) -> spaces.Box:
        """Declare the shared range including H1's observable iron balance."""

        del params
        observation_size = get_flat_map_obs_shape() + get_inventory_obs_shape()
        low = jnp.zeros((observation_size,), dtype=jnp.float32)
        iron_observation_index = get_flat_map_obs_shape() + 3
        low = low.at[iron_observation_index].set(-0.1)
        high = jnp.ones((observation_size,), dtype=jnp.float32)
        return spaces.Box(low, high, (observation_size,), dtype=jnp.float32)

