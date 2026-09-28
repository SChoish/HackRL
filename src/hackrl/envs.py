"""Craftax-Classic environments backed by HackRL mutation kernels."""

from __future__ import annotations

import jax
from craftax.craftax_classic.envs.common import compute_score
from craftax.craftax_classic.envs.craftax_symbolic_env import (
    CraftaxClassicSymbolicEnvNoAutoReset,
)

from hackrl.mutations import RootMutation, craftax_step, detect_violation


class HackRLClassicSymbolicEnvNoAutoReset(CraftaxClassicSymbolicEnvNoAutoReset):
    """Symbolic Craftax environment with a fixed per-instance mutation."""

    def __init__(self, mutation: RootMutation = RootMutation.FIXED, **kwargs):
        super().__init__(**kwargs)
        self.mutation = RootMutation(mutation)

    def step_env(self, rng, state, action, params):
        next_state, reward = craftax_step(
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
        info["HackRL/violation"] = detect_violation(
            self.mutation, state, action, next_state
        )
        info["HackRL/goal_success"] = jax.numpy.logical_and(
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

