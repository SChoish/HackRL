import jax
import jax.numpy as jnp
from craftax.craftax_classic import game_logic as base
from craftax.craftax_classic.constants import Action

from hackrl import HackRLClassicSymbolicEnvNoAutoReset, RootMutation
from hackrl.mutations import craftax_step


def test_fixed_step_matches_upstream(base_state, static_params):
    env = HackRLClassicSymbolicEnvNoAutoReset(
        mutation=RootMutation.FIXED,
        static_env_params=static_params,
    )
    params = env.default_params
    key = jax.random.PRNGKey(4)

    expected_state, expected_reward = base.craftax_step(
        key, base_state, Action.NOOP.value, params, static_params
    )
    actual_state, actual_reward = craftax_step(
        key,
        base_state,
        Action.NOOP.value,
        params,
        static_params,
        RootMutation.FIXED,
    )

    comparisons = jax.tree_util.tree_map(jnp.array_equal, expected_state, actual_state)
    assert all(bool(value) for value in jax.tree_util.tree_leaves(comparisons))
    assert bool(jnp.array_equal(expected_reward, actual_reward))


def test_environment_smoke_reset_and_jitted_step():
    env = HackRLClassicSymbolicEnvNoAutoReset(RootMutation.H1_IRON_LOWER_BOUND)
    params = env.default_params
    reset_key, step_key = jax.random.split(jax.random.PRNGKey(5))
    observation, state = env.reset(reset_key, params)

    assert observation.shape == (1345,)
    step = jax.jit(env.step)
    next_observation, next_state, reward, done, info = step(
        step_key, state, Action.NOOP.value, params
    )

    assert next_observation.shape == (1345,)
    assert int(next_state.timestep) == 1
    assert jnp.ndim(reward) == 0
    assert jnp.ndim(done) == 0
    assert "HackRL/violation" in info
    assert not bool(info["HackRL/violation"])

