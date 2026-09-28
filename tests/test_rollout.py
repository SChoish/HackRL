import jax
import jax.numpy as jnp
import pytest
from craftax.craftax_classic.constants import Action

from hackrl import EasyTask, HackRLEasySymbolicEnvNoAutoReset
from hackrl.rollout import HackRLBatchEnv


@pytest.mark.parametrize("task", list(EasyTask))
@pytest.mark.parametrize("mutant", [False, True])
def test_batch_adapter_runs_every_easy_pair(task, mutant):
    env = HackRLEasySymbolicEnvNoAutoReset(task, mutant=mutant)
    vector_env = HackRLBatchEnv(env, num_envs=2)
    observations, state = jax.jit(vector_env.reset)(
        jax.random.PRNGKey(80)
    )
    actions = jnp.full((2,), Action.NOOP.value, dtype=jnp.int32)

    policy_observations, next_state, transition = jax.jit(
        vector_env.step
    )(jax.random.PRNGKey(81), state, actions)

    assert observations.shape == (2, 1345)
    assert policy_observations.shape == (2, 1345)
    assert transition.terminal_observation.shape == (2, 1345)
    assert transition.reset_observation.shape == (2, 1345)
    assert next_state.env_state.map.shape == (2, 16, 16)


def test_batch_adapter_preserves_terminal_and_reset_observations():
    env = HackRLEasySymbolicEnvNoAutoReset(EasyTask.R_E, mutant=True)
    vector_env = HackRLBatchEnv(env, num_envs=2)
    _, state = vector_env.reset(jax.random.PRNGKey(82))
    actions = jnp.full(
        (2,), Action.MAKE_IRON_PICKAXE.value, dtype=jnp.int32
    )

    policy_observation, next_state, transition = jax.jit(
        vector_env.step
    )(jax.random.PRNGKey(83), state, actions)

    assert bool(jnp.all(transition.done))
    assert bool(jnp.all(transition.episode.goal_success))
    assert bool(jnp.all(transition.info["HackRL/violation"]))
    assert bool(
        jnp.array_equal(
            policy_observation, transition.reset_observation
        )
    )
    assert not bool(
        jnp.array_equal(
            transition.terminal_observation,
            transition.reset_observation,
        )
    )
    assert bool(
        jnp.all(next_state.env_state.inventory.iron_pickaxe == 0)
    )
    assert bool(
        jnp.all(next_state.tracker.episode_length == 0)
    )


def test_post_iron_reset_does_not_count_initial_inventory_as_acquisition():
    env = HackRLEasySymbolicEnvNoAutoReset(
        EasyTask.R_E,
        mutant=False,
        start_mode="r_e_post_iron",
    )
    vector_env = HackRLBatchEnv(env, num_envs=2)
    _, state = vector_env.reset(jax.random.PRNGKey(84))
    assert bool(jnp.all(state.tracker.iron_acquired_count == 0))
    assert bool(jnp.all(state.env_state.inventory.iron == 1))
