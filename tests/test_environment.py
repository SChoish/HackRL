import jax
import jax.numpy as jnp
from craftax.craftax_classic import game_logic as base
from craftax.craftax_classic.constants import Action, BlockType

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


def test_h2_natural_hunger_tick_is_not_a_violation(base_state, static_params):
    target_column = 8
    state = base_state.replace(
        map=base_state.map.at[-1, target_column].set(BlockType.IRON.value),
        player_position=jnp.array([0, target_column], dtype=jnp.int32),
        player_direction=jnp.asarray(Action.UP.value),
        player_hunger=jnp.asarray(25.0),
        inventory=base_state.inventory.replace(stone_pickaxe=0),
    )
    env = HackRLClassicSymbolicEnvNoAutoReset(
        RootMutation.H2_MISSING_MAP_BOUNDS,
        static_env_params=static_params,
    )
    params = env.default_params.replace(
        spawn_cow_chance=0.0,
        spawn_zombie_base_chance=0.0,
        spawn_zombie_night_chance=0.0,
        spawn_skeleton_chance=0.0,
    )

    _, next_state, _, _, info = jax.jit(env.step)(
        jax.random.PRNGKey(6), state, Action.DO.value, params
    )

    assert int(next_state.player_food) == 8
    assert not bool(info["HackRL/violation"])


def test_h0_sleeping_input_is_not_a_harvest(base_state, static_params):
    plant_position = jnp.array([7, 8], dtype=jnp.int32)
    state = base_state.replace(
        map=base_state.map.at[7, 8].set(BlockType.RIPE_PLANT.value),
        player_direction=jnp.asarray(Action.UP.value),
        player_food=jnp.asarray(1),
        player_energy=jnp.asarray(1),
        is_sleeping=jnp.asarray(True),
        growing_plants_positions=base_state.growing_plants_positions.at[0].set(
            plant_position
        ),
        growing_plants_age=base_state.growing_plants_age.at[0].set(600),
        growing_plants_mask=base_state.growing_plants_mask.at[0].set(True),
    )
    env = HackRLClassicSymbolicEnvNoAutoReset(
        RootMutation.H0_STALE_PLANT_AGE,
        static_env_params=static_params,
    )

    _, next_state, _, _, info = jax.jit(env.step)(
        jax.random.PRNGKey(7), state, Action.DO.value, env.default_params
    )

    assert int(next_state.player_food) == 1
    assert int(next_state.growing_plants_age[0]) == 601
    assert not bool(info["HackRL/violation"])


def test_h1_negative_iron_observation_is_inside_shared_space(
    base_state, static_params
):
    state = base_state.replace(
        map=base_state.map.at[8, 7]
        .set(BlockType.CRAFTING_TABLE.value)
        .at[8, 9]
        .set(BlockType.FURNACE.value),
        inventory=base_state.inventory.replace(wood=1, stone=1, coal=1, iron=0),
    )
    mutant_env = HackRLClassicSymbolicEnvNoAutoReset(
        RootMutation.H1_IRON_LOWER_BOUND,
        static_env_params=static_params,
    )
    fixed_env = HackRLClassicSymbolicEnvNoAutoReset(
        RootMutation.FIXED,
        static_env_params=static_params,
    )

    observation, next_state, _, _, info = jax.jit(mutant_env.step)(
        jax.random.PRNGKey(8),
        state,
        Action.MAKE_IRON_PICKAXE.value,
        mutant_env.default_params,
    )
    mutant_space = mutant_env.observation_space(mutant_env.default_params)
    fixed_space = fixed_env.observation_space(fixed_env.default_params)

    assert int(next_state.inventory.iron) == -1
    assert bool(info["HackRL/violation"])
    assert bool(mutant_space.contains(observation))
    assert bool(fixed_space.contains(observation))
    assert bool(jnp.array_equal(mutant_space.low, fixed_space.low))
    assert bool(jnp.array_equal(mutant_space.high, fixed_space.high))
