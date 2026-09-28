import jax
import jax.numpy as jnp
from craftax.craftax_classic import game_logic as base
from craftax.craftax_classic.constants import Action, BlockType
from craftax.craftax_classic.envs.craftax_state import EnvParams

from hackrl.mutations import (
    RootMutation,
    craftax_step_with_events,
    detect_violation,
    do_action,
    do_crafting,
)


def test_h0_preserves_harvested_plant_age_and_allows_immediate_reharvest(
    base_state, static_params
):
    plant_position = jnp.array([7, 8], dtype=jnp.int32)
    state = base_state.replace(
        map=base_state.map.at[7, 8].set(BlockType.RIPE_PLANT.value),
        player_direction=jnp.asarray(Action.UP.value),
        player_food=jnp.asarray(1),
        growing_plants_positions=base_state.growing_plants_positions.at[0].set(
            plant_position
        ),
        growing_plants_age=base_state.growing_plants_age.at[0].set(600),
        growing_plants_mask=base_state.growing_plants_mask.at[0].set(True),
    )
    key = jax.random.PRNGKey(1)

    fixed = do_action(
        key, state, Action.DO.value, static_params, RootMutation.FIXED
    )
    mutant_after_action = do_action(
        key,
        state,
        Action.DO.value,
        static_params,
        RootMutation.H0_STALE_PLANT_AGE,
    )
    assert bool(
        detect_violation(
            RootMutation.H0_STALE_PLANT_AGE,
            state,
            Action.DO.value,
            mutant_after_action,
        )
    )
    fixed = base.update_plants(fixed, static_params)
    mutant = base.update_plants(mutant_after_action, static_params)

    assert int(fixed.growing_plants_age[0]) == 1
    assert int(fixed.map[7, 8]) == BlockType.PLANT.value
    assert int(mutant.growing_plants_age[0]) == 601
    assert int(mutant.map[7, 8]) == BlockType.RIPE_PLANT.value
    fixed_twice = do_action(
        key, fixed, Action.DO.value, static_params, RootMutation.FIXED
    )
    mutant_twice = do_action(
        key,
        mutant,
        Action.DO.value,
        static_params,
        RootMutation.H0_STALE_PLANT_AGE,
    )
    assert int(fixed_twice.player_food) == 5
    assert int(mutant_twice.player_food) == 9


def test_h1_allows_exactly_one_zero_iron_pickaxe(base_state):
    state = base_state.replace(
        map=base_state.map.at[8, 7]
        .set(BlockType.CRAFTING_TABLE.value)
        .at[8, 9]
        .set(BlockType.FURNACE.value),
        inventory=base_state.inventory.replace(wood=1, stone=1, coal=1, iron=0),
    )

    fixed = do_crafting(
        state, Action.MAKE_IRON_PICKAXE.value, RootMutation.FIXED
    )
    mutant = do_crafting(
        state, Action.MAKE_IRON_PICKAXE.value, RootMutation.H1_IRON_LOWER_BOUND
    )

    assert int(fixed.inventory.iron_pickaxe) == 0
    assert int(fixed.inventory.iron) == 0
    assert int(mutant.inventory.iron_pickaxe) == 1
    assert int(mutant.inventory.iron) == -1
    assert bool(
        detect_violation(
            RootMutation.H1_IRON_LOWER_BOUND,
            state,
            Action.MAKE_IRON_PICKAXE.value,
            mutant,
        )
    )

    second_attempt = do_crafting(
        mutant,
        Action.MAKE_IRON_PICKAXE.value,
        RootMutation.H1_IRON_LOWER_BOUND,
    )
    assert int(second_attempt.inventory.iron_pickaxe) == 1
    assert int(second_attempt.inventory.iron) == -1


def test_h1_is_jittable(base_state):
    state = base_state.replace(
        map=base_state.map.at[8, 7]
        .set(BlockType.CRAFTING_TABLE.value)
        .at[8, 9]
        .set(BlockType.FURNACE.value),
        inventory=base_state.inventory.replace(wood=1, stone=1, coal=1, iron=0),
    )
    transition = jax.jit(
        lambda current: do_crafting(
            current,
            Action.MAKE_IRON_PICKAXE.value,
            RootMutation.H1_IRON_LOWER_BOUND,
        )
    )
    result = transition(state)
    assert int(result.inventory.iron) == -1
    assert int(result.inventory.iron_pickaxe) == 1


def test_h2_negative_index_mines_opposite_edge(base_state, static_params):
    target_column = 8
    state = base_state.replace(
        map=base_state.map.at[-1, target_column].set(BlockType.IRON.value),
        player_position=jnp.array([0, target_column], dtype=jnp.int32),
        player_direction=jnp.asarray(Action.UP.value),
        inventory=base_state.inventory.replace(stone_pickaxe=1),
    )
    key = jax.random.PRNGKey(2)

    fixed = do_action(
        key, state, Action.DO.value, static_params, RootMutation.FIXED
    )
    mutant = do_action(
        key,
        state,
        Action.DO.value,
        static_params,
        RootMutation.H2_MISSING_MAP_BOUNDS,
    )

    assert int(fixed.inventory.iron) == 0
    assert int(fixed.map[-1, target_column]) == BlockType.IRON.value
    assert not bool(
        detect_violation(
            RootMutation.H2_MISSING_MAP_BOUNDS, state, Action.DO.value, fixed
        )
    )
    assert int(mutant.inventory.iron) == 1
    assert int(mutant.map[-1, target_column]) == BlockType.PATH.value
    assert bool(
        detect_violation(
            RootMutation.H2_MISSING_MAP_BOUNDS, state, Action.DO.value, mutant
        )
    )


def test_unrelated_action_matches_fixed_kernel(base_state, static_params):
    key = jax.random.PRNGKey(3)
    expected = base.do_action(key, base_state, Action.NOOP.value, static_params)

    for mutation in RootMutation:
        actual = do_action(key, base_state, Action.NOOP.value, static_params, mutation)
        comparisons = jax.tree_util.tree_map(jnp.array_equal, expected, actual)
        assert all(bool(value) for value in jax.tree_util.tree_leaves(comparisons))


def test_transition_kernel_and_contract_are_selected_independently(
    base_state, static_params
):
    state = base_state.replace(
        map=base_state.map.at[8, 7]
        .set(BlockType.CRAFTING_TABLE.value)
        .at[8, 9]
        .set(BlockType.FURNACE.value),
        inventory=base_state.inventory.replace(wood=1, stone=1, coal=1, iron=0),
        zombies=base_state.zombies.replace(
            health=jnp.zeros_like(base_state.zombies.health)
        ),
        cows=base_state.cows.replace(
            health=jnp.zeros_like(base_state.cows.health)
        ),
        skeletons=base_state.skeletons.replace(
            health=jnp.zeros_like(base_state.skeletons.health)
        ),
        arrows=base_state.arrows.replace(
            health=jnp.zeros_like(base_state.arrows.health)
        ),
    )
    params = EnvParams(
        spawn_cow_chance=0.0,
        spawn_zombie_base_chance=0.0,
        spawn_zombie_night_chance=0.0,
        spawn_skeleton_chance=0.0,
    )
    key = jax.random.PRNGKey(4)

    fixed, _, fixed_events = craftax_step_with_events(
        key,
        state,
        Action.MAKE_IRON_PICKAXE.value,
        params,
        static_params,
        RootMutation.FIXED,
        contract=RootMutation.H1_IRON_LOWER_BOUND,
    )
    mutant, _, mutant_events = craftax_step_with_events(
        key,
        state,
        Action.MAKE_IRON_PICKAXE.value,
        params,
        static_params,
        RootMutation.H1_IRON_LOWER_BOUND,
        contract=RootMutation.H1_IRON_LOWER_BOUND,
    )

    assert int(fixed.inventory.iron_pickaxe) == 0
    assert not bool(fixed_events["violation"])
    assert int(mutant.inventory.iron_pickaxe) == 1
    assert bool(mutant_events["violation"])
