import jax
import jax.numpy as jnp
import pytest
from craftax.craftax_classic.constants import Action, BlockType

from hackrl import EasyTask, FixtureVersion, HackRLEasySymbolicEnvNoAutoReset
from hackrl.metrics import init_episode_tracker, update_episode_tracker
from hackrl.scripted_paths import (
    EASY_EXPLOIT_PATHS,
    EASY_NORMAL_PATHS,
    R_E_WOOD_WASTE_RECOVERY_PATH,
)
from hackrl.tasks import (
    R_E_REPLENISH_COALS,
    R_E_REPLENISH_STONES,
    R_E_REPLENISH_TREES,
)


def _rollout(env, actions, seed=0):
    params = env.default_params
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    _, state = env.reset(reset_key, params)
    tracker = init_episode_tracker(env.static_env_params.max_growing_plants)
    transitions = []
    for action in actions:
        key, step_key = jax.random.split(key)
        _, state, reward, done, info = env.step(
            step_key, state, action, params
        )
        tracker, _ = update_episode_tracker(
            tracker, reward, info, state.timestep
        )
        transitions.append((reward, done, info, state))
        if bool(done):
            break
    return state, tracker, transitions


def _replenish_env(mutant):
    return HackRLEasySymbolicEnvNoAutoReset(
        EasyTask.R_E,
        mutant=mutant,
        fixture=FixtureVersion.R_E_REPLENISH,
    )


def test_replenish_keeps_original_start_and_workshop():
    original = HackRLEasySymbolicEnvNoAutoReset(EasyTask.R_E, mutant=False)
    replenish = _replenish_env(False)
    key = jax.random.PRNGKey(11)
    _, original_state = original.reset(key, original.default_params)
    _, replenish_state = replenish.reset(key, replenish.default_params)

    assert tuple(map(int, replenish_state.player_position)) == (8, 8)
    assert int(replenish_state.player_direction) == Action.UP.value
    assert int(replenish_state.inventory.wood) == 1
    assert int(replenish_state.inventory.stone) == 1
    assert int(replenish_state.inventory.coal) == 1
    assert int(replenish_state.inventory.iron) == 0
    assert int(replenish_state.map[8, 7]) == BlockType.CRAFTING_TABLE.value
    assert int(replenish_state.map[8, 9]) == BlockType.FURNACE.value
    assert int(replenish_state.map[6, 8]) == BlockType.IRON.value
    assert int(original_state.map[9, 10]) == BlockType.GRASS.value
    assert int(replenish_state.map[9, 10]) == BlockType.TREE.value


def test_replenish_places_three_of_each_resource_off_the_iron_path():
    env = _replenish_env(False)
    _, state = env.reset(jax.random.PRNGKey(12), env.default_params)
    for row, col in R_E_REPLENISH_TREES:
        assert int(state.map[row, col]) == BlockType.TREE.value
    for row, col in R_E_REPLENISH_STONES:
        assert int(state.map[row, col]) == BlockType.STONE.value
    for row, col in R_E_REPLENISH_COALS:
        assert int(state.map[row, col]) == BlockType.COAL.value
    assert int(state.map[7, 8]) == BlockType.GRASS.value
    assert int(state.map[8, 8]) == BlockType.GRASS.value


def test_replenish_fixed_and_mutant_share_the_same_start():
    fixed = _replenish_env(False)
    mutant = _replenish_env(True)
    key = jax.random.PRNGKey(13)
    fixed_obs, fixed_state = fixed.reset(key, fixed.default_params)
    mutant_obs, mutant_state = mutant.reset(key, mutant.default_params)
    comparisons = jax.tree_util.tree_map(
        jnp.array_equal, fixed_state, mutant_state
    )
    assert all(bool(value) for value in jax.tree_util.tree_leaves(comparisons))
    assert bool(jnp.array_equal(fixed_obs, mutant_obs))


@pytest.mark.parametrize("mutant", [False, True])
def test_replenish_keeps_normal_three_step_path(mutant):
    state, _, transitions = _rollout(
        _replenish_env(mutant), EASY_NORMAL_PATHS[EasyTask.R_E]
    )
    assert len(transitions) == 3
    assert bool(transitions[-1][1])
    assert float(transitions[-1][0]) == 1.0
    assert int(state.inventory.iron_pickaxe) >= 1


@pytest.mark.parametrize("mutant", [False, True])
def test_replenish_keeps_mutant_one_step_split(mutant):
    state, _, transitions = _rollout(
        _replenish_env(mutant), EASY_EXPLOIT_PATHS[EasyTask.R_E]
    )
    succeeded = bool(transitions[-1][1]) and float(transitions[-1][0]) == 1.0
    assert succeeded is mutant
    if mutant:
        assert int(state.inventory.iron_pickaxe) >= 1


def test_replenish_recovers_after_wasted_wood_on_fixed():
    state, tracker, transitions = _rollout(
        _replenish_env(False), R_E_WOOD_WASTE_RECOVERY_PATH
    )
    wasted = transitions[0][3]
    harvested = transitions[3][3]
    assert int(wasted.inventory.wood) == 0
    assert int(harvested.inventory.wood) >= 1
    assert int(harvested.map[9, 10]) == BlockType.GRASS.value
    assert float(transitions[-1][0]) == 1.0
    assert bool(transitions[-1][1])
    assert int(state.inventory.iron_pickaxe) >= 1
    assert bool(tracker.wood_depleted_before_pickaxe)
    assert bool(tracker.wood_replenished_after_depletion)
    assert bool(tracker.goal_success)


def test_replenish_is_rejected_for_other_tasks():
    with pytest.raises(ValueError, match="r_e_replenish"):
        HackRLEasySymbolicEnvNoAutoReset(
            EasyTask.B_E, fixture=FixtureVersion.R_E_REPLENISH
        )
