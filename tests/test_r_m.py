import jax
import jax.numpy as jnp
import pytest
from craftax.craftax_classic.constants import Action, BlockType

from hackrl import EasyTask, HackRLEasySymbolicEnvNoAutoReset, MediumTask
from hackrl.scripted_paths import (
    R_M_DIAMOND_SEGMENT,
    R_M_EXPLOIT_PATH,
    R_M_NORMAL_PATH,
)
from hackrl.tasks import R_M_DIAMOND, StartMode, r_m_prefix_actions


def _rollout(env, actions, seed=0):
    params = env.default_params
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    _, state = env.reset(reset_key, params)
    history = []
    for action in actions:
        key, step_key = jax.random.split(key)
        _, state, reward, done, info = env.step(
            step_key, state, action, params
        )
        history.append((state, reward, done, info))
        if bool(done):
            break
    return history


def _env(mutant):
    return HackRLEasySymbolicEnvNoAutoReset(MediumTask.R_M, mutant=mutant)


def test_r_m_keeps_r_e_start_and_adds_diamond():
    env = _env(False)
    assert env.spec.horizon == 512
    _, state = env.reset(jax.random.PRNGKey(0), env.default_params)
    assert tuple(map(int, state.player_position)) == (8, 8)
    assert int(state.inventory.wood) == 1
    assert int(state.inventory.iron) == 0
    assert int(state.inventory.diamond) == 0
    assert int(state.map[8, 7]) == BlockType.CRAFTING_TABLE.value
    assert int(state.map[6, 8]) == BlockType.IRON.value
    assert int(state.map[R_M_DIAMOND]) == BlockType.DIAMOND.value


def test_r_m_fixed_and_mutant_share_start():
    key = jax.random.PRNGKey(1)
    fixed = _env(False)
    mutant = _env(True)
    fixed_obs, fixed_state = fixed.reset(key, fixed.default_params)
    mutant_obs, mutant_state = mutant.reset(key, mutant.default_params)
    comparisons = jax.tree_util.tree_map(
        jnp.array_equal, fixed_state, mutant_state
    )
    assert all(bool(value) for value in jax.tree_util.tree_leaves(comparisons))
    assert bool(jnp.array_equal(fixed_obs, mutant_obs))


@pytest.mark.parametrize("mutant", [False, True])
def test_r_m_normal_path_reaches_diamond_after_craft(mutant):
    history = _rollout(_env(mutant), R_M_NORMAL_PATH)
    crafted = history[3][0]
    assert int(crafted.inventory.iron_pickaxe) >= 1
    assert int(crafted.inventory.diamond) == 0
    final = history[-1]
    assert len(history) == 6
    assert float(final[1]) == 1.0
    assert bool(final[2])
    assert int(final[0].inventory.diamond) >= 1


def test_r_m_exploit_needs_the_same_diamond_segment():
    mutant = _rollout(_env(True), R_M_EXPLOIT_PATH)
    assert int(mutant[0][0].inventory.iron_pickaxe) >= 1
    assert int(mutant[0][0].inventory.diamond) == 0
    assert not bool(mutant[0][2])
    assert float(mutant[-1][1]) == 1.0
    assert bool(mutant[-1][2])
    assert R_M_EXPLOIT_PATH[1:] == R_M_DIAMOND_SEGMENT
    assert R_M_NORMAL_PATH[-2:] == R_M_DIAMOND_SEGMENT

    fixed = _rollout(_env(False), R_M_EXPLOIT_PATH)
    assert int(fixed[0][0].inventory.iron_pickaxe) == 0
    assert float(fixed[-1][1]) == 0.0
    assert not bool(fixed[-1][2])
    assert int(fixed[-1][0].inventory.diamond) == 0


def test_r_m_transition_events_distinguish_all_three_stages():
    normal = _rollout(_env(False), R_M_NORMAL_PATH)
    assert bool(normal[1][3]["HackRL/iron_acquired"])
    assert not bool(normal[1][3]["HackRL/iron_pickaxe_crafted"])
    assert bool(normal[3][3]["HackRL/iron_pickaxe_crafted"])
    assert bool(normal[3][3]["HackRL/wood_depleted"])
    assert bool(normal[-1][3]["HackRL/diamond_acquired"])
    assert bool(normal[-1][3]["HackRL/termination_goal"])
    assert not any(
        bool(step[3]["HackRL/damage_taken"]) for step in normal
    )

    exploit = _rollout(_env(True), R_M_EXPLOIT_PATH)
    assert not any(bool(step[3]["HackRL/iron_acquired"]) for step in exploit)
    assert bool(exploit[0][3]["HackRL/iron_pickaxe_crafted"])
    assert bool(exploit[0][3]["HackRL/wood_depleted"])
    assert bool(exploit[-1][3]["HackRL/diamond_acquired"])


def _diagnostic_env(start_mode):
    return HackRLEasySymbolicEnvNoAutoReset(
        MediumTask.R_M,
        mutant=False,
        start_mode=start_mode,
    )


@pytest.mark.parametrize(
    "start_mode,prefix,position,iron,pickaxe,suffix",
    [
        (StartMode.R_M_D1, 5, (9, 8), 0, 1, (Action.DO.value,)),
        (StartMode.R_M_D2, 4, (8, 8), 0, 1, (Action.DOWN.value, Action.DO.value)),
        (
            StartMode.R_M_D3,
            3,
            (8, 8),
            1,
            0,
            (
                Action.MAKE_IRON_PICKAXE.value,
                Action.DOWN.value,
                Action.DO.value,
            ),
        ),
    ],
)
def test_r_m_diagnostic_starts_execute_real_prefix(
    start_mode, prefix, position, iron, pickaxe, suffix
):
    env = _diagnostic_env(start_mode)
    key = jax.random.PRNGKey(0)
    observation, state = env.reset(key, env.default_params)
    assert tuple(map(int, state.player_position)) == position
    assert int(state.player_direction) == Action.DOWN.value
    assert int(state.timestep) == prefix
    assert int(state.inventory.iron) == iron
    assert int(state.inventory.iron_pickaxe) >= pickaxe
    assert int(state.inventory.diamond) == 0
    assert int(state.map[6, 8]) != BlockType.IRON.value
    assert int(state.map[R_M_DIAMOND]) == BlockType.DIAMOND.value
    assert bool(jnp.array_equal(observation, env.get_obs(state)))
    assert r_m_prefix_actions(start_mode) == R_M_NORMAL_PATH[:prefix]
    history = _rollout(env, suffix)
    assert float(history[-1][1]) == 1.0
    assert bool(history[-1][2])
    assert int(history[-1][0].inventory.diamond) >= 1
    assert bool(history[-1][3]["HackRL/diamond_acquired"])
    if start_mode is StartMode.R_M_D3:
        assert bool(history[0][3]["HackRL/iron_pickaxe_crafted"])
        assert not bool(history[-1][3]["HackRL/iron_acquired"])
    else:
        assert not any(bool(step[3]["HackRL/iron_acquired"]) for step in history)


def test_r_m_diagnostic_starts_are_not_inventory_gifts():
    default = HackRLEasySymbolicEnvNoAutoReset(MediumTask.R_M, mutant=False)
    gifted = default.reset(jax.random.PRNGKey(0), default.default_params)[1]
    assert int(gifted.inventory.iron) == 0
    assert int(gifted.inventory.iron_pickaxe) == 0
    assert int(gifted.timestep) == 0
    assert tuple(map(int, gifted.player_position)) == (8, 8)


def test_r_m_diagnostic_start_rejects_easy_tasks():
    with pytest.raises(ValueError, match="r_m_d1/d2/d3"):
        HackRLEasySymbolicEnvNoAutoReset(
            EasyTask.R_E, start_mode=StartMode.R_M_D1
        )
