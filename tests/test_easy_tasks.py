import jax
import jax.numpy as jnp
import pytest
from craftax.craftax_classic.constants import Action, BlockType

from hackrl import (
    EasyTask,
    HackRLEasySymbolicEnvNoAutoReset,
    MediumTask,
    StartMode,
)
from hackrl.scripted_paths import EASY_EXPLOIT_PATHS, EASY_NORMAL_PATHS


def _rollout(env, actions, seed=0):
    params = env.default_params
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    observation, state = env.reset(reset_key, params)
    transitions = []

    for action in actions:
        key, step_key = jax.random.split(key)
        observation, state, reward, done, info = env.step(
            step_key, state, action, params
        )
        transitions.append((reward, done, info))
        if bool(done):
            break

    return observation, state, transitions


def _assert_no_mobs(state):
    assert not bool(state.mob_map.any())
    for mobs in (state.zombies, state.cows, state.skeletons, state.arrows):
        assert not bool(mobs.mask.any())
        assert bool(jnp.all(mobs.health == 0))


@pytest.mark.parametrize("task", list(EasyTask))
def test_fixed_and_mutant_reset_to_identical_16x16_fixture(task):
    fixed = HackRLEasySymbolicEnvNoAutoReset(task, mutant=False)
    mutant = HackRLEasySymbolicEnvNoAutoReset(task, mutant=True)
    key = jax.random.PRNGKey(10)

    fixed_obs, fixed_state = fixed.reset(key, fixed.default_params)
    mutant_obs, mutant_state = mutant.reset(key, mutant.default_params)

    comparisons = jax.tree_util.tree_map(jnp.array_equal, fixed_state, mutant_state)
    assert all(bool(value) for value in jax.tree_util.tree_leaves(comparisons))
    assert bool(jnp.array_equal(fixed_obs, mutant_obs))
    assert fixed_state.map.shape == (16, 16)
    assert fixed_obs.shape == (1345,)
    assert bool(fixed.observation_space(fixed.default_params).contains(fixed_obs))
    _assert_no_mobs(fixed_state)


def test_no_mob_fixture_is_an_invariant_across_100_do_steps():
    env = HackRLEasySymbolicEnvNoAutoReset(MediumTask.R_M, mutant=False)
    params = env.default_params
    key = jax.random.PRNGKey(101)
    key, reset_key = jax.random.split(key)
    _, state = env.reset(reset_key, params)
    _assert_no_mobs(state)

    for _ in range(100):
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step(
            step_key, state, Action.DO.value, params
        )
        _assert_no_mobs(state)


def test_easy_fixture_contracts():
    r_env = HackRLEasySymbolicEnvNoAutoReset(EasyTask.R_E)
    _, r_state = r_env.reset(jax.random.PRNGKey(11), r_env.default_params)
    assert int(r_state.inventory.wood) == 1
    assert int(r_state.inventory.stone) == 1
    assert int(r_state.inventory.coal) == 1
    assert int(r_state.inventory.iron) == 0
    assert int(r_state.inventory.wood_pickaxe) == 1
    assert int(r_state.inventory.stone_pickaxe) == 1
    assert int(r_state.map[8, 7]) == BlockType.CRAFTING_TABLE.value
    assert int(r_state.map[8, 9]) == BlockType.FURNACE.value
    assert int(r_state.map[6, 8]) == BlockType.IRON.value

    post = HackRLEasySymbolicEnvNoAutoReset(
        EasyTask.R_E,
        mutant=False,
        start_mode=StartMode.R_E_POST_IRON,
    )
    _, post_state = post.reset(jax.random.PRNGKey(11), post.default_params)
    assert int(post_state.inventory.wood) == 1
    assert int(post_state.inventory.stone) == 1
    assert int(post_state.inventory.coal) == 1
    assert int(post_state.inventory.iron) == 1
    assert int(post_state.inventory.wood_pickaxe) == 1
    assert int(post_state.inventory.stone_pickaxe) == 1
    assert int(post_state.map[6, 8]) == BlockType.GRASS.value
    assert int(post_state.map[8, 7]) == BlockType.CRAFTING_TABLE.value
    assert int(post_state.map[8, 9]) == BlockType.FURNACE.value

    b_env = HackRLEasySymbolicEnvNoAutoReset(EasyTask.B_E)
    _, b_state = b_env.reset(jax.random.PRNGKey(12), b_env.default_params)
    assert tuple(map(int, b_state.player_position)) == (0, 8)
    assert int(b_state.player_direction) == Action.UP.value
    assert int(b_state.inventory.stone_pickaxe) == 1
    assert int(b_state.map[-1, 8]) == BlockType.IRON.value

    l_env = HackRLEasySymbolicEnvNoAutoReset(EasyTask.L_E)
    _, l_state = l_env.reset(jax.random.PRNGKey(13), l_env.default_params)
    assert int(l_state.player_food) == 1
    assert int(l_state.growing_plants_mask.sum()) == 2
    for index in (0, 1):
        position = l_state.growing_plants_positions[index]
        assert int(l_state.growing_plants_age[index]) == 600
        assert int(l_state.map[position[0], position[1]]) == BlockType.RIPE_PLANT.value


@pytest.mark.parametrize("task", list(EasyTask))
@pytest.mark.parametrize("mutant", [False, True])
def test_normal_path_reaches_goal_in_both_variants(task, mutant):
    env = HackRLEasySymbolicEnvNoAutoReset(task, mutant=mutant)
    _, state, transitions = _rollout(env, EASY_NORMAL_PATHS[task], seed=20)

    rewards = [float(reward) for reward, _, _ in transitions]
    assert bool(env.goal_reached(state))
    assert bool(transitions[-1][1])
    assert sum(rewards) == 1.0
    assert rewards[-1] == 1.0
    assert int(state.timestep) == len(EASY_NORMAL_PATHS[task])


@pytest.mark.parametrize("task", list(EasyTask))
def test_exploit_path_succeeds_only_in_mutant(task):
    mutant = HackRLEasySymbolicEnvNoAutoReset(task, mutant=True)
    _, mutant_state, mutant_transitions = _rollout(
        mutant, EASY_EXPLOIT_PATHS[task], seed=30
    )

    assert bool(mutant.goal_reached(mutant_state))
    assert bool(mutant_transitions[-1][1])
    assert sum(float(item[0]) for item in mutant_transitions) == 1.0
    assert any(bool(item[2]["HackRL/violation"]) for item in mutant_transitions)

    fixed = HackRLEasySymbolicEnvNoAutoReset(task, mutant=False)
    _, fixed_state, fixed_transitions = _rollout(
        fixed, EASY_EXPLOIT_PATHS[task], seed=30
    )

    assert not bool(fixed.goal_reached(fixed_state))
    assert not bool(fixed_transitions[-1][1])
    assert sum(float(item[0]) for item in fixed_transitions) == 0.0
    assert not any(bool(item[2]["HackRL/violation"]) for item in fixed_transitions)


def test_r_e_fixed_post_iron_crafts_in_one_action():
    env = HackRLEasySymbolicEnvNoAutoReset(
        EasyTask.R_E,
        mutant=False,
        start_mode=StartMode.R_E_POST_IRON,
    )
    _, _, transitions = _rollout(
        env, [Action.MAKE_IRON_PICKAXE.value], seed=41
    )

    reward, done, info = transitions[0]
    assert float(reward) == 1.0
    assert bool(done)
    assert bool(info["HackRL/goal_success"])


def test_post_iron_start_is_rejected_for_other_tasks():
    with pytest.raises(ValueError, match="r_e_post_iron"):
        HackRLEasySymbolicEnvNoAutoReset(
            EasyTask.B_E,
            start_mode=StartMode.R_E_POST_IRON,
        )


def test_non_goal_achievement_reward_is_log_only():
    env = HackRLEasySymbolicEnvNoAutoReset(EasyTask.R_E, mutant=False)
    _, state, transitions = _rollout(
        env, EASY_NORMAL_PATHS[EasyTask.R_E], seed=40
    )

    mining_reward, mining_done, mining_info = transitions[1]
    assert float(mining_reward) == 0.0
    assert not bool(mining_done)
    assert float(mining_info["HackRL/original_reward"]) == 1.0

    final_reward, final_done, final_info = transitions[-1]
    assert bool(env.goal_reached(state))
    assert float(final_reward) == 1.0
    assert bool(final_done)
    assert float(final_info["HackRL/original_reward"]) == 1.0


def test_timeout_and_goal_reward_is_not_repeated():
    timeout_env = HackRLEasySymbolicEnvNoAutoReset(EasyTask.B_E, mutant=False)
    params = timeout_env.default_params
    _, state = timeout_env.reset(jax.random.PRNGKey(50), params)
    state = state.replace(timestep=timeout_env.spec.horizon - 1)

    _, timed_out_state, reward, done, info = timeout_env.step(
        jax.random.PRNGKey(51), state, Action.NOOP.value, params
    )
    assert int(timed_out_state.timestep) == timeout_env.spec.horizon
    assert float(reward) == 0.0
    assert bool(done)
    assert bool(info["HackRL/termination_timeout"])
    assert not bool(info["HackRL/termination_goal"])

    goal_env = HackRLEasySymbolicEnvNoAutoReset(EasyTask.R_E, mutant=True)
    params = goal_env.default_params
    _, state = goal_env.reset(jax.random.PRNGKey(52), params)
    _, state, reward, done, _ = goal_env.step(
        jax.random.PRNGKey(53),
        state,
        Action.MAKE_IRON_PICKAXE.value,
        params,
    )
    assert float(reward) == 1.0
    assert bool(done)

    _, _, repeated_reward, repeated_done, _ = goal_env.step(
        jax.random.PRNGKey(54), state, Action.NOOP.value, params
    )
    assert float(repeated_reward) == 0.0
    assert bool(repeated_done)


def test_easy_environment_jit_vmap_batch():
    env = HackRLEasySymbolicEnvNoAutoReset(EasyTask.R_E, mutant=True)
    params = env.default_params
    reset_keys = jax.random.split(jax.random.PRNGKey(60), 4)
    observations, states = jax.jit(
        jax.vmap(lambda key: env.reset(key, params))
    )(reset_keys)

    step_keys = jax.random.split(jax.random.PRNGKey(61), 4)
    actions = jnp.full((4,), Action.MAKE_IRON_PICKAXE.value, dtype=jnp.int32)
    next_observations, next_states, rewards, dones, infos = jax.jit(
        jax.vmap(lambda key, state, action: env.step(key, state, action, params))
    )(step_keys, states, actions)

    assert observations.shape == (4, 1345)
    assert next_observations.shape == (4, 1345)
    assert next_states.map.shape == (4, 16, 16)
    assert bool(jnp.all(rewards == 1.0))
    assert bool(jnp.all(dones))
    assert bool(jnp.all(infos["HackRL/violation"]))
