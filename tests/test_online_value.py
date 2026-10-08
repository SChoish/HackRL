import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hackrl.online_value import (
    AllGoalTransition,
    GoalQTransition,
    all_goal_td_loss,
    combine_dual_q,
    epsilon_at_phase_step,
    init_all_goal_q,
    init_goal_q,
    leo_one_step_targets,
    online_one_step_targets,
    reset_phase_counter,
    update_all_goal_q,
    update_goal_q,
    weighted_td_loss,
)


def _examples(batch_size=4):
    maps = jnp.arange(batch_size * 3 * 3 * 2, dtype=jnp.float32).reshape(
        batch_size, 3, 3, 2
    ) / 100.0
    numeric = jnp.arange(batch_size * 4, dtype=jnp.float32).reshape(
        batch_size, 4
    ) / 10.0
    goals = jax.nn.one_hot(jnp.arange(batch_size) % 3, 3)
    return maps, numeric, goals


def _all_finite(tree):
    return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(tree))


def test_online_one_step_targets_match_hand_calculation():
    result = online_one_step_targets(
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([False, True]),
        jnp.asarray([[3.0, 4.0], [5.0, 6.0]]),
        0.5,
    )
    np.testing.assert_allclose(result, [3.0, 2.0])


def test_leo_targets_use_per_goal_and_world_terminal_masks():
    result = leo_one_step_targets(
        jnp.asarray([[True, False], [False, False]]),
        jnp.asarray([False, True]),
        jnp.asarray([[[9.0, 8.0], [2.0, 4.0]], [[7.0, 6.0], [5.0, 3.0]]]),
        0.5,
    )
    np.testing.assert_allclose(result, [[1.0, 2.0], [0.0, 0.0]])


def test_loss_denominators_are_valid_physical_transitions():
    valid = jnp.asarray([True, False])
    assert float(weighted_td_loss(jnp.zeros(2), jnp.ones(2), valid)) == pytest.approx(0.5)
    assert float(
        all_goal_td_loss(jnp.zeros((2, 2)), jnp.ones((2, 2)), valid)
    ) == pytest.approx(1.0)


def test_dual_combination_and_phase_schedule_are_exact():
    pqn = jnp.asarray([[1.0, 3.0]])
    leo = jnp.asarray([[5.0, 1.0]])
    np.testing.assert_allclose(combine_dual_q(pqn, leo), [[2.2, 2.4]])
    with pytest.raises(ValueError):
        combine_dual_q(pqn, leo, leo_weight=1.1)
    assert float(
        epsilon_at_phase_step(50, start=1.0, finish=0.1, decay_transitions=100)
    ) == pytest.approx(0.55)


def test_pqn_state_has_no_replay_target_or_q_lambda_and_updates_finitely():
    maps, numeric, goals = _examples()
    network, state = init_goal_q(
        jax.random.PRNGKey(0),
        maps,
        numeric,
        goals,
        num_actions=5,
        hidden_size=8,
        learning_rate=1e-3,
        max_grad_norm=1.0,
    )
    for forbidden in ("target_params", "replay", "q_lambda"):
        assert not hasattr(state, forbidden)
    transition = GoalQTransition(
        map_channels=maps,
        numeric_features=numeric,
        goal_one_hot=goals,
        action=jnp.asarray([0, 1, 2, 3]),
        reward=jnp.asarray([0.0, 1.0, 0.5, 0.0]),
        next_map_channels=maps + 0.01,
        next_numeric_features=numeric + 0.01,
        next_goal_one_hot=goals,
        done=jnp.asarray([False, True, False, False]),
        valid=jnp.asarray([True, True, True, False]),
    )
    updated, metrics = update_goal_q(network, state, transition, gamma=0.99)
    assert int(updated.environment_steps) == 4
    assert int(updated.phase_steps) == 4
    assert _all_finite(updated)
    assert _all_finite(metrics)
    restarted = reset_phase_counter(updated)
    assert int(restarted.phase_steps) == 0
    assert int(restarted.environment_steps) == 4


def test_leo_update_is_separate_all_goal_td_and_finite():
    maps, numeric, _ = _examples()
    network, state = init_all_goal_q(
        jax.random.PRNGKey(1),
        maps,
        numeric,
        num_goals=3,
        num_actions=5,
        hidden_size=8,
        learning_rate=1e-3,
        max_grad_norm=1.0,
    )
    transition = AllGoalTransition(
        map_channels=maps,
        numeric_features=numeric,
        action=jnp.asarray([0, 1, 2, 3]),
        terminal_goals=jnp.asarray(
            [[False, False, False], [True, False, False], [False, True, False], [False, False, False]]
        ),
        next_map_channels=maps + 0.01,
        next_numeric_features=numeric + 0.01,
        world_done=jnp.asarray([False, False, True, False]),
        valid=jnp.asarray([True, True, True, False]),
    )
    updated, metrics = update_all_goal_q(network, state, transition, gamma=0.99)
    assert int(updated.environment_steps) == 4
    assert int(metrics["td_targets"]) == 9
    assert _all_finite(updated)
    assert _all_finite(metrics)
