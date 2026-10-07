import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from hackrl.gc_double_dqn import (
    ReplayTransition,
    append_replay,
    double_dqn_targets,
    exploration_epsilon,
    init_double_dqn,
    init_replay_buffer,
    load_checkpoint,
    make_replay_updates,
    save_checkpoint,
    select_goal_q,
)


def _transition(count, *, value=0.0, valid=True):
    maps = jnp.full((count, 3, 3, 2), value, dtype=jnp.float32)
    numeric = jnp.full((count, 3), value, dtype=jnp.float32)
    return ReplayTransition(
        map_channels=maps,
        numeric_features=numeric,
        goal_index=jnp.arange(count, dtype=jnp.int32) % 2,
        action=jnp.arange(count, dtype=jnp.int32) % 3,
        reward=jnp.zeros((count,), dtype=jnp.float32),
        next_map_channels=maps + 1,
        next_numeric_features=numeric + 1,
        next_goal_index=jnp.arange(count, dtype=jnp.int32) % 2,
        done=jnp.zeros((count,), dtype=jnp.bool_),
        valid=jnp.full((count,), valid, dtype=jnp.bool_),
        explored=jnp.zeros((count,), dtype=jnp.bool_),
    )


def test_double_dqn_uses_online_argmax_and_target_value():
    online = jnp.asarray([[1.0, 5.0, 3.0], [9.0, 1.0, 0.0]])
    target = jnp.asarray([[20.0, 7.0, 99.0], [4.0, 8.0, 2.0]])
    reward = jnp.asarray([0.0, 1.0])
    done = jnp.asarray([False, True])
    result = double_dqn_targets(online, target, reward, done, 0.5)
    np.testing.assert_allclose(result, [3.5, 1.0])


def test_goal_head_selection_is_per_example():
    values = jnp.arange(2 * 3 * 4).reshape((2, 3, 4))
    selected = select_goal_q(values, jnp.asarray([2, 0]))
    np.testing.assert_array_equal(selected, np.asarray([values[0, 2], values[1, 0]]))


def test_epsilon_schedule_clamps_at_end():
    assert float(
        exploration_epsilon(0, start=1.0, end=0.05, decay_transitions=100)
    ) == 1.0
    np.testing.assert_allclose(
        float(
            exploration_epsilon(
                50, start=1.0, end=0.05, decay_transitions=100
            )
        ),
        0.525,
    )
    np.testing.assert_allclose(
        float(
            exploration_epsilon(
                1000, start=1.0, end=0.05, decay_transitions=100
            )
        ),
        0.05,
        atol=1e-7,
    )


def test_replay_wraps_and_tracks_generated_transitions():
    replay = init_replay_buffer(6, (3, 3, 2), (3,))
    replay = append_replay(replay, _transition(4, value=1.0))
    assert int(replay.size) == 4
    assert int(replay.cursor) == 4
    replay = append_replay(replay, _transition(4, value=2.0))
    assert int(replay.size) == 6
    assert int(replay.cursor) == 2
    assert int(replay.total_inserted) == 8
    np.testing.assert_allclose(
        np.asarray(replay.numeric_features[:2]), 2.0
    )


def test_replay_updates_online_and_hard_target_state():
    maps = jnp.zeros((8, 3, 3, 2), dtype=jnp.float32)
    numeric = jnp.zeros((8, 3), dtype=jnp.float32)
    network, state = init_double_dqn(
        jax.random.PRNGKey(0),
        maps,
        numeric,
        num_goals=2,
        num_actions=3,
        hidden_size=8,
        learning_rate=1e-3,
        max_grad_norm=1.0,
    )
    replay = init_replay_buffer(16, maps.shape[1:], numeric.shape[1:])
    transition = _transition(16, value=0.25)
    transition = transition.replace(
        reward=jnp.ones((16,), dtype=jnp.float32),
        done=jnp.ones((16,), dtype=jnp.bool_),
    )
    replay = append_replay(replay, transition)
    update = jax.jit(
        make_replay_updates(
            network,
            batch_size=8,
            gradient_steps=2,
            target_update_interval=1,
            gamma=0.995,
        )
    )
    updated, _, metrics = update(state, replay, jax.random.PRNGKey(1))
    assert int(updated.step) == 2
    assert int(metrics["applied_gradient_steps"]) == 2
    assert int(metrics["target_syncs"]) == 2
    leaves = jax.tree.leaves(
        jax.tree.map(
            lambda left, right: jnp.array_equal(left, right),
            updated.params,
            updated.target_params,
        )
    )
    assert all(bool(value) for value in leaves)


class _CheckpointRunner(struct.PyTreeNode):
    global_update: jax.Array
    env_steps: jax.Array
    rng: jax.Array


def _trees_equal(left, right):
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    return len(left_leaves) == len(right_leaves) and all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(left_leaves, right_leaves)
    )


def test_invalid_replay_rows_are_retained_but_not_sampled_for_learning():
    maps = jnp.zeros((8, 3, 3, 2), dtype=jnp.float32)
    numeric = jnp.zeros((8, 3), dtype=jnp.float32)
    network, state = init_double_dqn(
        jax.random.PRNGKey(3),
        maps,
        numeric,
        num_goals=2,
        num_actions=3,
        hidden_size=8,
        learning_rate=1e-3,
        max_grad_norm=1.0,
    )
    replay = init_replay_buffer(16, maps.shape[1:], numeric.shape[1:])
    replay = append_replay(replay, _transition(1, value=0.25, valid=True))
    replay = append_replay(replay, _transition(15, value=0.75, valid=False))
    assert int(jnp.sum(replay.valid)) == 1
    update = jax.jit(
        make_replay_updates(
            network,
            batch_size=8,
            gradient_steps=2,
            target_update_interval=99,
            gamma=0.995,
        )
    )
    updated, _, metrics = update(state, replay, jax.random.PRNGKey(4))
    assert int(updated.step) == 2
    assert int(metrics["sampled_valid_transitions"]) == 16


def test_checkpoint_round_trip_restores_full_value_and_replay_state(tmp_path):
    maps = jnp.zeros((8, 3, 3, 2), dtype=jnp.float32)
    numeric = jnp.zeros((8, 3), dtype=jnp.float32)
    _, state = init_double_dqn(
        jax.random.PRNGKey(5),
        maps,
        numeric,
        num_goals=2,
        num_actions=3,
        hidden_size=8,
        learning_rate=1e-3,
        max_grad_norm=1.0,
    )
    replay = init_replay_buffer(16, maps.shape[1:], numeric.shape[1:])
    replay = append_replay(replay, _transition(7, value=0.5))
    runner = _CheckpointRunner(
        global_update=jnp.asarray(12, dtype=jnp.int32),
        env_steps=jnp.asarray(393216, dtype=jnp.int32),
        rng=jax.random.PRNGKey(6),
    )
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        runner=runner,
        train_state=state,
        replay=replay,
        environment_config={"variant": "fixed"},
        algorithm_config={"algorithm": "double_dqn"},
    )
    restored = load_checkpoint(
        checkpoint,
        runner=runner,
        train_state=state,
        replay=replay,
    )
    assert _trees_equal(restored["runner"], runner)
    assert _trees_equal(restored["train_state"], state)
    assert _trees_equal(restored["replay"], replay)
    assert not list(checkpoint.glob("*.tmp"))
