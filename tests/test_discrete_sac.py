import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hackrl.discrete_sac import (
    SDSACBatch,
    append_sd_sac_replay,
    categorical_terms,
    double_average_soft_value,
    init_sd_sac,
    init_sd_sac_replay,
    q_clip_loss,
    record_sd_sac_environment_steps,
    sample_sd_sac_replay,
    sample_valid_sd_sac_replay,
    sd_sac_actor_loss,
    sd_sac_critic_target,
    temperature_loss,
    update_sd_sac,
    valid_sd_sac_replay_count,
)


def _all_finite(tree):
    return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(tree))


def _batch(rewards=None):
    size = 4 if rewards is None else len(rewards)
    maps = jnp.arange(size * 3 * 3 * 2, dtype=jnp.float32).reshape(
        size, 3, 3, 2
    ) / 100.0
    numeric = jnp.arange(size * 4, dtype=jnp.float32).reshape(size, 4) / 10.0
    goals = jax.nn.one_hot(jnp.arange(size) % 3, 3)
    return SDSACBatch(
        map_channels=maps,
        numeric_features=numeric,
        goal_one_hot=goals,
        action=jnp.arange(size, dtype=jnp.int32) % 5,
        reward=jnp.asarray(
            np.arange(size, dtype=np.float32) if rewards is None else rewards
        ),
        next_map_channels=maps + 0.01,
        next_numeric_features=numeric + 0.01,
        next_goal_one_hot=goals,
        done=jnp.arange(size) == 1,
        valid=jnp.arange(size) != size - 1,
        behavior_entropy=jnp.full((size,), np.log(5.0), dtype=jnp.float32),
    )


def test_categorical_expectation_entropy_target_and_terminal_mask():
    logits = jnp.asarray([[0.0, 0.0]])
    probabilities, log_probabilities, entropy = categorical_terms(logits)
    np.testing.assert_allclose(probabilities, [[0.5, 0.5]])
    np.testing.assert_allclose(log_probabilities, [[-np.log(2.0), -np.log(2.0)]])
    np.testing.assert_allclose(entropy, [np.log(2.0)])
    value = double_average_soft_value(
        logits, jnp.asarray([[1.0, 3.0]]), jnp.asarray([[3.0, 5.0]]), 0.5
    )
    np.testing.assert_allclose(value, [3.0 + 0.5 * np.log(2.0)])
    result = sd_sac_critic_target(
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([False, True]),
        jnp.repeat(logits, 2, axis=0),
        jnp.asarray([[1.0, 3.0], [1.0, 3.0]]),
        jnp.asarray([[3.0, 5.0], [3.0, 5.0]]),
        0.5,
        0.9,
    )
    np.testing.assert_allclose(result, [1.0 + 0.9 * (3.0 + 0.5 * np.log(2.0)), 2.0])


def test_q_clip_actor_penalty_and_temperature_direction_match_hand_values():
    loss = q_clip_loss(
        jnp.asarray([3.0]),
        jnp.asarray([0.0]),
        jnp.asarray([1.0]),
        jnp.asarray([True]),
        0.5,
    )
    assert float(loss) == pytest.approx(2.0)
    actor_loss, metrics = sd_sac_actor_loss(
        jnp.asarray([[0.0, 0.0]]),
        jnp.zeros((1, 2)),
        jnp.zeros((1, 2)),
        jnp.asarray([np.log(2.0)]),
        jnp.asarray([True]),
        alpha=0.5,
        beta=0.2,
    )
    assert float(actor_loss) == pytest.approx(-0.5 * np.log(2.0))
    assert float(metrics["entropy_penalty"]) == pytest.approx(0.0, abs=1e-7)
    gradient = jax.grad(temperature_loss)(
        jnp.asarray(0.0), jnp.asarray([0.2]), jnp.asarray([True]), 0.5
    )
    assert float(gradient) < 0.0


def test_replay_wraparound_and_sampling_preserve_behavior_entropy():
    replay = init_sd_sac_replay(3, (3, 3, 2), (4,), (3,))
    replay = append_sd_sac_replay(replay, _batch([0.0, 1.0]))
    replay = append_sd_sac_replay(replay, _batch([2.0, 3.0]))
    assert int(replay.size) == 3
    assert int(replay.cursor) == 1
    assert int(replay.total_inserted) == 4
    sampled = sample_sd_sac_replay(replay, jnp.asarray([0, 1, 2]))
    np.testing.assert_allclose(sampled.reward, [3.0, 1.0, 2.0])
    np.testing.assert_allclose(sampled.behavior_entropy, np.log(5.0))
    assert int(valid_sd_sac_replay_count(replay)) == 1
    valid_batch, indices = sample_valid_sd_sac_replay(
        jax.random.PRNGKey(9), replay, 32
    )
    assert set(np.asarray(indices).tolist()) == {2}
    assert bool(jnp.all(valid_batch.valid))


def test_sd_sac_initialization_and_update_are_finite_and_count_replay_correctly():
    batch = _batch()
    actor, critic_1, critic_2, state = init_sd_sac(
        jax.random.PRNGKey(4),
        batch.map_channels,
        batch.numeric_features,
        batch.goal_one_hot,
        num_actions=5,
        hidden_size=8,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        temperature_learning_rate=1e-3,
        initial_alpha=0.2,
        max_grad_norm=1.0,
    )
    updated, metrics = update_sd_sac(
        actor,
        critic_1,
        critic_2,
        state,
        batch,
        gamma=0.99,
        beta=0.1,
        clip_range=0.5,
        tau=0.01,
        target_entropy=0.8 * np.log(5.0),
    )
    assert int(updated.environment_steps) == 0
    assert int(updated.update_steps) == 1
    assert int(updated.gradient_steps) == 4
    counted = record_sd_sac_environment_steps(updated, 4)
    assert int(counted.environment_steps) == 4
    assert _all_finite(updated)
    assert _all_finite(metrics)
