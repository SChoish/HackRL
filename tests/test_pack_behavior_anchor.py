import jax
import jax.numpy as jnp
import numpy as np

from hackrl.pack_behavior_anchor import (
    PackBehaviorAnchor,
    behavior_anchor_kl,
    make_behavior_anchored_pack_restore_update,
)
from hackrl.pack_restore_gc import (
    PackRestoreGCConfig,
    _batch_inputs,
    categorical_source_kl,
    initialize_pack_restore_gc,
    make_pack_restore_gc_update,
)


def _assert_tree_equal(left, right):
    left_leaves, left_structure = jax.tree.flatten(jax.device_get(left))
    right_leaves, right_structure = jax.tree.flatten(jax.device_get(right))
    assert left_structure == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _fixture():
    config = PackRestoreGCConfig(
        num_envs=2,
        num_steps=2,
        minibatch_size=4,
        num_updates=1,
        update_epochs=1,
        policy_hidden_size=16,
        teacher_hidden_size=16,
        goal_mode="deliver_3",
        seed=91,
    )
    network, runner = initialize_pack_restore_gc(config)
    inputs = _batch_inputs(runner.env_state, runner.current_goal)
    policy, _ = network.apply(runner.train_state.params, *inputs)
    anchor = PackBehaviorAnchor(
        map_channels=inputs[0],
        numeric_features=inputs[1],
        goal_one_hot=inputs[2],
        source_logits=jax.lax.stop_gradient(policy.logits),
    )
    return config, network, runner, anchor


def test_full_categorical_source_kl_matches_manual_action_sum():
    source_logits = jnp.asarray([[1.2, -0.3, 0.7], [-1.0, 0.5, 0.2]])
    current_logits = jnp.asarray([[0.8, 0.1, 0.4], [-0.2, 0.9, -0.4]])
    source_log_probs = jax.nn.log_softmax(source_logits, axis=-1)
    current_log_probs = jax.nn.log_softmax(current_logits, axis=-1)
    expected = jnp.sum(
        jnp.exp(source_log_probs) * (source_log_probs - current_log_probs),
        axis=-1,
    )
    np.testing.assert_allclose(
        categorical_source_kl(source_logits, current_logits),
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


def test_kl_stops_source_gradient_and_is_stationary_at_equal_outputs():
    logits = jnp.asarray([[1.0, -0.2, 0.4], [0.1, 0.3, -0.7]])
    source_gradient = jax.grad(
        lambda source: jnp.mean(categorical_source_kl(source, logits))
    )(logits)
    current_gradient = jax.grad(
        lambda current: jnp.mean(categorical_source_kl(logits, current))
    )(logits)
    np.testing.assert_array_equal(source_gradient, jnp.zeros_like(source_gradient))
    np.testing.assert_allclose(current_gradient, 0.0, atol=1e-7)
    assert abs(float(jnp.mean(categorical_source_kl(logits, logits)))) < 1e-7


def test_kl_gradient_step_moves_current_distribution_toward_source():
    source_logits = jnp.asarray([[2.0, 0.0, -1.0]])
    current_logits = jnp.asarray([[-0.5, 0.3, 1.1]])
    objective = lambda current: jnp.mean(
        categorical_source_kl(source_logits, current)
    )
    before = objective(current_logits)
    gradient = jax.grad(objective)(current_logits)
    after = objective(current_logits - 0.5 * gradient)
    assert float(after) < float(before)


def test_soft_source_targets_are_not_argmax_behavior_cloning():
    current_a = jnp.asarray([[1.0, 0.9, -3.0]])
    current_b = jnp.asarray([[4.0, -1.0, -3.0]])
    source = jnp.asarray([[1.1, 1.0, -2.0]])
    assert int(jnp.argmax(current_a)) == int(jnp.argmax(current_b)) == int(
        jnp.argmax(source)
    )
    kl_a = float(categorical_source_kl(source, current_a)[0])
    kl_b = float(categorical_source_kl(source, current_b)[0])
    assert not np.isclose(kl_a, kl_b)


def test_beta_zero_wrapper_is_bitwise_equal_to_existing_updater():
    config, network, runner, anchor = _fixture()
    legacy_update = jax.jit(make_pack_restore_gc_update(network, config))
    zero_update = jax.jit(
        make_behavior_anchored_pack_restore_update(
            network, config, anchor, coefficient=0.0
        )
    )
    legacy_runner, legacy_metrics = legacy_update(runner)
    zero_runner, zero_metrics = zero_update(runner)
    _assert_tree_equal(legacy_runner, zero_runner)
    _assert_tree_equal(legacy_metrics, zero_metrics)


def test_positive_anchor_uses_one_optimizer_application_and_keeps_anchor_fixed():
    config, network, runner, anchor = _fixture()
    shifted_logits = anchor.source_logits + jnp.linspace(
        -1.0, 1.0, anchor.source_logits.shape[-1]
    )
    shifted_anchor = anchor.replace(source_logits=shifted_logits)
    anchor_before = jax.tree.map(lambda value: np.asarray(value).copy(), shifted_anchor)
    update = jax.jit(
        make_behavior_anchored_pack_restore_update(
            network, config, shifted_anchor, coefficient=0.1
        )
    )
    ordinary, _ = jax.jit(
        make_behavior_anchored_pack_restore_update(network, config, anchor, coefficient=0.0)
    )(runner)
    updated, metrics = update(runner)
    assert int(updated.train_state.step) - int(runner.train_state.step) == 1
    assert int(updated.train_state.step) == int(ordinary.train_state.step)
    assert int(metrics["empty_minibatches"]) == 0
    assert int(metrics["behavior_anchor_state_count"]) == shifted_anchor.state_count
    assert int(metrics["behavior_anchor_forward_state_evaluations"]) == (
        shifted_anchor.state_count
    )
    assert np.isclose(float(metrics["behavior_anchor_coefficient"]), 0.1)
    assert float(metrics["behavior_anchor_kl"]) > 0.0
    assert np.isclose(
        float(metrics["behavior_anchor_penalty"]),
        0.1 * float(metrics["behavior_anchor_kl"]),
        rtol=1e-6,
    )
    _assert_tree_equal(anchor_before, shifted_anchor)


def test_empty_minibatch_skips_anchor_and_optimizer_application():
    config, network, runner, anchor = _fixture()
    inactive = runner.replace(command_active=jnp.zeros_like(runner.command_active))
    update = jax.jit(
        make_behavior_anchored_pack_restore_update(
            network, config, anchor, coefficient=0.1
        )
    )
    updated, metrics = update(inactive)
    assert int(updated.train_state.step) == int(inactive.train_state.step)
    assert int(metrics["empty_minibatches"]) == 1
    assert int(metrics["behavior_anchor_forward_state_evaluations"]) == 0


def test_actor_has_no_mutable_normalization_collection_and_anchor_has_no_params():
    _, network, runner, anchor = _fixture()
    assert tuple(runner.train_state.params.keys()) == ("params",)
    assert len(jax.tree.leaves(anchor)) == 4
    assert np.isclose(
        float(behavior_anchor_kl(network, runner.train_state.params, anchor)),
        0.0,
        atol=1e-7,
    )
