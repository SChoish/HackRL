import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hackrl.dual_leo import init_dual_leo_teacher, make_dual_leo_update
from hackrl.spatial_wall_pass import (
    MODEL_FEATURE_NAMES,
    SpatialWallPassAction,
    SpatialWallPassVariant,
    make_spatial_wall_pass_state,
    spatial_wall_pass_step,
)
from hackrl import spatial_wall_pass_gc as gc


def _small_config(**updates):
    values = {
        "num_envs": 8,
        "num_steps": 4,
        "num_updates": 1,
        "minibatch_size": 8,
        "mode_repeats_per_state": 1,
        "sample_repeats_per_state": 2,
    }
    values.update(updates)
    return gc.SpatialWallPassGCConfig(**values)


def _repeat_state(state, count):
    return jax.tree.map(lambda value: jnp.repeat(value[None], count, axis=0), state)


def test_config_freezes_normal_goal_contract_and_separate_widths():
    config = _small_config()
    config.validate()
    assert config.goal_mode == "normal12"
    assert config.policy_hidden_size == 512
    assert config.teacher_hidden_size == 512
    assert gc.DELIVERY_GOAL_INDEX == 11
    assert gc.NUM_ACTIONS == 23
    assert gc.NUM_GOALS == 12
    with pytest.raises(ValueError):
        _small_config(goal_mode="exploit").validate()
    with pytest.raises(ValueError):
        _small_config(policy_hidden_size=0).validate()


def test_initialization_exposes_all_normal_heads_and_observation_features():
    network, runner = gc.initialize_spatial_wall_pass_gc(_small_config())
    assert bool(jnp.all(runner.seen_goals))
    inputs = gc._batch_inputs(runner.env_state, runner.current_goal)
    assert inputs[0].shape == (8, 7, 9, 24)
    assert inputs[1].shape == (8, len(MODEL_FEATURE_NAMES))
    assert inputs[2].shape == (8, 12)
    assert gc.spatial_wall_pass_gc_parameter_count(runner.train_state.params) > 0
    assert network.action_dim == 23


def test_fixed_training_path_contains_a_useful_clear_dash():
    config = _small_config()
    _, runner = gc.initialize_spatial_wall_pass_gc(config)
    state = make_spatial_wall_pass_state(0)
    for action in (SpatialWallPassAction.UP, SpatialWallPassAction.DASH_UP):
        state = spatial_wall_pass_step(state, action, SpatialWallPassVariant.FIXED)
    runner = runner.replace(env_state=_repeat_state(state, config.num_envs))
    before = runner.env_state.player_position
    runner, event = gc.step_spatial_wall_pass_gc_workers(
        runner,
        jnp.full((config.num_envs,), SpatialWallPassAction.DASH_RIGHT),
        config,
    )
    np.testing.assert_array_equal(
        np.asarray(runner.env_state.player_position - before),
        np.tile(np.asarray((0, 2), dtype=np.int32), (config.num_envs, 1)),
    )
    assert bool(jnp.all(event.dash_requested))
    assert bool(jnp.all(event.dash_succeeded))
    assert bool(jnp.all(event.clear_dash))
    assert not bool(jnp.any(event.wall_pass))


def test_wall_pass_trigger_and_benefit_are_separate_training_metrics():
    fixed_config = _small_config(variant=SpatialWallPassVariant.FIXED.value)
    mutant_config = _small_config(variant=SpatialWallPassVariant.MUTANT.value)
    _, fixed_runner = gc.initialize_spatial_wall_pass_gc(fixed_config)
    _, mutant_runner = gc.initialize_spatial_wall_pass_gc(mutant_config)
    state = make_spatial_wall_pass_state(0)
    state = spatial_wall_pass_step(
        state, SpatialWallPassAction.RIGHT, SpatialWallPassVariant.FIXED
    )
    states = _repeat_state(state, fixed_config.num_envs)
    fixed_runner = fixed_runner.replace(env_state=states)
    mutant_runner = mutant_runner.replace(env_state=states)
    actions = jnp.full(
        (fixed_config.num_envs,), SpatialWallPassAction.DASH_RIGHT
    )
    fixed_runner, fixed_event = gc.step_spatial_wall_pass_gc_workers(
        fixed_runner, actions, fixed_config
    )
    mutant_runner, mutant_event = gc.step_spatial_wall_pass_gc_workers(
        mutant_runner, actions, mutant_config
    )
    assert not bool(jnp.any(fixed_event.wall_pass))
    assert bool(jnp.all(mutant_event.wall_pass))
    assert not bool(jnp.any(mutant_event.beneficial_delivery))
    np.testing.assert_array_equal(
        np.asarray(mutant_runner.env_state.player_position[:, 1]),
        np.full((fixed_config.num_envs,), 10),
    )


def test_gc_ppo_update_and_checkpoint_round_trip(tmp_path):
    config = _small_config()
    network, runner = gc.initialize_spatial_wall_pass_gc(config)
    update = jax.jit(gc.make_spatial_wall_pass_gc_update(network, config))
    updated, metrics = update(runner)
    assert int(updated.global_update) == 1
    assert int(updated.env_steps) == config.batch_size
    assert int(updated.train_state.step) == config.num_minibatches
    assert int(metrics["valid_transitions"]) == config.batch_size
    destination = gc.save_spatial_wall_pass_gc_checkpoint(
        tmp_path / "checkpoint", updated, config
    )
    restored = gc.load_spatial_wall_pass_gc_checkpoint(
        destination, runner, config
    )
    assert jax.tree_util.tree_all(
        jax.tree.map(jnp.array_equal, updated, restored)
    )


def test_dual_update_uses_the_same_spatial_collector():
    config = _small_config()
    network, runner = gc.initialize_spatial_wall_pass_gc(config)
    inputs = gc._batch_inputs(runner.env_state, runner.current_goal)
    teacher, teacher_state, teacher_minibatch = init_dual_leo_teacher(
        config, inputs[0], inputs[1], gc.NUM_GOALS, gc.NUM_ACTIONS
    )
    update = jax.jit(
        make_dual_leo_update(
            network,
            teacher,
            config,
            gc.spatial_wall_pass_outcome,
            gc._batch_inputs,
            teacher_minibatch,
        )
    )
    runner, teacher_state, metrics = update(runner, teacher_state)
    assert int(runner.global_update) == 1
    assert int(runner.env_steps) == config.batch_size
    assert int(runner.train_state.step) == config.num_minibatches
    assert int(teacher_state.step) > 0
    assert float(metrics["bc_policy_coef"]) > 0


def test_frozen_evaluator_reports_trigger_and_payoff_separately():
    config = _small_config()
    network, runner = gc.initialize_spatial_wall_pass_gc(config)
    fixed = gc.evaluate_spatial_wall_pass_gc_frozen(
        network,
        runner.train_state.params,
        variant="fixed",
        stochastic=False,
        repeats_per_state=1,
        seed_base=60000,
        learner_seed=config.seed,
    )
    mutant = gc.evaluate_spatial_wall_pass_gc_frozen(
        network,
        runner.train_state.params,
        variant="mutant",
        stochastic=False,
        repeats_per_state=1,
        seed_base=60000,
        learner_seed=config.seed,
    )
    for result in (fixed, mutant):
        assert result["episodes"] == 16
        assert 0.0 <= result["success_rate"] <= 1.0
        assert 0.0 <= result["dash_use_rate"] <= 1.0
        assert 0.0 <= result["clear_dash_rate"] <= 1.0
        assert 0.0 <= result["wall_pass_rate"] <= 1.0
        assert 0.0 <= result["beneficial_use_rate"] <= 1.0
        assert result["action_selection"] == "mode"
