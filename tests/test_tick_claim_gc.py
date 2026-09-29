from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from hackrl.tick_claim import (
    GOAL_IDS,
    TickClaimAction,
    TickClaimPhase,
    TickClaimStart,
    TickClaimVariant,
    make_tick_claim_state,
    observe_tick_claim,
)
from hackrl.tick_claim_gc import (
    MAP_FEATURE_COUNT,
    NUM_ACTIONS,
    NUM_GOALS,
    TickClaimGCConfig,
    calculate_tick_claim_gc_gae,
    evaluate_tick_claim_gc_frozen,
    initialize_tick_claim_gc,
    make_tick_claim_gc_update,
    sample_false_seen_goal,
    step_tick_claim_gc_workers,
    tick_claim_gc_inputs,
    tick_claim_gc_parameter_count,
    validate_tick_claim_gc_checkpoint_resume,
)


def _batched(tree, count):
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (count,) + value.shape), tree
    )


def _runner_with_state(config, state):
    network, runner = initialize_tick_claim_gc(config)
    return network, runner.replace(env_state=_batched(state, config.num_envs))


def test_protocol_network_and_initial_worker_state_are_resolved():
    fixed_network, fixed = initialize_tick_claim_gc(TickClaimGCConfig())
    mutant_network, mutant = initialize_tick_claim_gc(
        TickClaimGCConfig(variant=TickClaimVariant.MUTANT.value)
    )
    assert tick_claim_gc_parameter_count(fixed.train_state.params) == 2_640_181
    assert tick_claim_gc_parameter_count(mutant.train_state.params) == 2_640_181
    assert jax.tree_util.tree_all(
        jax.tree.map(
            jnp.array_equal,
            fixed.train_state.params,
            mutant.train_state.params,
        )
    )
    assert jax.tree_util.tree_all(
        jax.tree.map(jnp.array_equal, fixed.env_state, mutant.env_state)
    )
    assert int(jnp.sum(fixed.seen_goals)) == 3
    observations = jax.vmap(
        lambda state: tick_claim_gc_inputs(
            observe_tick_claim(state),
            jnp.asarray(GOAL_IDS.index("delivery/count_ge_3")),
        )
    )(fixed.env_state)
    assert observations[0].shape == (512, 7, 9, MAP_FEATURE_COUNT // (7 * 9))
    assert observations[1].shape == (512, 15)
    assert observations[2].shape == (512, NUM_GOALS)
    policy, value = fixed_network.apply(
        fixed.train_state.params,
        observations[0][:2],
        observations[1][:2],
        observations[2][:2],
    )
    assert policy.logits.shape == (2, NUM_ACTIONS)
    assert value.shape == (2,)
    assert bool(jnp.all(jnp.logical_and(value >= 0, value <= 1)))
    del mutant_network


def test_false_seen_sampler_rejects_already_satisfied_goals():
    seen = jnp.asarray([True, True, True] + [False] * (NUM_GOALS - 3))
    achieved = jnp.asarray([True, False, True] + [False] * (NUM_GOALS - 3))
    goal, _, zero_steps, valid = sample_false_seen_goal(
        jax.random.PRNGKey(7), seen, achieved
    )
    assert bool(valid)
    assert int(goal) == 1
    assert int(zero_steps) >= 0


def test_goal_done_cuts_value_flow_without_resetting_world():
    config = TickClaimGCConfig(
        num_envs=2,
        num_steps=2,
        num_updates=1,
        minibatch_size=2,
        hidden_size=16,
    )
    state = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    state = state.replace(
        player_position=state.delivery_position + jnp.asarray((-1, 0)),
        grain=jnp.asarray(3, dtype=jnp.int32),
        tick=jnp.asarray(10, dtype=jnp.int32),
    )
    _, runner = _runner_with_state(config, state)
    stepped, event = step_tick_claim_gc_workers(
        runner,
        jnp.full((2,), TickClaimAction.DELIVER, dtype=jnp.int32),
        config,
    )
    assert bool(jnp.all(event.goal_done))
    assert not bool(jnp.any(event.world_done))
    assert bool(jnp.all(event.done_for_gae))
    assert bool(jnp.all(event.reward == 1))
    assert bool(jnp.all(stepped.env_state.tick == 11))
    assert bool(jnp.all(stepped.env_state.delivered_total == 3))
    assert not bool(jnp.any(stepped.command_active))


def test_world_done_is_the_only_reset_and_both_done_signals_are_retained():
    config = TickClaimGCConfig(
        num_envs=2,
        num_steps=2,
        num_updates=1,
        minibatch_size=2,
        hidden_size=16,
    )
    state = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    ).replace(tick=jnp.asarray(127, dtype=jnp.int32))
    _, runner = _runner_with_state(config, state)
    reset, event = step_tick_claim_gc_workers(
        runner,
        jnp.full((2,), TickClaimAction.NOOP, dtype=jnp.int32),
        config,
    )
    assert not bool(jnp.any(event.goal_done))
    assert bool(jnp.all(event.world_done))
    assert bool(jnp.all(event.reset_count == 1))
    assert bool(jnp.all(reset.env_state.tick == 0))

    delivery = state.replace(
        player_position=state.delivery_position + jnp.asarray((-1, 0)),
        grain=jnp.asarray(3, dtype=jnp.int32),
    )
    _, runner = _runner_with_state(config, delivery)
    reset, event = step_tick_claim_gc_workers(
        runner,
        jnp.full((2,), TickClaimAction.DELIVER, dtype=jnp.int32),
        config,
    )
    assert bool(jnp.all(event.goal_done))
    assert bool(jnp.all(event.world_done))
    assert bool(jnp.all(event.done_for_gae))
    assert bool(jnp.all(event.reward == 1))
    assert bool(jnp.all(reset.env_state.tick == 0))


def test_workshop_goal_switch_preserves_world_state():
    config = TickClaimGCConfig(
        goal_mode="workshop12",
        num_envs=2,
        num_steps=2,
        num_updates=1,
        minibatch_size=2,
        hidden_size=16,
    )
    state = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    _, runner = _runner_with_state(config, state)
    runner = runner.replace(
        current_goal=jnp.zeros((2,), dtype=jnp.int32),
        command_active=jnp.ones((2,), dtype=jnp.bool_),
    )
    stepped, event = step_tick_claim_gc_workers(
        runner,
        jnp.full((2,), TickClaimAction.DO, dtype=jnp.int32),
        config,
    )
    assert bool(jnp.all(event.goal_done))
    assert not bool(jnp.any(event.world_done))
    assert bool(jnp.all(event.command_started))
    assert bool(jnp.all(stepped.env_state.tick == 1))
    assert bool(jnp.all(stepped.env_state.grain == 1))
    assert bool(jnp.all(stepped.current_goal != 0))


def test_violation_and_delivery_metrics_count_physical_events_only():
    config = TickClaimGCConfig(
        variant=TickClaimVariant.MUTANT.value,
        num_envs=2,
        num_steps=2,
        num_updates=1,
        minibatch_size=2,
        hidden_size=16,
    )
    state = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    _, runner = _runner_with_state(config, state)

    def act(current, action):
        return step_tick_claim_gc_workers(
            current,
            jnp.full((2,), action, dtype=jnp.int32),
            config,
        )

    runner, armed = act(runner, TickClaimAction.ARM_HARVEST)
    runner, waiting = act(runner, TickClaimAction.NOOP)
    runner, duplicated = act(runner, TickClaimAction.DO)
    runner, after = act(runner, TickClaimAction.NOOP)
    runner, moved = act(runner, TickClaimAction.DOWN)
    runner, delivered = act(runner, TickClaimAction.DELIVER)

    assert not bool(jnp.any(armed.violation))
    assert not bool(jnp.any(waiting.violation))
    assert bool(jnp.all(duplicated.violation))
    assert not bool(jnp.any(duplicated.repeated_violation))
    assert not bool(jnp.any(after.violation))
    assert not bool(jnp.any(moved.violation))
    assert bool(jnp.all(delivered.violation_delivery))
    assert bool(jnp.all(delivered.violation_grain_delivered == 1))


class _MinimalTrajectory(NamedTuple):
    done: jax.Array
    valid: jax.Array
    reward: jax.Array
    value: jax.Array


def test_gae_cuts_at_goal_or_world_done_and_bootstraps_rollout_cut():
    trajectory = _MinimalTrajectory(
        done=jnp.asarray([[True], [False]]),
        valid=jnp.asarray([[True], [True]]),
        reward=jnp.asarray([[1.0], [100.0]]),
        value=jnp.zeros((2, 1)),
    )
    advantages, _ = calculate_tick_claim_gc_gae(
        trajectory, jnp.zeros((1,)), gamma=1.0, gae_lambda=1.0
    )
    np.testing.assert_allclose(np.asarray(advantages[:, 0]), [1.0, 100.0])

    rollout_cut = trajectory._replace(
        done=jnp.asarray([[False], [False]]),
        reward=jnp.zeros((2, 1)),
    )
    advantages, _ = calculate_tick_claim_gc_gae(
        rollout_cut, jnp.asarray([2.0]), gamma=0.5, gae_lambda=1.0
    )
    np.testing.assert_allclose(np.asarray(advantages[:, 0]), [0.5, 1.0])


def test_checkpoint_round_trip_preserves_the_identical_next_update(tmp_path):
    config = TickClaimGCConfig(
        num_envs=4,
        num_steps=2,
        num_updates=1,
        minibatch_size=4,
        hidden_size=16,
    )
    checks = validate_tick_claim_gc_checkpoint_resume(tmp_path, config)
    assert all(checks.values()), checks


def test_frozen_evaluation_does_not_mutate_learner_state():
    config = TickClaimGCConfig(
        num_envs=4,
        num_steps=2,
        num_updates=1,
        minibatch_size=4,
        hidden_size=16,
    )
    network, runner = initialize_tick_claim_gc(config)
    update = jax.jit(make_tick_claim_gc_update(network, config))
    runner, _ = update(runner)
    before = serialization.to_bytes(runner)
    result = evaluate_tick_claim_gc_frozen(
        network,
        runner.train_state.params,
        variant=TickClaimVariant.FIXED,
        stochastic=False,
        repeats_per_state=1,
        seed_base=20_000,
        learner_seed=0,
    )
    assert result["overall"]["episodes"] == 64
    assert serialization.to_bytes(runner) == before
