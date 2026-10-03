"""Dual LEO checks that have to pass before the comparison queue starts."""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from hackrl.dual_leo import (
    BC_HORIZON_UPDATES,
    BC_POLICY_COEF,
    BC_VALUE_COEF,
    bc_policy_coefficient,
    dual_leo_q_targets,
    init_dual_leo_teacher,
    load_dual_checkpoint,
    make_dual_leo_update,
    save_dual_checkpoint,
)
from hackrl.pack_restore_gc import (
    evaluate_pack_restore_gc_frozen,
    NUM_ACTIONS as PACK_ACTIONS,
    NUM_GOALS as PACK_GOALS,
    DELIVER_3_GOAL_INDEX as PACK_DELIVER,
    PackRestoreGCConfig,
    _batch_inputs as pack_inputs,
    initialize_pack_restore_gc,
    step_pack_restore_gc_workers,
)
from hackrl.pack_restore import pack_restore_goal_vector
from hackrl.tick_claim import observe_tick_claim, tick_claim_goal_vector
from hackrl.tick_claim_gc import (
    evaluate_tick_claim_gc_frozen,
    NUM_ACTIONS as TICK_ACTIONS,
    NUM_GOALS as TICK_GOALS,
    DELIVER_3_GOAL_INDEX as TICK_DELIVER,
    TickClaimGCConfig,
    _batch_inputs as tick_inputs,
    initialize_tick_claim_gc,
    step_tick_claim_gc_workers,
)


def test_per_goal_bootstrap_stops_only_for_that_goal_or_a_real_reset():
    achieved = jnp.asarray([[[True, False], [False, True]]])
    nxt = jnp.asarray([[[0.4, 0.8], [0.2, 0.6]]])
    open_world = jnp.asarray([[False, False]])
    targets = dual_leo_q_targets(achieved, nxt, open_world, 0.995)
    np.testing.assert_allclose(targets[0, 0], [1.0, 0.995 * 0.8])
    np.testing.assert_allclose(targets[0, 1], [0.995 * 0.2, 1.0])

    closed = dual_leo_q_targets(achieved, nxt, jnp.asarray([[True, False]]), 0.995)
    np.testing.assert_allclose(closed[0, 0], [1.0, 0.0])
    np.testing.assert_allclose(closed[0, 1], [0.995 * 0.2, 1.0])


def test_bc_schedule_is_locked_across_the_full_horizon():
    assert BC_HORIZON_UPDATES == 4608
    assert BC_VALUE_COEF == 0.0
    np.testing.assert_allclose(float(bc_policy_coefficient(0)), BC_POLICY_COEF)
    np.testing.assert_allclose(float(bc_policy_coefficient(512)), 0.1 * (1.0 - 512 / 4608))
    np.testing.assert_allclose(float(bc_policy_coefficient(4608)), 0.0)
    assert float(bc_policy_coefficient(4609)) == 0.0


def _terminal_reset(kind):
    if kind == "tick":
        config = TickClaimGCConfig(
            seed=20, num_envs=2, num_steps=4, num_updates=1, minibatch_size=8, hidden_size=32
        )
        _, runner = initialize_tick_claim_gc(config)
        runner = runner.replace(
            env_state=runner.env_state.replace(
                tick=jnp.full((2,), 127, dtype=jnp.int32),
                delivered_total=jnp.full((2,), 3, dtype=jnp.int32),
                grain=jnp.zeros((2,), dtype=jnp.int32),
            ),
            current_goal=jnp.full((2,), 2, dtype=jnp.int32),
        )
        stepped, event = step_tick_claim_gc_workers(
            runner, jnp.zeros((2,), dtype=jnp.int32), config
        )
        reset_goals = jax.vmap(
            lambda state: tick_claim_goal_vector(observe_tick_claim(state))
        )(stepped.env_state)
        return event, stepped.env_state.delivered_total, stepped.env_state.tick, reset_goals
    config = PackRestoreGCConfig(
        seed=20, num_envs=2, num_steps=4, num_updates=1, minibatch_size=8, hidden_size=32
    )
    _, runner = initialize_pack_restore_gc(config)
    runner = runner.replace(
        env_state=runner.env_state.replace(
            tick=jnp.full((2,), 127, dtype=jnp.int32),
            delivered_total=jnp.full((2,), 3, dtype=jnp.int32),
            carried_grain=jnp.zeros((2,), dtype=jnp.int32),
        ),
        current_goal=jnp.full((2,), 2, dtype=jnp.int32),
    )
    stepped, event = step_pack_restore_gc_workers(
        runner, jnp.zeros((2,), dtype=jnp.int32), config
    )
    reset_goals = jax.vmap(pack_restore_goal_vector)(stepped.env_state)
    return event, stepped.env_state.delivered_total, stepped.env_state.tick, reset_goals


def test_terminal_predicates_survive_the_world_reset():
    for kind, deliver in (("tick", TICK_DELIVER), ("pack", PACK_DELIVER)):
        event, delivered, tick, reset_goals = _terminal_reset(kind)
        assert bool(jnp.all(event.world_done))
        assert bool(jnp.all(event.terminal_goals[:, deliver]))
        assert bool(jnp.all(delivered == 0))
        assert bool(jnp.all(tick == 0))
        assert bool(jnp.all(jnp.logical_not(reset_goals[:, deliver])))
        command = event.terminal_goals[:, 2]
        np.testing.assert_array_equal(np.asarray(event.reward > 0), np.asarray(command))
        assert bool(jnp.all(jnp.sum(event.terminal_goals, axis=-1) > event.reward))


def _tick_outcome(runner, actions, config):
    runner, event = step_tick_claim_gc_workers(runner, actions, config)
    return (
        runner,
        event.done_for_gae,
        event.valid_transition,
        event.reward,
        event.terminal_goals,
        event.world_done,
        event.goal_done,
        event.observed_goals,
    )


def _pack_outcome(runner, actions, config):
    runner, event = step_pack_restore_gc_workers(runner, actions, config)
    return (
        runner,
        event.done,
        event.valid,
        event.reward,
        event.terminal_goals,
        event.world_done,
        event.goal_done,
        event.observed_goals,
    )


def _trees_equal(left, right):
    compared = jax.tree.map(lambda a, b: bool(jnp.array_equal(a, b)), left, right)
    return all(jax.tree.leaves(compared))


def _round_trip(kind, directory):
    if kind == "tick":
        config = TickClaimGCConfig(
            seed=21,
            num_envs=2,
            num_steps=4,
            num_updates=2,
            update_epochs=1,
            minibatch_size=8,
            hidden_size=32,
            goal_mode="workshop12",
        )
        network, runner = initialize_tick_claim_gc(config)
        inputs = tick_inputs(runner.env_state, runner.current_goal)
        outcome, goals, actions = _tick_outcome, TICK_GOALS, TICK_ACTIONS
    else:
        config = PackRestoreGCConfig(
            seed=21,
            num_envs=2,
            num_steps=4,
            num_updates=2,
            update_epochs=1,
            minibatch_size=8,
            hidden_size=32,
            goal_mode="workshop12",
        )
        network, runner = initialize_pack_restore_gc(config)
        inputs = pack_inputs(runner.env_state, runner.current_goal)
        outcome, goals, actions = _pack_outcome, PACK_GOALS, PACK_ACTIONS
    teacher, leo, minibatch = init_dual_leo_teacher(
        config, inputs[0], inputs[1], goals, actions
    )
    fresh_network, fresh_runner = (
        initialize_tick_claim_gc(config) if kind == "tick" else initialize_pack_restore_gc(config)
    )
    del fresh_network
    assert _trees_equal(runner.train_state.params, fresh_runner.train_state.params)
    update = jax.jit(
        make_dual_leo_update(network, teacher, config, outcome, tick_inputs if kind == "tick" else pack_inputs, minibatch)
    )
    updated_runner, updated_leo, metrics = update(runner, leo)
    assert int(updated_runner.global_update) == 1
    assert int(updated_leo.step) == 2
    assert float(metrics["bc_value_coef"]) == 0.0
    np.testing.assert_allclose(float(metrics["bc_policy_coef"]), 0.1)
    assert int(metrics["teacher_grad_steps"]) == 2
    assert int(metrics["teacher_applied_minibatches"]) == int(metrics["teacher_scheduled_minibatches"])
    assert int(metrics["ppo_valid_transitions"]) == int(metrics["valid_transitions"])
    assert int(metrics["teacher_valid_samples"]) > 0
    initial_stats = leo.batch_stats
    assert not _trees_equal(updated_leo.batch_stats, initial_stats)
    save_dual_checkpoint(directory, updated_runner, updated_leo, {"seed": config.seed})
    restored_runner, restored_leo = load_dual_checkpoint(directory, updated_runner, updated_leo)
    assert _trees_equal(restored_runner, updated_runner)
    assert _trees_equal(restored_leo, updated_leo)
    continued_runner, continued_leo, _ = update(updated_runner, updated_leo)
    reloaded_runner, reloaded_leo, _ = update(restored_runner, restored_leo)
    assert _trees_equal(continued_runner, reloaded_runner)
    assert _trees_equal(continued_leo, reloaded_leo)
    evaluate = evaluate_tick_claim_gc_frozen if kind == "tick" else evaluate_pack_restore_gc_frozen
    before = evaluate(
        network,
        updated_runner.train_state.params,
        variant="fixed",
        stochastic=False,
        repeats_per_state=1,
        seed_base=20000,
        learner_seed=int(config.seed),
        record_episodes=True,
    )
    after = evaluate(
        network,
        restored_runner.train_state.params,
        variant="fixed",
        stochastic=False,
        repeats_per_state=1,
        seed_base=20000,
        learner_seed=int(config.seed),
        record_episodes=True,
    )
    assert before["episode_records"] == after["episode_records"]


def test_checkpoint_restores_policy_teacher_optimizer_and_schedule_position():
    with tempfile.TemporaryDirectory() as directory:
        _round_trip("tick", Path(directory) / "tick")
        _round_trip("pack", Path(directory) / "pack")


def test_split_arms_freeze_stats_and_keep_the_teacher_shuffle():
    config = TickClaimGCConfig(
        seed=21,
        num_envs=2,
        num_steps=4,
        num_updates=1,
        update_epochs=1,
        minibatch_size=8,
        hidden_size=32,
        goal_mode="workshop12",
    )
    network, runner = initialize_tick_claim_gc(config)
    inputs = tick_inputs(runner.env_state, runner.current_goal)
    teacher, leo, minibatch = init_dual_leo_teacher(
        config, inputs[0], inputs[1], TICK_GOALS, TICK_ACTIONS
    )
    arms = {}
    for learn, imitate in ((True, True), (True, False), (False, True)):
        update = jax.jit(
            make_dual_leo_update(
                network,
                teacher,
                config,
                _tick_outcome,
                tick_inputs,
                minibatch,
                learn_teacher=learn,
                imitate_teacher=imitate,
            )
        )
        arms[(learn, imitate)] = update(runner, leo)
    full_runner, full_leo, full_metrics = arms[(True, True)]
    quiet_runner, quiet_leo, quiet_metrics = arms[(True, False)]
    frozen_runner, frozen_leo, frozen_metrics = arms[(False, True)]
    assert _trees_equal(full_runner.rng, quiet_runner.rng)
    assert _trees_equal(full_runner.rng, frozen_runner.rng)
    assert not _trees_equal(full_runner.rng, runner.rng)
    assert _trees_equal(frozen_leo.params, leo.params)
    assert _trees_equal(frozen_leo.batch_stats, leo.batch_stats)
    assert _trees_equal(frozen_leo.opt_state, leo.opt_state)
    assert int(frozen_leo.step) == 0
    assert int(frozen_metrics["teacher_applied_minibatches"]) == 0
    assert int(frozen_metrics["teacher_scheduled_minibatches"]) > 0
    assert not _trees_equal(quiet_leo.batch_stats, leo.batch_stats)
    assert int(quiet_leo.step) == 2
    assert _trees_equal(frozen_runner.train_state.params, full_runner.train_state.params)
    assert not _trees_equal(quiet_runner.train_state.params, full_runner.train_state.params)
    np.testing.assert_allclose(float(quiet_metrics["bc_policy_coef"]), 0.0)
    np.testing.assert_allclose(float(frozen_metrics["bc_policy_coef"]), 0.1)
    np.testing.assert_allclose(float(full_metrics["bc_policy_coef"]), 0.1)


def test_adapt_bc_off_branches_from_full_dual_pretraining():
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from run_dual_adapt_bc_off import ORIGIN_ARM, build_jobs
    from run_dual_leo_compare import _check_arm

    jobs = build_jobs()
    assert len(jobs) == 20
    assert {job["env"] for job in jobs} == {"tick", "pack"}
    assert {job["variant"] for job in jobs} == {"fixed", "mutant"}
    assert {job["seed"] for job in jobs} == {20, 21, 22, 23, 24}
    for job in jobs:
        assert job["learn_teacher"] is True
        assert job["imitate_teacher"] is False
        assert job["origin_arm"] == ORIGIN_ARM
        source = Path(job["pretrain_checkpoint"])
        assert (source / "state.msgpack").is_file()
        _check_arm(source, job["origin_arm"])
        try:
            _check_arm(source, {"learn_teacher": True, "imitate_teacher": False})
        except RuntimeError:
            pass
        else:
            raise AssertionError("a full Dual pretrain must not be treated as the BC-off branch")


def test_teacher_imitation_queue_adds_sixty_jobs_from_pretraining():
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from run_teacher_imitation_split import build_jobs

    jobs = build_jobs()
    assert len(jobs) == 60
    assert sum(job["kind"] == "pretrain" for job in jobs) == 20
    assert sum(job["kind"] == "adapt" for job in jobs) == 40
    assert {job["method"] for job in jobs} == {"teacher_only", "frozen_imitation"}
    assert all(job["learn_teacher"] and not job["imitate_teacher"] for job in jobs if job["method"] == "teacher_only")
    assert all(not job["learn_teacher"] and job["imitate_teacher"] for job in jobs if job["method"] == "frozen_imitation")
    assert all(job["kind"] == "pretrain" or "pretrain" in job["depends_on"][0] for job in jobs)
