import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from hackrl.pack_restore import (
    PackRestoreAction,
    PackRestorePhase,
    PackRestoreStart,
    PackRestoreVariant,
    make_pack_restore_state,
    pack_restore_step,
)
from hackrl.pack_restore_gc import (
    PackRestoreGCConfig,
    evaluate_pack_restore_gc_frozen,
    initialize_pack_restore_gc,
    step_pack_restore_gc_workers,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = _load_script(
    "run_pack_inventory_overwrite_ppo",
    "scripts/run_pack_inventory_overwrite_ppo.py",
)


def _metric(value):
    return jnp.asarray(value)


def test_manifest_locks_exact_three_condition_budget():
    manifest = json.loads(
        (
            REPOSITORY
            / "docs/manifests/pack_inventory_overwrite_ppo_v1.json"
        ).read_text()
    )
    assert manifest["seeds"] == [30, 31, 32, 33, 34]
    assert set(manifest["conditions"]) == {
        "frozen_harmful",
        "normal_continue",
        "harmful_adapt",
    }
    optimization = manifest["optimization"]
    assert optimization["num_envs"] * optimization["num_steps"] == 32768
    assert optimization["updates"] * 32768 == 1048576
    assert manifest["budget"]["total_collector_slots"] == 15 * 1048576
    assert manifest["evaluation"]["adaptation_updates"] == [0, 2, 8, 32]
    assert manifest["budget"]["automatic_extension"] is False
    assert manifest["conditions"]["frozen_harmful"]["ppo_updates"] == 0
    assert all(
        condition["teacher_updates"] == condition["bc_updates"] == 0
        for condition in manifest["conditions"].values()
    )


def test_job_matrix_has_five_paired_seeds_and_ten_learning_jobs():
    jobs = RUNNER.build_jobs()
    assert len(jobs) == 15
    assert sum(job["learning"] for job in jobs) == 10
    for seed in range(30, 35):
        assert [
            job["condition"] for job in jobs if job["seed"] == seed
        ] == list(RUNNER.CONDITIONS)


def test_episode_loss_tracker_spans_two_64_step_blocks():
    tracker = RUNNER._empty_tracker(2)
    first = {
        "loss_worker_mask": _metric([True, False]),
        "world_done_worker_mask": _metric([False, False]),
        "transitions": _metric(8),
        "valid_transitions": _metric(7),
        "empty_minibatches": _metric(0),
        "world_resets": _metric(0),
        "lost_grain": _metric(2),
        "physical_loss_events": _metric(1),
        "overwrite_trigger_events": _metric(1),
        "goal_successes": _metric(0),
        "harvest_events": _metric(1),
        "storage_events": _metric(0),
        "delivery_events": _metric(0),
    }
    tracker = RUNNER.update_exposure_tracker(
        tracker, first, learning=True, optimizer_applications=1
    )
    assert tracker["completed_world_episodes"] == 0
    second = dict(first)
    second.update(
        loss_worker_mask=_metric([False, False]),
        world_done_worker_mask=_metric([True, True]),
        world_resets=_metric(2),
        lost_grain=_metric(0),
        physical_loss_events=_metric(0),
        overwrite_trigger_events=_metric(0),
    )
    tracker = RUNNER.update_exposure_tracker(
        tracker, second, learning=True, optimizer_applications=1
    )
    assert tracker["completed_world_episodes"] == 2
    assert tracker["loss_episodes"] == 1
    assert tracker["loss_episode_rate"] == 0.5
    assert tracker["cumulative_lost_grain"] == 2
    assert tracker["valid_ppo_samples"] == 14
    assert tracker["optimizer_applications"] == 2


def _harvest_ready_state(carried=2):
    state = make_pack_restore_state(
        0,
        int(PackRestorePhase.EMPTY),
        start=PackRestoreStart.PATH_CHECK,
        source_growth_period=8,
    )
    state = pack_restore_step(
        state, int(PackRestoreAction.RIGHT), PackRestoreVariant.FIXED
    )
    return state.replace(
        carried_grain=jnp.asarray(carried, dtype=jnp.int32),
        source_grain=jnp.asarray(2, dtype=jnp.int32),
        source_age=jnp.asarray(8, dtype=jnp.int32),
    )


def test_worker_transition_reports_exact_overwrite_loss():
    config = PackRestoreGCConfig(
        variant=PackRestoreVariant.INVENTORY_OVERWRITE.value,
        num_envs=2,
        num_steps=4,
        minibatch_size=8,
        policy_hidden_size=32,
        goal_mode="deliver_3",
    )
    _, runner = initialize_pack_restore_gc(config)
    state = _harvest_ready_state(carried=2)
    states = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), state)
    runner = runner.replace(
        env_state=states,
        command_active=jnp.ones((2,), dtype=jnp.bool_),
    )
    actions = jnp.full((2,), int(PackRestoreAction.DO), dtype=jnp.int32)
    _, event = step_pack_restore_gc_workers(runner, actions, config)
    assert np.asarray(event.physical_loss).tolist() == [True, True]
    assert np.asarray(event.lost_grain).tolist() == [2, 2]
    assert np.asarray(event.overwrite_trigger).tolist() == [True, True]
    assert np.asarray(event.harvest_event).tolist() == [True, True]


def test_frozen_collection_does_not_change_train_state():
    config = PackRestoreGCConfig(
        variant=PackRestoreVariant.INVENTORY_OVERWRITE.value,
        seed=7,
        num_envs=2,
        num_steps=4,
        minibatch_size=8,
        num_updates=1,
        policy_hidden_size=32,
        goal_mode="deliver_3",
    )
    network, runner = initialize_pack_restore_gc(config)
    before = RUNNER._train_state_sha256(runner.train_state)
    collect = jax.jit(RUNNER.make_frozen_collection(network, config))
    after_runner, metrics = collect(runner)
    jax.block_until_ready(after_runner.env_steps)
    assert RUNNER._train_state_sha256(after_runner.train_state) == before
    assert int(after_runner.train_state.step) == int(runner.train_state.step)
    assert int(metrics["transitions"]) == config.batch_size


def test_evaluator_reports_strategy_order_without_changing_old_metrics():
    config = PackRestoreGCConfig(
        seed=9,
        num_envs=2,
        num_steps=4,
        minibatch_size=8,
        num_updates=1,
        policy_hidden_size=32,
        goal_mode="deliver_3",
    )
    network, runner = initialize_pack_restore_gc(config)
    result = evaluate_pack_restore_gc_frozen(
        network,
        runner.train_state.params,
        variant=PackRestoreVariant.INVENTORY_OVERWRITE,
        stochastic=False,
        repeats_per_state=1,
        seed_base=9000,
        learner_seed=9,
        record_episodes=True,
        natural_only=True,
    )
    aggregate = result["overall"]
    assert aggregate["episodes"] == 32
    for key in (
        "success_rate",
        "loss_episode_rate",
        "deliver_before_first_harvest_rate",
        "store_before_first_harvest_rate",
        "protect_before_first_harvest_rate",
        "successful_zero_loss_rate",
        "no_loss_failure_rate",
    ):
        assert key in aggregate
    assert len(result["episode_records"]) == 32
    assert "first_harvest_step" in result["episode_records"][0]
