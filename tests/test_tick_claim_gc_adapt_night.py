import json
from dataclasses import replace

import jax
import numpy as np

from hackrl.tick_claim_gc import (
    TickClaimGCConfig,
    _BRANCH_LOCKED_FIELDS,
    _FROZEN_EVAL_CACHE,
    evaluate_tick_claim_gc_frozen,
    initialize_tick_claim_gc,
    load_tick_claim_gc_branch,
    make_tick_claim_gc_update,
    save_tick_claim_gc_checkpoint,
)
from run_tick_claim_gc_adapt_night import (
    ADAPT_UPDATES,
    BATCH,
    SCIENCE_UPDATES,
    _metric_rows,
    build_jobs,
    curve_complete,
    latest_saved_update,
    science_transitions,
)


def test_job_matrix_is_seventy_adapt_cells_in_phase_order():
    jobs = build_jobs()
    adapt = [job for job in jobs if job["kind"] == "adapt"]
    assert len(adapt) == 70
    phases = [job["phase"] for job in adapt]
    assert phases == ["A"] * 30 + ["B"] * 20 + ["C"] * 20
    assert {job["entropy"] for job in adapt if job["phase"] == "B"} == {0.002, 0.01}
    assert {job["hidden"] for job in adapt if job["phase"] == "C"} == {256, 1024}
    assert science_transitions() == (
        0,
        1_048_576,
        4_194_304,
        8_388_608,
        16_777_216,
        33_554_432,
        67_108_864,
        134_217_728,
    )
    assert SCIENCE_UPDATES[-1] == ADAPT_UPDATES
    assert ADAPT_UPDATES * BATCH == 134_217_728


def test_resume_uses_the_latest_checkpoint_not_adapt_0(tmp_path):
    source_update = 512
    for update in (0, 128):
        directory = tmp_path / "checkpoints" / f"adapt_{update}"
        directory.mkdir(parents=True)
        (directory / "state.msgpack").write_bytes(b"state")
        (directory / "config.json").write_text("{}\n", encoding="utf-8")
        (directory / "metadata.json").write_text(
            json.dumps({"global_update": source_update + update}),
            encoding="utf-8",
        )
    assert latest_saved_update(tmp_path, source_update) == 128


def test_metric_log_keeps_only_updates_through_the_checkpoint(tmp_path):
    path = tmp_path / "updates.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {"adaptation_update": update, "aggregation": "this_update_only"}
            )
            + "\n"
            for update in (1, 2, 3)
        ),
        encoding="utf-8",
    )
    kept = _metric_rows(path, 2)
    assert [row["adaptation_update"] for row in kept] == [1, 2]
    assert "this_update_only" in path.read_text(encoding="utf-8")


def test_curve_file_requires_both_dynamics_and_families(tmp_path):
    path = tmp_path / "adapt_0.json"
    assert curve_complete(path, 0) is False
    family = {
        "violation_delivery_rate": 0.0,
        "opportunity_exposure_rate": 0.0,
        "violation_rate_given_opportunity": None,
    }
    path.write_text(
        json.dumps(
            {
                "adaptation_updates": 0,
                "fixed": {
                    "mode": {"natural_reset": family, "common_setup": family},
                    "sample": {"natural_reset": family, "common_setup": family},
                },
                "mutant": {
                    "mode": {"natural_reset": family, "common_setup": family},
                    "sample": {"natural_reset": family, "common_setup": family},
                },
            }
        ),
        encoding="utf-8",
    )
    assert curve_complete(path, 0) is True
    assert curve_complete(path, 32) is False


def test_entropy_can_change_when_branching(tmp_path):
    assert "entropy_coefficient" not in _BRANCH_LOCKED_FIELDS
    config = TickClaimGCConfig(
        num_envs=2,
        num_steps=2,
        num_updates=1,
        minibatch_size=2,
        hidden_size=8,
        goal_mode="workshop12",
        entropy_coefficient=0.005,
    )
    _network, runner = initialize_tick_claim_gc(config)
    save_tick_claim_gc_checkpoint(tmp_path, runner, config)
    branch = replace(
        config,
        variant="mutant",
        goal_mode="deliver_3",
        num_updates=4,
        entropy_coefficient=0.01,
    )
    _branch_network, template = initialize_tick_claim_gc(branch)
    loaded = load_tick_claim_gc_branch(tmp_path, template, branch)
    assert int(loaded.global_update) == 0


def test_empty_minibatch_does_not_apply_adam():
    config = TickClaimGCConfig(
        num_envs=2,
        num_steps=2,
        num_updates=1,
        minibatch_size=2,
        hidden_size=8,
        goal_mode="deliver_3",
    )
    network, runner = initialize_tick_claim_gc(config)
    runner = runner.replace(
        command_active=jax.numpy.zeros_like(runner.command_active)
    )
    update = jax.jit(make_tick_claim_gc_update(network, config))
    updated, metrics = update(runner)
    jax.block_until_ready(updated.global_update)
    assert int(updated.train_state.step) == int(runner.train_state.step)

    def max_abs(left, right):
        return max(
            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for a, b in zip(
                jax.tree_util.tree_leaves(left),
                jax.tree_util.tree_leaves(right),
            )
        )

    assert max_abs(runner.train_state.params, updated.train_state.params) == 0
    assert max_abs(runner.train_state.opt_state, updated.train_state.opt_state) == 0
    assert int(np.asarray(jax.device_get(metrics["valid_transitions"]))) == 0
    assert int(np.asarray(jax.device_get(metrics["empty_minibatches"]))) == (
        config.update_epochs * config.num_minibatches
    )


def test_frozen_eval_reuses_compilation():
    config = TickClaimGCConfig(
        num_envs=2,
        num_steps=2,
        num_updates=1,
        minibatch_size=2,
        hidden_size=8,
        goal_mode="workshop12",
    )
    network, runner = initialize_tick_claim_gc(config)
    before = len(_FROZEN_EVAL_CACHE)
    kwargs = dict(
        variant="fixed",
        stochastic=False,
        repeats_per_state=1,
        seed_base=20000,
        learner_seed=0,
    )
    evaluate_tick_claim_gc_frozen(network, runner.train_state.params, **kwargs)
    after_first = len(_FROZEN_EVAL_CACHE)
    evaluate_tick_claim_gc_frozen(network, runner.train_state.params, **kwargs)
    assert after_first == before + 1
    assert len(_FROZEN_EVAL_CACHE) == after_first
