import json
from dataclasses import replace

import jax
import numpy as np

from hackrl.tick_claim_gc import (
    TickClaimGCConfig,
    evaluate_tick_claim_gc_frozen,
    initialize_tick_claim_gc,
    load_tick_claim_gc_history_branch,
    make_tick_claim_gc_update,
    reset_tick_claim_gc_optimizer,
    save_tick_claim_gc_checkpoint,
)
from run_tick_claim_gc_history import (
    ADAPT_UPDATES,
    BATCH,
    EPISODE_KEYS,
    ORIGINS,
    SEEDS,
    build_jobs,
    curve_complete,
    science_transitions,
)


def test_history_jobs_are_ten_pretrains_and_forty_adapt_cells():
    jobs = build_jobs()
    pretrain = [job for job in jobs if job["kind"] == "pretrain"]
    adapt = [job for job in jobs if job["kind"] == "adapt"]
    assert len(pretrain) == 10
    assert len(adapt) == 40
    assert [job["seed"] for job in adapt][:8] == [5, 5, 6, 6, 7, 7, 8, 8]
    assert {job["seed"] for job in adapt} == set(SEEDS)
    assert {job["origin"] for job in adapt} == set(ORIGINS)
    assert [job["origin"] for job in adapt if job["seed"] == 5 and job["variant"] == "fixed"] == list(ORIGINS)
    none = [job for job in adapt if job["origin"] == "none"]
    assert all(job["depends_on"] == [] for job in none)
    assert all(
        job["depends_on"] == [f"pretrain-deliver3-s{job['seed']}"]
        for job in adapt
        if job["origin"] == "deliver3"
    )
    assert all(
        job["depends_on"] == [f"pretrain-workshop-s{job['seed']}"]
        for job in adapt
        if job["origin"] in {"workshop12", "adam_reset"}
    )
    assert science_transitions()[-1] == ADAPT_UPDATES * BATCH == 134_217_728


def test_adam_reset_keeps_parameters_and_rng():
    config = TickClaimGCConfig(
        num_envs=2,
        num_steps=2,
        num_updates=1,
        minibatch_size=2,
        hidden_size=8,
        goal_mode="deliver_3",
    )
    network, runner = initialize_tick_claim_gc(config)
    update = jax.jit(make_tick_claim_gc_update(network, config))
    stepped, metrics = update(runner)
    jax.block_until_ready(stepped.global_update)
    assert int(np.asarray(jax.device_get(metrics["valid_transitions"]))) > 0
    assert int(stepped.train_state.step) != 0
    reset = reset_tick_claim_gc_optimizer(stepped, config)

    def leaves(value):
        return [
            np.asarray(jax.device_get(leaf))
            for leaf in jax.tree_util.tree_leaves(value)
        ]

    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            leaves(stepped.train_state.params), leaves(reset.train_state.params)
        )
    )
    fresh = initialize_tick_claim_gc(config)[1]
    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            leaves(reset.train_state.opt_state), leaves(fresh.train_state.opt_state)
        )
    )
    assert int(reset.train_state.step) == 0
    assert np.array_equal(
        np.asarray(jax.device_get(stepped.rng)),
        np.asarray(jax.device_get(reset.rng)),
    )
    assert np.array_equal(
        np.asarray(jax.device_get(stepped.env_keys)),
        np.asarray(jax.device_get(reset.env_keys)),
    )


def test_deliver3_history_branch_keeps_parameters(tmp_path):
    config = TickClaimGCConfig(
        num_envs=2,
        num_steps=2,
        num_updates=1,
        minibatch_size=2,
        hidden_size=8,
        goal_mode="deliver_3",
        seed=5,
    )
    _, runner = initialize_tick_claim_gc(config)
    save_tick_claim_gc_checkpoint(tmp_path, runner, config)
    branch = replace(config, variant="mutant", num_updates=2)
    _, template = initialize_tick_claim_gc(branch)
    loaded = load_tick_claim_gc_history_branch(tmp_path, template, branch)
    assert int(loaded.global_update) == 0
    left = np.asarray(jax.device_get(jax.tree_util.tree_leaves(runner.train_state.params)[0]))
    right = np.asarray(jax.device_get(jax.tree_util.tree_leaves(loaded.train_state.params)[0]))
    assert np.array_equal(left, right)


def test_episode_records_name_the_start_state():
    config = TickClaimGCConfig(
        num_envs=2,
        num_steps=2,
        num_updates=1,
        minibatch_size=2,
        hidden_size=8,
        goal_mode="deliver_3",
    )
    network, runner = initialize_tick_claim_gc(config)
    result = evaluate_tick_claim_gc_frozen(
        network,
        runner.train_state.params,
        variant="mutant",
        stochastic=False,
        repeats_per_state=1,
        seed_base=20000,
        learner_seed=5,
        record_episodes=True,
    )
    rows = result["episode_records"]
    assert len(rows) == 64
    assert EPISODE_KEYS <= set(rows[0])
    assert sum(row["family"] == "natural_reset" for row in rows) == 32
    assert sum(row["family"] == "common_setup" for row in rows) == 32
    assert rows[16]["state_index"] == 16
    assert rows[16]["layout"] == 0
    assert rows[16]["phase"] == 1
    assert "episode_records" not in evaluate_tick_claim_gc_frozen(
        network,
        runner.train_state.params,
        variant="mutant",
        stochastic=False,
        repeats_per_state=1,
        seed_base=20000,
        learner_seed=5,
    )


def test_curve_requires_both_dynamics_and_episode_rows(tmp_path):
    path = tmp_path / "adapt_0.json"
    family = {
        "violation_delivery_rate": 0.0,
        "opportunity_exposure_rate": 0.0,
        "violation_rate_given_opportunity": None,
        "success_rate": 1.0,
        "mean_length": 3.0,
    }
    episode = {key: 0 for key in EPISODE_KEYS}
    episode["family"] = "natural_reset"
    view = {
        "natural_reset": family,
        "common_setup": family,
        "episodes": [dict(episode) for _ in range(64)],
    }
    sample = dict(view)
    sample["episodes"] = [dict(episode) for _ in range(256)]
    document = {
        "adaptation_updates": 0,
        "fixed": {"mode": view, "sample": sample},
        "mutant": {"mode": view, "sample": sample},
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    assert curve_complete(path, 0)
    del document["fixed"]
    path.write_text(json.dumps(document), encoding="utf-8")
    assert not curve_complete(path, 0)
