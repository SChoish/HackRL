import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hackrl.mine_expedition_ppo import (
    MineExpeditionPPOConfig,
    MineExpeditionTransition,
    calculate_mine_expedition_gae,
    evaluate_mine_expedition_frozen,
    initialize_mine_expedition_ppo,
    load_mine_expedition_checkpoint,
    make_mine_expedition_update,
    mine_expedition_checkpoint_files_present,
    mine_expedition_parameter_count,
    save_mine_expedition_checkpoint,
)


def _small_config():
    return MineExpeditionPPOConfig(
        seed=0,
        num_envs=2,
        num_steps=4,
        num_updates=1,
        update_epochs=1,
        minibatch_size=8,
        hidden_size=16,
        mode_eval_episodes=1,
        sample_eval_episodes=2,
    )


def test_fixed_gate_rejects_non_training_starts_and_bad_batches():
    with pytest.raises(ValueError, match="curriculum or natural"):
        MineExpeditionPPOConfig(training_start="target_ready").validate()
    with pytest.raises(ValueError, match="divide evenly"):
        MineExpeditionPPOConfig(
            num_envs=2, num_steps=3, minibatch_size=4
        ).validate()
    with pytest.raises(ValueError, match="environment shaping discount"):
        MineExpeditionPPOConfig(gamma=0.99).validate()


def test_timeout_transition_does_not_bootstrap_successor_value():
    shape = (1, 1)
    zeros = jnp.zeros(shape, dtype=jnp.float32)
    flags = jnp.zeros(shape, dtype=jnp.bool_)
    trajectory = MineExpeditionTransition(
        done=jnp.ones(shape, dtype=jnp.bool_),
        action=jnp.zeros(shape, dtype=jnp.int32),
        value=jnp.full(shape, 2.0, dtype=jnp.float32),
        reward=jnp.ones(shape, dtype=jnp.float32),
        log_prob=zeros,
        map_channels=jnp.zeros((1, 1, 1), dtype=jnp.float32),
        numeric_features=jnp.zeros((1, 1, 1), dtype=jnp.float32),
        success=flags,
        timeout=jnp.ones(shape, dtype=jnp.bool_),
        crafted_pickaxe=flags,
        mined_target=flags,
        returned_target=flags,
        iron_increase=flags,
        indirect_use=flags,
        completed_return=zeros,
        completed_length=jnp.zeros(shape, dtype=jnp.int32),
        reset_count=jnp.ones(shape, dtype=jnp.int32),
    )
    config = MineExpeditionPPOConfig()
    advantages, targets = calculate_mine_expedition_gae(
        trajectory,
        last_value=jnp.asarray([100.0], dtype=jnp.float32),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    np.testing.assert_allclose(np.asarray(advantages), [[-1.0]])
    np.testing.assert_allclose(np.asarray(targets), [[1.0]])


def test_manifest_training_contract_matches_executable_defaults():
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/manifests/mine_expedition_fixed_learnability_v1.json"
        ).read_text(encoding="utf-8")
    )
    contract = manifest["training_contract"]
    config = MineExpeditionPPOConfig()
    assert contract["num_envs"] == config.num_envs
    assert contract["num_steps"] == config.num_steps
    assert contract["num_updates"] == config.num_updates
    assert contract["transitions_per_seed"] == config.batch_size * config.num_updates
    assert contract["update_epochs"] == config.update_epochs
    assert contract["minibatch_size"] == config.minibatch_size
    assert contract["hidden_size"] == config.hidden_size
    assert contract["learning_rate"] == config.learning_rate
    assert contract["gamma"] == config.gamma
    assert contract["gae_lambda"] == config.gae_lambda
    assert contract["entropy_coefficient"] == config.entropy_coefficient


def test_one_update_changes_parameters_and_reports_fixed_only_metrics():
    config = _small_config()
    network, runner = initialize_mine_expedition_ppo(config)
    before = jax.tree.map(lambda value: np.asarray(value).copy(), runner.train_state.params)
    updated, metrics = jax.jit(make_mine_expedition_update(network, config))(runner)
    changed = [
        not np.array_equal(left, np.asarray(right))
        for left, right in zip(
            jax.tree.leaves(before),
            jax.tree.leaves(updated.train_state.params),
            strict=True,
        )
    ]
    assert any(changed)
    assert int(updated.global_update) == 1
    assert int(updated.env_steps) == config.batch_size
    assert int(metrics["transitions"]) == config.batch_size
    assert int(metrics["iron_increase_events"]) == 0
    assert int(metrics["indirect_use_events"]) == 0
    assert math.isfinite(float(metrics["loss"]))
    assert math.isfinite(float(metrics["entropy"]))


def test_frozen_natural_evaluation_does_not_mutate_parameters():
    config = _small_config()
    network, runner = initialize_mine_expedition_ppo(config)
    before = jax.tree.map(lambda value: np.asarray(value).copy(), runner.train_state.params)
    result = evaluate_mine_expedition_frozen(
        network,
        runner.train_state.params,
        stochastic=False,
        episodes=1,
        seed_base=100,
        learner_seed=config.seed,
    )
    assert result["variant"] == "fixed"
    assert result["start"] == "natural"
    assert result["episodes"] == 1
    assert 0.0 <= result["success_rate"] <= 1.0
    assert 0.0 <= result["completed_rate"] <= 1.0
    assert all(
        np.array_equal(left, np.asarray(right))
        for left, right in zip(
            jax.tree.leaves(before),
            jax.tree.leaves(runner.train_state.params),
            strict=True,
        )
    )


def test_checkpoint_round_trip_preserves_full_runner(tmp_path):
    config = _small_config()
    _, runner = initialize_mine_expedition_ppo(config)
    destination = save_mine_expedition_checkpoint(tmp_path / "update_0", runner, config)
    _, template = initialize_mine_expedition_ppo(config)
    restored = load_mine_expedition_checkpoint(destination, template, config)
    assert int(restored.global_update) == 0
    assert mine_expedition_parameter_count(restored.train_state.params) > 0
    assert all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree.leaves(runner), jax.tree.leaves(restored), strict=True
        )
    )
    assert not (destination / "state.msgpack.tmp").exists()
    with (destination / "state.msgpack").open("ab") as handle:
        handle.write(b"corrupt")
    assert not mine_expedition_checkpoint_files_present(destination)
