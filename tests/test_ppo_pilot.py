import math

import pytest

from hackrl import EasyTask, MediumTask
from hackrl.ppo import PPOConfig, run_ppo_pilot


def test_protocol_default_keeps_learning_rate_fixed():
    assert not PPOConfig().anneal_learning_rate


def test_r_m_diagnostic_start_modes_validate_only_for_r_m():
    with pytest.raises(ValueError, match="r_m_d1/d2/d3"):
        PPOConfig(task=EasyTask.R_E, start_mode="r_m_d1").validate()
    PPOConfig(task=MediumTask.R_M, start_mode="r_m_d1").validate()
    assert PPOConfig().eval_sample_episodes is None
    assert PPOConfig(eval_sample_episodes=128).resolved_eval_sample_episodes == 128


def test_minimal_official_ppo_training_and_evaluation_flow(tmp_path):
    result = run_ppo_pilot(
        PPOConfig(
            task=EasyTask.R_E,
            mutant=True,
            seed=0,
            num_envs=2,
            num_steps=4,
            num_updates=1,
            update_epochs=1,
            num_minibatches=1,
            layer_size=16,
            eval_episodes=2,
            log_dir=str(tmp_path),
        )
    )

    assert result["task"] == "R-E"
    assert result["variant"] == "mutant"
    assert result["start_mode"] == "default"
    assert result["transitions"] == 8
    assert result["eval_episodes"] == 2
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["entropy"])
    assert math.isfinite(result["eval_mean_return"])
    assert result["final_parameter_norm"] != result["initial_parameter_norm"]
    assert 0.0 <= result["eval_success_rate"] <= 1.0
    assert 0.0 <= result["eval_sample_success_rate"] <= 1.0
    assert 0.0 <= result["eval_violation_episode_rate"] <= 1.0
    assert 0.0 <= result["eval_repeat_harvest_episode_rate"] <= 1.0
    assert 0.0 <= result["eval_iron_acquisition_episode_rate"] <= 1.0
    assert 0.0 <= result["eval_iron_pickaxe_craft_episode_rate"] <= 1.0
    assert 0.0 <= result["completed_wood_depletion_rate"] <= 1.0
    assert 0.0 <= result["completed_damage_episode_rate"] <= 1.0
    assert 0.0 <= result["completed_timeout_termination_rate"] <= 1.0
    assert 0.0 <= result["eval_death_before_iron_rate"] <= 1.0
    assert 0.0 <= result["eval_timeout_before_iron_rate"] <= 1.0
    assert result["metrics_schema_version"] == 2
    assert result["fixture_dynamics_version"] == 2
    assert result["fixture"] == "default"
    assert result["git_sha"]
    assert (tmp_path / "params.msgpack").is_file()
    assert (tmp_path / "updates.csv").is_file()
    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "working_tree.patch").is_file()


def test_checkpoint_updates_write_params_and_eval(tmp_path):
    result = run_ppo_pilot(
        PPOConfig(
            task=EasyTask.R_E,
            mutant=True,
            seed=0,
            num_envs=2,
            num_steps=4,
            num_updates=1,
            update_epochs=1,
            num_minibatches=1,
            layer_size=16,
            eval_episodes=2,
            learning_rate=2e-4,
            checkpoint_updates=(0, 1),
            log_dir=str(tmp_path),
        )
    )
    assert result["transitions"] == 8
    assert (tmp_path / "checkpoints" / "transitions_0" / "params.msgpack").is_file()
    assert (tmp_path / "checkpoints" / "transitions_8" / "eval.json").is_file()
