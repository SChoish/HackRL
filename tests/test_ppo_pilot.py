import math

from hackrl import EasyTask
from hackrl.ppo import PPOConfig, run_ppo_pilot


def test_minimal_official_ppo_training_and_evaluation_flow():
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
        )
    )

    assert result["task"] == "R-E"
    assert result["variant"] == "mutant"
    assert result["transitions"] == 8
    assert result["eval_episodes"] == 2
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["entropy"])
    assert math.isfinite(result["eval_mean_return"])
    assert result["final_parameter_norm"] != result["initial_parameter_norm"]
    assert 0.0 <= result["eval_success_rate"] <= 1.0
    assert 0.0 <= result["eval_violation_episode_rate"] <= 1.0
    assert 0.0 <= result["eval_repeat_harvest_episode_rate"] <= 1.0
