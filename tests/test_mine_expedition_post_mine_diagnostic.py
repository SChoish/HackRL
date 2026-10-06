import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest
from flax import serialization


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_mine_expedition_post_mine",
    REPOSITORY / "scripts/diagnose_mine_expedition_post_mine.py",
)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DIAGNOSTIC)


def _empty_trace(steps=3, episodes=2):
    scalar_int = np.zeros((steps, episodes), dtype=np.int32)
    scalar_float = np.zeros((steps, episodes), dtype=np.float32)
    scalar_bool = np.zeros((steps, episodes), dtype=bool)
    positions = np.zeros((steps, episodes, 2), dtype=np.int32)
    before_camp_distance = np.full((steps, episodes), 2, dtype=np.int32)
    after_camp_distance = np.full((steps, episodes), 1, dtype=np.int32)
    if steps > 1 and episodes > 0:
        before_camp_distance[1:, 0] = 0
        after_camp_distance[1:, 0] = 0
    return {
        "active": np.ones((steps, episodes), dtype=bool),
        "action": scalar_int.copy(),
        "value": scalar_float.copy(),
        "reward": scalar_float.copy(),
        "done": scalar_bool.copy(),
        "success": scalar_bool.copy(),
        "timeout": scalar_bool.copy(),
        "crafted": scalar_bool.copy(),
        "mined": scalar_bool.copy(),
        "returned": scalar_bool.copy(),
        "before_tick": scalar_int.copy(),
        "after_tick": scalar_int.copy(),
        "before_position": positions.copy(),
        "after_position": positions.copy(),
        "before_carried_target": np.ones((steps, episodes), dtype=np.int32),
        "after_carried_target": np.ones((steps, episodes), dtype=np.int32),
        "before_returned_target": scalar_int.copy(),
        "after_returned_target": scalar_int.copy(),
        "before_target_remaining": scalar_int.copy(),
        "after_target_remaining": scalar_int.copy(),
        "before_task_distance": scalar_int.copy(),
        "after_task_distance": scalar_int.copy(),
        "before_camp_distance": before_camp_distance,
        "after_camp_distance": after_camp_distance,
    }


def test_canonical_mining_transition_switches_goal_without_termination():
    _, _, row = DIAGNOSTIC._canonical_transition()
    assert row["position"] == row["expected_target_pose"]
    assert row["target_remaining_before"] == 1
    assert row["target_remaining_after"] == 0
    assert row["carried_target_before"] == 0
    assert row["carried_target_after"] == 1
    assert row["task_distance_before"] == 0
    assert row["task_distance_after"] > 0
    assert row["observation_carried_target_before"] == 0.0
    assert row["observation_carried_target_after"] > 0.0
    assert row["observation_target_channel_sum_before"] == 1.0
    assert row["observation_target_channel_sum_after"] == 0.0
    assert row["gae_nonterminal_mask"] == 1.0
    assert not row["done"]
    assert not row["success"]
    assert not row["timeout"]


def test_episode_metrics_separate_delivery_from_navigation_failure():
    trace = _empty_trace()
    trace["action"][1, 0] = int(DIAGNOSTIC.MineExpeditionAction.RETURN_TARGET)
    trace["returned"][1, 0] = True
    trace["success"][1, 0] = True
    trace["done"][1, 0] = True
    trace["after_carried_target"][1, 0] = 0
    trace["after_returned_target"][1:, 0] = 1
    trace["active"][2, 0] = False
    trace["timeout"][2, 1] = True
    trace["done"][2, 1] = True

    result = DIAGNOSTIC._episode_metrics(trace, start_is_post_mine=True)

    assert result["success_count"] == 1
    assert result["timeout_count"] == 1
    assert result["reached_return_pose_count"] == 1
    assert result["legal_return_action_attempt_count"] == 1
    assert result["failed_after_mining_count"] == 1
    assert result["failed_after_mining_never_reached_return_pose_count"] == 1
    assert result["failed_after_mining_reached_pose_without_legal_return_count"] == 0
    assert result["success_state_mismatch_steps"] == 0
    assert result["return_event_mismatch_steps"] == 0
    assert result["terminal_classification_mismatch_steps"] == 0


def test_final_step_arrival_is_not_classified_as_never_reached():
    trace = _empty_trace(steps=1, episodes=1)
    trace["before_camp_distance"][0, 0] = 1
    trace["after_camp_distance"][0, 0] = 0
    trace["timeout"][0, 0] = True
    trace["done"][0, 0] = True

    result = DIAGNOSTIC._episode_metrics(trace, start_is_post_mine=True)

    assert result["reached_return_pose_count"] == 1
    assert result["timed_out_after_mining_never_reached_return_pose_count"] == 0
    assert (
        result["timed_out_after_mining_reached_pose_without_legal_return_count"]
        == 1
    )


def test_agreement_mismatch_is_fatal():
    evaluations = {
        "canonical_post_mine_mode": {
            "success_state_mismatch_steps": 0,
            "return_event_mismatch_steps": 0,
            "terminal_classification_mismatch_steps": 1,
        }
    }
    with pytest.raises(RuntimeError, match="classification mismatch"):
        DIAGNOSTIC._validate_agreement([{"evaluations": evaluations}])


def test_manifest_protocol_controls_episode_counts_and_seeds():
    manifest = DIAGNOSTIC._read_json(DIAGNOSTIC.MANIFEST)
    protocol = DIAGNOSTIC._evaluation_protocol(manifest)
    assert protocol == {
        "mode_episodes": 1,
        "sample_episodes": 128,
        "mode_action_seed": 61000,
        "sample_action_seed": 62000,
        "horizon": 256,
    }


def test_rollout_shapes_cover_full_horizon_and_do_not_require_training():
    config = DIAGNOSTIC.MineExpeditionPPOConfig(
        seed=0,
        num_envs=1,
        num_steps=1,
        num_updates=1,
        minibatch_size=1,
        hidden_size=8,
        training_start="target_ready",
        mode_eval_episodes=1,
        sample_eval_episodes=1,
        checkpoint_updates=(1,),
    )
    network, runner = DIAGNOSTIC.initialize_mine_expedition_ppo(config)
    rollout = DIAGNOSTIC.make_rollout(network, stochastic=False, episodes=1)
    state = DIAGNOSTIC.reset_mine_expedition(
        jax.random.PRNGKey(0), DIAGNOSTIC.MineExpeditionStart.TARGET_READY
    )
    before = serialization.to_bytes(runner)
    _, trace = rollout(
        runner.train_state.params,
        state,
        jax.random.split(jax.random.PRNGKey(1), 1),
    )
    assert trace["action"].shape == (DIAGNOSTIC.WORLD_HORIZON, 1)
    assert before == serialization.to_bytes(runner)
