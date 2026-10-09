import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import serialization

from hackrl.dual_leo import (
    init_dual_leo_teacher,
    load_dual_checkpoint,
    save_dual_checkpoint,
)
from hackrl import spatial_wall_pass_gc as gc


REPOSITORY = Path(__file__).resolve().parents[1]


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


def _command_only_digest(runner, teacher):
    return serialization.to_bytes(
        {
            "train_state": runner.train_state,
            "env_state": runner.env_state,
            "env_keys": runner.env_keys,
            "sampler_keys": runner.sampler_keys,
            "rng": runner.rng,
            "seen_goals": runner.seen_goals,
            "global_update": runner.global_update,
            "env_steps": runner.env_steps,
            "rollout_cursor": runner.rollout_cursor,
            "teacher": teacher,
        }
    )


def test_dual_adaptation_command_switch_preserves_inherited_state(tmp_path):
    origin = _small_config(goal_mode="normal12", variant="fixed")
    _, origin_runner = gc.initialize_spatial_wall_pass_gc(origin)
    inputs = gc._batch_inputs(
        origin_runner.env_state, origin_runner.current_goal
    )
    _, origin_teacher, _ = init_dual_leo_teacher(
        origin, inputs[0], inputs[1], gc.NUM_GOALS, gc.NUM_ACTIONS
    )
    checkpoint = save_dual_checkpoint(
        tmp_path / "source",
        origin_runner,
        origin_teacher,
        gc.spatial_wall_pass_gc_config_payload(origin),
    )

    branch = _small_config(
        goal_mode="delivery", variant="mutant", num_updates=2
    )
    _, template = gc.initialize_spatial_wall_pass_gc(branch)
    template_inputs = gc._batch_inputs(
        template.env_state, template.current_goal
    )
    _, teacher_template, _ = init_dual_leo_teacher(
        branch,
        template_inputs[0],
        template_inputs[1],
        gc.NUM_GOALS,
        gc.NUM_ACTIONS,
    )
    loaded, loaded_teacher = load_dual_checkpoint(
        checkpoint, template, teacher_template
    )
    before = _command_only_digest(loaded, loaded_teacher)
    commanded = gc.command_delivery(loaded)
    after = _command_only_digest(commanded, loaded_teacher)

    assert before == after
    assert bool(jnp.all(commanded.current_goal == gc.DELIVERY_GOAL_INDEX))
    achieved = jax.vmap(gc._goals)(commanded.env_state)[
        :, gc.DELIVERY_GOAL_INDEX
    ]
    assert bool(jnp.all(commanded.command_active == jnp.logical_not(achieved)))


def test_frozen_evaluation_records_first_wall_pass_route_fields():
    config = _small_config()
    network, runner = gc.initialize_spatial_wall_pass_gc(config)
    result = gc.evaluate_spatial_wall_pass_gc_frozen(
        network,
        runner.train_state.params,
        variant="mutant",
        stochastic=True,
        repeats_per_state=2,
        seed_base=61000,
        learner_seed=130,
        record_episodes=True,
    )
    assert len(result["episode_records"]) == 32
    required = {
        "first_wall_pass_action",
        "first_wall_pass_action_name",
        "first_wall_pass_from",
        "first_wall_pass_intermediate",
        "first_wall_pass_destination",
    }
    assert all(required <= set(record) for record in result["episode_records"])
    for record in result["episode_records"]:
        if record["first_wall_pass_step"] < 0:
            assert record["first_wall_pass_action"] is None
            assert record["first_wall_pass_from"] is None
        else:
            assert record["first_wall_pass_action"] is not None
            assert len(record["first_wall_pass_from"]) == 2
            assert len(record["first_wall_pass_intermediate"]) == 2
            assert len(record["first_wall_pass_destination"]) == 2



def _adaptation_runner_module():
    path = (
        REPOSITORY
        / "scripts/run_spatial_wall_pass_dual_adaptation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_spatial_wall_pass_dual_adaptation_runner", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_adaptation_evaluation_uses_configured_learner_seed(monkeypatch):
    module = _adaptation_runner_module()
    config = _small_config(seed=130)
    network, runner = gc.initialize_spatial_wall_pass_gc(config)
    calls = []

    def fake_evaluation(*args, **kwargs):
        del args
        calls.append((kwargs["variant"], kwargs["learner_seed"]))
        return {"episode_records": []}

    monkeypatch.setattr(
        module, "evaluate_spatial_wall_pass_gc_frozen", fake_evaluation
    )
    result = module._evaluate(network, runner, config, 0)

    assert calls == [
        ("fixed", 130),
        ("fixed", 130),
        ("mutant", 130),
        ("mutant", 130),
    ]
    assert result["adaptation_update"] == 0
    assert result["runner_state_immutable"] is True

def test_dual_adaptation_manifest_freezes_four_job_budget():
    manifest = json.loads(
        (
            REPOSITORY
            / "docs/manifests/spatial_wall_pass_dual_adaptation_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["training"]["total_jobs"] == 4
    assert manifest["training"]["adaptation_updates"] == 4096
    assert manifest["training"]["total_transitions"] == 536_870_912
    assert manifest["branches"]["gc_adaptation"] is False
    assert manifest["evaluation"]["same_frozen_policy_across_kernels"] is True
    assert manifest["storage"]["destination"].startswith(
        "/raid/ext_csv/HackRL/runs/"
    )
