import importlib.util
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))


def _load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = _load_script(
    "run_pack_inventory_overwrite_lr_ppo_test",
    "scripts/run_pack_inventory_overwrite_lr_ppo.py",
)
MANIFEST = json.loads(
    (
        REPOSITORY
        / "docs/manifests/pack_inventory_overwrite_lr_ppo_v1.json"
    ).read_text(encoding="utf-8")
)


def test_manifest_locks_twenty_job_single_intervention():
    assert MANIFEST["seeds"] == [35, 36, 37, 38, 39]
    assert {
        name: (
            condition["overwrite_probability"],
            condition["learning_rate_multiplier"],
        )
        for name, condition in MANIFEST["conditions"].items()
    } == {
        "p05_lr1": (0.05, 1.0),
        "p05_lr025": (0.05, 0.25),
        "p30_lr1": (0.30, 1.0),
        "p30_lr025": (0.30, 0.25),
    }
    optimization = MANIFEST["optimization"]
    assert optimization["num_envs"] * optimization["num_steps"] == 32768
    assert optimization["updates"] * 32768 == 1_048_576
    assert MANIFEST["budget"]["total_jobs"] == 20
    assert MANIFEST["budget"]["total_collector_slots"] == 20_971_520
    assert MANIFEST["evaluation"]["adaptation_updates"] == [0, 2, 8, 32]
    assert MANIFEST["budget"]["automatic_extension"] is False
    assert MANIFEST["budget"]["learning_rate_sweep"] is False
    assert optimization["teacher_updates"] == optimization["bc_updates"] == 0


def test_branch_configuration_changes_only_probability_rate_and_run_length():
    configs = {
        condition: RUNNER._branch_config(35, condition)
        for condition in MANIFEST["conditions"]
    }
    standard = configs["p05_lr1"]
    assert {
        condition: (
            config.inventory_overwrite_probability,
            config.learning_rate,
        )
        for condition, config in configs.items()
    } == {
        "p05_lr1": (0.05, 0.0002),
        "p05_lr025": (0.05, 0.00005),
        "p30_lr1": (0.30, 0.0002),
        "p30_lr025": (0.30, 0.00005),
    }
    ignored = {
        "learning_rate",
        "inventory_overwrite_probability",
        "num_updates",
        "checkpoint_updates",
    }
    standard_payload = vars(standard)
    for key in standard_payload:
        if key not in ignored:
            assert all(
                vars(config)[key] == standard_payload[key]
                for config in configs.values()
            )
    assert all(config.num_updates == 32 for config in configs.values())


def test_categorical_drift_is_zero_for_identical_actor():
    logits = jnp.asarray([[1.0, 0.0], [-2.0, 3.0]])
    result = RUNNER._categorical_drift(logits, logits)
    assert abs(result["mean_kl_source_to_current"]) < 1e-8
    assert result["argmax_action_change_count"] == 0
    assert result["argmax_action_change_rate"] == 0.0


def test_categorical_drift_detects_distribution_and_argmax_change():
    source = jnp.asarray([[3.0, 0.0], [0.0, 2.0]])
    current = jnp.asarray([[0.0, 3.0], [0.0, 2.0]])
    result = RUNNER._categorical_drift(source, current)
    assert result["mean_kl_source_to_current"] > 0.0
    assert result["argmax_action_change_count"] == 1
    assert result["argmax_action_change_rate"] == 0.5


def test_tracker_counts_probabilistic_fires_separately_from_opportunities():
    tracker = RUNNER._empty_tracker(2)
    metrics = {
        "loss_worker_mask": jnp.asarray([True, False]),
        "world_done_worker_mask": jnp.asarray([True, True]),
        "transitions": jnp.asarray(8),
        "valid_transitions": jnp.asarray(7),
        "empty_minibatches": jnp.asarray(0),
        "world_resets": jnp.asarray(2),
        "lost_grain": jnp.asarray(1),
        "physical_loss_events": jnp.asarray(1),
        "overwrite_trigger_events": jnp.asarray(4),
        "overwrite_fire_events": jnp.asarray(1),
        "goal_successes": jnp.asarray(1),
        "harvest_events": jnp.asarray(4),
        "storage_events": jnp.asarray(0),
        "delivery_events": jnp.asarray(3),
    }
    updated = RUNNER.update_exposure_tracker(
        tracker, metrics, learning=True, optimizer_applications=1
    )
    assert updated["overwrite_trigger_events"] == 4
    assert updated["overwrite_fire_events"] == 1
    assert updated["loss_episodes"] == 1
    assert updated["valid_ppo_samples"] == 7


def test_prerequisite_hashes_and_prior_status_are_pinned():
    RUNNER._validate_prerequisite()
    assert (
        RUNNER.SPEC["prerequisites"]["prior_result_sha256"]
        == "5eceabb11aba5eacd5e3a1087edff7a5e9771cb43f3ac74cca9937b0dc1979fa"
    )
