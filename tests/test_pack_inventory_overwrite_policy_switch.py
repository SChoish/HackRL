import importlib.util
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))


def _load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load_script(
    "evaluate_pack_inventory_overwrite_policy_switch_test",
    "scripts/evaluate_pack_inventory_overwrite_policy_switch.py",
)
MANIFEST = json.loads(
    (
        REPOSITORY
        / "docs/manifests/pack_inventory_overwrite_policy_switch_v1.json"
    ).read_text(encoding="utf-8")
)


def test_manifest_locks_zero_update_1280_episode_diagnostic():
    assert MANIFEST["status"] == "frozen_before_execution"
    assert MANIFEST["seeds"] == [35, 36, 37, 38, 39]
    assert list(MANIFEST["execution_modes"]) == [
        "source",
        "adapted",
        "one_step",
        "persistent",
    ]
    assert MANIFEST["evaluation"]["direct_kernels"] == [
        "fixed",
        "inventory_overwrite_p1",
    ]
    assert MANIFEST["budget"]["learning_updates"] == 0
    assert MANIFEST["budget"]["teacher_updates"] == 0
    assert MANIFEST["budget"]["bc_updates"] == 0
    assert MANIFEST["budget"]["episodes"] == 1280
    assert MANIFEST["budget"]["maximum_environment_steps"] == 163840
    assert MANIFEST["budget"]["automatic_extension"] is False


def test_one_step_switches_only_on_current_source_risk():
    source = jnp.asarray([1, 2, 3])
    adapted = jnp.asarray([4, 5, 6])
    risk = jnp.asarray([False, True, False])
    persistent = jnp.asarray([True, False, False])
    chosen, use_adapted, switch = EVALUATOR.choose_actions(
        "one_step", source, adapted, risk, persistent
    )
    assert chosen.tolist() == [1, 5, 3]
    assert use_adapted.tolist() == [False, True, False]
    assert switch.tolist() == [False, True, False]


def test_persistent_switch_retains_existing_episode_state():
    source = jnp.asarray([1, 2, 3])
    adapted = jnp.asarray([4, 5, 6])
    risk = jnp.asarray([False, True, False])
    persistent = jnp.asarray([True, False, False])
    chosen, use_adapted, switch = EVALUATOR.choose_actions(
        "persistent", source, adapted, risk, persistent
    )
    assert chosen.tolist() == [4, 5, 3]
    assert use_adapted.tolist() == [True, True, False]
    assert switch.tolist() == [False, True, False]


def test_exact_profile_endpoints_have_expected_semantics():
    episodes = [
        {"success": True, "discounted_return": 0.9, "executed_risk_count": 0},
        {"success": True, "discounted_return": 0.8, "executed_risk_count": 1},
        {"success": False, "discounted_return": 0.0, "executed_risk_count": 2},
    ]
    profile = EVALUATOR.polynomial_profile(episodes)
    at_zero = EVALUATOR.evaluate_profile(profile, 0.0)
    at_one = EVALUATOR.evaluate_profile(profile, 1.0)
    assert at_zero["success_rate"] == pytest.approx(2 / 3)
    assert at_zero["expected_discounted_return"] == pytest.approx(1.7 / 3)
    assert at_zero["overwrite_loss_episode_probability"] == pytest.approx(0.0)
    assert at_one["success_rate"] == pytest.approx(1 / 3)
    assert at_one["expected_discounted_return"] == pytest.approx(0.3)
    assert at_one["overwrite_loss_episode_probability"] == pytest.approx(2 / 3)


def test_execution_source_set_covers_manifest_evaluator_and_kernel_dependencies():
    assert set(MANIFEST["execution_sources"]) == {
        "docs/manifests/pack_inventory_overwrite_policy_switch_v1.json",
        "scripts/evaluate_pack_inventory_overwrite_policy_switch.py",
        "src/hackrl/pack_restore.py",
        "src/hackrl/pack_restore_gc.py",
        "src/hackrl/pack_inventory_overwrite_rare.py",
        "src/hackrl/dual_leo.py",
    }
