import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))


def _load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = _load_script(
    "run_pack_inventory_overwrite_train_probability_transfer_test",
    "scripts/run_pack_inventory_overwrite_train_probability_transfer.py",
)
SUMMARY = _load_script(
    "summarize_pack_inventory_overwrite_train_probability_transfer_test",
    "scripts/summarize_pack_inventory_overwrite_train_probability_transfer.py",
)
MANIFEST = json.loads(
    (
        REPOSITORY
        / "docs/manifests/pack_inventory_overwrite_train_probability_transfer_v1.json"
    ).read_text(encoding="utf-8")
)


def test_manifest_locks_only_ten_missing_standard_rate_jobs():
    assert MANIFEST["seeds"] == [35, 36, 37, 38, 39]
    assert {
        name: (
            condition["overwrite_probability"],
            condition["learning_rate_multiplier"],
        )
        for name, condition in MANIFEST["conditions"].items()
    } == {"p00_lr1": (0.0, 1.0), "p100_lr1": (1.0, 1.0)}
    assert MANIFEST["evaluation_probabilities"] == {
        "p00": 0.0,
        "p05": 0.05,
        "p30": 0.30,
        "p100": 1.0,
    }
    assert set(MANIFEST["reused_conditions"]) == {
        "p05_lr1",
        "p05_lr025",
        "p30_lr1",
        "p30_lr025",
    }
    assert MANIFEST["budget"]["total_jobs"] == 10
    assert MANIFEST["budget"]["total_collector_slots"] == 10_485_760
    assert MANIFEST["budget"]["automatic_extension"] is False
    assert MANIFEST["budget"]["probability_sweep"] is False
    assert MANIFEST["budget"]["learning_rate_sweep"] is False
    assert MANIFEST["optimization"]["teacher_updates"] == 0
    assert MANIFEST["optimization"]["bc_updates"] == 0


def test_new_branch_configuration_changes_only_probability_and_run_length():
    RUNNER._configure_lr_module()
    configs = {
        condition: RUNNER.lr._branch_config(35, condition)
        for condition in MANIFEST["conditions"]
    }
    assert {
        condition: (
            config.inventory_overwrite_probability,
            config.learning_rate,
        )
        for condition, config in configs.items()
    } == {"p00_lr1": (0.0, 0.0002), "p100_lr1": (1.0, 0.0002)}
    ignored = {
        "inventory_overwrite_probability",
        "num_updates",
        "checkpoint_updates",
    }
    left, right = configs.values()
    for key, value in vars(left).items():
        if key not in ignored:
            assert vars(right)[key] == value
    assert left.num_updates == right.num_updates == 32


def test_prerequisites_and_all_reused_final_evaluations_are_pinned():
    RUNNER._configure_lr_module()
    RUNNER._validate_prerequisite()
    reused = MANIFEST["prerequisites"]["reused_final_evaluation_sha256"]
    assert len(reused) == 20
    expected = {
        str(
            Path(MANIFEST["prerequisites"]["prior_raw_result"]).parent
            / condition
            / f"seed{seed}"
            / "evaluations/adapt_32.json"
        )
        for condition in MANIFEST["reused_conditions"]
        for seed in MANIFEST["seeds"]
    }
    assert set(reused) == expected


def test_reused_p30_policy_exact_p05_return_matches_precomputed_clue():
    values = []
    for seed in MANIFEST["seeds"]:
        evaluation = SUMMARY.reused_evaluation("p30_lr1", seed)
        point = SUMMARY.evaluate_regime_point(evaluation["risk_profile"], 0.05)
        values.append(point["existing_expected_discounted_return"])
    assert SUMMARY.mean(values) == pytest.approx(0.88496, abs=5e-6)


def test_cross_probability_exact_endpoints_preserve_normal_panel_semantics():
    evaluation = SUMMARY.reused_evaluation("p30_lr1", 35)
    fixed = evaluation["fixed_kernel"]["mode"]["overall"]
    at_zero = SUMMARY.evaluate_regime_point(evaluation["risk_profile"], 0.0)
    at_one = SUMMARY.evaluate_regime_point(evaluation["risk_profile"], 1.0)
    assert at_zero["existing_expected_discounted_return"] == pytest.approx(
        fixed["mean_discounted_return"], abs=1e-15
    )
    assert at_zero["existing_success_rate"] == pytest.approx(
        fixed["success_rate"], abs=1e-15
    )
    risk_free_successes = [
        row
        for row in evaluation["risk_profile"]["episodes"]
        if row["fixed_success"] and row["risk_count"] == 0
    ]
    assert at_one["existing_success_rate"] == pytest.approx(
        len(risk_free_successes) / 32, abs=1e-15
    )


def test_execution_source_set_includes_every_runtime_dependency():
    assert set(MANIFEST["execution_sources"]) == {
        "docs/manifests/pack_inventory_overwrite_train_probability_transfer_v1.json",
        "scripts/run_pack_inventory_overwrite_train_probability_transfer.py",
        "scripts/summarize_pack_inventory_overwrite_train_probability_transfer.py",
        "docs/manifests/pack_inventory_overwrite_lr_ppo_v1.json",
        "scripts/run_pack_inventory_overwrite_lr_ppo.py",
        "scripts/summarize_pack_inventory_overwrite_lr_ppo.py",
        "scripts/run_pack_inventory_overwrite_ppo.py",
        "src/hackrl/pack_restore.py",
        "src/hackrl/pack_restore_gc.py",
        "src/hackrl/pack_inventory_overwrite_rare.py",
        "src/hackrl/dual_leo.py",
    }


def test_evaluation_seed_and_exactness_basis_are_pinned():
    assert MANIFEST["evaluation"]["seed_base"] == 71000
    assert MANIFEST["prerequisites"]["exact_formula_basis_sha256"] == (
        "3462a38e05bdef89c5658a165542a48028fe7c8134bcd7dc42a7f1fd7b545d19"
    )
    assert len(MANIFEST["prerequisites"]["reused_source_origin_sha256"]) == 5


def test_reused_evaluation_payload_contract_is_semantically_validated():
    evaluation = SUMMARY.reused_evaluation("p30_lr025", 39)
    SUMMARY.validate_evaluation_payload(evaluation, "p30_lr025")
    invalid = json.loads(json.dumps(evaluation))
    invalid["risk_profile"]["episodes"][0]["fixed_length"] = 129
    with pytest.raises(AssertionError, match="invalid risk-profile episode"):
        SUMMARY.validate_evaluation_payload(invalid, "p30_lr025")


def test_run_manifest_materializes_full_execution_contract(monkeypatch):
    RUNNER._install_overrides()
    monkeypatch.setattr(RUNNER.lr, "_BASE_RUN_MANIFEST", lambda: {})
    run_manifest = RUNNER._run_manifest()
    assert run_manifest["behavior_drift_anchor"] == MANIFEST["evaluation"][
        "behavior_drift_anchor"
    ]
    assert run_manifest["new_training_job_count"] == 10
