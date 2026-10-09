import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


def _load_script(name, relative):
    path = REPOSITORY / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _replication():
    return _load_script(
        "test_spatial_wall_pass_dual_replication_runner",
        "scripts/run_spatial_wall_pass_dual_replication.py",
    )


def _adaptation():
    return _load_script(
        "test_spatial_wall_pass_dual_replication_adaptation",
        "scripts/run_spatial_wall_pass_dual_adaptation.py",
    )


def _fixed():
    return _load_script(
        "test_spatial_wall_pass_dual_replication_fixed",
        "scripts/run_spatial_wall_pass_fixed_gate.py",
    )


def test_replication_manifest_freezes_seed_budget_and_denominators():
    module = _replication()
    manifest = module.validate_manifest()
    assert manifest["seeds"]["learner"] == [132, 133, 134, 135, 136]
    assert manifest["seeds"]["replacement"] is False
    assert manifest["budget"]["maximum_jobs"] == 15
    assert manifest["budget"]["maximum_transitions"] == 1_426_063_360
    assert len(manifest["evaluation"]["report_two_denominators"]) == 2


def test_replication_provenance_covers_every_child_execution_source():
    replication = _replication()
    fixed = _fixed()
    adaptation = _adaptation()
    child_sources = set(fixed.REPLICATION_EXECUTION_SOURCES) | set(
        adaptation.REPLICATION_EXECUTION_SOURCES
    )
    assert child_sources <= set(replication.EXECUTION_SOURCES)


def test_manifest_validation_rejects_critical_training_drift(
    tmp_path, monkeypatch
):
    module = _replication()
    manifest = json.loads(module.MANIFEST.read_text(encoding="utf-8"))
    manifest["training"]["num_envs"] = 256
    changed = tmp_path / "changed_manifest.json"
    changed.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(module, "MANIFEST", changed)
    with pytest.raises(RuntimeError, match="replication manifest changed"):
        module.validate_manifest()


def test_replication_fixed_contract_is_dual_only_and_uses_new_seeds():
    settings = _fixed()._contract_settings("replication")
    assert settings["seeds"] == (132, 133, 134, 135, 136)
    assert settings["methods"] == ("dual",)
    assert str(settings["root"]).endswith(
        "/spatial_wall_pass_dual_replication_v1/pretraining"
    )


@pytest.mark.parametrize(
    ("mode", "sample", "passes"),
    [
        (1.0, 1.0, True),
        (0.9375, 0.90, True),
        (0.875, 1.0, False),
        (1.0, 0.899, False),
    ],
)
def test_normal_gate_requires_both_final_success_rates(mode, sample, passes):
    module = _replication()
    summary = {
        "final_evaluation": {
            "fixed": {
                "mode": {"success_rate": mode},
                "sample": {"success_rate": sample},
            }
        }
    }
    assert module.classify_normal(summary)["passes_normal_gate"] is passes


def test_exploit_gate_and_repaired_dependency_are_separate():
    module = _replication()
    evaluation = {
        "mutant": {
            "mode": {
                "success_rate": 1.0,
                "wall_pass_rate": 1.0,
                "beneficial_use_rate": 1.0,
                "mean_discounted_return": 0.96,
            }
        },
        "fixed": {
            "mode": {
                "success_rate": 0.0,
                "mean_discounted_return": 0.0,
            }
        },
    }
    decision = module.classify_exploit(evaluation)
    assert decision["passes_exploit_gate"] is True
    assert decision["passes_defect_dependency_gate"] is True

    evaluation["fixed"]["mode"]["success_rate"] = 1.0
    evaluation["fixed"]["mode"]["mean_discounted_return"] = 0.94
    decision = module.classify_exploit(evaluation)
    assert decision["passes_exploit_gate"] is True
    assert decision["passes_defect_dependency_gate"] is False


def test_adaptation_accepts_only_preregistered_replication_gate(tmp_path):
    module = _adaptation()
    gate = {
        "schema_version": (
            "hackrl_spatial_wall_pass_dual_replication_gate_v1"
        ),
        "status": "complete",
        "execution_code_sha": "abc123",
        "all_seeds": [132, 133, 134, 135, 136],
        "passing_seeds": [132, 135],
        "pretrain_root": (
            "/raid/ext_csv/HackRL/runs/"
            "spatial_wall_pass_dual_replication_v1/pretraining"
        ),
        "adaptation_root": (
            "/raid/ext_csv/HackRL/runs/"
            "spatial_wall_pass_dual_replication_v1/adaptation"
        ),
        "final_checkpoint_state_sha256": {
            "132": "hash132",
            "135": "hash135",
        },
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    module.configure_replication(path)
    assert module.IS_REPLICATION is True
    assert module.SEEDS == (132, 135)
    assert module.EXPECTED_SOURCE_STATE_SHA256 == {
        132: "hash132",
        135: "hash135",
    }

    gate["all_seeds"] = [132, 133, 134, 135, 137]
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(RuntimeError, match="all-seed set changed"):
        module.configure_replication(path)


def test_adaptation_rejects_a_different_code_commit(tmp_path, monkeypatch):
    module = _adaptation()
    gate = {
        "schema_version": (
            "hackrl_spatial_wall_pass_dual_replication_gate_v1"
        ),
        "status": "complete",
        "execution_code_sha": "pretraining-sha",
        "all_seeds": [132, 133, 134, 135, 136],
        "passing_seeds": [132],
        "pretrain_root": (
            "/raid/ext_csv/HackRL/runs/"
            "spatial_wall_pass_dual_replication_v1/pretraining"
        ),
        "adaptation_root": (
            "/raid/ext_csv/HackRL/runs/"
            "spatial_wall_pass_dual_replication_v1/adaptation"
        ),
        "final_checkpoint_state_sha256": {"132": "hash132"},
        "method_gate": {
            "dual": {"eligible_for_future_adaptation": True}
        },
        "decision": {"mutant_training_started_at_gate_time": False},
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    module.configure_replication(path)
    monkeypatch.setattr(module, "_git", lambda *args: "different-sha")
    with pytest.raises(RuntimeError, match="code commit differs"):
        module.validate_contract()


def test_result_keeps_all_seed_and_normal_pass_denominators_separate(
    tmp_path, monkeypatch
):
    module = _replication()
    monkeypatch.setattr(module, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(module, "ADAPTATION_ROOT", tmp_path / "adaptation")
    monkeypatch.setattr(module, "_git", lambda *args: "test-sha")
    gate = {
        "passing_seeds": [132, 133, 134],
        "cells": [
            {
                "seed": seed,
                "passes_normal_gate": seed in {132, 133, 134},
                "fixed_mode_success_rate": (
                    1.0 if seed in {132, 133, 134} else 0.0
                ),
                "fixed_sample_success_rate": (
                    1.0 if seed in {132, 133, 134} else 0.0
                ),
            }
            for seed in module.SEEDS
        ],
    }

    def evaluation(exploit, dependent):
        return {
            "mutant": {
                "mode": {
                    "success_rate": 1.0 if exploit else 0.0,
                    "wall_pass_rate": 1.0 if exploit else 0.0,
                    "beneficial_use_rate": 1.0 if exploit else 0.0,
                    "mean_discounted_return": 0.96 if exploit else 0.0,
                }
            },
            "fixed": {
                "mode": {
                    "success_rate": 0.0 if dependent else 1.0,
                    "mean_discounted_return": 0.0 if dependent else 0.94,
                }
            },
        }

    adaptation = {
        "cells": {
            "dual_mutant_seed_132": {
                "final_evaluation": evaluation(True, True)
            },
            "dual_mutant_seed_133": {
                "final_evaluation": evaluation(True, False)
            },
            "dual_mutant_seed_134": {
                "final_evaluation": evaluation(False, False)
            },
        }
    }
    result = module.summarize(gate, adaptation)
    assert result["all_seed_denominator"] == 5
    assert result["normal_gate_pass_count"] == 3
    assert result["normal_gate_pass_rate"] == pytest.approx(3 / 5)
    assert result["exploit_count_all_seeds"] == 2
    assert result["normal_and_exploit_rate_all_seeds"] == pytest.approx(2 / 5)
    assert result["exploit_rate_given_normal_gate"] == pytest.approx(2 / 3)
    assert result["defect_dependency_count_all_seeds"] == 1
    assert result["defect_dependency_rate_given_normal_gate"] == pytest.approx(
        1 / 3
    )


def test_zero_normal_gate_passes_still_produces_a_complete_result(
    tmp_path, monkeypatch
):
    module = _replication()
    monkeypatch.setattr(module, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(module, "ADAPTATION_ROOT", tmp_path / "adaptation")
    monkeypatch.setattr(module, "_git", lambda *args: "test-sha")
    gate = {
        "passing_seeds": [],
        "cells": [
            {
                "seed": seed,
                "passes_normal_gate": False,
                "fixed_mode_success_rate": 0.0,
                "fixed_sample_success_rate": 0.0,
            }
            for seed in module.SEEDS
        ],
    }
    result = module.summarize(gate, None)
    assert result["status"] == "complete"
    assert result["all_seed_denominator"] == 5
    assert result["normal_gate_pass_count"] == 0
    assert result["normal_gate_pass_rate"] == 0.0
    assert result["exploit_count_all_seeds"] == 0
    assert result["exploit_rate_given_normal_gate"] is None
    assert result["defect_dependency_rate_given_normal_gate"] is None
    assert result["actual_jobs"] == 5
    assert result["actual_transitions"] == 83_886_080
    assert (tmp_path / "result.json").exists()


def test_production_queue_rejects_direct_python_launch(
    tmp_path, monkeypatch
):
    module = _replication()
    monkeypatch.setattr(module, "RUN_ROOT", tmp_path)
    for name in (
        "HACKRL_DEVICE",
        "JAX_PLATFORMS",
        "CUDA_VISIBLE_DEVICES",
        "HACKRL_DUMMY_KEEPALIVE_CLEARED",
        "HACKRL_GPU_GUARD_ACTIVE",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="must use the GPU entrypoint"):
        module.run_queue()
