from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/manifests/cross_environment_ppo_dual_dreamer_v1.json"


def _load() -> dict:
    return json.loads(MANIFEST.read_text())


def test_design_has_no_execution_authority() -> None:
    manifest = _load()
    assert manifest["status"] == (
        "scope_and_estimands_frozen_implementation_budget_authority_pending"
    )
    assert not any(manifest["authority"].values())
    assert "grants no authority" in manifest["gate_semantics"]
    assert "stable-repetition latency rule" in manifest["pending_fields"]
    assert manifest["budget_contract"]["normal_pretraining_budget_per_cell"] is None
    assert manifest["budget_contract"]["adaptation_budget_per_branch"] is None


def test_matrix_is_three_systems_by_three_environments() -> None:
    manifest = _load()
    assert set(manifest["comparison_axis"]["algorithms"]) == {
        "PPO",
        "Dual-PPO-task",
        "DreamerV3",
    }
    assert set(manifest["environments"]) == {
        "tick_claim",
        "pack_restore",
        "craft_remain",
    }
    dual = manifest["comparison_axis"]["algorithms"]["Dual-PPO-task"]
    assert dual["frozen_condition"] == "D-task"
    assert "Do not select D-on" in dual["reason"]


def test_seeds_jobs_and_evaluation_caps_are_predeclared() -> None:
    manifest = _load()
    assert manifest["learner_seeds"]["normal_only_development"] == [48, 49]
    assert manifest["learner_seeds"]["held_out_main"] == [50, 51, 52, 53, 54]
    assert manifest["main_protocol"]["main_jobs_if_all_new"] == {
        "source_pretraining": 45,
        "adaptation_branches": 90,
        "total": 135,
    }
    evaluation = manifest["evaluation_contract"]
    assert evaluation["task_receipts"] == 1080
    assert evaluation["episode_cap"] == 59400
    assert evaluation["step_cap"] == 7603200


def test_estimand_and_claim_boundaries_are_explicit() -> None:
    manifest = _load()
    assert "fixed-trained" in manifest["estimands"]["within_system_adaptation"]
    assert "Delta_U" in manifest["estimands"]["between_system_contrast"]
    assert "not assumed statistical pairing" in (
        manifest["estimands"]["between_system_contrast"]
    )
    assert "do not by themselves create statistical pairing" in (
        manifest["learner_seeds"]["rule"]
    )
    assert manifest["evaluation_contract"]["real_kernel_rule"].startswith(
        "Success, return, trigger, and exploitation are scored only from actual"
    )
    prohibited = " ".join(manifest["reporting_contract"]["prohibited_claims"])
    assert "world model alone" in prohibited
    assert "episode count" in prohibited
    assert manifest["external_sources"][0]["doi"] == "10.1038/s41586-025-08744-2"
    assert manifest["external_sources"][1]["license_observed"] == "MIT"
    assert "not external ground truth" in (
        manifest["evaluation_contract"]["oracle_independence_boundary"]
    )
