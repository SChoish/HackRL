from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/manifests/cross_environment_teacher_matrix_v1.json"


def _load() -> dict:
    return json.loads(MANIFEST.read_text())


def test_matrix_authorizes_exact_main_execution_and_common_evaluation() -> None:
    manifest = _load()
    assert manifest["status"] == "main_training_and_common_evaluation_authorized_not_started"
    authorization = manifest["authorization"]
    assert authorization["read_only_audit"] is True
    assert authorization["matrix_and_validator_implementation"] is True
    for key in (
        "missing_condition_runner_implementation",
        "common_trace_evaluator_implementation",
        "contract_smoke",
        "smoke_checkpoint_writes",
        "integration_smoke_complete",
    ):
        assert authorization[key] is True
    for key in (
        "common_frozen_evaluation",
        "main_training",
        "main_training_checkpoint_writes",
        "queue_or_watcher_launch",
    ):
        assert authorization[key] is True
    assert authorization["dreamer_execution"] is False
    assert "full 4,096 updates" in manifest["understood_as"]
    assert "explicit authorization" in manifest["understood_as"]
    assert "later explicit authorization" not in manifest["understood_as"]


def test_condition_set_and_task_goal_mapping_are_canonical() -> None:
    manifest = _load()
    assert set(manifest["canonical_conditions"]) == {
        "source",
        "D-on",
        "D-task",
        "D-frozen",
        "D-off",
    }
    assert manifest["canonical_conditions"]["D-task"]["teacher_target_heads"] == [11]
    for mapping in manifest["shared_learning_contract"]["task_goal_mapping"].values():
        assert mapping == {"goal_id": "delivery/count_ge_3", "goal_index": 11}


def test_missing_jobs_and_slots_match_environment_matrix() -> None:
    manifest = _load()
    matrix = manifest["environment_matrix"]
    assert {
        environment: {
            name: condition["new_learning_jobs"]
            for name, condition in spec["conditions"].items()
            if condition["new_learning_jobs"]
        }
        for environment, spec in matrix.items()
    } == {
        "tick_claim": {"D-task": 10, "D-frozen": 10},
        "pack_restore": {},
        "craft_remain": {"D-task": 10, "D-frozen": 10, "D-off": 10},
    }
    budget = manifest["missing_learning_budget"]
    assert budget["total_jobs"] == 50
    assert budget["collector_slots_per_job"] == 134_217_728
    assert budget["main_training_collection_slots"] == 6_710_886_400
    assert budget["completed_smoke_collection_slots"] == 163_840
    assert budget["planned_total_including_smoke"] == 6_711_050_240
    assert budget["main_training_collection_slots"] == (
        budget["total_jobs"] * budget["collector_slots_per_job"]
    )
    assert budget["planned_total_including_smoke"] == (
        budget["main_training_collection_slots"]
        + budget["completed_smoke_collection_slots"]
    )


def test_smoke_receipt_includes_independent_trace_replay() -> None:
    manifest = _load()
    receipt = json.loads(
        (ROOT / manifest["execution_implementation"]["smoke_receipt"]).read_text()
    )
    trace_smoke = receipt["trace_evaluator_smoke"]
    assert receipt["trace_replay_code_commit"] == (
        manifest["common_trace_implementation"]["posthoc_replay_commit"]
    )
    assert trace_smoke["independent_replay_passed"] == 4
    assert len(trace_smoke["traces"]) == 4
    assert all(row["independent_replay_passed"] for row in trace_smoke["traces"])
    assert all(row["transition_kernel_calls"] == 0 for row in trace_smoke["traces"])
    assert all(row["policy_or_teacher_calls"] == 0 for row in trace_smoke["traces"])


def test_comparison_evidence_classes_prevent_single_factor_overclaim() -> None:
    evidence = _load()["comparison_evidence_status"]
    assert evidence["new_same_code_and_initialization"]
    assert evidence["historical_equivalence_verified"]
    assert evidence["exploratory_due_to_unresolved_code_difference"]
    assert "must not be called single-factor" in evidence["single_factor_claim_rule"]
    assert set(evidence["substitute_evidence_without_independent_source_manifest"]) == {
        "tick_claim", "craft_remain"
    }


def test_old_teacher_split_is_explicitly_non_equivalent() -> None:
    manifest = _load()
    exclusion = manifest["non_equivalent_existing_runs"]["teacher_imitation_split_v1"]
    assert "pretraining" in exclusion["reason"]
    assert "unlearned teacher" in exclusion["reason"]


def test_completed_curves_do_not_claim_full_trace_completion() -> None:
    evaluation = _load()["shared_evaluation_contract"]
    assert "complete action sequence" in evaluation["required_episode_record"]
    assert "authorized" in evaluation["current_gap"]
    assert "scientific performance" in evaluation["validator_scope"]
    assert any(
        "evaluation RNG seed schedule" in item
        for item in evaluation["pre_execution_freeze_required"]
    )
    assert any(
        "actual fixed or mutant kernel" in rule
        for rule in evaluation["independence_controls_required"]
    )
