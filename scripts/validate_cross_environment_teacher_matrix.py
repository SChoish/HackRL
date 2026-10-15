#!/usr/bin/env python3
"""Read-only validator for the cross-environment teacher matrix audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hackrl.craft_remain_env import GOAL_IDS as CRAFT_GOAL_IDS
from hackrl.cross_environment_teacher_matrix import build_jobs
from hackrl.pack_restore import GOAL_IDS as PACK_GOAL_IDS
from hackrl.tick_claim import GOAL_IDS as TICK_GOAL_IDS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / "docs/manifests/cross_environment_teacher_matrix_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _glob_count(root: Path, pattern: str) -> int:
    patterns = (
        [pattern.replace("{fixed,mutant}", variant) for variant in ("fixed", "mutant")]
        if "{fixed,mutant}" in pattern
        else [pattern]
    )
    return sum(len(list(root.glob(item))) for item in patterns)


def validate(manifest: dict) -> dict:
    errors: list[str] = []
    artifact_results = []
    for check in manifest["artifact_checks"]:
        root = _resolve(check["root"])
        observed = _glob_count(root, check["glob"]) if root.exists() else 0
        passed = observed == check["expected"]
        artifact_results.append(
            {
                "id": check["id"],
                "observed": observed,
                "expected": check["expected"],
                "passed": passed,
            }
        )
        if not passed:
            errors.append(
                f"{check['id']}: observed {observed}, expected {check['expected']}"
            )

    aggregate_results = []
    for check in manifest["aggregate_checks"]:
        path = _resolve(check["path"])
        if not path.exists():
            aggregate_results.append({"id": check["id"], "passed": False})
            errors.append(f"{check['id']}: missing {path}")
            continue
        data = json.loads(path.read_text())
        passed = (
            data.get("execution_complete") is check["execution_complete"]
            and data.get("validated_curve_count") == check["validated_curve_count"]
            and not data.get("errors")
            and not data.get("missing_curves")
        )
        aggregate_results.append(
            {
                "id": check["id"],
                "execution_complete": data.get("execution_complete"),
                "validated_curve_count": data.get("validated_curve_count"),
                "passed": passed,
            }
        )
        if not passed:
            errors.append(f"{check['id']}: aggregate contract failed")

    hash_results = []
    for raw_path, expected in manifest["audit_provenance"]["source_sha256"].items():
        path = _resolve(raw_path)
        observed = _sha256(path) if path.exists() else None
        passed = observed == expected
        hash_results.append(
            {
                "path": raw_path,
                "observed": observed,
                "expected": expected,
                "passed": passed,
            }
        )
        if not passed:
            errors.append(f"source hash mismatch: {raw_path}")

    goal_catalogs = {
        "tick_claim": TICK_GOAL_IDS,
        "pack_restore": PACK_GOAL_IDS,
        "craft_remain": CRAFT_GOAL_IDS,
    }
    goal_results = {}
    for environment, expected in manifest["shared_learning_contract"][
        "task_goal_mapping"
    ].items():
        catalog = goal_catalogs[environment]
        observed_index = catalog.index(expected["goal_id"])
        passed = observed_index == expected["goal_index"]
        goal_results[environment] = {
            "goal_id": expected["goal_id"],
            "observed_index": observed_index,
            "expected_index": expected["goal_index"],
            "passed": passed,
        }
        if not passed:
            errors.append(f"task goal mismatch: {environment}")

    budget = manifest["missing_learning_budget"]
    matrix_jobs = sum(
        condition["new_learning_jobs"]
        for environment in manifest["environment_matrix"].values()
        for condition in environment["conditions"].values()
    )
    per_environment_jobs = sum(
        budget[name]["jobs"]
        for name in ("tick_claim", "pack_restore", "craft_remain")
    )
    budget_passed = (
        matrix_jobs == budget["total_jobs"]
        and per_environment_jobs == budget["total_jobs"]
        and budget["main_training_collection_slots"]
        == budget["total_jobs"] * budget["collector_slots_per_job"]
        == 6_710_886_400
        and budget["completed_smoke_collection_slots"] == 163_840
        and budget["planned_total_including_smoke"] == 6_711_050_240
        and budget["planned_total_including_smoke"]
        == budget["main_training_collection_slots"]
        + budget["completed_smoke_collection_slots"]
        and budget["smoke_does_not_reduce_main_updates"] is True
    )
    if not budget_passed:
        errors.append("missing-learning budget arithmetic failed")

    jobs = build_jobs(manifest)
    job_builder_passed = len(jobs) == 50 and not any(
        job["environment"] == "pack_restore" for job in jobs
    )
    if not job_builder_passed:
        errors.append("common job builder did not produce the exact 50-job matrix")

    panel_path = _resolve(manifest["common_trace_implementation"]["panel"])
    panel = json.loads(panel_path.read_text()) if panel_path.is_file() else {}
    cap = panel.get("evaluation_upper_bound", {})
    panel_passed = (
        cap.get("total_episodes") == 88200
        and cap.get("total_steps") == 11289600
        and manifest["common_trace_implementation"]["episode_cap"] == 88200
        and manifest["common_trace_implementation"]["step_cap"] == 11289600
        and manifest["common_trace_implementation"]["automatic_re_evaluation"] is False
        and _resolve(manifest["common_trace_implementation"]["orchestrator"]).is_file()
    )
    if not panel_passed:
        errors.append("common trace panel/cap contract failed")

    evidence_path = _resolve(manifest["evidence_freeze"]["manifest"] )
    evidence = json.loads(evidence_path.read_text()) if evidence_path.is_file() else {}
    evidence_passed = (
        len(evidence.get("checkpoint_index", {})) == 130
        and len(evidence.get("artifact_index", {})) == 123
        and evidence.get("preservation", {}).get("reset_stash_delete_used") is False
    )
    if not evidence_passed:
        errors.append("evidence freeze contract failed")

    smoke_path = _resolve(manifest["execution_implementation"]["smoke_receipt"])
    smoke = json.loads(smoke_path.read_text()) if smoke_path.is_file() else {}
    smoke_accounting = smoke.get("collection_accounting", {})
    smoke_passed = (
        smoke_accounting.get("started_attempts") == 5
        and smoke_accounting.get("passed_attempts") == 5
        and smoke_accounting.get("failed_attempts") == 0
        and smoke_accounting.get("slots_charged") == 163840
        and smoke_accounting.get("remaining_main_training_slots") == 6710886400
        and smoke_accounting.get("planned_total_including_smoke") == 6711050240
        and smoke_accounting.get("completed_smoke_collection_slots") == 163840
        and smoke.get("main_queue_launched") is False
        and len(smoke.get("checkpoint_receipts", [])) == 5
        and smoke.get("trace_evaluator_smoke", {}).get("episodes") == 4
        and smoke.get("trace_evaluator_smoke", {}).get("independent_replay_passed") == 4
        and all(
            trace.get("independent_replay_passed")
            and trace.get("transition_kernel_calls") == 0
            and trace.get("policy_or_teacher_calls") == 0
            for trace in smoke.get("trace_evaluator_smoke", {}).get("traces", [])
        )
    )
    if not smoke_passed:
        errors.append("integration smoke receipt failed")

    comparison_status = manifest.get("comparison_evidence_status", {})
    comparison_status_passed = (
        bool(comparison_status.get("new_same_code_and_initialization"))
        and bool(comparison_status.get("historical_equivalence_verified"))
        and bool(comparison_status.get("exploratory_due_to_unresolved_code_difference"))
        and "must not be called single-factor"
        in comparison_status.get("single_factor_claim_rule", "")
        and set(comparison_status.get(
            "substitute_evidence_without_independent_source_manifest", {}
        )) == {"tick_claim", "craft_remain"}
    )
    if not comparison_status_passed:
        errors.append("comparison evidence status contract failed")

    authorization = manifest["authorization"]
    authorization_passed = (
        authorization["read_only_audit"]
        and authorization["matrix_and_validator_implementation"]
        and authorization["missing_condition_runner_implementation"]
        and authorization["common_trace_evaluator_implementation"]
        and authorization["contract_smoke"]
        and authorization["smoke_checkpoint_writes"]
        and authorization["integration_smoke_complete"]
        and authorization["common_frozen_evaluation"]
        and authorization["main_training"]
        and authorization["main_training_checkpoint_writes"]
        and authorization["queue_or_watcher_launch"]
        and not authorization["dreamer_execution"]
    )
    if not authorization_passed:
        errors.append("authorization boundary failed")

    return {
        "schema_version": "hackrl_cross_environment_teacher_matrix_validation_v1",
        "manifest_id": manifest["manifest_id"],
        "artifact_checks": artifact_results,
        "aggregate_checks": aggregate_results,
        "source_hash_checks": hash_results,
        "task_goal_checks": goal_results,
        "budget": {
            "matrix_jobs": matrix_jobs,
            "manifest_total_jobs": budget["total_jobs"],
            "main_training_collection_slots": budget["main_training_collection_slots"],
            "completed_smoke_collection_slots": budget["completed_smoke_collection_slots"],
            "planned_total_including_smoke": budget["planned_total_including_smoke"],
            "passed": budget_passed,
        },
        "job_builder_passed": job_builder_passed,
        "trace_panel_passed": panel_passed,
        "evidence_freeze_passed": evidence_passed,
        "integration_smoke_passed": smoke_passed,
        "comparison_evidence_status_passed": comparison_status_passed,
        "authorization_passed": authorization_passed,
        "errors": errors,
        "outcome": (
            "main_training_and_common_evaluation_authorized" if not errors else "validation_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    result = validate(json.loads(args.manifest.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
