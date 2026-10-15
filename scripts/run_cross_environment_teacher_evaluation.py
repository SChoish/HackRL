#!/usr/bin/env python3
"""Run the frozen common teacher-matrix evaluation exactly once per task."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

import eval_cross_environment_teacher_traces as trace_eval
import run_dual_leo_compare as engine
from hackrl.cross_environment_teacher_matrix import (
    DEFAULT_MATRIX,
    ROOT,
    build_jobs,
    load_matrix,
    sha256_file,
)
from hackrl.cross_environment_teacher_trace import FAMILIES


DEFAULT_MAIN_RUN_ROOT = Path("/raid/ext_csv/HackRL/runs/cross_environment_teacher_matrix_v1_main")
DEFAULT_OUTPUT_ROOT = Path("/raid/ext_csv/HackRL/evaluations/cross_environment_teacher_matrix_v1")
DEFAULT_PANEL = ROOT / "docs/manifests/cross_environment_teacher_trace_panel_v1.json"
EVALUATION_CODE = (
    "docs/manifests/cross_environment_teacher_matrix_v1.json",
    "docs/manifests/cross_environment_teacher_trace_panel_v1.json",
    "scripts/eval_cross_environment_teacher_traces.py",
    "scripts/run_cross_environment_teacher_evaluation.py",
    "src/hackrl/cross_environment_teacher_matrix.py",
    "src/hackrl/cross_environment_teacher_trace.py",
    "src/hackrl/dual_leo.py",
    "src/hackrl/tick_claim.py",
    "src/hackrl/pack_restore.py",
    "src/hackrl/craft_remain_env.py",
)


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _assert_frozen_evaluator() -> dict:
    dirty = subprocess.run(
        ("git", "diff", "--quiet", "--", *EVALUATION_CODE), cwd=ROOT
    ).returncode
    staged = subprocess.run(
        ("git", "diff", "--cached", "--quiet", "--", *EVALUATION_CODE), cwd=ROOT
    ).returncode
    untracked = [
        path for path in EVALUATION_CODE
        if subprocess.run(
            ("git", "ls-files", "--error-unmatch", path), cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode
    ]
    if dirty or staged or untracked:
        raise RuntimeError(
            "evaluation sources must be committed before scientific evaluation; "
            f"dirty={bool(dirty)} staged={bool(staged)} untracked={untracked}"
        )
    return {
        "execution_commit": _git("rev-parse", "HEAD"),
        "evaluation_source_sha256": {
            path: sha256_file(ROOT / path) for path in EVALUATION_CODE
        },
    }


def _reused_checkpoint(environment: str, condition: str, seed: int, variant: str | None) -> Path:
    if environment == "tick_claim":
        base = ROOT / "runs"
        if condition == "source":
            return base / f"dual_leo_compare_v1/tick/dual/pretrain/seed{seed}/checkpoints/update_512"
        if condition == "D-on":
            return base / f"dual_leo_compare_v1/tick/dual/{variant}/seed{seed}/checkpoints/adapt_4096"
        if condition == "D-off":
            return base / f"dual_adapt_bc_off_v1/tick/adapt_bc_off/{variant}/seed{seed}/checkpoints/adapt_4096"
    elif environment == "craft_remain":
        base = ROOT / "runs/craft_remain_compare_v1/dual"
        if condition == "source":
            return base / f"pretrain/seed{seed}/checkpoints/update_512"
        if condition == "D-on":
            return base / f"{variant}/seed{seed}/checkpoints/adapt_4096"
    elif environment == "pack_restore":
        base = Path("/raid/ext_csv/HackRL/runs")
        if condition == "source":
            return base / f"pack_restore_scale_bc_v1/size_s/pack/dual/pretrain/seed{seed}/checkpoints/update_512"
        roots = {
            "D-on": "pack_restore_scale_bc_v1/size_s/pack/dual",
            "D-off": "pack_restore_scale_bc_v1/size_s/pack/dual_bc_off",
            "D-task": "pack_restore_delivery_only_teacher_v1/size_s/pack/dual_teacher_delivery_only",
            "D-frozen": "pack_restore_pretrained_teacher_freeze_v1/size_s/pack/dual_teacher_frozen",
        }
        if condition in roots:
            return base / roots[condition] / f"{variant}/seed{seed}/checkpoints/adapt_4096"
    raise KeyError((environment, condition, seed, variant))


def build_policy_inventory(matrix: dict, main_run_root: Path) -> list[dict]:
    new_jobs = {
        (job["environment"], job["condition"], int(job["seed"]), job["variant"]): job
        for job in build_jobs(matrix)
    }
    policies: list[dict] = []
    for environment, environment_spec in matrix["environment_matrix"].items():
        for seed in environment_spec["learner_seeds"]:
            source = _reused_checkpoint(environment, "source", int(seed), None)
            policies.append({
                "id": f"{environment}-source-s{seed}",
                "environment": environment,
                "condition": "source",
                "seed": int(seed),
                "trained_variant": None,
                "checkpoint_origin": "reused_result",
                "checkpoint": str(source),
                "code_provenance": "historical_source_checkpoint_reused",
            })
            for condition, condition_spec in environment_spec["conditions"].items():
                if condition == "source":
                    continue
                for variant in ("fixed", "mutant"):
                    key = (environment, condition, int(seed), variant)
                    if condition_spec["new_learning_jobs"]:
                        job = new_jobs[key]
                        checkpoint = engine._cell_dir(main_run_root, job) / "checkpoints/adapt_4096"
                        origin = "new_matrix_result"
                        provenance = "new_frozen_execution_code"
                    else:
                        checkpoint = _reused_checkpoint(environment, condition, int(seed), variant)
                        origin = "reused_result"
                        provenance = "historical_code_see_comparison_evidence_status"
                    policies.append({
                        "id": f"{environment}-{condition}-{variant}-s{seed}",
                        "environment": environment,
                        "condition": condition,
                        "seed": int(seed),
                        "trained_variant": variant,
                        "checkpoint_origin": origin,
                        "checkpoint": str(checkpoint),
                        "code_provenance": provenance,
                    })
    counts = Counter(policy["environment"] for policy in policies)
    if counts != {"tick_claim": 45, "pack_restore": 90, "craft_remain": 45}:
        raise RuntimeError(f"policy inventory drift: {counts}")
    if len({policy["id"] for policy in policies}) != 180:
        raise RuntimeError("policy id collision")
    return policies


def build_tasks(policies: list[dict]) -> list[dict]:
    tasks = []
    for policy in policies:
        for kernel in ("fixed", "mutant"):
            for view in ("mode", "sample"):
                for family in FAMILIES[policy["environment"]]:
                    tasks.append({
                        "id": f"{policy['id']}__{kernel}-{view}-{family}",
                        "policy_id": policy["id"],
                        "kernel": kernel,
                        "view": view,
                        "family": family,
                    })
    if len(tasks) != 1440:
        raise RuntimeError(f"evaluation task drift: {len(tasks)}")
    return tasks


def _artifact_receipt(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _verify_completed_task(receipt_path: Path) -> dict:
    receipt = json.loads(receipt_path.read_text())
    for artifact in receipt["artifacts"].values():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(
                f"recorded evaluation artifact is missing or changed; do not auto re-evaluate: {path}"
            )
    return receipt


def _ensure_evaluation_manifest(
    matrix: dict,
    output_root: Path,
    main_run_root: Path,
    policies: list[dict],
    tasks: list[dict],
    provenance: dict,
) -> dict:
    panel = json.loads(DEFAULT_PANEL.read_text())
    payload = {
        "schema_version": "hackrl_cross_environment_teacher_evaluation_run_v1",
        "scientific_result": True,
        "understood_as": (
            "Evaluate the frozen 180-policy inventory on every predeclared kernel, view, and "
            "start family after all 50 main jobs complete; never auto re-evaluate missing records."
        ),
        "recorded_utc": _utc(),
        "execution_commit": provenance["execution_commit"],
        "evaluation_source_sha256": provenance["evaluation_source_sha256"],
        "matrix": str(DEFAULT_MATRIX),
        "matrix_sha256": sha256_file(DEFAULT_MATRIX),
        "panel": str(DEFAULT_PANEL),
        "panel_sha256": sha256_file(DEFAULT_PANEL),
        "main_run_root": str(main_run_root),
        "output_root": str(output_root),
        "policies": policies,
        "policy_count": len(policies),
        "task_count": len(tasks),
        "episode_cap": panel["evaluation_upper_bound"]["total_episodes"],
        "step_cap": panel["evaluation_upper_bound"]["total_steps"],
        "automatic_re_evaluation": False,
    }
    path = output_root / "evaluation_manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text())
        comparable = dict(existing)
        comparable["recorded_utc"] = payload["recorded_utc"]
        if comparable != payload:
            raise RuntimeError(f"evaluation manifest drift: {path}")
    else:
        _write_json(path, payload)
    return payload


def validate_ready(matrix: dict, main_run_root: Path) -> dict:
    policies = build_policy_inventory(matrix, main_run_root)
    missing = [policy["checkpoint"] for policy in policies if not Path(policy["checkpoint"]).is_dir()]
    tasks = build_tasks(policies)
    panel = json.loads(DEFAULT_PANEL.read_text())["evaluation_upper_bound"]
    return {
        "policies": len(policies),
        "tasks": len(tasks),
        "missing_checkpoints": missing,
        "episode_cap": panel["total_episodes"],
        "step_cap": panel["total_steps"],
        "ready": not missing,
    }


def run_all(matrix: dict, main_run_root: Path, output_root: Path) -> dict:
    provenance = _assert_frozen_evaluator()
    policies = build_policy_inventory(matrix, main_run_root)
    tasks = build_tasks(policies)
    readiness = validate_ready(matrix, main_run_root)
    if not readiness["ready"]:
        raise RuntimeError(f"main training incomplete; missing checkpoints: {readiness['missing_checkpoints']}")
    manifest = _ensure_evaluation_manifest(
        matrix, output_root, main_run_root, policies, tasks, provenance
    )
    task_lookup = {task["policy_id"]: [] for task in tasks}
    for task in tasks:
        task_lookup[task["policy_id"]].append(task)
    total_episodes = 0
    total_steps = 0
    completed = 0
    for policy in policies:
        checkpoint = Path(policy["checkpoint"])
        fingerprint = trace_eval._checkpoint_fingerprint(checkpoint)
        actor_bundle = None
        for task in task_lookup[policy["id"]]:
            receipt_path = output_root / "task_receipts" / f"{task['id']}.json"
            if receipt_path.is_file():
                receipt = _verify_completed_task(receipt_path)
                total_episodes += int(receipt["episodes"])
                total_steps += int(receipt["steps"])
                completed += 1
                continue
            attempt = {
                "recorded_utc": _utc(),
                "task": task["id"],
                "policy": policy["id"],
                "status": "started",
            }
            _append_jsonl(output_root / "evaluation_ledger.jsonl", attempt)
            started = time.perf_counter()
            try:
                if actor_bundle is None:
                    actor_bundle = trace_eval._load_actor(policy["environment"], checkpoint)
                summary = trace_eval.evaluate(
                    environment=policy["environment"],
                    checkpoint=checkpoint,
                    checkpoint_origin=policy["checkpoint_origin"],
                    learner_seed=policy["seed"],
                    variant=task["kernel"],
                    view=task["view"],
                    family=task["family"],
                    matrix_path=DEFAULT_MATRIX,
                    panel_path=DEFAULT_PANEL,
                    output_root=output_root,
                    scientific=True,
                    actor_bundle=actor_bundle,
                    checkpoint_fingerprint=fingerprint,
                )
                trace_path = Path(summary["trace"])
                summary_path = trace_path.with_name("summary.json")
                replay_path = trace_path.with_name("independent_replay.json")
                receipt = attempt | {
                    "recorded_utc": _utc(),
                    "status": "passed",
                    "elapsed_seconds": time.perf_counter() - started,
                    "episodes": int(summary["episodes"]),
                    "steps": int(summary["steps"]),
                    "artifacts": {
                        "trace": _artifact_receipt(trace_path),
                        "summary": _artifact_receipt(summary_path),
                        "independent_replay": _artifact_receipt(replay_path),
                    },
                }
                _write_json(receipt_path, receipt)
                _append_jsonl(output_root / "evaluation_ledger.jsonl", receipt)
            except BaseException as error:
                failed = attempt | {
                    "recorded_utc": _utc(),
                    "status": "failed",
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": repr(error),
                    "automatic_re_evaluation": False,
                }
                _append_jsonl(output_root / "evaluation_ledger.jsonl", failed)
                raise
            total_episodes += int(receipt["episodes"])
            total_steps += int(receipt["steps"])
            completed += 1
    result = {
        "schema_version": "hackrl_cross_environment_teacher_evaluation_receipt_v1",
        "recorded_utc": _utc(),
        "execution_commit": manifest["execution_commit"],
        "matrix_sha256": manifest["matrix_sha256"],
        "panel_sha256": manifest["panel_sha256"],
        "policies": len(policies),
        "tasks": len(tasks),
        "completed_tasks": completed,
        "episodes": total_episodes,
        "steps": total_steps,
        "episode_cap": manifest["episode_cap"],
        "step_cap": manifest["step_cap"],
        "passed": completed == len(tasks) and total_episodes <= manifest["episode_cap"] and total_steps <= manifest["step_cap"],
        "automatic_re_evaluation": False,
    }
    _write_json(output_root / "execution_receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--main-run-root", type=Path, default=DEFAULT_MAIN_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", action="store_true")
    group.add_argument("--list-policies", action="store_true")
    group.add_argument("--run-all", action="store_true")
    parser.add_argument("--authorize-common-evaluation", action="store_true")
    args = parser.parse_args()
    matrix = load_matrix(args.matrix)
    if args.validate:
        print(json.dumps(validate_ready(matrix, args.main_run_root), indent=2, sort_keys=True))
        return
    policies = build_policy_inventory(matrix, args.main_run_root)
    if args.list_policies:
        print(json.dumps(policies, indent=2, sort_keys=True))
        return
    if not args.authorize_common_evaluation:
        raise SystemExit("--run-all requires --authorize-common-evaluation")
    print(json.dumps(run_all(matrix, args.main_run_root, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
