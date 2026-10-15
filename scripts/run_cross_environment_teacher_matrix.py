#!/usr/bin/env python3
"""Validated runner for the frozen cross-environment teacher matrix v1.

There is deliberately no queue mode.  Full jobs require an explicit command-line
acknowledgement, while ``--smoke`` advances five predeclared official jobs only
to adapt_1 so every collected slot remains resumable and inside the fixed cap.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import jax
import numpy as np

import run_dual_leo_compare as engine
from hackrl.craft_remain_gc import (
    NUM_ACTIONS as CRAFT_ACTIONS,
    NUM_GOALS as CRAFT_GOALS,
    CraftRemainGCActorCritic,
    CraftRemainGCConfig,
    _BRANCH_LOCKED_FIELDS as CRAFT_LOCKED,
    _batch_inputs as craft_inputs,
    command_deliver_3 as craft_command,
    config_from_craft_remain_gc_payload,
    craft_remain_gc_config_payload,
    craft_remain_gc_parameter_count,
    craft_remain_outcome,
    evaluate_craft_remain_gc_frozen,
    initialize_craft_remain_gc,
    load_craft_remain_gc_checkpoint,
    load_craft_remain_gc_history_branch,
    make_craft_remain_gc_update,
    save_craft_remain_gc_checkpoint,
)
from hackrl.cross_environment_teacher_matrix import (
    DEFAULT_MATRIX,
    ROOT,
    build_jobs,
    job_by_id,
    load_matrix,
    sha256_file,
)
from hackrl.dual_leo import load_dual_checkpoint


DEFAULT_RUN_ROOT = Path("/raid/ext_csv/HackRL/runs/cross_environment_teacher_matrix_v1")
SLOTS_PER_UPDATE = 512 * 64
SMOKE_SELECTION = (
    ("tick_claim", "D-task", "fixed", 20),
    ("tick_claim", "D-frozen", "mutant", 20),
    ("craft_remain", "D-task", "fixed", 20),
    ("craft_remain", "D-frozen", "mutant", 20),
    ("craft_remain", "D-off", "mutant", 20),
)
SCIENTIFIC_CODE = (
    "docs/manifests/cross_environment_teacher_matrix_v1.json",
    "scripts/run_cross_environment_teacher_matrix.py",
    "scripts/run_dual_leo_compare.py",
    "src/hackrl/cross_environment_teacher_matrix.py",
    "src/hackrl/dual_leo.py",
    "src/hackrl/tick_claim.py",
    "src/hackrl/tick_claim_gc.py",
    "src/hackrl/craft_remain_env.py",
    "src/hackrl/craft_remain_gc.py",
)


def _register_craft() -> None:
    engine.ENVS["craft"] = {
        "config": CraftRemainGCConfig,
        "actor": CraftRemainGCActorCritic,
        "initialize": initialize_craft_remain_gc,
        "gc_update": make_craft_remain_gc_update,
        "save": save_craft_remain_gc_checkpoint,
        "load": load_craft_remain_gc_checkpoint,
        "branch": load_craft_remain_gc_history_branch,
        "command": craft_command,
        "evaluate": evaluate_craft_remain_gc_frozen,
        "inputs": craft_inputs,
        "goals": CRAFT_GOALS,
        "actions": CRAFT_ACTIONS,
        "payload": craft_remain_gc_config_payload,
        "from_payload": config_from_craft_remain_gc_payload,
        "parameters": craft_remain_gc_parameter_count,
        "locked": CRAFT_LOCKED,
        "outcome": craft_remain_outcome,
        "families": ("natural_reset", "path_check"),
    }


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


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _assert_frozen_sources() -> dict:
    head = _git("rev-parse", "HEAD")
    dirty = subprocess.run(
        ("git", "diff", "--quiet", "--", *SCIENTIFIC_CODE), cwd=ROOT
    ).returncode
    untracked = [
        path for path in SCIENTIFIC_CODE
        if subprocess.run(("git", "ls-files", "--error-unmatch", path), cwd=ROOT,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    ]
    if dirty or untracked:
        raise RuntimeError(
            "scientific sources must be committed before checkpoint writes; "
            f"dirty={bool(dirty)} untracked={untracked}"
        )
    return {
        "git_commit": head,
        "scientific_source_sha256": {
            path: sha256_file(ROOT / path) for path in SCIENTIFIC_CODE
        },
    }


def _capacity_snapshot(run_root: Path, jobs: list[dict]) -> dict:
    home = shutil.disk_usage("/home/ext_csv")
    raid = shutil.disk_usage("/raid/ext_csv")
    source_sizes = [_directory_bytes(Path(job["pretrain_checkpoint"])) for job in jobs]
    checkpoint_upper = max(source_sizes)
    retained_final = checkpoint_upper * len(jobs)
    predecessor_successor_overlap = checkpoint_upper * len(jobs)
    evaluation_artifacts = 24 * 1024**3
    logs_and_curves = 8 * 1024**3
    projected_remaining_writes = (
        retained_final + predecessor_successor_overlap + evaluation_artifacts + logs_and_curves
    )
    reserve = max(8 * 1024**3, int(np.ceil(0.20 * projected_remaining_writes)))
    passed = raid.free >= projected_remaining_writes + reserve
    record = {
        "recorded_utc": _utc(),
        "intended_filesystem": "/raid/ext_csv",
        "run_root": str(run_root),
        "existing_run_bytes": _directory_bytes(run_root),
        "home": {"total": home.total, "used": home.used, "free": home.free},
        "raid": {"total": raid.total, "used": raid.used, "free": raid.free},
        "source_checkpoint_bytes": {
            "minimum": min(source_sizes),
            "maximum": checkpoint_upper,
            "mean": int(np.mean(source_sizes)),
        },
        "forecast": {
            "retained_final_checkpoints": retained_final,
            "active_predecessor_successor_overlap": predecessor_successor_overlap,
            "evaluation_artifacts_allowance": evaluation_artifacts,
            "logs_and_curves_allowance": logs_and_curves,
            "projected_remaining_writes": projected_remaining_writes,
            "safety_reserve": reserve,
            "required_free": projected_remaining_writes + reserve,
        },
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"RAID capacity forecast failed: {record}")
    return record


def _tree_digest(tree: object) -> str:
    digest = hashlib.sha256()
    for index, leaf in enumerate(jax.tree.leaves(tree)):
        array = np.asarray(jax.device_get(leaf))
        digest.update(str(index).encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _load_smoke_checkpoint(job: dict, checkpoint: Path):
    spec = engine.ENVS[job["env"]]
    config = engine._job_config(
        job, goal_mode="deliver_3", variant=job["variant"], updates=engine.ADAPT_UPDATES
    )
    _, template = spec["initialize"](config)
    _, leo_template, _ = engine._start_teacher(spec, config, template)
    return load_dual_checkpoint(checkpoint, template, leo_template)


def _teacher_state_digests(leo: object) -> dict:
    return {
        "parameters": _tree_digest(leo.params),
        "adam_state": _tree_digest(leo.opt_state),
        "batch_renorm_statistics": _tree_digest(leo.batch_stats),
        "optimizer_step": _tree_digest(leo.step),
        "all_teacher_state": _tree_digest(leo),
    }


def _last_metrics(cell: Path) -> dict:
    rows = [json.loads(line) for line in (cell / "updates.jsonl").read_text().splitlines() if line]
    rows = [row for row in rows if int(row["adaptation_update"]) == 1]
    if not rows:
        raise RuntimeError(f"no adapt_1 metric row: {cell}")
    return rows[-1]


def _check_update_contract(job: dict, cell: Path, metrics: dict) -> dict:
    config = json.loads((cell / "checkpoints/adapt_1/config.json").read_text())
    if config["variant"] != job["variant"]:
        raise RuntimeError(f"wrong kernel in saved config: {job['id']}")
    arm = json.loads((cell / "checkpoints/adapt_1/arm.json").read_text())
    expected_arm = engine._arm(job)
    if arm != expected_arm:
        raise RuntimeError(f"arm mismatch: {arm} != {expected_arm}")
    checks = {
        "saved_kernel_matches_job": True,
        "saved_arm_matches_job": True,
        "learn_teacher_metric": int(metrics["learn_teacher"]),
        "imitate_teacher_metric": int(metrics["imitate_teacher"]),
        "bc_policy_coef": float(metrics["bc_policy_coef"]),
        "teacher_target_head_count": int(metrics["teacher_target_head_count"]),
        "teacher_target_terms": int(metrics["teacher_target_terms"]),
        "teacher_valid_samples": int(metrics["teacher_valid_samples"]),
        "teacher_applied_minibatches": int(metrics["teacher_applied_minibatches"]),
    }
    if job["condition"] == "D-task":
        if not (
            checks["learn_teacher_metric"] == 1
            and checks["imitate_teacher_metric"] == 1
            and checks["teacher_target_head_count"] == 1
            and checks["teacher_target_terms"] == checks["teacher_valid_samples"]
            and checks["teacher_applied_minibatches"] > 0
        ):
            raise RuntimeError(f"D-task update contract failed: {checks}")
        checks["task_loss_scaling"] = "one selected head; target terms equal valid samples"
    elif job["condition"] == "D-frozen":
        before = _load_smoke_checkpoint(job, cell / "checkpoints/adapt_0")[1]
        after = _load_smoke_checkpoint(job, cell / "checkpoints/adapt_1")[1]
        before_digests = _teacher_state_digests(before)
        after_digests = _teacher_state_digests(after)
        if not (
            checks["learn_teacher_metric"] == 0
            and checks["imitate_teacher_metric"] == 1
            and checks["teacher_applied_minibatches"] == 0
            and before_digests == after_digests
        ):
            raise RuntimeError("D-frozen changed teacher state")
        checks["teacher_before_sha256"] = before_digests
        checks["teacher_after_sha256"] = after_digests
        checks["all_teacher_state_frozen"] = True
    elif job["condition"] == "D-off":
        if not (
            checks["learn_teacher_metric"] == 1
            and checks["imitate_teacher_metric"] == 0
            and checks["bc_policy_coef"] == 0.0
            and checks["teacher_applied_minibatches"] > 0
        ):
            raise RuntimeError(f"D-off update contract failed: {checks}")
        checks["policy_imitation_disabled"] = True
    return checks



def _reuse_update_zero_curve(job: dict, run_root: Path) -> dict:
    if job["environment"] == "tick_claim":
        source = (
            ROOT / "runs/dual_leo_compare_v1/tick/dual"
            / job["variant"] / f"seed{job['seed']}/curve/adapt_0.json"
        )
    elif job["environment"] == "craft_remain":
        source = (
            ROOT / "runs/craft_remain_compare_v1/dual"
            / job["variant"] / f"seed{job['seed']}/curve/adapt_0.json"
        )
    else:
        raise ValueError(job["environment"])
    cell = engine._cell_dir(run_root, job)
    destination = cell / "curve/adapt_0.json"
    provenance_path = cell / "curve/adapt_0.provenance.json"
    source_hash = sha256_file(source)
    if destination.is_file():
        if sha256_file(destination) != source_hash:
            raise RuntimeError(f"reused update-zero curve changed: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    provenance = {
        "schema_version": "hackrl_reused_evaluation_provenance_v1",
        "origin": "reused_result",
        "reason": "same frozen source actor before any condition-specific adaptation update",
        "source": str(source.resolve()),
        "source_sha256": source_hash,
        "destination": str(destination),
        "destination_sha256": sha256_file(destination),
        "new_evaluation_episodes": 0,
        "new_evaluation_steps": 0,
    }
    if provenance_path.is_file():
        if json.loads(provenance_path.read_text()) != provenance:
            raise RuntimeError(f"update-zero provenance changed: {provenance_path}")
    else:
        _write_json(provenance_path, provenance)
    return provenance

def _smoke_jobs(jobs: list[dict]) -> list[dict]:
    lookup = {
        (job["environment"], job["condition"], job["variant"], job["seed"]): job
        for job in jobs
    }
    selected = [lookup[key] for key in SMOKE_SELECTION]
    if len(selected) != 5:
        raise RuntimeError("smoke selection drift")
    return selected


def run_smoke(matrix: dict, run_root: Path) -> dict:
    _register_craft()
    jobs = build_jobs(matrix)
    provenance = _assert_frozen_sources()
    capacity = _capacity_snapshot(run_root, jobs)
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "capacity_before_smoke.json", capacity)
    _write_json(
        run_root / "run_manifest.json",
        {
            "schema_version": "hackrl_cross_environment_teacher_matrix_run_v1",
            "scientific_result": False,
            "matrix": str(DEFAULT_MATRIX),
            "matrix_sha256": sha256_file(DEFAULT_MATRIX),
            "total_main_jobs": len(jobs),
            "main_queue_launched": False,
            "collection_cap_slots": int(matrix["missing_learning_budget"]["total_collector_slots"]),
            **provenance,
        },
    )
    records = []
    for job in _smoke_jobs(jobs):
        cell = engine._cell_dir(run_root, job)
        _reuse_update_zero_curve(job, run_root)
        already = engine._meta_update(cell / "checkpoints/adapt_1") == engine.PRETRAIN_UPDATES + 1
        attempt = {
            "recorded_utc": _utc(),
            "job": job["id"],
            "environment": job["environment"],
            "condition": job["condition"],
            "variant": job["variant"],
            "target_adaptation_update": 1,
            "attempt_kind": "resume_validation" if already else "new_collection",
            "slots_charged": 0 if already else SLOTS_PER_UPDATE,
            "status": "started",
        }
        _append_jsonl(run_root / "collection_ledger.jsonl", attempt)
        started = time.perf_counter()
        try:
            if not already:
                engine.run_adapt(
                    run_root,
                    job,
                    target_adapt_updates=1,
                    write_science_curves=False,
                )
            metrics = _last_metrics(cell)
            checks = _check_update_contract(job, cell, metrics)
            record = attempt | {
                "status": "passed",
                "elapsed_seconds": time.perf_counter() - started,
                "checks": checks,
                "checkpoint": str(cell / "checkpoints/adapt_1"),
            }
        except Exception as error:
            record = attempt | {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - started,
                "error": repr(error),
            }
            _append_jsonl(run_root / "collection_ledger.jsonl", record)
            raise
        _append_jsonl(run_root / "collection_ledger.jsonl", record)
        records.append(record)
    newly_charged = sum(row["slots_charged"] for row in records)
    total_cap = int(matrix["missing_learning_budget"]["total_collector_slots"])
    result = {
        "schema_version": "hackrl_cross_environment_teacher_matrix_smoke_v1",
        "scientific_performance_result": False,
        "recorded_utc": _utc(),
        "git_commit": provenance["git_commit"],
        "jobs": records,
        "smoke_jobs": len(records),
        "new_slots_charged_this_invocation": newly_charged,
        "official_job_slots_completed": 5 * SLOTS_PER_UPDATE,
        "collection_cap_slots": total_cap,
        "remaining_official_collection_slots": total_cap - 5 * SLOTS_PER_UPDATE,
        "main_queue_launched": False,
        "passed": all(row["status"] == "passed" for row in records),
    }
    _write_json(run_root / "smoke.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", action="store_true")
    group.add_argument("--list-jobs", action="store_true")
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--run-job")
    parser.add_argument("--authorize-main-training", action="store_true")
    args = parser.parse_args()
    matrix = load_matrix(args.matrix)
    _register_craft()
    jobs = build_jobs(matrix)
    if args.validate:
        print(json.dumps({"jobs": len(jobs), "matrix_sha256": sha256_file(args.matrix)}, indent=2))
        return
    if args.list_jobs:
        print(json.dumps(jobs, indent=2, sort_keys=True))
        return
    if args.smoke:
        print(json.dumps(run_smoke(matrix, args.run_root), indent=2, sort_keys=True))
        return
    if not args.authorize_main_training:
        raise SystemExit("--run-job requires --authorize-main-training; no queue mode exists")
    _assert_frozen_sources()
    capacity = _capacity_snapshot(args.run_root, jobs)
    args.run_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.run_root / "capacity_before_main_job.json", capacity)
    job = job_by_id(args.run_job, matrix)
    _reuse_update_zero_curve(job, args.run_root)
    engine.run_adapt(args.run_root, job)


if __name__ == "__main__":
    main()
