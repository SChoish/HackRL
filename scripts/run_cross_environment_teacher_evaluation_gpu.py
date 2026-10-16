#!/usr/bin/env python3
"""Run the frozen common teacher-matrix evaluation with GPU-batched traces."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import gzip
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import eval_cross_environment_teacher_traces as base_trace_eval
import eval_cross_environment_teacher_traces_gpu as trace_eval
import run_dual_leo_compare as engine
from hackrl.cross_environment_teacher_matrix import (
    DEFAULT_MATRIX,
    ROOT,
    build_jobs,
    load_matrix,
    sha256_file,
)
from hackrl.cross_environment_teacher_trace import FAMILIES, HORIZON


DEFAULT_MAIN_RUN_ROOT = Path("/raid/ext_csv/HackRL/runs/cross_environment_teacher_matrix_v1_main")
DEFAULT_OUTPUT_ROOT = Path("/raid/ext_csv/HackRL/evaluations/cross_environment_teacher_matrix_v1_gpu")
DEFAULT_PANEL = ROOT / "docs/manifests/cross_environment_teacher_trace_panel_v1.json"
VALIDATION_RECEIPT = (
    ROOT / "docs/manifests/cross_environment_teacher_gpu_evaluator_validation_v1.json"
)
TRACE_BYTES_PER_STEP_FORECAST = 800
EVALUATION_FIXED_OVERHEAD_BYTES = 1024**3
SAFETY_RESERVE_BYTES = 8 * 1024**3
EVALUATION_CODE = (
    "docs/manifests/cross_environment_teacher_matrix_v1.json",
    "docs/manifests/cross_environment_teacher_trace_panel_v1.json",
    "docs/manifests/cross_environment_teacher_gpu_evaluator_validation_v1.json",
    "scripts/eval_cross_environment_teacher_traces.py",
    "scripts/eval_cross_environment_teacher_traces_gpu.py",
    "scripts/run_cross_environment_teacher_evaluation_gpu.py",
    "scripts/run_dual_leo_compare.py",
    "scripts/validate_cross_environment_teacher_gpu_evaluator.py",
    "src/hackrl/cross_environment_teacher_matrix.py",
    "src/hackrl/cross_environment_teacher_trace.py",
    "src/hackrl/dual_leo.py",
    "src/hackrl/tick_claim.py",
    "src/hackrl/tick_claim_gc.py",
    "src/hackrl/tick_claim_oracle.py",
    "src/hackrl/pack_restore.py",
    "src/hackrl/pack_restore_gc.py",
    "src/hackrl/craft_remain_env.py",
)


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _assert_gpu_backend() -> dict:
    devices = trace_eval.jax.devices()
    platforms = sorted({device.platform for device in devices})
    backend = trace_eval.jax.default_backend()
    if backend != "gpu" or platforms != ["gpu"]:
        raise RuntimeError(
            f"scientific GPU evaluation requires only GPU devices: "
            f"backend={backend!r} platforms={platforms!r}"
        )
    return {
        "default_backend": backend,
        "devices": [str(device) for device in devices],
    }


def _assert_validation_receipt() -> dict:
    receipt = json.loads(VALIDATION_RECEIPT.read_text())
    if receipt.get("passed") is not True:
        raise RuntimeError("GPU evaluator validation receipt did not pass")
    if receipt.get("scientific_performance_result") is not False:
        raise RuntimeError("GPU evaluator validation is not marked as smoke-only")
    expected = {
        "scripts/eval_cross_environment_teacher_traces.py": sha256_file(
            ROOT / "scripts/eval_cross_environment_teacher_traces.py"
        ),
        "scripts/eval_cross_environment_teacher_traces_gpu.py": sha256_file(
            ROOT / "scripts/eval_cross_environment_teacher_traces_gpu.py"
        ),
    }
    if receipt.get("validated_source_sha256") != expected:
        raise RuntimeError("GPU evaluator changed after validation receipt")
    for environment in ("tick_claim", "pack_restore", "craft_remain"):
        row = receipt.get("environments", {}).get(environment, {})
        if not (
            row.get("passed") is True
            and row.get("actual_defect_trigger_observed") is True
            and row.get("non_logit_records_exact") is True
            and row.get("actions_equal") is True
        ):
            raise RuntimeError(f"incomplete GPU validation coverage: {environment}")
    return {
        "path": str(VALIDATION_RECEIPT),
        "sha256": sha256_file(VALIDATION_RECEIPT),
    }


def _capacity_preflight(output_root: Path, step_cap: int) -> dict:
    target = output_root
    while not target.exists():
        target = target.parent
    usage = shutil.disk_usage(target)
    existing_bytes = (
        sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file())
        if output_root.exists() else 0
    )
    projected_total_bytes = (
        int(step_cap) * TRACE_BYTES_PER_STEP_FORECAST
        + EVALUATION_FIXED_OVERHEAD_BYTES
    )
    projected_remaining_bytes = max(projected_total_bytes - existing_bytes, 0)
    safety_reserve_bytes = max(
        SAFETY_RESERVE_BYTES,
        (projected_remaining_bytes + 4) // 5,
    )
    passed = usage.free >= projected_remaining_bytes + safety_reserve_bytes
    result = {
        "target_filesystem": str(target),
        "free_bytes": usage.free,
        "existing_evaluation_bytes": existing_bytes,
        "projected_total_bytes": projected_total_bytes,
        "projected_remaining_bytes": projected_remaining_bytes,
        "safety_reserve_bytes": safety_reserve_bytes,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"insufficient evaluation capacity: {result}")
    return result


def _checkpoint_complete(path: Path, checkpoint_origin: str) -> bool:
    required = ["state.msgpack", "config.json", "metadata.json"]
    if checkpoint_origin == "new_matrix_result":
        required.append("arm.json")
    return path.is_dir() and all((path / name).is_file() for name in required)


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
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _expected_task_episodes(task: dict) -> int:
    panel = json.loads(DEFAULT_PANEL.read_text())
    starts = int(
        panel["families"][task["environment"]][task["family"]]["count"]
    )
    repeats = 1 if task["view"] == "mode" else 4
    return starts * repeats


def _verify_completed_task(
    receipt_path: Path,
    task: dict,
    policy: dict,
    manifest: dict,
) -> dict:
    receipt = json.loads(receipt_path.read_text())
    if not (
        receipt_path.stem == task["id"]
        and receipt.get("status") == "passed"
        and receipt.get("task") == task["id"]
        and receipt.get("policy") == policy["id"]
        and receipt.get("evaluation_execution_commit")
        == manifest["evaluation_execution_commit"]
        and receipt.get("matrix_sha256") == manifest["matrix_sha256"]
        and receipt.get("panel_sha256") == manifest["panel_sha256"]
        and receipt.get("evaluation_backend") == manifest["evaluation_backend"]
    ):
        raise RuntimeError(f"task receipt provenance drift: {receipt_path}")
    expected_episodes = _expected_task_episodes(
        task | {"environment": policy["environment"]}
    )
    episodes = int(receipt.get("episodes", -1))
    steps = int(receipt.get("steps", -1))
    if episodes != expected_episodes or not 0 <= steps <= episodes * HORIZON:
        raise RuntimeError(f"task accounting drift: {receipt_path}")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "trace", "summary", "independent_replay"
    }:
        raise RuntimeError(f"task artifact set drift: {receipt_path}")
    output_root = Path(manifest["output_root"]).resolve()
    resolved = {}
    for name, artifact in artifacts.items():
        path = Path(artifact["path"])
        try:
            path.resolve().relative_to(output_root)
        except ValueError as error:
            raise RuntimeError(f"artifact outside output root: {path}") from error
        if (
            not path.is_file()
            or path.stat().st_size != int(artifact["bytes"])
            or sha256_file(path) != artifact["sha256"]
        ):
            raise RuntimeError(
                "recorded evaluation artifact is missing or changed; "
                f"do not auto re-evaluate: {path}"
            )
        resolved[name] = path
    summary = json.loads(resolved["summary"].read_text())
    replay = json.loads(resolved["independent_replay"].read_text())
    if not (
        summary.get("scientific_performance_result") is True
        and summary.get("policy_id") == policy["id"]
        and summary.get("evaluation_backend") == manifest["evaluation_backend"]
        and int(summary.get("episodes", -1)) == episodes
        and int(summary.get("steps", -1)) == steps
        and Path(summary.get("trace", "")).resolve() == resolved["trace"].resolve()
        and replay.get("passed") is True
        and int(replay.get("episodes_checked", -1)) == episodes
        and int(replay.get("steps_checked", -1)) == steps
    ):
        raise RuntimeError(f"task summary or replay drift: {receipt_path}")
    with gzip.open(resolved["trace"], "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
    if not (
        header.get("policy_id") == policy["id"]
        and header.get("matrix_sha256") == manifest["matrix_sha256"]
        and header.get("panel_sha256") == manifest["panel_sha256"]
        and header.get("evaluation_backend") == manifest["evaluation_backend"]
        and header.get("scientific_performance_result") is True
    ):
        raise RuntimeError(f"trace header provenance drift: {receipt_path}")
    return receipt


def _checkpoint_execution_provenance(main_run_root: Path) -> dict:
    path = main_run_root / "run_manifest.json"
    payload = json.loads(path.read_text())
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "execution_commit": payload["execution_commit"],
        "matrix_sha256": payload["matrix_sha256"],
    }


def _ensure_evaluation_manifest(
    matrix: dict,
    output_root: Path,
    main_run_root: Path,
    policies: list[dict],
    tasks: list[dict],
    provenance: dict,
) -> dict:
    panel = json.loads(DEFAULT_PANEL.read_text())
    checkpoint_provenance = _checkpoint_execution_provenance(main_run_root)
    validation = _assert_validation_receipt()
    payload = {
        "schema_version": "hackrl_cross_environment_teacher_gpu_evaluation_run_v1",
        "scientific_result": True,
        "understood_as": (
            "Evaluate the frozen 180-policy inventory on every predeclared kernel, view, and "
            "start family with one GPU-batched implementation; preserve the superseded CPU "
            "evaluation separately and never combine its partial receipts with this run."
        ),
        "recorded_utc": _utc(),
        "evaluation_execution_commit": provenance["execution_commit"],
        "checkpoint_execution_provenance": checkpoint_provenance,
        "evaluation_source_sha256": provenance["evaluation_source_sha256"],
        "evaluation_backend": trace_eval.BACKEND_VERSION,
        "trace_gzip_level": trace_eval.TRACE_GZIP_LEVEL,
        "validation_receipt": validation,
        "oracle_independence_scope": (
            "Actor actions are selected before oracle computation. Independent replay "
            "recomputes conservation events from stored pre/post states without policy, "
            "teacher, reward, or transition-kernel calls; it is code-path independent "
            "but not independently authored by an external evaluator."
        ),
        "recommended_shard_count": 4,
        "superseded_partial_cpu_evaluation": (
            "/raid/ext_csv/HackRL/evaluations/cross_environment_teacher_matrix_v1"
        ),
        "partial_cpu_receipts_reused": 0,
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
    missing = [
        policy["checkpoint"]
        for policy in policies
        if not _checkpoint_complete(
            Path(policy["checkpoint"]), policy["checkpoint_origin"]
        )
    ]
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


def prepare_run(matrix: dict, main_run_root: Path, output_root: Path) -> dict:
    provenance = _assert_frozen_evaluator()
    policies = build_policy_inventory(matrix, main_run_root)
    tasks = build_tasks(policies)
    readiness = validate_ready(matrix, main_run_root)
    if not readiness["ready"]:
        raise RuntimeError(
            f"main training incomplete; missing checkpoints: {readiness['missing_checkpoints']}"
        )
    manifest = _ensure_evaluation_manifest(
        matrix, output_root, main_run_root, policies, tasks, provenance
    )
    capacity = _capacity_preflight(output_root, manifest["step_cap"])
    _write_json(output_root / "capacity_preflight.json", capacity)
    return {
        "prepared": True,
        "output_root": str(output_root),
        "evaluation_execution_commit": manifest["evaluation_execution_commit"],
        "policies": len(policies),
        "tasks": len(tasks),
        "episode_cap": manifest["episode_cap"],
        "step_cap": manifest["step_cap"],
        "capacity_preflight": capacity,
        "validation_receipt": manifest["validation_receipt"],
    }


def run_shard(
    matrix: dict,
    main_run_root: Path,
    output_root: Path,
    shard_index: int,
    shard_count: int,
) -> dict:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError((shard_index, shard_count))
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = (
        output_root / "shard_locks"
        / f"shard_{shard_index:02d}_of_{shard_count:02d}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"evaluation shard is already running: {lock_path}") from error
    provenance = _assert_frozen_evaluator()
    policies = build_policy_inventory(matrix, main_run_root)
    tasks = build_tasks(policies)
    readiness = validate_ready(matrix, main_run_root)
    if not readiness["ready"]:
        raise RuntimeError(
            f"main training incomplete; missing checkpoints: {readiness['missing_checkpoints']}"
        )
    manifest = _ensure_evaluation_manifest(
        matrix, output_root, main_run_root, policies, tasks, provenance
    )
    gpu_backend = _assert_gpu_backend()
    capacity = _capacity_preflight(output_root, manifest["step_cap"])
    selected_policies = [
        policy for index, policy in enumerate(policies)
        if index % shard_count == shard_index
    ]
    task_lookup = {policy["id"]: [] for policy in selected_policies}
    for task in tasks:
        if task["policy_id"] in task_lookup:
            task_lookup[task["policy_id"]].append(task)
    expected_tasks = sum(len(rows) for rows in task_lookup.values())
    ledger_path = (
        output_root / "shard_ledgers"
        / f"shard_{shard_index:02d}_of_{shard_count:02d}.jsonl"
    )
    total_episodes = 0
    total_steps = 0
    completed = 0
    worker_started = time.perf_counter()
    for policy in selected_policies:
        checkpoint = Path(policy["checkpoint"])
        fingerprint = base_trace_eval._checkpoint_fingerprint(checkpoint)
        actor_bundle = None
        for task in task_lookup[policy["id"]]:
            receipt_path = output_root / "task_receipts" / f"{task['id']}.json"
            if receipt_path.is_file():
                receipt = _verify_completed_task(
                    receipt_path, task, policy, manifest
                )
                total_episodes += int(receipt["episodes"])
                total_steps += int(receipt["steps"])
                completed += 1
                continue
            attempt = {
                "recorded_utc": _utc(),
                "task": task["id"],
                "policy": policy["id"],
                "shard_index": shard_index,
                "shard_count": shard_count,
                "evaluation_execution_commit": manifest["evaluation_execution_commit"],
                "matrix_sha256": manifest["matrix_sha256"],
                "panel_sha256": manifest["panel_sha256"],
                "evaluation_backend": manifest["evaluation_backend"],
                "status": "started",
            }
            _append_jsonl(ledger_path, attempt)
            started = time.perf_counter()
            try:
                if actor_bundle is None:
                    actor_bundle = base_trace_eval._load_actor(
                        policy["environment"], checkpoint
                    )
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
                    policy_id=policy["id"],
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
                    "evaluation_backend": summary["evaluation_backend"],
                    "artifacts": {
                        "trace": _artifact_receipt(trace_path),
                        "summary": _artifact_receipt(summary_path),
                        "independent_replay": _artifact_receipt(replay_path),
                    },
                }
                _write_json(receipt_path, receipt)
                _append_jsonl(ledger_path, receipt)
            except BaseException as error:
                failed = attempt | {
                    "recorded_utc": _utc(),
                    "status": "failed",
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": repr(error),
                    "automatic_re_evaluation": False,
                }
                _append_jsonl(ledger_path, failed)
                raise
            total_episodes += int(receipt["episodes"])
            total_steps += int(receipt["steps"])
            completed += 1
            elapsed = time.perf_counter() - worker_started
            progress = {
                "shard_index": shard_index,
                "completed_tasks": completed,
                "expected_tasks": expected_tasks,
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": (
                    elapsed / completed * (expected_tasks - completed)
                    if completed else None
                ),
                "last_task": task["id"],
            }
            print(json.dumps(progress, sort_keys=True), flush=True)
    result = {
        "schema_version": "hackrl_cross_environment_teacher_gpu_shard_receipt_v1",
        "recorded_utc": _utc(),
        "evaluation_execution_commit": manifest["evaluation_execution_commit"],
        "matrix_sha256": manifest["matrix_sha256"],
        "panel_sha256": manifest["panel_sha256"],
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected_policies": len(selected_policies),
        "expected_tasks": expected_tasks,
        "completed_tasks": completed,
        "episodes": total_episodes,
        "steps": total_steps,
        "elapsed_seconds": time.perf_counter() - worker_started,
        "passed": completed == expected_tasks,
        "automatic_re_evaluation": False,
        "gpu_backend": gpu_backend,
        "capacity_preflight": capacity,
    }
    path = (
        output_root / "shard_receipts"
        / f"shard_{shard_index:02d}_of_{shard_count:02d}.json"
    )
    _write_json(path, result)
    fcntl.flock(lock_handle, fcntl.LOCK_UN)
    lock_handle.close()
    return result


def finalize_run(matrix: dict, main_run_root: Path, output_root: Path) -> dict:
    provenance = _assert_frozen_evaluator()
    policies = build_policy_inventory(matrix, main_run_root)
    policy_lookup = {policy["id"]: policy for policy in policies}
    tasks = build_tasks(policies)
    manifest = _ensure_evaluation_manifest(
        matrix, output_root, main_run_root, policies, tasks, provenance
    )
    unexpected = sorted(
        path.name for path in (output_root / "task_receipts").glob("*.json")
        if path.stem not in {task["id"] for task in tasks}
    )
    total_episodes = 0
    total_steps = 0
    missing = []
    for task in tasks:
        receipt_path = output_root / "task_receipts" / f"{task['id']}.json"
        if not receipt_path.is_file():
            missing.append(task["id"])
            continue
        policy = policy_lookup[task["policy_id"]]
        receipt = _verify_completed_task(
            receipt_path, task, policy, manifest
        )
        total_episodes += int(receipt["episodes"])
        total_steps += int(receipt["steps"])
    if missing or unexpected:
        raise RuntimeError(
            f"evaluation incomplete: missing={len(missing)} "
            f"unexpected_receipts={unexpected}"
        )
    capacity = _capacity_preflight(output_root, manifest["step_cap"])
    passed = (
        total_episodes == manifest["episode_cap"]
        and 0 < total_steps <= manifest["step_cap"]
    )
    if not passed:
        raise RuntimeError(
            f"evaluation accounting mismatch: episodes={total_episodes} "
            f"steps={total_steps}"
        )
    result = {
        "schema_version": "hackrl_cross_environment_teacher_gpu_evaluation_receipt_v1",
        "recorded_utc": _utc(),
        "evaluation_execution_commit": manifest["evaluation_execution_commit"],
        "checkpoint_execution_commit": (
            manifest["checkpoint_execution_provenance"]["execution_commit"]
        ),
        "matrix_sha256": manifest["matrix_sha256"],
        "panel_sha256": manifest["panel_sha256"],
        "evaluation_backend": manifest["evaluation_backend"],
        "policies": len(policies),
        "tasks": len(tasks),
        "completed_tasks": len(tasks),
        "episodes": total_episodes,
        "steps": total_steps,
        "episode_cap": manifest["episode_cap"],
        "step_cap": manifest["step_cap"],
        "independent_replays_passed": len(tasks),
        "partial_cpu_receipts_reused": 0,
        "capacity_preflight": capacity,
        "passed": passed,
        "automatic_re_evaluation": False,
    }
    path = output_root / "execution_receipt.json"
    if path.is_file():
        existing = json.loads(path.read_text())
        comparable = dict(existing)
        comparable["recorded_utc"] = result["recorded_utc"]
        if comparable != result:
            raise RuntimeError(f"execution receipt drift: {path}")
    else:
        _write_json(path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--main-run-root", type=Path, default=DEFAULT_MAIN_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", action="store_true")
    group.add_argument("--list-policies", action="store_true")
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run-shard", action="store_true")
    group.add_argument("--finalize", action="store_true")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--authorize-common-evaluation", action="store_true")
    args = parser.parse_args()
    if args.matrix.resolve() != DEFAULT_MATRIX.resolve():
        raise SystemExit(
            f"scientific GPU evaluation is frozen to {DEFAULT_MATRIX}"
        )
    matrix = load_matrix(DEFAULT_MATRIX)
    if args.validate:
        print(json.dumps(
            validate_ready(matrix, args.main_run_root), indent=2, sort_keys=True
        ))
        return
    policies = build_policy_inventory(matrix, args.main_run_root)
    if args.list_policies:
        print(json.dumps(policies, indent=2, sort_keys=True))
        return
    if not args.authorize_common_evaluation:
        raise SystemExit(
            "prepare, shard execution, and finalization require "
            "--authorize-common-evaluation"
        )
    if args.prepare:
        result = prepare_run(matrix, args.main_run_root, args.output_root)
    elif args.run_shard:
        if args.shard_index is None:
            raise SystemExit("--run-shard requires --shard-index")
        result = run_shard(
            matrix,
            args.main_run_root,
            args.output_root,
            args.shard_index,
            args.shard_count,
        )
    else:
        result = finalize_run(matrix, args.main_run_root, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
