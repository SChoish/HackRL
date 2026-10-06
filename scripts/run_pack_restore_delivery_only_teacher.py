#!/usr/bin/env python3
"""Run the 20-job PACK-RESTORE delivery-only teacher experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import uuid
from pathlib import Path

if os.environ.get("HACKRL_DEVICE", "cpu") != "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

from hackrl.dual_leo import (
    LEO_EPOCHS,
    LEO_MINIBATCH_SIZE,
    load_dual_checkpoint,
)
from hackrl.pack_restore_gc import (
    DELIVER_3_GOAL_INDEX,
    NUM_ACTIONS,
    NUM_GOALS,
)
from run_dual_leo_compare import (
    ADAPT_UPDATES,
    ENVS,
    PRETRAIN_UPDATES,
    _cell_dir,
    _checkpoint_fingerprint,
    _dual_update,
    _job_complete,
    _job_config,
    _load_adapt_start,
    _start_teacher,
    run_adapt,
)


REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(
    "docs/manifests/pack_restore_delivery_only_teacher_v1.json"
)
SOURCE_RUN_ROOT = Path(
    "/raid/ext_csv/HackRL/runs/pack_restore_scale_bc_v1"
)
SEEDS = tuple(range(30, 40))
VARIANTS = ("fixed", "mutant")
FULL_DUAL_ARM = {"learn_teacher": True, "imitate_teacher": True}
DELIVERY_ONLY_ARM = {
    "learn_teacher": True,
    "imitate_teacher": True,
    "teacher_goal_indices": [DELIVER_3_GOAL_INDEX],
}
EXPECTED_SOURCE_EXECUTION_SHA = "eaf603b1ab378b34bf2a788927f59446cabdde50"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "85ee37b55b4e7904391b817d924c381e613911cd91d13574b4fd4c07bf4dc534"
)
EXPECTED_SOURCE_FINGERPRINTS = {
    30: "72a22fd391b4ead94ec4f27f6eecf3ec329573ae087b9a51587ce4027c06ae47",
    31: "3de211aaf701de9e867ecca30f132f60d31a110b3d08205373f202a6facabb9d",
    32: "167f68663089ffa94d8180d74ea680023f35ea237b8b1e00800922d87b76ef6d",
    33: "b4c8d414e778b1021ab1d24dfe0d6b2ff5bd23f8e3b7606ce94f13d257a5ad09",
    34: "18e67f005537a4ee83ccc6e8f9f29af29596e345f1845de5486a01f38b5f5cc2",
    35: "6c526121ca8ce95c1081466169e6bbd7eb9a7d9cd3802209f4bf5476293ccf3f",
    36: "9993e0ebdbe1c02c3591727a4fc51671c026bddacd5df2374daa8ac05dbbf9c5",
    37: "73fb64ba75c3cbe570dfeb5bcf649af689ca89376267cbc0a2cf3659a9462376",
    38: "446cd0ab191477f05c1f5c853754e5a31810b82d0610cc926dd9cc3f7d8f3b7d",
    39: "9e2fe3947e58f1ae10f54243061f4fbb98d3b632b4f82c68b7c95cf12f505730",
}
EXECUTION_SOURCES = (
    MANIFEST_PATH,
    Path("scripts/run_pack_restore_delivery_only_teacher.py"),
    Path("scripts/run_pack_restore_delivery_only_teacher_queue.sh"),
    Path("scripts/run_pack_restore_delivery_only_teacher_watchdog.sh"),
    Path("scripts/summarize_pack_restore_delivery_only_teacher.py"),
    Path("scripts/audit_pack_restore_teacher_evaluation.py"),
    Path("scripts/analyze_pack_restore_common_state_teachers.py"),
    Path("scripts/run_pack_restore_pretrained_teacher_freeze.py"),
    Path("scripts/trace_pack_restore_teacher_recommendations.py"),
    Path("scripts/run_dual_leo_compare.py"),
    Path("src/hackrl/pack_restore.py"),
    Path("src/hackrl/pack_restore_gc.py"),
    Path("src/hackrl/dual_leo.py"),
    Path("src/hackrl/batch_renorm.py"),
    Path("src/hackrl/tick_claim.py"),
    Path("src/hackrl/tick_claim_gc.py"),
    Path("src/hackrl/tick_claim_oracle.py"),
)
MEASURED_DUAL_CHECKPOINT_BYTES = 49_568_320
RETAINED_FINAL_CHECKPOINTS = len(SEEDS) * len(VARIANTS)
ACTIVE_OVERLAP_CHECKPOINTS = 1
LOG_AND_EVALUATION_ALLOWANCE_BYTES = 1024**3
PROJECTED_REMAINING_WRITE_BYTES = (
    MEASURED_DUAL_CHECKPOINT_BYTES
    * (RETAINED_FINAL_CHECKPOINTS + ACTIVE_OVERLAP_CHECKPOINTS)
    + LOG_AND_EVALUATION_ALLOWANCE_BYTES
)
SAFETY_RESERVE_BYTES = max(
    8 * 1024**3,
    (PROJECTED_REMAINING_WRITE_BYTES + 4) // 5,
)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments):
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def source_checkpoint(seed):
    return (
        SOURCE_RUN_ROOT
        / "size_s"
        / "pack"
        / "dual"
        / "pretrain"
        / f"seed{seed}"
        / "checkpoints"
        / "update_512"
    )


def build_jobs():
    jobs = []
    for seed in SEEDS:
        for variant in VARIANTS:
            jobs.append(
                {
                    "id": f"pack-s-d-delivery-s{seed}-{variant}",
                    "kind": "adapt",
                    "env": "pack",
                    "method": "dual_teacher_delivery_only",
                    "condition": "D-delivery",
                    "size": "S",
                    "policy_hidden_size": 512,
                    "teacher_hidden_size": 512,
                    "seed": seed,
                    "variant": variant,
                    "learn_teacher": True,
                    "imitate_teacher": True,
                    "teacher_goal_indices": [DELIVER_3_GOAL_INDEX],
                    "origin_arm": dict(FULL_DUAL_ARM),
                    "pretrain_checkpoint": str(source_checkpoint(seed)),
                    "rolling_checkpoints": True,
                    "record_checkpoint_fingerprint": True,
                }
            )
    return jobs


def _validate_source_run():
    run_manifest = json.loads(
        (SOURCE_RUN_ROOT / "run_manifest.json").read_text(encoding="utf-8")
    )
    if run_manifest.get("execution_code_sha") != EXPECTED_SOURCE_EXECUTION_SHA:
        raise RuntimeError("source scale run execution SHA mismatch")
    if (
        run_manifest.get("execution_source_sha256", {}).get(
            "docs/manifests/pack_restore_scale_bc_v1.json"
        )
        != EXPECTED_SOURCE_MANIFEST_SHA256
    ):
        raise RuntimeError("source scale run manifest digest mismatch")
    rows = []
    for seed in SEEDS:
        checkpoint = source_checkpoint(seed)
        fingerprint = _checkpoint_fingerprint(checkpoint)
        arm = json.loads((checkpoint / "arm.json").read_text(encoding="utf-8"))
        metadata = json.loads(
            (checkpoint / "metadata.json").read_text(encoding="utf-8")
        )
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        passed = (
            fingerprint["sha256"] == EXPECTED_SOURCE_FINGERPRINTS[seed]
            and arm == FULL_DUAL_ARM
            and metadata.get("global_update") == PRETRAIN_UPDATES
            and config.get("seed") == seed
            and config.get("goal_mode") == "workshop12"
            and config.get("policy_hidden_size") == 512
            and config.get("teacher_hidden_size") == 512
            and config.get("num_envs") == 512
            and config.get("num_steps") == 64
        )
        rows.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "fingerprint": fingerprint["sha256"],
                "passed": passed,
            }
        )
    if not all(row["passed"] for row in rows):
        raise RuntimeError(f"source checkpoint validation failed: {rows}")
    return rows


def _canonical_tree_sha256(tree):
    """Hash numerical PyTree state independent of JAX host/device containers."""

    digest = hashlib.sha256()
    path_leaves, structure = jax.tree_util.tree_flatten_with_path(tree)
    digest.update(str(structure).encode("utf-8"))
    digest.update(b"\n")
    for path, value in path_leaves:
        array = np.ascontiguousarray(jax.device_get(value))
        digest.update(jax.tree_util.keystr(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _teacher_digests(leo):
    def digest(value):
        return _canonical_tree_sha256(value)

    return {
        "params_sha256": digest(leo.params),
        "opt_state_sha256": digest(leo.opt_state),
        "batch_stats_sha256": digest(leo.batch_stats),
        "step_sha256": digest(leo.step),
        "step": int(leo.step),
    }


def _load_teacher(checkpoint, config):
    spec = ENVS["pack"]
    _, template = spec["initialize"](config)
    _, leo_template, _ = _start_teacher(spec, config, template)
    _, leo = load_dual_checkpoint(checkpoint, template, leo_template)
    return leo


def _verify_delivery_only_teacher(log_dir, job):
    source = source_checkpoint(job["seed"])
    source_config = ENVS["pack"]["from_payload"](
        json.loads((source / "config.json").read_text(encoding="utf-8"))
    )
    branch = _job_config(
        job,
        goal_mode="deliver_3",
        variant=job["variant"],
        updates=ADAPT_UPDATES,
    )
    cell = _cell_dir(log_dir, job)
    final = cell / "checkpoints" / f"adapt_{ADAPT_UPDATES}"
    recorded_arm = json.loads((final / "arm.json").read_text(encoding="utf-8"))
    source_teacher = _load_teacher(source, source_config)
    final_teacher = _load_teacher(final, branch)
    source_digest = _teacher_digests(source_teacher)
    final_digest = _teacher_digests(final_teacher)

    source_kernel = np.asarray(
        jax.device_get(source_teacher.params["q_output"]["kernel"])
    ).reshape((-1, NUM_GOALS, NUM_ACTIONS))
    final_kernel = np.asarray(
        jax.device_get(final_teacher.params["q_output"]["kernel"])
    ).reshape((-1, NUM_GOALS, NUM_ACTIONS))
    source_bias = np.asarray(
        jax.device_get(source_teacher.params["q_output"]["bias"])
    ).reshape((NUM_GOALS, NUM_ACTIONS))
    final_bias = np.asarray(
        jax.device_get(final_teacher.params["q_output"]["bias"])
    ).reshape((NUM_GOALS, NUM_ACTIONS))
    non_delivery = np.arange(NUM_GOALS) != DELIVER_3_GOAL_INDEX
    scheduled_step_delta = (
        ADAPT_UPDATES
        * (branch.batch_size // LEO_MINIBATCH_SIZE)
        * LEO_EPOCHS
    )
    log_rows = [
        json.loads(line)
        for line in (cell / "updates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((cell / "summary.json").read_text(encoding="utf-8"))
    applied_step_delta = sum(
        int(row.get("teacher_applied_minibatches", -scheduled_step_delta))
        for row in log_rows
    )
    observed_step_delta = final_digest["step"] - source_digest["step"]
    target_metrics_valid = len(log_rows) == ADAPT_UPDATES and all(
        int(row.get("teacher_target_head_count", -1)) == 1
        and int(row.get("teacher_target_terms", -1))
        == int(row.get("teacher_valid_samples", -2))
        for row in log_rows
    )
    checks = {
        "arm_exact": recorded_arm == DELIVERY_ONLY_ARM,
        "teacher_state_changed": source_digest != final_digest,
        "teacher_step_delta_matches_applied_log": (
            observed_step_delta == applied_step_delta
        ),
        "teacher_step_delta_positive_and_within_schedule": (
            0 < observed_step_delta <= scheduled_step_delta
        ),
        "summary_matches_final_teacher_step": (
            int(summary.get("teacher_applied_grad_steps", -1))
            == final_digest["step"]
        ),
        "delivery_output_changed": (
            not np.array_equal(
                source_kernel[:, DELIVER_3_GOAL_INDEX, :],
                final_kernel[:, DELIVER_3_GOAL_INDEX, :],
            )
            or not np.array_equal(
                source_bias[DELIVER_3_GOAL_INDEX, :],
                final_bias[DELIVER_3_GOAL_INDEX, :],
            )
        ),
        "target_metrics_valid": target_metrics_valid,
    }
    document = {
        "schema_version": "hackrl_delivery_only_teacher_verification_v1",
        "job": job["id"],
        "source_checkpoint": str(source.resolve()),
        "final_checkpoint": str(final.resolve()),
        "source": source_digest,
        "final": final_digest,
        "teacher_goal_indices": [DELIVER_3_GOAL_INDEX],
        "teacher_target_head_count": 1,
        "all_goal_baseline_target_head_count": NUM_GOALS,
        "loss_normalization": (
            "sum selected-head half-squared TD errors per valid transition; "
            "the delivery head contribution keeps the baseline per-transition scale"
        ),
        "scheduled_teacher_step_delta": scheduled_step_delta,
        "applied_teacher_step_delta_from_log": applied_step_delta,
        "observed_teacher_step_delta": observed_step_delta,
        "update_log_rows": len(log_rows),
        "optimizer_state_inherited_from_multi_goal_pretraining": True,
        "non_delivery_output_parameter_drift": {
            "kernel_max_abs": float(np.max(np.abs(
                source_kernel[:, non_delivery, :] - final_kernel[:, non_delivery, :]
            ))),
            "bias_max_abs": float(np.max(np.abs(
                source_bias[non_delivery, :] - final_bias[non_delivery, :]
            ))),
            "expected_reason": "inherited Adam momentum; shared layers also change non-delivery Q values",
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not document["passed"]:
        raise RuntimeError(f"delivery-only verification failed: {job['id']}")
    _write_json(cell / "delivery_only_verification.json", document)
    return document


def _run_provenance(jobs):
    sources = [str(path) for path in EXECUTION_SOURCES]
    dirty = _git("status", "--porcelain", "--", *sources)
    if dirty:
        raise RuntimeError("execution sources must be committed:\n" + dirty)
    tracked = set(_git("ls-files", "--", *sources).splitlines())
    missing = sorted(set(sources) - tracked)
    if missing:
        raise RuntimeError(f"untracked execution sources: {missing}")
    return {
        "schema_version": "hackrl_pack_restore_delivery_only_teacher_run_v1",
        "experiment_id": "pack_restore_delivery_only_teacher_v1",
        "execution_code_sha": _git("rev-parse", "HEAD"),
        "manifest_sha256": _sha256_file(REPOSITORY / MANIFEST_PATH),
        "execution_source_sha256": {
            name: _sha256_file(REPOSITORY / name) for name in sources
        },
        "source_scale_run": {
            "root": str(SOURCE_RUN_ROOT),
            "execution_code_sha": EXPECTED_SOURCE_EXECUTION_SHA,
            "manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        },
        "job_count": len(jobs),
        "job_ids": [job["id"] for job in jobs],
        "projected_remaining_write_bytes": PROJECTED_REMAINING_WRITE_BYTES,
        "safety_reserve_bytes": SAFETY_RESERVE_BYTES,
        "chosen_filesystem": "/raid/ext_csv",
    }


def _directory_bytes(root):
    root = Path(root)
    if not root.exists():
        return 0
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    )


def record_capacity(log_dir, event):
    log_dir = Path(log_dir).resolve()
    if not str(log_dir).startswith("/raid/ext_csv/HackRL/runs/"):
        raise RuntimeError(f"run root must be on RAID: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)
    raid = shutil.disk_usage("/raid/ext_csv")
    home = shutil.disk_usage("/home/ext_csv")
    required = PROJECTED_REMAINING_WRITE_BYTES + SAFETY_RESERVE_BYTES
    row = {
        "event": event,
        "utc_unix_seconds": time.time(),
        "run_bytes": _directory_bytes(log_dir),
        "projected_remaining_write_bytes": PROJECTED_REMAINING_WRITE_BYTES,
        "safety_reserve_bytes": SAFETY_RESERVE_BYTES,
        "required_free_bytes": required,
        "chosen_filesystem": "/raid/ext_csv",
        "raid": {
            "total_bytes": raid.total,
            "used_bytes": raid.used,
            "free_bytes": raid.free,
        },
        "home": {
            "total_bytes": home.total,
            "used_bytes": home.used,
            "free_bytes": home.free,
            "utilization_fraction": home.used / home.total,
        },
    }
    if raid.free < required:
        raise RuntimeError(
            f"insufficient RAID capacity: free={raid.free}, required={required}"
        )
    path = log_dir / "capacity_checks.json"
    history = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    history.append(row)
    _write_json(path, history)
    return row


def _prepare_run(log_dir, jobs):
    log_dir = Path(log_dir)
    provenance = _run_provenance(jobs)
    path = log_dir / "run_manifest.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != provenance:
            raise RuntimeError("existing run manifest differs from current provenance")
    else:
        record_capacity(log_dir, "prelaunch")
        _write_json(path, provenance)
    return provenance


def _process_start_ticks(pid):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8").split()
        return int(fields[21])
    except (OSError, ValueError, IndexError):
        return None


def _reclaim_claims(log_dir):
    claims = Path(log_dir) / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    for path in claims.iterdir():
        try:
            owner = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            path.unlink(missing_ok=True)
            continue
        alive_start = _process_start_ticks(owner.get("pid"))
        if alive_start != owner.get("process_start_ticks"):
            path.unlink(missing_ok=True)


def _claim(log_dir, job_id):
    path = Path(log_dir) / "claims" / job_id
    owner = {
        "pid": os.getpid(),
        "process_start_ticks": _process_start_ticks(os.getpid()),
        "token": uuid.uuid4().hex,
        "host": platform.node(),
        "created_unix_seconds": time.time(),
    }
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return None
    os.write(descriptor, (json.dumps(owner, sort_keys=True) + "\n").encode())
    os.close(descriptor)
    return owner


def _release(log_dir, job_id, owner):
    path = Path(log_dir) / "claims" / job_id
    if not path.is_file():
        return
    recorded = json.loads(path.read_text(encoding="utf-8"))
    if recorded != owner:
        raise RuntimeError(f"refusing to release a foreign claim: {job_id}")
    path.unlink()


def validate_contract():
    jobs = build_jobs()
    sources = _validate_source_run()
    if len(jobs) != 20:
        raise RuntimeError(f"expected 20 jobs, found {len(jobs)}")
    if sum(
        ADAPT_UPDATES * 512 * 64 for _ in jobs
    ) != 2_684_354_560:
        raise RuntimeError("transition budget mismatch")
    for job in jobs:
        if (
            job["size"] != "S"
            or job["method"] != "dual_teacher_delivery_only"
            or job["condition"] != "D-delivery"
            or job["learn_teacher"] is not True
            or job["imitate_teacher"] is not True
            or job["teacher_goal_indices"] != [DELIVER_3_GOAL_INDEX]
            or job["origin_arm"] != FULL_DUAL_ARM
            or job["pretrain_checkpoint"] != str(source_checkpoint(job["seed"]))
        ):
            raise RuntimeError(f"invalid delivery-only job: {job}")
    return {
        "jobs": len(jobs),
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "transitions": 2_684_354_560,
        "source_checkpoints": sources,
        "arm": DELIVERY_ONLY_ARM,
        "projected_remaining_write_bytes": PROJECTED_REMAINING_WRITE_BYTES,
        "safety_reserve_bytes": SAFETY_RESERVE_BYTES,
    }


def smoke():
    _validate_source_run()
    job = build_jobs()[0]
    branch = _job_config(
        job, goal_mode="deliver_3", variant=job["variant"], updates=ADAPT_UPDATES
    )
    spec = ENVS["pack"]
    network, template = spec["initialize"](branch)
    teacher, _, minibatch = _start_teacher(spec, branch, template)
    runner, leo = _load_adapt_start(Path("/unused"), job, template, branch)
    before = _teacher_digests(leo)
    before_kernel = np.asarray(
        jax.device_get(leo.params["q_output"]["kernel"])
    ).reshape((-1, NUM_GOALS, NUM_ACTIONS))
    before_bias = np.asarray(
        jax.device_get(leo.params["q_output"]["bias"])
    ).reshape((NUM_GOALS, NUM_ACTIONS))
    update = _dual_update(
        spec, network, teacher, branch, minibatch, DELIVERY_ONLY_ARM
    )
    started = time.perf_counter()
    updated, adapted, metrics = update(runner, leo)
    jax.block_until_ready((updated.global_update, adapted.step, metrics["loss"]))
    after = _teacher_digests(adapted)
    after_kernel = np.asarray(
        jax.device_get(adapted.params["q_output"]["kernel"])
    ).reshape((-1, NUM_GOALS, NUM_ACTIONS))
    after_bias = np.asarray(
        jax.device_get(adapted.params["q_output"]["bias"])
    ).reshape((NUM_GOALS, NUM_ACTIONS))
    non_delivery = np.arange(NUM_GOALS) != DELIVER_3_GOAL_INDEX
    valid_samples = int(metrics["teacher_valid_samples"])
    target_terms = int(metrics["teacher_target_terms"])
    expected_teacher_step_delta = (
        branch.batch_size // LEO_MINIBATCH_SIZE
    ) * LEO_EPOCHS
    result = {
        "shape": "512x64",
        "source_seed": job["seed"],
        "seconds_compile_and_update": time.perf_counter() - started,
        "global_update_before": int(runner.global_update),
        "global_update_after": int(updated.global_update),
        "teacher_state_before": before,
        "teacher_state_after": after,
        "teacher_state_changed": before != after,
        "teacher_step_delta": after["step"] - before["step"],
        "expected_teacher_step_delta": expected_teacher_step_delta,
        "non_delivery_output_kernel_max_abs_drift": float(np.max(np.abs(
            before_kernel[:, non_delivery, :] - after_kernel[:, non_delivery, :]
        ))),
        "non_delivery_output_bias_max_abs_drift": float(np.max(np.abs(
            before_bias[non_delivery, :] - after_bias[non_delivery, :]
        ))),
        "delivery_output_changed": (
            not np.array_equal(
                before_kernel[:, DELIVER_3_GOAL_INDEX, :],
                after_kernel[:, DELIVER_3_GOAL_INDEX, :],
            )
            or not np.array_equal(
                before_bias[DELIVER_3_GOAL_INDEX, :],
                after_bias[DELIVER_3_GOAL_INDEX, :],
            )
        ),
        "learn_teacher_metric": int(metrics["learn_teacher"]),
        "imitate_teacher_metric": int(metrics["imitate_teacher"]),
        "teacher_target_head_count": int(metrics["teacher_target_head_count"]),
        "teacher_valid_samples": valid_samples,
        "teacher_target_terms": target_terms,
        "bc_policy_coef": float(metrics["bc_policy_coef"]),
    }
    if not (
        result["teacher_state_changed"]
        and result["teacher_step_delta"] == expected_teacher_step_delta
        and result["delivery_output_changed"]
        and result["global_update_after"] == PRETRAIN_UPDATES + 1
        and result["learn_teacher_metric"] == 1
        and result["imitate_teacher_metric"] == 1
        and result["teacher_target_head_count"] == 1
        and target_terms == valid_samples
        and result["bc_policy_coef"] > 0
    ):
        raise RuntimeError(f"delivery-only teacher smoke failed: {result}")
    return result


def work(log_dir, worker):
    jobs = build_jobs()
    _validate_source_run()
    provenance = _prepare_run(log_dir, jobs)
    print(f"[provenance] execution_code_sha={provenance['execution_code_sha']}", flush=True)
    print(f"[worker] {worker} jobs={len(jobs)}", flush=True)
    while True:
        _reclaim_claims(log_dir)
        chosen = None
        owner = None
        for job in jobs:
            if _job_complete(log_dir, job):
                verification = _cell_dir(log_dir, job) / "delivery_only_verification.json"
                if not verification.is_file():
                    _verify_delivery_only_teacher(log_dir, job)
                continue
            owner = _claim(log_dir, job["id"])
            if owner is not None:
                chosen = job
                break
        if chosen is None:
            if all(_job_complete(log_dir, job) for job in jobs):
                record_capacity(log_dir, "queue_complete")
                print(f"[worker] {worker} sweep complete", flush=True)
                return
            time.sleep(15)
            continue
        try:
            record_capacity(log_dir, f"before_{chosen['id']}")
            print(f"[job] {chosen['id']}", flush=True)
            run_adapt(log_dir, chosen)
            _verify_delivery_only_teacher(log_dir, chosen)
            record_capacity(log_dir, f"after_{chosen['id']}")
        finally:
            _release(log_dir, chosen["id"], owner)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir")
    parser.add_argument("--worker")
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--validate-contract", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.list_jobs:
        print(json.dumps(build_jobs(), indent=2, sort_keys=True))
        return
    if arguments.validate_contract:
        print(json.dumps(validate_contract(), indent=2, sort_keys=True))
        return
    if arguments.smoke:
        print(json.dumps(smoke(), indent=2, sort_keys=True))
        return
    if not arguments.log_dir or not arguments.worker:
        raise SystemExit("queue mode requires --log-dir and --worker")
    work(Path(arguments.log_dir).resolve(), arguments.worker)


if __name__ == "__main__":
    main()
