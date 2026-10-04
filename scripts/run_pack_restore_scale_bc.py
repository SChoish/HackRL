#!/usr/bin/env python3
"""PACK-RESTORE PPO scale by adaptation-time teacher guidance.

The queue owns two pretraining histories per size and seed: GC-PPO and full
Dual. Full-Dual pretraining is then loaded twice for D-on and D-off, so their
policy, critic, Adam, teacher, BatchRenorm, environment, and RNG states are
identical before the adaptation arm changes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

if os.environ.get("HACKRL_DEVICE", "cpu") != "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

from hackrl.dual_leo import init_dual_leo_teacher, teacher_parameter_count
from hackrl.pack_restore_gc import (
    NUM_ACTIONS,
    NUM_GOALS,
    _batch_inputs,
    initialize_pack_restore_gc,
    make_pack_restore_gc_update,
    pack_restore_gc_parameter_count,
)
from run_dual_leo_compare import (
    ENVS,
    _claim,
    _dual_update,
    _job_complete,
    _job_config,
    _ready,
    _reclaim,
    _release,
    run_adapt,
    run_pretrain,
)

SEEDS = tuple(range(30, 40))
SIZE_SPECS = (
    {
        "size": "S",
        "policy_hidden_size": 512,
        "expected_ppo_parameters": 2_644_057,
    },
    {
        "size": "M",
        "policy_hidden_size": 1024,
        "expected_ppo_parameters": 8_426_585,
    },
    {
        "size": "L",
        "policy_hidden_size": 2048,
        "expected_ppo_parameters": 29_428_825,
    },
)
TEACHER_HIDDEN_SIZE = 512
EXPECTED_TEACHER_PARAMETERS = 1_469_366
FULL_DUAL_ARM = {"learn_teacher": True, "imitate_teacher": True}
REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path("docs/manifests/pack_restore_scale_bc_v1.json")
EXECUTION_SOURCES = (
    MANIFEST_PATH,
    Path("scripts/run_pack_restore_scale_bc.py"),
    Path("scripts/run_pack_restore_scale_bc_queue.sh"),
    Path("scripts/summarize_pack_restore_scale_bc.py"),
    Path("scripts/run_dual_leo_compare.py"),
    Path("src/hackrl/pack_restore.py"),
    Path("src/hackrl/pack_restore_gc.py"),
    Path("src/hackrl/dual_leo.py"),
    Path("src/hackrl/batch_renorm.py"),
    Path("src/hackrl/tick_claim.py"),
    Path("src/hackrl/tick_claim_gc.py"),
    Path("src/hackrl/tick_claim_oracle.py"),
)
PROJECTED_FINAL_CHECKPOINT_BYTES = 41_573_442_180
PRELAUNCH_FREE_SPACE_MARGIN_BYTES = 8 * 1024**3


def _sized(spec):
    return {
        "env": "pack",
        "size": spec["size"],
        "policy_hidden_size": spec["policy_hidden_size"],
        "teacher_hidden_size": TEACHER_HIDDEN_SIZE,
        "rolling_checkpoints": True,
        "record_checkpoint_fingerprint": True,
    }


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


def _run_provenance(jobs):
    relative_sources = [str(path) for path in EXECUTION_SOURCES]
    dirty = _git("status", "--porcelain", "--", *relative_sources)
    if dirty:
        raise RuntimeError(
            "execution sources must be committed before launch:\n" + dirty
        )
    tracked = set(_git("ls-files", "--", *relative_sources).splitlines())
    missing = sorted(set(relative_sources) - tracked)
    if missing:
        raise RuntimeError(f"execution sources are not tracked: {missing}")
    return {
        "schema_version": "hackrl_pack_restore_scale_bc_run_v1",
        "manifest_id": "pack_restore_scale_bc_v1",
        "execution_code_sha": _git("rev-parse", "HEAD"),
        "manifest_sha256": _sha256_file(REPOSITORY / MANIFEST_PATH),
        "execution_source_sha256": {
            name: _sha256_file(REPOSITORY / name) for name in relative_sources
        },
        "job_count": len(jobs),
        "job_ids": [job["id"] for job in jobs],
        "checkpoint_retention": (
            "all science curves; one rolling resume state while training; "
            "final state for every pretraining and adaptation job"
        ),
        "projected_final_checkpoint_bytes": PROJECTED_FINAL_CHECKPOINT_BYTES,
    }


def _prepare_run(log_dir, jobs):
    """Lock the run to committed sources and check initial storage headroom."""

    document = _run_provenance(jobs)
    log_dir = Path(log_dir)
    path = log_dir / "run_manifest.json"
    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded != document:
            raise RuntimeError(
                f"run provenance does not match the current checkout: {path}"
            )
        return document
    required = PROJECTED_FINAL_CHECKPOINT_BYTES + PRELAUNCH_FREE_SPACE_MARGIN_BYTES
    log_dir.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(log_dir.parent).free
    if free < required:
        raise RuntimeError(
            f"need at least {required} free bytes before launch; found {free}"
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document


def build_jobs():
    """Return 60 pretraining and 180 adaptation jobs in seed-size bundles."""

    jobs = []
    for seed in SEEDS:
        pretraining = {}
        for spec in SIZE_SPECS:
            size = spec["size"]
            common = _sized(spec) | {"seed": seed}
            for method, condition in (("gc", "G"), ("dual", "Dual")):
                identifier = f"pack-{size.lower()}-{method}-pretrain-s{seed}"
                pretraining[(size, method)] = identifier
                jobs.append(
                    common
                    | {
                        "id": identifier,
                        "kind": "pretrain",
                        "method": method,
                        "condition": condition,
                    }
                )

        for spec in SIZE_SPECS:
            size = spec["size"]
            common = _sized(spec) | {"seed": seed}
            arms = (
                (
                    "gc",
                    "G",
                    pretraining[(size, "gc")],
                    {},
                ),
                (
                    "dual",
                    "D-on",
                    pretraining[(size, "dual")],
                    {},
                ),
                (
                    "dual_bc_off",
                    "D-off",
                    pretraining[(size, "dual")],
                    {
                        "pretrain_method": "dual",
                        "learn_teacher": True,
                        "imitate_teacher": False,
                        "origin_arm": dict(FULL_DUAL_ARM),
                    },
                ),
            )
            for method, condition, dependency, extra in arms:
                for variant in ("fixed", "mutant"):
                    jobs.append(
                        common
                        | {
                            "id": (
                                f"pack-{size.lower()}-{condition.lower()}-"
                                f"s{seed}-{variant}"
                            ),
                            "kind": "adapt",
                            "method": method,
                            "condition": condition,
                            "variant": variant,
                            "depends_on": [dependency],
                        }
                        | extra
                    )
    return jobs


def initialized_parameter_counts():
    """Initialize each declared width and verify counts from the real modules."""

    rows = []
    for spec in SIZE_SPECS:
        config = _job_config(
            _sized(spec) | {"seed": SEEDS[0]},
            goal_mode="workshop12",
            variant="fixed",
            updates=1,
        )
        config = replace(config, num_envs=2, num_steps=4, minibatch_size=8)
        network, runner = initialize_pack_restore_gc(config)
        model_inputs = _batch_inputs(runner.env_state, runner.current_goal)
        _, teacher, _ = init_dual_leo_teacher(
            config,
            model_inputs[0],
            model_inputs[1],
            NUM_GOALS,
            NUM_ACTIONS,
        )
        row = {
            "size": spec["size"],
            "policy_hidden_size": spec["policy_hidden_size"],
            "ppo_parameters": pack_restore_gc_parameter_count(
                runner.train_state.params
            ),
            "teacher_hidden_size": config.teacher_hidden_size,
            "teacher_parameters": teacher_parameter_count(teacher),
        }
        row["matches_contract"] = (
            row["ppo_parameters"] == spec["expected_ppo_parameters"]
            and row["teacher_parameters"] == EXPECTED_TEACHER_PARAMETERS
        )
        rows.append(row)
        del network, runner, teacher, model_inputs
        gc.collect()
    if not all(row["matches_contract"] for row in rows):
        raise RuntimeError(f"parameter count contract failed: {rows}")
    return rows


def measure(size, method, updates):
    """Measure the real 512-env by 64-step update for one declared size."""

    spec = next(item for item in SIZE_SPECS if item["size"] == size)
    job = _sized(spec) | {"seed": SEEDS[0], "method": method}
    config = _job_config(
        job,
        goal_mode="workshop12",
        variant="fixed",
        updates=updates,
    )
    network, runner = initialize_pack_restore_gc(config)
    if method == "gc":
        update = jax.jit(make_pack_restore_gc_update(network, config))
        teacher = None
    else:
        model_inputs = _batch_inputs(runner.env_state, runner.current_goal)
        teacher_network, teacher, minibatch = init_dual_leo_teacher(
            config,
            model_inputs[0],
            model_inputs[1],
            NUM_GOALS,
            NUM_ACTIONS,
        )
        update = _dual_update(
            ENVS["pack"],
            network,
            teacher_network,
            config,
            minibatch,
            FULL_DUAL_ARM,
        )
    samples = []
    for _ in range(int(updates)):
        started = time.perf_counter()
        if method == "gc":
            runner, metrics = update(runner)
            ready = (runner.global_update, metrics["valid_transitions"])
        else:
            runner, teacher, metrics = update(runner, teacher)
            ready = (
                runner.global_update,
                teacher.step,
                metrics["teacher_td_loss"],
            )
        jax.block_until_ready(ready)
        samples.append(time.perf_counter() - started)
    steady = samples[1:] if len(samples) > 1 else samples
    result = {
        "size": size,
        "method": method,
        "updates": int(updates),
        "compile_and_first_seconds": samples[0],
        "steady_seconds_per_update": float(np.mean(steady)),
        "ppo_parameters": pack_restore_gc_parameter_count(
            runner.train_state.params
        ),
        "teacher_parameters": (
            0 if teacher is None else teacher_parameter_count(teacher)
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir")
    parser.add_argument("--worker")
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--validate-contract", action="store_true")
    parser.add_argument("--measure-size", choices=tuple(s["size"] for s in SIZE_SPECS))
    parser.add_argument("--measure-method", choices=("gc", "dual"), default="dual")
    parser.add_argument("--measure-updates", type=int, default=0)
    arguments = parser.parse_args()
    jobs = build_jobs()
    if arguments.list_jobs:
        print(json.dumps(jobs, indent=2, sort_keys=True))
        return
    if arguments.validate_contract:
        print(json.dumps(initialized_parameter_counts(), indent=2, sort_keys=True))
        return
    if arguments.measure_updates:
        measure(
            arguments.measure_size or "S",
            arguments.measure_method,
            arguments.measure_updates,
        )
        return
    if not arguments.log_dir or not arguments.worker:
        raise SystemExit("queue mode needs --log-dir and --worker")

    log_dir = Path(arguments.log_dir)
    if len(jobs) != 240:
        raise RuntimeError(f"expected 240 jobs, found {len(jobs)}")
    provenance = _prepare_run(log_dir, jobs)
    print(
        f"[provenance] execution_code_sha={provenance['execution_code_sha']}",
        flush=True,
    )
    print(f"[worker] {arguments.worker} jobs={len(jobs)}", flush=True)
    while True:
        _reclaim(log_dir)
        chosen = None
        for job in jobs:
            if _job_complete(log_dir, job) or not _ready(log_dir, job, jobs):
                continue
            if _claim(log_dir, job["id"]):
                chosen = job
                break
        if chosen is None:
            if all(_job_complete(log_dir, job) for job in jobs):
                print(f"[worker] {arguments.worker} sweep complete", flush=True)
                return
            time.sleep(15)
            continue
        try:
            print(f"[job] {chosen['id']}", flush=True)
            if chosen["kind"] == "pretrain":
                run_pretrain(log_dir, chosen)
            else:
                run_adapt(log_dir, chosen)
        finally:
            _release(log_dir, chosen["id"])


if __name__ == "__main__":
    main()
