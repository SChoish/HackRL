#!/usr/bin/env python3
"""Overnight A/B/C deliver_3 adaptation sweep in a new run directory.

The finished 18-cell run is left untouched. Every adaptation update is
recorded on its own, and a restart continues from the latest mid-training
checkpoint rather than from adapt_0.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

if os.environ.get("HACKRL_DEVICE", "cpu") != "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

from hackrl.tick_claim_gc import (
    TickClaimGCActorCritic,
    TickClaimGCConfig,
    checkpoint_files_present,
    command_deliver_3,
    config_from_tick_claim_gc_payload,
    evaluate_tick_claim_gc_frozen,
    initialize_tick_claim_gc,
    load_tick_claim_gc_branch,
    load_tick_claim_gc_checkpoint,
    make_tick_claim_gc_update,
    run_tick_claim_gc_pilot,
    save_tick_claim_gc_checkpoint,
    tick_claim_gc_parameter_count,
)

BATCH = 512 * 64
ADAPT_UPDATES = 4096
SCIENCE_UPDATES = (0, 32, 128, 256, 512, 1024, 2048, 4096)
SAVE_UPDATES = tuple(
    sorted(set(SCIENCE_UPDATES) | set(range(256, ADAPT_UPDATES + 1, 256)))
)
FAMILIES = ("natural_reset", "common_setup")
BASE_ROOT = Path("/home/ext_csv/HackRL/runs/tick_claim_gc_workshop12_base_v1")
_NETWORKS = {}


def science_transitions(updates=SCIENCE_UPDATES, batch=BATCH):
    return tuple(int(update) * int(batch) for update in updates)


def build_jobs():
    """A, then B, then C. Pretraining sits immediately before the cells that need it."""

    jobs = [
        {"id": "pretrain-h512-s3", "kind": "pretrain", "hidden": 512, "seed": 3},
        {"id": "pretrain-h512-s4", "kind": "pretrain", "hidden": 512, "seed": 4},
    ]
    for source in (0, 128, 512):
        for seed in range(5):
            for variant in ("fixed", "mutant"):
                jobs.append(
                    {
                        "id": f"A-pre{source}-s{seed}-{variant}",
                        "kind": "adapt",
                        "phase": "A",
                        "hidden": 512,
                        "seed": seed,
                        "variant": variant,
                        "entropy": 0.005,
                        "source_update": source,
                        "depends_on": (
                            [f"pretrain-h512-s{seed}"] if seed >= 3 else []
                        ),
                    }
                )
    for entropy in (0.002, 0.01):
        token = str(entropy).replace(".", "p")
        for seed in range(5):
            for variant in ("fixed", "mutant"):
                jobs.append(
                    {
                        "id": f"B-e{token}-s{seed}-{variant}",
                        "kind": "adapt",
                        "phase": "B",
                        "hidden": 512,
                        "seed": seed,
                        "variant": variant,
                        "entropy": entropy,
                        "source_update": 512,
                        "depends_on": (
                            [f"pretrain-h512-s{seed}"] if seed >= 3 else []
                        ),
                    }
                )
    for hidden in (256, 1024):
        for seed in range(5):
            jobs.append(
                {
                    "id": f"pretrain-h{hidden}-s{seed}",
                    "kind": "pretrain",
                    "hidden": hidden,
                    "seed": seed,
                }
            )
    for hidden in (256, 1024):
        for seed in range(5):
            for variant in ("fixed", "mutant"):
                jobs.append(
                    {
                        "id": f"C-h{hidden}-s{seed}-{variant}",
                        "kind": "adapt",
                        "phase": "C",
                        "hidden": hidden,
                        "seed": seed,
                        "variant": variant,
                        "entropy": 0.005,
                        "source_update": 512,
                        "depends_on": [f"pretrain-h{hidden}-s{seed}"],
                    }
                )
    return jobs


def pretrain_config(hidden, seed):
    return TickClaimGCConfig(
        variant="fixed",
        seed=seed,
        num_envs=512,
        num_steps=64,
        num_updates=512,
        minibatch_size=1024,
        hidden_size=hidden,
        entropy_coefficient=0.005,
        goal_mode="workshop12",
        checkpoint_updates=(0, 128, 512),
    )


def pretrain_dir(log_dir, hidden, seed):
    return Path(log_dir) / "pretrain" / f"hidden{hidden}_seed{seed}"


def source_checkpoint(log_dir, hidden, seed, update):
    if hidden == 512 and seed <= 2:
        return (
            BASE_ROOT
            / f"fixed_seed{seed}"
            / "checkpoints"
            / f"update_{update}"
        )
    return pretrain_dir(log_dir, hidden, seed) / "checkpoints" / f"update_{update}"


def cell_dir(log_dir, job):
    log_dir = Path(log_dir)
    name = f"{job['variant']}_seed{job['seed']}"
    if job["phase"] == "A":
        return log_dir / "A" / f"pre_{job['source_update'] * BATCH}" / name
    if job["phase"] == "B":
        token = str(job["entropy"]).replace(".", "p")
        return log_dir / "B" / f"entropy_{token}" / name
    return log_dir / "C" / f"hidden_{job['hidden']}" / name


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_config(checkpoint):
    if not checkpoint_files_present(checkpoint):
        raise FileNotFoundError(f"source checkpoint is missing: {checkpoint}")
    recorded = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    return config_from_tick_claim_gc_payload(recorded)


def _network(hidden):
    if hidden not in _NETWORKS:
        _NETWORKS[hidden] = TickClaimGCActorCritic(hidden_size=hidden)
    return _NETWORKS[hidden]


def _host_metrics(metrics, adaptation_update):
    host = {
        key: np.asarray(jax.device_get(value)).tolist()
        for key, value in metrics.items()
    }
    host["adaptation_update"] = int(adaptation_update)
    host["aggregation"] = "this_update_only"
    return host


def _metric_rows(path, completed):
    path = Path(path)
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("adaptation_update", -1)) <= completed:
                rows.append(row)
    rows.sort(key=lambda row: int(row["adaptation_update"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def _append_metric(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _checkpoint_dir(cell, adaptation_update):
    return Path(cell) / "checkpoints" / f"adapt_{adaptation_update}"


def latest_saved_update(cell, source_update):
    """Highest checkpoint whose stored update matches this adaptation step."""

    best = None
    for update in SAVE_UPDATES:
        directory = _checkpoint_dir(cell, update)
        if not checkpoint_files_present(directory):
            continue
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if int(meta.get("global_update", -1)) != int(source_update) + int(update):
            continue
        best = int(update)
    return best


def _curve_path(cell, adaptation_update):
    return Path(cell) / "curve" / f"adapt_{adaptation_update}.json"


def curve_complete(path, adaptation_update):
    path = Path(path)
    if not path.is_file():
        return False
    document = json.loads(path.read_text(encoding="utf-8"))
    if int(document.get("adaptation_updates", -1)) != int(adaptation_update):
        return False
    for dynamics in ("fixed", "mutant"):
        block = document.get(dynamics)
        if not isinstance(block, dict):
            return False
        for name in ("mode", "sample"):
            families = block.get(name)
            if not isinstance(families, dict):
                return False
            if not set(FAMILIES) <= set(families):
                return False
            for family in FAMILIES:
                if not {
                    "violation_delivery_rate",
                    "opportunity_exposure_rate",
                    "violation_rate_given_opportunity",
                } <= set(families[family]):
                    return False
    return True


def _evaluate(network, params, *, variant, seed):
    views = {}
    for stochastic, repeats, name in ((False, 1, "mode"), (True, 4, "sample")):
        result = evaluate_tick_claim_gc_frozen(
            network,
            params,
            variant=variant,
            stochastic=stochastic,
            repeats_per_state=repeats,
            seed_base=20000,
            learner_seed=seed,
        )
        views[name] = {
            family: {
                key: result[family][key]
                for key in (
                    "episodes",
                    "success_rate",
                    "mean_length",
                    "violation_rate",
                    "repeated_violation_rate",
                    "violation_delivery_rate",
                    "mean_violation_grain_delivered",
                    "mean_first_violation_step",
                    "opportunity_exposure_rate",
                    "violation_rate_given_opportunity",
                )
            }
            for family in FAMILIES
        }
    return views


def _write_curve(cell, job, runner, adaptation_update, network):
    destination = _curve_path(cell, adaptation_update)
    if curve_complete(destination, adaptation_update):
        return
    params = runner.train_state.params
    seed = int(job["seed"])
    _write_json(
        destination,
        {
            "schema_version": "tick_claim_gc_adapt_night_curve_v1",
            "phase": job["phase"],
            "seed": seed,
            "trained_variant": job["variant"],
            "hidden_size": int(job["hidden"]),
            "entropy_coefficient": float(job["entropy"]),
            "source_update": int(job["source_update"]),
            "source_transitions": int(job["source_update"]) * BATCH,
            "adaptation_updates": int(adaptation_update),
            "adaptation_transitions": int(adaptation_update) * BATCH,
            "global_update": int(runner.global_update),
            "families_not_pooled": list(FAMILIES),
            "fixed": _evaluate(network, params, variant="fixed", seed=seed),
            "mutant": _evaluate(network, params, variant="mutant", seed=seed),
        },
    )
    print(f"[curve] {destination}", flush=True)


def _branch_config(origin, job):
    return replace(
        origin,
        variant=job["variant"],
        goal_mode="deliver_3",
        num_updates=ADAPT_UPDATES,
        checkpoint_updates=(),
        entropy_coefficient=float(job["entropy"]),
    )


def _cell_finished(cell, source_update):
    summary_path = Path(cell) / "summary.json"
    final_dir = _checkpoint_dir(cell, ADAPT_UPDATES)
    if not summary_path.is_file() or not checkpoint_files_present(final_dir):
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    meta = json.loads((final_dir / "metadata.json").read_text(encoding="utf-8"))
    if int(summary.get("adaptation_updates", -1)) != ADAPT_UPDATES:
        return False
    if int(meta.get("global_update", -1)) != int(source_update) + ADAPT_UPDATES:
        return False
    return all(
        curve_complete(_curve_path(cell, update), update)
        for update in SCIENCE_UPDATES
    )


def _write_pair(log_dir, job):
    if job["variant"] not in ("fixed", "mutant"):
        return
    own = cell_dir(log_dir, job)
    sibling_job = dict(job)
    sibling_job["variant"] = "mutant" if job["variant"] == "fixed" else "fixed"
    sibling = cell_dir(log_dir, sibling_job)
    if not _cell_finished(own, job["source_update"]) or not _cell_finished(
        sibling, job["source_update"]
    ):
        return
    mutant_job = sibling_job if job["variant"] == "fixed" else job
    fixed_job = job if job["variant"] == "fixed" else sibling_job
    points = []
    for update in SCIENCE_UPDATES:
        mutant = json.loads(
            _curve_path(cell_dir(log_dir, mutant_job), update).read_text(
                encoding="utf-8"
            )
        )
        fixed = json.loads(
            _curve_path(cell_dir(log_dir, fixed_job), update).read_text(
                encoding="utf-8"
            )
        )
        difference = {}
        for name in ("mode", "sample"):
            difference[name] = {}
            for family in FAMILIES:
                difference[name][family] = (
                    mutant["mutant"][name][family]["violation_delivery_rate"]
                    - fixed["mutant"][name][family]["violation_delivery_rate"]
                )
        points.append(
            {
                "adaptation_updates": update,
                "adaptation_transitions": update * BATCH,
                "violation_delivery_rate_difference": difference,
            }
        )
    parent = own.parent
    _write_json(
        parent / f"pair_seed{job['seed']}.json",
        {
            "schema_version": "tick_claim_gc_adapt_night_pair_v1",
            "phase": job["phase"],
            "seed": int(job["seed"]),
            "hidden_size": int(job["hidden"]),
            "entropy_coefficient": float(job["entropy"]),
            "source_update": int(job["source_update"]),
            "metric": "mutant_eval violation_delivery_rate: mutant-adapted minus fixed-continued",
            "families_not_pooled": list(FAMILIES),
            "points": points,
        },
    )


def _assert_shared_start(log_dir, job):
    other = dict(job)
    other["variant"] = "mutant" if job["variant"] == "fixed" else "fixed"
    left = _checkpoint_dir(cell_dir(log_dir, job), 0) / "state.msgpack"
    right = _checkpoint_dir(cell_dir(log_dir, other), 0) / "state.msgpack"
    if not left.is_file() or not right.is_file():
        return
    if left.read_bytes() != right.read_bytes():
        raise RuntimeError(f"fixed and mutant adapt_0 states differ for {job['id']}")


def run_adapt_cell(log_dir, job):
    cell = cell_dir(log_dir, job)
    source_update = int(job["source_update"])
    if _cell_finished(cell, source_update):
        print(f"[skip] {cell}", flush=True)
        _write_pair(log_dir, job)
        return
    source = source_checkpoint(
        log_dir, int(job["hidden"]), int(job["seed"]), source_update
    )
    origin = _read_config(source)
    if int(origin.seed) != int(job["seed"]) or int(origin.hidden_size) != int(job["hidden"]):
        raise RuntimeError(f"source config does not match the job at {source}")
    branch = _branch_config(origin, job)
    _discard, template = initialize_tick_claim_gc(branch)
    del _discard
    network = _network(int(job["hidden"]))
    update = jax.jit(make_tick_claim_gc_update(network, branch))
    saved = latest_saved_update(cell, source_update)
    if saved is None:
        runner = load_tick_claim_gc_branch(source, template, branch)
        if int(runner.global_update) != source_update:
            raise RuntimeError(
                f"source update {source_update} loaded as {int(runner.global_update)}"
            )
        before = jax.tree_util.tree_leaves(runner.train_state.params)
        before_bytes = tuple(np.asarray(jax.device_get(leaf)).tobytes() for leaf in before)
        runner = command_deliver_3(runner)
        after_bytes = tuple(
            np.asarray(jax.device_get(leaf)).tobytes()
            for leaf in jax.tree_util.tree_leaves(runner.train_state.params)
        )
        if before_bytes != after_bytes:
            raise RuntimeError("command switch changed parameters")
        save_tick_claim_gc_checkpoint(_checkpoint_dir(cell, 0), runner, branch)
        _assert_shared_start(log_dir, job)
        saved = 0
    else:
        runner = load_tick_claim_gc_checkpoint(
            _checkpoint_dir(cell, saved), template, branch
        )
    rows = _metric_rows(cell / "updates.jsonl", saved)
    present = {int(row["adaptation_update"]) for row in rows}
    expected = set(range(1, saved + 1))
    if present != expected:
        print(
            f"[metrics] {cell} has {len(present)} per-update rows through saved step {saved}",
            flush=True,
        )
    if saved in SCIENCE_UPDATES:
        _write_curve(cell, job, runner, saved, network)
    target = source_update + ADAPT_UPDATES
    parameter_count = tick_claim_gc_parameter_count(runner.train_state.params)
    started = time.perf_counter()
    trained = 0
    while int(runner.global_update) < target:
        step_started = time.perf_counter()
        runner, metrics = update(runner)
        jax.block_until_ready(runner.global_update)
        finished = int(runner.global_update) - source_update
        _append_metric(cell / "updates.jsonl", _host_metrics(metrics, finished))
        trained += 1
        if finished in SAVE_UPDATES:
            save_tick_claim_gc_checkpoint(
                _checkpoint_dir(cell, finished), runner, branch
            )
        if finished in SCIENCE_UPDATES:
            _write_curve(cell, job, runner, finished, network)
            elapsed = time.perf_counter() - step_started
            host = _host_metrics(metrics, finished)
            print(
                f"[adapt] {job['id']} step {finished}/{ADAPT_UPDATES} "
                f"seconds={elapsed:.2f} "
                f"this_update_violation_events={host.get('violation_events')}",
                flush=True,
            )
        elif finished % 32 == 0:
            host = _host_metrics(metrics, finished)
            print(
                f"[adapt] {job['id']} step {finished}/{ADAPT_UPDATES} "
                f"this_update_violation_events={host.get('violation_events')}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    _write_json(
        cell / "summary.json",
        {
            "schema_version": "tick_claim_gc_adapt_night_cell_v1",
            "job_id": job["id"],
            "phase": job["phase"],
            "variant": job["variant"],
            "seed": int(job["seed"]),
            "hidden_size": int(job["hidden"]),
            "entropy_coefficient": float(job["entropy"]),
            "parameter_count": int(parameter_count),
            "source_update": source_update,
            "source_checkpoint": str(source),
            "adaptation_updates": ADAPT_UPDATES,
            "adaptation_transitions": ADAPT_UPDATES * BATCH,
            "global_update": int(runner.global_update),
            "environment_steps": int(runner.env_steps),
            "resumed_from_adaptation_update": int(saved),
            "updates_trained_this_process": int(trained),
            "seconds_this_process": elapsed,
            "per_update_metrics": "updates.jsonl aggregation=this_update_only",
        },
    )
    _record_throughput(log_dir, int(job["hidden"]), elapsed, trained)
    _write_pair(log_dir, job)
    print(f"[done] {cell}", flush=True)


def _record_throughput(log_dir, hidden, elapsed, trained):
    if trained < 128:
        return
    path = Path(log_dir) / "throughput.json"
    document = {}
    if path.is_file():
        document = json.loads(path.read_text(encoding="utf-8"))
    key = f"hidden_{hidden}"
    if key in document:
        return
    per_update = elapsed / trained
    same_width = [
        job
        for job in build_jobs()
        if job["kind"] == "adapt" and int(job["hidden"]) == hidden
    ]
    document[key] = {
        "seconds_per_update": per_update,
        "measured_updates": int(trained),
        "projected_cell_hours": per_update * ADAPT_UPDATES / 3600,
        "projected_width_hours_one_gpu": per_update
        * ADAPT_UPDATES
        * len(same_width)
        / 3600,
    }
    _write_json(path, document)
    print(
        f"[eta] width {hidden}: {per_update:.3f}s/update, "
        f"one cell {document[key]['projected_cell_hours']:.2f}h",
        flush=True,
    )


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reclaim_stale_claims(log_dir):
    claims = Path(log_dir) / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    for path in claims.iterdir():
        try:
            pid = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if not _pid_alive(pid):
            path.unlink(missing_ok=True)


def _job_complete(log_dir, job):
    if job["kind"] == "pretrain":
        directory = pretrain_dir(log_dir, job["hidden"], job["seed"])
        final = directory / "checkpoints" / "update_512"
        if not checkpoint_files_present(final):
            return False
        meta = json.loads((final / "metadata.json").read_text(encoding="utf-8"))
        return int(meta.get("global_update", -1)) == 512
    return _cell_finished(cell_dir(log_dir, job), int(job["source_update"]))


def _dependencies_ready(log_dir, job):
    lookup = {item["id"]: item for item in build_jobs()}
    return all(_job_complete(log_dir, lookup[name]) for name in job.get("depends_on", []))


def _claim(log_dir, job_id):
    path = Path(log_dir) / "claims" / job_id
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def _release(log_dir, job_id):
    (Path(log_dir) / "claims" / job_id).unlink(missing_ok=True)


def _ensure_deadline(log_dir, hours):
    path = Path(log_dir) / "deadline.json"
    if path.is_file():
        return float(json.loads(path.read_text(encoding="utf-8"))["deadline_epoch"])
    payload = {
        "started_epoch": time.time(),
        "deadline_epoch": time.time() + float(hours) * 3600,
        "hours": float(hours),
    }
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return float(json.loads(path.read_text(encoding="utf-8"))["deadline_epoch"])
    os.write(fd, json.dumps(payload, sort_keys=True).encode())
    os.close(fd)
    return float(payload["deadline_epoch"])


def _next_job(log_dir):
    for job in build_jobs():
        if _job_complete(log_dir, job):
            continue
        if not _dependencies_ready(log_dir, job):
            continue
        if _claim(log_dir, job["id"]):
            return job
    return None


def _run_job(log_dir, job):
    print(f"[job] {job['id']}", flush=True)
    if job["kind"] == "pretrain":
        run_tick_claim_gc_pilot(
            pretrain_config(int(job["hidden"]), int(job["seed"])),
            pretrain_dir(log_dir, int(job["hidden"]), int(job["seed"])),
        )
        print(f"[pretrain-done] {job['id']}", flush=True)
        return
    run_adapt_cell(log_dir, job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--hours", type=float, default=8.0)
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    deadline = _ensure_deadline(log_dir, args.hours)
    print(
        f"[worker] {args.worker} deadline_epoch={deadline:.0f} jobs={len(build_jobs())}",
        flush=True,
    )
    while True:
        _reclaim_stale_claims(log_dir)
        if time.time() >= deadline:
            pending = [
                job["id"]
                for job in build_jobs()
                if not _job_complete(log_dir, job)
            ]
            print(
                f"[deadline] {args.worker} stops claiming; pending={len(pending)}",
                flush=True,
            )
            return
        job = _next_job(log_dir)
        if job is None:
            if all(_job_complete(log_dir, item) for item in build_jobs()):
                print(f"[worker] {args.worker} sweep complete", flush=True)
                return
            time.sleep(15)
            continue
        try:
            _run_job(log_dir, job)
        finally:
            _release(log_dir, job["id"])


if __name__ == "__main__":
    main()
