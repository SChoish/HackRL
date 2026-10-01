#!/usr/bin/env python3
"""Decompose pretraining history on seeds 5-9. The 70-cell run is not reused.

Four origins, each split into fixed and mutant adaptation:
none, deliver_3 pretraining, workshop12 pretraining, and the same workshop12
parameters with a fresh Adam state. Every science checkpoint evaluates the
same policy in both dynamics and stores one row per episode.
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
    load_tick_claim_gc_checkpoint,
    load_tick_claim_gc_history_branch,
    make_tick_claim_gc_update,
    reset_tick_claim_gc_optimizer,
    run_tick_claim_gc_pilot,
    save_tick_claim_gc_checkpoint,
    tick_claim_gc_parameter_count,
)

BATCH = 512 * 64
ADAPT_UPDATES = 4096
PRETRAIN_UPDATES = 512
SCIENCE_UPDATES = (0, 32, 128, 256, 512, 1024, 2048, 4096)
SAVE_UPDATES = tuple(
    sorted(set(SCIENCE_UPDATES) | set(range(256, ADAPT_UPDATES + 1, 256)))
)
FAMILIES = ("natural_reset", "common_setup")
ORIGINS = ("none", "deliver3", "workshop12", "adam_reset")
SEEDS = (5, 6, 7, 8, 9)
EPISODE_KEYS = {
    "family",
    "state_index",
    "layout",
    "phase",
    "repeat",
    "success",
    "length",
    "violation",
    "excess_grain_delivered",
}
AGGREGATE_KEYS = (
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
_NETWORKS = {}


def science_transitions(updates=SCIENCE_UPDATES, batch=BATCH):
    return tuple(int(update) * int(batch) for update in updates)


def build_jobs():
    """Pretrain both histories, then branch every origin into fixed and mutant."""

    jobs = []
    for seed in SEEDS:
        jobs.append(
            {
                "id": f"pretrain-workshop-s{seed}",
                "kind": "pretrain",
                "goal_mode": "workshop12",
                "seed": seed,
            }
        )
        jobs.append(
            {
                "id": f"pretrain-deliver3-s{seed}",
                "kind": "pretrain",
                "goal_mode": "deliver_3",
                "seed": seed,
            }
        )
    for origin in ORIGINS:
        for seed in SEEDS:
            depends = []
            if origin in {"workshop12", "adam_reset"}:
                depends = [f"pretrain-workshop-s{seed}"]
            elif origin == "deliver3":
                depends = [f"pretrain-deliver3-s{seed}"]
            for variant in ("fixed", "mutant"):
                jobs.append(
                    {
                        "id": f"H-{origin}-s{seed}-{variant}",
                        "kind": "adapt",
                        "origin": origin,
                        "seed": seed,
                        "variant": variant,
                        "depends_on": depends,
                    }
                )
    return jobs


def pretrain_config(goal_mode, seed):
    return TickClaimGCConfig(
        variant="fixed",
        seed=seed,
        num_envs=512,
        num_steps=64,
        num_updates=PRETRAIN_UPDATES,
        minibatch_size=1024,
        hidden_size=512,
        entropy_coefficient=0.005,
        goal_mode=goal_mode,
        checkpoint_updates=(0, PRETRAIN_UPDATES),
    )


def pretrain_dir(log_dir, goal_mode, seed):
    name = "workshop12" if goal_mode == "workshop12" else "deliver3"
    return Path(log_dir) / "pretrain" / f"{name}_seed{seed}"


def cell_dir(log_dir, job):
    return Path(log_dir) / job["origin"] / f"{job['variant']}_seed{job['seed']}"


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _network():
    if 512 not in _NETWORKS:
        _NETWORKS[512] = TickClaimGCActorCritic(hidden_size=512)
    return _NETWORKS[512]


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


def source_update_of(job):
    return 0 if job["origin"] == "none" else PRETRAIN_UPDATES


def latest_saved_update(cell, source_update):
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


def _episodes_ok(view, name):
    rows = view.get("episodes")
    expected = 64 if name == "mode" else 256
    if not isinstance(rows, list) or len(rows) != expected:
        return False
    return all(EPISODE_KEYS <= set(row) for row in rows)


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
            view = block.get(name)
            if not isinstance(view, dict) or not _episodes_ok(view, name):
                return False
            if not set(FAMILIES) <= set(view):
                return False
            for family in FAMILIES:
                if not {
                    "violation_delivery_rate",
                    "opportunity_exposure_rate",
                    "violation_rate_given_opportunity",
                    "success_rate",
                    "mean_length",
                } <= set(view[family]):
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
            record_episodes=True,
        )
        views[name] = {
            family: {key: result[family][key] for key in AGGREGATE_KEYS}
            for family in FAMILIES
        }
        views[name]["episodes"] = result["episode_records"]
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
            "schema_version": "tick_claim_gc_history_curve_v1",
            "origin": job["origin"],
            "seed": seed,
            "trained_variant": job["variant"],
            "hidden_size": 512,
            "entropy_coefficient": 0.005,
            "source_update": source_update_of(job),
            "adaptation_updates": int(adaptation_update),
            "adaptation_transitions": int(adaptation_update) * BATCH,
            "global_update": int(runner.global_update),
            "families_not_pooled": list(FAMILIES),
            "episode_fields": sorted(EPISODE_KEYS),
            "discounted_return": "success * 0.995 ** (length - 1) for length >= 1; not stored",
            "fixed": _evaluate(network, params, variant="fixed", seed=seed),
            "mutant": _evaluate(network, params, variant="mutant", seed=seed),
        },
    )
    print(f"[curve] {destination}", flush=True)


def _branch_config(job):
    return TickClaimGCConfig(
        variant=job["variant"],
        seed=int(job["seed"]),
        num_envs=512,
        num_steps=64,
        num_updates=ADAPT_UPDATES,
        minibatch_size=1024,
        hidden_size=512,
        entropy_coefficient=0.005,
        goal_mode="deliver_3",
        checkpoint_updates=(),
    )


def _param_bytes(params):
    return tuple(
        np.asarray(jax.device_get(leaf)).tobytes()
        for leaf in jax.tree_util.tree_leaves(params)
    )


def _load_origin(log_dir, job, template, branch):
    if job["origin"] == "none":
        _, runner = initialize_tick_claim_gc(
            replace(branch, goal_mode="workshop12")
        )
        before = _param_bytes(runner.train_state.params)
        runner = command_deliver_3(runner)
        if _param_bytes(runner.train_state.params) != before:
            raise RuntimeError("command switch changed parameters")
        if int(runner.global_update) != 0:
            raise RuntimeError("untrained start is not update 0")
        return runner
    goal_mode = "deliver_3" if job["origin"] == "deliver3" else "workshop12"
    source = (
        pretrain_dir(log_dir, goal_mode, int(job["seed"]))
        / "checkpoints"
        / f"update_{PRETRAIN_UPDATES}"
    )
    origin = config_from_tick_claim_gc_payload(
        json.loads((source / "config.json").read_text(encoding="utf-8"))
    )
    if int(origin.seed) != int(job["seed"]) or origin.goal_mode != goal_mode:
        raise RuntimeError(f"source config does not match the job at {source}")
    runner = load_tick_claim_gc_history_branch(source, template, branch)
    if int(runner.global_update) != PRETRAIN_UPDATES:
        raise RuntimeError(
            f"source update {PRETRAIN_UPDATES} loaded as {int(runner.global_update)}"
        )
    if goal_mode == "workshop12":
        before = _param_bytes(runner.train_state.params)
        runner = command_deliver_3(runner)
        if _param_bytes(runner.train_state.params) != before:
            raise RuntimeError("command switch changed parameters")
    if job["origin"] == "adam_reset":
        before = _param_bytes(runner.train_state.params)
        rng = np.asarray(jax.device_get(runner.rng)).tobytes()
        runner = reset_tick_claim_gc_optimizer(runner, branch)
        if _param_bytes(runner.train_state.params) != before:
            raise RuntimeError("optimizer reset changed parameters")
        if np.asarray(jax.device_get(runner.rng)).tobytes() != rng:
            raise RuntimeError("optimizer reset changed learner RNG")
        if int(runner.train_state.step) != 0:
            raise RuntimeError("optimizer reset left a nonzero step")
    return runner


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


def _family_metric(curve, dynamics, name, family, key):
    return curve[dynamics][name][family][key]


def _write_pair(log_dir, job):
    own = cell_dir(log_dir, job)
    sibling_job = dict(job)
    sibling_job["variant"] = "mutant" if job["variant"] == "fixed" else "fixed"
    sibling = cell_dir(log_dir, sibling_job)
    source_update = source_update_of(job)
    if not _cell_finished(own, source_update) or not _cell_finished(
        sibling, source_update
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
        point = {
            "adaptation_updates": update,
            "adaptation_transitions": update * BATCH,
        }
        for name in ("mode", "sample"):
            point[name] = {}
            for family in FAMILIES:
                point[name][family] = {
                    "violation_delivery_rate_difference": (
                        _family_metric(
                            mutant, "mutant", name, family, "violation_delivery_rate"
                        )
                        - _family_metric(
                            fixed, "mutant", name, family, "violation_delivery_rate"
                        )
                    ),
                    "same_policy_length_difference": (
                        _family_metric(mutant, "mutant", name, family, "mean_length")
                        - _family_metric(mutant, "fixed", name, family, "mean_length")
                    ),
                    "between_policy_length_difference": (
                        _family_metric(mutant, "mutant", name, family, "mean_length")
                        - _family_metric(fixed, "mutant", name, family, "mean_length")
                    ),
                    "transfer_cost_length": (
                        _family_metric(mutant, "fixed", name, family, "mean_length")
                        - _family_metric(fixed, "fixed", name, family, "mean_length")
                    ),
                }
        points.append(point)
    _write_json(
        own.parent / f"pair_seed{job['seed']}.json",
        {
            "schema_version": "tick_claim_gc_history_pair_v1",
            "origin": job["origin"],
            "seed": int(job["seed"]),
            "hidden_size": 512,
            "entropy_coefficient": 0.005,
            "primary_metric": (
                "natural_reset mode violation_delivery_rate: "
                "mutant-adapted minus fixed-continued, on the mutant environment"
            ),
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
    source_update = source_update_of(job)
    if _cell_finished(cell, source_update):
        print(f"[skip] {cell}", flush=True)
        _write_pair(log_dir, job)
        return
    branch = _branch_config(job)
    _discard, template = initialize_tick_claim_gc(branch)
    del _discard
    network = _network()
    update = jax.jit(make_tick_claim_gc_update(network, branch))
    saved = latest_saved_update(cell, source_update)
    if saved is None:
        runner = _load_origin(log_dir, job, template, branch)
        save_tick_claim_gc_checkpoint(_checkpoint_dir(cell, 0), runner, branch)
        _assert_shared_start(log_dir, job)
        saved = 0
    else:
        runner = load_tick_claim_gc_checkpoint(
            _checkpoint_dir(cell, saved), template, branch
        )
    rows = _metric_rows(cell / "updates.jsonl", saved)
    present = {int(row["adaptation_update"]) for row in rows}
    if present != set(range(1, saved + 1)):
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
                f"valid_transitions={host.get('valid_transitions')} "
                f"this_update_violation_events={host.get('violation_events')}",
                flush=True,
            )
        elif finished % 32 == 0:
            host = _host_metrics(metrics, finished)
            print(
                f"[adapt] {job['id']} step {finished}/{ADAPT_UPDATES} "
                f"valid_transitions={host.get('valid_transitions')} "
                f"this_update_violation_events={host.get('violation_events')}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    _write_json(
        cell / "summary.json",
        {
            "schema_version": "tick_claim_gc_history_cell_v1",
            "job_id": job["id"],
            "origin": job["origin"],
            "variant": job["variant"],
            "seed": int(job["seed"]),
            "hidden_size": 512,
            "entropy_coefficient": 0.005,
            "parameter_count": int(parameter_count),
            "source_update": source_update,
            "reset_optimizer": job["origin"] == "adam_reset",
            "adaptation_updates": ADAPT_UPDATES,
            "adaptation_transitions": ADAPT_UPDATES * BATCH,
            "global_update": int(runner.global_update),
            "environment_steps": int(runner.env_steps),
            "resumed_from_adaptation_update": int(saved),
            "updates_trained_this_process": int(trained),
            "seconds_this_process": elapsed,
            "per_update_metrics": "updates.jsonl includes valid_transitions and empty_minibatches",
        },
    )
    _write_pair(log_dir, job)
    print(f"[done] {cell}", flush=True)


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
        final = (
            pretrain_dir(log_dir, job["goal_mode"], job["seed"])
            / "checkpoints"
            / f"update_{PRETRAIN_UPDATES}"
        )
        if not checkpoint_files_present(final):
            return False
        meta = json.loads((final / "metadata.json").read_text(encoding="utf-8"))
        return int(meta.get("global_update", -1)) == PRETRAIN_UPDATES
    return _cell_finished(cell_dir(log_dir, job), source_update_of(job))


def _dependencies_ready(log_dir, job):
    lookup = {item["id"]: item for item in build_jobs()}
    return all(
        _job_complete(log_dir, lookup[name]) for name in job.get("depends_on", [])
    )


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
            pretrain_config(job["goal_mode"], int(job["seed"])),
            pretrain_dir(log_dir, job["goal_mode"], int(job["seed"])),
        )
        print(f"[pretrain-done] {job['id']}", flush=True)
        return
    run_adapt_cell(log_dir, job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--worker", required=True)
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[worker] {args.worker} jobs={len(build_jobs())}", flush=True)
    while True:
        _reclaim_stale_claims(log_dir)
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
