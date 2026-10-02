#!/usr/bin/env python3
"""PACK-RESTORE history comparison on the source-growth fixture.

Three origins, each split into fixed continued learning and mutant adaptation:
no pretraining, deliver_3 pretraining, and workshop12 pretraining. Every
science checkpoint evaluates the same policy in both kernels.
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

from hackrl.pack_restore_gc import (
    PackRestoreGCActorCritic,
    PackRestoreGCConfig,
    checkpoint_files_present,
    command_deliver_3,
    config_from_pack_restore_gc_payload,
    evaluate_pack_restore_gc_frozen,
    initialize_pack_restore_gc,
    load_pack_restore_gc_checkpoint,
    load_pack_restore_gc_history_branch,
    make_pack_restore_gc_update,
    pack_restore_gc_parameter_count,
    save_pack_restore_gc_checkpoint,
)

BATCH = 512 * 64
ADAPT_UPDATES = 4096
PRETRAIN_UPDATES = 512
SCIENCE_UPDATES = (0, 32, 128, 256, 512, 1024, 2048, 4096)
SAVE_UPDATES = tuple(sorted(set(SCIENCE_UPDATES) | set(range(256, ADAPT_UPDATES + 1, 256))))
FAMILIES = ("natural_reset", "common_setup")
ORIGINS = ("none", "deliver3", "workshop12")
SEEDS = (0, 1, 2)
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
FAMILY_KEYS = (
    "success_rate",
    "mean_length",
    "mean_discounted_return",
    "violation_rate",
    "violation_delivery_rate",
    "opportunity_exposure_rate",
    "violation_rate_given_opportunity",
)
_NETWORKS = {}


def build_jobs():
    jobs = []
    for seed in SEEDS:
        for goal_mode in ("workshop12", "deliver_3"):
            jobs.append(
                {
                    "id": f"pretrain-{goal_mode}-s{seed}",
                    "kind": "pretrain",
                    "goal_mode": goal_mode,
                    "seed": seed,
                }
            )
    for origin in ORIGINS:
        for seed in SEEDS:
            depends = []
            if origin == "workshop12":
                depends = [f"pretrain-workshop12-s{seed}"]
            elif origin == "deliver3":
                depends = [f"pretrain-deliver_3-s{seed}"]
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
    return PackRestoreGCConfig(
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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _network():
    if 512 not in _NETWORKS:
        _NETWORKS[512] = PackRestoreGCActorCritic(hidden_size=512)
    return _NETWORKS[512]


def _host_metrics(metrics, adaptation_update):
    host = {
        key: np.asarray(jax.device_get(value)).tolist()
        for key, value in metrics.items()
    }
    host["adaptation_update"] = int(adaptation_update)
    return host


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
            for family in FAMILIES:
                if not set(FAMILY_KEYS) <= set(view.get(family, {})):
                    return False
    return True


def _evaluate(network, params, *, variant, seed):
    views = {}
    for stochastic, repeats, name in ((False, 1, "mode"), (True, 4, "sample")):
        result = evaluate_pack_restore_gc_frozen(
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
            family: {key: result[family][key] for key in FAMILY_KEYS} for family in FAMILIES
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
            "schema_version": "pack_restore_gc_history_curve_v1",
            "origin": job["origin"],
            "seed": seed,
            "trained_variant": job["variant"],
            "source_update": source_update_of(job),
            "adaptation_updates": int(adaptation_update),
            "global_update": int(runner.global_update),
            "discounted_return": "success * 0.995 ** (length - 1) for length >= 1",
            "fixed": _evaluate(network, params, variant="fixed", seed=seed),
            "mutant": _evaluate(network, params, variant="mutant", seed=seed),
        },
    )
    print(f"[curve] {destination}", flush=True)


def _branch_config(job):
    return PackRestoreGCConfig(
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
        np.asarray(jax.device_get(leaf)).tobytes() for leaf in jax.tree_util.tree_leaves(params)
    )


def _load_origin(log_dir, job, template, branch):
    if job["origin"] == "none":
        _, runner = initialize_pack_restore_gc(replace(branch, goal_mode="workshop12"))
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
    origin = config_from_pack_restore_gc_payload(
        json.loads((source / "config.json").read_text(encoding="utf-8"))
    )
    if int(origin.seed) != int(job["seed"]) or origin.goal_mode != goal_mode:
        raise RuntimeError(f"source config does not match the job at {source}")
    runner = load_pack_restore_gc_history_branch(source, template, branch)
    if int(runner.global_update) != PRETRAIN_UPDATES:
        raise RuntimeError(f"source update loaded as {int(runner.global_update)}")
    if goal_mode == "workshop12":
        before = _param_bytes(runner.train_state.params)
        runner = command_deliver_3(runner)
        if _param_bytes(runner.train_state.params) != before:
            raise RuntimeError("command switch changed parameters")
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
    return all(curve_complete(_curve_path(cell, update), update) for update in SCIENCE_UPDATES)


def _family_metric(curve, dynamics, name, family, key):
    return curve[dynamics][name][family][key]


def _write_pair(log_dir, job):
    sibling_job = dict(job)
    sibling_job["variant"] = "mutant" if job["variant"] == "fixed" else "fixed"
    source_update = source_update_of(job)
    if not _cell_finished(cell_dir(log_dir, job), source_update) or not _cell_finished(
        cell_dir(log_dir, sibling_job), source_update
    ):
        return
    mutant_job = sibling_job if job["variant"] == "fixed" else job
    fixed_job = job if job["variant"] == "fixed" else sibling_job
    points = []
    for update in SCIENCE_UPDATES:
        mutant = json.loads(
            _curve_path(cell_dir(log_dir, mutant_job), update).read_text(encoding="utf-8")
        )
        fixed = json.loads(
            _curve_path(cell_dir(log_dir, fixed_job), update).read_text(encoding="utf-8")
        )
        point = {"adaptation_updates": update}
        for name in ("mode", "sample"):
            point[name] = {}
            for family in FAMILIES:
                point[name][family] = {
                    "success_rate_mutant_policy_mutant_env": _family_metric(
                        mutant, "mutant", name, family, "success_rate"
                    ),
                    "success_rate_fixed_policy_mutant_env": _family_metric(
                        fixed, "mutant", name, family, "success_rate"
                    ),
                    "violation_rate_mutant_policy_mutant_env": _family_metric(
                        mutant, "mutant", name, family, "violation_rate"
                    ),
                    "violation_delivery_rate_difference": (
                        _family_metric(mutant, "mutant", name, family, "violation_delivery_rate")
                        - _family_metric(fixed, "mutant", name, family, "violation_delivery_rate")
                    ),
                    "same_policy_length_difference": (
                        _family_metric(mutant, "mutant", name, family, "mean_length")
                        - _family_metric(mutant, "fixed", name, family, "mean_length")
                    ),
                    "same_policy_return_difference": (
                        _family_metric(mutant, "mutant", name, family, "mean_discounted_return")
                        - _family_metric(mutant, "fixed", name, family, "mean_discounted_return")
                    ),
                }
        points.append(point)
    _write_json(
        cell_dir(log_dir, job).parent / f"pair_seed{job['seed']}.json",
        {
            "schema_version": "pack_restore_gc_history_pair_v1",
            "origin": job["origin"],
            "seed": int(job["seed"]),
            "primary_metric": (
                "natural_reset mode violation_delivery_rate: "
                "mutant-adapted minus fixed-continued, on the mutant environment"
            ),
            "reference_path_wording": "shortening versus the 20-step normal reference, not an optimum",
            "points": points,
        },
    )


def run_pretrain(log_dir, job):
    config = pretrain_config(job["goal_mode"], int(job["seed"]))
    destination = pretrain_dir(log_dir, job["goal_mode"], int(job["seed"]))
    final = destination / "checkpoints" / f"update_{PRETRAIN_UPDATES}"
    if checkpoint_files_present(final):
        meta = json.loads((final / "metadata.json").read_text(encoding="utf-8"))
        if int(meta.get("global_update", -1)) == PRETRAIN_UPDATES:
            print(f"[skip] {destination}", flush=True)
            return
    network, runner = initialize_pack_restore_gc(config)
    update = jax.jit(make_pack_restore_gc_update(network, config))
    save_pack_restore_gc_checkpoint(destination / "checkpoints" / "update_0", runner, config)
    started = time.perf_counter()
    while int(runner.global_update) < PRETRAIN_UPDATES:
        runner, metrics = update(runner)
        jax.block_until_ready(runner.global_update)
        finished = int(runner.global_update)
        if finished % 32 == 0 or finished == PRETRAIN_UPDATES:
            host = _host_metrics(metrics, finished)
            print(
                f"[pretrain] {job['id']} step {finished}/{PRETRAIN_UPDATES} "
                f"valid_transitions={host.get('valid_transitions')} "
                f"goal_successes={host.get('goal_successes')}",
                flush=True,
            )
    save_pack_restore_gc_checkpoint(final, runner, config)
    _write_json(
        destination / "summary.json",
        {
            "goal_mode": job["goal_mode"],
            "seed": int(job["seed"]),
            "updates": PRETRAIN_UPDATES,
            "parameter_count": pack_restore_gc_parameter_count(runner.train_state.params),
            "seconds": time.perf_counter() - started,
            "source_growth_period": 8,
        },
    )


def run_adapt_cell(log_dir, job):
    cell = cell_dir(log_dir, job)
    source_update = source_update_of(job)
    if _cell_finished(cell, source_update):
        print(f"[skip] {cell}", flush=True)
        _write_pair(log_dir, job)
        return
    branch = _branch_config(job)
    _discard, template = initialize_pack_restore_gc(branch)
    del _discard
    network = _network()
    update = jax.jit(make_pack_restore_gc_update(network, branch))
    saved = latest_saved_update(cell, source_update)
    if saved is None:
        runner = _load_origin(log_dir, job, template, branch)
        save_pack_restore_gc_checkpoint(_checkpoint_dir(cell, 0), runner, branch)
        saved = 0
    else:
        runner = load_pack_restore_gc_checkpoint(_checkpoint_dir(cell, saved), template, branch)
    if saved in SCIENCE_UPDATES:
        _write_curve(cell, job, runner, saved, network)
    target = source_update + ADAPT_UPDATES
    parameter_count = pack_restore_gc_parameter_count(runner.train_state.params)
    started = time.perf_counter()
    trained = 0
    while int(runner.global_update) < target:
        runner, metrics = update(runner)
        jax.block_until_ready(runner.global_update)
        finished = int(runner.global_update) - source_update
        _append_metric(cell / "updates.jsonl", _host_metrics(metrics, finished))
        trained += 1
        if finished in SAVE_UPDATES:
            save_pack_restore_gc_checkpoint(_checkpoint_dir(cell, finished), runner, branch)
        if finished in SCIENCE_UPDATES:
            _write_curve(cell, job, runner, finished, network)
            host = _host_metrics(metrics, finished)
            print(
                f"[adapt] {job['id']} step {finished}/{ADAPT_UPDATES} "
                f"valid_transitions={host.get('valid_transitions')} "
                f"violation_events={host.get('violation_events')} "
                f"goal_successes={host.get('goal_successes')}",
                flush=True,
            )
        elif finished % 32 == 0:
            host = _host_metrics(metrics, finished)
            print(
                f"[adapt] {job['id']} step {finished}/{ADAPT_UPDATES} "
                f"valid_transitions={host.get('valid_transitions')}",
                flush=True,
            )
    _write_json(
        cell / "summary.json",
        {
            "schema_version": "pack_restore_gc_history_cell_v1",
            "job_id": job["id"],
            "origin": job["origin"],
            "variant": job["variant"],
            "seed": int(job["seed"]),
            "parameter_count": int(parameter_count),
            "source_update": source_update,
            "adaptation_updates": ADAPT_UPDATES,
            "global_update": int(runner.global_update),
            "updates_trained_this_process": int(trained),
            "seconds_this_process": time.perf_counter() - started,
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


def _next_job(log_dir):
    for job in build_jobs():
        if _job_complete(log_dir, job):
            continue
        if not _dependencies_ready(log_dir, job):
            continue
        if _claim(log_dir, job["id"]):
            return job
    return None


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
            print(f"[job] {job['id']}", flush=True)
            if job["kind"] == "pretrain":
                run_pretrain(log_dir, job)
            else:
                run_adapt_cell(log_dir, job)
        finally:
            _release(log_dir, job["id"])


if __name__ == "__main__":
    main()
