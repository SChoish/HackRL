#!/usr/bin/env python3
"""Compare reservation goals with the rest of the workshop goal set.

Pretraining uses one masked-goal contract for every condition: sample a
currently false allowed goal, and natural-reset when none remain. Adaptation
is deliver_3 for 4096 updates. This run is separate from the history sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

if os.environ.get("HACKRL_DEVICE", "cpu") != "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
from flax import serialization

from hackrl.tick_claim import GOAL_IDS
from hackrl.tick_claim_gc import (
    DELIVER_3_GOAL_INDEX,
    NUM_GOALS,
    TickClaimGCActorCritic,
    TickClaimGCConfig,
    checkpoint_files_present,
    command_deliver_3,
    config_from_tick_claim_gc_payload,
    evaluate_tick_claim_gc_frozen,
    initialize_tick_claim_gc,
    load_tick_claim_gc_checkpoint,
    make_tick_claim_gc_update,
    reinit_tick_claim_gc_adaptation_start,
    reset_tick_claim_gc_optimizer,
    save_tick_claim_gc_checkpoint,
    tick_claim_gc_parameter_count,
)
from eval_tick_claim_reservation_probe import (
    build_reservation_probe_states,
    evaluate_policy,
)

BATCH = 512 * 64
VALID_BUDGET = 16_777_216
PHYSICAL_CAP_UPDATES = 2048
ADAPT_UPDATES = 4096
SCIENCE_UPDATES = (0, 32, 128, 256, 512, 1024, 2048, 4096)
SAVE_UPDATES = tuple(
    sorted(set(SCIENCE_UPDATES) | set(range(256, ADAPT_UPDATES + 1, 256)))
)
FAMILIES = ("natural_reset", "common_setup")
CONDITIONS = ("A", "AR", "B", "BR")
SEEDS = (10, 11, 12, 13, 14)
PRESENT = GOAL_IDS.index("facility/reservation_present")
ABSENT = GOAL_IDS.index("facility/reservation_absent")
NON_RESERVATION = tuple(
    index for index in range(NUM_GOALS) if index not in {PRESENT, ABSENT}
)
ALLOWED = {
    "A": (DELIVER_3_GOAL_INDEX,),
    "AR": (DELIVER_3_GOAL_INDEX, PRESENT, ABSENT),
    "B": NON_RESERVATION,
    "BR": tuple(range(NUM_GOALS)),
}
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
_PROBE_STATES = None


def sampling_weights(allowed):
    """Match reservation-goal weights in the two conditions that include them.

    When every allowed goal is false, each reservation goal has probability
    1/12, the same as one goal in the uniform 12-goal set. The remaining
    mass stays on the other allowed goals. Conditions without reservation
    goals stay uniform on their own set.
    """

    weights = [0.0] * NUM_GOALS
    reservation = [goal for goal in allowed if goal in {PRESENT, ABSENT}]
    others = [goal for goal in allowed if goal not in {PRESENT, ABSENT}]
    if reservation:
        share = (NUM_GOALS - len(reservation)) / len(others)
        for goal in reservation:
            weights[goal] = 1.0
        for goal in others:
            weights[goal] = share
    else:
        for goal in allowed:
            weights[goal] = 1.0
    return tuple(weights)


def build_jobs():
    jobs = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            jobs.append(
                {
                    "id": f"pretrain-{condition}-s{seed}",
                    "kind": "pretrain",
                    "condition": condition,
                    "seed": seed,
                }
            )
    for condition in CONDITIONS:
        for seed in SEEDS:
            for variant in ("fixed", "mutant"):
                jobs.append(
                    {
                        "id": f"G-{condition}-s{seed}-{variant}",
                        "kind": "adapt",
                        "condition": condition,
                        "seed": seed,
                        "variant": variant,
                        "depends_on": [f"pretrain-{condition}-s{seed}"],
                    }
                )
    return jobs


def pretrain_config(condition, seed):
    allowed = ALLOWED[condition]
    return TickClaimGCConfig(
        variant="fixed",
        seed=int(seed),
        num_envs=512,
        num_steps=64,
        num_updates=PHYSICAL_CAP_UPDATES,
        minibatch_size=1024,
        hidden_size=512,
        learning_rate=2e-4,
        entropy_coefficient=0.005,
        goal_mode="masked",
        allowed_goals=allowed,
        goal_sampling_weights=sampling_weights(allowed),
        checkpoint_updates=(),
    )


def adapt_config(job):
    return TickClaimGCConfig(
        variant=job["variant"],
        seed=int(job["seed"]),
        num_envs=512,
        num_steps=64,
        num_updates=ADAPT_UPDATES,
        minibatch_size=1024,
        hidden_size=512,
        learning_rate=2e-4,
        entropy_coefficient=0.005,
        goal_mode="deliver_3",
        checkpoint_updates=(),
    )


def pretrain_dir(log_dir, condition, seed):
    return Path(log_dir) / "pretrain" / f"{condition}_seed{seed}"


def cell_dir(log_dir, job):
    return (
        Path(log_dir) / job["condition"] / f"{job['variant']}_seed{job['seed']}"
    )


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


def _host_metrics(metrics, update):
    host = {
        key: np.asarray(jax.device_get(value)).tolist()
        for key, value in metrics.items()
    }
    host["update"] = int(update)
    host["aggregation"] = "this_update_only"
    return host


def _metric_rows(path, completed, key):
    path = Path(path)
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get(key, -1)) <= completed:
                rows.append(row)
    rows.sort(key=lambda row: int(row[key]))
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


def _sum_field(rows, name):
    total = 0
    for row in rows:
        value = row.get(name, 0)
        if isinstance(value, list):
            continue
        total += int(value)
    return total


def _sum_vector(rows, name):
    total = [0] * NUM_GOALS
    for row in rows:
        value = row.get(name)
        if not isinstance(value, list) or len(value) != NUM_GOALS:
            continue
        for index, item in enumerate(value):
            total[index] += int(item)
    return total


def _checkpoint_dir(cell, adaptation_update):
    return Path(cell) / "checkpoints" / f"adapt_{adaptation_update}"


def _pretrain_checkpoint(directory, update):
    return Path(directory) / "checkpoints" / f"update_{update}"


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
    if "reservation_probe" not in document:
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


def _discounted_return(episodes, family):
    rows = [row for row in episodes if row["family"] == family]
    if not rows:
        raise ValueError(f"no episodes for {family}")
    total = 0.0
    for row in rows:
        length = int(row["length"])
        if int(row["success"]) and length >= 1:
            total += 0.995 ** (length - 1)
    return total / len(rows)


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


def _probe(network, params):
    global _PROBE_STATES
    if _PROBE_STATES is None:
        _PROBE_STATES = build_reservation_probe_states()
    evaluated = evaluate_policy(network, params, _PROBE_STATES)
    return {
        name: {
            "do_probability": row["do_probability"],
            "mode_action": row["mode_action"],
            "mode_is_do": row["mode_is_do"],
            "mutant_payout": row["mutant"]["payout"],
            "fixed_payout": row["fixed"]["payout"],
        }
        for name, row in evaluated.items()
    }


def _curve_path(cell, adaptation_update):
    return Path(cell) / "curve" / f"adapt_{adaptation_update}.json"


def _write_curve(cell, job, runner, adaptation_update, network):
    destination = _curve_path(cell, adaptation_update)
    if curve_complete(destination, adaptation_update):
        return
    params = runner.train_state.params
    seed = int(job["seed"])
    _write_json(
        destination,
        {
            "schema_version": "tick_claim_gc_goalset_curve_v1",
            "condition": job["condition"],
            "seed": seed,
            "trained_variant": job["variant"],
            "hidden_size": 512,
            "entropy_coefficient": 0.005,
            "learning_rate": 2e-4,
            "adaptation_updates": int(adaptation_update),
            "adaptation_transitions": int(adaptation_update) * BATCH,
            "global_update": int(runner.global_update),
            "families_not_pooled": list(FAMILIES),
            "episode_fields": sorted(EPISODE_KEYS),
            "discounted_return": (
                "success * 0.995 ** (length - 1) for length >= 1; not stored"
            ),
            "reservation_probe": _probe(network, params),
            "fixed": _evaluate(network, params, variant="fixed", seed=seed),
            "mutant": _evaluate(network, params, variant="mutant", seed=seed),
        },
    )
    print(f"[curve] {destination}", flush=True)


def _latest_pretrain_update(directory):
    best = None
    root = Path(directory) / "checkpoints"
    if not root.is_dir():
        return None
    for path in root.iterdir():
        if not path.name.startswith("update_"):
            continue
        if not checkpoint_files_present(path):
            continue
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        update = int(path.name.split("_", 1)[1])
        if int(meta.get("global_update", -1)) != update:
            continue
        if best is None or update > best:
            best = update
    return best


def _pretrain_summary_ok(directory):
    path = Path(directory) / "summary.json"
    final = _latest_pretrain_update(directory)
    if not path.is_file() or final is None:
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    return (
        int(summary.get("valid_transitions", -1)) >= VALID_BUDGET
        and int(summary.get("global_update", -1)) == int(final)
    )


def run_pretrain(log_dir, job):
    directory = pretrain_dir(log_dir, job["condition"], job["seed"])
    if _pretrain_summary_ok(directory):
        print(f"[skip] {directory}", flush=True)
        return
    config = pretrain_config(job["condition"], job["seed"])
    directory.mkdir(parents=True, exist_ok=True)
    network, template = initialize_tick_claim_gc(config)
    update = jax.jit(make_tick_claim_gc_update(network, config))
    saved = _latest_pretrain_update(directory)
    if saved is None:
        runner = template
        save_tick_claim_gc_checkpoint(_pretrain_checkpoint(directory, 0), runner, config)
        saved = 0
    else:
        runner = load_tick_claim_gc_checkpoint(
            _pretrain_checkpoint(directory, saved), template, config
        )
    rows = _metric_rows(directory / "updates.jsonl", saved, "update")
    valid_total = _sum_field(rows, "valid_transitions")
    while valid_total < VALID_BUDGET:
        if int(runner.global_update) >= PHYSICAL_CAP_UPDATES:
            raise RuntimeError(
                f"{job['id']} reached {PHYSICAL_CAP_UPDATES} updates "
                f"with only {valid_total} valid transitions"
            )
        runner, metrics = update(runner)
        jax.block_until_ready(runner.global_update)
        finished = int(runner.global_update)
        host = _host_metrics(metrics, finished)
        _append_metric(directory / "updates.jsonl", host)
        rows.append(host)
        valid_total += int(host["valid_transitions"])
        if finished % 128 == 0 or valid_total >= VALID_BUDGET:
            save_tick_claim_gc_checkpoint(
                _pretrain_checkpoint(directory, finished), runner, config
            )
        if finished % 32 == 0 or valid_total >= VALID_BUDGET:
            print(
                f"[pretrain] {job['id']} update {finished} "
                f"valid={valid_total}/{VALID_BUDGET} "
                f"this_valid={host['valid_transitions']}",
                flush=True,
            )
    physical = _sum_field(rows, "transitions")
    _write_json(
        directory / "summary.json",
        {
            "schema_version": "tick_claim_gc_goalset_pretrain_v1",
            "condition": job["condition"],
            "seed": int(job["seed"]),
            "allowed_goals": list(ALLOWED[job["condition"]]),
            "goal_sampling_weights": list(
                sampling_weights(ALLOWED[job["condition"]])
            ),
            "valid_budget": VALID_BUDGET,
            "valid_transitions": valid_total,
            "physical_transitions": physical,
            "global_update": int(runner.global_update),
            "empty_minibatches": _sum_field(rows, "empty_minibatches"),
            "valid_by_goal": _sum_vector(rows, "valid_by_goal"),
            "commands_by_goal": _sum_vector(rows, "commands_by_goal"),
            "goal_ids": list(GOAL_IDS),
        },
    )
    print(f"[pretrain-done] {job['id']} valid={valid_total} physical={physical}", flush=True)


def _prepare_branch(log_dir, job, template, branch):
    source_dir = pretrain_dir(log_dir, job["condition"], job["seed"])
    update = _latest_pretrain_update(source_dir)
    source = _pretrain_checkpoint(source_dir, update)
    recorded = json.loads((source / "config.json").read_text(encoding="utf-8"))
    origin = config_from_tick_claim_gc_payload(recorded)
    if origin.goal_mode != "masked" or int(origin.seed) != int(job["seed"]):
        raise RuntimeError(f"pretrain config does not match {job['id']}")
    if tuple(origin.allowed_goals) != tuple(ALLOWED[job["condition"]]):
        raise RuntimeError(f"pretrain mask does not match {job['id']}")
    runner = serialization.from_bytes(template, (source / "state.msgpack").read_bytes())
    before = tuple(
        np.asarray(jax.device_get(leaf)).tobytes()
        for leaf in jax.tree_util.tree_leaves(runner.train_state.params)
    )
    runner = reset_tick_claim_gc_optimizer(runner, branch)
    runner = reinit_tick_claim_gc_adaptation_start(runner, int(job["seed"]))
    runner = command_deliver_3(runner)
    after = tuple(
        np.asarray(jax.device_get(leaf)).tobytes()
        for leaf in jax.tree_util.tree_leaves(runner.train_state.params)
    )
    if after != before:
        raise RuntimeError("adaptation start changed parameters")
    if int(runner.train_state.step) != 0 or int(runner.global_update) != 0:
        raise RuntimeError("adaptation start did not reset the counters")
    return runner


def _assert_shared_start(log_dir, job):
    other = dict(job)
    other["variant"] = "mutant" if job["variant"] == "fixed" else "fixed"
    left = _checkpoint_dir(cell_dir(log_dir, job), 0) / "state.msgpack"
    right = _checkpoint_dir(cell_dir(log_dir, other), 0) / "state.msgpack"
    if left.is_file() and right.is_file() and left.read_bytes() != right.read_bytes():
        raise RuntimeError(f"fixed and mutant adapt_0 states differ for {job['id']}")


def _family_metric(curve, dynamics, name, family, key):
    return curve[dynamics][name][family][key]


def _write_pair(log_dir, job):
    own = cell_dir(log_dir, job)
    sibling_job = dict(job)
    sibling_job["variant"] = "mutant" if job["variant"] == "fixed" else "fixed"
    sibling = cell_dir(log_dir, sibling_job)
    if not _cell_finished(own) or not _cell_finished(sibling):
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
            "mutant_reservation_probe": mutant["reservation_probe"],
            "fixed_reservation_probe": fixed["reservation_probe"],
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
                    "same_policy_return_difference": (
                        _discounted_return(mutant["mutant"][name]["episodes"], family)
                        - _discounted_return(mutant["fixed"][name]["episodes"], family)
                    ),
                }
        points.append(point)
    _write_json(
        own.parent / f"pair_seed{job['seed']}.json",
        {
            "schema_version": "tick_claim_gc_goalset_pair_v1",
            "condition": job["condition"],
            "seed": int(job["seed"]),
            "primary_metric": (
                "natural_reset mode violation_delivery_rate: "
                "mutant-adapted minus fixed-continued, on the mutant environment"
            ),
            "families_not_pooled": list(FAMILIES),
            "points": points,
        },
    )


def latest_saved_update(cell):
    best = None
    for update in SAVE_UPDATES:
        directory = _checkpoint_dir(cell, update)
        if not checkpoint_files_present(directory):
            continue
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if int(meta.get("global_update", -1)) != int(update):
            continue
        best = int(update)
    return best


def _cell_finished(cell):
    summary_path = Path(cell) / "summary.json"
    final_dir = _checkpoint_dir(cell, ADAPT_UPDATES)
    if not summary_path.is_file() or not checkpoint_files_present(final_dir):
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    meta = json.loads((final_dir / "metadata.json").read_text(encoding="utf-8"))
    if int(summary.get("adaptation_updates", -1)) != ADAPT_UPDATES:
        return False
    if int(meta.get("global_update", -1)) != ADAPT_UPDATES:
        return False
    return all(
        curve_complete(_curve_path(cell, update), update) for update in SCIENCE_UPDATES
    )


def run_adapt_cell(log_dir, job):
    cell = cell_dir(log_dir, job)
    if _cell_finished(cell):
        print(f"[skip] {cell}", flush=True)
        _write_pair(log_dir, job)
        return
    branch = adapt_config(job)
    _discard, template = initialize_tick_claim_gc(branch)
    del _discard
    network = _network()
    update = jax.jit(make_tick_claim_gc_update(network, branch))
    saved = latest_saved_update(cell)
    if saved is None:
        runner = _prepare_branch(log_dir, job, template, branch)
        save_tick_claim_gc_checkpoint(_checkpoint_dir(cell, 0), runner, branch)
        _assert_shared_start(log_dir, job)
        saved = 0
    else:
        runner = load_tick_claim_gc_checkpoint(
            _checkpoint_dir(cell, saved), template, branch
        )
    rows = _metric_rows(cell / "updates.jsonl", saved, "update")
    if saved in SCIENCE_UPDATES:
        _write_curve(cell, job, runner, saved, network)
    parameter_count = tick_claim_gc_parameter_count(runner.train_state.params)
    started = time.perf_counter()
    trained = 0
    while int(runner.global_update) < ADAPT_UPDATES:
        runner, metrics = update(runner)
        jax.block_until_ready(runner.global_update)
        finished = int(runner.global_update)
        _append_metric(cell / "updates.jsonl", _host_metrics(metrics, finished))
        trained += 1
        if finished in SAVE_UPDATES:
            save_tick_claim_gc_checkpoint(
                _checkpoint_dir(cell, finished), runner, branch
            )
        if finished in SCIENCE_UPDATES:
            _write_curve(cell, job, runner, finished, network)
            host = _host_metrics(metrics, finished)
            print(
                f"[adapt] {job['id']} step {finished}/{ADAPT_UPDATES} "
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
    _write_json(
        cell / "summary.json",
        {
            "schema_version": "tick_claim_gc_goalset_cell_v1",
            "job_id": job["id"],
            "condition": job["condition"],
            "variant": job["variant"],
            "seed": int(job["seed"]),
            "hidden_size": 512,
            "entropy_coefficient": 0.005,
            "learning_rate": 2e-4,
            "parameter_count": int(parameter_count),
            "reset_optimizer": True,
            "reinitialized_env_and_rng": True,
            "adaptation_updates": ADAPT_UPDATES,
            "adaptation_transitions": ADAPT_UPDATES * BATCH,
            "global_update": int(runner.global_update),
            "environment_steps": int(runner.env_steps),
            "resumed_from_adaptation_update": int(saved),
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
        return _pretrain_summary_ok(pretrain_dir(log_dir, job["condition"], job["seed"]))
    return _cell_finished(cell_dir(log_dir, job))


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


def _run_job(log_dir, job):
    print(f"[job] {job['id']}", flush=True)
    if job["kind"] == "pretrain":
        run_pretrain(log_dir, job)
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
