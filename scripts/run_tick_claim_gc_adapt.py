#!/usr/bin/env python3
"""Deliver_3 adaptation from workshop12 checkpoints, with fixed/mutant baselines."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

if os.environ.get("HACKRL_DEVICE", "cpu") != "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
from flax import serialization

from hackrl.tick_claim_gc import (
    TickClaimGCActorCritic,
    checkpoint_files_present,
    command_deliver_3,
    config_from_tick_claim_gc_payload,
    initialize_tick_claim_gc,
    load_tick_claim_gc_branch,
    load_tick_claim_gc_checkpoint,
    make_tick_claim_gc_update,
    save_tick_claim_gc_checkpoint,
    evaluate_tick_claim_gc_frozen,
)


ADAPTATION_UPDATES = 128
SOURCE_UPDATES = (0, 128, 512)
FAMILIES = ("natural_reset", "common_setup")
EVAL_KEYS = (
    "episodes",
    "success_rate",
    "mean_length",
    "violation_rate",
    "repeated_violation_rate",
    "violation_delivery_rate",
    "mean_violation_grain_delivered",
    "mean_first_violation_step",
    "opportunity_exposure_rate",
    "completed_rate",
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).strip()


def _source_checkpoint(run_root, seed, update):
    return run_root / f"fixed_seed{seed}" / "checkpoints" / f"update_{update}"


def _cell_dir(log_dir, update, variant, seed):
    return log_dir / f"from_update_{update}" / f"{variant}_seed{seed}"


def _read_config(checkpoint):
    if not checkpoint_files_present(checkpoint):
        raise FileNotFoundError(f"source checkpoint is missing: {checkpoint}")
    recorded = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    return config_from_tick_claim_gc_payload(recorded)


def _branch_config(origin, variant):
    return replace(
        origin,
        variant=variant,
        goal_mode="deliver_3",
        num_updates=ADAPTATION_UPDATES,
        checkpoint_updates=(),
    )


def _family_view(result):
    return {
        family: {key: result[family][key] for key in EVAL_KEYS}
        for family in FAMILIES
    }


def _evaluate_policy(network, params, *, variant, seed):
    """Score deliver_3. Natural reset and common setup stay separate."""

    views = {}
    for stochastic, repeats, name in (
        (False, 1, "mode"),
        (True, 4, "sample"),
    ):
        result = evaluate_tick_claim_gc_frozen(
            network,
            params,
            variant=variant,
            stochastic=stochastic,
            repeats_per_state=repeats,
            seed_base=20000,
            learner_seed=seed,
        )
        views[name] = _family_view(result)
    return views


def _pre_complete(path):
    if not path.is_file():
        return False
    document = json.loads(path.read_text(encoding="utf-8"))
    for variant in ("fixed", "mutant"):
        block = document.get(variant)
        if not isinstance(block, dict):
            return False
        for name in ("mode", "sample"):
            families = block.get(name)
            if not isinstance(families, dict):
                return False
            if not set(FAMILIES) <= set(families):
                return False
    return True


def write_pre_adaptation_eval(network, checkpoint, destination, *, seed, update):
    if _pre_complete(destination):
        print(f"[skip-pre] seed {seed} update {update}", flush=True)
        return
    _config, runner = _load_exact(checkpoint, network)
    payload = {
        "schema_version": "tick_claim_gc_pre_adapt_eval_v1",
        "seed": seed,
        "source_update": update,
        "command": "deliver_3",
        "families_not_pooled": list(FAMILIES),
        "fixed": _evaluate_policy(
            network, runner.train_state.params, variant="fixed", seed=seed
        ),
        "mutant": _evaluate_policy(
            network, runner.train_state.params, variant="mutant", seed=seed
        ),
    }
    _write_json(destination, payload)
    print(f"[pre] {destination}", flush=True)


def _load_exact(checkpoint, network):
    recorded = _read_config(checkpoint)
    _network, template = initialize_tick_claim_gc(recorded)
    del _network
    runner = load_tick_claim_gc_checkpoint(checkpoint, template, recorded)
    return recorded, runner


def _train_metrics(metrics):
    return {
        key: np.asarray(jax.device_get(value)).tolist()
        for key, value in metrics.items()
    }


def _cell_complete(cell):
    summary_path = cell / "summary.json"
    final_dir = cell / "checkpoints" / "adapt_128"
    start_dir = cell / "checkpoints" / "adapt_0"
    cross_path = cell / "mutant_cross_eval.json"
    if not summary_path.is_file() or not cross_path.is_file():
        return False
    if not checkpoint_files_present(final_dir) or not checkpoint_files_present(start_dir):
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    final = json.loads((final_dir / "metadata.json").read_text(encoding="utf-8"))
    return (
        int(summary.get("adaptation_updates", -1)) == ADAPTATION_UPDATES
        and int(summary.get("adaptation_transitions", -1)) == 4_194_304
        and int(final.get("global_update", -1))
        == int(summary.get("source_update", -2)) + ADAPTATION_UPDATES
        and json.loads(cross_path.read_text(encoding="utf-8")).get("dynamics")
        == "mutant"
    )


def run_adaptation_cell(source, cell, *, variant, seed, source_update):
    if _cell_complete(cell):
        print(f"[skip] {cell}", flush=True)
        return
    origin = _read_config(source)
    if int(origin.seed) != seed:
        raise RuntimeError(f"source seed mismatch at {source}")
    branch = _branch_config(origin, variant)
    _discard, template = initialize_tick_claim_gc(branch)
    del _discard
    network = TickClaimGCActorCritic(hidden_size=branch.hidden_size)
    update = jax.jit(make_tick_claim_gc_update(network, branch))
    start_dir = cell / "checkpoints" / "adapt_0"
    if checkpoint_files_present(start_dir):
        runner = load_tick_claim_gc_checkpoint(start_dir, template, branch)
    else:
        runner = load_tick_claim_gc_branch(source, template, branch)
        if int(runner.global_update) != source_update:
            raise RuntimeError(
                f"source update {source_update} loaded as {int(runner.global_update)}"
            )
        before = serialization.to_bytes(runner.train_state.params)
        runner = command_deliver_3(runner)
        if serialization.to_bytes(runner.train_state.params) != before:
            raise RuntimeError("command switch changed parameters")
        save_tick_claim_gc_checkpoint(start_dir, runner, branch)
    started = int(runner.global_update)
    target = source_update + ADAPTATION_UPDATES
    if started > target:
        raise RuntimeError(f"adaptation already past its budget at {cell}")
    while int(runner.global_update) < target:
        runner, metrics = update(runner)
        jax.block_until_ready(runner.global_update)
        finished = int(runner.global_update) - source_update
        if finished == 1 or finished % 32 == 0:
            host = _train_metrics(metrics)
            print(
                f"[adapt] {variant} seed {seed} from {source_update} "
                f"step {finished}/{ADAPTATION_UPDATES} "
                f"violations={host.get('violation_events')}",
                flush=True,
            )
    final_dir = cell / "checkpoints" / "adapt_128"
    save_tick_claim_gc_checkpoint(final_dir, runner, branch)
    own_eval = _evaluate_policy(
        network, runner.train_state.params, variant=variant, seed=seed
    )
    mutant_eval = _evaluate_policy(
        network, runner.train_state.params, variant="mutant", seed=seed
    )
    _write_json(
        cell / "mutant_cross_eval.json",
        {
            "schema_version": "tick_claim_gc_mutant_cross_eval_v1",
            "dynamics": "mutant",
            "trained_variant": variant,
            "seed": seed,
            "source_update": source_update,
            "families_not_pooled": list(FAMILIES),
            **mutant_eval,
        },
    )
    _write_json(
        cell / "summary.json",
        {
            "schema_version": "tick_claim_gc_adapt_cell_v1",
            "variant": variant,
            "seed": seed,
            "goal_mode": "deliver_3",
            "source_update": source_update,
            "source_checkpoint": str(source),
            "adaptation_updates": ADAPTATION_UPDATES,
            "adaptation_transitions": 4_194_304,
            "global_update": int(runner.global_update),
            "environment_steps": int(runner.env_steps),
            "training_dynamics_eval": own_eval,
            "git_sha": _git_sha(),
        },
    )
    print(f"[done] {cell}", flush=True)


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--updates", default="0,128,512")
    return parser.parse_args()


def main():
    args = _arguments()
    run_root = Path(args.run_root)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(item) for item in args.seeds.split(",") if item]
    updates = [int(item) for item in args.updates.split(",") if item]
    if any(update not in SOURCE_UPDATES for update in updates):
        raise ValueError(f"updates must be chosen from {SOURCE_UPDATES}")
    first = _source_checkpoint(run_root, seeds[0], updates[0])
    origin = _read_config(first)
    network = TickClaimGCActorCritic(hidden_size=origin.hidden_size)
    for seed in seeds:
        for update in updates:
            source = _source_checkpoint(run_root, seed, update)
            pre_path = log_dir / "pre_adapt" / f"seed{seed}_update{update}.json"
            write_pre_adaptation_eval(
                network, source, pre_path, seed=seed, update=update
            )
            for variant in ("fixed", "mutant"):
                run_adaptation_cell(
                    source,
                    _cell_dir(log_dir, update, variant, seed),
                    variant=variant,
                    seed=seed,
                    source_update=update,
                )
            fixed_state = (
                _cell_dir(log_dir, update, "fixed", seed)
                / "checkpoints"
                / "adapt_0"
                / "state.msgpack"
            ).read_bytes()
            mutant_state = (
                _cell_dir(log_dir, update, "mutant", seed)
                / "checkpoints"
                / "adapt_0"
                / "state.msgpack"
            ).read_bytes()
            if fixed_state != mutant_state:
                raise RuntimeError(
                    f"fixed/mutant adapt_0 states differ for seed {seed} update {update}"
                )
    print("[queue-step] adaptation shard complete", flush=True)


if __name__ == "__main__":
    main()
