#!/usr/bin/env python3
"""Frozen 12-goal evaluation of the workshop12 base checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import fields
from pathlib import Path

if os.environ.get("HACKRL_DEVICE", "cpu") != "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
from flax import serialization

from hackrl.tick_claim import GOAL_IDS
from hackrl.tick_claim_gc import (
    TickClaimGCConfig,
    initialize_tick_claim_gc,
    load_tick_claim_gc_checkpoint,
    make_tick_claim_gc_update,
    tick_claim_gc_config_payload,
)
from hackrl.tick_claim_gc_goal_eval import (
    GOAL_GROUPS,
    evaluate_tick_claim_gc_goal_grid,
)


def _config_from_recorded(recorded):
    names = {item.name for item in fields(TickClaimGCConfig)}
    config = TickClaimGCConfig(**{name: recorded[name] for name in names})
    payload = tick_claim_gc_config_payload(config)
    if payload != recorded:
        mismatched = [
            key
            for key in sorted(set(payload) | set(recorded))
            if payload.get(key) != recorded.get(key)
        ]
        raise ValueError(f"checkpoint config mismatch: {mismatched}")
    return config


def _load_runner(checkpoint_dir, template):
    recorded = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    config = _config_from_recorded(recorded)
    runner = load_tick_claim_gc_checkpoint(checkpoint_dir, template, config)
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    if int(runner.global_update) != int(metadata["global_update"]):
        raise ValueError("loaded global_update does not match metadata")
    if int(runner.env_steps) != int(metadata["environment_steps"]):
        raise ValueError("loaded environment_steps do not match metadata")
    return config, runner


def _tree_equal(left, right):
    return bool(
        jax.tree_util.tree_all(
            jax.tree.map(lambda a, b: bool(jax.numpy.array_equal(a, b)), left, right)
        )
    )


def verify_checkpoint(checkpoint_dir, template, update_fn):
    _config, runner = _load_runner(checkpoint_dir, template)
    direct_next, direct_metrics = update_fn(runner)
    restored = load_tick_claim_gc_checkpoint(
        checkpoint_dir, template, _config
    )
    restored_next, restored_metrics = update_fn(restored)
    jax.block_until_ready(direct_next.global_update)
    return {
        "global_update": int(runner.global_update),
        "environment_steps": int(runner.env_steps),
        "next_state_equal": _tree_equal(direct_next, restored_next),
        "next_metrics_equal": _tree_equal(direct_metrics, restored_metrics),
    }


def _group_rates(goal_rows, family):
    groups = {}
    for name, members in GOAL_GROUPS:
        rates = [
            goal_rows[goal_id][family]["success_rate_eligible"]
            for goal_id in members
            if goal_rows[goal_id][family]["eligible_episodes"] > 0
            and goal_rows[goal_id][family]["success_rate_eligible"] is not None
        ]
        groups[name] = float(sum(rates) / len(rates)) if rates else None
    return groups


def evaluate_checkpoint(network, runner, *, seed):
    parameters = runner.train_state.params
    before = serialization.to_bytes(parameters)
    mode = evaluate_tick_claim_gc_goal_grid(
        network,
        parameters,
        variant="fixed",
        stochastic=False,
        repeats_per_state=1,
        learner_seed=seed,
    )
    sample = evaluate_tick_claim_gc_goal_grid(
        network,
        parameters,
        variant="fixed",
        stochastic=True,
        repeats_per_state=4,
        learner_seed=seed,
    )
    return {
        "mode": mode,
        "sample": sample,
        "parameter_bytes_unchanged": serialization.to_bytes(parameters) == before,
        "group_success_eligible": {
            "mode": {
                "natural_reset": _group_rates(mode["goals"], "natural_reset"),
                "common_setup": _group_rates(mode["goals"], "common_setup"),
            },
            "sample": {
                "natural_reset": _group_rates(sample["goals"], "natural_reset"),
                "common_setup": _group_rates(sample["goals"], "common_setup"),
            },
        },
    }


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--updates", default="0,32,128,256,512")
    return parser.parse_args()


def main():
    args = _arguments()
    run_root = Path(args.run_root)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(item) for item in args.seeds.split(",") if item]
    updates = [int(item) for item in args.updates.split(",") if item]
    first = run_root / f"fixed_seed{seeds[0]}" / "checkpoints" / f"update_{updates[0]}"
    recorded = json.loads((first / "config.json").read_text(encoding="utf-8"))
    template_config = _config_from_recorded(recorded)
    network, template = initialize_tick_claim_gc(template_config)
    update_fn = jax.jit(make_tick_claim_gc_update(network, template_config))
    restore_path = log_dir / "restore_checks.json"
    restore = json.loads(restore_path.read_text(encoding="utf-8")) if restore_path.is_file() else {}
    for seed in seeds:
        for update in updates:
            key = f"fixed_seed{seed}/update_{update}"
            checkpoint = run_root / f"fixed_seed{seed}" / "checkpoints" / f"update_{update}"
            if key not in restore:
                print(f"[restore] {key}", flush=True)
                checked = verify_checkpoint(checkpoint, template, update_fn)
                if not (checked["next_state_equal"] and checked["next_metrics_equal"]):
                    raise RuntimeError(f"restore failed for {key}: {checked}")
                if checked["global_update"] != update:
                    raise RuntimeError(f"update index mismatch for {key}")
                restore[key] = checked
                restore_path.write_text(
                    json.dumps(restore, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            destination = log_dir / f"fixed_seed{seed}" / f"update_{update}.json"
            if destination.is_file():
                print(f"[skip] {key}", flush=True)
                continue
            print(f"[eval] {key}", flush=True)
            _config, runner = _load_runner(checkpoint, template)
            if int(runner.global_update) != update or int(_config.seed) != seed:
                raise RuntimeError(f"refusing to score mismatched checkpoint {key}")
            payload = {
                "schema_version": "tick_claim_gc_goal_eval_cell_v1",
                "manifest": "docs/manifests/tick_claim_gc_workshop12_goal_eval_v1.json",
                "git_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=Path(__file__).resolve().parents[1],
                    text=True,
                ).strip(),
                "seed": seed,
                "update": update,
                "environment_steps": int(runner.env_steps),
                "goal_ids": list(GOAL_IDS),
                **evaluate_checkpoint(network, runner, seed=seed),
            }
            if not payload["parameter_bytes_unchanged"]:
                raise RuntimeError(f"evaluation mutated parameters for {key}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"[wrote] {destination}", flush=True)


if __name__ == "__main__":
    main()
