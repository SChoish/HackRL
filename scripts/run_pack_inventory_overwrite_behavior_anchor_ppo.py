#!/usr/bin/env python3
"""Run the fixed-budget PACK behavior-anchored PPO comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

if os.environ.get("HACKRL_DEVICE", "cpu") != "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

import run_pack_inventory_overwrite_ppo as base
from hackrl.pack_behavior_anchor import (
    PackBehaviorAnchor,
    build_pack_behavior_anchor,
    make_behavior_anchored_pack_restore_update,
)
from hackrl.pack_restore import PackRestoreSplit, PackRestoreVariant
from hackrl.pack_restore_gc import (
    _FROZEN_EVAL_CACHE,
    categorical_source_kl,
    evaluate_pack_restore_gc_frozen,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC_PATH = (
    REPOSITORY
    / "docs/manifests/pack_inventory_overwrite_behavior_anchor_ppo_v1.json"
)
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
VALIDATION_PATH = (
    REPOSITORY
    / "docs/manifests/pack_inventory_overwrite_behavior_anchor_ppo_v1_implementation_validation.json"
)
VALIDATION = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
RUN_ROOT = Path(SPEC["run_root"])
SEEDS = tuple(int(seed) for seed in SPEC["seeds"])
CONDITIONS = tuple(SPEC["conditions"])
LEARNING_CONDITIONS = CONDITIONS
SCIENCE_UPDATES = tuple(
    int(update) for update in SPEC["evaluation"]["adaptation_updates"]
)
EVAL_SEED_BASE = 73000
EXECUTION_SOURCES = tuple(Path(path) for path in SPEC["execution_sources"])
PANELS = tuple(SPEC["evaluation"]["development_panels"])
_PROBABILITIES = {
    float(row["overwrite_probability"]) for row in SPEC["conditions"].values()
}
if len(_PROBABILITIES) != 1:
    raise RuntimeError("behavior-anchor conditions must share one probability")
TARGET_PROBABILITY = _PROBABILITIES.pop()

_BASE_CAPACITY = base._capacity
_BASE_EMPTY_TRACKER = base._empty_tracker
_BASE_RUN_MANIFEST = base._run_manifest
_BASE_SUMMARIZE_CELL = base._summarize_cell
_BASE_UPDATE_TRACKER = base.update_exposure_tracker
_SOURCE_ANCHORS: dict[tuple[int, str], tuple[PackBehaviorAnchor, dict]] = {}


def _prerequisite_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY / path


def _coefficient(condition: str) -> float:
    return float(SPEC["conditions"][condition]["behavior_anchor_coefficient"])


def _install_spec() -> None:
    base.SPEC_PATH = SPEC_PATH
    base.SPEC = SPEC
    base.RUN_ROOT = RUN_ROOT
    base.SOURCE_ROOT = Path(SPEC["source_root"])
    base.SEEDS = SEEDS
    base.CONDITIONS = CONDITIONS
    base.LEARNING_CONDITIONS = LEARNING_CONDITIONS
    base.SCIENCE_UPDATES = SCIENCE_UPDATES
    base.EVAL_SEED_BASE = EVAL_SEED_BASE
    base.EXECUTION_SOURCES = EXECUTION_SOURCES


def _capacity() -> dict:
    return _BASE_CAPACITY(RUN_ROOT)


def _validate_prerequisite() -> None:
    prerequisites = SPEC["prerequisites"]
    for name in (
        "implementation_validation",
        "manual_gate_closeout",
        "learning_rate_result",
        "learning_rate_audit",
    ):
        path = _prerequisite_path(prerequisites[name])
        if base._sha256(path) != prerequisites[f"{name}_sha256"]:
            raise RuntimeError(f"pinned prerequisite changed: {path}")
    if VALIDATION.get("status") != "implementation_contract_valid_training_not_started":
        raise RuntimeError("implementation validation record changed")
    if VALIDATION.get("manifest_id") != SPEC["manifest_id"]:
        raise RuntimeError("implementation validation targets another manifest")
    for seed in SEEDS:
        checkpoint = base._source_checkpoint(seed)
        for name, expected in SPEC["source_checkpoint_sha256"][str(seed)].items():
            if base._sha256(checkpoint / name) != expected:
                raise RuntimeError(
                    f"preregistered source checkpoint changed: {checkpoint / name}"
                )


def _branch_config(seed: int, condition: str):
    source = base._source_config(seed)
    if not math.isclose(
        float(source.learning_rate),
        float(SPEC["optimization"]["learning_rate"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError(f"source learning rate changed for seed {seed}")
    config = replace(
        source,
        variant=PackRestoreVariant.INVENTORY_OVERWRITE.value,
        inventory_overwrite_probability=TARGET_PROBABILITY,
        num_updates=int(SPEC["optimization"]["updates"]),
        checkpoint_updates=(),
    )
    config.validate()
    return config


def _source_anchor(network, seed: int, split: str):
    key = (seed, split)
    cached = _SOURCE_ANCHORS.get(key)
    if cached is not None:
        return cached
    _, source_runner, source_config, _ = base._fresh_runner(seed, CONDITIONS[0])
    anchor, metadata = build_pack_behavior_anchor(
        network,
        source_runner.train_state.params,
        source_growth_period=source_config.source_growth_period,
        horizon_ticks=int(SPEC["evaluation"]["horizon_ticks"]),
        split=PackRestoreSplit(split),
    )
    if (
        metadata.source_has_mutable_normalization_statistics
        or metadata.source_variable_collections != ("params",)
    ):
        raise RuntimeError(f"source actor variables changed for seed {seed}")
    metadata_document = asdict(metadata)
    metadata_document["anchor_sha256"] = base._tree_sha256(anchor)
    if split == PackRestoreSplit.TRAIN.value:
        expected = {
            int(item["seed"]): item["anchor_sha256"]
            for item in VALIDATION["anchor_contracts"]
        }[seed]
        if metadata_document["anchor_sha256"] != expected:
            raise RuntimeError(f"TRAIN behavior-anchor content changed for seed {seed}")
    result = (anchor, metadata_document)
    _SOURCE_ANCHORS[key] = result
    return result


def _make_learning_update(network, config, runner, condition: str, seed: int):
    del runner
    coefficient = _coefficient(condition)
    if coefficient == 0.0:
        return make_behavior_anchored_pack_restore_update(
            network, config, None, coefficient=0.0
        )
    anchor, metadata = _source_anchor(network, seed, PackRestoreSplit.TRAIN.value)
    expected = {
        int(item["seed"]): int(item["metadata"]["state_count"])
        for item in VALIDATION["anchor_contracts"]
    }[seed]
    if metadata["state_count"] != int(expected):
        raise RuntimeError(f"TRAIN behavior anchor changed for seed {seed}")
    return make_behavior_anchored_pack_restore_update(
        network, config, anchor, coefficient=coefficient
    )


def _empty_tracker(num_envs: int) -> dict:
    tracker = _BASE_EMPTY_TRACKER(num_envs)
    tracker["behavior_anchor_forward_state_evaluations"] = 0
    return tracker


def update_exposure_tracker(
    tracker: dict,
    metrics: dict,
    *,
    learning: bool,
    optimizer_applications: int,
) -> dict:
    prior = int(tracker.get("behavior_anchor_forward_state_evaluations", 0))
    output = _BASE_UPDATE_TRACKER(
        tracker,
        metrics,
        learning=learning,
        optimizer_applications=optimizer_applications,
    )
    output["behavior_anchor_forward_state_evaluations"] = prior + int(
        np.asarray(
            jax.device_get(metrics["behavior_anchor_forward_state_evaluations"])
        )
    )
    return output


def _behavior_change(network, parameters, anchor: PackBehaviorAnchor) -> dict:
    current_policy, _ = network.apply(
        parameters,
        anchor.map_channels,
        anchor.numeric_features,
        anchor.goal_one_hot,
    )
    per_state = categorical_source_kl(anchor.source_logits, current_policy.logits)
    changed = jnp.argmax(anchor.source_logits, axis=-1) != jnp.argmax(
        current_policy.logits, axis=-1
    )
    return {
        "state_count": anchor.state_count,
        "mean_kl_source_to_current": max(
            0.0, float(np.asarray(jax.device_get(jnp.mean(per_state))))
        ),
        "argmax_action_change_count": int(
            np.asarray(jax.device_get(jnp.sum(changed)))
        ),
        "argmax_action_change_rate": float(
            np.asarray(jax.device_get(jnp.mean(changed.astype(jnp.float32))))
        ),
    }


def _exact_p05(records: list[dict], *, gamma: float, state_count: int) -> dict:
    if len(records) != state_count:
        raise AssertionError(
            f"exact evaluation requires {state_count} mode trajectories"
        )
    success_terms = []
    return_terms = []
    no_overwrite_terms = []
    distribution: dict[str, int] = {}
    risky_starts = 0
    for record in records:
        risk = int(record["overwrite_trigger_count"])
        success = bool(record["success"])
        survival = (1.0 - TARGET_PROBABILITY) ** risk
        discounted = gamma ** (int(record["length"]) - 1) if success else 0.0
        success_terms.append(float(success) * survival)
        return_terms.append(discounted * survival)
        no_overwrite_terms.append(survival)
        distribution[str(risk)] = distribution.get(str(risk), 0) + 1
        risky_starts += int(risk > 0)
    denominator = len(records)
    success_rate = math.fsum(success_terms) / denominator
    return {
        "overwrite_probability": TARGET_PROBABILITY,
        "success_rate": success_rate,
        "expected_discounted_return": math.fsum(return_terms) / denominator,
        "failure_probability": 1.0 - success_rate,
        "loss_episode_probability": 1.0
        - math.fsum(no_overwrite_terms) / denominator,
        "risk_count_distribution": distribution,
        "risky_start_count": risky_starts,
    }


def _evaluate_panel(network, parameters, seed: int, split: str) -> dict:
    source_config = base._source_config(seed)
    state_count = int(SPEC["evaluation"]["development_panels"][split]["states"])
    common = {
        "split": split,
        "seed_base": EVAL_SEED_BASE + 1000 * PANELS.index(split),
        "learner_seed": seed,
        "source_growth_period": source_config.source_growth_period,
        "record_episodes": True,
        "natural_only": True,
    }
    fixed_mode = evaluate_pack_restore_gc_frozen(
        network,
        parameters,
        variant=PackRestoreVariant.FIXED.value,
        stochastic=False,
        repeats_per_state=int(
            SPEC["evaluation"]["fixed_mode_repeats_per_state"]
        ),
        **common,
    )
    fixed_sample = evaluate_pack_restore_gc_frozen(
        network,
        parameters,
        variant=PackRestoreVariant.FIXED.value,
        stochastic=True,
        repeats_per_state=int(
            SPEC["evaluation"]["normal_sample_repeats_per_state"]
        ),
        **common,
    )
    target_sample = evaluate_pack_restore_gc_frozen(
        network,
        parameters,
        variant=PackRestoreVariant.INVENTORY_OVERWRITE.value,
        overwrite_probability=TARGET_PROBABILITY,
        stochastic=True,
        repeats_per_state=int(
            SPEC["evaluation"]["target_sample_repeats_per_state"]
        ),
        **common,
    )
    for result, repeats in (
        (fixed_mode, SPEC["evaluation"]["fixed_mode_repeats_per_state"]),
        (fixed_sample, SPEC["evaluation"]["normal_sample_repeats_per_state"]),
        (target_sample, SPEC["evaluation"]["target_sample_repeats_per_state"]),
    ):
        if int(result["overall"]["episodes"]) != state_count * int(repeats):
            raise AssertionError(f"{split} evaluation panel size changed")
    state_indices = {
        int(record["state_index"]) for record in fixed_mode["episode_records"]
    }
    if state_indices != set(range(state_count)):
        raise AssertionError(f"{split} mode state coverage changed")
    anchor, metadata = _source_anchor(network, seed, split)
    return {
        "split": split,
        "fixed_kernel": {"mode": fixed_mode, "sample": fixed_sample},
        "target_kernel": {
            "exact_mode": _exact_p05(
                fixed_mode["episode_records"],
                gamma=float(source_config.gamma),
                state_count=state_count,
            ),
            "sample": target_sample,
        },
        "behavior_change_from_source": _behavior_change(
            network, parameters, anchor
        ),
        "source_trajectory_anchor": metadata,
    }


def _write_evaluation(cell: Path, update: int, network, runner, seed: int) -> None:
    path = base._evaluation_path(cell, update)
    if path.is_file():
        return
    condition = cell.parent.name
    train_anchor, train_metadata = _source_anchor(
        network, seed, PackRestoreSplit.TRAIN.value
    )
    base._write_json(
        path,
        {
            "schema_version": "hackrl_pack_inventory_overwrite_behavior_anchor_eval_v1",
            "condition": condition,
            "seed": seed,
            "adaptation_update": update,
            "collector_slots": update
            * int(SPEC["optimization"]["collector_slots_per_update"]),
            "train_state_sha256": base._train_state_sha256(runner.train_state),
            "behavior_anchor_coefficient": _coefficient(condition),
            "train_anchor_behavior_change": _behavior_change(
                network, runner.train_state.params, train_anchor
            ),
            "train_anchor_metadata": train_metadata,
            "panels": {
                split: _evaluate_panel(
                    network, runner.train_state.params, seed, split
                )
                for split in PANELS
            },
        },
    )
    print(f"[evaluation] {path}", flush=True)


def _write_origin(cell: Path, condition: str, seed: int, provenance: dict) -> None:
    document = {
        "condition": condition,
        "seed": seed,
        "kernel": PackRestoreVariant.INVENTORY_OVERWRITE.value,
        "overwrite_probability": TARGET_PROBABILITY,
        "behavior_anchor_coefficient": _coefficient(condition),
        **provenance,
    }
    path = cell / "origin.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != document:
        raise RuntimeError(f"origin changed: {path}")
    base._write_json(path, document)


def _summarize_cell(
    cell: Path,
    condition: str,
    seed: int,
    runner,
    tracker: dict,
    origin: dict,
) -> dict:
    summary = _BASE_SUMMARIZE_CELL(
        cell, condition, seed, runner, tracker, origin
    )
    final = json.loads(
        base._evaluation_path(
            cell, int(SPEC["optimization"]["updates"])
        ).read_text(encoding="utf-8")
    )
    summary.update(
        {
            "schema_version": "hackrl_pack_inventory_overwrite_behavior_anchor_cell_v1",
            "overwrite_probability": TARGET_PROBABILITY,
            "behavior_anchor_coefficient": _coefficient(condition),
            "behavior_anchor_forward_state_evaluations": int(
                tracker["behavior_anchor_forward_state_evaluations"]
            ),
            "final_panels": {
                split: {
                    "exact_mode": final["panels"][split]["target_kernel"][
                        "exact_mode"
                    ],
                    "fixed_mode": final["panels"][split]["fixed_kernel"][
                        "mode"
                    ]["overall"],
                    "behavior_change_from_source": final["panels"][split][
                        "behavior_change_from_source"
                    ],
                }
                for split in PANELS
            },
            "final_train_anchor_behavior_change": final[
                "train_anchor_behavior_change"
            ],
        }
    )
    return summary


def _run_manifest() -> dict:
    manifest = _BASE_RUN_MANIFEST()
    manifest.update(
        {
            "schema_version": "hackrl_pack_inventory_overwrite_behavior_anchor_run_v1",
            "conditions": {
                condition: {
                    "overwrite_probability": TARGET_PROBABILITY,
                    "behavior_anchor_coefficient": _coefficient(condition),
                }
                for condition in CONDITIONS
            },
            "development_panels": list(PANELS),
            "defect_probability_in_observation": False,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "jax": jax.__version__,
                "numpy": np.__version__,
                "devices": [str(device) for device in jax.devices()],
                "backend": jax.default_backend(),
                "x64_enabled": bool(jax.config.x64_enabled),
            },
        }
    )
    return manifest


def _install_overrides() -> None:
    _install_spec()
    base._capacity = _capacity
    base._validate_prerequisite = _validate_prerequisite
    base._branch_config = _branch_config
    base._make_learning_update = _make_learning_update
    base._empty_tracker = _empty_tracker
    base.update_exposure_tracker = update_exposure_tracker
    base._write_evaluation = _write_evaluation
    base._write_origin = _write_origin
    base._summarize_cell = _summarize_cell
    base._run_manifest = _run_manifest


def main() -> None:
    _install_overrides()
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-contract", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    arguments = parser.parse_args()
    if arguments.validate_contract:
        print(json.dumps(base.validate_contract(), indent=2, sort_keys=True))
        return
    if not arguments.all and (
        arguments.condition is None or arguments.seed is None
    ):
        raise SystemExit("use --all or provide both --condition and --seed")
    manifest = base._prepare_run()
    print(f"[provenance] execution_head={manifest['execution_head']}", flush=True)
    jobs = (
        base.build_jobs()
        if arguments.all
        else [
            {
                "condition": arguments.condition,
                "seed": arguments.seed,
                "learning": True,
            }
        ]
    )
    for job in jobs:
        condition = str(job["condition"])
        seed = int(job["seed"])
        if base._job_complete(job):
            print(f"[skip] {condition} seed={seed}", flush=True)
            continue
        try:
            base.run_learning(condition, seed)
            failed = base._cell(condition, seed) / "failed.json"
            if failed.is_file():
                recovery = base._cell(condition, seed) / "recovered_failures"
                recovery.mkdir(parents=True, exist_ok=True)
                failed.replace(recovery / f"failed_{int(time.time())}.json")
            base._capacity_checkpoint()
        except Exception:
            base._write_json(
                base._cell(condition, seed) / "failed.json",
                {"failed_unix": time.time(), "traceback": traceback.format_exc()},
            )
            base._capacity_checkpoint()
            raise
        finally:
            _FROZEN_EVAL_CACHE.clear()
            jax.clear_caches()
    if arguments.all and all(
        base._job_complete(job) for job in base.build_jobs()
    ):
        base._write_json(
            RUN_ROOT / "training_complete.json",
            {
                "status": "all_10_jobs_complete_pending_aggregation",
                "completed_unix": time.time(),
                "execution_head": manifest["execution_head"],
            },
        )


if __name__ == "__main__":
    main()
