#!/usr/bin/env python3
"""Run the preregistered stochastic inventory-overwrite PPO comparison."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path

if os.environ.get("HACKRL_DEVICE", "cpu") != "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

import run_pack_inventory_overwrite_ppo as base
from hackrl.pack_inventory_overwrite_rare import (
    build_risk_profile,
    evaluate_regime_point,
)
from hackrl.pack_restore import PackRestoreVariant
from hackrl.pack_restore_gc import (
    _FROZEN_EVAL_CACHE,
    evaluate_pack_restore_gc_frozen,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC_PATH = (
    REPOSITORY
    / "docs/manifests/pack_inventory_overwrite_stochastic_ppo_v1.json"
)
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
RUN_ROOT = Path(SPEC["run_root"])
SEEDS = tuple(int(seed) for seed in SPEC["seeds"])
CONDITIONS = tuple(SPEC["conditions"])
LEARNING_CONDITIONS = CONDITIONS
SCIENCE_UPDATES = tuple(
    int(update) for update in SPEC["evaluation"]["adaptation_updates"]
)
EVAL_SEED_BASE = 53000
EXECUTION_SOURCES = (
    Path("docs/manifests/pack_inventory_overwrite_stochastic_ppo_v1.json"),
    Path("scripts/run_pack_inventory_overwrite_stochastic_ppo.py"),
    Path("scripts/run_pack_inventory_overwrite_ppo.py"),
    Path("src/hackrl/pack_restore.py"),
    Path("src/hackrl/pack_restore_gc.py"),
    Path("src/hackrl/pack_inventory_overwrite_rare.py"),
    Path("src/hackrl/dual_leo.py"),
)

_BASE_EMPTY_TRACKER = base._empty_tracker
_BASE_UPDATE_TRACKER = base.update_exposure_tracker
_BASE_CAPACITY = base._capacity
_BASE_SUMMARIZE_CELL = base._summarize_cell
_BASE_RUN_MANIFEST = base._run_manifest


def _install_spec() -> None:
    """Point the proven checkpoint/resume machinery at this frozen design."""

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


def _probability(condition: str) -> float:
    return float(SPEC["conditions"][condition]["overwrite_probability"])


def _capacity() -> dict:
    return _BASE_CAPACITY(RUN_ROOT)


def _validate_prerequisite() -> None:
    prerequisites = SPEC["prerequisites"]
    for path_key, hash_key in (
        ("policy_panel_result", "policy_panel_result_sha256"),
        ("policy_panel_risk_profiles", "policy_panel_risk_profiles_sha256"),
        ("reference_paths", "reference_paths_sha256"),
    ):
        path = Path(prerequisites[path_key])
        if base._sha256(path) != prerequisites[hash_key]:
            raise RuntimeError(f"pinned prerequisite changed: {path}")
    panel = json.loads(
        Path(prerequisites["policy_panel_result"]).read_text(encoding="utf-8")
    )
    if panel.get("status") != prerequisites["policy_panel_required_status"]:
        raise RuntimeError("the frozen four-policy panel is not complete")


def _branch_config(seed: int, condition: str):
    config = replace(
        base._source_config(seed),
        variant=PackRestoreVariant.INVENTORY_OVERWRITE.value,
        inventory_overwrite_probability=_probability(condition),
        num_updates=int(SPEC["optimization"]["updates"]),
        checkpoint_updates=(),
    )
    config.validate()
    return config


def _empty_tracker(num_envs: int) -> dict:
    tracker = _BASE_EMPTY_TRACKER(num_envs)
    tracker["overwrite_fire_events"] = 0
    return tracker


def update_exposure_tracker(
    tracker: dict,
    metrics: dict,
    *,
    learning: bool,
    optimizer_applications: int,
) -> dict:
    previous_fires = int(tracker.get("overwrite_fire_events", 0))
    output = _BASE_UPDATE_TRACKER(
        tracker,
        metrics,
        learning=learning,
        optimizer_applications=optimizer_applications,
    )
    output["overwrite_fire_events"] = previous_fires + int(
        np.asarray(jax.device_get(metrics["overwrite_fire_events"]))
    )
    return output


def _reference_episodes() -> list[dict]:
    document = json.loads(
        Path(SPEC["prerequisites"]["reference_paths"]).read_text(
            encoding="utf-8"
        )
    )
    return document["episodes"]


def _baseline_point(policy: str, seed: int, probability: float) -> dict:
    result = json.loads(
        Path(SPEC["prerequisites"]["policy_panel_result"]).read_text(
            encoding="utf-8"
        )
    )
    row = next(
        candidate
        for candidate in result["rows"][policy]
        if int(candidate["seed"]) == seed
    )
    fixed_records = row["evaluations"]["fixed"]["episode_records"]
    risk_counts = sorted(
        {int(record["overwrite_trigger_count"]) for record in fixed_records}
    )
    point = next(
        candidate
        for candidate in row["regime_map"]
        if abs(float(candidate["overwrite_probability"]) - probability) < 1e-15
    )
    return {
        "policy": policy,
        "seed": seed,
        "overwrite_probability": probability,
        "success_rate": float(point["existing_success_rate"]),
        "expected_discounted_return": float(
            point["existing_expected_discounted_return"]
        ),
        "loss_episode_probability": float(
            point["existing_loss_episode_probability"]
        ),
        "risk_count_distribution": {
            str(risk_count): sum(
                int(record["overwrite_trigger_count"] == risk_count)
                for record in fixed_records
            )
            for risk_count in risk_counts
        },
    }


def _evaluate(network, parameters, seed: int, condition: str) -> dict:
    probability = _probability(condition)
    source_config = base._source_config(seed)
    common = {
        "split": SPEC["evaluation"]["split"],
        "seed_base": EVAL_SEED_BASE,
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
    stochastic_sample = evaluate_pack_restore_gc_frozen(
        network,
        parameters,
        variant=PackRestoreVariant.INVENTORY_OVERWRITE.value,
        overwrite_probability=probability,
        stochastic=True,
        repeats_per_state=int(
            SPEC["evaluation"]["stochastic_sample_repeats_per_state"]
        ),
        **common,
    )
    for result, repeats in (
        (fixed_mode, SPEC["evaluation"]["fixed_mode_repeats_per_state"]),
        (fixed_sample, SPEC["evaluation"]["normal_sample_repeats_per_state"]),
        (
            stochastic_sample,
            SPEC["evaluation"]["stochastic_sample_repeats_per_state"],
        ),
    ):
        expected = int(SPEC["evaluation"]["states"]) * int(repeats)
        if int(result["overall"]["episodes"]) != expected:
            raise AssertionError("evaluation panel size changed")
    profile = build_risk_profile(
        fixed_mode["episode_records"],
        _reference_episodes(),
        require_all_fixed_success=False,
    )
    exact = evaluate_regime_point(profile, probability)
    baselines = {
        policy: _baseline_point(policy, seed, probability)
        for policy in ("original", "normal_continue", "p1_adapt")
    }
    return {
        "overwrite_probability": probability,
        "fixed_kernel": {"mode": fixed_mode, "sample": fixed_sample},
        "target_kernel": {
            "exact_mode": exact,
            "sample": stochastic_sample,
        },
        "risk_profile": profile,
        "frozen_baselines": baselines,
        "same_seed_exact_return_contrasts": {
            f"adapt_minus_{policy}": float(
                exact["existing_expected_discounted_return"]
            )
            - float(baselines[policy]["expected_discounted_return"])
            for policy in baselines
        },
        "interpretation": {
            "mode_is_exact_finite_panel_computation": True,
            "sample_is_reported_separately": True,
            "reference_is_not_claimed_optimal": True,
        },
    }


def _write_evaluation(
    cell: Path,
    update: int,
    network,
    runner,
    seed: int,
) -> None:
    path = base._evaluation_path(cell, update)
    if path.is_file():
        return
    condition = cell.parent.name
    base._write_json(
        path,
        {
            "schema_version": "hackrl_pack_inventory_overwrite_stochastic_eval_v1",
            "condition": condition,
            "seed": seed,
            "adaptation_update": update,
            "collector_slots": update
            * int(SPEC["optimization"]["collector_slots_per_update"]),
            "train_state_sha256": base._train_state_sha256(runner.train_state),
            "evaluation": _evaluate(
                network, runner.train_state.params, seed, condition
            ),
        },
    )
    print(f"[evaluation] {path}", flush=True)


def _write_origin(
    cell: Path,
    condition: str,
    seed: int,
    provenance: dict,
) -> None:
    document = {
        "condition": condition,
        "seed": seed,
        "kernel": SPEC["conditions"][condition]["kernel"],
        "overwrite_probability": _probability(condition),
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
    final_evaluation = json.loads(
        base._evaluation_path(
            cell, int(SPEC["optimization"]["updates"])
        ).read_text(encoding="utf-8")
    )["evaluation"]
    summary.update(
        {
            "schema_version": "hackrl_pack_inventory_overwrite_stochastic_cell_v1",
            "overwrite_probability": _probability(condition),
            "overwrite_fire_events": int(tracker["overwrite_fire_events"]),
            "final_exact_mode": final_evaluation["target_kernel"][
                "exact_mode"
            ],
            "final_same_seed_exact_return_contrasts": final_evaluation[
                "same_seed_exact_return_contrasts"
            ],
            "fixed_mode_risk_count_distribution": final_evaluation[
                "risk_profile"
            ]["risk_count_distribution"],
        }
    )
    return summary


def _run_manifest() -> dict:
    manifest = _BASE_RUN_MANIFEST()
    manifest.update(
        {
            "schema_version": "hackrl_pack_inventory_overwrite_stochastic_ppo_run_v1",
            "pinned_prerequisite_sha256": {
                key: SPEC["prerequisites"][key]
                for key in (
                    "policy_panel_result_sha256",
                    "policy_panel_risk_profiles_sha256",
                    "reference_paths_sha256",
                )
            },
            "probabilities": {
                condition: _probability(condition) for condition in CONDITIONS
            },
            "defect_probability_in_observation": False,
        }
    )
    return manifest


def _install_overrides() -> None:
    _install_spec()
    base._capacity = _capacity
    base._validate_prerequisite = _validate_prerequisite
    base._branch_config = _branch_config
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
            base._capacity_checkpoint()
        except Exception:
            base._write_json(
                base._cell(condition, seed) / "failed.json",
                {"failed_unix": time.time(), "traceback": traceback.format_exc()},
            )
            raise
        finally:
            _FROZEN_EVAL_CACHE.clear()
            jax.clear_caches()
    if arguments.all and all(base._job_complete(job) for job in base.build_jobs()):
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
