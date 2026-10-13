#!/usr/bin/env python3
"""Run the missing p=0 and p=1 PPO training-probability branches."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

if os.environ.get("HACKRL_DEVICE", "cpu") != "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

import run_pack_inventory_overwrite_lr_ppo as lr
from hackrl.pack_restore_gc import _FROZEN_EVAL_CACHE


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC_PATH = (
    REPOSITORY
    / "docs/manifests/pack_inventory_overwrite_train_probability_transfer_v1.json"
)
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
RUN_ROOT = Path(SPEC["run_root"])
SEEDS = tuple(int(seed) for seed in SPEC["seeds"])
CONDITIONS = tuple(SPEC["conditions"])
UPDATES = tuple(
    int(update) for update in SPEC["evaluation"]["adaptation_updates"]
)
EXECUTION_SOURCES = tuple(Path(path) for path in SPEC["execution_sources"])
EVAL_SEED_BASE = int(SPEC["evaluation"]["seed_base"])
base = lr.base


def _prerequisite_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY / path


def _configure_lr_module() -> None:
    lr.SPEC_PATH = SPEC_PATH
    lr.SPEC = SPEC
    lr.RUN_ROOT = RUN_ROOT
    lr.SEEDS = SEEDS
    lr.CONDITIONS = CONDITIONS
    lr.LEARNING_CONDITIONS = CONDITIONS
    lr.SCIENCE_UPDATES = UPDATES
    lr.EVAL_SEED_BASE = EVAL_SEED_BASE
    lr.EXECUTION_SOURCES = EXECUTION_SOURCES
    lr._BEHAVIOR_ANCHOR_CACHE.clear()


def _validate_prerequisite() -> None:
    lr._validate_prerequisite()
    prerequisites = SPEC["prerequisites"]
    prior_run_manifest = _prerequisite_path(
        prerequisites["prior_run_manifest"]
    )
    if (
        base._sha256(prior_run_manifest)
        != prerequisites["prior_run_manifest_sha256"]
    ):
        raise RuntimeError("pinned prior run manifest changed")
    formula_basis = _prerequisite_path(prerequisites["exact_formula_basis"])
    if base._sha256(formula_basis) != prerequisites["exact_formula_basis_sha256"]:
        raise RuntimeError("pinned exact-formula basis changed")
    basis = json.loads(formula_basis.read_text(encoding="utf-8"))
    if basis.get("status") != "complete_no_learning_training_not_started":
        raise RuntimeError("exact-formula basis is incomplete")
    prior_root = _prerequisite_path(prerequisites["prior_raw_result"]).parent
    origins = prerequisites["reused_source_origin_sha256"]
    expected_origins = {
        str(prior_root / "p05_lr1" / f"seed{seed}" / "origin.json")
        for seed in SEEDS
    }
    if set(origins) != expected_origins:
        raise RuntimeError("reused source-origin file set changed")
    for path, expected_hash in origins.items():
        if base._sha256(Path(path)) != expected_hash:
            raise RuntimeError(f"reused source origin changed: {path}")
    reused = prerequisites["reused_final_evaluation_sha256"]
    expected = {
        str(
            prior_root
            / condition
            / f"seed{seed}"
            / "evaluations/adapt_32.json"
        )
        for condition in SPEC["reused_conditions"]
        for seed in SEEDS
    }
    if set(reused) != expected:
        raise RuntimeError("reused final-evaluation file set changed")
    for path, expected_hash in reused.items():
        if base._sha256(Path(path)) != expected_hash:
            raise RuntimeError(f"reused final evaluation changed: {path}")


def _write_evaluation(cell: Path, update: int, network, runner, seed: int) -> None:
    path = base._evaluation_path(cell, update)
    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        condition = cell.parent.name
        if (
            recorded.get("schema_version")
            != "hackrl_pack_inventory_overwrite_train_probability_transfer_eval_v1"
            or recorded.get("condition") != condition
            or recorded.get("seed") != seed
            or recorded.get("adaptation_update") != update
            or recorded.get("collector_slots")
            != update * int(SPEC["optimization"]["collector_slots_per_update"])
            or recorded.get("train_state_sha256")
            != base._train_state_sha256(runner.train_state)
            or not isinstance(recorded.get("evaluation"), dict)
        ):
            raise RuntimeError(f"stale or incompatible evaluation: {path}")
        return
    condition = cell.parent.name
    base._write_json(
        path,
        {
            "schema_version": (
                "hackrl_pack_inventory_overwrite_train_probability_transfer_eval_v1"
            ),
            "condition": condition,
            "seed": seed,
            "adaptation_update": update,
            "collector_slots": update
            * int(SPEC["optimization"]["collector_slots_per_update"]),
            "train_state_sha256": base._train_state_sha256(runner.train_state),
            "evaluation": lr._evaluate(
                network, runner.train_state.params, seed, condition
            ),
        },
    )
    print(f"[evaluation] {path}", flush=True)


def _summarize_cell(
    cell: Path,
    condition: str,
    seed: int,
    runner,
    tracker: dict,
    origin: dict,
) -> dict:
    summary = lr._summarize_cell(
        cell, condition, seed, runner, tracker, origin
    )
    summary["schema_version"] = (
        "hackrl_pack_inventory_overwrite_train_probability_transfer_cell_v1"
    )
    return summary


def _run_manifest() -> dict:
    manifest = lr._run_manifest()
    manifest.update(
        {
            "schema_version": (
                "hackrl_pack_inventory_overwrite_train_probability_transfer_run_v1"
            ),
            "reused_final_evaluation_sha256": SPEC["prerequisites"][
                "reused_final_evaluation_sha256"
            ],
            "reused_source_origin_sha256": SPEC["prerequisites"][
                "reused_source_origin_sha256"
            ],
            "exact_formula_basis_sha256": SPEC["prerequisites"][
                "exact_formula_basis_sha256"
            ],
            "evaluation_probabilities": SPEC["evaluation_probabilities"],
            "new_training_job_count": SPEC["budget"]["total_jobs"],
        }
    )
    return manifest


def _install_overrides() -> None:
    _configure_lr_module()
    lr._install_overrides()
    base._validate_prerequisite = _validate_prerequisite
    base._write_evaluation = _write_evaluation
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
        except Exception:
            base._write_json(
                base._cell(condition, seed) / "failed.json",
                {"failed_unix": time.time(), "traceback": traceback.format_exc()},
            )
            raise
        else:
            base._capacity_checkpoint()
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
