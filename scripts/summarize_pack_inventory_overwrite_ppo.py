#!/usr/bin/env python3
"""Verify and aggregate the PACK inventory-overwrite PPO comparison."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from run_pack_inventory_overwrite_ppo import (
    CONDITIONS,
    EXECUTION_SOURCES,
    LEARNING_CONDITIONS,
    REPOSITORY,
    RUN_ROOT,
    SCIENCE_UPDATES,
    SEEDS,
    SPEC,
    _cell,
    _evaluation_path,
    _sha256,
    _source_checkpoint_hashes,
    _write_json,
    build_jobs,
)


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric(evaluation: dict, kernel: str, view: str, name: str):
    return evaluation["kernels"][kernel][view]["overall"][name]


def _mean(values) -> float:
    return float(np.mean([float(value) for value in values]))


def _current_execution_hashes() -> dict[str, str]:
    return {
        str(path): _sha256(REPOSITORY / path) for path in EXECUTION_SOURCES
    }


def _verify_provenance() -> dict:
    manifest = _load(RUN_ROOT / "run_manifest.json")
    if manifest["execution_source_sha256"] != _current_execution_hashes():
        raise RuntimeError("execution source hashes changed after launch")
    if manifest["source_checkpoint_sha256"] != _source_checkpoint_hashes():
        raise RuntimeError("source checkpoints changed after launch")
    return manifest


def _evaluation_episode_count(document: dict) -> int:
    return sum(
        int(document["kernels"][kernel][view]["overall"]["episodes"])
        for kernel in ("fixed", "inventory_overwrite")
        for view in ("mode", "sample")
    )


def _final_block(evaluation: dict, kernel: str, view: str) -> dict:
    metrics = evaluation["kernels"][kernel][view]["overall"]
    keys = (
        "success_rate",
        "mean_discounted_return",
        "mean_length",
        "loss_episode_rate",
        "mean_lost_grain",
        "deliver_before_first_harvest_rate",
        "store_before_first_harvest_rate",
        "protect_before_first_harvest_rate",
        "successful_zero_loss_rate",
        "no_loss_failure_rate",
    )
    return {key: metrics[key] for key in keys}


def _curve(condition: str, seed: int) -> dict:
    return {
        str(update): {
            "collector_slots": int(
                SPEC["optimization"]["collector_slots_per_block"]
            )
            * update,
            "harmful_mode": _final_block(
                _load(_evaluation_path(_cell(condition, seed), update)),
                "inventory_overwrite",
                "mode",
            ),
            "fixed_mode": _final_block(
                _load(_evaluation_path(_cell(condition, seed), update)),
                "fixed",
                "mode",
            ),
        }
        for update in SCIENCE_UPDATES
    }


def aggregate() -> dict:
    provenance = _verify_provenance()
    summaries = {}
    evaluations = {}
    total_evaluation_episodes = 0
    for condition in CONDITIONS:
        summaries[condition] = {}
        evaluations[condition] = {}
        for seed in SEEDS:
            summary_path = _cell(condition, seed) / "summary.json"
            if not summary_path.is_file():
                raise RuntimeError(f"missing summary: {summary_path}")
            summary = _load(summary_path)
            if summary.get("status") != "complete":
                raise RuntimeError(f"incomplete summary: {summary_path}")
            summaries[condition][seed] = summary
            evaluations[condition][seed] = {}
            for update in SCIENCE_UPDATES:
                path = _evaluation_path(_cell(condition, seed), update)
                if not path.is_file():
                    raise RuntimeError(f"missing evaluation: {path}")
                document = _load(path)
                evaluations[condition][seed][update] = document
                total_evaluation_episodes += _evaluation_episode_count(document)

    paired_start_checks = []
    rows = []
    for seed in SEEDS:
        origins = {
            condition: _load(_cell(condition, seed) / "origin.json")
            for condition in CONDITIONS
        }
        train_hashes = {
            origin["source_train_state_sha256"] for origin in origins.values()
        }
        worker_hashes = {
            origin["worker_initial_sha256"] for origin in origins.values()
        }
        initial_evaluations = {
            json.dumps(
                evaluations[condition][seed][0]["kernels"], sort_keys=True
            )
            for condition in CONDITIONS
        }
        paired_start_checks.append(
            {
                "seed": seed,
                "train_state_equal": len(train_hashes) == 1,
                "fresh_worker_state_equal": len(worker_hashes) == 1,
                "update0_cross_evaluation_equal": len(initial_evaluations) == 1,
            }
        )
        if not all(
            paired_start_checks[-1][key]
            for key in (
                "train_state_equal",
                "fresh_worker_state_equal",
                "update0_cross_evaluation_equal",
            )
        ):
            raise AssertionError(f"paired initialization failed for seed {seed}")

        final = {
            condition: evaluations[condition][seed][int(SPEC["optimization"]["updates"])]
            for condition in CONDITIONS
        }
        normal_harmful = _final_block(
            final["normal_continue"], "inventory_overwrite", "mode"
        )
        adapt_harmful = _final_block(
            final["harmful_adapt"], "inventory_overwrite", "mode"
        )
        row = {
            "seed": seed,
            "final_mode": {
                condition: {
                    "harmful": _final_block(
                        final[condition], "inventory_overwrite", "mode"
                    ),
                    "fixed": _final_block(final[condition], "fixed", "mode"),
                }
                for condition in CONDITIONS
            },
            "final_sample": {
                condition: {
                    "harmful": _final_block(
                        final[condition], "inventory_overwrite", "sample"
                    ),
                    "fixed": _final_block(final[condition], "fixed", "sample"),
                }
                for condition in CONDITIONS
            },
            "paired_harmful_adapt_minus_normal_continue": {
                key: float(adapt_harmful[key]) - float(normal_harmful[key])
                for key in (
                    "success_rate",
                    "mean_discounted_return",
                    "mean_length",
                    "loss_episode_rate",
                    "mean_lost_grain",
                    "protect_before_first_harvest_rate",
                    "successful_zero_loss_rate",
                    "no_loss_failure_rate",
                )
            },
            "collection_exposure": {
                condition: {
                    key: summaries[condition][seed][key]
                    for key in (
                        "collector_slots",
                        "physical_transitions",
                        "valid_command_transitions",
                        "valid_ppo_samples",
                        "optimizer_applications",
                        "completed_world_episodes",
                        "loss_episodes",
                        "loss_episode_rate",
                        "cumulative_lost_grain",
                        "physical_loss_events",
                        "goal_successes_during_collection",
                    )
                }
                for condition in CONDITIONS
            },
            "curves": {
                condition: _curve(condition, seed) for condition in CONDITIONS
            },
        }
        rows.append(row)

    metric_names = tuple(
        rows[0]["paired_harmful_adapt_minus_normal_continue"].keys()
    )
    paired_mean = {
        key: _mean(
            row["paired_harmful_adapt_minus_normal_continue"][key]
            for row in rows
        )
        for key in metric_names
    }
    condition_means = {}
    for condition in CONDITIONS:
        condition_means[condition] = {
            "final_harmful_mode": {
                key: _mean(
                    row["final_mode"][condition]["harmful"][key] for row in rows
                )
                for key in rows[0]["final_mode"][condition]["harmful"]
            },
            "final_fixed_mode": {
                key: _mean(
                    row["final_mode"][condition]["fixed"][key] for row in rows
                )
                for key in rows[0]["final_mode"][condition]["fixed"]
            },
            "collection_exposure": {
                key: _mean(
                    row["collection_exposure"][condition][key] for row in rows
                )
                for key in rows[0]["collection_exposure"][condition]
            },
        }

    total_collector_slots = sum(
        int(summary["collector_slots"])
        for condition in CONDITIONS
        for summary in summaries[condition].values()
    )
    total_physical_transitions = sum(
        int(summary["physical_transitions"])
        for condition in CONDITIONS
        for summary in summaries[condition].values()
    )
    total_optimizer_applications = sum(
        int(summary["optimizer_applications"])
        for condition in CONDITIONS
        for summary in summaries[condition].values()
    )
    if total_collector_slots != int(SPEC["budget"]["total_collector_slots"]):
        raise AssertionError("total collector slots do not match the preregistration")
    if total_physical_transitions != total_collector_slots:
        raise AssertionError("physical transitions and collector slots diverged")
    if total_evaluation_episodes != int(
        SPEC["evaluation"]["planned_evaluation_episodes"]
    ):
        raise AssertionError("evaluation episode count changed")
    if any(
        summaries["frozen_harmful"][seed]["optimizer_applications"] != 0
        for seed in SEEDS
    ):
        raise AssertionError("frozen condition changed optimizer state")

    if any(
        summaries["normal_continue"][seed]["cumulative_lost_grain"] != 0
        for seed in SEEDS
    ):
        raise AssertionError("fixed-kernel continuation recorded physical loss")
    return {
        "schema_version": "hackrl_pack_inventory_overwrite_ppo_result_v1",
        "status": "complete_fixed_budget_no_automatic_extension",
        "manifest_id": SPEC["manifest_id"],
        "execution_head": provenance["execution_head"],
        "independent_unit": "learner seed",
        "seeds": list(SEEDS),
        "jobs_completed": len(build_jobs()),
        "paired_start_checks": paired_start_checks,
        "rows": rows,
        "condition_means": condition_means,
        "paired_harmful_adapt_minus_normal_continue_mean": paired_mean,
        "accounting": {
            "collector_slots": total_collector_slots,
            "physical_transitions": total_physical_transitions,
            "valid_ppo_samples": sum(
                int(summaries[condition][seed]["valid_ppo_samples"])
                for condition in CONDITIONS
                for seed in SEEDS
            ),
            "optimizer_applications": total_optimizer_applications,
            "evaluation_episodes": total_evaluation_episodes,
            "teacher_updates": 0,
            "bc_updates": 0,
        },
        "interpretation_guardrails": {
            "lower_loss_without_success_is_not_recovery": True,
            "dual_history_is_not_compared_against_pure_ppo_pretraining": True,
            "mode_and_sample_are_separate": True,
            "episode_count_is_not_a_replication_unit": True,
            "automatic_extension": False,
        },
        "completed_unix": time.time(),
    }


def main() -> None:
    result = aggregate()
    _write_json(RUN_ROOT / "result.json", result)
    _write_json(
        RUN_ROOT / "verified_complete.json",
        {
            "status": result["status"],
            "result_sha256": _sha256(RUN_ROOT / "result.json"),
            "completed_unix": time.time(),
        },
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "jobs_completed": result["jobs_completed"],
                "accounting": result["accounting"],
                "paired_mean": result[
                    "paired_harmful_adapt_minus_normal_continue_mean"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
