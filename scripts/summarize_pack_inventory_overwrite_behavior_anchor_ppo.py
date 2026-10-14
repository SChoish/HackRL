#!/usr/bin/env python3
"""Validate and aggregate the PACK behavior-anchored PPO experiment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC_PATH = (
    REPOSITORY
    / "docs/manifests/pack_inventory_overwrite_behavior_anchor_ppo_v1.json"
)
RESULT_PATH = (
    REPOSITORY
    / "docs/manifests/pack_inventory_overwrite_behavior_anchor_ppo_v1_results.json"
)
AUDIT_PATH = (
    REPOSITORY
    / "docs/manifests/pack_inventory_overwrite_behavior_anchor_ppo_v1_computation.json"
)
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
RUN_ROOT = Path(SPEC["run_root"])
SEEDS = tuple(int(seed) for seed in SPEC["seeds"])
CONDITIONS = tuple(SPEC["conditions"])
PANELS = tuple(SPEC["evaluation"]["development_panels"])
UPDATES = tuple(int(value) for value in SPEC["evaluation"]["adaptation_updates"])
STANDARD = next(
    name
    for name, row in SPEC["conditions"].items()
    if float(row["behavior_anchor_coefficient"]) == 0.0
)
ANCHORED = next(
    name
    for name, row in SPEC["conditions"].items()
    if float(row["behavior_anchor_coefficient"]) > 0.0
)
AGGREGATOR_RELATIVE_PATH = "scripts/summarize_pack_inventory_overwrite_behavior_anchor_ppo.py"
PRE_CORRECTION_AGGREGATE_SHA256 = (
    "03c96588799889c974d3ae28f1e577bf0a02bc2138556d43cece40f52bb42939"
)


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values) -> float:
    values = [float(value) for value in values]
    return math.fsum(values) / len(values)


def sign_counts(values) -> dict:
    values = [float(value) for value in values]
    tolerance = float(SPEC["statistics"]["sign_tolerance"])
    return {
        "positive": sum(value > tolerance for value in values),
        "equal_within_tolerance": sum(
            abs(value) <= tolerance for value in values
        ),
        "negative": sum(value < -tolerance for value in values),
    }


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY, text=True
    ).strip()


def _expected_jobs() -> list[dict]:
    return [
        {
            "id": f"{condition}-seed{seed}",
            "condition": condition,
            "seed": seed,
            "learning": True,
        }
        for seed in SEEDS
        for condition in CONDITIONS
    ]


def validate_run() -> dict:
    manifest = read(RUN_ROOT / "run_manifest.json")
    completion = read(RUN_ROOT / "training_complete.json")
    if completion.get("status") != "all_10_jobs_complete_pending_aggregation":
        raise AssertionError("ten-job completion marker is missing")
    if completion.get("execution_head") != manifest.get("execution_head"):
        raise AssertionError("completion and execution heads differ")
    if manifest.get("jobs") != _expected_jobs():
        raise AssertionError("job matrix changed")
    if manifest.get("manifest_sha256") != sha256(SPEC_PATH):
        raise AssertionError("design manifest changed after launch")
    if manifest.get("planned_collector_slots") != SPEC["budget"][
        "collector_slots"
    ]:
        raise AssertionError("collector budget changed")
    if manifest.get("automatic_extension") is not False:
        raise AssertionError("automatic extension occurred")
    for relative, expected in manifest["execution_source_sha256"].items():
        if relative == AGGREGATOR_RELATIVE_PATH:
            continue
        if sha256(REPOSITORY / relative) != expected:
            raise AssertionError(f"execution source changed: {relative}")
    for path, expected in manifest["source_checkpoint_sha256"].items():
        if sha256(Path(path)) != expected:
            raise AssertionError(f"source checkpoint changed: {path}")
    if list(RUN_ROOT.rglob("failed.json")):
        raise AssertionError("failed-job marker remains")
    return manifest


def validate_cell(condition: str, seed: int) -> dict:
    cell = RUN_ROOT / condition / f"seed{seed}"
    summary = read(cell / "summary.json")
    origin = read(cell / "origin.json")
    coefficient = float(
        SPEC["conditions"][condition]["behavior_anchor_coefficient"]
    )
    if (
        summary.get("status") != "complete"
        or summary.get("condition") != condition
        or summary.get("seed") != seed
        or summary.get("schema_version")
        != "hackrl_pack_inventory_overwrite_behavior_anchor_cell_v1"
        or float(summary.get("behavior_anchor_coefficient")) != coefficient
        or float(origin.get("behavior_anchor_coefficient")) != coefficient
    ):
        raise AssertionError(f"cell identity changed: {condition} seed {seed}")
    if summary["collector_slots"] != SPEC["optimization"][
        "collector_slots_per_job"
    ]:
        raise AssertionError("cell collector budget changed")
    if (
        summary["optimizer_applications"] + summary["empty_minibatches"]
        != SPEC["optimization"]["scheduled_optimizer_applications_per_job"]
    ):
        raise AssertionError("optimizer schedule changed")
    if summary["teacher_updates"] != 0 or summary["bc_updates"] != 0:
        raise AssertionError("teacher or BC update occurred")
    expected_anchor_compute = (
        summary["optimizer_applications"]
        * int(summary["final_train_anchor_behavior_change"]["state_count"])
        if coefficient > 0.0
        else 0
    )
    if (
        summary["behavior_anchor_forward_state_evaluations"]
        != expected_anchor_compute
    ):
        raise AssertionError("behavior-anchor compute accounting changed")
    update_files = sorted(
        (cell / "updates").glob("update_*.json"),
        key=lambda path: int(path.stem.removeprefix("update_")),
    )
    if len(update_files) != SPEC["optimization"]["updates"]:
        raise AssertionError("update record count changed")
    slots_per_update = int(SPEC["optimization"]["collector_slots_per_update"])
    applications_per_update = (
        int(SPEC["optimization"]["num_envs"])
        * int(SPEC["optimization"]["num_steps"])
        // int(SPEC["optimization"]["minibatch_size"])
    )
    for update, path in enumerate(update_files, start=1):
        record = read(path)
        if (
            int(record["adaptation_update"]) != update
            or int(record["collector_slots_this_update"]) != slots_per_update
            or int(record["metrics"]["transitions"]) != slots_per_update
            or int(record["cumulative"]["collector_slots"])
            != update * slots_per_update
            or int(record["optimizer_applications_this_update"])
            + int(record["metrics"]["empty_minibatches"])
            != applications_per_update
        ):
            raise AssertionError(
                f"update accounting changed: {condition} seed {seed} update {update}"
            )
    evaluations = {}
    checkpoint_names = sorted(
        path.name for path in (cell / "checkpoints").glob("adapt_*")
    )
    if checkpoint_names != sorted(f"adapt_{update}" for update in UPDATES):
        raise AssertionError("retained checkpoint set changed")
    for update in UPDATES:
        checkpoint = cell / "checkpoints" / f"adapt_{update}"
        receipt = read(checkpoint / "receipt.json")
        evaluation_path = cell / "evaluations" / f"adapt_{update}.json"
        evaluation = read(evaluation_path)
        if receipt["adaptation_update"] != update:
            raise AssertionError("checkpoint coordinate changed")
        if sha256(checkpoint / "state.msgpack") != receipt["state_sha256"]:
            raise AssertionError("checkpoint receipt hash mismatch")
        if evaluation["train_state_sha256"] != receipt["train_state_sha256"]:
            raise AssertionError("checkpoint/evaluation identity mismatch")
        if update == 0 and (
            receipt["train_state_sha256"]
            != origin["source_train_state_sha256"]
            or receipt["train_state_sha256"]
            != origin["branch_train_state_sha256"]
        ):
            raise AssertionError("update-zero checkpoint differs from source")
        if set(evaluation["panels"]) != set(PANELS):
            raise AssertionError("development-panel set changed")
        evaluations[update] = evaluation
    resume = cell / "resume"
    if resume.is_dir() and list(resume.iterdir()):
        raise AssertionError("rolling resume checkpoint was not pruned")
    return {
        "cell": cell,
        "summary": summary,
        "origin": origin,
        "evaluations": evaluations,
        "update_files": update_files,
    }


def _normal_retention(initial: dict, final: dict) -> dict:
    before = {
        int(row["state_index"]): bool(row["success"])
        for row in initial["fixed_kernel"]["mode"]["episode_records"]
    }
    after = {
        int(row["state_index"]): bool(row["success"])
        for row in final["fixed_kernel"]["mode"]["episode_records"]
    }
    return {
        "source_successes": sum(before.values()),
        "retained_source_successes": sum(
            before[index] and after[index] for index in before
        ),
        "lost_source_successes": sum(
            before[index] and not after[index] for index in before
        ),
        "new_successes": sum(
            not before[index] and after[index] for index in before
        ),
    }


def seed_row(condition: str, seed: int, cell: dict) -> dict:
    initial_doc = cell["evaluations"][0]
    final_doc = cell["evaluations"][UPDATES[-1]]
    panels = {}
    for split in PANELS:
        initial = initial_doc["panels"][split]
        final = final_doc["panels"][split]
        exact = final["target_kernel"]["exact_mode"]
        source_exact = initial["target_kernel"]["exact_mode"]
        panels[split] = {
            "exact_mode": exact,
            "source_exact_mode": source_exact,
            "final_minus_source_exact_return": exact[
                "expected_discounted_return"
            ]
            - source_exact["expected_discounted_return"],
            "normal_mode": final["fixed_kernel"]["mode"]["overall"],
            "normal_sample": final["fixed_kernel"]["sample"]["overall"],
            "target_sample": final["target_kernel"]["sample"]["overall"],
            "behavior_change_from_source": final[
                "behavior_change_from_source"
            ],
            "normal_retention": _normal_retention(initial, final),
        }
    summary = cell["summary"]
    return {
        "condition": condition,
        "seed": seed,
        "behavior_anchor_coefficient": summary["behavior_anchor_coefficient"],
        "panels": panels,
        "train_anchor_behavior_change": summary[
            "final_train_anchor_behavior_change"
        ],
        "collection": {
            key: summary[key]
            for key in (
                "collector_slots",
                "valid_ppo_samples",
                "optimizer_applications",
                "empty_minibatches",
                "completed_world_episodes",
                "loss_episodes",
                "loss_episode_rate",
                "cumulative_lost_grain",
                "physical_loss_events",
                "overwrite_trigger_events",
                "goal_successes_during_collection",
                "behavior_anchor_forward_state_evaluations",
            )
        },
    }


def aggregate_condition(rows: list[dict]) -> dict:
    output = {"seeds": len(rows), "panels": {}}
    for split in PANELS:
        panel_rows = [row["panels"][split] for row in rows]
        output["panels"][split] = {
            "exact_mode_mean": {
                key: mean(item["exact_mode"][key] for item in panel_rows)
                for key in (
                    "success_rate",
                    "expected_discounted_return",
                    "failure_probability",
                    "loss_episode_probability",
                    "risky_start_count",
                )
            },
            "source_exact_mode_mean": {
                key: mean(item["source_exact_mode"][key] for item in panel_rows)
                for key in ("success_rate", "expected_discounted_return")
            },
            "normal_mode_mean": {
                key: mean(item["normal_mode"][key] for item in panel_rows)
                for key in ("success_rate", "mean_discounted_return", "mean_length")
            },
            "normal_sample_mean": {
                key: mean(item["normal_sample"][key] for item in panel_rows)
                for key in ("success_rate", "mean_discounted_return")
            },
            "target_sample_mean": {
                key: mean(item["target_sample"][key] for item in panel_rows)
                for key in ("success_rate", "mean_discounted_return")
            },
            "behavior_change_mean": {
                key: mean(
                    item["behavior_change_from_source"][key]
                    for item in panel_rows
                )
                for key in (
                    "mean_kl_source_to_current",
                    "argmax_action_change_rate",
                )
            },
            "normal_retention_totals": {
                key: sum(item["normal_retention"][key] for item in panel_rows)
                for key in panel_rows[0]["normal_retention"]
            },
            "final_minus_source_exact_return_signs": sign_counts(
                item["final_minus_source_exact_return"] for item in panel_rows
            ),
        }
    output["train_anchor_behavior_change_mean"] = {
        key: mean(row["train_anchor_behavior_change"][key] for row in rows)
        for key in ("mean_kl_source_to_current", "argmax_action_change_rate")
    }
    output["collection_totals"] = {
        key: sum(int(row["collection"][key]) for row in rows)
        for key in rows[0]["collection"]
        if key != "loss_episode_rate"
    }
    output["mean_collection_loss_episode_rate"] = mean(
        row["collection"]["loss_episode_rate"] for row in rows
    )
    return output


def paired_contrast(rows: dict[str, list[dict]], split: str) -> dict:
    standard = {row["seed"]: row for row in rows[STANDARD]}
    anchored = {row["seed"]: row for row in rows[ANCHORED]}
    items = []
    for seed in SEEDS:
        left = standard[seed]["panels"][split]
        right = anchored[seed]["panels"][split]
        source = right["source_exact_mode"]
        items.append(
            {
                "seed": seed,
                "anchored_minus_ppo_exact_return": right["exact_mode"][
                    "expected_discounted_return"
                ]
                - left["exact_mode"]["expected_discounted_return"],
                "anchored_minus_source_exact_return": right["exact_mode"][
                    "expected_discounted_return"
                ]
                - source["expected_discounted_return"],
                "anchored_minus_ppo_exact_success": right["exact_mode"][
                    "success_rate"
                ]
                - left["exact_mode"]["success_rate"],
                "anchored_minus_source_exact_success": right["exact_mode"][
                    "success_rate"
                ]
                - source["success_rate"],
                "anchored_minus_ppo_normal_success": right["normal_mode"][
                    "success_rate"
                ]
                - left["normal_mode"]["success_rate"],
                "anchored_minus_ppo_normal_return": right["normal_mode"][
                    "mean_discounted_return"
                ]
                - left["normal_mode"]["mean_discounted_return"],
                "anchored_minus_ppo_risky_starts": right["exact_mode"][
                    "risky_start_count"
                ]
                - left["exact_mode"]["risky_start_count"],
            }
        )
    fields = tuple(key for key in items[0] if key != "seed")
    return {
        "panel": split,
        "left": ANCHORED,
        "right": STANDARD,
        "per_seed": items,
        "means": {field: mean(item[field] for item in items) for field in fields},
        "anchored_minus_ppo_exact_return_signs": sign_counts(
            item["anchored_minus_ppo_exact_return"] for item in items
        ),
        "anchored_minus_source_exact_return_signs": sign_counts(
            item["anchored_minus_source_exact_return"] for item in items
        ),
    }


def learning_curve(condition: str, cells: dict, split: str) -> list[dict]:
    output = []
    for update in UPDATES:
        panels = [
            cells[(condition, seed)]["evaluations"][update]["panels"][split]
            for seed in SEEDS
        ]
        output.append(
            {
                "adaptation_update": update,
                "collector_slots_per_seed": update
                * SPEC["optimization"]["collector_slots_per_update"],
                "exact_return": mean(
                    panel["target_kernel"]["exact_mode"][
                        "expected_discounted_return"
                    ]
                    for panel in panels
                ),
                "exact_success": mean(
                    panel["target_kernel"]["exact_mode"]["success_rate"]
                    for panel in panels
                ),
                "normal_success": mean(
                    panel["fixed_kernel"]["mode"]["overall"]["success_rate"]
                    for panel in panels
                ),
                "normal_return": mean(
                    panel["fixed_kernel"]["mode"]["overall"][
                        "mean_discounted_return"
                    ]
                    for panel in panels
                ),
            }
        )
    return output


def main() -> None:
    run_manifest = validate_run()
    cells = {
        (condition, seed): validate_cell(condition, seed)
        for condition in CONDITIONS
        for seed in SEEDS
    }
    for seed in SEEDS:
        left = cells[(STANDARD, seed)]["evaluations"][0]
        right = cells[(ANCHORED, seed)]["evaluations"][0]
        if left["train_state_sha256"] != right["train_state_sha256"]:
            raise AssertionError(f"paired update-zero state differs for seed {seed}")
        if left["panels"] != right["panels"]:
            raise AssertionError(f"paired update-zero evaluation differs for seed {seed}")
        left_origin = cells[(STANDARD, seed)]["origin"]
        right_origin = cells[(ANCHORED, seed)]["origin"]
        for key in (
            "source_checkpoint",
            "source_train_state_sha256",
            "branch_train_state_sha256",
            "worker_initial_sha256",
            "source_optimizer_step",
            "source_global_update",
            "fresh_environment_steps",
            "fresh_rollout_cursor",
        ):
            if left_origin[key] != right_origin[key]:
                raise AssertionError(
                    f"paired origin field {key} differs for seed {seed}"
                )
    rows = {
        condition: [
            seed_row(condition, seed, cells[(condition, seed)]) for seed in SEEDS
        ]
        for condition in CONDITIONS
    }
    contrasts = {split: paired_contrast(rows, split) for split in PANELS}
    aggregation_revision = {
        "launch_aggregator_sha256": run_manifest["execution_source_sha256"][
            AGGREGATOR_RELATIVE_PATH
        ],
        "current_aggregator_sha256": sha256(
            REPOSITORY / AGGREGATOR_RELATIVE_PATH
        ),
        "pre_correction_aggregate_sha256": PRE_CORRECTION_AGGREGATE_SHA256,
        "scope": "Materialize sign-count iterables before the positive, equal, and negative passes. Seed rows, means, checkpoints, evaluations, and training artifacts are unchanged.",
    }
    result = {
        "schema_version": "hackrl_pack_inventory_overwrite_behavior_anchor_results_v1",
        "status": "complete_fixed_budget_no_automatic_extension",
        "manifest_id": SPEC["manifest_id"],
        "execution_head": run_manifest["execution_head"],
        "primary_panel": "validation",
        "aggregation_revision": aggregation_revision,
        "conditions": {
            condition: aggregate_condition(rows[condition])
            for condition in CONDITIONS
        },
        "paired_contrasts": contrasts,
        "seed_rows": rows,
        "learning_curves": {
            condition: {
                split: learning_curve(condition, cells, split)
                for split in PANELS
            }
            for condition in CONDITIONS
        },
        "interpretation_contract": {
            "ordinary_comparison": "anchored minus ordinary PPO within seed",
            "source_comparison": "anchored final minus its byte-identical update-zero source within seed",
            "development_panels_only": True,
            "no_population_significance_claim": True,
            "no_automatic_extension_or_beta_tuning": True,
        },
    }
    raw_path = RUN_ROOT / "aggregate.json"
    write(raw_path, result)
    write(RESULT_PATH, result)
    input_hashes = {}
    for cell in cells.values():
        paths = [cell["cell"] / "summary.json", *cell["update_files"]]
        paths.extend(
            cell["cell"] / "evaluations" / f"adapt_{update}.json"
            for update in UPDATES
        )
        paths.extend(
            cell["cell"] / "checkpoints" / f"adapt_{update}" / "state.msgpack"
            for update in UPDATES
        )
        input_hashes.update({str(path): sha256(path) for path in paths})
    audit = {
        "schema_version": "hackrl_pack_inventory_overwrite_behavior_anchor_computation_v1",
        "status": "complete",
        "created_unix": time.time(),
        "execution_head": run_manifest["execution_head"],
        "aggregator_head": git("rev-parse", "HEAD"),
        "manifest_sha256": sha256(SPEC_PATH),
        "aggregate_sha256": sha256(raw_path),
        "result_sha256": sha256(RESULT_PATH),
        "aggregation_revision": aggregation_revision,
        "validated_cells": len(cells),
        "validated_update_records": sum(
            len(cell["update_files"]) for cell in cells.values()
        ),
        "validated_retained_checkpoints": len(cells) * len(UPDATES),
        "input_sha256": input_hashes,
        "capacity_checks": sum(
            1 for _ in (RUN_ROOT / "capacity_checks.jsonl").open(encoding="utf-8")
        ),
    }
    write(AUDIT_PATH, audit)
    write(RUN_ROOT / "aggregation_complete.json", audit)
    print(json.dumps(result["paired_contrasts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
