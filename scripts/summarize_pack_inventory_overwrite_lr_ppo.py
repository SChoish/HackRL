#!/usr/bin/env python3
"""Validate and aggregate the fixed-budget PPO learning-rate comparison."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC_PATH = (
    REPOSITORY / "docs/manifests/pack_inventory_overwrite_lr_ppo_v1.json"
)
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
RUN_ROOT = Path(SPEC["run_root"])
SEEDS = tuple(int(seed) for seed in SPEC["seeds"])
CONDITIONS = tuple(SPEC["conditions"])
UPDATES = tuple(
    int(update) for update in SPEC["evaluation"]["adaptation_updates"]
)
PAIRS = {
    "p05": ("p05_lr1", "p05_lr025"),
    "p30": ("p30_lr1", "p30_lr025"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def mean(values) -> float:
    values = list(values)
    return math.fsum(float(value) for value in values) / len(values)


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY, text=True
    ).strip()


def prerequisite_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY / path


def validate_inputs() -> tuple[dict, dict]:
    run_manifest = read(RUN_ROOT / "run_manifest.json")
    completion = read(RUN_ROOT / "training_complete.json")
    if completion.get("status") != "all_20_jobs_complete_pending_aggregation":
        raise AssertionError("training completion marker is missing")
    if completion.get("execution_head") != run_manifest["execution_head"]:
        raise AssertionError("completion and run-manifest heads differ")
    expected_jobs = [
        {
            "id": f"{condition}-seed{seed}",
            "condition": condition,
            "seed": seed,
            "learning": True,
        }
        for seed in SEEDS
        for condition in CONDITIONS
    ]
    if run_manifest.get("jobs") != expected_jobs:
        raise AssertionError("run-manifest job matrix changed")
    if (
        run_manifest.get("manifest_id") != SPEC["manifest_id"]
        or run_manifest.get("planned_collector_slots")
        != SPEC["budget"]["total_collector_slots"]
        or run_manifest.get("automatic_extension") is not False
    ):
        raise AssertionError("run-manifest experiment contract changed")
    if run_manifest["manifest_sha256"] != sha256(SPEC_PATH):
        raise AssertionError("experiment manifest changed after launch")
    if set(run_manifest["execution_source_sha256"]) != set(
        SPEC["execution_sources"]
    ):
        raise AssertionError("execution-source set changed")
    for relative, expected in run_manifest["execution_source_sha256"].items():
        if sha256(REPOSITORY / relative) != expected:
            raise AssertionError(
                f"execution source changed after launch: {relative}"
            )
    expected_source_files = {
        str(
            Path(SPEC["source_root"])
            / f"seed{seed}"
            / "checkpoints"
            / SPEC["source_checkpoint"] / name
        )
        for seed in SEEDS
        for name in ("arm.json", "config.json", "metadata.json", "state.msgpack")
    }
    if set(run_manifest["source_checkpoint_sha256"]) != expected_source_files:
        raise AssertionError("source-checkpoint file set changed")
    for path, expected in run_manifest["source_checkpoint_sha256"].items():
        if sha256(Path(path)) != expected:
            raise AssertionError(f"source checkpoint changed: {path}")
    prerequisites = SPEC["prerequisites"]
    for path_key, hash_key in (
        ("prior_result", "prior_result_sha256"),
        ("prior_raw_result", "prior_raw_result_sha256"),
        ("reference_paths", "reference_paths_sha256"),
    ):
        if (
            sha256(prerequisite_path(prerequisites[path_key]))
            != prerequisites[hash_key]
        ):
            raise AssertionError(f"prerequisite changed: {path_key}")
    if list(RUN_ROOT.rglob("failed.json")):
        raise AssertionError("a failed-job marker remains")
    return run_manifest, completion


def validate_cell(condition: str, seed: int) -> dict:
    cell = RUN_ROOT / condition / f"seed{seed}"
    summary = read(cell / "summary.json")
    origin = read(cell / "origin.json")
    expected_condition = SPEC["conditions"][condition]
    expected_lr = (
        SPEC["optimization"]["source_learning_rate"]
        * expected_condition["learning_rate_multiplier"]
    )
    expected_source_path = (
        Path(SPEC["source_root"])
        / f"seed{seed}"
        / "checkpoints"
        / SPEC["source_checkpoint"]
    )
    source_config = read(expected_source_path / "config.json")
    if (
        summary.get("condition") != condition
        or summary.get("seed") != seed
        or origin.get("condition") != condition
        or origin.get("seed") != seed
        or summary.get("kernel") != expected_condition["kernel"]
        or origin.get("kernel") != expected_condition["kernel"]
        or summary.get("schema_version")
        != "hackrl_pack_inventory_overwrite_lr_cell_v1"
    ):
        raise AssertionError("cell identity changed")
    if summary.get("status") != "complete":
        raise AssertionError(f"incomplete cell: {condition} seed {seed}")
    if summary["collector_slots"] != SPEC["optimization"][
        "collector_slots_per_job"
    ]:
        raise AssertionError("cell collector budget changed")
    if summary["physical_transitions"] != summary["collector_slots"]:
        raise AssertionError("physical-transition accounting changed")
    if (
        summary["optimizer_applications"] + summary["empty_minibatches"]
        != SPEC["optimization"]["scheduled_optimizer_applications_per_job"]
    ):
        raise AssertionError("scheduled optimizer accounting changed")
    if summary["teacher_updates"] != 0 or summary["bc_updates"] != 0:
        raise AssertionError("teacher or BC update occurred")
    if (
        summary["overwrite_probability"]
        != expected_condition["overwrite_probability"]
        or summary["learning_rate_multiplier"]
        != expected_condition["learning_rate_multiplier"]
        or not math.isclose(
            summary["branch_learning_rate"],
            expected_lr,
            rel_tol=0.0, abs_tol=0.0
        )
    ):
        raise AssertionError("cell intervention changed")
    for key, expected in (
        ("overwrite_probability", expected_condition["overwrite_probability"]),
        (
            "learning_rate_multiplier",
            expected_condition["learning_rate_multiplier"],
        ),
        ("branch_learning_rate", expected_lr),
    ):
        if not math.isclose(
            float(origin[key]), float(expected), rel_tol=0.0, abs_tol=0.0
        ):
            raise AssertionError(f"origin {key} changed")
    if (
        origin["source_checkpoint"] != str(expected_source_path)
        or origin["source_global_update"] != SPEC["source_global_update"]
    ):
        raise AssertionError("origin does not match prescribed source")
    update_files = sorted(
        (cell / "updates").glob("update_*.json"),
        key=lambda path: int(path.stem.removeprefix("update_")),
    )
    if len(update_files) != SPEC["optimization"]["updates"]:
        raise AssertionError("update record count changed")
    for update, path in enumerate(update_files, start=1):
        record = read(path)
        if int(record["adaptation_update"]) != update:
            raise AssertionError("update ordering changed")
    checkpoint_root = cell / "checkpoints"
    checkpoint_names = sorted(
        path.name for path in checkpoint_root.glob("adapt_*")
    )
    expected_names = sorted(f"adapt_{update}" for update in UPDATES)
    if checkpoint_names != expected_names:
        raise AssertionError("retained checkpoint set changed")
    evaluations = {}
    for update in UPDATES:
        checkpoint = checkpoint_root / f"adapt_{update}"
        receipt = read(checkpoint / "receipt.json")
        config = read(checkpoint / "config.json")
        evaluation_doc = read(
            cell / "evaluations" / f"adapt_{update}.json"
        )
        if receipt["adaptation_update"] != update:
            raise AssertionError("checkpoint receipt update changed")
        if sha256(checkpoint / "state.msgpack") != receipt["state_sha256"]:
            raise AssertionError("checkpoint state hash changed")
        if (
            evaluation_doc["train_state_sha256"]
            != receipt["train_state_sha256"]
        ):
            raise AssertionError("evaluation and checkpoint identity differ")
        if evaluation_doc["collector_slots"] != update * SPEC[
            "optimization"
        ]["collector_slots_per_update"]:
            raise AssertionError("evaluation slot coordinate changed")
        if (
            config["variant"] != expected_condition["kernel"]
            or config["inventory_overwrite_probability"]
            != expected_condition["overwrite_probability"]
            or not math.isclose(
                config["learning_rate"],
                expected_lr,
                rel_tol=0.0, abs_tol=0.0
            )
        ):
            raise AssertionError("checkpoint configuration changed")
        if (
            source_config["learning_rate"]
            != SPEC["optimization"]["source_learning_rate"]
        ):
            raise AssertionError("source learning rate changed")
        intervention_fields = {
            "variant",
            "inventory_overwrite_probability",
            "learning_rate",
            "num_updates",
            "checkpoint_updates",
        }
        for key, source_value in source_config.items():
            if (
                key not in intervention_fields
                and config.get(key) != source_value
            ):
                raise AssertionError(
                    f"non-intervention config changed: {key}"
                )
        if (
            evaluation_doc.get("condition") != condition
            or evaluation_doc.get("seed") != seed
            or evaluation_doc.get("adaptation_update") != update
            or evaluation_doc.get("schema_version")
            != "hackrl_pack_inventory_overwrite_lr_eval_v1"
        ):
            raise AssertionError("evaluation identity changed")
        if update == 0 and (
            receipt["train_state_sha256"]
            != origin["source_train_state_sha256"]
            or receipt["train_state_sha256"]
            != origin["branch_train_state_sha256"]
        ):
            raise AssertionError(
                "update-zero state does not match restored source state"
            )
        evaluations[update] = evaluation_doc["evaluation"]
    resume = cell / "resume"
    if resume.is_dir() and list(resume.iterdir()):
        raise AssertionError("rolling resume checkpoint was not pruned")
    return {
        "summary": summary,
        "origin": origin,
        "evaluations": evaluations,
    }


def seed_row(condition: str, seed: int, cell: dict) -> dict:
    initial = cell["evaluations"][UPDATES[0]]
    final = cell["evaluations"][UPDATES[-1]]
    initial_exact = initial["target_kernel"]["exact_mode"]
    exact = final["target_kernel"]["exact_mode"]
    fixed_mode = final["fixed_kernel"]["mode"]["overall"]
    fixed_sample = final["fixed_kernel"]["sample"]["overall"]
    target_sample = final["target_kernel"]["sample"]["overall"]
    behavior = final["behavior_change_from_source"]
    summary = cell["summary"]
    return {
        "seed": seed,
        "condition": condition,
        "overwrite_probability": final["overwrite_probability"],
        "learning_rate_multiplier": final["learning_rate_multiplier"],
        "branch_learning_rate": summary["branch_learning_rate"],
        "exact_mode": {
            "success_rate": exact["existing_success_rate"],
            "expected_discounted_return": exact[
                "existing_expected_discounted_return"
            ],
            "failure_probability": exact["existing_failure_probability"],
            "loss_episode_probability": exact[
                "existing_loss_episode_probability"
            ],
            "source_success_rate": initial_exact["existing_success_rate"],
            "source_expected_discounted_return": initial_exact[
                "existing_expected_discounted_return"
            ],
            "final_minus_source_return": (
                exact["existing_expected_discounted_return"]
                - initial_exact["existing_expected_discounted_return"]
            ),
        },
        "fixed_mode": {
            key: fixed_mode[key]
            for key in (
                "success_rate",
                "mean_discounted_return",
                "mean_length",
                "deliver_before_first_harvest_rate",
                "store_before_first_harvest_rate",
                "protect_before_first_harvest_rate",
                "successful_zero_loss_rate",
                "no_loss_failure_rate",
            )
        },
        "fixed_sample": {
            key: fixed_sample[key]
            for key in ("success_rate", "mean_discounted_return")
        },
        "target_sample": {
            key: target_sample[key]
            for key in ("success_rate", "mean_discounted_return")
        },
        "risk_count_distribution": final["risk_profile"][
            "risk_count_distribution"
        ],
        "risky_start_count": sum(
            int(value)
            for key, value in final["risk_profile"][
                "risk_count_distribution"
            ].items()
            if int(key) > 0
        ),
        "behavior_change_from_source": behavior,
        "collection": {
            key: summary[key]
            for key in (
                "collector_slots",
                "physical_transitions",
                "valid_ppo_samples",
                "optimizer_applications",
                "empty_minibatches",
                "completed_world_episodes",
                "loss_episodes",
                "loss_episode_rate",
                "cumulative_lost_grain",
                "physical_loss_events",
                "overwrite_trigger_events",
                "overwrite_fire_events",
                "goal_successes_during_collection",
            )
        },
    }


def aggregate_rows(rows: list[dict]) -> dict:
    risk_counts = Counter()
    for row in rows:
        risk_counts.update(
            {
                int(key): int(value)
                for key, value in row["risk_count_distribution"].items()
            }
        )
    exact_keys = (
        "success_rate",
        "expected_discounted_return",
        "failure_probability",
        "loss_episode_probability",
        "source_success_rate",
        "source_expected_discounted_return",
        "final_minus_source_return",
    )
    fixed_keys = tuple(rows[0]["fixed_mode"])
    behavior_keys = (
        "mean_kl_source_to_current",
        "argmax_action_change_count",
        "argmax_action_change_rate",
    )
    collection_sum_keys = tuple(rows[0]["collection"])
    changes = [
        float(row["exact_mode"]["final_minus_source_return"])
        for row in rows
    ]
    trigger_total = sum(
        int(row["collection"]["overwrite_trigger_events"]) for row in rows
    )
    fire_total = sum(
        int(row["collection"]["overwrite_fire_events"]) for row in rows
    )
    return {
        "seeds": len(rows),
        "exact_mode_mean": {
            key: mean(row["exact_mode"][key] for row in rows)
            for key in exact_keys
        },
        "normal_kernel_mode_mean": {
            key: mean(row["fixed_mode"][key] for row in rows)
            for key in fixed_keys
        },
        "normal_sample_mean": {
            key: mean(row["fixed_sample"][key] for row in rows)
            for key in rows[0]["fixed_sample"]
        },
        "target_sample_mean": {
            key: mean(row["target_sample"][key] for row in rows)
            for key in rows[0]["target_sample"]
        },
        "behavior_change_mean": {
            key: mean(
                row["behavior_change_from_source"][key] for row in rows
            )
            for key in behavior_keys
        },
        "risk_count_distribution": {
            str(key): risk_counts[key] for key in sorted(risk_counts)
        },
        "risky_start_count": sum(row["risky_start_count"] for row in rows),
        "final_minus_source_signs": sign_counts(changes),
        "collection_totals": {
            key: sum(
                (
                    float(row["collection"][key])
                    if key == "loss_episode_rate"
                    else int(row["collection"][key])
                )
                for row in rows
            )
            for key in collection_sum_keys
            if key != "loss_episode_rate"
        },
        "mean_collection_loss_episode_rate": mean(
            row["collection"]["loss_episode_rate"] for row in rows
        ),
        "empirical_fire_per_risky_harvest": (
            fire_total / trigger_total if trigger_total else None
        ),
    }


def sign_counts(values: list[float]) -> dict:
    tolerance = float(SPEC["statistics"]["sign_tolerance"])
    return {
        "positive": sum(value > tolerance for value in values),
        "equal_within_tolerance": sum(
            abs(value) <= tolerance for value in values
        ),
        "negative": sum(value < -tolerance for value in values),
    }


def learning_curve(condition: str, cells: dict) -> list[dict]:
    output = []
    for update in UPDATES:
        evaluations = [
            cells[(condition, seed)]["evaluations"][update]
            for seed in SEEDS
        ]
        initial = [
            cells[(condition, seed)]["evaluations"][0]
            for seed in SEEDS
        ]
        exact = [
            evaluation["target_kernel"]["exact_mode"]
            for evaluation in evaluations
        ]
        output.append(
            {
                "adaptation_update": update,
                "collector_slots_per_seed": update
                * SPEC["optimization"]["collector_slots_per_update"],
                "exact_mode_success_rate": mean(
                    item["existing_success_rate"] for item in exact
                ),
                "exact_mode_expected_discounted_return": mean(
                    item["existing_expected_discounted_return"]
                    for item in exact
                ),
                "current_minus_source_return": mean(
                    item["existing_expected_discounted_return"]
                    - source["target_kernel"]["exact_mode"][
                        "existing_expected_discounted_return"
                    ]
                    for item, source in zip(exact, initial, strict=True)
                ),
                "exact_mode_loss_episode_probability": mean(
                    item["existing_loss_episode_probability"]
                    for item in exact
                ),
                "normal_kernel_mode_success_rate": mean(
                    evaluation["fixed_kernel"]["mode"]["overall"][
                        "success_rate"
                    ]
                    for evaluation in evaluations
                ),
                "normal_kernel_mode_return": mean(
                    evaluation["fixed_kernel"]["mode"]["overall"][
                        "mean_discounted_return"
                    ]
                    for evaluation in evaluations
                ),
                "risky_start_count": sum(
                    int(value)
                    for evaluation in evaluations
                    for key, value in evaluation["risk_profile"][
                        "risk_count_distribution"
                    ].items()
                    if int(key) > 0
                ),
                "mean_actor_kl_source_to_current": mean(
                    evaluation["behavior_change_from_source"][
                        "mean_kl_source_to_current"
                    ]
                    for evaluation in evaluations
                ),
                "argmax_action_change_rate": mean(
                    evaluation["behavior_change_from_source"][
                        "argmax_action_change_rate"
                    ]
                    for evaluation in evaluations
                ),
            }
        )
    return output


def paired_contrast(label: str, rows: dict[str, list[dict]]) -> dict:
    standard_name, lower_name = PAIRS[label]
    standard = {row["seed"]: row for row in rows[standard_name]}
    lower = {row["seed"]: row for row in rows[lower_name]}
    items = []
    for seed in SEEDS:
        left = standard[seed]
        right = lower[seed]
        items.append(
            {
                "seed": seed,
                "lower_minus_standard_exact_return": right["exact_mode"][
                    "expected_discounted_return"
                ]
                - left["exact_mode"]["expected_discounted_return"],
                "lower_minus_standard_exact_success": right["exact_mode"][
                    "success_rate"
                ]
                - left["exact_mode"]["success_rate"],
                "lower_minus_standard_normal_return": right["fixed_mode"][
                    "mean_discounted_return"
                ]
                - left["fixed_mode"]["mean_discounted_return"],
                "lower_minus_standard_normal_success": right["fixed_mode"][
                    "success_rate"
                ]
                - left["fixed_mode"]["success_rate"],
                "lower_minus_standard_risky_starts": (
                    right["risky_start_count"] - left["risky_start_count"]
                ),
                "lower_minus_standard_actor_kl": right[
                    "behavior_change_from_source"
                ]["mean_kl_source_to_current"]
                - left["behavior_change_from_source"][
                    "mean_kl_source_to_current"
                ],
                "lower_minus_standard_argmax_change_rate": right[
                    "behavior_change_from_source"
                ]["argmax_action_change_rate"]
                - left["behavior_change_from_source"][
                    "argmax_action_change_rate"
                ],
            }
        )
    fields = tuple(key for key in items[0] if key != "seed")
    return {
        "standard_condition": standard_name,
        "lower_condition": lower_name,
        "per_seed": items,
        "means": {
            field: mean(item[field] for item in items) for field in fields
        },
        "exact_return_signs": sign_counts(
            [
                item["lower_minus_standard_exact_return"]
                for item in items
            ]
        ),
    }


def capacity() -> dict:
    home = shutil.disk_usage("/home/ext_csv")
    raid = shutil.disk_usage("/raid/ext_csv")
    run_bytes = sum(
        path.stat().st_size for path in RUN_ROOT.rglob("*") if path.is_file()
    )
    return {
        "measured_unix": time.time(),
        "home": home._asdict(),
        "raid": raid._asdict(),
        "run_bytes_before_result": run_bytes,
        "reserve_preserved": raid.free
        > SPEC["checkpoint_storage"]["minimum_required_free_bytes"],
    }


def main() -> None:
    run_manifest, completion = validate_inputs()
    cells = {
        (condition, seed): validate_cell(condition, seed)
        for condition in CONDITIONS
        for seed in SEEDS
    }
    for seed in SEEDS:
        origins = [
            cells[(condition, seed)]["origin"]
            for condition in CONDITIONS
        ]
        for key in (
            "source_checkpoint",
            "source_train_state_sha256",
            "branch_train_state_sha256",
            "worker_initial_sha256",
            "source_optimizer_step",
            "source_global_update",
        ):
            if len({origin[key] for origin in origins}) != 1:
                raise AssertionError(
                    f"paired initialization differs: seed {seed} {key}"
                )
        zero = [
            cells[(condition, seed)]["evaluations"][0]
            for condition in CONDITIONS
        ]
        if len(
            {
                json.dumps(item["risk_profile"], sort_keys=True)
                for item in zero
            }
        ) != 1:
            raise AssertionError(
                f"update-zero fixed profiles differ: seed {seed}"
            )
        for item in zero:
            drift = item["behavior_change_from_source"]
            if (
                abs(drift["mean_kl_source_to_current"]) > 1e-7
                or drift["argmax_action_change_count"] != 0
            ):
                raise AssertionError("update-zero actor is not the source actor")
        for standard_name, lower_name in PAIRS.values():
            left = cells[(standard_name, seed)]["evaluations"][0]
            right = cells[(lower_name, seed)]["evaluations"][0]
            if (
                left["target_kernel"]["exact_mode"]
                != right["target_kernel"]["exact_mode"]
            ):
                raise AssertionError(
                    "same-p update-zero exact evaluations differ"
                )
    rows = {
        condition: [
            seed_row(condition, seed, cells[(condition, seed)])
            for seed in SEEDS
        ]
        for condition in CONDITIONS
    }
    aggregates = {
        condition: aggregate_rows(condition_rows)
        for condition, condition_rows in rows.items()
    }
    contrasts = {
        label: paired_contrast(label, rows) for label in PAIRS
    }
    total_slots = sum(
        aggregate["collection_totals"]["collector_slots"]
        for aggregate in aggregates.values()
    )
    total_scheduled = sum(
        aggregate["collection_totals"]["optimizer_applications"]
        + aggregate["collection_totals"]["empty_minibatches"]
        for aggregate in aggregates.values()
    )
    result = {
        "schema_version": "hackrl_pack_inventory_overwrite_lr_ppo_result_v1",
        "status": "complete_fixed_budget_no_automatic_extension",
        "generated_unix": time.time(),
        "execution_head": run_manifest["execution_head"],
        "aggregation_head": git("rev-parse", "HEAD"),
        "manifest_sha256": sha256(SPEC_PATH),
        "training_completion": completion,
        "gates": {
            "all_20_jobs_complete": True,
            "total_collector_slots_exact": total_slots
            == SPEC["budget"]["total_collector_slots"],
            "scheduled_optimizer_accounting_exact": total_scheduled
            == SPEC["budget"]["jobs"]
            * SPEC["optimization"][
                "scheduled_optimizer_applications_per_job"
            ],
            "paired_restored_state_and_worker_initial_hashes_identical": True,
            "update_zero_behavior_matches_source_on_frozen_anchor": True,
            "all_update_zero_fixed_profiles_identical_within_seed": True,
            "all_same_probability_update_zero_exact_results_identical": True,
            "all_retained_state_file_receipts_valid": True,
            "source_checkpoint_file_set_and_hashes_match_manifest": True,
            "execution_source_set_matches_manifest": True,
            "job_matrix_and_run_contract_match_manifest": True,
            "non_intervention_checkpoint_config_matches_source": True,
            "execution_sources_unchanged": True,
            "prerequisites_unchanged": True,
            "no_teacher_updates": True,
            "no_bc_updates": True,
            "no_automatic_extension": True,
        },
        "rows": rows,
        "aggregates": aggregates,
        "learning_curves": {
            condition: learning_curve(condition, cells)
            for condition in CONDITIONS
        },
        "paired_learning_rate_contrasts": contrasts,
        "primary_result": {
            "contrast": SPEC["evaluation"]["primary_contrast"],
            **contrasts["p05"],
        },
        "secondary_p30_result": contrasts["p30"],
        "mathematical_surrogate": {
            "domain": (
                "IEEE-754 binary64 finite sums over five learner seeds "
                "by 32 natural starts"
            ),
            "formula": SPEC["evaluation"]["exact_mode_formula"],
            "mode_paths_per_condition": len(SEEDS)
            * SPEC["evaluation"]["states"],
            "risk_count_unrestricted": True,
            "sample_evaluation_separate": True,
        },
        "non_claims": [
            "The quarter-rate intervention is not claimed optimal.",
            "A learning-rate effect is not an identification of the sole cause of prior degradation.",
            "Actor KL on source visitation is not a guarantee of task preservation.",
            "The p=0.05 versus p=0.30 contrast is not attributed to experience frequency alone.",
            "Five learner seeds do not establish statistical significance or broad generalization.",
        ],
        "capacity_end": capacity(),
    }
    if not all(result["gates"].values()):
        raise AssertionError(f"result gate failed: {result['gates']}")
    if not result["capacity_end"]["reserve_preserved"]:
        raise AssertionError("RAID safety reserve was not preserved")
    write(RUN_ROOT / "capacity_end.json", result["capacity_end"])
    write(RUN_ROOT / "result.json", result)
    print(json.dumps(result["primary_result"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
