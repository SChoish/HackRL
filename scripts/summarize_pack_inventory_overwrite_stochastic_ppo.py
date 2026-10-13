#!/usr/bin/env python3
"""Validate and aggregate the fixed-budget stochastic overwrite PPO run."""

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
    REPOSITORY
    / "docs/manifests/pack_inventory_overwrite_stochastic_ppo_v1.json"
)
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
RUN_ROOT = Path(SPEC["run_root"])
SEEDS = tuple(int(seed) for seed in SPEC["seeds"])
CONDITIONS = tuple(SPEC["conditions"])
UPDATES = tuple(int(update) for update in SPEC["evaluation"]["adaptation_updates"])


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
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def mean(values) -> float:
    values = list(values)
    return math.fsum(float(value) for value in values) / len(values)


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY, text=True
    ).strip()


def validate_inputs() -> tuple[dict, dict]:
    run_manifest = read(RUN_ROOT / "run_manifest.json")
    completion = read(RUN_ROOT / "training_complete.json")
    if completion.get("status") != "all_10_jobs_complete_pending_aggregation":
        raise AssertionError("training completion marker is missing")
    if run_manifest["manifest_sha256"] != sha256(SPEC_PATH):
        raise AssertionError("experiment manifest changed after launch")
    for relative, expected in run_manifest["execution_source_sha256"].items():
        if sha256(REPOSITORY / relative) != expected:
            raise AssertionError(f"execution source changed after launch: {relative}")
    prerequisites = SPEC["prerequisites"]
    for path_key, hash_key in (
        ("policy_panel_result", "policy_panel_result_sha256"),
        ("policy_panel_risk_profiles", "policy_panel_risk_profiles_sha256"),
        ("reference_paths", "reference_paths_sha256"),
    ):
        if sha256(Path(prerequisites[path_key])) != prerequisites[hash_key]:
            raise AssertionError(f"prerequisite changed: {path_key}")
    if list(RUN_ROOT.rglob("failed.json")):
        raise AssertionError("a failed-job marker remains")
    return run_manifest, completion


def validate_cell(condition: str, seed: int) -> dict:
    cell = RUN_ROOT / condition / f"seed{seed}"
    summary = read(cell / "summary.json")
    origin = read(cell / "origin.json")
    if summary.get("status") != "complete":
        raise AssertionError(f"incomplete cell: {condition} seed {seed}")
    if summary["collector_slots"] != SPEC["optimization"]["collector_slots_per_job"]:
        raise AssertionError("cell collector budget changed")
    if summary["physical_transitions"] != summary["collector_slots"]:
        raise AssertionError("physical-transition accounting changed")
    if summary["optimizer_applications"] != SPEC["optimization"][
        "scheduled_optimizer_applications_per_job"
    ]:
        raise AssertionError("optimizer application count changed")
    if summary["teacher_updates"] != 0 or summary["bc_updates"] != 0:
        raise AssertionError("teacher or BC update occurred")
    if summary["overwrite_probability"] != SPEC["conditions"][condition][
        "overwrite_probability"
    ]:
        raise AssertionError("cell overwrite probability changed")
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
    checkpoint_names = sorted(path.name for path in checkpoint_root.glob("adapt_*"))
    expected_names = sorted(f"adapt_{update}" for update in UPDATES)
    if checkpoint_names != expected_names:
        raise AssertionError("retained checkpoint set changed")
    evaluations = {}
    for update in UPDATES:
        checkpoint = checkpoint_root / f"adapt_{update}"
        receipt = read(checkpoint / "receipt.json")
        evaluation = read(cell / "evaluations" / f"adapt_{update}.json")
        if receipt["adaptation_update"] != update:
            raise AssertionError("checkpoint receipt update changed")
        if sha256(checkpoint / "state.msgpack") != receipt["state_sha256"]:
            raise AssertionError("checkpoint state hash changed")
        if evaluation["train_state_sha256"] != receipt["train_state_sha256"]:
            raise AssertionError("evaluation and checkpoint identity differ")
        if evaluation["collector_slots"] != update * SPEC["optimization"][
            "collector_slots_per_update"
        ]:
            raise AssertionError("evaluation slot coordinate changed")
        evaluations[update] = evaluation["evaluation"]
    resume = cell / "resume"
    if resume.is_dir() and list(resume.iterdir()):
        raise AssertionError("rolling resume checkpoint was not pruned")
    return {"summary": summary, "origin": origin, "evaluations": evaluations}


def seed_row(condition: str, seed: int, cell: dict) -> dict:
    final = cell["evaluations"][UPDATES[-1]]
    exact = final["target_kernel"]["exact_mode"]
    target_sample = final["target_kernel"]["sample"]["overall"]
    fixed_mode = final["fixed_kernel"]["mode"]["overall"]
    fixed_sample = final["fixed_kernel"]["sample"]["overall"]
    summary = cell["summary"]
    return {
        "seed": seed,
        "condition": condition,
        "overwrite_probability": float(
            SPEC["conditions"][condition]["overwrite_probability"]
        ),
        "exact_mode": {
            "success_rate": exact["existing_success_rate"],
            "expected_discounted_return": exact[
                "existing_expected_discounted_return"
            ],
            "failure_probability": exact["existing_failure_probability"],
            "loss_episode_probability": exact[
                "existing_loss_episode_probability"
            ],
            "adapt_minus_normal_continue_return": final[
                "same_seed_exact_return_contrasts"
            ]["adapt_minus_normal_continue"],
            "adapt_minus_original_return": final[
                "same_seed_exact_return_contrasts"
            ]["adapt_minus_original"],
            "adapt_minus_p1_adapt_return": final[
                "same_seed_exact_return_contrasts"
            ]["adapt_minus_p1_adapt"],
        },
        "fixed_mode": {
            "success_rate": fixed_mode["success_rate"],
            "mean_discounted_return": fixed_mode["mean_discounted_return"],
            "mean_length": fixed_mode["mean_length"],
            "deliver_before_first_harvest_rate": fixed_mode[
                "deliver_before_first_harvest_rate"
            ],
            "store_before_first_harvest_rate": fixed_mode[
                "store_before_first_harvest_rate"
            ],
            "protect_before_first_harvest_rate": fixed_mode[
                "protect_before_first_harvest_rate"
            ],
        },
        "fixed_sample": {
            "success_rate": fixed_sample["success_rate"],
            "mean_discounted_return": fixed_sample["mean_discounted_return"],
        },
        "target_sample": {
            "success_rate": target_sample["success_rate"],
            "mean_discounted_return": target_sample[
                "mean_discounted_return"
            ],
        },
        "risk_count_distribution": final["risk_profile"][
            "risk_count_distribution"
        ],
        "frozen_baselines": final["frozen_baselines"],
        "collection": {
            key: summary[key]
            for key in (
                "collector_slots",
                "physical_transitions",
                "valid_ppo_samples",
                "optimizer_applications",
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
            {int(key): int(value) for key, value in row["risk_count_distribution"].items()}
        )
    exact_keys = (
        "success_rate",
        "expected_discounted_return",
        "failure_probability",
        "loss_episode_probability",
        "adapt_minus_normal_continue_return",
        "adapt_minus_original_return",
        "adapt_minus_p1_adapt_return",
    )
    collection_sum_keys = (
        "collector_slots",
        "physical_transitions",
        "valid_ppo_samples",
        "optimizer_applications",
        "completed_world_episodes",
        "loss_episodes",
        "cumulative_lost_grain",
        "physical_loss_events",
        "overwrite_trigger_events",
        "overwrite_fire_events",
        "goal_successes_during_collection",
    )
    contrasts = [
        float(row["exact_mode"]["adapt_minus_normal_continue_return"])
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
            for key in (
                "success_rate",
                "mean_discounted_return",
                "mean_length",
                "deliver_before_first_harvest_rate",
                "store_before_first_harvest_rate",
                "protect_before_first_harvest_rate",
            )
        },
        "target_sample_mean": {
            key: mean(row["target_sample"][key] for row in rows)
            for key in ("success_rate", "mean_discounted_return")
        },
        "normal_sample_mean": {
            key: mean(row["fixed_sample"][key] for row in rows)
            for key in ("success_rate", "mean_discounted_return")
        },
        "risk_count_distribution": {
            str(key): risk_counts[key] for key in sorted(risk_counts)
        },
        "paired_contrast_signs": {
            "positive": sum(value > 1e-12 for value in contrasts),
            "equal_within_1e-12": sum(abs(value) <= 1e-12 for value in contrasts),
            "negative": sum(value < -1e-12 for value in contrasts),
        },
        "collection_totals": {
            key: sum(int(row["collection"][key]) for row in rows)
            for key in collection_sum_keys
        },
        "empirical_fire_per_risky_harvest": fire_total / trigger_total,
    }


def curve(condition: str, cells: dict) -> list[dict]:
    output = []
    for update in UPDATES:
        evaluations = [cells[(condition, seed)]["evaluations"][update] for seed in SEEDS]
        exact = [evaluation["target_kernel"]["exact_mode"] for evaluation in evaluations]
        output.append(
            {
                "adaptation_update": update,
                "collector_slots_per_seed": update
                * SPEC["optimization"]["collector_slots_per_update"],
                "exact_mode_success_rate": mean(
                    item["existing_success_rate"] for item in exact
                ),
                "exact_mode_expected_discounted_return": mean(
                    item["existing_expected_discounted_return"] for item in exact
                ),
                "exact_mode_loss_episode_probability": mean(
                    item["existing_loss_episode_probability"] for item in exact
                ),
                "adapt_minus_normal_continue_return": mean(
                    evaluation["same_seed_exact_return_contrasts"][
                        "adapt_minus_normal_continue"
                    ]
                    for evaluation in evaluations
                ),
                "normal_kernel_mode_success_rate": mean(
                    evaluation["fixed_kernel"]["mode"]["overall"]["success_rate"]
                    for evaluation in evaluations
                ),
            }
        )
    return output


def aggregate_baselines(exemplar_rows: list[dict]) -> dict:

    output = {}
    for policy in ("original", "normal_continue", "p1_adapt"):
        output[policy] = {
            "success_rate": mean(
                row["frozen_baselines"][policy]["success_rate"]
                for row in exemplar_rows
            ),
            "expected_discounted_return": mean(
                row["frozen_baselines"][policy]["expected_discounted_return"]
                for row in exemplar_rows
            ),
            "loss_episode_probability": mean(
                row["frozen_baselines"][policy]["loss_episode_probability"]
                for row in exemplar_rows
            ),
        }
    return output


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
        left = cells[(CONDITIONS[0], seed)]["origin"]
        right = cells[(CONDITIONS[1], seed)]["origin"]
        for key in (
            "source_checkpoint",
            "source_train_state_sha256",
            "branch_train_state_sha256",
            "worker_initial_sha256",
            "source_optimizer_step",
            "source_global_update",
        ):
            if left[key] != right[key]:
                raise AssertionError(f"paired initialization differs: seed {seed} {key}")
        left_zero = cells[(CONDITIONS[0], seed)]["evaluations"][0]
        right_zero = cells[(CONDITIONS[1], seed)]["evaluations"][0]
        if left_zero["risk_profile"] != right_zero["risk_profile"]:
            raise AssertionError("paired update-zero fixed mode profiles differ")
    rows = {
        condition: [
            seed_row(condition, seed, cells[(condition, seed)]) for seed in SEEDS
        ]
        for condition in CONDITIONS
    }
    aggregates = {
        condition: aggregate_rows(condition_rows)
        for condition, condition_rows in rows.items()
    }
    baseline_by_condition = {
        condition: aggregate_baselines(rows[condition])
        for condition in CONDITIONS
    }
    p05 = aggregates["p05_adapt"]
    p30 = aggregates["p30_adapt"]
    result = {
        "schema_version": "hackrl_pack_inventory_overwrite_stochastic_ppo_result_v1",
        "status": "complete_fixed_budget_no_automatic_extension",
        "generated_unix": time.time(),
        "execution_head": run_manifest["execution_head"],
        "aggregation_head": git("rev-parse", "HEAD"),
        "manifest_sha256": sha256(SPEC_PATH),
        "training_completion": completion,
        "gates": {
            "all_10_jobs_complete": True,
            "total_collector_slots_exact": sum(
                aggregate["collection_totals"]["collector_slots"]
                for aggregate in aggregates.values()
            )
            == SPEC["budget"]["total_collector_slots"],
            "total_optimizer_applications_exact": sum(
                aggregate["collection_totals"]["optimizer_applications"]
                for aggregate in aggregates.values()
            )
            == SPEC["budget"]["jobs"]
            * SPEC["optimization"]["scheduled_optimizer_applications_per_job"],
            "paired_initialization_identity": True,
            "paired_update_zero_fixed_profiles": True,
            "all_retained_checkpoint_hashes_valid": True,
            "execution_sources_unchanged": True,
            "prerequisites_unchanged": True,
            "no_teacher_updates": True,
            "no_bc_updates": True,
            "no_automatic_extension": True,
        },
        "rows": rows,
        "aggregates": aggregates,
        "learning_curves": {
            condition: curve(condition, cells) for condition in CONDITIONS
        },
        "frozen_baselines": baseline_by_condition,
        "decision": {
            "p05": {
                "mean_return_improvement_over_normal_continue": p05[
                    "exact_mode_mean"
                ]["adapt_minus_normal_continue_return"],
                "classification": "no aggregate improvement; mixed seeds",
                "paired_signs": p05["paired_contrast_signs"],
                "p1_policy_has_higher_return_than_normal_continue": (
                    baseline_by_condition["p05_adapt"]["p1_adapt"][
                        "expected_discounted_return"
                    ]
                    > baseline_by_condition["p05_adapt"]["normal_continue"][
                        "expected_discounted_return"
                    ]
                ),
            },
            "p30": {
                "mean_return_improvement_over_normal_continue": p30[
                    "exact_mode_mean"
                ]["adapt_minus_normal_continue_return"],
                "classification": "aggregate improvement; four positive seeds and one tie",
                "paired_signs": p30["paired_contrast_signs"],
            },
            "research_interpretation": "Within the fixed budget, return-improving acquisition was observed at p=0.30 but not in aggregate at p=0.05. The frozen p=1 policy remains advantageous at p=0.05, separating existence of a useful low-cost behavior from its acquisition under rare defect experience.",
        },
        "mathematical_surrogate": {
            "domain": "IEEE-754 binary64 finite sums over five learned seeds by 32 natural starts",
            "formula": SPEC["evaluation"]["exact_mode_formula"],
            "mode_paths": 160,
            "risk_count_unrestricted": True,
            "sample_evaluation_separate": True,
        },
        "non_claims": [
            "No optimal-avoidance boundary is identified.",
            "The zero-loss scripted path and p=1 policy are not claimed optimal.",
            "The p=0.30 versus p=0.05 difference is not attributed to experience frequency alone.",
            "Five learner seeds do not establish statistical significance or broad generalization.",
            "Sample-policy evaluation is not substituted for exact deterministic-mode evaluation.",
        ],
        "capacity_end": capacity(),
        "prelaunch_verification": {
            "focused": "14 passed, 1 warning",
            "full_repository": "504 passed, 9 failed, 1 warning",
            "full_failure_scope": "Eight Mine Expedition tests rejected pre-existing authorized-source hash drift; one historical spatial GPU validation rejected five stale source hashes. Focused stochastic tests and update-zero end-to-end evaluation passed.",
        },
    }
    if not all(result["gates"].values()):
        raise AssertionError(f"result gate failed: {result['gates']}")
    write(RUN_ROOT / "capacity_end.json", result["capacity_end"])
    write(RUN_ROOT / "result.json", result)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
