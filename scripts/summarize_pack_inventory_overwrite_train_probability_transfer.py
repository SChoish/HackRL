#!/usr/bin/env python3
"""Validate and aggregate the overwrite training-probability transfer study."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import summarize_pack_inventory_overwrite_lr_ppo as lr_summary
from hackrl.pack_inventory_overwrite_rare import evaluate_regime_point


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC_PATH = (
    REPOSITORY
    / "docs/manifests/pack_inventory_overwrite_train_probability_transfer_v1.json"
)
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
RUN_ROOT = Path(SPEC["run_root"])
SEEDS = tuple(int(seed) for seed in SPEC["seeds"])
NEW_CONDITIONS = tuple(SPEC["conditions"])
REUSED_CONDITIONS = tuple(SPEC["reused_conditions"])
STANDARD_CONDITIONS = ("p00_lr1", "p05_lr1", "p30_lr1", "p100_lr1")
AUXILIARY_CONDITIONS = ("p05_lr025", "p30_lr025")
ALL_CONDITIONS = STANDARD_CONDITIONS + AUXILIARY_CONDITIONS
UPDATES = tuple(
    int(update) for update in SPEC["evaluation"]["adaptation_updates"]
)
EVALUATION_PROBABILITIES = {
    name: float(value)
    for name, value in SPEC["evaluation_probabilities"].items()
}
PRIOR_ROOT = Path(SPEC["prerequisites"]["prior_raw_result"]).parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def configure_lr_summary() -> None:
    lr_summary.SPEC_PATH = SPEC_PATH
    lr_summary.SPEC = SPEC
    lr_summary.RUN_ROOT = RUN_ROOT
    lr_summary.SEEDS = SEEDS
    lr_summary.CONDITIONS = NEW_CONDITIONS
    lr_summary.UPDATES = UPDATES


def validate_inputs() -> tuple[dict, dict, dict, dict]:
    run_manifest = read(RUN_ROOT / "run_manifest.json")
    completion = read(RUN_ROOT / "training_complete.json")
    if completion.get("status") != "all_10_jobs_complete_pending_aggregation":
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
        for condition in NEW_CONDITIONS
    ]
    if run_manifest.get("jobs") != expected_jobs:
        raise AssertionError("run-manifest job matrix changed")
    if (
        run_manifest.get("manifest_id") != SPEC["manifest_id"]
        or run_manifest.get("planned_collector_slots")
        != SPEC["budget"]["total_collector_slots"]
        or run_manifest.get("automatic_extension") is not False
        or run_manifest.get("new_training_job_count") != 10
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
            / SPEC["source_checkpoint"]
            / name
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
        ("prior_run_manifest", "prior_run_manifest_sha256"),
        ("reference_paths", "reference_paths_sha256"),
        ("exact_formula_basis", "exact_formula_basis_sha256"),
    ):
        if sha256(prerequisite_path(prerequisites[path_key])) != prerequisites[
            hash_key
        ]:
            raise AssertionError(f"prerequisite changed: {path_key}")
    if (
        run_manifest.get("reused_final_evaluation_sha256")
        != prerequisites["reused_final_evaluation_sha256"]
    ):
        raise AssertionError("run manifest reused-evaluation hashes changed")
    if (
        run_manifest.get("reused_source_origin_sha256")
        != prerequisites["reused_source_origin_sha256"]
        or run_manifest.get("exact_formula_basis_sha256")
        != prerequisites["exact_formula_basis_sha256"]
    ):
        raise AssertionError("run manifest prerequisite hashes changed")
    for path, expected in prerequisites["reused_source_origin_sha256"].items():
        if sha256(Path(path)) != expected:
            raise AssertionError(f"reused source origin changed: {path}")
    for path, expected in prerequisites[
        "reused_final_evaluation_sha256"
    ].items():
        if sha256(Path(path)) != expected:
            raise AssertionError(f"reused final evaluation changed: {path}")
    if list(RUN_ROOT.rglob("failed.json")):
        raise AssertionError("a failed-job marker remains")
    prior_raw = read(prerequisite_path(prerequisites["prior_raw_result"]))
    if prior_raw.get("status") != "complete_fixed_budget_no_automatic_extension":
        raise AssertionError("prior learning-rate result is incomplete")
    if set(prior_raw.get("rows", {})) != set(REUSED_CONDITIONS):
        raise AssertionError("prior reusable condition set changed")
    prior_run = read(prerequisite_path(prerequisites["prior_run_manifest"]))
    if prior_run["source_checkpoint_sha256"] != run_manifest["source_checkpoint_sha256"]:
        raise AssertionError("prior and new source checkpoints differ")
    required_shared = {
        "scripts/run_pack_inventory_overwrite_lr_ppo.py",
        "scripts/summarize_pack_inventory_overwrite_lr_ppo.py",
        "scripts/run_pack_inventory_overwrite_ppo.py",
        "src/hackrl/pack_restore.py",
        "src/hackrl/pack_restore_gc.py",
        "src/hackrl/pack_inventory_overwrite_rare.py",
        "src/hackrl/dual_leo.py",
    }
    shared_sources = set(prior_run["execution_source_sha256"]) & set(
        run_manifest["execution_source_sha256"]
    )
    if not required_shared.issubset(shared_sources):
        raise AssertionError("shared execution-source set is incomplete")
    for path in required_shared:
        if prior_run["execution_source_sha256"][path] != run_manifest[
            "execution_source_sha256"
        ][path]:
            raise AssertionError(f"shared execution source differs: {path}")
    return run_manifest, completion, prior_raw, prior_run


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
        != "hackrl_pack_inventory_overwrite_train_probability_transfer_cell_v1"
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
            summary["branch_learning_rate"], expected_lr, rel_tol=0.0, abs_tol=0.0
        )
    ):
        raise AssertionError("cell intervention changed")
    for key, expected in (
        ("overwrite_probability", expected_condition["overwrite_probability"]),
        ("learning_rate_multiplier", expected_condition["learning_rate_multiplier"]),
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
    update_records = []
    for update, path in enumerate(update_files, start=1):
        record = read(path)
        if int(record["adaptation_update"]) != update:
            raise AssertionError("update ordering changed")
        update_records.append(record)
    final_cumulative = update_records[-1]["cumulative"]
    cumulative_fields = {
        "collector_slots": "collector_slots",
        "physical_transitions": "physical_transitions",
        "valid_ppo_samples": "valid_ppo_samples",
        "optimizer_applications": "optimizer_applications",
        "empty_minibatches": "empty_minibatches",
        "completed_world_episodes": "completed_world_episodes",
        "loss_episodes": "loss_episodes",
        "cumulative_lost_grain": "cumulative_lost_grain",
        "physical_loss_events": "physical_loss_events",
        "overwrite_trigger_events": "overwrite_trigger_events",
        "overwrite_fire_events": "overwrite_fire_events",
        "goal_successes_during_collection": "goal_successes",
    }
    for summary_key, cumulative_key in cumulative_fields.items():
        if int(summary[summary_key]) != int(final_cumulative[cumulative_key]):
            raise AssertionError(
                f"summary/update cumulative mismatch: {summary_key}"
            )
    _assert_close(
        summary["loss_episode_rate"],
        final_cumulative["loss_episode_rate"],
        "summary/update loss episode rate",
    )
    checkpoint_root = cell / "checkpoints"
    checkpoint_names = sorted(path.name for path in checkpoint_root.glob("adapt_*"))
    expected_names = sorted(f"adapt_{update}" for update in UPDATES)
    if checkpoint_names != expected_names:
        raise AssertionError("retained checkpoint set changed")
    evaluations = {}
    for update in UPDATES:
        checkpoint = checkpoint_root / f"adapt_{update}"
        receipt = read(checkpoint / "receipt.json")
        config = read(checkpoint / "config.json")
        evaluation_doc = read(cell / "evaluations" / f"adapt_{update}.json")
        if receipt["adaptation_update"] != update:
            raise AssertionError("checkpoint receipt update changed")
        if sha256(checkpoint / "state.msgpack") != receipt["state_sha256"]:
            raise AssertionError("checkpoint state hash changed")
        if evaluation_doc["train_state_sha256"] != receipt["train_state_sha256"]:
            raise AssertionError("evaluation and checkpoint identity differ")
        if evaluation_doc["collector_slots"] != update * SPEC["optimization"][
            "collector_slots_per_update"
        ]:
            raise AssertionError("evaluation slot coordinate changed")
        if (
            config["variant"] != expected_condition["kernel"]
            or config["inventory_overwrite_probability"]
            != expected_condition["overwrite_probability"]
            or not math.isclose(
                config["learning_rate"], expected_lr, rel_tol=0.0, abs_tol=0.0
            )
        ):
            raise AssertionError("checkpoint configuration changed")
        if set(config) != set(source_config):
            raise AssertionError("checkpoint configuration key set changed")
        if (
            config["num_updates"] != SPEC["optimization"]["updates"]
            or config["checkpoint_updates"] != []
            or config["num_envs"] != SPEC["optimization"]["num_envs"]
            or config["num_steps"] != SPEC["optimization"]["num_steps"]
            or config["minibatch_size"] != SPEC["optimization"]["minibatch_size"]
            or config["update_epochs"] != SPEC["optimization"]["update_epochs"]
        ):
            raise AssertionError("checkpoint run-length configuration changed")
        intervention_fields = {
            "variant",
            "inventory_overwrite_probability",
            "learning_rate",
            "num_updates",
            "checkpoint_updates",
        }
        for key, source_value in source_config.items():
            if key not in intervention_fields and config.get(key) != source_value:
                raise AssertionError(f"non-intervention config changed: {key}")
        if (
            evaluation_doc.get("condition") != condition
            or evaluation_doc.get("seed") != seed
            or evaluation_doc.get("adaptation_update") != update
            or evaluation_doc.get("schema_version")
            != "hackrl_pack_inventory_overwrite_train_probability_transfer_eval_v1"
        ):
            raise AssertionError("evaluation identity changed")
        if update == 0 and (
            receipt["train_state_sha256"] != origin["source_train_state_sha256"]
            or receipt["train_state_sha256"]
            != origin["branch_train_state_sha256"]
        ):
            raise AssertionError("update-zero state does not match source state")
        validate_evaluation_payload(evaluation_doc["evaluation"], condition)
        evaluations[update] = evaluation_doc["evaluation"]
    resume = cell / "resume"
    if resume.is_dir() and list(resume.iterdir()):
        raise AssertionError("rolling resume checkpoint was not pruned")
    return {"summary": summary, "origin": origin, "evaluations": evaluations}


def reused_evaluation(condition: str, seed: int) -> dict:
    path = PRIOR_ROOT / condition / f"seed{seed}" / "evaluations/adapt_32.json"
    expected = SPEC["prerequisites"]["reused_final_evaluation_sha256"][str(path)]
    if sha256(path) != expected:
        raise AssertionError(f"reused evaluation changed: {path}")
    document = read(path)
    if (
        document.get("schema_version") != "hackrl_pack_inventory_overwrite_lr_eval_v1"
        or document.get("condition") != condition
        or document.get("seed") != seed
        or document.get("adaptation_update") != 32
    ):
        raise AssertionError("reused evaluation identity changed")
    evaluation = document["evaluation"]
    validate_evaluation_payload(evaluation, condition)
    return evaluation


def _assert_close(left: float, right: float, label: str) -> None:
    if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15):
        raise AssertionError(f"{label}: {left} != {right}")


def _assert_finite(value, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"non-finite evaluation value: {label}")


def validate_evaluation_payload(evaluation: dict, condition: str) -> None:
    if condition in SPEC["conditions"]:
        condition_spec = SPEC["conditions"][condition]
        expected_probability = condition_spec["overwrite_probability"]
        expected_lr_multiplier = condition_spec["learning_rate_multiplier"]
    else:
        condition_spec = SPEC["reused_conditions"][condition]
        expected_probability = condition_spec["training_probability"]
        expected_lr_multiplier = condition_spec["learning_rate_multiplier"]
    if (
        evaluation.get("overwrite_probability")
        != expected_probability
        or evaluation.get("learning_rate_multiplier")
        != expected_lr_multiplier
    ):
        raise AssertionError("evaluation intervention identity changed")
    _assert_finite(evaluation, f"evaluation.{condition}")
    required_episodes = {
        ("fixed_kernel", "mode"): SPEC["evaluation"]["states"]
        * SPEC["evaluation"]["fixed_mode_repeats_per_state"],
        ("fixed_kernel", "sample"): SPEC["evaluation"]["states"]
        * SPEC["evaluation"]["normal_sample_repeats_per_state"],
        ("target_kernel", "sample"): SPEC["evaluation"]["states"]
        * SPEC["evaluation"]["stochastic_sample_repeats_per_state"],
    }
    for path, expected_count in required_episodes.items():
        result = evaluation[path[0]][path[1]]["overall"]
        if int(result["episodes"]) != int(expected_count):
            raise AssertionError(f"evaluation episode count changed: {path}")
        if not 0.0 <= float(result["success_rate"]) <= 1.0:
            raise AssertionError(f"evaluation probability out of range: {path}")
    profile = evaluation["risk_profile"]
    episodes = profile["episodes"]
    if (
        int(profile["episode_count"]) != SPEC["evaluation"]["states"]
        or len(episodes) != SPEC["evaluation"]["states"]
        or sum(int(value) for value in profile["risk_count_distribution"].values())
        != SPEC["evaluation"]["states"]
        or len({int(row["state_index"]) for row in episodes})
        != SPEC["evaluation"]["states"]
    ):
        raise AssertionError("risk-profile panel changed")
    horizon = int(SPEC["evaluation"]["horizon_ticks"])
    for row in episodes:
        if (
            int(row["risk_count"]) < 0
            or float(row["fixed_success"]) not in (0.0, 1.0)
            or not 1 <= int(row["fixed_length"]) <= horizon
        ):
            raise AssertionError("invalid risk-profile episode")
    recomputed = evaluate_regime_point(
        profile, float(expected_probability)
    )
    recorded = evaluation["target_kernel"]["exact_mode"]
    for key in (
        "existing_success_rate",
        "existing_expected_discounted_return",
        "existing_failure_probability",
        "existing_loss_episode_probability",
    ):
        _assert_close(recorded[key], recomputed[key], f"exact payload {key}")


def validate_reused_row(row: dict, evaluation: dict) -> None:
    exact = evaluation["target_kernel"]["exact_mode"]
    for row_key, exact_key in (
        ("success_rate", "existing_success_rate"),
        ("expected_discounted_return", "existing_expected_discounted_return"),
        ("failure_probability", "existing_failure_probability"),
        ("loss_episode_probability", "existing_loss_episode_probability"),
    ):
        _assert_close(row["exact_mode"][row_key], exact[exact_key], row_key)
    if row["risk_count_distribution"] != evaluation["risk_profile"][
        "risk_count_distribution"
    ]:
        raise AssertionError("reused risk distribution differs from prior row")
    fixed = evaluation["fixed_kernel"]["mode"]["overall"]
    for key, value in row["fixed_mode"].items():
        _assert_close(value, fixed[key], f"fixed mode {key}")
    behavior = evaluation["behavior_change_from_source"]
    for key in (
        "mean_kl_source_to_current",
        "argmax_action_change_count",
        "argmax_action_change_rate",
    ):
        _assert_close(
            row["behavior_change_from_source"][key], behavior[key], f"behavior {key}"
        )


def policy_record(
    condition: str,
    seed: int,
    evaluation: dict,
    prior_or_new_row: dict,
    provenance: str,
) -> dict:
    profile = evaluation["risk_profile"]
    cross = {
        label: evaluate_regime_point(profile, probability)
        for label, probability in EVALUATION_PROBABILITIES.items()
    }
    fixed_mode = evaluation["fixed_kernel"]["mode"]["overall"]
    _assert_close(
        cross["p00"]["existing_expected_discounted_return"],
        fixed_mode["mean_discounted_return"],
        f"p=0 exact return {condition} seed {seed}",
    )
    _assert_close(
        cross["p00"]["existing_success_rate"],
        fixed_mode["success_rate"],
        f"p=0 exact success {condition} seed {seed}",
    )
    training_probability = (
        SPEC["conditions"][condition]["overwrite_probability"]
        if condition in SPEC["conditions"]
        else SPEC["reused_conditions"][condition]["training_probability"]
    )
    learning_rate_multiplier = (
        SPEC["conditions"][condition]["learning_rate_multiplier"]
        if condition in SPEC["conditions"]
        else SPEC["reused_conditions"][condition]["learning_rate_multiplier"]
    )
    return {
        "condition": condition,
        "seed": seed,
        "provenance": provenance,
        "training_probability": training_probability,
        "learning_rate_multiplier": learning_rate_multiplier,
        "normal_kernel_mode": {
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
        "cross_probability_exact": cross,
        "matching_probability_sample": {
            key: evaluation["target_kernel"]["sample"]["overall"][key]
            for key in ("success_rate", "mean_discounted_return")
        },
        "risk_count_distribution": profile["risk_count_distribution"],
        "risky_start_count": sum(
            int(value)
            for key, value in profile["risk_count_distribution"].items()
            if int(key) > 0
        ),
        "behavior_change_from_source": evaluation[
            "behavior_change_from_source"
        ],
        "collection": prior_or_new_row["collection"],
    }


def aggregate_policy_records(records: list[dict]) -> dict:
    risk_counts = Counter()
    for record in records:
        risk_counts.update(
            {
                int(key): int(value)
                for key, value in record["risk_count_distribution"].items()
            }
        )
    exact = {}
    for label in EVALUATION_PROBABILITIES:
        exact[label] = {
            "success_rate": mean(
                row["cross_probability_exact"][label]["existing_success_rate"]
                for row in records
            ),
            "expected_discounted_return": mean(
                row["cross_probability_exact"][label][
                    "existing_expected_discounted_return"
                ]
                for row in records
            ),
            "overwrite_incident_probability": mean(
                row["cross_probability_exact"][label][
                    "existing_loss_episode_probability"
                ]
                for row in records
            ),
        }
    normal = {
        key: mean(row["normal_kernel_mode"][key] for row in records)
        for key in records[0]["normal_kernel_mode"]
    }
    for label in EVALUATION_PROBABILITIES:
        exact[label]["defect_return_cost_from_p00"] = (
            exact["p00"]["expected_discounted_return"]
            - exact[label]["expected_discounted_return"]
        )
    collection_sum_keys = (
        "collector_slots",
        "physical_transitions",
        "valid_ppo_samples",
        "optimizer_applications",
        "empty_minibatches",
        "completed_world_episodes",
        "loss_episodes",
        "cumulative_lost_grain",
        "physical_loss_events",
        "overwrite_trigger_events",
        "overwrite_fire_events",
        "goal_successes_during_collection",
    )
    return {
        "seeds": len(records),
        "training_probability": records[0]["training_probability"],
        "learning_rate_multiplier": records[0]["learning_rate_multiplier"],
        "normal_kernel_mode_mean": normal,
        "cross_probability_exact_mean": exact,
        "matching_probability_sample_mean": {
            key: mean(row["matching_probability_sample"][key] for row in records)
            for key in records[0]["matching_probability_sample"]
        },
        "risk_count_distribution": {
            str(key): risk_counts[key] for key in sorted(risk_counts)
        },
        "risky_start_count": sum(row["risky_start_count"] for row in records),
        "behavior_change_mean": {
            key: mean(row["behavior_change_from_source"][key] for row in records)
            for key in (
                "mean_kl_source_to_current",
                "argmax_action_change_count",
                "argmax_action_change_rate",
            )
        },
        "collection_totals": {
            key: sum(int(row["collection"][key]) for row in records)
            for key in collection_sum_keys
        },
        "mean_collection_loss_episode_rate": mean(
            row["collection"]["loss_episode_rate"] for row in records
        ),
    }


def paired_difference(
    left: str,
    right: str,
    policies: dict[tuple[str, int], dict],
    evaluation_label: str,
) -> dict:
    items = []
    for seed in SEEDS:
        left_row = policies[(left, seed)]
        right_row = policies[(right, seed)]
        value = (
            right_row["cross_probability_exact"][evaluation_label][
                "existing_expected_discounted_return"
            ]
            - left_row["cross_probability_exact"][evaluation_label][
                "existing_expected_discounted_return"
            ]
        )
        items.append({"seed": seed, "right_minus_left": value})
    tolerance = float(SPEC["statistics"]["sign_tolerance"])
    return {
        "left": left,
        "right": right,
        "evaluation_probability": EVALUATION_PROBABILITIES[evaluation_label],
        "per_seed": items,
        "mean_right_minus_left": mean(item["right_minus_left"] for item in items),
        "signs": {
            "positive": sum(item["right_minus_left"] > tolerance for item in items),
            "equal_within_tolerance": sum(
                abs(item["right_minus_left"]) <= tolerance for item in items
            ),
            "negative": sum(item["right_minus_left"] < -tolerance for item in items),
        },
    }


def source_baseline(cells: dict) -> dict:
    records = []
    for seed in SEEDS:
        evaluation = cells[(NEW_CONDITIONS[0], seed)]["evaluations"][0]
        profile = evaluation["risk_profile"]
        records.append(
            {
                "seed": seed,
                "cross_probability_exact": {
                    label: evaluate_regime_point(profile, probability)
                    for label, probability in EVALUATION_PROBABILITIES.items()
                },
                "normal_return": evaluation["fixed_kernel"]["mode"]["overall"][
                    "mean_discounted_return"
                ],
                "normal_success": evaluation["fixed_kernel"]["mode"]["overall"][
                    "success_rate"
                ],
            }
        )
    return {
        "per_seed": records,
        "means": {
            "normal_return": mean(row["normal_return"] for row in records),
            "normal_success": mean(row["normal_success"] for row in records),
            "cross_probability_exact": {
                label: {
                    "success_rate": mean(
                        row["cross_probability_exact"][label][
                            "existing_success_rate"
                        ]
                        for row in records
                    ),
                    "expected_discounted_return": mean(
                        row["cross_probability_exact"][label][
                            "existing_expected_discounted_return"
                        ]
                        for row in records
                    ),
                }
                for label in EVALUATION_PROBABILITIES
            },
        },
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
    configure_lr_summary()
    run_manifest, completion, prior_raw, prior_run = validate_inputs()
    cells = {
        (condition, seed): validate_cell(condition, seed)
        for condition in NEW_CONDITIONS
        for seed in SEEDS
    }
    for seed in SEEDS:
        origins = [cells[(condition, seed)]["origin"] for condition in NEW_CONDITIONS]
        for key in (
            "source_checkpoint",
            "source_train_state_sha256",
            "branch_train_state_sha256",
            "worker_initial_sha256",
            "source_optimizer_step",
            "source_global_update",
        ):
            if len({origin[key] for origin in origins}) != 1:
                raise AssertionError(f"paired initialization differs: seed {seed} {key}")
        reused_origin_path = PRIOR_ROOT / "p05_lr1" / f"seed{seed}" / "origin.json"
        reused_origin = read(reused_origin_path)
        expected_origin_hash = SPEC["prerequisites"]["reused_source_origin_sha256"][
            str(reused_origin_path)
        ]
        if sha256(reused_origin_path) != expected_origin_hash:
            raise AssertionError(f"reused source origin changed: seed {seed}")
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
            if reused_origin[key] != origins[0][key]:
                raise AssertionError(
                    f"prior/new initialization differs: seed {seed} {key}"
                )
        zero = [cells[(condition, seed)]["evaluations"][0] for condition in NEW_CONDITIONS]
        if len({json.dumps(item["risk_profile"], sort_keys=True) for item in zero}) != 1:
            raise AssertionError(f"update-zero fixed profiles differ: seed {seed}")
        for item in zero:
            drift = item["behavior_change_from_source"]
            if (
                abs(drift["mean_kl_source_to_current"]) > 1e-7
                or drift["argmax_action_change_count"] != 0
            ):
                raise AssertionError("update-zero actor is not the source actor")

    new_rows = {
        condition: [
            lr_summary.seed_row(condition, seed, cells[(condition, seed)])
            for seed in SEEDS
        ]
        for condition in NEW_CONDITIONS
    }
    policies: dict[tuple[str, int], dict] = {}
    for condition in NEW_CONDITIONS:
        for row in new_rows[condition]:
            seed = int(row["seed"])
            evaluation = cells[(condition, seed)]["evaluations"][UPDATES[-1]]
            policies[(condition, seed)] = policy_record(
                condition, seed, evaluation, row, "new_training"
            )
    for condition in REUSED_CONDITIONS:
        prior_by_seed = {int(row["seed"]): row for row in prior_raw["rows"][condition]}
        if set(prior_by_seed) != set(SEEDS):
            raise AssertionError(f"prior seed set changed: {condition}")
        for seed in SEEDS:
            evaluation = reused_evaluation(condition, seed)
            row = prior_by_seed[seed]
            validate_reused_row(row, evaluation)
            policies[(condition, seed)] = policy_record(
                condition, seed, evaluation, row, "reused_pinned_evaluation"
            )

    policy_rows = {
        condition: [policies[(condition, seed)] for seed in SEEDS]
        for condition in ALL_CONDITIONS
    }
    aggregates = {
        condition: aggregate_policy_records(rows)
        for condition, rows in policy_rows.items()
    }
    baseline = source_baseline(cells)
    for label, prior_probability_condition in (("p05", "p05_lr1"), ("p30", "p30_lr1")):
        prior_source = prior_raw["aggregates"][prior_probability_condition][
            "exact_mode_mean"
        ]["source_expected_discounted_return"]
        _assert_close(
            baseline["means"]["cross_probability_exact"][label][
                "expected_discounted_return"
            ],
            prior_source,
            f"source baseline {label}",
        )

    comparisons = {
        "p00_vs_p05_direct": paired_difference(
            "p05_lr1", "p00_lr1", policies, "p05"
        ),
        "p30_vs_p05_direct": paired_difference(
            "p05_lr1", "p30_lr1", policies, "p05"
        ),
        "p100_vs_p05_direct": paired_difference(
            "p05_lr1", "p100_lr1", policies, "p05"
        ),
        "p100_vs_p05_quarter_rate": paired_difference(
            "p05_lr025", "p100_lr1", policies, "p05"
        ),
        "p100_vs_p30_direct": paired_difference(
            "p30_lr1", "p100_lr1", policies, "p05"
        ),
        "p100_vs_p30_quarter_rate": paired_difference(
            "p30_lr025", "p100_lr1", policies, "p05"
        ),
    }
    baseline_by_seed = {int(row["seed"]): row for row in baseline["per_seed"]}
    normal_task_changes_from_source = {
        condition: {
            "per_seed": [
                {
                    "seed": seed,
                    "success_rate_difference": policies[(condition, seed)][
                        "normal_kernel_mode"
                    ]["success_rate"]
                    - baseline_by_seed[seed]["normal_success"],
                    "discounted_return_difference": policies[(condition, seed)][
                        "normal_kernel_mode"
                    ]["mean_discounted_return"]
                    - baseline_by_seed[seed]["normal_return"],
                }
                for seed in SEEDS
            ]
        }
        for condition in ALL_CONDITIONS
    }
    for value in normal_task_changes_from_source.values():
        value["means"] = {
            key: mean(row[key] for row in value["per_seed"])
            for key in ("success_rate_difference", "discounted_return_difference")
        }
    total_slots = sum(
        aggregates[condition]["collection_totals"]["collector_slots"]
        for condition in NEW_CONDITIONS
    )
    total_scheduled = sum(
        aggregates[condition]["collection_totals"]["optimizer_applications"]
        + aggregates[condition]["collection_totals"]["empty_minibatches"]
        for condition in NEW_CONDITIONS
    )
    p00_equivalence = all(
        math.isclose(
            aggregates[condition]["cross_probability_exact_mean"]["p00"][
                "expected_discounted_return"
            ],
            aggregates[condition]["normal_kernel_mode_mean"][
                "mean_discounted_return"
            ],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for condition in ALL_CONDITIONS
    )
    result = {
        "schema_version": (
            "hackrl_pack_inventory_overwrite_train_probability_transfer_result_v1"
        ),
        "status": "complete_fixed_budget_no_automatic_extension",
        "generated_unix": time.time(),
        "execution_head": run_manifest["execution_head"],
        "aggregation_head": git("rev-parse", "HEAD"),
        "manifest_sha256": sha256(SPEC_PATH),
        "training_completion": completion,
        "gates": {
            "all_10_new_jobs_complete": True,
            "total_new_collector_slots_exact": total_slots
            == SPEC["budget"]["total_collector_slots"],
            "scheduled_optimizer_accounting_exact": total_scheduled
            == SPEC["budget"]["jobs"]
            * SPEC["optimization"]["scheduled_optimizer_applications_per_job"],
            "paired_new_branch_initialization_identical": True,
            "update_zero_behavior_matches_source_on_frozen_anchor": True,
            "all_new_update_zero_fixed_profiles_identical_within_seed": True,
            "all_retained_state_file_receipts_valid": True,
            "source_checkpoint_file_set_and_hashes_match_manifest": True,
            "execution_source_set_matches_manifest": True,
            "job_matrix_and_run_contract_match_manifest": True,
            "non_intervention_checkpoint_config_matches_source": True,
            "reused_final_evaluations_match_pinned_hashes": True,
            "reused_rows_match_reused_evaluations": True,
            "source_baseline_matches_prior_result": True,
            "cross_probability_matrix_complete": len(policies)
            == len(ALL_CONDITIONS) * len(SEEDS),
            "p00_exact_matches_normal_kernel": p00_equivalence,
            "no_teacher_updates": True,
            "no_bc_updates": True,
            "no_automatic_extension": True,
        },
        "source_baseline": baseline,
        "policy_rows": policy_rows,
        "aggregates": aggregates,
        "cross_probability_return_matrix": {
            condition: {
                label: aggregates[condition]["cross_probability_exact_mean"][label][
                    "expected_discounted_return"
                ]
                for label in EVALUATION_PROBABILITIES
            }
            for condition in ALL_CONDITIONS
        },
        "cross_probability_success_matrix": {
            condition: {
                label: aggregates[condition]["cross_probability_exact_mean"][label][
                    "success_rate"
                ]
                for label in EVALUATION_PROBABILITIES
            }
            for condition in ALL_CONDITIONS
        },
        "primary_p05_comparisons": comparisons,
        "normal_task_changes_from_source": normal_task_changes_from_source,
        "new_training_learning_curves_at_matching_probability": {
            condition: lr_summary.learning_curve(condition, cells)
            for condition in NEW_CONDITIONS
        },
        "mathematical_surrogate": {
            "domain": "IEEE-754 binary64 finite sums over five learner seeds by 32 natural starts",
            "formula": SPEC["evaluation"]["exact_mode_formula"],
            "policies": len(ALL_CONDITIONS) * len(SEEDS),
            "evaluation_probabilities": EVALUATION_PROBABILITIES,
            "mode_paths_per_condition": len(SEEDS) * SPEC["evaluation"]["states"],
            "risk_count_unrestricted": True,
            "matching_probability_sample_evaluation_separate": True,
            "cross_probability_values_are_formula_evaluations_not_new_rollouts": True,
            "legacy_field_mapping": (
                "existing_loss_episode_probability is interpreted here as the "
                "probability of at least one overwrite incident, not the total "
                "failure probability; fixed-kernel failures are reported separately."
            ),
        },
        "non_claims": SPEC["interpretation"]["non_claims"],
        "capacity_end": capacity(),
    }
    if not all(result["gates"].values()):
        raise AssertionError(f"result gate failed: {result['gates']}")
    if not result["capacity_end"]["reserve_preserved"]:
        raise AssertionError("RAID safety reserve was not preserved")
    write(RUN_ROOT / "capacity_end.json", result["capacity_end"])
    write(RUN_ROOT / "result.json", result)
    print(json.dumps(result["primary_p05_comparisons"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
