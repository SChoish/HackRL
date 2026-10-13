#!/usr/bin/env python3
"""Independently recompute the finite-panel transfer result with Decimal."""

from __future__ import annotations

from decimal import Decimal, getcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[1]
RUN_ROOT = Path(
    "/raid/ext_csv/HackRL/runs/pack_inventory_overwrite_train_probability_transfer_v1"
)
PRIOR_ROOT = Path(
    "/raid/ext_csv/HackRL/runs/pack_inventory_overwrite_lr_ppo_v1"
)
RAW_RESULT = RUN_ROOT / "result.json"
OUTPUT = (
    REPOSITORY
    / "docs/manifests/pack_inventory_overwrite_train_probability_transfer_v1_computation_audit.json"
)
SEEDS = (35, 36, 37, 38, 39)
NEW_CONDITIONS = ("p00_lr1", "p100_lr1")
CONDITIONS = (
    "p00_lr1",
    "p05_lr1",
    "p30_lr1",
    "p100_lr1",
    "p05_lr025",
    "p30_lr025",
)
PROBABILITIES = {
    "p00": Decimal("0"),
    "p05": Decimal("0.05"),
    "p30": Decimal("0.30"),
    "p100": Decimal("1"),
}
COMPARISONS = {
    "p00_vs_p05_direct": ("p05_lr1", "p00_lr1"),
    "p30_vs_p05_direct": ("p05_lr1", "p30_lr1"),
    "p100_vs_p05_direct": ("p05_lr1", "p100_lr1"),
    "p100_vs_p05_quarter_rate": ("p05_lr025", "p100_lr1"),
    "p100_vs_p30_direct": ("p30_lr1", "p100_lr1"),
    "p100_vs_p30_quarter_rate": ("p30_lr025", "p100_lr1"),
    "p30_vs_p05_quarter_rate": ("p05_lr025", "p30_lr025"),
    "p30_vs_p00_standard": ("p00_lr1", "p30_lr1"),
    "p100_vs_p00_standard": ("p00_lr1", "p100_lr1"),
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cell_root(condition: str, seed: int) -> Path:
    root = RUN_ROOT if condition in NEW_CONDITIONS else PRIOR_ROOT
    return root / condition / f"seed{seed}"


def decimal_point(condition: str, seed: int, probability: Decimal) -> dict:
    cell = cell_root(condition, seed)
    evaluation = read(cell / "evaluations/adapt_32.json")["evaluation"]
    config = read(cell / "checkpoints/adapt_32/config.json")
    gamma = Decimal(str(config["gamma"]))
    success = Decimal(0)
    discounted_return = Decimal(0)
    incident = Decimal(0)
    episodes = evaluation["risk_profile"]["episodes"]
    for row in episodes:
        fixed_success = Decimal(str(row["fixed_success"]))
        risk_count = int(row["risk_count"])
        survival = (
            Decimal(1)
            if risk_count == 0
            else (Decimal(1) - probability) ** risk_count
        )
        success += fixed_success * survival
        discounted_return += (
            fixed_success
            * gamma ** (int(row["fixed_length"]) - 1)
            * survival
        )
        incident += Decimal(1) - survival
    denominator = Decimal(len(episodes))
    return {
        "success_rate": success / denominator,
        "expected_discounted_return": discounted_return / denominator,
        "overwrite_incident_probability": incident / denominator,
    }


def mean(values) -> Decimal:
    values = list(values)
    return sum(values, Decimal(0)) / Decimal(len(values))


def as_string(value: Decimal) -> str:
    return format(value, "f")


def main() -> None:
    getcontext().prec = 80
    raw = read(RAW_RESULT)
    points = {
        (condition, seed, label): decimal_point(condition, seed, probability)
        for condition in CONDITIONS
        for seed in SEEDS
        for label, probability in PROBABILITIES.items()
    }
    matrix = {}
    maximum_absolute_difference = Decimal(0)
    for condition in CONDITIONS:
        matrix[condition] = {}
        for label in PROBABILITIES:
            point = {
                metric: mean(
                    points[(condition, seed, label)][metric] for seed in SEEDS
                )
                for metric in (
                    "success_rate",
                    "expected_discounted_return",
                    "overwrite_incident_probability",
                )
            }
            matrix[condition][label] = {
                key: as_string(value) for key, value in point.items()
            }
            for metric, raw_key in (
                ("success_rate", "cross_probability_success_matrix"),
                ("expected_discounted_return", "cross_probability_return_matrix"),
            ):
                reported = Decimal(str(raw[raw_key][condition][label]))
                difference = abs(point[metric] - reported)
                maximum_absolute_difference = max(
                    maximum_absolute_difference, difference
                )
    comparisons = {}
    for name, (left, right) in COMPARISONS.items():
        differences = [
            points[(right, seed, "p05")]["expected_discounted_return"]
            - points[(left, seed, "p05")]["expected_discounted_return"]
            for seed in SEEDS
        ]
        recomputed_mean = mean(differences)
        if name in raw["primary_p05_comparisons"]:
            reported_mean = Decimal(
                str(
                    raw["primary_p05_comparisons"][name][
                        "mean_right_minus_left"
                    ]
                )
            )
            maximum_absolute_difference = max(
                maximum_absolute_difference,
                abs(recomputed_mean - reported_mean),
            )
        comparisons[name] = {
            "per_seed_right_minus_left": [as_string(value) for value in differences],
            "mean_right_minus_left": as_string(recomputed_mean),
        }
    collection_checks = {}
    for condition in NEW_CONDITIONS:
        summaries = [
            read(cell_root(condition, seed) / "summary.json") for seed in SEEDS
        ]
        totals = {
            key: sum(int(row[key]) for row in summaries)
            for key in (
                "collector_slots",
                "physical_transitions",
                "valid_ppo_samples",
                "optimizer_applications",
                "empty_minibatches",
                "cumulative_lost_grain",
                "physical_loss_events",
                "overwrite_trigger_events",
                "overwrite_fire_events",
            )
        }
        expected = raw["aggregates"][condition]["collection_totals"]
        collection_checks[condition] = {
            "totals": totals,
            "matches_raw_result": all(
                totals[key] == int(expected[key]) for key in totals
            ),
        }
    direct_input_paths = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            cell = cell_root(condition, seed)
            direct_input_paths.extend(
                [
                    cell / "evaluations/adapt_32.json",
                    cell / "checkpoints/adapt_32/config.json",
                ]
            )
            if condition in NEW_CONDITIONS:
                direct_input_paths.append(cell / "summary.json")
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=REPOSITORY, text=True
    ).splitlines()
    audit = {
        "schema_version": (
            "hackrl_pack_inventory_overwrite_train_probability_transfer_"
            "computation_audit_v1"
        ),
        "status": "pass",
        "method": (
            "Independent Decimal arithmetic over every one of 30 final-policy "
            "risk profiles and all 32 natural starts; no project exact-evaluation "
            "helper or aggregate value is used in the recomputation."
        ),
        "precision_decimal_digits": getcontext().prec,
        "generated_unix": time.time(),
        "execution": {
            "git_head": git_head,
            "git_status_short": git_status,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "argv": sys.argv,
            "documented_command": (
                "env PYTHONPATH=src:scripts "
                "/home/ext_csv/miniconda3/envs/offrl/bin/python "
                "scripts/audit_pack_inventory_overwrite_train_probability_transfer.py"
            ),
        },
        "inputs": {
            "raw_result": str(RAW_RESULT),
            "raw_result_sha256": sha256(RAW_RESULT),
            "run_manifest_sha256": sha256(RUN_ROOT / "run_manifest.json"),
            "training_complete_sha256": sha256(
                RUN_ROOT / "training_complete.json"
            ),
            "audit_script_sha256": sha256(Path(__file__)),
            "direct_input_sha256": {
                str(path): sha256(path) for path in direct_input_paths
            },
        },
        "panel": {
            "learner_seeds": list(SEEDS),
            "conditions": list(CONDITIONS),
            "probabilities": {
                key: as_string(value) for key, value in PROBABILITIES.items()
            },
            "natural_starts_per_seed": 32,
            "policy_start_paths": len(CONDITIONS) * len(SEEDS) * 32,
        },
        "recomputed_matrix": matrix,
        "recomputed_p05_comparisons": comparisons,
        "collection_checks": collection_checks,
        "maximum_absolute_difference_from_binary64_result": as_string(
            maximum_absolute_difference
        ),
        "tolerance": "1e-15",
        "all_numeric_checks_within_tolerance": (
            maximum_absolute_difference <= Decimal("1e-15")
            and all(
                value["matches_raw_result"]
                for value in collection_checks.values()
            )
        ),
        "scope_limit": (
            "This audits arithmetic and recorded accounting, not the simulator, "
            "checkpoint semantics, or the causal interpretation of training probability."
        ),
    }
    if not audit["all_numeric_checks_within_tolerance"]:
        raise AssertionError(audit)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(OUTPUT)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "maximum_absolute_difference": audit[
                    "maximum_absolute_difference_from_binary64_result"
                ],
                "output": str(OUTPUT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
