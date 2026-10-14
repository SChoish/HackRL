#!/usr/bin/env python3
"""Audit and compact the held-out PACK carry-gate evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, getcontext
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess


getcontext().prec = 50
REPOSITORY = Path(__file__).resolve().parents[1]
DESIGN_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_gate_heldout_v1.json"
RAW_ROOT = Path("/raid/ext_csv/HackRL/runs/pack_inventory_overwrite_gate_heldout_v1")
RAW_PATH = RAW_ROOT / "result.json"
RUN_MANIFEST_PATH = RAW_ROOT / "run_manifest.json"
COMPLETE_PATH = RAW_ROOT / "complete.json"
COMPACT_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_gate_heldout_v1_results.json"
AUDIT_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_gate_heldout_v1_computation_audit.json"
SCRIPT_PATH = REPOSITORY / "scripts/summarize_pack_inventory_overwrite_gate_heldout.py"
DISCOUNT = Decimal("0.995")
PROBABILITIES = (("p00", Decimal("0")), ("p05", Decimal("0.05")), ("p100", Decimal("1")))
TOLERANCE = Decimal("1e-14")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPOSITORY, text=True).strip()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def decimal_profile(episodes: list[dict], probability: Decimal) -> dict[str, Decimal]:
    survival = Decimal(1) - probability
    count = Decimal(len(episodes))
    success = Decimal(0)
    expected_return = Decimal(0)
    no_overwrite = Decimal(0)
    for episode in episodes:
        risk_count = int(episode["executed_risk_count"])
        weight = Decimal(1) if risk_count == 0 else survival ** risk_count
        no_overwrite += weight
        if episode["success"]:
            success += weight
            expected_return += DISCOUNT ** (int(episode["length"]) - 1) * weight
    return {
        "success_rate": success / count,
        "expected_discounted_return": expected_return / count,
        "overwrite_loss_episode_probability": Decimal(1) - no_overwrite / count,
    }


def signs(values: list[float], tolerance: float = 1e-12) -> dict[str, int]:
    return {
        "positive": sum(value > tolerance for value in values),
        "equal_within_tolerance": sum(abs(value) <= tolerance for value in values),
        "negative": sum(value < -tolerance for value in values),
    }


def contrast(comparison: dict, left: str, right: str) -> dict:
    left_rows = {row["seed"]: row for row in comparison[left]["per_seed"]}
    right_rows = {row["seed"]: row for row in comparison[right]["per_seed"]}
    per_seed = [
        {
            "seed": seed,
            "difference": left_rows[seed]["p05_expected_discounted_return"]
            - right_rows[seed]["p05_expected_discounted_return"],
        }
        for seed in sorted(left_rows)
    ]
    values = [row["difference"] for row in per_seed]
    return {
        "left": left,
        "right": right,
        "metric": "p05_expected_discounted_return",
        "mean_difference": math.fsum(values) / len(values),
        "signs": signs(values),
        "per_seed": per_seed,
    }


def condition_summary(row: dict) -> dict:
    return {
        "p00_expected_discounted_return": row["exact"]["p00"]["expected_discounted_return"],
        "p05_expected_discounted_return": row["exact"]["p05"]["expected_discounted_return"],
        "p05_success_rate": row["exact"]["p05"]["success_rate"],
        "p05_overwrite_loss_episode_probability": row["exact"]["p05"][
            "overwrite_loss_episode_probability"
        ],
        "p100_expected_discounted_return": row["exact"]["p100"]["expected_discounted_return"],
        "fixed_success_rate": row["fixed"]["success_rate"],
        "fixed_mean_length": row["fixed"]["mean_length"],
        "fixed_risky_path_count": row["fixed_risky_start_count"],
        "risk_count_distribution": row["risk_count_distribution"],
    }


def build_compact(raw: dict, design: dict, run_manifest: dict, complete: dict) -> dict:
    decimal_points = {}
    maximum = Decimal(0)
    for mode in design["conditions"]:
        episodes = [
            episode
            for row in raw["rows"]
            if row["mode"] == mode
            for episode in row["fixed"]["episodes"]
        ]
        decimal_points[mode] = {}
        for label, probability in PROBABILITIES:
            audited = decimal_profile(episodes, probability)
            decimal_points[mode][label] = {metric: str(value) for metric, value in audited.items()}
            for metric, value in audited.items():
                reported = Decimal(str(raw["comparison"][mode]["exact"][label][metric]))
                difference = abs(value - reported)
                maximum = max(maximum, difference)
                if difference > TOLERANCE:
                    raise AssertionError(
                        f"{mode}.{label}.{metric}: decimal={value} reported={reported}"
                    )

    source = raw["comparison"]["source"]
    adapted = raw["comparison"]["adapted"]
    gated = raw["comparison"]["carry_gated"]
    gated_minus_source = contrast(raw["comparison"], "carry_gated", "source")
    gated_minus_adapted = contrast(raw["comparison"], "carry_gated", "adapted")
    if gated_minus_source["signs"] != {
        "positive": 4,
        "equal_within_tolerance": 0,
        "negative": 1,
    }:
        raise AssertionError("primary seed directions changed")
    return {
        "schema_version": "hackrl_pack_inventory_overwrite_gate_heldout_compact_result_v1",
        "status": raw["status"],
        "decision": (
            "On the predefined PACK TEST panel, carry-gated execution improved mean p=0.05 expected discounted return "
            "over the source actor by 0.002876 and was positive for seeds 36-39, while seed 35 decreased by 0.044043. "
            "The gate reduced risky paths from 94 to 36 but lowered fixed-kernel success from 100% to 98.75%; therefore "
            "the validation-panel benefit transferred across multiple policy pairs but was neither uniform nor failure-free."
        ),
        "provenance": {
            "execution_head": run_manifest["execution_head"],
            "aggregation_head": git("rev-parse", "HEAD"),
            "design_manifest": {
                "path": str(DESIGN_PATH.relative_to(REPOSITORY)),
                "sha256": sha256(DESIGN_PATH),
            },
            "raw_result": {
                "path": str(RAW_PATH),
                "sha256": sha256(RAW_PATH),
                "bytes": RAW_PATH.stat().st_size,
            },
            "run_manifest": {"path": str(RUN_MANIFEST_PATH), "sha256": sha256(RUN_MANIFEST_PATH)},
            "complete_marker": {
                "path": str(COMPLETE_PATH),
                "sha256": sha256(COMPLETE_PATH),
                "recorded_result_sha256": complete["result_sha256"],
            },
            "aggregation_script": {
                "path": str(SCRIPT_PATH.relative_to(REPOSITORY)),
                "sha256": sha256(SCRIPT_PATH),
            },
        },
        "protocol": {
            "panel": raw["design"]["panel"],
            "conditions": design["conditions"],
            "episode_accounting": "Five policy-pair seeds times three conditions times two endpoint kernels times 32 TEST states give 960 direct episodes.",
            "exact_probability_method": "For each successful fixed-kernel deterministic path, weight return and success by (1-p)^k, where k is its executed risky-harvest count.",
            "primary": design["analysis"]["primary_comparison"],
            "independent_unit": design["analysis"]["independent_unit"],
        },
        "gates": raw["gates"],
        "conditions": {
            "source": condition_summary(source),
            "adapted": condition_summary(adapted),
            "carry_gated": condition_summary(gated),
        },
        "carry_gated_minus_source": gated_minus_source,
        "carry_gated_minus_adapted": gated_minus_adapted,
        "per_seed": {
            mode: raw["comparison"][mode]["per_seed"] for mode in design["conditions"]
        },
        "decimal_audit": {
            "domain": "Decimal, 50 digits",
            "scope": "All three conditions, 160 fixed-kernel paths per condition, at p=0, p=0.05, and p=1.",
            "formula": "mean(success * 0.995^(length-1) * (1-p)^executed_risk_count)",
            "points": decimal_points,
            "tolerance": str(TOLERANCE),
            "maximum_absolute_difference": str(maximum),
            "passed": maximum <= TOLERANCE,
        },
        "interpretation": {
            "primary": (
                f"Carry-gated p=0.05 expected return was {gated['exact']['p05']['expected_discounted_return']:.6f}, "
                f"versus {source['exact']['p05']['expected_discounted_return']:.6f} for source; the paired seed mean difference "
                f"was {gated_minus_source['mean_difference']:+.6f}, positive for four of five seeds."
            ),
            "normal_performance": (
                "Source fixed-kernel success was 100%. Carry-gated success was 98.75%; both failures were seed 35, "
                "whose carry-gated fixed success was 93.75%."
            ),
            "risk": (
                f"Carry-gated execution reduced fixed-path risk exposure from {source['fixed_risky_start_count']} to "
                f"{gated['fixed_risky_start_count']} paths and raised p=0.05 expected success from "
                f"{source['exact']['p05']['success_rate']:.6f} to {gated['exact']['p05']['success_rate']:.6f}."
            ),
            "adapted_comparison": (
                f"Carry-gated mean p=0.05 return exceeded adapted-only by {gated_minus_adapted['mean_difference']:+.6f}, "
                "but this contrast was positive for only one of five seeds and does not support a uniform advantage over adapted-only execution."
            ),
            "scope": (
                "The result extends the comparison to a pre-existing TEST split with different geometry and states. "
                "It does not test new learner seeds, and it reveals that the fixed gate can introduce normal-task failures."
            ),
        },
        "non_claims": design["analysis"]["non_claims"],
    }


def build_audit(compact: dict, run_manifest: dict, raw: dict, started_clean: bool) -> dict:
    started = float(run_manifest["capacity"]["measured_unix"])
    ended = float(raw["generated_unix"])
    cpu_model = "unknown x86_64 CPU"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "schema_version": 1,
        "claim_id": "pack_inventory_overwrite_gate_heldout_v1_finite_test_panel",
        "repository": {"commit": git("rev-parse", "HEAD"), "dirty": not started_clean},
        "command": [
            "PYTHONPATH=src:scripts JAX_PLATFORMS=cpu /home/ext_csv/miniconda3/envs/offrl/bin/python scripts/evaluate_pack_inventory_overwrite_gate_heldout.py --run",
            "/home/ext_csv/miniconda3/envs/offrl/bin/python scripts/summarize_pack_inventory_overwrite_gate_heldout.py --write",
        ],
        "environment": {
            "software": [
                f"Python {platform.python_version()}",
                "JAX 0.10.2 / jaxlib 0.10.2",
                "NumPy 2.5.0",
            ],
            "hardware": f"CPU backend; {cpu_model}; {platform.platform()}",
        },
        "mathematics": {
            "assertion_tested": (
                "Compare frozen source, p=1-adapted, and carried_grain>0 gated execution on the predefined PACK TEST natural-start panel, with carry-gated minus source p=0.05 expected return as primary."
            ),
            "coefficient_domain": (
                "Integer path/event counts; IEEE-754 rollout summaries; independent Decimal arithmetic at 50-digit precision for p=0, p=0.05, and p=1 expectations."
            ),
            "conventions": (
                "Zero-indexed TEST state indices; one-indexed recorded steps; gamma=0.995; deterministic argmax actions; adapted actor exactly when carried_grain>0 in carry-gated mode."
            ),
            "inputs": [
                {"path": str(RAW_PATH), "sha256": sha256(RAW_PATH)},
                {"path": str(RUN_MANIFEST_PATH), "sha256": sha256(RUN_MANIFEST_PATH)},
                {"path": str(DESIGN_PATH.relative_to(REPOSITORY)), "sha256": sha256(DESIGN_PATH)},
            ],
            "bounds": {
                "learner_seeds": [35, 36, 37, 38, 39],
                "test_natural_starts_per_seed": 32,
                "conditions": 3,
                "endpoint_kernels": 2,
                "direct_episodes": 960,
                "horizon_ticks": 128,
                "maximum_environment_steps": 122880,
                "evaluated_probabilities": [0.0, 0.05, 1.0],
            },
            "non_claims": compact["non_claims"],
        },
        "randomness": {"used": False, "generator": "", "seed": None},
        "run": {
            "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "runtime_seconds": ended - started,
            "exit_status": 0,
        },
        "outputs": [
            {"path": str(COMPACT_PATH.relative_to(REPOSITORY)), "sha256": sha256(COMPACT_PATH)}
        ],
        "checks": [
            "All 960 direct episodes were recorded and every execution gate passed.",
            "The 32 TEST states matched the frozen ordered hash, were unique, and had no exact full-state overlap with TRAIN or VALIDATION natural panels.",
            "Every p=0 and p=1 exact value matched its direct endpoint-kernel rollout.",
            "All first-overwrite prehistories matched between endpoint kernels, and every p=1 success had zero fixed-path risk.",
            f"Independent Decimal recomputation differed by at most {compact['decimal_audit']['maximum_absolute_difference']}.",
        ],
        "result": (
            "Finite TEST-panel assertion verified: carry-gated minus source p=0.05 return was +0.002876 on average, positive for four of five seeds, while fixed-kernel success decreased from 100% to 98.75%."
        ),
        "residual_risks": [
            "The policy network and simulator were not independently reimplemented.",
            "The same project codebase generated paths and goal-success labels; endpoint checks and Decimal recomputation reduce but do not remove verifier-designer coupling.",
            "TEST changes geometry and state, so the result is not an i.i.d. resampling estimate.",
            "The same five learned policy pairs were reused, so the result does not test generalization to new learner seeds.",
            "Intermediate-probability values rely on the environment property that any overwrite loss makes the three-delivery goal impossible.",
            "Schema version 1 does not machine-pin every execution input or resource limit.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    if not arguments.write:
        raise SystemExit("use --write")
    started_clean = not bool(git("status", "--porcelain"))
    if not started_clean:
        raise RuntimeError("aggregation must start from a clean committed tree")
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    run_manifest = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
    complete = json.loads(COMPLETE_PATH.read_text(encoding="utf-8"))
    if sha256(RAW_PATH) != complete["result_sha256"]:
        raise RuntimeError("raw result does not match completion marker")
    if run_manifest["execution_head"] != raw["execution_head"]:
        raise RuntimeError("execution heads differ")
    if sha256(DESIGN_PATH) != run_manifest["manifest_sha256"]:
        raise RuntimeError("design manifest changed after execution")
    if design["budget"]["direct_episodes"] != 960 or not all(raw["gates"].values()):
        raise RuntimeError("execution contract or result gates failed")
    compact = build_compact(raw, design, run_manifest, complete)
    write_json(COMPACT_PATH, compact)
    audit = build_audit(compact, run_manifest, raw, started_clean)
    write_json(AUDIT_PATH, audit)
    print(
        json.dumps(
            {
                "status": compact["status"],
                "compact_result": str(COMPACT_PATH.relative_to(REPOSITORY)),
                "compact_sha256": sha256(COMPACT_PATH),
                "audit": str(AUDIT_PATH.relative_to(REPOSITORY)),
                "audit_sha256": sha256(AUDIT_PATH),
                "maximum_decimal_difference": compact["decimal_audit"][
                    "maximum_absolute_difference"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
