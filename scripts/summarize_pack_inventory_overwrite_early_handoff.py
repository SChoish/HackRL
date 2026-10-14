#!/usr/bin/env python3
"""Audit and compact the PACK inventory-phase handoff result."""

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
DESIGN_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_early_handoff_v1.json"
RAW_ROOT = Path("/raid/ext_csv/HackRL/runs/pack_inventory_overwrite_early_handoff_v1")
RAW_PATH = RAW_ROOT / "result.json"
RUN_MANIFEST_PATH = RAW_ROOT / "run_manifest.json"
COMPLETE_PATH = RAW_ROOT / "complete.json"
COMPACT_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_early_handoff_v1_results.json"
AUDIT_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_early_handoff_v1_computation_audit.json"
SCRIPT_PATH = REPOSITORY / "scripts/summarize_pack_inventory_overwrite_early_handoff.py"
PROBABILITIES = {
    "p00": Decimal("0"),
    "p05": Decimal("0.05"),
    "p30": Decimal("0.30"),
    "p100": Decimal("1"),
}
DISCOUNT = Decimal("0.995")
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


def paired_contrast(comparison: dict, left: str, right: str) -> dict:
    left_rows = {row["seed"]: row for row in comparison[left]["per_seed"]}
    right_rows = {row["seed"]: row for row in comparison[right]["per_seed"]}
    values = [
        left_rows[seed]["p05_expected_discounted_return"]
        - right_rows[seed]["p05_expected_discounted_return"]
        for seed in sorted(left_rows)
    ]
    return {
        "left": left,
        "right": right,
        "metric": "p05_expected_discounted_return",
        "mean_difference": math.fsum(values) / len(values),
        "per_seed": [
            {"seed": seed, "difference": value}
            for seed, value in zip(sorted(left_rows), values, strict=True)
        ],
        "signs": signs(values),
    }


def carry_diagnostic(raw: dict) -> dict:
    fixed = [episode for row in raw["rows"] for episode in row["fixed"]["episodes"]]
    switched = [episode for episode in fixed if episode["switch_count"] > 0]
    first_events = [
        next(step for step in episode["steps"] if step["switch_event"])
        for episode in switched
    ]
    return {
        "episodes": len(fixed),
        "episodes_with_handoff": len(switched),
        "handoff_count": sum(episode["switch_count"] for episode in fixed),
        "adapted_policy_steps": sum(episode["adapted_policy_steps"] for episode in fixed),
        "first_handoff_safe_action_count": sum(
            not event["executed_risky_harvest"] for event in first_events
        ),
        "first_handoff_risky_action_count": sum(
            event["executed_risky_harvest"] for event in first_events
        ),
        "failed_path_count": sum(not episode["success"] for episode in fixed),
        "risky_start_count": sum(episode["executed_risk_count"] > 0 for episode in fixed),
    }


def build_compact(raw: dict, run_manifest: dict, complete: dict) -> dict:
    fixed = [episode for row in raw["rows"] for episode in row["fixed"]["episodes"]]
    maximum_difference = Decimal(0)
    decimal_points = {}
    for label, probability in PROBABILITIES.items():
        audited = decimal_profile(fixed, probability)
        decimal_points[label] = {metric: str(value) for metric, value in audited.items()}
        for metric, value in audited.items():
            reported = Decimal(str(raw["aggregate"]["exact"][label][metric]))
            difference = abs(value - reported)
            maximum_difference = max(maximum_difference, difference)
            if difference > TOLERANCE:
                raise AssertionError(
                    f"carry_gated.{label}.{metric}: decimal={value} reported={reported} diff={difference}"
                )

    comparison = raw["comparison"]
    contrasts = {
        "carry_gated_minus_source": paired_contrast(comparison, "carry_gated", "source"),
        "carry_gated_minus_persistent": paired_contrast(comparison, "carry_gated", "persistent"),
        "carry_gated_minus_adapted": paired_contrast(comparison, "carry_gated", "adapted"),
    }
    source = comparison["source"]
    late = comparison["persistent"]
    carry = comparison["carry_gated"]
    adapted = comparison["adapted"]
    return {
        "schema_version": "hackrl_pack_inventory_overwrite_early_handoff_compact_result_v1",
        "status": raw["status"],
        "decision": (
            "The carry-gated schedule met the preregistered success rule in this frozen five-seed panel. "
            "Carry-gated execution had the highest p=0.05 expected return, retained 100% fixed-kernel success, "
            "and reduced risky starts from 94 for source-only and 82 for late persistent switching to 40. "
            "Its paired p=0.05 return difference was positive in all five seeds against both source-only and late switching."
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
            "run_manifest": {
                "path": str(RUN_MANIFEST_PATH),
                "sha256": sha256(RUN_MANIFEST_PATH),
            },
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
        "design": {
            "seeds": [35, 36, 37, 38, 39],
            "natural_starts_per_seed": 32,
            "new_execution_mode": "carry_gated",
            "direct_kernels": ["fixed", "inventory_overwrite_p1"],
            "new_episodes": 320,
            "maximum_environment_steps": 40960,
            "learning_updates": 0,
        },
        "protocol": {
            "defect": (
                "On a valid harvest with carried_grain > 0, the harmful kernel overwrites inventory to one. "
                "Any physical grain loss makes delivery of three impossible in this fixture."
            ),
            "source": "Execute the frozen source actor for the full episode.",
            "persistent": (
                "Execute the source actor until it proposes a risky harvest, then execute the adapted actor until delivery or storage "
                "empties inventory; repeat if another source-risk proposal occurs."
            ),
            "carry_gated": (
                "Execute the source actor whenever carried_grain is zero and the adapted actor whenever carried_grain is positive; "
                "the reversible gate can therefore hand off multiple times in one episode."
            ),
            "adapted": "Execute the frozen p=1-adapted actor for the full episode.",
            "risky_start": "A fixed-kernel deterministic path that executes at least one valid harvest while carrying grain.",
            "episode_accounting": (
                "Five seeds times 32 natural starts give 160 paths per kernel; fixed and always-fire endpoint kernels give 320 direct episodes."
            ),
            "success_rule": (
                "Improve or retain p=0.05 expected return and fixed-kernel task performance; lower risk alone is not success."
            ),
            "gate_source": "The evaluator supplies both switching rules; neither policy learned the gate.",
        },
        "gates": raw["gates"],
        "primary_p05": {
            mode: {
                "expected_discounted_return": comparison[mode]["exact"]["p05"]["expected_discounted_return"],
                "success_rate": comparison[mode]["exact"]["p05"]["success_rate"],
                "fixed_success_rate": comparison[mode]["fixed_success_rate"],
                "fixed_mean_discounted_return": comparison[mode]["fixed_mean_discounted_return"],
                "fixed_mean_length": comparison[mode]["fixed_mean_length"],
                "fixed_risky_start_count": comparison[mode]["fixed_risky_start_count"],
            }
            for mode in ("source", "persistent", "carry_gated", "adapted")
        },
        "paired_p05_return_contrasts": contrasts,
        "carry_gate_diagnostic": carry_diagnostic(raw),
        "normal_kernel_cost_vs_source": {
            "return_difference": carry["fixed_mean_discounted_return"]
            - source["fixed_mean_discounted_return"],
            "mean_length_difference": carry["fixed_mean_length"] - source["fixed_mean_length"],
            "success_rate_difference": carry["fixed_success_rate"] - source["fixed_success_rate"],
        },
        "risk_reduction": {
            "source_to_late": source["fixed_risky_start_count"] - late["fixed_risky_start_count"],
            "late_to_carry": late["fixed_risky_start_count"] - carry["fixed_risky_start_count"],
            "carry_to_adapted": carry["fixed_risky_start_count"] - adapted["fixed_risky_start_count"],
        },
        "exact_carry_gate": raw["aggregate"]["exact"],
        "per_seed": {
            mode: comparison[mode]["per_seed"]
            for mode in ("source", "persistent", "carry_gated", "adapted")
        },
        "decimal_audit": {
            "domain": "Decimal, 50 digits",
            "formula": "mean(success * 0.995^(length-1) * (1-p)^executed_risk_count)",
            "points": decimal_points,
            "tolerance": str(TOLERANCE),
            "maximum_absolute_difference": str(maximum_difference),
            "passed": maximum_difference <= TOLERANCE,
        },
        "interpretation": {
            "primary": (
                "The tested carry-gated composite reduced risky starts and improved p=0.05 return, whereas the tested late-risk "
                "composite retained most source-policy risky paths."
            ),
            "composition": (
                "Carry-gated execution outperformed full source execution in all five seeds, showing that the two frozen policies "
                "were composable under this supplied phase gate. Its mean also exceeded adapted-from-start, but that contrast was "
                "positive only in seed 35 and does not establish stable superiority over adapted-from-start."
            ),
            "cost": (
                "The fixed-kernel success rate stayed at 100%, while mean fixed return decreased slightly and path length increased, "
                "so the safer route was not free."
            ),
            "limit": (
                "Carry-gated and late-risk execution differ in the gate predicate, handoff timing, and states controlled by the adapted actor. "
                "The comparison therefore does not identify timing alone, explain the adapted actor internally, or show that a learned gate "
                "would recover the same benefit."
            ),
        },
        "non_claims": [
            "The evaluator supplied the carry gate; the policy did not learn when to switch.",
            "The result does not isolate state-distribution shift as the cause of the late-switch failure.",
            "The exact p=0.05 and p=0.30 values are finite-panel expectations from deterministic fixed paths, not fresh stochastic rollouts.",
            "Five Dual-history learner seeds do not establish population frequency or significance.",
        ],
    }


def build_audit(compact: dict, started_clean: bool) -> dict:
    run_manifest = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
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
        "claim_id": "pack_inventory_overwrite_early_handoff_v1_finite_panel",
        "repository": {"commit": git("rev-parse", "HEAD"), "dirty": not started_clean},
        "command": [
            "PYTHONPATH=src:scripts JAX_PLATFORMS=cpu /home/ext_csv/miniconda3/envs/offrl/bin/python scripts/evaluate_pack_inventory_overwrite_early_handoff.py --run",
            "/home/ext_csv/miniconda3/envs/offrl/bin/python scripts/summarize_pack_inventory_overwrite_early_handoff.py --write",
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
                "For the carry-gated frozen execution mode on 5 learner seeds x 32 natural starts, recompute exact expected success "
                "and discounted return at overwrite probabilities 0, 0.05, 0.30, and 1 from recorded fixed-kernel paths."
            ),
            "coefficient_domain": "Decimal arithmetic at 50 digits for the independent aggregate; binary64/JAX values in the recorded run.",
            "conventions": (
                "Discount 0.995; terminal success reward represented by 0.995^(T-1); independent overwrite survival (1-p)^k; "
                "one physical grain loss makes the three-grain goal impossible."
            ),
            "inputs": [
                {"path": str(RAW_PATH), "sha256": sha256(RAW_PATH)},
                {"path": str(RUN_MANIFEST_PATH), "sha256": sha256(RUN_MANIFEST_PATH)},
                {"path": str(DESIGN_PATH.relative_to(REPOSITORY)), "sha256": sha256(DESIGN_PATH)},
            ],
            "bounds": {
                "learner_seeds": [35, 36, 37, 38, 39],
                "natural_starts_per_seed": 32,
                "new_execution_modes": 1,
                "direct_kernels": 2,
                "direct_episodes": 320,
                "horizon_ticks": 128,
                "maximum_environment_steps": 40960,
                "evaluated_probabilities": [0.0, 0.05, 0.30, 1.0],
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
            "All 320 direct episodes were recorded.",
            "Exact p=0 and p=1 success and return matched direct endpoint kernels for every seed.",
            "Pre-overwrite deterministic histories matched between fixed and always-fire kernels.",
            "The prior source, late-switch, and adapted results remained hash-pinned.",
            f"Independent Decimal recomputation differed by at most {compact['decimal_audit']['maximum_absolute_difference']}.",
            "All five seeds and every failed fixed-kernel path were retained.",
        ],
        "result": "Implementation and finite assertion verified in the stated five-seed, 32-start panel.",
        "residual_risks": [
            "The policy network and simulator implementation were not independently reimplemented.",
            "The intermediate-probability result relies on the environment property that any overwrite loss makes the delivery goal impossible.",
            "The supplied phase gate is not an autonomous switching method.",
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
    if design["budget"]["episodes"] != 320 or not all(raw["gates"].values()):
        raise RuntimeError("execution contract or result gates failed")
    compact = build_compact(raw, run_manifest, complete)
    write_json(COMPACT_PATH, compact)
    audit = build_audit(compact, started_clean)
    write_json(AUDIT_PATH, audit)
    print(
        json.dumps(
            {
                "status": compact["status"],
                "compact_result": str(COMPACT_PATH.relative_to(REPOSITORY)),
                "compact_sha256": sha256(COMPACT_PATH),
                "audit": str(AUDIT_PATH.relative_to(REPOSITORY)),
                "audit_sha256": sha256(AUDIT_PATH),
                "maximum_decimal_difference": compact["decimal_audit"]["maximum_absolute_difference"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
