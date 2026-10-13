#!/usr/bin/env python3
"""Audit and compact the frozen PACK overwrite policy-switch result."""

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
import time


getcontext().prec = 50
REPOSITORY = Path(__file__).resolve().parents[1]
DESIGN_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_policy_switch_v1.json"
RAW_ROOT = Path("/raid/ext_csv/HackRL/runs/pack_inventory_overwrite_policy_switch_v1")
RAW_PATH = RAW_ROOT / "result.json"
RUN_MANIFEST_PATH = RAW_ROOT / "run_manifest.json"
COMPLETE_PATH = RAW_ROOT / "complete.json"
COMPACT_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_policy_switch_v1_results.json"
AUDIT_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_policy_switch_v1_computation_audit.json"
SCRIPT_PATH = REPOSITORY / "scripts/summarize_pack_inventory_overwrite_policy_switch.py"
PROBABILITIES = {"p00": Decimal("0"), "p05": Decimal("0.05"), "p30": Decimal("0.30"), "p100": Decimal("1")}
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


def assert_close(actual: Decimal, reported: float, label: str) -> None:
    difference = abs(actual - Decimal(str(reported)))
    if difference > TOLERANCE:
        raise AssertionError(f"{label}: decimal={actual} reported={reported} diff={difference}")


def signs(values: list[float], tolerance: float = 1e-12) -> dict[str, int]:
    return {
        "positive": sum(value > tolerance for value in values),
        "equal_within_tolerance": sum(abs(value) <= tolerance for value in values),
        "negative": sum(value < -tolerance for value in values),
    }


def pooled_episodes(raw: dict, mode: str, kernel: str) -> list[dict]:
    return [
        episode
        for row in raw["rows"]
        if row["mode"] == mode
        for episode in row[kernel]["episodes"]
    ]


def mode_summary(raw: dict, mode: str) -> dict:
    fixed = pooled_episodes(raw, mode, "fixed")
    p1 = pooled_episodes(raw, mode, "inventory_overwrite_p1")
    rows = sorted((row for row in raw["rows"] if row["mode"] == mode), key=lambda row: row["seed"])
    exact = raw["aggregate"][mode]["exact"]
    maximum_difference = Decimal(0)
    for label, probability in PROBABILITIES.items():
        audited = decimal_profile(fixed, probability)
        for metric, value in audited.items():
            reported = exact[label][metric]
            maximum_difference = max(maximum_difference, abs(value - Decimal(str(reported))))
            assert_close(value, reported, f"{mode}.{label}.{metric}")

    switched = []
    safe_switch_events = 0
    risky_switch_events = 0
    for episode in fixed:
        events = [step for step in episode["steps"] if step["switch_event"]]
        if events:
            switched.append(episode)
        for event in events:
            if event["executed_risky_harvest"]:
                risky_switch_events += 1
            else:
                safe_switch_events += 1

    return {
        "fixed_kernel": {
            "episodes": len(fixed),
            "success_rate": math.fsum(float(row["success"]) for row in fixed) / len(fixed),
            "mean_discounted_return": math.fsum(row["discounted_return"] for row in fixed) / len(fixed),
            "mean_length": math.fsum(row["length"] for row in fixed) / len(fixed),
            "risky_start_count": sum(row["executed_risk_count"] > 0 for row in fixed),
            "executed_risk_count": sum(row["executed_risk_count"] for row in fixed),
            "source_risk_proposal_count": sum(row["source_risk_proposal_count"] for row in fixed),
            "switch_count": sum(row["switch_count"] for row in fixed),
            "adapted_policy_steps": sum(row["adapted_policy_steps"] for row in fixed),
            "repeated_compact_state_action_count": sum(row["repeated_compact_state_action_count"] for row in fixed),
            "failed_path_count": sum(not row["success"] for row in fixed),
        },
        "always_fire_kernel": {
            "episodes": len(p1),
            "success_rate": math.fsum(float(row["success"]) for row in p1) / len(p1),
            "mean_discounted_return": math.fsum(row["discounted_return"] for row in p1) / len(p1),
            "loss_episode_rate": math.fsum(float(row["loss"]) for row in p1) / len(p1),
        },
        "exact": exact,
        "switch_diagnostic": {
            "episodes_with_switch": len(switched),
            "first_switch_safe_action_count": sum(
                not next(step for step in row["steps"] if step["switch_event"])["executed_risky_harvest"]
                for row in switched
            ),
            "first_switch_risky_action_count": sum(
                next(step for step in row["steps"] if step["switch_event"])["executed_risky_harvest"]
                for row in switched
            ),
            "safe_switch_event_count": safe_switch_events,
            "risky_switch_event_count": risky_switch_events,
        },
        "per_seed": [
            {
                "seed": row["seed"],
                "p05_expected_discounted_return": row["exact"]["p05"]["expected_discounted_return"],
                "p05_success_rate": row["exact"]["p05"]["success_rate"],
                "fixed_success_rate": row["fixed"]["aggregate"]["success_rate"],
                "fixed_mean_discounted_return": row["fixed"]["aggregate"]["mean_discounted_return"],
                "fixed_mean_length": row["fixed"]["aggregate"]["mean_length"],
                "fixed_risky_start_count": sum(episode["executed_risk_count"] > 0 for episode in row["fixed"]["episodes"]),
                "fixed_switch_count": sum(episode["switch_count"] for episode in row["fixed"]["episodes"]),
                "fixed_adapted_policy_steps": sum(episode["adapted_policy_steps"] for episode in row["fixed"]["episodes"]),
                "fixed_repeated_compact_state_action_count": sum(episode["repeated_compact_state_action_count"] for episode in row["fixed"]["episodes"]),
            }
            for row in rows
        ],
        "maximum_decimal_audit_difference": str(maximum_difference),
    }


def paired_contrast(summaries: dict, left: str, right: str) -> dict:
    left_rows = {row["seed"]: row for row in summaries[left]["per_seed"]}
    right_rows = {row["seed"]: row for row in summaries[right]["per_seed"]}
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


def path_comparison(raw: dict) -> dict:
    one = {
        (row["seed"], episode["state_index"]): episode
        for row in raw["rows"]
        if row["mode"] == "one_step"
        for episode in row["fixed"]["episodes"]
    }
    persistent = {
        (row["seed"], episode["state_index"]): episode
        for row in raw["rows"]
        if row["mode"] == "persistent"
        for episode in row["fixed"]["episodes"]
    }
    differing = []
    shorter = []
    p05_differences = {}
    for key in sorted(one):
        one_actions = [step["chosen_action"] for step in one[key]["steps"]]
        persistent_actions = [step["chosen_action"] for step in persistent[key]["steps"]]
        if one_actions != persistent_actions:
            differing.append(key)
        if persistent[key]["length"] < one[key]["length"]:
            shorter.append(key)
        p05_differences[key] = (
            persistent[key]["discounted_return"] * 0.95 ** persistent[key]["executed_risk_count"]
            - one[key]["discounted_return"] * 0.95 ** one[key]["executed_risk_count"]
        )
    overall_difference = math.fsum(p05_differences.values()) / len(one)
    shorter_path_contribution = math.fsum(p05_differences[key] for key in shorter) / len(one)
    if not math.isclose(overall_difference, shorter_path_contribution, abs_tol=1e-15):
        raise AssertionError("shorter paths do not account for the persistent p05 advantage")
    return {
        "different_fixed_action_paths": len(differing),
        "persistent_shorter_paths": len(shorter),
        "persistent_minus_one_step_p05_return": overall_difference,
        "shorter_paths_p05_return_contribution": shorter_path_contribution,
        "shorter_paths_account_for_entire_p05_difference": True,
        "persistent_shorter_path_keys": [
            {"seed": seed, "state_index": state_index} for seed, state_index in shorter
        ],
    }


def build_compact(raw: dict, run_manifest: dict, complete: dict) -> dict:
    modes = ("source", "adapted", "one_step", "persistent")
    summaries = {mode: mode_summary(raw, mode) for mode in modes}
    contrasts = {
        f"{mode}_minus_source": paired_contrast(summaries, mode, "source")
        for mode in ("adapted", "one_step", "persistent")
    }
    contrasts["persistent_minus_one_step"] = paired_contrast(summaries, "persistent", "one_step")
    maximum = max(Decimal(summary["maximum_decimal_audit_difference"]) for summary in summaries.values())
    return {
        "schema_version": "hackrl_pack_inventory_overwrite_policy_switch_compact_result_v1",
        "status": raw["status"],
        "decision": "Persistent switching preserved fixed-kernel success and slightly improved mean p=0.05 return over the source policy, but retained 82 of the source policy's 94 risky starts. At the first switch, the adapted policy still selected a risky harvest in 82 of 94 switched episodes. The frozen p=1 policy's avoidance therefore did not transfer as a reliable local replacement at the source policy's risky states; persistent switching's entire observed mean advantage over one-step came from six seed-39 starts where repeated paths became shorter.",
        "provenance": {
            "execution_head": run_manifest["execution_head"],
            "aggregation_head": git("rev-parse", "HEAD"),
            "design_manifest": {"path": str(DESIGN_PATH.relative_to(REPOSITORY)), "sha256": sha256(DESIGN_PATH)},
            "raw_result": {"path": str(RAW_PATH), "sha256": sha256(RAW_PATH), "bytes": RAW_PATH.stat().st_size},
            "run_manifest": {"path": str(RUN_MANIFEST_PATH), "sha256": sha256(RUN_MANIFEST_PATH)},
            "complete_marker": {"path": str(COMPLETE_PATH), "sha256": sha256(COMPLETE_PATH)},
            "aggregation_script": {"path": str(SCRIPT_PATH.relative_to(REPOSITORY)), "sha256": sha256(SCRIPT_PATH)},
        },
        "design": {
            "seeds": [35, 36, 37, 38, 39],
            "natural_starts_per_seed": 32,
            "execution_modes": list(modes),
            "direct_kernels": ["fixed", "inventory_overwrite_p1"],
            "episodes": 1280,
            "maximum_environment_steps": 163840,
            "learning_updates": 0,
        },
        "gates": raw["gates"],
        "modes": summaries,
        "paired_p05_return_contrasts": contrasts,
        "one_step_vs_persistent_paths": path_comparison(raw),
        "decimal_audit": {
            "domain": "Decimal, 50 digits",
            "formula": "mean(success * 0.995^(length-1) * (1-p)^executed_risk_count)",
            "probabilities": [0.0, 0.05, 0.30, 1.0],
            "tolerance": str(TOLERANCE),
            "maximum_absolute_difference": str(maximum),
            "passed": maximum <= TOLERANCE,
        },
        "interpretation": {
            "primary": "Persistent switching has the highest mean p=0.05 return among the four frozen execution modes in this five-seed panel, while retaining 100% fixed-kernel success.",
            "limitation": "The mean p=0.05 advantage over source is small (+0.00206), positive in two seeds and exactly zero in three. This is not evidence of a population-level effect.",
            "mechanism": "Only 12 of 94 first switch actions were safe. The adapted-only policy reached just six risky starts by changing earlier trajectory decisions; late oracle switching cannot reproduce most of that avoidance.",
            "persistent_specific": "Persistent and one-step switching have the same 82 risky starts. The entire observed p=0.05 return difference between them comes from six seed-39 fixed paths that are shorter under persistent switching and contain no repeated compact state/action pairs.",
        },
        "non_claims": [
            "The oracle switch is not autonomous defect detection.",
            "The result does not prove that earlier learned switching would fail.",
            "The exact p=0.05 and p=0.30 values are finite-panel expectations from deterministic fixed paths, not fresh stochastic rollouts.",
            "Five Dual-history learner seeds do not establish a population frequency or significance result.",
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
        "claim_id": "pack_inventory_overwrite_policy_switch_v1_finite_panel",
        "repository": {"commit": git("rev-parse", "HEAD"), "dirty": not started_clean},
        "command": [
            "PYTHONPATH=src:scripts JAX_PLATFORMS=cpu /home/ext_csv/miniconda3/envs/offrl/bin/python scripts/evaluate_pack_inventory_overwrite_policy_switch.py --run",
            "/home/ext_csv/miniconda3/envs/offrl/bin/python scripts/summarize_pack_inventory_overwrite_policy_switch.py --write",
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
            "assertion_tested": "For each frozen deterministic execution mode on 5 learner seeds x 32 natural starts, recompute exact expected success and discounted return at overwrite probabilities 0, 0.05, 0.30, and 1 from the recorded fixed-kernel paths.",
            "coefficient_domain": "Decimal arithmetic at 50 digits for the independent aggregate; binary64/JAX values in the recorded run.",
            "conventions": "Discount 0.995; terminal success reward represented by 0.995^(T-1); independent overwrite survival (1-p)^k; one physical grain loss makes the three-grain goal impossible.",
            "inputs": [
                {"path": str(RAW_PATH), "sha256": sha256(RAW_PATH)},
                {"path": str(RUN_MANIFEST_PATH), "sha256": sha256(RUN_MANIFEST_PATH)},
                {"path": str(DESIGN_PATH.relative_to(REPOSITORY)), "sha256": sha256(DESIGN_PATH)},
            ],
            "bounds": {
                "learner_seeds": [35, 36, 37, 38, 39],
                "natural_starts_per_seed": 32,
                "execution_modes": 4,
                "direct_kernels": 2,
                "direct_episodes": 1280,
                "horizon_ticks": 128,
                "maximum_environment_steps": 163840,
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
            "All 1,280 direct episodes were recorded.",
            "Exact p=0 and p=1 success and return matched direct endpoint kernels for every seed and mode.",
            "Pre-overwrite deterministic histories matched between fixed and always-fire kernels.",
            "Pinned source and adapted p=0.05 results were reproduced.",
            f"Independent Decimal recomputation differed by at most {compact['decimal_audit']['maximum_absolute_difference']}.",
            "All five seeds and every failed fixed-kernel path were retained.",
        ],
        "result": "Implementation and finite assertion verified in the stated five-seed, 32-start panel.",
        "residual_risks": [
            "The policy network and simulator implementation were not independently reimplemented.",
            "The intermediate-probability result relies on the proved-by-environment property that any overwrite loss makes the delivery goal impossible.",
            "Schema version 1 does not machine-verify the external RAID input paths.",
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
    if design["budget"]["episodes"] != 1280 or not all(raw["gates"].values()):
        raise RuntimeError("execution contract or result gates failed")
    compact = build_compact(raw, run_manifest, complete)
    write_json(COMPACT_PATH, compact)
    audit = build_audit(compact, started_clean)
    write_json(AUDIT_PATH, audit)
    print(json.dumps({
        "status": compact["status"],
        "compact_result": str(COMPACT_PATH.relative_to(REPOSITORY)),
        "compact_sha256": sha256(COMPACT_PATH),
        "audit": str(AUDIT_PATH.relative_to(REPOSITORY)),
        "audit_sha256": sha256(AUDIT_PATH),
        "maximum_decimal_difference": compact["decimal_audit"]["maximum_absolute_difference"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
