#!/usr/bin/env python3
"""Audit and compact the PACK carry-persistent comparison."""

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
DESIGN_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_carry_persistent_v1.json"
RAW_ROOT = Path("/raid/ext_csv/HackRL/runs/pack_inventory_overwrite_carry_persistent_v1")
RAW_PATH = RAW_ROOT / "result.json"
RUN_MANIFEST_PATH = RAW_ROOT / "run_manifest.json"
COMPLETE_PATH = RAW_ROOT / "complete.json"
COMPACT_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_carry_persistent_v1_results.json"
AUDIT_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_carry_persistent_v1_computation_audit.json"
SCRIPT_PATH = REPOSITORY / "scripts/summarize_pack_inventory_overwrite_carry_persistent.py"
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
    without_35 = [row["difference"] for row in per_seed if row["seed"] != 35]
    return {
        "left": left,
        "right": right,
        "metric": "p05_expected_discounted_return",
        "mean_difference": math.fsum(values) / len(values),
        "signs": signs(values),
        "per_seed": per_seed,
        "seed35_excluded_sensitivity": {
            "mean_difference": math.fsum(without_35) / len(without_35),
            "signs": signs(without_35),
            "seeds": [36, 37, 38, 39],
        },
    }


def build_compact(raw: dict, run_manifest: dict, complete: dict) -> dict:
    fixed = [episode for row in raw["rows"] for episode in row["fixed"]["episodes"]]
    maximum = Decimal(0)
    decimal_points = {}
    for label, probability in (("p00", Decimal("0")), ("p05", Decimal("0.05"))):
        audited = decimal_profile(fixed, probability)
        decimal_points[label] = {metric: str(value) for metric, value in audited.items()}
        for metric, value in audited.items():
            reported = Decimal(str(raw["comparison"]["B_carry_persistent"]["exact"][label][metric]))
            difference = abs(value - reported)
            maximum = max(maximum, difference)
            if difference > TOLERANCE:
                raise AssertionError(f"{label}.{metric}: decimal={value} reported={reported}")

    comparison = raw["comparison"]
    primary = contrast(comparison, "C_carry_gated", "B_carry_persistent")
    secondary = contrast(comparison, "B_carry_persistent", "A_late_risk_persistent")
    compact_gates = {
        "all_160_new_B_fixed_episodes_recorded": raw["gates"]["all_160_fixed_episodes_recorded"],
        "every_new_B_episode_has_exactly_one_positive_inventory_handoff": raw["gates"][
            "every_episode_has_one_positive_inventory_handoff"
        ],
        "B_never_relinquishes_policy_control_after_handoff": raw["gates"][
            "no_episode_returns_to_source_after_handoff"
        ],
        "prior_A_and_C_endpoint_validation_remains_pinned": raw["gates"][
            "prior_endpoint_validation_remains_pinned"
        ],
        "no_learning_updates": raw["gates"]["no_learning_updates"],
    }
    return {
        "schema_version": "hackrl_pack_inventory_overwrite_carry_persistent_compact_result_v1",
        "status": raw["status"],
        "decision": (
            "In this five-seed paired panel, inventory-gated return to the source actor raised mean p=0.05 discounted return by 0.02417 "
            "over persistent adapted control, but 97.0% of that mean difference came from seed 35; excluding seed 35, the mean "
            "difference was 0.000897. Both rules produced 40 fixed-kernel paths with at least one executed risky harvest, so no "
            "reduction was observed in that exposure count."
        ),
        "provenance": {
            "execution_head": run_manifest["execution_head"],
            "aggregation_head": git("rev-parse", "HEAD"),
            "design_manifest": {"path": str(DESIGN_PATH.relative_to(REPOSITORY)), "sha256": sha256(DESIGN_PATH)},
            "raw_result": {"path": str(RAW_PATH), "sha256": sha256(RAW_PATH), "bytes": RAW_PATH.stat().st_size},
            "run_manifest": {"path": str(RUN_MANIFEST_PATH), "sha256": sha256(RUN_MANIFEST_PATH)},
            "complete_marker": {
                "path": str(COMPLETE_PATH),
                "sha256": sha256(COMPLETE_PATH),
                "recorded_result_sha256": complete["result_sha256"],
            },
            "aggregation_script": {"path": str(SCRIPT_PATH.relative_to(REPOSITORY)), "sha256": sha256(SCRIPT_PATH)},
        },
        "protocol": {
            "A_late_risk_persistent": "Switch at the first source-policy risky-harvest proposal and never return.",
            "B_carry_persistent": "Switch at the first state with carried_grain > 0 and never return.",
            "C_carry_gated": "Use the adapted actor whenever carried_grain > 0 and the source actor whenever carried_grain == 0.",
            "primary": "C minus B p=0.05 expected discounted return.",
            "episode_accounting": (
                "Five seeds times 32 natural starts give 160 newly executed B paths. A and C each reuse 160 pinned paths from "
                "prior evaluations, for 480 condition-path records in the comparison."
            ),
            "gate_source": "The evaluator supplies all switching rules; no gate is learned.",
            "terms": {
                "fixed_kernel": "The normal PACK transition kernel with no overwrite firing; it exposes the deterministic path used for exact p evaluation.",
                "natural_start": "One of the 32 unmodified validation reset states used for every policy pair and condition.",
                "risky_harvest": "An executed valid harvest while carried_grain > 0, i.e. an action at which overwrite could fire.",
                "fixed_risky_start_count": "Number of the 160 fixed-kernel paths with executed_risk_count > 0.",
                "fixed_success_rate": "Goal-success fraction over the deterministic fixed-kernel paths.",
                "source_actor": "The frozen normal-training policy checkpoint in each seed-specific policy pair.",
                "adapted_actor": "The frozen p=1 overwrite-adapted policy checkpoint paired with that source actor.",
            },
        },
        "gates": compact_gates,
        "conditions": {
            mode: {
                "p00_expected_discounted_return": row["exact"]["p00"]["expected_discounted_return"],
                "p05_expected_discounted_return": row["exact"]["p05"]["expected_discounted_return"],
                "p05_success_rate": row["exact"]["p05"]["success_rate"],
                "fixed_success_rate": row["fixed_success_rate"],
                "fixed_mean_length": row["fixed_mean_length"],
                "fixed_risky_start_count": row["fixed_risky_start_count"],
                "adapted_policy_steps": row["adapted_policy_steps"],
                "total_active_steps": row["total_active_steps"],
                "adapted_step_fraction": row["adapted_step_fraction"],
            }
            for mode, row in comparison.items()
        },
        "primary_C_minus_B": primary,
        "secondary_B_minus_A": secondary,
        "per_seed": {mode: row["per_seed"] for mode, row in comparison.items()},
        "decimal_audit": {
            "domain": "Decimal, 50 digits",
            "scope": "The 160 newly executed B carry-persistent paths at p=0 and p=0.05.",
            "formula": "mean(success * 0.995^(length-1) * (1-p)^executed_risk_count)",
            "points": decimal_points,
            "tolerance": str(TOLERANCE),
            "maximum_absolute_difference": str(maximum),
            "passed": maximum <= TOLERANCE,
        },
        "interpretation": {
            "risk": (
                "B and C each had 40 fixed-kernel paths with at least one executed risky harvest. Returning control at empty inventory "
                "therefore did not reduce this measured exposure count; equal counts do not establish equal actions, paths, or broader risk."
            ),
            "retention": (
                "C exceeded B in four seeds and tied in one, but 97.0% of the mean C-minus-B difference came from seed 35, where B fixed-kernel "
                "success was 87.5%. Excluding seed 35, the mean difference was 0.000897."
            ),
            "exposure": (
                "The return rule reduced adapted-policy exposure from 84.0% to 35.8% of active steps, preserving source-policy control in empty-inventory phases."
            ),
            "secondary": (
                "B minus A was negative in the full mean because of seed 35, but positive for each of seeds 36-39. "
                "Changing the first-handoff predicate and its downstream state distribution was therefore policy-pair dependent when no return was allowed."
            ),
            "limit": (
                "C minus B isolates the supplied return rule after their common first handoff only until their trajectories diverge; "
                "it does not show that a learned selector would reproduce the result."
            ),
        },
        "non_claims": [
            "The evaluator supplied the gates; the policies did not learn when to switch.",
            "Equal risky-start counts do not imply identical trajectories or actions.",
            "The seed-35-excluded value is a declared sensitivity analysis, not the primary result.",
            "Five Dual-history learner seeds do not establish population frequency or significance.",
            "Task retention is not a latent construct here; the reported preservation measure is fixed-kernel goal success.",
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
        "claim_id": "pack_inventory_overwrite_carry_persistent_v1_finite_panel",
        "repository": {"commit": git("rev-parse", "HEAD"), "dirty": not started_clean},
        "command": [
            "PYTHONPATH=src:scripts JAX_PLATFORMS=cpu /home/ext_csv/miniconda3/envs/offrl/bin/python scripts/evaluate_pack_inventory_overwrite_carry_persistent.py --run",
            "/home/ext_csv/miniconda3/envs/offrl/bin/python scripts/summarize_pack_inventory_overwrite_carry_persistent.py --write",
        ],
        "environment": {
            "software": [f"Python {platform.python_version()}", "JAX 0.10.2 / jaxlib 0.10.2", "NumPy 2.5.0"],
            "hardware": f"CPU backend; {cpu_model}; {platform.platform()}",
        },
        "mathematics": {
            "assertion_tested": "Recompute p=0 and p=0.05 success and discounted return for 5 seeds x 32 carry-persistent fixed paths.",
            "inputs": [
                {"path": str(RAW_PATH), "sha256": sha256(RAW_PATH)},
                {"path": str(RUN_MANIFEST_PATH), "sha256": sha256(RUN_MANIFEST_PATH)},
                {"path": str(DESIGN_PATH.relative_to(REPOSITORY)), "sha256": sha256(DESIGN_PATH)},
            ],
            "bounds": {
                "learner_seeds": [35, 36, 37, 38, 39],
                "natural_starts_per_seed": 32,
                "new_fixed_paths": 160,
                "horizon_ticks": 128,
                "maximum_environment_steps": 20480,
                "evaluated_probabilities": [0.0, 0.05],
            },
            "non_claims": compact["non_claims"],
        },
        "randomness": {"used": False, "generator": "", "seed": None},
        "run": {
            "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "runtime_seconds": ended - started,
            "exit_status": 0,
        },
        "outputs": [{"path": str(COMPACT_PATH.relative_to(REPOSITORY)), "sha256": sha256(COMPACT_PATH)}],
        "checks": [
            "All 160 newly executed B fixed-kernel paths were recorded.",
            "Every new B path handed policy control to the adapted actor exactly once at positive inventory and never relinquished it.",
            "Pinned A and C results retained their endpoint-validation gates.",
            f"Independent Decimal recomputation differed by at most {compact['decimal_audit']['maximum_absolute_difference']}.",
            "All five seed contrasts and the declared seed-35-excluded sensitivity were retained."
        ],
        "result": "Implementation and finite assertion verified in the stated five-seed, 32-start panel.",
        "residual_risks": [
            "The policy network and simulator were not independently reimplemented.",
            "Intermediate-probability values rely on the pinned environment property that any overwrite loss makes the goal impossible.",
            "The supplied phase gates are not autonomous switching methods."
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
    if design["budget"]["episodes"] != 160 or not all(raw["gates"].values()):
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
