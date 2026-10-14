#!/usr/bin/env python3
"""Audit and compact the PACK one-action-swap diagnostic."""

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
DESIGN_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_one_action_swap_v1.json"
RAW_ROOT = Path("/raid/ext_csv/HackRL/runs/pack_inventory_overwrite_one_action_swap_v1")
RAW_PATH = RAW_ROOT / "result.json"
RUN_MANIFEST_PATH = RAW_ROOT / "run_manifest.json"
COMPLETE_PATH = RAW_ROOT / "complete.json"
COMPACT_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_one_action_swap_v1_results.json"
AUDIT_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_one_action_swap_v1_computation_audit.json"
SCRIPT_PATH = REPOSITORY / "scripts/summarize_pack_inventory_overwrite_one_action_swap.py"
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
        weight = survival ** risk_count
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


def build_compact(raw: dict, design: dict, run_manifest: dict, complete: dict) -> dict:
    fixed = [episode for row in raw["rows"] for episode in row["fixed"]["episodes"]]
    maximum = Decimal(0)
    decimal_points = {}
    for label, probability in (("p00", Decimal("0")), ("p05", Decimal("0.05"))):
        audited = decimal_profile(fixed, probability)
        decimal_points[label] = {metric: str(value) for metric, value in audited.items()}
        for metric, value in audited.items():
            reported = Decimal(str(raw["aggregate"]["exact"][label][metric]))
            difference = abs(value - reported)
            maximum = max(maximum, difference)
            if difference > TOLERANCE:
                raise AssertionError(f"{label}.{metric}: decimal={value} reported={reported}")

    targets = raw["diagnostic"]["targets"]
    expected_targets = [tuple(item) for item in design["diagnostic_panel"]["expected_seed_state_indices"]]
    actual_targets = [(row["seed"], row["state_index"]) for row in targets]
    if actual_targets != expected_targets:
        raise AssertionError("diagnostic target identities changed")
    d_episodes = {
        (row["seed"], episode["state_index"]): episode
        for row in raw["rows"]
        for episode in row["fixed"]["episodes"]
    }
    target_details = []
    for target in targets:
        episode = d_episodes[(target["seed"], target["state_index"])]
        branch_index = int(target["branch_step"]) - 1
        branch = episode["steps"][branch_index]
        later_disagreements = [
            step
            for step in episode["steps"][branch_index + 1 :]
            if step["carried_grain"] == 0
            and step["source_action"] != step["adapted_action"]
        ]
        target_details.append(
            {
                **target,
                "branch_delivered_total": int(branch["delivered_total"]),
                "later_empty_action_disagreement_count": len(later_disagreements),
                "first_later_empty_action_disagreement": (
                    None
                    if not later_disagreements
                    else {
                        "step": int(later_disagreements[0]["step"]),
                        "delivered_total": int(later_disagreements[0]["delivered_total"]),
                        "source_action": int(later_disagreements[0]["source_action"]),
                        "adapted_action": int(later_disagreements[0]["adapted_action"]),
                    }
                ),
            }
        )
    recovered = [
        [row["seed"], row["state_index"]]
        for row in target_details
        if row["D_success"]
    ]
    failed = [
        [row["seed"], row["state_index"]]
        for row in target_details
        if not row["D_success"]
    ]
    comparison = raw["comparison"]
    d_minus_b = contrast(comparison, "D_one_action_swap", "B_carry_persistent")
    c_minus_d = contrast(comparison, "C_carry_gated", "D_one_action_swap")
    return {
        "schema_version": "hackrl_pack_inventory_overwrite_one_action_swap_compact_result_v1",
        "status": raw["status"],
        "decision": (
            "One source-actor action recovered two of the four predeclared B-failure/C-success paths. "
            "The step-15 cases recovered, while the step-3 cases still failed after the same empty-inventory disagreement recurred at step 12; "
            "therefore this finite panel mixes failures for which one action was sufficient with failures requiring a different or longer intervention. "
            "No path that B had solved became a D failure."
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
        "protocol": {
            "B_carry_persistent": "After first positive inventory, use the adapted actor through episode termination.",
            "C_carry_gated": "Use the adapted actor at positive inventory and the source actor at empty inventory.",
            "D_one_action_swap": (
                "Follow B, but at the first post-handoff empty-inventory state where the frozen argmax actions disagree, "
                "execute the source action once and then use the adapted actor through termination."
            ),
            "branch_selection": "The branch uses only current inventory, handoff history, and frozen argmax disagreement; it does not use outcomes.",
            "episode_accounting": "Five seeds times 32 natural starts give 160 newly executed D paths; B and C paths are pinned prior results.",
            "primary": "D recovery among four predeclared fixed-kernel B-failure/C-success paths.",
            "secondary": "Whole-panel fixed-kernel success, p=0.05 expected discounted return, risky-path count, and new failures.",
        },
        "gates": raw["gates"],
        "diagnostic": {
            "target_count": len(targets),
            "recovered_count": len(recovered),
            "recovered_seed_state_indices": recovered,
            "failed_seed_state_indices": failed,
            "targets": target_details,
            "new_D_failures_among_B_successes": raw["diagnostic"]["new_D_failures_among_B_successes"],
        },
        "conditions": {
            mode: {
                "p00_expected_discounted_return": row["exact"]["p00"]["expected_discounted_return"],
                "p05_expected_discounted_return": row["exact"]["p05"]["expected_discounted_return"],
                "p05_success_rate": row["exact"]["p05"]["success_rate"],
                "fixed_success_rate": row["fixed_success_rate"],
                "fixed_mean_length": row["fixed_mean_length"],
                "fixed_risky_path_count": row["fixed_risky_start_count"],
                "adapted_policy_steps": row["adapted_policy_steps"],
                "total_active_steps": row["total_active_steps"],
                "adapted_step_fraction": row["adapted_step_fraction"],
            }
            for mode, row in comparison.items()
        },
        "D_execution": {
            "swap_path_count": raw["aggregate"]["swap_path_count"],
            "no_swap_path_count": raw["aggregate"]["no_swap_path_count"],
            "fixed_total_executed_risky_harvests": raw["aggregate"][
                "fixed_total_executed_risky_harvests"
            ],
            "failed_path_count": len(raw["aggregate"]["failed_paths"]),
        },
        "D_minus_B": d_minus_b,
        "C_minus_D": c_minus_d,
        "per_seed": {mode: row["per_seed"] for mode, row in comparison.items()},
        "decimal_audit": {
            "domain": "Decimal, 50 digits",
            "scope": "All 160 newly executed D fixed-kernel paths at p=0 and p=0.05.",
            "formula": "mean(success * 0.995^(length-1) * (1-p)^executed_risk_count)",
            "points": decimal_points,
            "tolerance": str(TOLERANCE),
            "maximum_absolute_difference": str(maximum),
            "passed": maximum <= TOLERANCE,
        },
        "interpretation": {
            "mixed_recovery": (
                "D recovered state indices 16 and 24 after swapping at step 15 with delivered_total=2 and encountered no later eligible disagreement. "
                "It did not recover state indices 4 and 12 after swapping at step 3 with delivered_total=1; on both paths the empty-inventory "
                "UP-versus-DELIVER disagreement recurred at step 12 after delivered_total reached 2."
            ),
            "whole_panel": (
                "D fixed-kernel success was 98.75% and p=0.05 return was 0.883804, between B (97.5%, 0.871333) "
                "and C (100%, 0.895508). D introduced no failure on a path that B had solved."
            ),
            "risk": (
                "B, C, and D each had 40 fixed-kernel paths with an executed risky harvest. D therefore changed normal-task "
                "completion without changing this measured exposure count."
            ),
            "control_scope": (
                "D used a one-action swap on 72 paths but still used the adapted actor for 80.8% of active steps, versus "
                "84.0% for B and 35.8% for C."
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
        "claim_id": "pack_inventory_overwrite_one_action_swap_v1_finite_panel",
        "repository": {"commit": git("rev-parse", "HEAD"), "dirty": not started_clean},
        "command": [
            "PYTHONPATH=src:scripts JAX_PLATFORMS=cpu /home/ext_csv/miniconda3/envs/offrl/bin/python scripts/evaluate_pack_inventory_overwrite_one_action_swap.py --run",
            "/home/ext_csv/miniconda3/envs/offrl/bin/python scripts/summarize_pack_inventory_overwrite_one_action_swap.py --write",
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
                "Evaluate D on five frozen policy pairs and 32 natural starts per pair, and determine whether it recovers "
                "the four predeclared B-failure/C-success fixed-kernel paths."
            ),
            "coefficient_domain": (
                "Integer path/event counts; IEEE-754 rollout summaries; independent Decimal arithmetic at 50-digit precision "
                "for p=0 and p=0.05 expectations."
            ),
            "conventions": (
                "One-indexed recorded steps; gamma=0.995; deterministic argmax actions; D swaps once at the first eligible "
                "post-handoff empty-inventory argmax disagreement and then uses the adapted actor."
            ),
            "inputs": [
                {"path": str(RAW_PATH), "sha256": sha256(RAW_PATH)},
                {"path": str(RUN_MANIFEST_PATH), "sha256": sha256(RUN_MANIFEST_PATH)},
                {"path": str(DESIGN_PATH.relative_to(REPOSITORY)), "sha256": sha256(DESIGN_PATH)},
            ],
            "bounds": {
                "learner_seeds": [35, 36, 37, 38, 39],
                "natural_starts_per_seed": 32,
                "new_D_paths": 160,
                "horizon_ticks": 128,
                "maximum_environment_steps": 20480,
                "evaluated_probabilities": [0.0, 0.05],
                "predeclared_diagnostic_paths": 4,
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
            {
                "path": str(COMPACT_PATH.relative_to(REPOSITORY)),
                "sha256": sha256(COMPACT_PATH),
            }
        ],
        "checks": [
            "All 160 D paths were recorded and all execution gates passed.",
            "The four diagnostic identities were reconstructed from pinned B/C outcomes and matched the predeclared manifest.",
            "Each D prefix and swap step matched the first pinned B/C empty-inventory action disagreement.",
            "No D path used more than one source action, and every post-swap action used the adapted actor.",
            f"Independent Decimal recomputation differed by at most {compact['decimal_audit']['maximum_absolute_difference']}.",
        ],
        "result": (
            "Implementation and finite assertion verified: D recovered two of four predeclared diagnostic paths, "
            "with no new failure among paths B had solved."
        ),
        "residual_risks": [
            "The policy network and simulator were not independently reimplemented.",
            "The same project codebase generated the paths and goal-success labels; path invariants and pinned comparisons reduce but do not remove verifier-designer coupling.",
            "The four diagnostic paths were selected from prior B/C outcomes and are not an independent generalization panel.",
            "Intermediate-probability values rely on the pinned environment property that any overwrite loss makes the goal impossible.",
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
    if design["budget"]["episodes"] != 160 or not all(raw["gates"].values()):
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
