#!/usr/bin/env python3
"""Validate and aggregate the six frozen GC calibration cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--resolved", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean_and_sample_std(values):
    return {
        "values_by_seed": values,
        "mean": statistics.mean(values),
        "sample_standard_deviation": (
            statistics.stdev(values) if len(values) > 1 else None
        ),
    }


def _cell_id(variant, seed):
    return f"{variant}_seed{seed}"


def main():
    arguments = _arguments()
    manifest_path = Path(arguments.manifest)
    resolved_path = Path(arguments.resolved)
    manifest = _read(manifest_path)
    resolved = _read(resolved_path)
    run_root = Path(arguments.run_root)
    expected_transitions = manifest["budget"]["transitions_per_cell"]
    expected_updates = manifest["budget"]["updates_per_cell"]
    expected_parameters = manifest["model"]["parameter_count_gate"]
    expected_revision = resolved["implementation_code_sha"]
    cells = []
    errors = []
    for declared in manifest["cells"]:
        variant = declared["variant"]
        seed = declared["learner_seed"]
        identifier = _cell_id(variant, seed)
        cell_dir = run_root / identifier
        try:
            run_manifest_path = cell_dir / "run_manifest.json"
            summary_path = cell_dir / "summary.json"
            updates_path = cell_dir / "updates.json"
            checkpoint_path = cell_dir / "checkpoint_final" / "metadata.json"
            run_manifest = _read(run_manifest_path)
            summary = _read(summary_path)
            checkpoint = _read(checkpoint_path)
        except (FileNotFoundError, json.JSONDecodeError) as error:
            errors.append(f"{identifier}: {error}")
            continue
        checks = {
            "run_complete": run_manifest.get("status") == "complete",
            "variant": summary.get("variant") == variant,
            "seed": summary.get("seed") == seed,
            "transitions": summary.get("transitions") == expected_transitions,
            "parameter_count": summary.get("parameter_count")
            == expected_parameters,
            "frozen_evaluation": summary.get("evaluation_state_immutable")
            is True,
            "checkpoint_steps": checkpoint.get("environment_steps")
            == expected_transitions,
            "checkpoint_update": checkpoint.get("global_update")
            == expected_updates,
            "code_revision": checkpoint.get("code_revision")
            == expected_revision,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            errors.append(f"{identifier}: failed {', '.join(failed)}")
        cells.append(
            {
                "id": identifier,
                "variant": variant,
                "learner_seed": seed,
                "checks": checks,
                "artifact_hashes": {
                    "run_manifest": _sha256(run_manifest_path),
                    "summary": _sha256(summary_path),
                    "updates": _sha256(updates_path),
                    "checkpoint_metadata": _sha256(checkpoint_path),
                },
                "summary": summary,
            }
        )

    scalar_metrics = {
        "mode_success_rate": lambda item: item["mode_evaluation"]["overall"][
            "success_rate"
        ],
        "sample_success_rate": lambda item: item["sample_evaluation"][
            "overall"
        ]["success_rate"],
        "sample_violation_rate": lambda item: item["sample_evaluation"][
            "overall"
        ]["violation_rate"],
        "sample_repeated_violation_rate": lambda item: item[
            "sample_evaluation"
        ]["overall"]["repeated_violation_rate"],
        "sample_violation_delivery_rate": lambda item: item[
            "sample_evaluation"
        ]["overall"]["violation_delivery_rate"],
        "sample_mean_length": lambda item: item["sample_evaluation"][
            "overall"
        ]["mean_length"],
    }
    by_variant = {}
    for variant in ("fixed", "mutant"):
        selected = sorted(
            (cell for cell in cells if cell["variant"] == variant),
            key=lambda item: item["learner_seed"],
        )
        by_variant[variant] = {
            name: _mean_and_sample_std(
                [extract(cell["summary"]) for cell in selected]
            )
            for name, extract in scalar_metrics.items()
        }
        by_variant[variant]["training_totals_by_seed"] = {
            cell["learner_seed"]: {
                key: cell["summary"][key]
                for key in (
                    "total_goal_successes",
                    "total_opportunity_exposures",
                    "total_reservation_creations",
                    "total_violation_events",
                    "total_repeated_violation_events",
                    "total_violation_grain_delivered",
                )
            }
            for cell in selected
        }

    by_id = {cell["id"]: cell for cell in cells}
    paired = {}
    for name, extract in scalar_metrics.items():
        contrasts = {
            seed: extract(by_id[_cell_id("mutant", seed)]["summary"])
            - extract(by_id[_cell_id("fixed", seed)]["summary"])
            for seed in (0, 1, 2)
            if _cell_id("mutant", seed) in by_id
            and _cell_id("fixed", seed) in by_id
        }
        paired[name] = _mean_and_sample_std(list(contrasts.values()))
        paired[name]["mutant_minus_fixed_by_seed"] = contrasts

    result = {
        "schema_version": "tick_claim_gc_calibration_aggregate_v1",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "resolved": str(resolved_path),
        "resolved_sha256": _sha256(resolved_path),
        "implementation_code_sha": expected_revision,
        "status": (
            "passed"
            if not errors and len(cells) == manifest["budget"]["cell_count"]
            else "failed"
        ),
        "errors": errors,
        "cell_count": len(cells),
        "transitions": sum(cell["summary"]["transitions"] for cell in cells),
        "by_variant": by_variant,
        "paired_mutant_minus_fixed": paired,
        "cells": cells,
        "interpretation_limit": (
            "descriptive wiring calibration on one synthetic mechanism and "
            "related D4/layout variants; not evidence of general bug discovery"
        ),
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
