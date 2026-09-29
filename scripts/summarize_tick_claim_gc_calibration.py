#!/usr/bin/env python3
"""Validate and aggregate the six manifest-declared calibration cells."""

import argparse
import json
from pathlib import Path


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _mean(values):
    return sum(values) / len(values) if values else None


def main():
    arguments = _arguments()
    manifest_path = Path(arguments.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_root = Path(arguments.run_root)
    cells = []
    errors = []
    for declared in manifest["cells"]:
        cell_dir = run_root / declared["directory"]
        try:
            run_manifest = json.loads(
                (cell_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (cell_dir / "summary.json").read_text(encoding="utf-8")
            )
            checkpoint = json.loads(
                (cell_dir / "checkpoint_final/metadata.json").read_text(
                    encoding="utf-8"
                )
            )
        except (FileNotFoundError, json.JSONDecodeError) as error:
            errors.append(f"{declared['id']}: {error}")
            continue
        checks = {
            "run_complete": run_manifest.get("status") == "complete",
            "variant": summary.get("variant") == declared["variant"],
            "seed": summary.get("seed") == declared["seed"],
            "transitions": summary.get("transitions")
            == manifest["transitions_per_cell"],
            "parameter_count": summary.get("parameter_count")
            == manifest["parameter_count"],
            "frozen_evaluation": summary.get("evaluation_state_immutable") is True,
            "checkpoint_steps": checkpoint.get("environment_steps")
            == manifest["transitions_per_cell"],
            "checkpoint_update": checkpoint.get("global_update")
            == manifest["updates_per_cell"],
            "code_revision": checkpoint.get("code_revision")
            == manifest["implementation_code_sha"],
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            errors.append(f"{declared['id']}: failed {', '.join(failed)}")
        cells.append(
            {
                "id": declared["id"],
                "variant": declared["variant"],
                "seed": declared["seed"],
                "checks": checks,
                "summary": summary,
            }
        )
    by_variant = {}
    for variant in ("fixed", "mutant"):
        selected = [
            cell["summary"] for cell in cells if cell["variant"] == variant
        ]
        by_variant[variant] = {
            "cells": len(selected),
            "mode_success_rate_mean": _mean(
                [
                    item["mode_evaluation"]["overall"]["success_rate"]
                    for item in selected
                ]
            ),
            "sample_success_rate_mean": _mean(
                [
                    item["sample_evaluation"]["overall"]["success_rate"]
                    for item in selected
                ]
            ),
            "total_goal_successes": sum(
                item["total_goal_successes"] for item in selected
            ),
            "total_violation_events": sum(
                item["total_violation_events"] for item in selected
            ),
            "total_repeated_violation_events": sum(
                item["total_repeated_violation_events"] for item in selected
            ),
            "total_opportunity_exposures": sum(
                item["total_opportunity_exposures"] for item in selected
            ),
            "total_reservation_creations": sum(
                item["total_reservation_creations"] for item in selected
            ),
            "total_violation_grain_delivered": sum(
                item["total_violation_grain_delivered"] for item in selected
            ),
        }
    result = {
        "schema_version": "tick_claim_gc_calibration_aggregate_v1",
        "manifest": str(manifest_path),
        "status": "passed" if not errors and len(cells) == 6 else "failed",
        "errors": errors,
        "cell_count": len(cells),
        "transitions": sum(
            cell["summary"]["transitions"] for cell in cells
        ),
        "by_variant": by_variant,
        "cells": cells,
        "interpretation_limit": (
            "wiring calibration on one synthetic mechanism and related D4/layout "
            "variants; not evidence of general bug-discovery ability"
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
