#!/usr/bin/env python3
"""Aggregate E, D, V, U, and paired BC effects for the scale study."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_dual_leo_compare import (
    SCIENCE_UPDATES,
    _cell_dir,
    _checkpoint_fingerprint,
    _curve,
)
from run_pack_restore_scale_bc import SEEDS, SIZE_SPECS, build_jobs

CONDITIONS = ("G", "D-on", "D-off")
POLICIES = ("mode", "sample")
FAMILIES = ("natural_reset", "common_setup")
EXPLOIT_METRIC = "violation_delivery_rate"
NORMAL_METRICS = ("success_rate", "mean_length", "mean_discounted_return")
VALUE_METRICS = ("success_rate", "mean_length", "mean_discounted_return")
BOOTSTRAP_RESAMPLES = 20_000


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _bootstrap_summary(values, rng):
    values = np.asarray(values, dtype=np.float64)
    result = {
        "n_seeds": int(values.size),
        "mean": float(np.mean(values)),
        "sample_standard_deviation": (
            float(np.std(values, ddof=1)) if values.size > 1 else None
        ),
    }
    if values.size == 1:
        low = high = float(values[0])
    else:
        draws = rng.choice(values, size=(BOOTSTRAP_RESAMPLES, values.size), replace=True)
        low, high = np.quantile(np.mean(draws, axis=1), (0.025, 0.975))
    result["bootstrap_95_percentile_interval"] = [float(low), float(high)]
    return result


def _summaries(points, group_fields):
    grouped = defaultdict(list)
    for point in points:
        grouped[tuple(point[name] for name in group_fields)].append(point["value"])
    rng = np.random.default_rng(20261003)
    rows = []
    for key in sorted(grouped, key=repr):
        rows.append(
            {name: value for name, value in zip(group_fields, key)}
            | _bootstrap_summary(grouped[key], rng)
        )
    return rows


def _adaptation_jobs():
    return [job for job in build_jobs() if job["kind"] == "adapt"]


def _shared_dual_origin_checks(origins):
    checks = []
    for spec in SIZE_SPECS:
        for seed in SEEDS:
            expected = {
                (spec["size"], condition, seed, variant)
                for condition in ("D-on", "D-off")
                for variant in ("fixed", "mutant")
            }
            records = [origins[key] for key in expected if key in origins]
            fingerprints = {
                record.get("source_checkpoint_sha256") for record in records
            }
            checks.append(
                {
                    "size": spec["size"],
                    "seed": seed,
                    "expected_branches": 4,
                    "observed_branches": len(records),
                    "source_checkpoint_sha256": (
                        next(iter(fingerprints)) if len(fingerprints) == 1 else None
                    ),
                    "passed": len(records) == 4
                    and len(fingerprints) == 1
                    and None not in fingerprints,
                }
            )
    return checks


def collect(run_root):
    """Read curve artifacts without treating evaluation episodes as replicates."""

    curves = {}
    missing = []
    errors = []
    parameter_checks = []
    origins = {}
    verified_sources = {}
    expected_by_size = {
        spec["size"]: spec["expected_ppo_parameters"] for spec in SIZE_SPECS
    }
    for job in _adaptation_jobs():
        cell = _cell_dir(run_root, job)
        origin_path = cell / "origin.json"
        if origin_path.is_file():
            try:
                origin = _read(origin_path)
                key = (job["size"], job["condition"], job["seed"], job["variant"])
                if (
                    origin.get("size") != job["size"]
                    or origin.get("condition") != job["condition"]
                    or origin.get("seed") != job["seed"]
                    or origin.get("variant") != job["variant"]
                    or not origin.get("source_checkpoint_sha256")
                ):
                    raise ValueError("origin does not match adaptation job")
                source = str(Path(origin["source_checkpoint"]).resolve())
                if source not in verified_sources:
                    verified_sources[source] = _checkpoint_fingerprint(source)
                actual = verified_sources[source]
                if (
                    origin.get("source_checkpoint_sha256") != actual["sha256"]
                    or origin.get("source_checkpoint_files") != actual["files"]
                ):
                    raise ValueError("origin fingerprint does not match checkpoint")
                origins[key] = origin
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
                errors.append(f"{origin_path}: {error}")
        else:
            errors.append(f"missing origin: {origin_path}")
        summary_path = cell / "summary.json"
        if summary_path.is_file():
            try:
                summary = _read(summary_path)
                parameter_checks.append(
                    {
                        "job": job["id"],
                        "ppo_parameters": summary.get("ppo_parameters"),
                        "expected_ppo_parameters": expected_by_size[job["size"]],
                        "teacher_parameters": summary.get("teacher_parameters"),
                        "passed": (
                            summary.get("ppo_parameters")
                            == expected_by_size[job["size"]]
                            and (
                                job["condition"] == "G"
                                or summary.get("teacher_parameters") == 1_469_366
                            )
                        ),
                    }
                )
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"{summary_path}: {error}")
        else:
            errors.append(f"missing summary: {summary_path}")
        for update in SCIENCE_UPDATES:
            path = _curve(cell, update)
            if not path.is_file():
                missing.append({"job": job["id"], "adaptation_update": update})
                continue
            try:
                document = _read(path)
                if int(document.get("adaptation_updates", -1)) != update:
                    raise ValueError("adaptation update does not match path")
                curves[(job["size"], job["condition"], job["seed"], job["variant"], update)] = document
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                errors.append(f"{path}: {error}")
    return (
        curves,
        missing,
        errors,
        parameter_checks,
        origins,
        _shared_dual_origin_checks(origins),
    )


def _block(
    curves, size, condition, seed, variant, update, dynamics, policy, family
):
    document = curves.get((size, condition, seed, variant, update))
    if document is None:
        return None
    try:
        return document[dynamics][policy][family]
    except (KeyError, TypeError):
        return None


def compute_outputs(curves):
    u_points = []
    normal_points = []
    value_points = []
    timing = []
    for spec in SIZE_SPECS:
        size = spec["size"]
        for condition in CONDITIONS:
            for seed in SEEDS:
                for policy in POLICIES:
                    for family in FAMILIES:
                        observed_updates = []
                        available_updates = []
                        for update in SCIENCE_UPDATES:
                            fixed = _block(
                                curves,
                                size,
                                condition,
                                seed,
                                "fixed",
                                update,
                                "mutant",
                                policy,
                                family,
                            )
                            mutant = _block(
                                curves,
                                size,
                                condition,
                                seed,
                                "mutant",
                                update,
                                "mutant",
                                policy,
                                family,
                            )
                            if fixed is not None and mutant is not None:
                                value = float(mutant[EXPLOIT_METRIC]) - float(
                                    fixed[EXPLOIT_METRIC]
                                )
                                u_points.append(
                                    {
                                        "size": size,
                                        "condition": condition,
                                        "seed": seed,
                                        "adaptation_update": update,
                                        "policy": policy,
                                        "family": family,
                                        "value": value,
                                    }
                                )
                            if mutant is not None and float(mutant[EXPLOIT_METRIC]) > 0:
                                observed_updates.append(update)
                            if mutant is not None:
                                available_updates.append(update)

                            normal = _block(
                                curves,
                                size,
                                condition,
                                seed,
                                "fixed",
                                update,
                                "fixed",
                                policy,
                                family,
                            )
                            if normal is not None:
                                for metric in NORMAL_METRICS:
                                    normal_points.append(
                                        {
                                            "size": size,
                                            "condition": condition,
                                            "seed": seed,
                                            "adaptation_update": update,
                                            "policy": policy,
                                            "family": family,
                                            "metric": metric,
                                            "value": float(normal[metric]),
                                        }
                                    )

                            fixed_kernel = _block(
                                curves,
                                size,
                                condition,
                                seed,
                                "mutant",
                                update,
                                "fixed",
                                policy,
                                family,
                            )
                            mutant_kernel = _block(
                                curves,
                                size,
                                condition,
                                seed,
                                "mutant",
                                update,
                                "mutant",
                                policy,
                                family,
                            )
                            if fixed_kernel is not None and mutant_kernel is not None:
                                for metric in VALUE_METRICS:
                                    value_points.append(
                                        {
                                            "size": size,
                                            "condition": condition,
                                            "seed": seed,
                                            "adaptation_update": update,
                                            "policy": policy,
                                            "family": family,
                                            "metric": metric,
                                            "value": float(mutant_kernel[metric])
                                            - float(fixed_kernel[metric]),
                                        }
                                    )
                        timing.append(
                            {
                                "size": size,
                                "condition": condition,
                                "seed": seed,
                                "policy": policy,
                                "family": family,
                                "observed_updates": observed_updates,
                                "available_updates": available_updates,
                                "first_observed_update": (
                                    observed_updates[0] if observed_updates else None
                                ),
                                "last_observed_update": (
                                    observed_updates[-1] if observed_updates else None
                                ),
                                "observed_at_final": 4096 in observed_updates,
                                "lost_after_first_observation": (
                                    None
                                    if available_updates != list(SCIENCE_UPDATES)
                                    else bool(observed_updates)
                                    and observed_updates
                                    != [
                                        item
                                        for item in SCIENCE_UPDATES
                                        if item >= observed_updates[0]
                                    ]
                                ),
                            }
                        )

    u_lookup = {
        (
            point["size"],
            point["condition"],
            point["seed"],
            point["adaptation_update"],
            point["policy"],
            point["family"],
        ): point["value"]
        for point in u_points
    }
    delta_points = []
    for size in (spec["size"] for spec in SIZE_SPECS):
        for seed in SEEDS:
            for update in SCIENCE_UPDATES:
                for policy in POLICIES:
                    for family in FAMILIES:
                        on = u_lookup.get((size, "D-on", seed, update, policy, family))
                        off = u_lookup.get((size, "D-off", seed, update, policy, family))
                        if on is not None and off is not None:
                            delta_points.append(
                                {
                                    "size": size,
                                    "seed": seed,
                                    "adaptation_update": update,
                                    "policy": policy,
                                    "family": family,
                                    "value": on - off,
                                }
                            )

    u_groups = ("size", "condition", "adaptation_update", "policy", "family")
    delta_groups = ("size", "adaptation_update", "policy", "family")
    axis_groups = (
        "size",
        "condition",
        "adaptation_update",
        "policy",
        "family",
        "metric",
    )

    def primary(point):
        return point["policy"] == "mode" and point["family"] == "natural_reset"

    return {
        "U": {
            "definition": "mutant-adapted minus fixed-continued violation_delivery_rate, both evaluated on the mutant kernel",
            "seed_points": u_points,
            "summaries": _summaries(u_points, u_groups),
            "primary_seed_points": [point for point in u_points if primary(point)],
        },
        "Delta_BC": {
            "definition": "paired U_D-on minus U_D-off within size, seed, update, policy, and start family",
            "seed_points": delta_points,
            "summaries": _summaries(delta_points, delta_groups),
            "primary_seed_points": [point for point in delta_points if primary(point)],
        },
        "E_normal_performance": {
            "definition": "fixed-continued policy evaluated on the fixed kernel",
            "seed_points": normal_points,
            "summaries": _summaries(normal_points, axis_groups),
        },
        "D_exploitation_timing": timing,
        "V_realized_gain": {
            "definition": "same mutant-adapted policy evaluated on mutant minus fixed kernel",
            "seed_points": value_points,
            "summaries": _summaries(value_points, axis_groups),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    run_root = Path(arguments.run_root)
    (
        curves,
        missing,
        errors,
        parameter_checks,
        origins,
        origin_checks,
    ) = collect(run_root)
    outputs = compute_outputs(curves)
    failed_parameter_checks = [row for row in parameter_checks if not row["passed"]]
    failed_origin_checks = [row for row in origin_checks if not row["passed"]]
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest = None
    if run_manifest_path.is_file():
        try:
            run_manifest = _read(run_manifest_path)
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{run_manifest_path}: {error}")
    else:
        errors.append(f"missing run manifest: {run_manifest_path}")
    status = (
        "complete"
        if not missing
        and not errors
        and not failed_parameter_checks
        and not failed_origin_checks
        and len(parameter_checks) == 180
        and len(origins) == 180
        and len(origin_checks) == 30
        and len(curves) == 180 * len(SCIENCE_UPDATES)
        else "incomplete"
    )
    result = {
        "schema_version": "hackrl_pack_restore_scale_bc_results_v1",
        "status": status,
        "run_root": str(arguments.run_root),
        "expected_adaptation_curves": 180 * len(SCIENCE_UPDATES),
        "loaded_adaptation_curves": len(curves),
        "missing": missing,
        "errors": errors,
        "parameter_checks": parameter_checks,
        "failed_parameter_checks": failed_parameter_checks,
        "run_provenance": run_manifest,
        "adaptation_origins": list(origins.values()),
        "shared_dual_pretraining_checks": origin_checks,
        "failed_shared_dual_pretraining_checks": failed_origin_checks,
        "inference_unit": "learner seed; evaluation episodes are not treated as independent repetitions",
        "bootstrap": {
            "method": "learner-seed resampling of the mean",
            "resamples": BOOTSTRAP_RESAMPLES,
            "rng_seed": 20261003,
            "interval": "95% percentile"
        },
        **outputs,
    }
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "loaded_adaptation_curves": len(curves),
                "missing_count": len(missing),
                "error_count": len(errors),
                "failed_parameter_checks": len(failed_parameter_checks),
                "failed_shared_dual_pretraining_checks": len(
                    failed_origin_checks
                ),
                "output": str(destination),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if arguments.require_complete and status != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
