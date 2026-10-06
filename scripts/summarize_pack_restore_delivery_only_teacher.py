#!/usr/bin/env python3
"""Validate and aggregate the S-policy pretrained-delivery-only teacher experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_dual_leo_compare import SCIENCE_UPDATES, _cell_dir, _checkpoint_fingerprint
from run_pack_restore_delivery_only_teacher import (
    ADAPT_UPDATES,
    EXPECTED_SOURCE_EXECUTION_SHA,
    EXPECTED_SOURCE_FINGERPRINTS,
    EXPECTED_SOURCE_MANIFEST_SHA256,
    MANIFEST_PATH,
    REPOSITORY,
    SEEDS,
    SOURCE_RUN_ROOT,
    build_jobs,
    source_checkpoint,
)


CONDITIONS = ("D-delivery", "D-on")
POLICIES = ("mode", "sample")
FAMILIES = ("natural_reset", "common_setup")
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20261006


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _baseline_cell(condition, seed, variant):
    method = {"D-on": "dual"}[condition]
    return SOURCE_RUN_ROOT / "size_s" / "pack" / method / variant / f"seed{seed}"


def _new_cell(run_root, seed, variant):
    job = next(
        job
        for job in build_jobs()
        if job["seed"] == seed and job["variant"] == variant
    )
    return _cell_dir(run_root, job)


def _curve_cell(run_root, condition, seed, variant):
    if condition == "D-delivery":
        return _new_cell(run_root, seed, variant)
    return _baseline_cell(condition, seed, variant)


def _bootstrap(values, rng):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n_seeds": 0, "mean": None, "bootstrap_95_percentile_interval": None}
    if values.size == 1:
        interval = [float(values[0]), float(values[0])]
    else:
        draws = rng.choice(
            values, size=(BOOTSTRAP_RESAMPLES, values.size), replace=True
        )
        interval = [
            float(value)
            for value in np.quantile(np.mean(draws, axis=1), (0.025, 0.975))
        ]
    return {
        "n_seeds": int(values.size),
        "mean": float(np.mean(values)),
        "sample_standard_deviation": (
            float(np.std(values, ddof=1)) if values.size > 1 else None
        ),
        "bootstrap_95_percentile_interval": interval,
    }


def _validate_provenance(
    run_root, expected_execution_sha, expected_manifest_sha256, errors
):
    try:
        manifest = _read(REPOSITORY / MANIFEST_PATH)
        if _sha256(REPOSITORY / MANIFEST_PATH) != expected_manifest_sha256:
            errors.append("experiment manifest differs from authorized digest")
        for relative, expected in manifest.get("authorized_source_sha256", {}).items():
            if _sha256(REPOSITORY / relative) != expected:
                errors.append(f"authorized source hash mismatch: {relative}")
        run_manifest = _read(Path(run_root) / "run_manifest.json")
        if run_manifest.get("execution_code_sha") != expected_execution_sha:
            errors.append("run execution SHA mismatch")
        if run_manifest.get("manifest_sha256") != expected_manifest_sha256:
            errors.append("run manifest digest mismatch")
        if run_manifest.get("job_count") != 20:
            errors.append("run manifest job count mismatch")
        source = _read(SOURCE_RUN_ROOT / "run_manifest.json")
        if source.get("execution_code_sha") != EXPECTED_SOURCE_EXECUTION_SHA:
            errors.append("baseline execution SHA mismatch")
        if (
            source.get("execution_source_sha256", {}).get(
                "docs/manifests/pack_restore_scale_bc_v1.json"
            )
            != EXPECTED_SOURCE_MANIFEST_SHA256
        ):
            errors.append("baseline manifest digest mismatch")
    except (OSError, json.JSONDecodeError, KeyError) as error:
        errors.append(f"provenance validation failed: {error}")


def _collect(run_root):
    curves = {}
    seed_checks = []
    errors = []
    missing = []
    for seed in SEEDS:
        actual_fingerprint = _checkpoint_fingerprint(source_checkpoint(seed))["sha256"]
        if actual_fingerprint != EXPECTED_SOURCE_FINGERPRINTS[seed]:
            errors.append(f"seed {seed}: source checkpoint fingerprint changed")
        for variant in ("fixed", "mutant"):
            new_cell = _new_cell(run_root, seed, variant)
            try:
                summary = _read(new_cell / "summary.json")
                origin = _read(new_cell / "origin.json")
                delivery = _read(new_cell / "delivery_only_verification.json")
                passed = (
                    summary.get("condition") == "D-delivery"
                    and summary.get("seed") == seed
                    and summary.get("variant") == variant
                    and summary.get("global_update") == 4608
                    and summary.get("adaptation_updates") == ADAPT_UPDATES
                    and summary.get("learn_teacher") is True
                    and summary.get("imitate_teacher") is True
                    and summary.get("teacher_goal_indices") == [11]
                    and delivery.get("passed") is True
                    and all(delivery.get("checks", {}).values())
                    and origin.get("source_checkpoint_sha256")
                    == EXPECTED_SOURCE_FINGERPRINTS[seed]
                    and Path(origin.get("source_checkpoint", "")).resolve()
                    == source_checkpoint(seed).resolve()
                )
                if not passed:
                    errors.append(f"seed {seed} {variant}: delivery-only arm verification failed")
                seed_checks.append(
                    {"seed": seed, "variant": variant, "passed": passed}
                )
            except (OSError, json.JSONDecodeError, TypeError) as error:
                errors.append(f"seed {seed} {variant}: invalid new artifact: {error}")
        for condition in CONDITIONS:
            for variant in ("fixed", "mutant"):
                cell = _curve_cell(run_root, condition, seed, variant)
                for update in SCIENCE_UPDATES:
                    path = cell / "curve" / f"adapt_{update}.json"
                    if not path.is_file():
                        missing.append(
                            {
                                "condition": condition,
                                "seed": seed,
                                "variant": variant,
                                "adaptation_update": update,
                            }
                        )
                        continue
                    try:
                        document = _read(path)
                        if (
                            document.get("condition") != condition
                            or document.get("size") != "S"
                            or document.get("seed") != seed
                            or document.get("trained_variant") != variant
                            or document.get("adaptation_updates") != update
                        ):
                            raise ValueError("curve identity mismatch")
                        curves[(condition, seed, variant, update)] = document
                    except (OSError, json.JSONDecodeError, ValueError) as error:
                        errors.append(f"{path}: {error}")
    return curves, seed_checks, missing, errors


def _block(curves, condition, seed, variant, update, kernel, policy, family):
    try:
        return curves[(condition, seed, variant, update)][kernel][policy][family]
    except (KeyError, TypeError):
        return None


def _compute(curves):
    u_points = []
    normal_points = []
    value_points = []
    first_observed = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            for policy in POLICIES:
                for family in FAMILIES:
                    observed = []
                    for update in SCIENCE_UPDATES:
                        fixed_on_mutant = _block(
                            curves, condition, seed, "fixed", update,
                            "mutant", policy, family,
                        )
                        mutant_on_mutant = _block(
                            curves, condition, seed, "mutant", update,
                            "mutant", policy, family,
                        )
                        mutant_on_fixed = _block(
                            curves, condition, seed, "mutant", update,
                            "fixed", policy, family,
                        )
                        fixed_on_fixed = _block(
                            curves, condition, seed, "fixed", update,
                            "fixed", policy, family,
                        )
                        if fixed_on_mutant is not None and mutant_on_mutant is not None:
                            value = float(
                                mutant_on_mutant["violation_delivery_rate"]
                            ) - float(fixed_on_mutant["violation_delivery_rate"])
                            u_points.append(
                                {
                                    "condition": condition,
                                    "seed": seed,
                                    "adaptation_update": update,
                                    "policy": policy,
                                    "family": family,
                                    "value": value,
                                }
                            )
                            if float(mutant_on_mutant["violation_delivery_rate"]) > 0:
                                observed.append(update)
                        if fixed_on_fixed is not None:
                            for metric in (
                                "success_rate", "mean_length", "mean_discounted_return"
                            ):
                                normal_points.append(
                                    {
                                        "condition": condition,
                                        "seed": seed,
                                        "adaptation_update": update,
                                        "policy": policy,
                                        "family": family,
                                        "metric": metric,
                                        "value": float(fixed_on_fixed[metric]),
                                    }
                                )
                        if mutant_on_mutant is not None and mutant_on_fixed is not None:
                            for metric in (
                                "success_rate", "mean_length", "mean_discounted_return"
                            ):
                                value_points.append(
                                    {
                                        "condition": condition,
                                        "seed": seed,
                                        "adaptation_update": update,
                                        "policy": policy,
                                        "family": family,
                                        "metric": metric,
                                        "value": float(mutant_on_mutant[metric])
                                        - float(mutant_on_fixed[metric]),
                                    }
                                )
                    first_observed.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "policy": policy,
                            "family": family,
                            "first_saved_update_with_violation_delivery": (
                                min(observed) if observed else None
                            ),
                        }
                    )

    lookup = {
        (
            row["condition"], row["seed"], row["adaptation_update"],
            row["policy"], row["family"],
        ): row["value"]
        for row in u_points
    }
    paired = []
    for left, right, label in (
        ("D-on", "D-delivery", "D-on_minus_D-delivery"),
    ):
        for seed in SEEDS:
            for update in SCIENCE_UPDATES:
                for policy in POLICIES:
                    for family in FAMILIES:
                        left_key = (left, seed, update, policy, family)
                        right_key = (right, seed, update, policy, family)
                        if left_key in lookup and right_key in lookup:
                            paired.append(
                                {
                                    "contrast": label,
                                    "seed": seed,
                                    "adaptation_update": update,
                                    "policy": policy,
                                    "family": family,
                                    "value": lookup[left_key] - lookup[right_key],
                                }
                            )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    grouped = defaultdict(list)
    for row in u_points:
        grouped[(
            row["condition"], row["adaptation_update"], row["policy"], row["family"]
        )].append(row["value"])
    u_summaries = [
        {
            "condition": key[0],
            "adaptation_update": key[1],
            "policy": key[2],
            "family": key[3],
            **_bootstrap(values, rng),
        }
        for key, values in sorted(grouped.items(), key=repr)
    ]
    paired_grouped = defaultdict(list)
    for row in paired:
        paired_grouped[(
            row["contrast"], row["adaptation_update"], row["policy"], row["family"]
        )].append(row["value"])
    paired_summaries = [
        {
            "contrast": key[0],
            "adaptation_update": key[1],
            "policy": key[2],
            "family": key[3],
            **_bootstrap(values, rng),
        }
        for key, values in sorted(paired_grouped.items(), key=repr)
    ]
    primary_points = [
        row for row in u_points
        if row["adaptation_update"] == ADAPT_UPDATES
        and row["policy"] == "mode"
        and row["family"] == "natural_reset"
    ]
    primary_paired = [
        row for row in paired
        if row["adaptation_update"] == ADAPT_UPDATES
        and row["policy"] == "mode"
        and row["family"] == "natural_reset"
    ]
    return {
        "u_seed_points": u_points,
        "u_summaries": u_summaries,
        "paired_seed_points": paired,
        "paired_summaries": paired_summaries,
        "normal_performance_seed_points": normal_points,
        "same_policy_kernel_gain_seed_points": value_points,
        "first_saved_use": first_observed,
        "primary_final_mode_natural": {
            "u_seed_points": primary_points,
            "paired_seed_points": primary_paired,
        },
    }


def summarize(
    run_root,
    *,
    expected_execution_sha,
    expected_manifest_sha256,
    preflight_audit=None,
):
    run_root = Path(run_root).resolve()
    errors = []
    _validate_provenance(
        run_root, expected_execution_sha, expected_manifest_sha256, errors
    )
    curves, seed_checks, missing, collection_errors = _collect(run_root)
    errors.extend(collection_errors)
    audit_record = None
    if preflight_audit is not None:
        path = Path(preflight_audit)
        try:
            audit = _read(path)
            if (
                audit.get("schema_version")
                != "hackrl_pack_restore_teacher_evaluation_audit_v1"
                or audit.get("passed") is not True
            ):
                raise ValueError("preflight evaluation audit did not pass")
            audit_record = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "passed": True,
                "training_or_optimizer_updates": audit.get("training_or_optimizer_updates"),
            }
        except (OSError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"preflight audit invalid: {error}")
    execution_complete = (
        not errors
        and not missing
        and len(curves) == len(CONDITIONS) * len(SEEDS) * 2 * len(SCIENCE_UPDATES)
        and len(seed_checks) == len(SEEDS) * 2
        and all(row["passed"] for row in seed_checks)
        and audit_record is not None
    )
    return {
        "schema_version": "hackrl_pack_restore_delivery_only_teacher_result_v1",
        "experiment_id": "pack_restore_delivery_only_teacher_v1",
        "execution_complete": execution_complete,
        "errors": errors,
        "missing_curves": missing,
        "validated_curve_count": len(curves),
        "delivery_only_teacher_checks": seed_checks,
        "preflight_evaluation_audit": audit_record,
        "statistics": {
            "independent_unit": "learner_seed",
            "episode_count_is_not_a_replication_unit": True,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "interpretation_limits": [
            "The same seeds and pretraining checkpoints are reused; this is a paired retrospective mechanism experiment, not an independent replication.",
            "D-on is a retrospective baseline from an earlier execution SHA and wall-clock period. Matching source hashes narrows but does not eliminate every runtime or dependency confound.",
            "Both conditions start from the same multi-goal-pretrained teacher, so this tests adaptation-time target restriction, not the necessity of multi-goal pretraining.",
            "The delivery-only arm has the same optimizer steps but one target head instead of twelve; target-term count and shared-layer gradient aggregation differ by design.",
            "Later visitation and policy states diverge, so the contrast is the full mediated effect of the adaptation-time teacher objective.",
            "Normal success, conservation violation, violation delivery, and same-policy kernel return gain remain separate outcomes.",
            "Claims are limited to the encoded PACK-RESTORE fixture and tested optimization contract.",
        ],
        "outputs": _compute(curves) if not missing else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-execution-sha", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--preflight-audit", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = summarize(
        arguments.run_root,
        expected_execution_sha=arguments.expected_execution_sha,
        expected_manifest_sha256=arguments.expected_manifest_sha256,
        preflight_audit=arguments.preflight_audit,
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps({
        "execution_complete": result["execution_complete"],
        "errors": result["errors"],
        "missing_curve_count": len(result["missing_curves"]),
        "output": str(output.resolve()),
    }, sort_keys=True))
    if not result["execution_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
