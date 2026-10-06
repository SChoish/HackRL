#!/usr/bin/env python3
"""Read-only audit of PACK teacher evaluation before the next training run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import distrax
import jax
import jax.numpy as jnp
import numpy as np

from analyze_pack_restore_common_state_teachers import (
    EVALUATORS,
    FREEZE_RUN_ROOT,
    SEEDS,
    SOURCE_RUN_ROOT,
    _checkpoint,
)
from hackrl.dual_leo import init_dual_leo_teacher, load_dual_checkpoint
from hackrl.pack_restore import PackRestoreAction, PackRestoreVariant
from hackrl.pack_restore_gc import (
    DELIVER_3_GOAL_INDEX,
    NUM_ACTIONS,
    NUM_GOALS,
    _batch_inputs,
    config_from_pack_restore_gc_payload,
    evaluate_pack_restore_gc_frozen,
    initialize_pack_restore_gc,
)


COMMON_RESULT = FREEZE_RUN_ROOT / "common_state_teacher_analysis.json"
COMMON_STATES = FREEZE_RUN_ROOT / "common_state_teacher_analysis_states.npz"
PRIOR_RESULT = FREEZE_RUN_ROOT / "result.json"
EXPECTED_COMMON_RESULT_SHA256 = (
    "3033e2d70571a87df3d1493689ce526ef821aec109eabfdd150035fe1308ea36"
)
EXPECTED_COMMON_STATES_SHA256 = (
    "0b070196af0d9e36ce8cfc0d0acf8173cab358933347e46ee4aaa679caded48b"
)
SCIENCE_UPDATES = (0, 32, 128, 256, 512, 1024, 2048, 4096)
FAMILIES = ("natural_reset", "common_setup")
POLICIES = ("mode", "sample")
CURVE_CONDITIONS = ("D-on", "D-off", "D-frozen")
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


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _checkpoint_state_sha256(checkpoint):
    return _sha256(Path(checkpoint) / "state.msgpack")


def _bootstrap(values, rng):
    values = np.asarray(values, dtype=np.float64)
    draws = rng.choice(
        values, size=(BOOTSTRAP_RESAMPLES, values.size), replace=True
    )
    means = np.mean(draws, axis=1)
    return {
        "n_seeds": int(values.size),
        "mean": float(np.mean(values)),
        "sample_standard_deviation": float(np.std(values, ddof=1)),
        "bootstrap_95_percentile_interval": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


def _templates():
    reference = _checkpoint("D-frozen", SEEDS[0])
    config = config_from_pack_restore_gc_payload(
        _read(reference / "config.json")
    )
    network, template = initialize_pack_restore_gc(config)
    inputs = _batch_inputs(template.env_state, template.current_goal)
    teacher_network, leo_template, _ = init_dual_leo_teacher(
        config, inputs[0], inputs[1], NUM_GOALS, NUM_ACTIONS
    )
    return config, network, template, teacher_network, leo_template


def _teacher_q(teacher_network, leo, maps, numeric):
    return np.asarray(
        jax.device_get(
            teacher_network.apply(
                {"params": leo.params, "batch_stats": leo.batch_stats},
                jnp.asarray(maps),
                jnp.asarray(numeric),
                train=False,
            )
        )
    )


def _chunked_teacher_q(teacher_network, leo, maps, numeric, chunk_size=64):
    blocks = []
    for start in range(0, maps.shape[0], chunk_size):
        stop = min(start + chunk_size, maps.shape[0])
        count = stop - start
        map_block = maps[start:stop]
        numeric_block = numeric[start:stop]
        if count < chunk_size:
            pad = chunk_size - count
            map_block = np.concatenate(
                (map_block, np.repeat(map_block[-1:], pad, axis=0)), axis=0
            )
            numeric_block = np.concatenate(
                (numeric_block, np.repeat(numeric_block[-1:], pad, axis=0)), axis=0
            )
        blocks.append(
            _teacher_q(teacher_network, leo, map_block, numeric_block)[:count]
        )
    return np.concatenate(blocks, axis=0)


def _batch_renorm_audit(template, teacher_network, leo_template, common):
    states = np.load(COMMON_STATES)
    rows = []
    immutable_before = {}
    for seed in SEEDS:
        mask = np.asarray(states["seed"]) == seed
        maps = np.asarray(states["map_channels"])[mask]
        numeric = np.asarray(states["numeric_features"])[mask]
        for evaluator in EVALUATORS:
            checkpoint = _checkpoint(evaluator, seed)
            immutable_before[(seed, evaluator)] = _checkpoint_state_sha256(checkpoint)
            _, leo = load_dual_checkpoint(checkpoint, template, leo_template)
            baseline = _teacher_q(teacher_network, leo, maps, numeric)
            reverse = _teacher_q(
                teacher_network, leo, maps[::-1], numeric[::-1]
            )[::-1]
            chunked = _chunked_teacher_q(
                teacher_network, leo, maps, numeric
            )
            baseline_delivery = baseline[:, DELIVER_3_GOAL_INDEX, :]
            reverse_delivery = reverse[:, DELIVER_3_GOAL_INDEX, :]
            chunked_delivery = chunked[:, DELIVER_3_GOAL_INDEX, :]
            baseline_actions = np.argmax(baseline_delivery, axis=-1)
            reverse_actions = np.argmax(reverse_delivery, axis=-1)
            chunked_actions = np.argmax(chunked_delivery, axis=-1)
            reverse_disagreement = baseline_actions != reverse_actions
            chunked_disagreement = baseline_actions != chunked_actions
            row = {
                "seed": int(seed),
                "evaluator": evaluator,
                "observations": int(maps.shape[0]),
                "train_flag": False,
                "mutable_collections_requested": False,
                "reverse_recommendations_identical": bool(
                    np.array_equal(baseline_actions, reverse_actions)
                ),
                "chunked_recommendations_identical": bool(
                    np.array_equal(baseline_actions, chunked_actions)
                ),
                "reverse_recommendation_disagreement_rate": float(
                    np.mean(reverse_disagreement)
                ),
                "chunked_recommendation_disagreement_rate": float(
                    np.mean(chunked_disagreement)
                ),
                "reverse_max_abs_q_difference": float(
                    np.max(np.abs(baseline_delivery - reverse_delivery))
                ),
                "chunked_max_abs_q_difference": float(
                    np.max(np.abs(baseline_delivery - chunked_delivery))
                ),
            }
            row["passed"] = bool(
                row["reverse_recommendations_identical"]
                and row["reverse_max_abs_q_difference"] <= 1e-5
                and row["chunked_max_abs_q_difference"] <= 1e-3
                and row["chunked_recommendation_disagreement_rate"] <= 0.0025
            )
            rows.append(row)
    immutable = []
    for seed in SEEDS:
        for evaluator in EVALUATORS:
            checkpoint = _checkpoint(evaluator, seed)
            after = _checkpoint_state_sha256(checkpoint)
            before = immutable_before[(seed, evaluator)]
            immutable.append(
                {
                    "seed": int(seed),
                    "evaluator": evaluator,
                    "before_sha256": before,
                    "after_sha256": after,
                    "unchanged": before == after,
                }
            )
    return {
        "rows": rows,
        "robustness_thresholds": {
            "maximum_abs_delivery_q_difference": 0.001,
            "maximum_delivery_argmax_disagreement_rate": 0.0025,
        },
        "exact_chunk_invariance_required": False,
        "all_batch_and_order_checks_passed": all(row["passed"] for row in rows),
        "checkpoint_immutability": immutable,
        "all_checkpoints_immutable": all(row["unchanged"] for row in immutable),
    }


def _weighting_audit(common):
    grouped = defaultdict(lambda: {"unique": [], "weighted": []})
    for record in common["records"]:
        weight = len(record["provenance"])
        for action in record["eligible_target_actions"]:
            for evaluator in EVALUATORS:
                evaluation = record["evaluations"][evaluator]
                value = float(evaluation["teacher_recommendation"] == action)
                key = (int(record["seed"]), evaluator)
                grouped[key]["unique"].append(value)
                grouped[key]["weighted"].extend([value] * weight)
    seed_rows = []
    for (seed, evaluator), values in sorted(grouped.items()):
        seed_rows.append(
            {
                "seed": seed,
                "evaluator": evaluator,
                "unique_state_action_pairs": len(values["unique"]),
                "visit_weighted_state_action_pairs": len(values["weighted"]),
                "unique_equal_weight_teacher_recommendation_rate": float(
                    np.mean(values["unique"])
                ),
                "visit_weighted_teacher_recommendation_rate": float(
                    np.mean(values["weighted"])
                ),
            }
        )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summaries = []
    for evaluator in EVALUATORS:
        rows = [row for row in seed_rows if row["evaluator"] == evaluator]
        for metric in (
            "unique_equal_weight_teacher_recommendation_rate",
            "visit_weighted_teacher_recommendation_rate",
        ):
            summaries.append(
                {
                    "evaluator": evaluator,
                    "metric": metric,
                    **_bootstrap([row[metric] for row in rows], rng),
                }
            )
    lookup = {(row["seed"], row["evaluator"]): row for row in seed_rows}
    contrasts = []
    for metric in (
        "unique_equal_weight_teacher_recommendation_rate",
        "visit_weighted_teacher_recommendation_rate",
    ):
        points = [
            lookup[(seed, "D-on")][metric] - lookup[(seed, "D-frozen")][metric]
            for seed in SEEDS
        ]
        contrasts.append(
            {
                "contrast": "D-on_minus_D-frozen",
                "metric": metric,
                "seed_points": [
                    {"seed": int(seed), "value": float(value)}
                    for seed, value in zip(SEEDS, points)
                ],
                **_bootstrap(points, rng),
            }
        )
    return {
        "independent_unit": "learner_seed",
        "observations_are_not_replication_units": True,
        "seed_rows": seed_rows,
        "summaries": summaries,
        "paired_contrasts": contrasts,
    }


def _curve_path(condition, seed, variant, update):
    if condition == "D-frozen":
        method = "dual_teacher_frozen"
        root = FREEZE_RUN_ROOT
    else:
        method = {"D-on": "dual", "D-off": "dual_bc_off"}[condition]
        root = SOURCE_RUN_ROOT
    return (
        root
        / "size_s"
        / "pack"
        / method
        / variant
        / f"seed{seed}"
        / "curve"
        / f"adapt_{update}.json"
    )


def _metric_separation_audit(prior):
    required = {
        "success_rate",
        "mean_length",
        "mean_discounted_return",
        "violation_rate",
        "violation_delivery_rate",
        "mean_violation_grain_delivered",
    }
    missing = []
    checked = 0
    for condition in CURVE_CONDITIONS:
        for seed in SEEDS:
            for variant in ("fixed", "mutant"):
                for update in SCIENCE_UPDATES:
                    document = _read(_curve_path(condition, seed, variant, update))
                    for kernel in ("fixed", "mutant"):
                        for policy in POLICIES:
                            for family in FAMILIES:
                                block = document[kernel][policy][family]
                                absent = sorted(required - set(block))
                                if absent:
                                    missing.append(
                                        {
                                            "condition": condition,
                                            "seed": int(seed),
                                            "variant": variant,
                                            "update": int(update),
                                            "kernel": kernel,
                                            "policy": policy,
                                            "family": family,
                                            "missing": absent,
                                        }
                                    )
                                checked += 1
    output_keys = set(prior.get("outputs", {}))
    separated_outputs = {
        "exploitation": "u_seed_points" in output_keys,
        "normal_performance": "normal_performance_seed_points" in output_keys,
        "same_policy_kernel_gain": "same_policy_kernel_gain_seed_points" in output_keys,
    }
    return {
        "curve_blocks_checked": checked,
        "required_block_metrics": sorted(required),
        "missing": missing,
        "separated_output_families": separated_outputs,
        "passed": (
            prior.get("execution_complete") is True
            and not missing
            and all(separated_outputs.values())
        ),
    }


class _TeacherGreedyPolicy:
    def __init__(self, teacher_network):
        self.teacher_network = teacher_network

    def apply(self, parameters, maps, numeric, goal_one_hot):
        q_values = self.teacher_network.apply(
            {
                "params": parameters["teacher_params"],
                "batch_stats": parameters["teacher_batch_stats"],
            },
            maps,
            numeric,
            train=False,
        )
        goal_index = jnp.argmax(goal_one_hot, axis=-1)
        selected = jnp.take_along_axis(
            q_values, goal_index[:, None, None], axis=1
        )[:, 0, :]
        return distrax.Categorical(logits=selected), jnp.zeros(
            (maps.shape[0],), dtype=jnp.float32
        )


def _greedy_teacher_audit(config, template, teacher_network, leo_template):
    policy = _TeacherGreedyPolicy(teacher_network)
    rows = []
    for seed in SEEDS:
        seed_results = {}
        for evaluator in EVALUATORS:
            checkpoint = _checkpoint(evaluator, seed)
            _, leo = load_dual_checkpoint(checkpoint, template, leo_template)
            parameters = {
                "teacher_params": leo.params,
                "teacher_batch_stats": leo.batch_stats,
            }
            seed_results[evaluator] = {}
            for variant in (PackRestoreVariant.FIXED, PackRestoreVariant.MUTANT):
                result = evaluate_pack_restore_gc_frozen(
                    policy,
                    parameters,
                    variant=variant,
                    stochastic=False,
                    repeats_per_state=1,
                    seed_base=20000,
                    learner_seed=int(seed),
                    source_growth_period=config.source_growth_period,
                    record_episodes=False,
                )
                seed_results[evaluator][variant.value] = result
                rows.append(
                    {
                        "seed": int(seed),
                        "evaluator": evaluator,
                        "kernel": variant.value,
                        "mode": True,
                        "natural_reset": result["natural_reset"],
                        "common_setup": result["common_setup"],
                    }
                )
        if seed_results["pretrain"] != seed_results["D-frozen"]:
            raise RuntimeError(
                f"seed {seed}: identical pretrain/frozen teachers differ as greedy policies"
            )
    return {
        "descriptive_only": True,
        "claim": (
            "Tests whether a saved teacher can execute a complete delivery policy; "
            "it does not identify whether teacher advice caused student exploitation."
        ),
        "rows": rows,
        "pretrain_and_frozen_exact_for_all_seeds": True,
    }


def audit():
    if _sha256(COMMON_RESULT) != EXPECTED_COMMON_RESULT_SHA256:
        raise RuntimeError("common-state result digest mismatch")
    if _sha256(COMMON_STATES) != EXPECTED_COMMON_STATES_SHA256:
        raise RuntimeError("common-state tensor digest mismatch")
    common = _read(COMMON_RESULT)
    prior = _read(PRIOR_RESULT)
    if (
        common.get("execution_complete") is not True
        or common.get("training_or_optimizer_updates") != 0
    ):
        raise RuntimeError("common-state input is incomplete or mutated training state")
    config, _, template, teacher_network, leo_template = _templates()
    batch_renorm = _batch_renorm_audit(
        template, teacher_network, leo_template, common
    )
    weighting = _weighting_audit(common)
    separation = _metric_separation_audit(prior)
    greedy = _greedy_teacher_audit(
        config, template, teacher_network, leo_template
    )
    checks = {
        "batch_and_order_robust": batch_renorm[
            "all_batch_and_order_checks_passed"
        ],
        "checkpoints_immutable": batch_renorm["all_checkpoints_immutable"],
        "common_state_weightings_separated": bool(weighting["seed_rows"]),
        "outcome_families_separated": separation["passed"],
        "greedy_pretrain_frozen_control_exact": greedy[
            "pretrain_and_frozen_exact_for_all_seeds"
        ],
    }
    return {
        "schema_version": "hackrl_pack_restore_teacher_evaluation_audit_v1",
        "passed": all(checks.values()),
        "training_or_optimizer_updates": 0,
        "checks": checks,
        "inputs": {
            "common_state_result": {
                "path": str(COMMON_RESULT),
                "sha256": _sha256(COMMON_RESULT),
            },
            "common_state_tensors": {
                "path": str(COMMON_STATES),
                "sha256": _sha256(COMMON_STATES),
            },
            "prior_result": {
                "path": str(PRIOR_RESULT),
                "sha256": _sha256(PRIOR_RESULT),
            },
        },
        "batch_renorm_evaluation": batch_renorm,
        "common_state_weighting": weighting,
        "metric_separation": separation,
        "greedy_teacher_policy": greedy,
        "limits": [
            "The fixture and evaluation oracle are internally constructed, not external validation.",
            "The common observations are descriptive states; learner seed is the independent unit.",
            "Greedy teacher rollouts change visitation and cannot establish advice-to-student causality.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = audit()
    _write_json(arguments.output, result)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "checks": result["checks"],
                "output": str(Path(arguments.output).resolve()),
            },
            sort_keys=True,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
