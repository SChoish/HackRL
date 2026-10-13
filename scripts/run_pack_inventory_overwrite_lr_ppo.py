#!/usr/bin/env python3
"""Run the fixed-budget learning-rate preservation comparison."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path

if os.environ.get("HACKRL_DEVICE", "cpu") != "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

import run_pack_inventory_overwrite_ppo as base
from hackrl.pack_inventory_overwrite_rare import (
    build_risk_profile,
    evaluate_regime_point,
)
from hackrl.pack_restore import (
    PackRestoreAction,
    PackRestoreSplit,
    PackRestoreVariant,
    pack_restore_goal_vector,
    pack_restore_step,
    pack_restore_world_done,
)
from hackrl.pack_restore_gc import (
    DELIVER_3_GOAL_INDEX,
    _FROZEN_EVAL_CACHE,
    _batch_inputs,
    _evaluation_states,
    _tree_where,
    evaluate_pack_restore_gc_frozen,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_lr_ppo_v1.json"
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
RUN_ROOT = Path(SPEC["run_root"])
SEEDS = tuple(int(seed) for seed in SPEC["seeds"])
CONDITIONS = tuple(SPEC["conditions"])
LEARNING_CONDITIONS = CONDITIONS
SCIENCE_UPDATES = tuple(
    int(update) for update in SPEC["evaluation"]["adaptation_updates"]
)
EVAL_SEED_BASE = 71000
EXECUTION_SOURCES = tuple(Path(path) for path in SPEC["execution_sources"])

_BASE_EMPTY_TRACKER = base._empty_tracker
_BASE_UPDATE_TRACKER = base.update_exposure_tracker
_BASE_CAPACITY = base._capacity
_BASE_SUMMARIZE_CELL = base._summarize_cell
_BASE_RUN_MANIFEST = base._run_manifest
_BEHAVIOR_ANCHOR_CACHE: dict[int, dict[str, np.ndarray | int]] = {}


def _prerequisite_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY / path


def _probability(condition: str) -> float:
    return float(SPEC["conditions"][condition]["overwrite_probability"])


def _learning_rate_multiplier(condition: str) -> float:
    return float(SPEC["conditions"][condition]["learning_rate_multiplier"])


def _install_spec() -> None:
    base.SPEC_PATH = SPEC_PATH
    base.SPEC = SPEC
    base.RUN_ROOT = RUN_ROOT
    base.SOURCE_ROOT = Path(SPEC["source_root"])
    base.SEEDS = SEEDS
    base.CONDITIONS = CONDITIONS
    base.LEARNING_CONDITIONS = LEARNING_CONDITIONS
    base.SCIENCE_UPDATES = SCIENCE_UPDATES
    base.EVAL_SEED_BASE = EVAL_SEED_BASE
    base.EXECUTION_SOURCES = EXECUTION_SOURCES


def _capacity() -> dict:
    return _BASE_CAPACITY(RUN_ROOT)


def _validate_prerequisite() -> None:
    prerequisites = SPEC["prerequisites"]
    for path_key, hash_key in (
        ("prior_result", "prior_result_sha256"),
        ("prior_raw_result", "prior_raw_result_sha256"),
        ("reference_paths", "reference_paths_sha256"),
    ):
        path = _prerequisite_path(prerequisites[path_key])
        if base._sha256(path) != prerequisites[hash_key]:
            raise RuntimeError(f"pinned prerequisite changed: {path}")
    prior = json.loads(
        _prerequisite_path(prerequisites["prior_result"]).read_text(
            encoding="utf-8"
        )
    )
    if prior.get("status") != "complete_fixed_budget_no_automatic_extension":
        raise RuntimeError("the prior stochastic PPO result is not complete")


def _branch_config(seed: int, condition: str):
    source = base._source_config(seed)
    expected_source_rate = float(SPEC["optimization"]["source_learning_rate"])
    if source.learning_rate != expected_source_rate:
        raise RuntimeError(f"source learning rate changed for seed {seed}")
    config = replace(
        source,
        variant=PackRestoreVariant.INVENTORY_OVERWRITE.value,
        inventory_overwrite_probability=_probability(condition),
        learning_rate=source.learning_rate * _learning_rate_multiplier(condition),
        num_updates=int(SPEC["optimization"]["updates"]),
        checkpoint_updates=(),
    )
    config.validate()
    return config


def _empty_tracker(num_envs: int) -> dict:
    tracker = _BASE_EMPTY_TRACKER(num_envs)
    tracker["overwrite_fire_events"] = 0
    return tracker


def update_exposure_tracker(
    tracker: dict,
    metrics: dict,
    *,
    learning: bool,
    optimizer_applications: int,
) -> dict:
    previous_fires = int(tracker.get("overwrite_fire_events", 0))
    output = _BASE_UPDATE_TRACKER(
        tracker,
        metrics,
        learning=learning,
        optimizer_applications=optimizer_applications,
    )
    output["overwrite_fire_events"] = previous_fires + int(
        np.asarray(jax.device_get(metrics["overwrite_fire_events"]))
    )
    return output


def _reference_episodes() -> list[dict]:
    document = json.loads(
        _prerequisite_path(SPEC["prerequisites"]["reference_paths"])
        .read_text(encoding="utf-8")
    )
    return document["episodes"]


def _categorical_drift(reference_logits, current_logits) -> dict:
    reference_logits = jnp.asarray(reference_logits, dtype=jnp.float32)
    current_logits = jnp.asarray(current_logits, dtype=jnp.float32)
    if reference_logits.shape != current_logits.shape or reference_logits.ndim != 2:
        raise ValueError("behavior logits must be equal two-dimensional arrays")
    reference_log_probs = jax.nn.log_softmax(reference_logits, axis=-1)
    current_log_probs = jax.nn.log_softmax(current_logits, axis=-1)
    reference_probs = jnp.exp(reference_log_probs)
    per_state_kl = jnp.sum(
        reference_probs * (reference_log_probs - current_log_probs), axis=-1
    )
    reference_actions = jnp.argmax(reference_logits, axis=-1)
    current_actions = jnp.argmax(current_logits, axis=-1)
    changed = reference_actions != current_actions
    state_count = int(reference_logits.shape[0])
    mean_kl = float(
        np.asarray(jax.device_get(jnp.mean(per_state_kl)))
    )
    return {
        "anchor_state_count": state_count,
        "mean_kl_source_to_current": max(0.0, mean_kl),
        "argmax_action_change_count": int(
            np.asarray(jax.device_get(jnp.sum(changed)))
        ),
        "argmax_action_change_rate": float(
            np.asarray(jax.device_get(jnp.mean(changed.astype(jnp.float32))))
        ),
    }


def _build_source_behavior_anchor(seed: int) -> dict[str, np.ndarray | int]:
    network, source_runner, _, _ = base._fresh_runner(seed, CONDITIONS[0])
    source_config = base._source_config(seed)
    episode_count = int(SPEC["evaluation"]["states"])
    state, _, _, _ = _evaluation_states(
        PackRestoreSplit(SPEC["evaluation"]["split"]),
        1,
        source_config.source_growth_period,
        natural_only=True,
    )
    state_count = int(jax.tree.leaves(state)[0].shape[0])
    if state_count != episode_count:
        raise AssertionError(
            f"behavior anchor has {state_count} states, expected {episode_count}"
        )
    goals = jnp.full(
        (episode_count,), DELIVER_3_GOAL_INDEX, dtype=jnp.int32
    )
    done = jnp.zeros((episode_count,), dtype=jnp.bool_)
    input_batches: list[list[np.ndarray]] = [[], [], []]
    logit_batches: list[np.ndarray] = []
    active_counts: list[int] = []
    for _ in range(int(SPEC["evaluation"]["horizon_ticks"])):
        inputs = _batch_inputs(state, goals)
        policy, _ = network.apply(source_runner.train_state.params, *inputs)
        active = jnp.logical_not(done)
        active_host = np.asarray(jax.device_get(active), dtype=bool)
        active_counts.append(int(np.sum(active_host)))
        for index, values in enumerate(inputs):
            input_batches[index].append(
                np.asarray(jax.device_get(values))[active_host]
            )
        logit_batches.append(
            np.asarray(jax.device_get(policy.logits))[active_host]
        )
        actions = jnp.argmax(policy.logits, axis=-1)
        actions = jnp.where(active, actions, int(PackRestoreAction.NOOP))
        stepped = jax.vmap(
            lambda item, action: pack_restore_step(
                item, action, PackRestoreVariant.FIXED
            )
        )(state, actions)
        achieved = jax.vmap(pack_restore_goal_vector)(stepped)[
            :, DELIVER_3_GOAL_INDEX
        ]
        world_done = jax.vmap(pack_restore_world_done)(stepped)
        just_done = jnp.logical_and(
            active, jnp.logical_or(achieved, world_done)
        )
        state = _tree_where(active, stepped, state)
        done = jnp.logical_or(done, just_done)
        if bool(np.all(np.asarray(jax.device_get(done)))):
            break
    anchor = {
        "map_channels": np.concatenate(input_batches[0], axis=0),
        "numeric_features": np.concatenate(input_batches[1], axis=0),
        "goal_one_hot": np.concatenate(input_batches[2], axis=0),
        "source_logits": np.concatenate(logit_batches, axis=0),
        "source_episode_count": episode_count,
        "source_path_max_length": len(active_counts),
    }
    if int(anchor["source_logits"].shape[0]) != sum(active_counts):
        raise AssertionError("source behavior anchor count changed")
    return anchor


def _behavior_change(network, parameters, seed: int) -> dict:
    anchor = _BEHAVIOR_ANCHOR_CACHE.get(seed)
    if anchor is None:
        anchor = _build_source_behavior_anchor(seed)
        _BEHAVIOR_ANCHOR_CACHE[seed] = anchor
    policy, _ = network.apply(
        parameters,
        jnp.asarray(anchor["map_channels"]),
        jnp.asarray(anchor["numeric_features"]),
        jnp.asarray(anchor["goal_one_hot"]),
    )
    output = _categorical_drift(anchor["source_logits"], policy.logits)
    output.update(
        {
            "anchor": "all active states on the source policy's deterministic fixed-kernel trajectories from the 32 natural validation starts",
            "state_weighting": "equal weight per active source-trajectory state",
            "source_episode_count": int(anchor["source_episode_count"]),
            "source_path_max_length": int(anchor["source_path_max_length"]),
            "kl_direction": "D_KL(source_policy || current_policy)",
        }
    )
    return output


def _evaluate(network, parameters, seed: int, condition: str) -> dict:
    probability = _probability(condition)
    source_config = base._source_config(seed)
    common = {
        "split": SPEC["evaluation"]["split"],
        "seed_base": EVAL_SEED_BASE,
        "learner_seed": seed,
        "source_growth_period": source_config.source_growth_period,
        "record_episodes": True,
        "natural_only": True,
    }
    fixed_mode = evaluate_pack_restore_gc_frozen(
        network,
        parameters,
        variant=PackRestoreVariant.FIXED.value,
        stochastic=False,
        repeats_per_state=int(
            SPEC["evaluation"]["fixed_mode_repeats_per_state"]
        ),
        **common,
    )
    fixed_sample = evaluate_pack_restore_gc_frozen(
        network,
        parameters,
        variant=PackRestoreVariant.FIXED.value,
        stochastic=True,
        repeats_per_state=int(
            SPEC["evaluation"]["normal_sample_repeats_per_state"]
        ),
        **common,
    )
    target_sample = evaluate_pack_restore_gc_frozen(
        network,
        parameters,
        variant=PackRestoreVariant.INVENTORY_OVERWRITE.value,
        overwrite_probability=probability,
        stochastic=True,
        repeats_per_state=int(
            SPEC["evaluation"]["stochastic_sample_repeats_per_state"]
        ),
        **common,
    )
    for result, repeats in (
        (fixed_mode, SPEC["evaluation"]["fixed_mode_repeats_per_state"]),
        (fixed_sample, SPEC["evaluation"]["normal_sample_repeats_per_state"]),
        (target_sample, SPEC["evaluation"]["stochastic_sample_repeats_per_state"]),
    ):
        expected = int(SPEC["evaluation"]["states"]) * int(repeats)
        if int(result["overall"]["episodes"]) != expected:
            raise AssertionError("evaluation panel size changed")
    profile = build_risk_profile(
        fixed_mode["episode_records"],
        _reference_episodes(),
        require_all_fixed_success=False,
    )
    return {
        "overwrite_probability": probability,
        "learning_rate_multiplier": _learning_rate_multiplier(condition),
        "fixed_kernel": {"mode": fixed_mode, "sample": fixed_sample},
        "target_kernel": {
            "exact_mode": evaluate_regime_point(profile, probability),
            "sample": target_sample,
        },
        "risk_profile": profile,
        "behavior_change_from_source": _behavior_change(
            network, parameters, seed
        ),
        "interpretation": {
            "mode_is_exact_finite_panel_computation": True,
            "sample_is_reported_separately": True,
            "behavior_anchor_is_source_visitation_not_current_visitation": True,
        },
    }


def _write_evaluation(cell: Path, update: int, network, runner, seed: int) -> None:
    path = base._evaluation_path(cell, update)
    if path.is_file():
        return
    condition = cell.parent.name
    base._write_json(
        path,
        {
            "schema_version": "hackrl_pack_inventory_overwrite_lr_eval_v1",
            "condition": condition,
            "seed": seed,
            "adaptation_update": update,
            "collector_slots": update
            * int(SPEC["optimization"]["collector_slots_per_update"]),
            "train_state_sha256": base._train_state_sha256(
                runner.train_state
            ),
            "evaluation": _evaluate(
                network, runner.train_state.params, seed, condition
            ),
        },
    )
    print(f"[evaluation] {path}", flush=True)


def _write_origin(
    cell: Path, condition: str, seed: int, provenance: dict
) -> None:
    config = _branch_config(seed, condition)
    document = {
        "condition": condition,
        "seed": seed,
        "kernel": SPEC["conditions"][condition]["kernel"],
        "overwrite_probability": _probability(condition),
        "learning_rate_multiplier": _learning_rate_multiplier(condition),
        "branch_learning_rate": float(config.learning_rate),
        "source_learning_rate": float(base._source_config(seed).learning_rate),
        **provenance,
    }
    path = cell / "origin.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != document:
        raise RuntimeError(f"origin changed: {path}")
    base._write_json(path, document)


def _summarize_cell(
    cell: Path,
    condition: str,
    seed: int,
    runner,
    tracker: dict,
    origin: dict,
) -> dict:
    summary = _BASE_SUMMARIZE_CELL(
        cell, condition, seed, runner, tracker, origin
    )
    final = json.loads(
        base._evaluation_path(
            cell, int(SPEC["optimization"]["updates"])
        ).read_text(encoding="utf-8")
    )["evaluation"]
    summary.update(
        {
            "schema_version": "hackrl_pack_inventory_overwrite_lr_cell_v1",
            "overwrite_probability": _probability(condition),
            "learning_rate_multiplier": _learning_rate_multiplier(condition),
            "branch_learning_rate": float(
                _branch_config(seed, condition).learning_rate
            ),
            "overwrite_fire_events": int(tracker["overwrite_fire_events"]),
            "final_exact_mode": final["target_kernel"]["exact_mode"],
            "fixed_mode_risk_count_distribution": final["risk_profile"][
                "risk_count_distribution"
            ],
            "final_behavior_change_from_source": final[
                "behavior_change_from_source"
            ],
        }
    )
    return summary


def _run_manifest() -> dict:
    manifest = _BASE_RUN_MANIFEST()
    manifest.update(
        {
            "schema_version": "hackrl_pack_inventory_overwrite_lr_ppo_run_v1",
            "pinned_prerequisite_sha256": {
                key: SPEC["prerequisites"][key]
                for key in (
                    "prior_result_sha256",
                    "prior_raw_result_sha256",
                    "reference_paths_sha256",
                )
            },
            "conditions": {
                condition: {
                    "overwrite_probability": _probability(condition),
                    "learning_rate_multiplier": _learning_rate_multiplier(
                        condition
                    ),
                    "learning_rate": float(
                        _branch_config(SEEDS[0], condition).learning_rate
                    ),
                }
                for condition in CONDITIONS
            },
            "behavior_drift_anchor": SPEC["evaluation"][
                "behavior_drift_anchor"
            ],
            "defect_probability_in_observation": False,
        }
    )
    return manifest


def _install_overrides() -> None:
    _install_spec()
    base._capacity = _capacity
    base._validate_prerequisite = _validate_prerequisite
    base._branch_config = _branch_config
    base._empty_tracker = _empty_tracker
    base.update_exposure_tracker = update_exposure_tracker
    base._write_evaluation = _write_evaluation
    base._write_origin = _write_origin
    base._summarize_cell = _summarize_cell
    base._run_manifest = _run_manifest


def main() -> None:
    _install_overrides()
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-contract", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    arguments = parser.parse_args()
    if arguments.validate_contract:
        print(json.dumps(base.validate_contract(), indent=2, sort_keys=True))
        return
    if not arguments.all and (
        arguments.condition is None or arguments.seed is None
    ):
        raise SystemExit("use --all or provide both --condition and --seed")
    manifest = base._prepare_run()
    print(f"[provenance] execution_head={manifest['execution_head']}", flush=True)
    jobs = (
        base.build_jobs()
        if arguments.all
        else [
            {
                "condition": arguments.condition,
                "seed": arguments.seed,
                "learning": True,
            }
        ]
    )
    for job in jobs:
        condition = str(job["condition"])
        seed = int(job["seed"])
        if base._job_complete(job):
            print(f"[skip] {condition} seed={seed}", flush=True)
            continue
        try:
            base.run_learning(condition, seed)
            base._capacity_checkpoint()
        except Exception:
            base._write_json(
                base._cell(condition, seed) / "failed.json",
                {"failed_unix": time.time(), "traceback": traceback.format_exc()},
            )
            raise
        finally:
            _FROZEN_EVAL_CACHE.clear()
            jax.clear_caches()
    if arguments.all and all(
        base._job_complete(job) for job in base.build_jobs()
    ):
        base._write_json(
            RUN_ROOT / "training_complete.json",
            {
                "status": "all_20_jobs_complete_pending_aggregation",
                "completed_unix": time.time(),
                "execution_head": manifest["execution_head"],
            },
        )


if __name__ == "__main__":
    main()
