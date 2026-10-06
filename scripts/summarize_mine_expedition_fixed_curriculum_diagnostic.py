#!/usr/bin/env python3
"""Adjudicate one phase of the fixed mine-expedition curriculum diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from flax import serialization

from hackrl.mine_expedition_ppo import (
    MineExpeditionPPOConfig,
    evaluate_mine_expedition_frozen,
    initialize_mine_expedition_ppo,
    load_mine_expedition_checkpoint,
    mine_expedition_config_payload,
)


REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST = (
    REPOSITORY
    / "docs/manifests/mine_expedition_fixed_curriculum_diagnostic_v1.json"
)
PHASE_DIRECTORIES = {
    "stage_a_craft_ready": "stage_a",
    "stage_b_natural_late": "stage_b",
    "fallback_target_ready": "fallback",
}
GIB = 1024**3


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _expected_config(manifest, phase_name, seed):
    optimizer = manifest["optimizer"]
    phase = manifest["phases"][phase_name]
    return mine_expedition_config_payload(
        MineExpeditionPPOConfig(
            seed=seed,
            num_envs=optimizer["num_envs"],
            num_steps=optimizer["num_steps"],
            num_updates=phase["updates"],
            update_epochs=optimizer["update_epochs"],
            minibatch_size=optimizer["minibatch_size"],
            hidden_size=optimizer["hidden_size"],
            learning_rate=optimizer["learning_rate"],
            gamma=optimizer["gamma"],
            gae_lambda=optimizer["gae_lambda"],
            entropy_coefficient=optimizer["entropy_coefficient"],
            training_start=phase["training_start"],
            mode_eval_episodes=optimizer["mode_eval_episodes_per_seed"],
            sample_eval_episodes=optimizer["sample_eval_episodes_per_seed"],
            checkpoint_updates=tuple(phase["checkpoint_updates"]),
        )
    )


def _validate_evaluation(evaluation, optimizer, location, errors):
    if not isinstance(evaluation, dict):
        errors.append(f"{location}: evaluation is not an object")
        return False
    valid = evaluation.get("runner_state_immutable") is True
    if not valid:
        errors.append(f"{location}: evaluation changed runner state")
    for policy, episodes in (
        ("mode", optimizer["mode_eval_episodes_per_seed"]),
        ("sample", optimizer["sample_eval_episodes_per_seed"]),
    ):
        block = evaluation.get(policy, {})
        expected_stochastic = policy == "sample"
        if not isinstance(block, dict) or any(
            (
                block.get("variant") != "fixed",
                block.get("start") != "natural",
                block.get("episodes") != episodes,
                block.get("stochastic") is not expected_stochastic,
            )
        ):
            errors.append(f"{location}: invalid natural-start {policy} evaluation")
            valid = False
            continue
        rate = block.get("success_rate")
        if not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
            errors.append(f"{location}: invalid {policy} success rate")
            valid = False
    return valid


def _validate_capacity(records, destination, phase, seed, errors):
    if not isinstance(records, list):
        errors.append(f"seed {seed}: capacity history is not a list")
        return
    required = {"start_or_resume"}
    required.update(
        f"before_checkpoint_{update}" for update in phase["checkpoint_updates"]
    )
    required.update(
        f"periodic_update_{update}"
        for update in range(64, phase["updates"] + 1, 64)
    )
    observed = set()
    for index, record in enumerate(records):
        try:
            observed.add(record["event"])
            projected = int(record["projected_remaining_write_bytes"])
            reserve = int(record["safety_reserve_bytes"])
            required_free = int(record["required_free_bytes"])
            target_free = int(record["target_filesystem"]["free_bytes"])
            recorded_destination = Path(record["destination"]).resolve()
        except (KeyError, TypeError, ValueError):
            errors.append(f"seed {seed}: malformed capacity record {index}")
            continue
        if (
            recorded_destination != destination.resolve()
            or reserve < 8 * GIB
            or reserve < int(0.2 * projected)
            or required_free != projected + reserve
            or target_free < required_free
        ):
            errors.append(f"seed {seed}: invalid capacity record {index}")
    missing = sorted(required - observed)
    if missing:
        errors.append(f"seed {seed}: missing capacity events: {missing}")


def _rollout_window(updates, width):
    window = updates[-width:]
    totals = {
        name: sum(float(row.get(name, 0)) for row in window)
        for name in (
            "completed_episodes",
            "completed_successes",
            "completed_timeouts",
            "crafted_pickaxes",
            "mined_targets",
            "returned_targets",
        )
    }
    totals["success_fraction"] = totals["completed_successes"] / max(
        totals["completed_episodes"], 1.0
    )
    totals["updates"] = width
    return totals


def _re_evaluate(checkpoint, config):
    network, template = initialize_mine_expedition_ppo(config)
    runner = load_mine_expedition_checkpoint(checkpoint, template, config)
    state_before = serialization.to_bytes(runner)
    mode = evaluate_mine_expedition_frozen(
        network,
        runner.train_state.params,
        stochastic=False,
        episodes=config.mode_eval_episodes,
        seed_base=30000,
        learner_seed=config.seed,
        discount=config.gamma,
    )
    sample = evaluate_mine_expedition_frozen(
        network,
        runner.train_state.params,
        stochastic=True,
        episodes=config.sample_eval_episodes,
        seed_base=30000,
        learner_seed=config.seed,
        discount=config.gamma,
    )
    return {
        "mode": mode,
        "sample": sample,
        "runner_state_immutable": state_before == serialization.to_bytes(runner),
    }


def summarize(run_root, phase_name, *, reevaluate=True):
    run_root = Path(run_root).resolve()
    manifest = _read(MANIFEST)
    phase = manifest["phases"][phase_name]
    optimizer = manifest["optimizer"]
    seeds = manifest["fixed_contract"]["seeds"]
    phase_root = run_root / PHASE_DIRECTORIES[phase_name]
    errors = []
    seed_results = []
    execution_shas = set()
    authorized = manifest.get("authorized_source_sha256", {})
    if not authorized:
        errors.append("diagnostic manifest has no authorized source hashes")
    for relative, expected in authorized.items():
        try:
            actual = _sha256(REPOSITORY / relative)
        except OSError as error:
            errors.append(f"authorized source missing: {relative}: {error}")
            continue
        if actual != expected:
            errors.append(f"authorized source hash mismatch: {relative}")

    for seed in seeds:
        destination = phase_root / f"seed{seed}"
        location = f"{phase_name} seed {seed}"
        try:
            summary = _read(destination / "summary.json")
            provenance = _read(destination / "run_manifest.json")
            capacity = _read(destination / "capacity_checks.json")
            updates = _read(destination / "updates.json")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{location}: missing or invalid run artifact: {error}")
            continue
        expected_config = _expected_config(manifest, phase_name, seed)
        if provenance.get("config") != expected_config:
            errors.append(f"{location}: run config differs from diagnostic manifest")
        recorded_sources = provenance.get("execution_source_sha256", {})
        if any(recorded_sources.get(path) != digest for path, digest in authorized.items()):
            errors.append(f"{location}: run provenance differs from authorized sources")
        execution_sha = summary.get("execution_code_sha")
        if not execution_sha or execution_sha != provenance.get("execution_code_sha"):
            errors.append(f"{location}: inconsistent execution SHA")
        else:
            execution_shas.add(execution_sha)
        if (
            summary.get("status") != "complete"
            or summary.get("seed") != seed
            or summary.get("variant") != "fixed"
            or summary.get("training_start") != phase["training_start"]
            or summary.get("evaluation_start") != "natural"
            or summary.get("updates") != phase["updates"]
            or summary.get("transitions") != phase["transitions_per_seed"]
        ):
            errors.append(f"{location}: summary identity or budget mismatch")
        initialization = provenance.get("initialization")
        if phase_name == "stage_b_natural_late":
            expected_source = (
                run_root
                / "stage_a"
                / f"seed{seed}"
                / "checkpoints"
                / "update_512"
            ).resolve()
            if (
                not isinstance(initialization, dict)
                or initialization.get("kind") != "fixed_checkpoint_transfer"
                or Path(initialization.get("checkpoint", "")).resolve()
                != expected_source
            ):
                errors.append(f"{location}: invalid stage-A checkpoint transfer")
        elif initialization != {"kind": "random"}:
            errors.append(f"{location}: phase must start from random initialization")

        expected_updates = list(range(1, phase["updates"] + 1))
        if [row.get("update") for row in updates] != expected_updates:
            errors.append(f"{location}: update history is missing or non-contiguous")
        rollout = _rollout_window(
            updates, int(phase.get("route_window_updates", min(128, len(updates))))
        )
        _validate_capacity(capacity, destination, phase, seed, errors)

        checkpoint = destination / "checkpoints" / f"update_{phase['updates']}"
        try:
            checkpoint_config = _read(checkpoint / "config.json")
            metadata = _read(checkpoint / "metadata.json")
            stored_evaluation = _read(checkpoint / "evaluation.json")
            state_digest = _sha256(checkpoint / "state.msgpack")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{location}: invalid final checkpoint: {error}")
            continue
        if (
            checkpoint_config != expected_config
            or metadata.get("state_sha256") != state_digest
            or metadata.get("global_update") != phase["updates"]
            or metadata.get("environment_steps")
            != phase["transitions_per_seed"]
            or metadata.get("variant") != "fixed"
            or metadata.get("evaluation_start") != "natural"
        ):
            errors.append(f"{location}: final checkpoint identity mismatch")
        evaluation_valid = _validate_evaluation(
            stored_evaluation, optimizer, location, errors
        )
        if summary.get("final_evaluation") != stored_evaluation:
            errors.append(f"{location}: summary and checkpoint evaluation differ")
        if reevaluate and evaluation_valid:
            reproduced = _re_evaluate(
                checkpoint, MineExpeditionPPOConfig(**expected_config)
            )
            if reproduced != stored_evaluation:
                errors.append(f"{location}: frozen re-evaluation mismatch")
        seed_results.append(
            {
                "seed": seed,
                "rollout_final_window": rollout,
                "natural_evaluation": stored_evaluation,
                "state_sha256": state_digest,
                "initialization": initialization,
            }
        )

    execution_complete = not errors and len(seed_results) == len(seeds)
    if phase_name == "stage_a_craft_ready":
        qualified = [
            row
            for row in seed_results
            if row["rollout_final_window"]["success_fraction"] >= 0.8
            and row["rollout_final_window"]["crafted_pickaxes"] > 0
            and row["rollout_final_window"]["mined_targets"] > 0
            and row["rollout_final_window"]["returned_targets"] > 0
        ]
        phase_passed = execution_complete and len(qualified) >= 2
        status = "advance" if phase_passed else "fallback"
    elif phase_name == "fallback_target_ready":
        qualified = [
            row
            for row in seed_results
            if row["rollout_final_window"]["success_fraction"] >= 0.8
            and row["rollout_final_window"]["mined_targets"] > 0
            and row["rollout_final_window"]["returned_targets"] > 0
        ]
        phase_passed = execution_complete and len(qualified) >= 2
        status = "localized" if phase_passed else "localization_failed"
    else:
        mode_passes = [
            row["natural_evaluation"]["mode"]["success_rate"] == 1.0
            for row in seed_results
        ]
        sample_passes = [
            row["natural_evaluation"]["sample"]["success_rate"] >= 0.8
            for row in seed_results
        ]
        qualified = [
            row for row, passed in zip(seed_results, sample_passes) if passed
        ]
        phase_passed = (
            execution_complete and all(mode_passes) and sum(sample_passes) >= 2
        )
        status = "pass" if phase_passed else "fail"
    if not execution_complete:
        status = "incomplete"

    return {
        "schema_version": "hackrl_mine_expedition_curriculum_phase_result_v1",
        "diagnostic_id": manifest["diagnostic_id"],
        "phase": phase_name,
        "status": status,
        "execution_complete": execution_complete,
        "phase_passed": phase_passed if execution_complete else None,
        "qualified_seeds": [row["seed"] for row in qualified],
        "fixed_natural_gate_passed": (
            phase_passed if phase_name == "stage_b_natural_late" else False
        ),
        "mutant_training_authorized": (
            phase_passed if phase_name == "stage_b_natural_late" else False
        ),
        "execution_code_shas": sorted(execution_shas),
        "errors": errors,
        "seed_results": seed_results,
        "claim_limit": manifest["evaluation_and_leakage"],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--phase", required=True, choices=tuple(PHASE_DIRECTORIES))
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-reevaluate", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    result = summarize(
        args.run_root, args.phase, reevaluate=not args.no_reevaluate
    )
    _write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["execution_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
