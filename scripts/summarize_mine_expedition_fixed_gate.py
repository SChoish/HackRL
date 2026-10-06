#!/usr/bin/env python3
"""Validate three fixed mine-expedition runs and adjudicate the gate."""

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
)


REPOSITORY = Path(__file__).resolve().parents[1]
GATE_MANIFEST = REPOSITORY / "docs/manifests/mine_expedition_fixed_learnability_v1.json"
GIB = 1024**3


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _expected_config(contract, seed):
    return {
        "seed": seed,
        "num_envs": contract["num_envs"],
        "num_steps": contract["num_steps"],
        "num_updates": contract["num_updates"],
        "update_epochs": contract["update_epochs"],
        "minibatch_size": contract["minibatch_size"],
        "hidden_size": contract["hidden_size"],
        "learning_rate": contract["learning_rate"],
        "gamma": contract["gamma"],
        "gae_lambda": contract["gae_lambda"],
        "clip_epsilon": 0.2,
        "entropy_coefficient": contract["entropy_coefficient"],
        "value_coefficient": 0.5,
        "max_grad_norm": 1.0,
        "training_start": "curriculum",
        "mode_eval_episodes": contract["mode_eval_episodes_per_seed"],
        "sample_eval_episodes": contract["sample_eval_episodes_per_seed"],
        "checkpoint_updates": contract["checkpoint_updates"],
    }


def _validate_evaluation(evaluation, contract, location, errors):
    valid = True
    if not isinstance(evaluation, dict):
        errors.append(f"{location}: evaluation is not an object")
        return False
    if evaluation.get("runner_state_immutable") is not True:
        errors.append(f"{location}: evaluation changed runner state")
        valid = False
    for policy, episodes in (
        ("mode", contract["mode_eval_episodes_per_seed"]),
        ("sample", contract["sample_eval_episodes_per_seed"]),
    ):
        block = evaluation.get(policy, {})
        if not isinstance(block, dict) or (
            block.get("variant") != "fixed"
            or block.get("start") != "natural"
            or block.get("episodes") != episodes
            or block.get("stochastic") is not (policy == "sample")
        ):
            errors.append(f"{location}: invalid {policy} evaluation contract")
            valid = False
            continue
        rate = block.get("success_rate")
        if not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
            errors.append(f"{location}: invalid {policy} success rate")
            valid = False
    return valid


def _validate_capacity(history, destination, contract, seed, errors):
    if not isinstance(history, list):
        errors.append(f"seed {seed}: capacity history is not a list")
        return
    required_events = {"start_or_resume"}
    required_events.update(
        f"before_checkpoint_{update}" for update in contract["checkpoint_updates"]
    )
    required_events.update(
        f"periodic_update_{update}"
        for update in range(64, contract["num_updates"] + 1, 64)
    )
    observed = set()
    for index, record in enumerate(history):
        if not isinstance(record, dict):
            errors.append(f"seed {seed}: capacity record {index} is not an object")
            continue
        observed.add(record.get("event"))
        try:
            projected = int(record["projected_remaining_write_bytes"])
            reserve = int(record["safety_reserve_bytes"])
            required = int(record["required_free_bytes"])
            target_free = int(record["target_filesystem"]["free_bytes"])
            recorded_destination = Path(record["destination"]).resolve()
        except (KeyError, TypeError, ValueError):
            errors.append(f"seed {seed}: malformed capacity record {index}")
            continue
        if (
            recorded_destination != destination.resolve()
            or reserve < 8 * GIB
            or reserve < int(0.2 * projected)
            or required != projected + reserve
            or target_free < required
        ):
            errors.append(f"seed {seed}: invalid capacity arithmetic at record {index}")
    missing = sorted(required_events - observed)
    if missing:
        errors.append(f"seed {seed}: missing capacity events: {missing}")


def _re_evaluate_final(checkpoint, expected_config):
    config = MineExpeditionPPOConfig(**expected_config)
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


def adjudicate(run_root, *, reevaluate=True):
    run_root = Path(run_root)
    manifest = _read(GATE_MANIFEST)
    contract = manifest["training_contract"]
    seeds = list(contract["seeds"])
    updates = list(contract["checkpoint_updates"])
    batch_size = contract["num_envs"] * contract["num_steps"]
    errors = []
    seed_results = []
    execution_shas = set()
    authorized = manifest.get("authorized_source_sha256")
    if not isinstance(authorized, dict) or not authorized:
        errors.append("gate manifest has no authorized source hash set")
        authorized = {}
    # Re-evaluation executes the current checkout, so it must be the exact
    # authorized source set. Artifact-only adjudication instead verifies the
    # source hashes recorded inside each run and is used by synthetic tests.
    if reevaluate:
        for relative, expected_digest in authorized.items():
            try:
                actual_digest = _sha256(REPOSITORY / relative)
            except OSError as error:
                errors.append(f"authorized source missing: {relative}: {error}")
                continue
            if actual_digest != expected_digest:
                errors.append(f"authorized source hash mismatch: {relative}")

    for seed in seeds:
        destination = run_root / f"seed{seed}"
        try:
            summary = _read(destination / "summary.json")
            provenance = _read(destination / "run_manifest.json")
            capacity = _read(destination / "capacity_checks.json")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"seed {seed}: missing or invalid run artifact: {error}")
            continue
        if not isinstance(summary, dict) or not isinstance(provenance, dict):
            errors.append(f"seed {seed}: summary or provenance is not an object")
            continue
        expected_config = _expected_config(contract, seed)
        recorded_config = provenance.get("config")
        if recorded_config != expected_config:
            errors.append(f"seed {seed}: run config differs from gate manifest")
        recorded_sources = provenance.get("execution_source_sha256", {})
        if not isinstance(recorded_sources, dict):
            errors.append(f"seed {seed}: source provenance is not an object")
            recorded_sources = {}
        if any(recorded_sources.get(path) != digest for path, digest in authorized.items()):
            errors.append(f"seed {seed}: run provenance differs from authorized sources")
        if summary.get("status") != "complete":
            errors.append(f"seed {seed}: execution status is not complete")
        if (
            summary.get("seed") != seed
            or summary.get("variant") != "fixed"
            or summary.get("evaluation_start") != "natural"
            or summary.get("updates") != contract["num_updates"]
            or summary.get("transitions") != contract["transitions_per_seed"]
        ):
            errors.append(f"seed {seed}: summary identity or budget mismatch")
        execution_sha = summary.get("execution_code_sha")
        if not execution_sha or execution_sha != provenance.get("execution_code_sha"):
            errors.append(f"seed {seed}: missing or inconsistent execution SHA")
        else:
            execution_shas.add(execution_sha)
        if summary.get("runtime") != provenance.get("runtime"):
            errors.append(f"seed {seed}: runtime provenance mismatch")
        _validate_capacity(capacity, destination, contract, seed, errors)

        evaluations = {}
        parameter_counts = set()
        for update in updates:
            checkpoint = destination / "checkpoints" / f"update_{update}"
            try:
                metadata = _read(checkpoint / "metadata.json")
                checkpoint_config = _read(checkpoint / "config.json")
                evaluation = _read(checkpoint / "evaluation.json")
                state_path = checkpoint / "state.msgpack"
                state_digest = _sha256(state_path)
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"seed {seed} update {update}: invalid checkpoint: {error}")
                continue
            if not isinstance(metadata, dict):
                errors.append(f"seed {seed} update {update}: metadata is not an object")
                continue
            if (
                metadata.get("schema_version")
                != "hackrl_mine_expedition_fixed_checkpoint_v1"
                or metadata.get("global_update") != update
                or metadata.get("environment_steps") != update * batch_size
                or metadata.get("variant") != "fixed"
                or metadata.get("evaluation_start") != "natural"
                or metadata.get("state_sha256") != state_digest
                or checkpoint_config != expected_config
            ):
                errors.append(f"seed {seed} update {update}: checkpoint identity mismatch")
            parameter_count = metadata.get("parameter_count")
            if isinstance(parameter_count, int) and parameter_count > 0:
                parameter_counts.add(parameter_count)
            else:
                errors.append(f"seed {seed} update {update}: invalid parameter count")
            valid_evaluation = _validate_evaluation(
                evaluation, contract, f"seed {seed} update {update}", errors
            )
            if valid_evaluation:
                evaluations[str(update)] = evaluation

        if len(parameter_counts) != 1:
            errors.append(f"seed {seed}: parameter count changed across checkpoints")
        parameter_count = (
            next(iter(parameter_counts)) if len(parameter_counts) == 1 else None
        )
        if summary.get("parameter_count") != parameter_count:
            errors.append(f"seed {seed}: summary parameter count mismatch")
        final = evaluations.get(str(contract["num_updates"]))
        if final is None or summary.get("final_evaluation") != final:
            errors.append(f"seed {seed}: final evaluation does not match checkpoint")
            continue
        if reevaluate:
            try:
                reproduced = _re_evaluate_final(
                    destination
                    / "checkpoints"
                    / f"update_{contract['num_updates']}",
                    expected_config,
                )
            except Exception as error:
                errors.append(f"seed {seed}: final checkpoint reevaluation failed: {error}")
                continue
            if reproduced != final:
                errors.append(f"seed {seed}: final checkpoint reevaluation mismatch")
                continue
            evidence = reproduced
        else:
            evidence = final
        mode_rate = evidence["mode"]["success_rate"]
        sample_rate = evidence["sample"]["success_rate"]
        seed_results.append(
            {
                "seed": seed,
                "mode_success_rate": mode_rate,
                "sample_success_rate": sample_rate,
                "mode_pass": mode_rate == 1.0,
                "sample_pass": sample_rate >= 0.8,
                "execution_code_sha": execution_sha,
                "parameter_count": parameter_count,
            }
        )

    if len(execution_shas) > 1:
        errors.append("training seeds used different execution SHAs")
    execution_complete = not errors and len(seed_results) == len(seeds)
    if execution_complete:
        gate_passed = all(item["mode_pass"] for item in seed_results) and sum(
            item["sample_pass"] for item in seed_results
        ) >= 2
        status = "pass" if gate_passed else "fail"
    else:
        gate_passed = None
        status = "incomplete"
    return {
        "schema_version": "hackrl_mine_expedition_fixed_gate_result_v1",
        "gate_id": manifest["gate_id"],
        "status": status,
        "execution_complete": execution_complete,
        "gate_passed": gate_passed,
        "inference_unit": "training seed; evaluation episodes are not independent learning repetitions",
        "seed_results": seed_results,
        "errors": errors,
        "execution_code_sha": (
            next(iter(execution_shas)) if len(execution_shas) == 1 else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = adjudicate(arguments.run_root)
    _write_json(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "incomplete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
