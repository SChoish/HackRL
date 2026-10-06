#!/usr/bin/env python3
"""Adjudicate the bounded fixed return curriculum and its natural gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hackrl.mine_expedition_adjudication import (
    re_evaluate,
    rollout_window,
    validate_capacity,
    validate_evaluation,
)
from hackrl.mine_expedition_ppo import (
    MineExpeditionPPOConfig,
    mine_expedition_config_payload,
)


REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST = (
    REPOSITORY
    / "docs/manifests/mine_expedition_fixed_return_curriculum_v1.json"
)
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


def _source_errors(manifest):
    errors = []
    authorized = manifest.get("authorized_source_sha256", {})
    if not authorized:
        return ["experiment manifest has no authorized source hashes"]
    for relative, expected in authorized.items():
        try:
            actual = _sha256(REPOSITORY / relative)
        except OSError as error:
            errors.append(f"authorized source missing: {relative}: {error}")
            continue
        if actual != expected:
            errors.append(f"authorized source hash mismatch: {relative}")
    return errors


def _trust_root_errors(expected_manifest_sha256):
    actual = _sha256(MANIFEST)
    if actual != expected_manifest_sha256:
        return [
            "experiment manifest differs from the externally authorized digest"
        ]
    return []


def _expected_initialization(run_root, manifest, phase_name, seed):
    index = manifest["phase_order"].index(phase_name)
    if index == 0:
        return None
    previous_name = manifest["phase_order"][index - 1]
    previous = manifest["phases"][previous_name]
    return (
        run_root
        / previous["directory"]
        / f"seed{seed}"
        / "checkpoints"
        / f"update_{previous['updates']}"
    ).resolve()


def summarize_phase(
    run_root,
    phase_name,
    *,
    expected_execution_sha,
    expected_manifest_sha256,
    reevaluate=True,
):
    run_root = Path(run_root).resolve()
    manifest = _read(MANIFEST)
    phase = manifest["phases"][phase_name]
    optimizer = manifest["optimizer"]
    phase_root = run_root / phase["directory"]
    errors = _trust_root_errors(expected_manifest_sha256)
    errors.extend(_source_errors(manifest))
    seed_results = []
    execution_shas = set()

    for seed in manifest["fixed_contract"]["seeds"]:
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
            errors.append(f"{location}: run config differs from experiment manifest")
        recorded_sources = provenance.get("execution_source_sha256", {})
        if any(
            recorded_sources.get(path) != digest
            for path, digest in manifest["authorized_source_sha256"].items()
        ):
            errors.append(f"{location}: provenance differs from authorized sources")
        manifest_relative = str(MANIFEST.relative_to(REPOSITORY))
        if recorded_sources.get(manifest_relative) != expected_manifest_sha256:
            errors.append(
                f"{location}: provenance has a different experiment manifest"
            )
        execution_sha = summary.get("execution_code_sha")
        if not execution_sha or execution_sha != provenance.get("execution_code_sha"):
            errors.append(f"{location}: inconsistent execution SHA")
        else:
            execution_shas.add(execution_sha)
        if any(
            (
                summary.get("status") != "complete",
                summary.get("seed") != seed,
                summary.get("variant") != "fixed",
                summary.get("training_start") != phase["training_start"],
                summary.get("evaluation_start") != "natural",
                summary.get("updates") != phase["updates"],
                summary.get("transitions") != phase["transitions_per_seed"],
            )
        ):
            errors.append(f"{location}: summary identity or budget mismatch")

        initialization = provenance.get("initialization")
        expected_source = _expected_initialization(
            run_root, manifest, phase_name, seed
        )
        if expected_source is None:
            if initialization != {"kind": "random"}:
                errors.append(f"{location}: first phase must start randomly")
        else:
            try:
                source_config = _read(expected_source / "config.json")
                source_metadata = _read(expected_source / "metadata.json")
                source_state_digest = _sha256(expected_source / "state.msgpack")
                source_provenance = _read(
                    expected_source.parent.parent / "run_manifest.json"
                )
            except (OSError, json.JSONDecodeError) as error:
                errors.append(
                    f"{location}: invalid prior-phase checkpoint: {error}"
                )
            else:
                previous_index = manifest["phase_order"].index(phase_name) - 1
                previous_name = manifest["phase_order"][previous_index]
                expected_source_config = _expected_config(
                    manifest, previous_name, seed
                )
                if (
                    not isinstance(initialization, dict)
                    or initialization.get("kind")
                    != "fixed_checkpoint_transfer"
                    or Path(initialization.get("checkpoint", "")).resolve()
                    != expected_source
                    or initialization.get("preserved")
                    != ["policy", "critic", "adam", "action_rng"]
                    or initialization.get("state_sha256") != source_state_digest
                    or source_metadata.get("state_sha256") != source_state_digest
                    or initialization.get("source_config") != source_config
                    or source_config != expected_source_config
                    or initialization.get("source_execution_code_sha")
                    != source_provenance.get("execution_code_sha")
                    or source_provenance.get("execution_code_sha")
                    != provenance.get("execution_code_sha")
                ):
                    errors.append(
                        f"{location}: invalid prior-phase checkpoint transfer"
                    )

        if [row.get("update") for row in updates] != list(
            range(1, phase["updates"] + 1)
        ):
            errors.append(f"{location}: update history is missing or non-contiguous")
        window_width = int(phase.get("route_window_updates", 128))
        rollout = rollout_window(updates, window_width)
        validate_capacity(capacity, destination, phase, seed, errors)

        checkpoint = (
            destination / "checkpoints" / f"update_{phase['updates']}"
        )
        try:
            checkpoint_config = _read(checkpoint / "config.json")
            metadata = _read(checkpoint / "metadata.json")
            stored_evaluation = _read(checkpoint / "evaluation.json")
            state_digest = _sha256(checkpoint / "state.msgpack")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{location}: invalid final checkpoint: {error}")
            continue
        if any(
            (
                checkpoint_config != expected_config,
                metadata.get("state_sha256") != state_digest,
                metadata.get("global_update") != phase["updates"],
                metadata.get("environment_steps")
                != phase["transitions_per_seed"],
                metadata.get("variant") != "fixed",
                metadata.get("evaluation_start") != "natural",
            )
        ):
            errors.append(f"{location}: final checkpoint identity mismatch")
        evaluation_valid = validate_evaluation(
            stored_evaluation, optimizer, location, errors
        )
        if summary.get("final_evaluation") != stored_evaluation:
            errors.append(f"{location}: summary and checkpoint evaluation differ")
        if reevaluate and evaluation_valid:
            reproduced = re_evaluate(
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

    if execution_shas != {expected_execution_sha}:
        errors.append(
            "phase execution SHAs differ from the externally authorized SHA: "
            f"{sorted(execution_shas)}"
        )

    execution_complete = not errors and len(seed_results) == len(
        manifest["fixed_contract"]["seeds"]
    )
    qualified = []
    if phase_name == "natural_full":
        minimum = phase["final_natural_gate"][
            "sample_success_rate_minimum_each_seed"
        ]
        for row in seed_results:
            evaluation = row["natural_evaluation"]
            if (
                evaluation["mode"]["success_rate"] == 1.0
                and evaluation["sample"]["success_rate"] >= minimum
            ):
                qualified.append(row)
        phase_passed = execution_complete and len(qualified) == len(seed_results)
        status = "pass" if phase_passed else "fail"
    else:
        for row in seed_results:
            rollout = row["rollout_final_window"]
            if (
                rollout["success_fraction"] >= phase["advance_success_fraction"]
                and all(rollout[event] > 0 for event in phase["advance_events"])
            ):
                qualified.append(row)
        phase_passed = execution_complete and len(qualified) == len(seed_results)
        status = "advance" if phase_passed else "stop"
    if not execution_complete:
        phase_passed = None
        status = "incomplete"

    return {
        "schema_version": "hackrl_mine_expedition_return_curriculum_phase_result_v1",
        "experiment_id": manifest["experiment_id"],
        "phase": phase_name,
        "status": status,
        "execution_complete": execution_complete,
        "phase_passed": phase_passed,
        "qualified_seeds": [row["seed"] for row in qualified],
        "fixed_natural_gate_passed": phase_passed
        if phase_name == "natural_full" and execution_complete
        else False,
        "mutant_training_authorized": False,
        "execution_code_shas": sorted(execution_shas),
        "authorized_execution_sha": expected_execution_sha,
        "authorized_manifest_sha256": expected_manifest_sha256,
        "errors": errors,
        "seed_results": seed_results,
        "claim_limit": manifest["evaluation_and_leakage"],
    }


def summarize_experiment(
    run_root,
    *,
    expected_execution_sha,
    expected_manifest_sha256,
    reevaluate=True,
):
    run_root = Path(run_root).resolve()
    manifest = _read(MANIFEST)
    phase_results = []
    errors = []
    stopped_phase = None
    for phase_name in manifest["phase_order"]:
        path = run_root / "phase_results" / f"{phase_name}.json"
        if not path.is_file():
            if stopped_phase is None:
                errors.append(f"missing phase result before a declared stop: {phase_name}")
            break
        recorded = _read(path)
        recomputed = summarize_phase(
            run_root,
            phase_name,
            expected_execution_sha=expected_execution_sha,
            expected_manifest_sha256=expected_manifest_sha256,
            reevaluate=reevaluate,
        )
        if recorded != recomputed:
            errors.append(
                f"recorded phase result differs from recomputed evidence: {phase_name}"
            )
            break
        result = recomputed
        if not result["execution_complete"] or result["errors"]:
            errors.append(f"invalid phase evidence: {phase_name}")
            break
        phase_results.append(result)
        if not result["phase_passed"]:
            stopped_phase = phase_name
            break

    later_results = []
    if stopped_phase is not None:
        stop_index = manifest["phase_order"].index(stopped_phase)
        for phase_name in manifest["phase_order"][stop_index + 1 :]:
            if (run_root / "phase_results" / f"{phase_name}.json").exists():
                later_results.append(phase_name)
    if later_results:
        errors.append(f"phase results exist after declared stop: {later_results}")

    final_seen = bool(phase_results) and phase_results[-1]["phase"] == "natural_full"
    natural_passed = final_seen and phase_results[-1]["phase_passed"] is True
    execution_complete = not errors and (
        stopped_phase is not None or final_seen
    )
    if not execution_complete:
        status = "incomplete"
    elif natural_passed:
        status = "pass"
    else:
        status = f"scientific_fail_{stopped_phase or 'natural_full'}"
    execution_shas = sorted(
        {
            sha
            for result in phase_results
            for sha in result.get("execution_code_shas", [])
        }
    )
    return {
        "schema_version": "hackrl_mine_expedition_fixed_return_curriculum_result_v1",
        "experiment_id": manifest["experiment_id"],
        "status": status,
        "execution_complete": execution_complete,
        "completed_phases": [result["phase"] for result in phase_results],
        "stopped_phase": stopped_phase,
        "fixed_natural_gate_passed": natural_passed,
        "mutant_training_authorized": natural_passed,
        "execution_code_shas": execution_shas,
        "authorized_execution_sha": expected_execution_sha,
        "authorized_manifest_sha256": expected_manifest_sha256,
        "errors": errors,
        "phase_summaries": [
            {
                "phase": result["phase"],
                "status": result["status"],
                "qualified_seeds": result["qualified_seeds"],
                "seed_results": result["seed_results"],
            }
            for result in phase_results
        ],
        "budget_limit": manifest["budget"],
        "claim_limit": manifest["evaluation_and_leakage"],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--phase", choices=tuple(_read(MANIFEST)["phase_order"]))
    group.add_argument("--final", action="store_true")
    parser.add_argument("--expected-execution-sha", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.final:
        result = summarize_experiment(
            args.run_root,
            expected_execution_sha=args.expected_execution_sha,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    else:
        result = summarize_phase(
            args.run_root,
            args.phase,
            expected_execution_sha=args.expected_execution_sha,
            expected_manifest_sha256=args.expected_manifest_sha256,
            reevaluate=True,
        )
    _write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["execution_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
