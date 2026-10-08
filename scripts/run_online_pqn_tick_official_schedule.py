#!/usr/bin/env python3
"""Diagnose and rerun the bounded PQN TICK schedule-restoration gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

if os.environ.get("HACKRL_DEVICE", "cpu") == "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
else:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

import run_online_algorithm_development as base
from hackrl.online_algorithm_env import (
    PQN,
    environment_adapter,
    make_online_value_update,
)
from hackrl.tick_claim_gc import TickClaimGCConfig

REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY / "docs/manifests/online_pqn_tick_official_schedule_v1.json"
SOURCE_PATHS = (
    "docs/manifests/online_pqn_tick_official_schedule_v1.json",
    "scripts/run_online_pqn_tick_official_schedule.py",
    "scripts/run_online_pqn_tick_official_schedule_gpu.sh",
    "scripts/guard_identity_checked_gpu_keepalive.py",
    "scripts/stop_identity_checked_gpu_keepalive.py",
    "scripts/run_online_algorithm_development.py",
    "src/hackrl/online_algorithm_env.py",
    "src/hackrl/online_value.py",
    "src/hackrl/batch_renorm.py",
    "src/hackrl/discrete_sac.py",
    "src/hackrl/dual_leo.py",
    "src/hackrl/pack_restore.py",
    "src/hackrl/pack_restore_gc.py",
    "src/hackrl/tick_claim.py",
    "src/hackrl/tick_claim_gc.py",
    "src/hackrl/tick_claim_oracle.py",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {relative: sha256_file(REPOSITORY / relative) for relative in SOURCE_PATHS}


def validate_manifest(manifest):
    training = manifest["training"]
    common = manifest["common_training"]
    if manifest["budget"] != {
        "jobs": 2,
        "training_transitions": 33_554_432,
        "maximum_evaluation_episodes": 128,
        "automatic_extension": False,
    }:
        raise RuntimeError("the bounded gate budget changed")
    if (
        training["method"] != PQN
        or training["environment"] != "TICK-CLAIM"
        or training["kernel"] != "fixed"
        or training["goal_mode"] != "workshop12"
        or training["learning_rate"] != 2e-4
        or training["gamma"] != 0.995
        or training["max_grad_norm"] != 1.0
        or training["pqn_hidden_size"] != 1024
        or training["epsilon_start"] != 1.0
        or training["epsilon_finish"] != 0.1
        or training["epsilon_decay_transitions"] != 16_777_216
    ):
        raise RuntimeError("the fixed GC-PQN candidate contract changed")
    if training["learner_seeds"] != [110, 111]:
        raise RuntimeError("learner seeds changed")
    if (
        training["num_envs"] != 512
        or training["num_steps"] != 2
        or training["minibatch_size"] != 256
        or training["num_minibatches"] != 4
        or training["update_epochs"] != 1
    ):
        raise RuntimeError("the restored PQN schedule changed")
    if training["num_envs"] * training["num_steps"] != training["rollout_transitions"]:
        raise RuntimeError("rollout arithmetic changed")
    expected_updates = training["transitions_per_job"] // training["rollout_transitions"]
    if training["updates"] != expected_updates or expected_updates != 16384:
        raise RuntimeError("physical-transition budget changed")
    expected_gradients = (
        training["updates"]
        * training["num_minibatches"]
        * training["update_epochs"]
    )
    if training["optimizer_applications_per_job"] != expected_gradients:
        raise RuntimeError("optimizer schedule arithmetic changed")
    mirrored = {
        "kernel": "kernel",
        "goal_mode": "goal_mode",
        "learner_seeds": "learner_seeds",
        "num_envs": "num_envs",
        "num_steps": "num_steps",
        "updates": "updates",
        "transitions_per_job": "transitions_per_job",
        "gamma": "gamma",
        "max_grad_norm": "max_grad_norm",
        "log_interval_updates": "log_interval_updates",
    }
    if any(common[left] != training[right] for left, right in mirrored.items()):
        raise RuntimeError("common training compatibility view disagrees with training")
    if common["transitions_per_update"] != training["rollout_transitions"]:
        raise RuntimeError("common rollout size disagrees with training")
    if common["evaluation"] != manifest["evaluation"]:
        raise RuntimeError("common evaluation view disagrees with evaluation")
    expected_candidate = {
        "id": training["candidate_id"],
        "parent_candidate_id": training["parent_candidate_id"],
        "pqn_hidden_size": training["pqn_hidden_size"],
        "leo_hidden_size": 512,
        "learning_rate": training["learning_rate"],
        "epsilon_start": training["epsilon_start"],
        "epsilon_finish": training["epsilon_finish"],
        "epsilon_decay_transitions": training["epsilon_decay_transitions"],
    }
    if manifest["candidate"] != expected_candidate:
        raise RuntimeError("candidate compatibility view disagrees with training")
    checkpoints = manifest["execution"].get("checkpoint_updates", training["checkpoint_updates"])
    if checkpoints != training["checkpoint_updates"] or checkpoints != [0, 4096, 8192, 12288, 16384]:
        raise RuntimeError("checkpoint schedule changed")
    evaluation = manifest["evaluation"]
    if evaluation["natural_start_episodes"] != 32 or evaluation["common_setup_diagnostic_episodes"] != 32:
        raise RuntimeError("evaluation episode contract changed")
    if evaluation["bug_and_mutant_metrics_persisted_or_used"] is not False:
        raise RuntimeError("this is a fixed-normal-only gate")


def jobs(manifest):
    candidate = manifest["candidate"]
    return [
        {
            "id": f"tick-pqn-{candidate['id']}-s{seed}",
            "environment": "tick",
            "method": PQN,
            "registered_name": PQN,
            "candidate_id": candidate["id"],
            "candidate": candidate,
            "seed": seed,
        }
        for seed in manifest["training"]["learner_seeds"]
    ]


def environment_config(job, manifest):
    training = manifest["training"]
    return TickClaimGCConfig(
        variant=training["kernel"],
        seed=job["seed"],
        num_envs=training["num_envs"],
        num_steps=training["num_steps"],
        num_updates=training["updates"],
        update_epochs=training["update_epochs"],
        minibatch_size=training["minibatch_size"],
        hidden_size=8,
        learning_rate=training["learning_rate"],
        gamma=training["gamma"],
        max_grad_norm=training["max_grad_norm"],
        goal_mode=training["goal_mode"],
        evaluation_split=manifest["evaluation"]["split"],
        mode_repeats_per_state=1,
        sample_repeats_per_state=1,
    )


def checkpoint_complete(path, job, update, hashes):
    path = Path(path)
    rollout_transitions = read_json(MANIFEST_PATH)["training"]["rollout_transitions"]
    state_path = path / "state.msgpack"
    metadata_path = path / "metadata.json"
    if not state_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = read_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return False
    expected_steps = int(update) * rollout_transitions
    return (
        metadata.get("schema_version") == "hackrl_online_algorithm_development_checkpoint_v1"
        and metadata.get("job_id") == job["id"]
        and metadata.get("method") == PQN
        and metadata.get("environment") == "tick"
        and metadata.get("candidate_id") == job["candidate_id"]
        and metadata.get("seed") == job["seed"]
        and metadata.get("update") == int(update)
        and metadata.get("environment_steps") == expected_steps
        and metadata.get("state_sha256") == sha256_file(state_path)
        and metadata.get("execution_source_hashes") == hashes
    )


def summary_complete(path, job, hashes, manifest):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        summary = read_json(path)
        final = manifest["training"]["updates"]
        checkpoint = base._checkpoint_dir(path.parent, final)
        return (
            summary.get("schema_version") == "hackrl_online_algorithm_development_summary_v1"
            and summary.get("execution_complete") is True
            and summary.get("job") == job["id"]
            and summary.get("method") == PQN
            and summary.get("environment") == "tick"
            and summary.get("candidate_id") == job["candidate_id"]
            and summary.get("seed") == job["seed"]
            and summary.get("global_update") == final
            and summary.get("environment_steps") == manifest["training"]["transitions_per_job"]
            and summary.get("execution_source_hashes") == hashes
            and summary.get("checkpoint") == str(checkpoint.resolve())
            and summary.get("checkpoint_state_sha256") == sha256_file(checkpoint / "state.msgpack")
            and checkpoint_complete(checkpoint, job, final, hashes)
            and base._validate_evaluation(summary.get("evaluation", {}), manifest)
        )
    except (KeyError, OSError, json.JSONDecodeError):
        return False


def require_smoke(manifest):
    path = Path(manifest["execution"]["run_root"]) / "gpu_smoke.json"
    if not path.is_file():
        raise RuntimeError("the bound GPU smoke is missing")
    smoke = read_json(path)
    if smoke.get("passed") is not True:
        raise RuntimeError("the bound GPU smoke did not pass")
    if smoke.get("git_head") != subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip():
        raise RuntimeError("GPU smoke git head changed")
    if smoke.get("execution_source_hashes") != source_hashes():
        raise RuntimeError("GPU smoke source hashes changed")
    training = manifest["training"]
    gradients = training["num_minibatches"] * training["update_epochs"]
    expected = {
        "environment_steps": training["rollout_transitions"],
        "rollout_updates": 1,
        "gradient_steps": gradients,
        "optimizer_step": gradients,
        "batch_renorm_steps": [gradients, gradients],
    }
    if smoke.get("counters") != expected:
        raise RuntimeError("GPU smoke counters changed")


def compiled_updates(job, adapter, networks, config):
    training = read_json(MANIFEST_PATH)["training"]
    update = jax.jit(
        make_online_value_update(
            adapter,
            config,
            networks,
            method=PQN,
            epsilon_start=job["candidate"]["epsilon_start"],
            epsilon_finish=job["candidate"]["epsilon_finish"],
            epsilon_decay_transitions=job["candidate"]["epsilon_decay_transitions"],
            minibatch_size=training["minibatch_size"],
            update_epochs=training["update_epochs"],
        )
    )
    return update, None


def batch_renorm_steps(batch_stats):
    found = []

    def visit(value):
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key == "steps":
                    found.append(int(np.asarray(jax.device_get(child))))
                else:
                    visit(child)

    visit(batch_stats)
    return sorted(found)


def _expand_all_goals(inputs, num_goals):
    maps, numeric, _ = inputs
    batch = maps.shape[0]
    maps = jnp.repeat(maps, num_goals, axis=0)
    numeric = jnp.repeat(numeric, num_goals, axis=0)
    goals = jnp.tile(jnp.eye(num_goals, dtype=jnp.float32), (batch, 1))
    return maps, numeric, goals


def run_batch_renorm_diagnostic(manifest):
    contract = manifest["normalization_diagnostic"]
    output = Path(contract["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if (
            sha256_file(output) == contract.get("output_sha256")
            and output.stat().st_size == contract.get("output_bytes")
        ):
            result = read_json(output)
            print(json.dumps(result, indent=2, sort_keys=True))
            return result
        raise RuntimeError(
            "preserving a diagnostic artifact that differs from the pinned result"
        )
    if contract.get("output_sha256") is not None:
        raise RuntimeError(
            "the pinned diagnostic is missing; a reviewed manifest revision is required"
        )
    old_manifest = read_json(
        REPOSITORY / "docs/manifests/online_algorithm_expansion_v1_development_seen_goals_fix_v1.json"
    )
    candidates = old_manifest["candidate_registry"][PQN]
    old_root = Path(manifest["normalization_diagnostic"]["source_run"])
    adapter = environment_adapter("tick")
    fixed_job = {
        "id": "diagnostic-fixed-input",
        "environment": "tick",
        "method": PQN,
        "registered_name": PQN,
        "candidate_id": candidates[0]["id"],
        "candidate": candidates[0],
        "seed": 110,
    }
    fixed_config = base.environment_config(fixed_job, old_manifest)
    _, fixture_runner = adapter.initialize(fixed_config)
    fixed_inputs = _expand_all_goals(
        adapter.batch_inputs(fixture_runner.env_state, fixture_runner.current_goal),
        adapter.num_goals,
    )
    fixed_hash = hashlib.sha256(serialization.to_bytes(fixed_inputs)).hexdigest()
    rows = []
    for candidate in candidates:
        for seed in (110, 111):
            job = {
                "id": f"tick-pqn-{candidate['id']}-s{seed}",
                "environment": "tick",
                "method": PQN,
                "registered_name": PQN,
                "candidate_id": candidate["id"],
                "candidate": candidate,
                "seed": seed,
            }
            config = base.environment_config(job, old_manifest)
            _, networks, runner, replay = base.initialize_job(job, config)
            checkpoint = (
                old_root / "pqn" / candidate["id"] / "tick" / f"seed{seed}"
                / "checkpoints" / "update_000512"
            )
            metadata = read_json(checkpoint / "metadata.json")
            state_path = checkpoint / "state.msgpack"
            before_hash = sha256_file(state_path)
            if before_hash != metadata["state_sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
            restored = serialization.from_bytes(
                base._payload(runner, replay), state_path.read_bytes()
            )
            state = restored["runner"].train_state
            variables = jax.tree.map(
                lambda leaf: jnp.array(leaf, copy=True),
                {"params": state.params, "batch_stats": state.batch_stats},
            )
            running_q = networks.pqn.apply(variables, *fixed_inputs, train=False)
            batch_q, _ = networks.pqn.apply(
                variables, *fixed_inputs, train=True, mutable=["batch_stats"]
            )
            running_q, batch_q = jax.device_get((running_q, batch_q))
            difference = np.abs(np.asarray(running_q) - np.asarray(batch_q))
            running_actions = np.argmax(np.asarray(running_q), axis=-1)
            batch_actions = np.argmax(np.asarray(batch_q), axis=-1)
            goal_indexes = np.tile(np.arange(adapter.num_goals), fixed_inputs[0].shape[0] // adapter.num_goals)
            per_goal = []
            for goal in range(adapter.num_goals):
                mask = goal_indexes == goal
                per_goal.append({
                    "goal": goal,
                    "episodes": int(mask.sum()),
                    "greedy_action_disagreement_count": int(np.sum(running_actions[mask] != batch_actions[mask])),
                    "greedy_action_disagreement_rate": float(np.mean(running_actions[mask] != batch_actions[mask])),
                    "q_mean_absolute_difference": float(np.mean(difference[mask])),
                    "q_max_absolute_difference": float(np.max(difference[mask])),
                })
            after_hash = sha256_file(state_path)
            if before_hash != after_hash:
                raise RuntimeError("diagnostic mutated a checkpoint")
            rows.append({
                "candidate_id": candidate["id"],
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256_before": before_hash,
                "checkpoint_sha256_after": after_hash,
                "batch_renorm_steps": batch_renorm_steps(state.batch_stats),
                "initial_running_stat_weight_at_momentum_0_999": float(0.999 ** 512),
                "fixed_batch_rows": int(fixed_inputs[0].shape[0]),
                "q_mean_absolute_difference": float(np.mean(difference)),
                "q_max_absolute_difference": float(np.max(difference)),
                "greedy_action_disagreement_count": int(np.sum(running_actions != batch_actions)),
                "greedy_action_disagreement_rate": float(np.mean(running_actions != batch_actions)),
                "per_goal": per_goal,
            })
    result = {
        "schema_version": "hackrl_online_pqn_tick_batch_renorm_diagnostic_v1",
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_or_optimizer_updates_performed": 0,
        "checkpoint_files_mutated": False,
        "fixed_observation_batch_sha256": fixed_hash,
        "fixed_observation_batch_contract": manifest["normalization_diagnostic"]["fixed_observation_batch"],
        "rows": rows,
        "all_saved_steps_below_correction_threshold": all(max(row["batch_renorm_steps"]) < 1000 for row in rows),
        "interpretation_boundary": manifest["normalization_diagnostic"]["interpretation_boundary"],
    }
    base._atomic_write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_gpu_smoke(manifest):
    run_root = Path(manifest["execution"]["run_root"])
    base._require_execution_runtime(manifest, run_root)
    base._require_clean_execution_sources()
    snapshot = base.capacity_snapshot(run_root, manifest, "before_gpu_smoke")
    base._append_jsonl(run_root / "capacity_checks.jsonl", snapshot)
    diagnostic_contract = manifest["normalization_diagnostic"]
    diagnostic = Path(diagnostic_contract["output"])
    if not diagnostic.is_file():
        raise RuntimeError("BatchRenorm diagnostic must run before GPU smoke")
    if sha256_file(diagnostic) != diagnostic_contract["output_sha256"]:
        raise RuntimeError("BatchRenorm diagnostic hash changed")
    if diagnostic.stat().st_size != diagnostic_contract["output_bytes"]:
        raise RuntimeError("BatchRenorm diagnostic size changed")
    job = jobs(manifest)[0]
    config = environment_config(job, manifest)
    config.validate()
    adapter, networks, runner, replay = base.initialize_job(job, config)
    update, _ = compiled_updates(job, adapter, networks, config)
    runner, metrics = update(runner)
    jax.block_until_ready(runner.global_update)
    counters = {
        "environment_steps": int(runner.train_state.environment_steps),
        "rollout_updates": int(runner.train_state.update_steps),
        "gradient_steps": int(runner.train_state.gradient_steps),
        "optimizer_step": int(runner.train_state.step),
        "batch_renorm_steps": batch_renorm_steps(runner.train_state.batch_stats),
    }
    training = manifest["training"]
    gradients = training["num_minibatches"] * training["update_epochs"]
    expected = {
        "environment_steps": training["rollout_transitions"],
        "rollout_updates": 1,
        "gradient_steps": gradients,
        "optimizer_step": gradients,
        "batch_renorm_steps": [gradients, gradients],
    }
    result = {
        "schema_version": "hackrl_online_pqn_tick_official_schedule_gpu_smoke_v1",
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": counters == expected and base._tree_is_finite(metrics) and base._tree_is_finite(runner.train_state),
        "counters": counters,
        "expected_counters": expected,
        "metrics": base._scalarize(metrics),
        "checkpoint_written": False,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
        "execution_source_hashes": source_hashes(),
        "runtime": base._runtime_provenance(),
    }
    base._atomic_write_json(run_root / "gpu_smoke.json", result)
    if not result["passed"]:
        raise RuntimeError("GPU smoke failed")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def adjudicate(manifest, job_rows):
    minimum = 29
    points = []
    for job, summary in job_rows:
        natural = summary["evaluation"]["natural_reset"]
        points.append({
            "seed": job["seed"],
            "success_count": natural["success_count"],
            "success_rate": natural["success_rate"],
            "mean_discounted_return": natural["mean_discounted_return"],
        })
    result = {
        "schema_version": "hackrl_online_pqn_tick_official_schedule_gate_v1",
        "execution_complete": True,
        "method": PQN,
        "environment": "tick",
        "candidate_id": manifest["candidate"]["id"],
        "seed_points": points,
        "passed": all(point["success_count"] >= minimum for point in points),
        "minimum_successes_per_32": minimum,
        "main_automatically_launched": False,
    }
    base._atomic_write_json(Path(manifest["execution"]["run_root"]) / "gate.json", result)
    return result


def final_audit(manifest, job_rows, hashes, gate):
    rows = []
    for job, summary in job_rows:
        config = environment_config(job, manifest)
        _, _, runner, replay = base.initialize_job(job, config)
        checkpoint = base._checkpoint_dir(base._cell_path(manifest["execution"]["run_root"], job), manifest["training"]["updates"])
        restored = base.load_checkpoint(checkpoint, base._payload(runner, replay))
        state = restored["runner"].train_state
        row = {
            "job_id": job["id"],
            "seed": job["seed"],
            "environment_steps": int(state.environment_steps),
            "rollout_updates": int(state.update_steps),
            "gradient_steps": int(state.gradient_steps),
            "optimizer_step": int(state.step),
            "batch_renorm_steps": batch_renorm_steps(state.batch_stats),
            "seen_goal_count": int(np.sum(np.asarray(restored["runner"].seen_goals))),
            "evaluation": summary["evaluation"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint / "state.msgpack"),
            "all_finite": base._tree_is_finite(state),
        }
        expected = manifest["training"]
        row["counter_contract_passed"] = (
            row["environment_steps"] == expected["transitions_per_job"]
            and row["rollout_updates"] == expected["updates"]
            and row["gradient_steps"] == expected["optimizer_applications_per_job"]
            and row["optimizer_step"] == expected["optimizer_applications_per_job"]
            and row["batch_renorm_steps"] == [expected["optimizer_applications_per_job"]] * 2
        )
        rows.append(row)
    result = {
        "schema_version": "hackrl_online_pqn_tick_official_schedule_result_v1",
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
        "execution_source_hashes": hashes,
        "jobs_complete": len(rows),
        "all_counter_contracts_passed": all(row["counter_contract_passed"] for row in rows),
        "gate": gate,
        "jobs": rows,
        "automatic_extension_launched": False,
    }
    base._atomic_write_json(Path(manifest["execution"]["run_root"]) / "result.json", result)
    return result


def configure_base():
    base.MANIFEST_PATH = MANIFEST_PATH
    base.SOURCE_PATHS = SOURCE_PATHS
    base.validate_manifest_contract = validate_manifest
    base._checkpoint_complete = checkpoint_complete
    base._summary_complete = summary_complete
    base._require_smoke_passed = require_smoke
    base._compiled_updates = compiled_updates
    base.environment_config = environment_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose-batch-renorm", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--job")
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--log-dir")
    parser.add_argument("--worker", default="manual")
    args = parser.parse_args()
    manifest = read_json(MANIFEST_PATH)
    validate_manifest(manifest)
    run_jobs = jobs(manifest)
    if args.list_jobs:
        print(json.dumps({"jobs": run_jobs}, indent=2, sort_keys=True))
        return
    if args.diagnose_batch_renorm:
        run_batch_renorm_diagnostic(manifest)
        return
    configure_base()
    if args.smoke:
        run_gpu_smoke(manifest)
        return
    if args.job:
        run_jobs = [job for job in run_jobs if job["id"] == args.job]
        if not run_jobs:
            raise RuntimeError(f"unknown job: {args.job}")
    expected_run_root = Path(manifest["execution"]["run_root"]).resolve()
    run_root = Path(args.log_dir or expected_run_root).resolve()
    if run_root != expected_run_root:
        raise RuntimeError("run-root overrides are forbidden")
    lock = base._acquire_runner_lock(manifest["execution"]["runner_lock"])
    hashes = source_hashes()
    base.prepare_run(run_root, manifest, jobs(manifest), hashes)
    completed = []
    for job in run_jobs:
        summary = base.run_job(run_root, manifest, job, hashes)
        completed.append((job, summary))
    if args.job:
        return
    gate = adjudicate(manifest, completed)
    result = final_audit(manifest, completed, hashes, gate)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
