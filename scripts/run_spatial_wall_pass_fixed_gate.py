#!/usr/bin/env python3
"""Run one resumable fixed-normal-learning SPATIAL-WALL-PASS cell."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

REQUESTED_DEVICE = os.environ.get("HACKRL_DEVICE", "cpu")
if REQUESTED_DEVICE not in {"cpu", "cuda"}:
    raise RuntimeError("HACKRL_DEVICE must be 'cpu' or 'cuda'")
if REQUESTED_DEVICE == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import numpy as np
from flax import serialization

from hackrl.dual_leo import (
    init_dual_leo_teacher,
    load_dual_checkpoint,
    make_dual_leo_update,
    save_dual_checkpoint,
    teacher_parameter_count,
)
from hackrl.spatial_wall_pass import SpatialWallPassSplit, SpatialWallPassVariant
from hackrl.spatial_wall_pass_gc import (
    NUM_ACTIONS,
    NUM_GOALS,
    SpatialWallPassGCConfig,
    _batch_inputs,
    evaluate_spatial_wall_pass_gc_frozen,
    initialize_spatial_wall_pass_gc,
    load_spatial_wall_pass_gc_checkpoint,
    make_spatial_wall_pass_gc_update,
    save_spatial_wall_pass_gc_checkpoint,
    spatial_wall_pass_gc_config_payload,
    spatial_wall_pass_gc_parameter_count,
    spatial_wall_pass_outcome,
)

REPOSITORY = Path(__file__).resolve().parents[1]
RUN_ID = "spatial_wall_pass_fixed_learnability_v1"
RUN_ROOT = Path("/raid/ext_csv/HackRL/runs") / RUN_ID
EXECUTION_SOURCES = (
    Path("scripts/run_spatial_wall_pass_fixed_gate.py"),
    Path("scripts/run_spatial_wall_pass_fixed_gate_queue.sh"),
    Path("src/hackrl/spatial_wall_pass.py"),
    Path("src/hackrl/spatial_wall_pass_oracle.py"),
    Path("src/hackrl/spatial_wall_pass_gc.py"),
    Path("src/hackrl/dual_leo.py"),
    Path("docs/manifests/spatial_wall_pass_v1.json"),
    Path("docs/manifests/spatial_wall_pass_fixed_learnability_v1.json"),
    Path("tests/test_spatial_wall_pass.py"),
    Path("tests/test_spatial_wall_pass_gc.py"),
)
GIB = 1024**3
MINIMUM_RESERVE_BYTES = 8 * GIB
LOG_AND_EVALUATION_ALLOWANCE_BYTES = 512 * 1024**2
CHECKPOINT_UPDATES = (0, 32, 128, 256, 512)


def _git(*arguments):
    result = subprocess.run(
        ["git", *arguments], cwd=REPOSITORY, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _runtime_record():
    return {
        "requested_device": REQUESTED_DEVICE,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "flax", "optax", "distrax")
        },
    }


def _directory_bytes(path):
    root = Path(path)
    if not root.exists():
        return 0
    return sum(
        item.stat().st_size for item in root.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def _filesystem_record(path):
    usage = shutil.disk_usage(path)
    return {
        "path": str(Path(path).resolve()),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "utilization": usage.used / usage.total,
    }


def _state_bytes(method, runner, teacher_state):
    state = runner if method == "gc" else {"gc": runner, "leo": teacher_state}
    return len(serialization.to_bytes(state))


def _checkpoint_dir(destination, update):
    return Path(destination) / "checkpoints" / f"update_{update}"


def _checkpoint_matches(path, update, method, config):
    path = Path(path)
    required = ("state.msgpack", "config.json", "metadata.json")
    if not all((path / name).is_file() for name in required):
        return False
    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        recorded = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    schema = (
        "hackrl_spatial_wall_pass_gc_checkpoint_v1"
        if method == "gc" else "hackrl_dual_leo_checkpoint_v1"
    )
    return (
        metadata.get("schema_version") == schema
        and int(metadata.get("global_update", -1)) == update
        and recorded == spatial_wall_pass_gc_config_payload(config)
    )


def _capacity_record(destination, method, runner, teacher_state, config):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_bytes = _state_bytes(method, runner, teacher_state)
    retained = len(set(config.checkpoint_updates) | {config.num_updates})
    pending = sum(
        not _checkpoint_matches(_checkpoint_dir(destination, update), update, method, config)
        for update in set(config.checkpoint_updates) | {config.num_updates}
    )
    overlap = 2 if pending else 0
    projected_remaining = (
        checkpoint_bytes * (pending + overlap) + LOG_AND_EVALUATION_ALLOWANCE_BYTES
    )
    reserve = max(MINIMUM_RESERVE_BYTES, int(np.ceil(0.2 * projected_remaining)))
    target = _filesystem_record(destination.parent)
    home = _filesystem_record("/home/ext_csv")
    raid = _filesystem_record("/raid/ext_csv")
    record = {
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "destination": str(destination.resolve()),
        "existing_run_bytes": _directory_bytes(destination),
        "serialized_checkpoint_bytes": checkpoint_bytes,
        "retained_checkpoint_count": retained,
        "pending_checkpoint_count": pending,
        "active_predecessor_successor_overlap_count": overlap,
        "log_and_evaluation_allowance_bytes": LOG_AND_EVALUATION_ALLOWANCE_BYTES,
        "projected_remaining_write_bytes": projected_remaining,
        "safety_reserve_bytes": reserve,
        "required_free_bytes": projected_remaining + reserve,
        "target_filesystem": target,
        "home_filesystem": home,
        "raid_filesystem": raid,
    }
    if not str(destination.resolve()).startswith("/raid/ext_csv/HackRL/runs/"):
        raise RuntimeError("checkpoint-producing gate runs must target /raid/ext_csv/HackRL/runs")
    if target["free_bytes"] < record["required_free_bytes"]:
        raise RuntimeError("checkpoint destination lacks projected writes plus reserve")
    return record


def _append_capacity_record(destination, record, event):
    path = Path(destination) / "capacity_checks.json"
    history = []
    if path.is_file():
        history = json.loads(path.read_text(encoding="utf-8"))
    history.append({"event": event, **record})
    _write_json(path, history)


def _provenance(method, config, runtime, parameter_counts):
    relative = [str(path) for path in EXECUTION_SOURCES]
    dirty = _git("status", "--porcelain", "--", *relative)
    if dirty:
        raise RuntimeError("execution sources must be committed before launch:\n" + dirty)
    tracked = set(_git("ls-files", "--", *relative).splitlines())
    missing = sorted(set(relative) - tracked)
    if missing:
        raise RuntimeError(f"execution sources are not tracked: {missing}")
    return {
        "schema_version": "hackrl_spatial_wall_pass_fixed_run_v1",
        "status": "started",
        "run_id": RUN_ID,
        "method": method,
        "execution_code_sha": _git("rev-parse", "HEAD"),
        "execution_source_sha256": {
            name: _sha256_file(REPOSITORY / name) for name in relative
        },
        "config": spatial_wall_pass_gc_config_payload(config),
        "training_kernel": "fixed",
        "training_split": "train",
        "training_start": "natural",
        "evaluation_start": "natural",
        "evaluation_kernels": ["fixed", "mutant"],
        "evaluation_split": config.evaluation_split,
        "parameter_counts": parameter_counts,
        "runtime": runtime,
    }


def _initialize(method, config):
    network, runner = initialize_spatial_wall_pass_gc(config)
    teacher_network = None
    teacher_state = None
    teacher_minibatch = None
    if method == "dual":
        inputs = _batch_inputs(runner.env_state, runner.current_goal)
        teacher_network, teacher_state, teacher_minibatch = init_dual_leo_teacher(
            config, inputs[0], inputs[1], NUM_GOALS, NUM_ACTIONS
        )
    return network, runner, teacher_network, teacher_state, teacher_minibatch


def _parameter_counts(method, runner, teacher_state):
    result = {"ppo_actor_critic": spatial_wall_pass_gc_parameter_count(runner.train_state.params)}
    if method == "dual":
        result["dual_teacher"] = teacher_parameter_count(teacher_state)
    return result


def _save_checkpoint(path, method, runner, teacher_state, config):
    if method == "gc":
        return save_spatial_wall_pass_gc_checkpoint(path, runner, config)
    return save_dual_checkpoint(
        path, runner, teacher_state, spatial_wall_pass_gc_config_payload(config)
    )


def _load_checkpoint(path, method, runner, teacher_state, config):
    if method == "gc":
        return load_spatial_wall_pass_gc_checkpoint(path, runner, config), teacher_state
    return load_dual_checkpoint(path, runner, teacher_state)


def _latest_checkpoint(destination, method, config):
    root = Path(destination) / "checkpoints"
    if not root.is_dir():
        return None
    candidates = []
    for path in root.glob("update_*"):
        suffix = path.name.removeprefix("update_")
        if suffix.isdigit() and int(suffix) <= config.num_updates:
            update = int(suffix)
            if _checkpoint_matches(path, update, method, config):
                candidates.append(update)
    return max(candidates, default=None)


def _load_evaluations(destination, through_update):
    evaluations = {}
    root = Path(destination) / "checkpoints"
    if not root.is_dir():
        return evaluations
    for path in root.glob("update_*"):
        suffix = path.name.removeprefix("update_")
        evaluation = path / "evaluation.json"
        if suffix.isdigit() and int(suffix) <= through_update and evaluation.is_file():
            evaluations[suffix] = json.loads(evaluation.read_text(encoding="utf-8"))
    return evaluations


def _evaluate(network, runner, config, update):
    before = serialization.to_bytes(runner)
    results = {}
    for variant in (SpatialWallPassVariant.FIXED, SpatialWallPassVariant.MUTANT):
        results[variant.value] = {}
        for stochastic, label, repeats, base in (
            (False, "mode", config.mode_repeats_per_state, 60000),
            (True, "sample", config.sample_repeats_per_state, 61000),
        ):
            results[variant.value][label] = evaluate_spatial_wall_pass_gc_frozen(
                network,
                runner.train_state.params,
                variant=variant.value,
                stochastic=stochastic,
                repeats_per_state=repeats,
                seed_base=base,
                learner_seed=config.seed,
                split=config.evaluation_split,
                record_episodes=True,
            )
    results["training_update"] = int(update)
    results["adaptation_update"] = 0
    results["runner_state_immutable"] = before == serialization.to_bytes(runner)
    return results


def _save_snapshot(destination, update, method, network, runner, teacher_state, config):
    capacity = _capacity_record(destination, method, runner, teacher_state, config)
    _append_capacity_record(destination, capacity, f"before_checkpoint_{update}")
    checkpoint = _save_checkpoint(
        _checkpoint_dir(destination, update), method, runner, teacher_state, config
    )
    evaluation = _evaluate(network, runner, config, update)
    _write_json(checkpoint / "evaluation.json", evaluation)
    print(json.dumps({
        "event": "checkpoint",
        "method": method,
        "seed": config.seed,
        "update": update,
        "environment_steps": int(runner.env_steps),
        "fixed_mode_success_rate": evaluation["fixed"]["mode"]["success_rate"],
        "fixed_sample_success_rate": evaluation["fixed"]["sample"]["success_rate"],
        "mutant_mode_wall_pass_rate": evaluation["mutant"]["mode"]["wall_pass_rate"],
        "path": str(checkpoint),
    }, sort_keys=True), flush=True)
    return evaluation


def _host_metrics(metrics, update):
    result = {
        key: np.asarray(jax.device_get(value)).tolist()
        for key, value in metrics.items()
    }
    result["update"] = int(update)
    return result


def _assert_preregistered_contract(method, config, destination):
    expected = {
        "variant": "fixed",
        "seed": config.seed,
        "num_envs": 512,
        "num_steps": 64,
        "num_updates": 512,
        "update_epochs": 1,
        "minibatch_size": 1024,
        "policy_hidden_size": 512,
        "teacher_hidden_size": 512,
        "learning_rate": 2e-4,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "entropy_coefficient": 0.005,
        "value_coefficient": 0.5,
        "max_grad_norm": 1.0,
        "goal_mode": "normal12",
        "evaluation_split": "validation",
        "mode_repeats_per_state": 1,
        "sample_repeats_per_state": 64,
        "checkpoint_updates": CHECKPOINT_UPDATES,
    }
    if config.seed not in {130, 131}:
        raise ValueError("checkpoint-producing gate seed must be 130 or 131")
    mismatches = {
        name: (getattr(config, name), value)
        for name, value in expected.items()
        if getattr(config, name) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint-producing run differs from preregistration: {mismatches}")
    expected_destination = (RUN_ROOT / f"{method}_seed_{config.seed}").resolve()
    if destination.resolve() != expected_destination:
        raise ValueError(
            f"gate destination must be {expected_destination}, got {destination.resolve()}"
        )


def run_cell(method, config, log_dir=None):
    if method not in {"gc", "dual"}:
        raise ValueError("method must be gc or dual")
    config.validate()
    if config.variant != "fixed" or config.goal_mode != "normal12":
        raise ValueError("this gate trains fixed normal12 only")
    destination = None if log_dir is None else Path(log_dir)
    if destination is not None:
        _assert_preregistered_contract(method, config, destination)
    runtime = _runtime_record()
    expected_backend = "gpu" if REQUESTED_DEVICE == "cuda" else "cpu"
    if runtime["jax_backend"] != expected_backend:
        raise RuntimeError(
            f"requested {REQUESTED_DEVICE}, got JAX backend {runtime['jax_backend']}"
        )
    network, runner, teacher_network, teacher_state, teacher_minibatch = _initialize(
        method, config
    )
    if method == "gc":
        update_fn = jax.jit(make_spatial_wall_pass_gc_update(network, config))
    else:
        update_fn = jax.jit(make_dual_leo_update(
            network,
            teacher_network,
            config,
            spatial_wall_pass_outcome,
            _batch_inputs,
            teacher_minibatch,
        ))
    metrics_history = []
    evaluations = {}
    completed = 0
    capacity = None
    provenance = None
    counts = _parameter_counts(method, runner, teacher_state)

    if destination is not None:
        capacity = _capacity_record(destination, method, runner, teacher_state, config)
        provenance = _provenance(method, config, runtime, counts)
        destination.mkdir(parents=True, exist_ok=True)
        _append_capacity_record(destination, capacity, "start_or_resume")
        manifest_path = destination / "run_manifest.json"
        if manifest_path.is_file():
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if recorded != provenance:
                raise RuntimeError("run manifest differs from requested committed run")
        else:
            _write_json(manifest_path, provenance)
        latest = _latest_checkpoint(destination, method, config)
        if latest is not None:
            runner, teacher_state = _load_checkpoint(
                _checkpoint_dir(destination, latest), method, runner, teacher_state, config
            )
            completed = latest
            evaluations = _load_evaluations(destination, completed)
            metrics_path = destination / "updates.json"
            if metrics_path.is_file():
                metrics_history = [
                    item for item in json.loads(metrics_path.read_text(encoding="utf-8"))
                    if int(item["update"]) <= completed
                ]
            if [int(item["update"]) for item in metrics_history] != list(range(1, completed + 1)):
                raise RuntimeError("checkpoint update history is missing or non-contiguous")
            if completed in config.checkpoint_updates and str(completed) not in evaluations:
                evaluations[str(completed)] = _save_snapshot(
                    destination, completed, method, network, runner, teacher_state, config
                )
        elif 0 in config.checkpoint_updates:
            evaluations["0"] = _save_snapshot(
                destination, 0, method, network, runner, teacher_state, config
            )

    started = time.perf_counter()
    first_update_seconds = None
    steady_update_seconds = []
    for index in range(completed, config.num_updates):
        update_started = time.perf_counter()
        if method == "gc":
            runner, metrics = update_fn(runner)
        else:
            runner, teacher_state, metrics = update_fn(runner, teacher_state)
        jax.block_until_ready(runner.train_state.params)
        elapsed = time.perf_counter() - update_started
        if first_update_seconds is None:
            first_update_seconds = elapsed
        else:
            steady_update_seconds.append(elapsed)
        finished = index + 1
        metrics_history.append(_host_metrics(metrics, finished))
        if destination is not None:
            _write_json(destination / "updates.json", metrics_history)
            if finished % 64 == 0:
                periodic = _capacity_record(
                    destination, method, runner, teacher_state, config
                )
                _append_capacity_record(destination, periodic, f"periodic_update_{finished}")
            if finished in config.checkpoint_updates:
                evaluations[str(finished)] = _save_snapshot(
                    destination, finished, method, network, runner, teacher_state, config
                )
        rate = None if not steady_update_seconds else config.batch_size / np.mean(steady_update_seconds[-8:])
        print(json.dumps({
            "event": "update",
            "method": method,
            "seed": config.seed,
            "update": finished,
            "updates_total": config.num_updates,
            "seconds": elapsed,
            "recent_transitions_per_second": rate,
        }, sort_keys=True), flush=True)

    training_seconds = time.perf_counter() - started
    final_evaluation = evaluations.get(str(config.num_updates))
    if final_evaluation is None:
        final_evaluation = _save_snapshot(
            destination, config.num_updates, method, network, runner, teacher_state, config
        ) if destination is not None else _evaluate(network, runner, config, config.num_updates)
        evaluations[str(config.num_updates)] = final_evaluation
    summary = {
        "schema_version": "hackrl_spatial_wall_pass_fixed_result_v1",
        "status": "complete",
        "method": method,
        "seed": config.seed,
        "updates": config.num_updates,
        "transitions": config.num_updates * config.batch_size,
        "resumed_from_update": completed,
        "parameter_counts": _parameter_counts(method, runner, teacher_state),
        "training_seconds_this_invocation": training_seconds,
        "compile_and_first_update_seconds": first_update_seconds,
        "steady_state_transitions_per_second": (
            None if not steady_update_seconds
            else config.batch_size / float(np.mean(steady_update_seconds))
        ),
        "final_evaluation": final_evaluation,
        "checkpoint_evaluations": evaluations,
        "capacity": capacity,
        "execution_code_sha": None if provenance is None else provenance["execution_code_sha"],
        "runtime": runtime,
    }
    if destination is not None:
        _write_json(destination / "summary.json", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("gc", "dual"), required=True)
    parser.add_argument("--seed", type=int, choices=(130, 131), required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--num-updates", type=int, default=512)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--policy-hidden-size", type=int, default=512)
    parser.add_argument("--teacher-hidden-size", type=int, default=512)
    parser.add_argument("--mode-repeats", type=int, default=1)
    parser.add_argument("--sample-repeats", type=int, default=64)
    parser.add_argument("--checkpoint-updates", default="0,32,128,256,512")
    parser.add_argument("--log-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = SpatialWallPassGCConfig(
        variant=SpatialWallPassVariant.FIXED.value,
        seed=args.seed,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_updates=args.num_updates,
        update_epochs=1,
        minibatch_size=args.minibatch_size,
        policy_hidden_size=args.policy_hidden_size,
        teacher_hidden_size=args.teacher_hidden_size,
        learning_rate=2e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coefficient=0.005,
        value_coefficient=0.5,
        max_grad_norm=1.0,
        goal_mode="normal12",
        evaluation_split=SpatialWallPassSplit.VALIDATION.value,
        mode_repeats_per_state=args.mode_repeats,
        sample_repeats_per_state=args.sample_repeats,
        checkpoint_updates=tuple(
            int(item) for item in args.checkpoint_updates.split(",") if item
        ),
    )
    summary = run_cell(args.method, config, args.log_dir)
    print(
        json.dumps(
            {
                "event": "complete",
                "status": summary["status"],
                "method": summary["method"],
                "seed": summary["seed"],
                "updates": summary["updates"],
                "transitions": summary["transitions"],
                "steady_state_transitions_per_second": summary[
                    "steady_state_transitions_per_second"
                ],
                "result_path": (
                    None if args.log_dir is None
                    else str(Path(args.log_dir).resolve() / "summary.json")
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
