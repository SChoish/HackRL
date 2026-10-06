#!/usr/bin/env python3
"""Run one resumable fixed-kernel mine-expedition learnability cell."""

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

from hackrl.mine_expedition_ppo import (
    MineExpeditionPPOConfig,
    evaluate_mine_expedition_frozen,
    initialize_mine_expedition_ppo,
    load_mine_expedition_checkpoint,
    make_mine_expedition_update,
    mine_expedition_checkpoint_files_present,
    mine_expedition_config_payload,
    mine_expedition_parameter_count,
    save_mine_expedition_checkpoint,
)


REPOSITORY = Path(__file__).resolve().parents[1]
EXECUTION_SOURCES = (
    Path("scripts/run_mine_expedition_fixed_gate.py"),
    Path("scripts/run_mine_expedition_fixed_return_curriculum_queue.sh"),
    Path("scripts/summarize_mine_expedition_fixed_return_curriculum.py"),
    Path("src/hackrl/mine_expedition.py"),
    Path("src/hackrl/mine_expedition_adjudication.py"),
    Path("src/hackrl/mine_expedition_env.py"),
    Path("src/hackrl/mine_expedition_ppo.py"),
    Path("docs/manifests/mine_expedition_fixed_return_curriculum_v1.json"),
    Path("tests/test_mine_expedition.py"),
    Path("tests/test_mine_expedition_env.py"),
    Path("tests/test_mine_expedition_gate.py"),
    Path("tests/test_mine_expedition_ppo.py"),
    Path("tests/test_mine_expedition_return_curriculum.py"),
)
GIB = 1024**3
MINIMUM_RESERVE_BYTES = 8 * GIB
LOG_AND_EVALUATION_ALLOWANCE_BYTES = 256 * 1024**2


def _git(*arguments):
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
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
            for name in ("jax", "jaxlib", "flax", "optax", "distrax", "craftax")
        },
    }


def _directory_bytes(path):
    root = Path(path)
    if not root.exists():
        return 0
    return sum(
        item.stat().st_size
        for item in root.rglob("*")
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


def _capacity_record(destination, runner, config):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_bytes = len(serialization.to_bytes(runner))
    retained = len(set(config.checkpoint_updates) | {config.num_updates})
    pending = sum(
        not _checkpoint_matches(_checkpoint_dir(destination, update), update, config)
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
    if target["free_bytes"] < record["required_free_bytes"]:
        raise RuntimeError(
            "checkpoint destination lacks projected writes plus safety reserve: "
            f"need {record['required_free_bytes']}, found {target['free_bytes']}"
        )
    resolved_destination = str(destination.resolve())
    destination_is_home = resolved_destination == "/home/ext_csv" or (
        resolved_destination.startswith("/home/ext_csv/")
    )
    if destination_is_home and target["utilization"] > 0.90:
        raise RuntimeError(
            "refusing a new checkpoint run on /home/ext_csv above 90% utilization; "
            "use /raid/ext_csv/HackRL/runs and keep a repository symlink if needed"
        )
    return record


def _append_capacity_record(destination, record, event):
    path = Path(destination) / "capacity_checks.json"
    history = []
    if path.is_file():
        history = json.loads(path.read_text(encoding="utf-8"))
    history.append({"event": event, **record})
    _write_json(path, history)


def _provenance(config, runtime, initialization):
    relative = [str(path) for path in EXECUTION_SOURCES]
    dirty = _git("status", "--porcelain", "--", *relative)
    if dirty:
        raise RuntimeError("execution sources must be committed before launch:\n" + dirty)
    tracked = set(_git("ls-files", "--", *relative).splitlines())
    missing = sorted(set(relative) - tracked)
    if missing:
        raise RuntimeError(f"execution sources are not tracked: {missing}")
    return {
        "schema_version": "hackrl_mine_expedition_fixed_run_v1",
        "status": "started",
        "execution_code_sha": _git("rev-parse", "HEAD"),
        "execution_source_sha256": {
            name: _sha256_file(REPOSITORY / name) for name in relative
        },
        "config": mine_expedition_config_payload(config),
        "kernel_variant": "fixed",
        "evaluation_start": "natural",
        "initialization": initialization,
        "runtime": runtime,
    }


_TRANSFER_COMPATIBILITY_FIELDS = (
    "seed",
    "num_envs",
    "num_steps",
    "update_epochs",
    "minibatch_size",
    "hidden_size",
    "learning_rate",
    "gamma",
    "gae_lambda",
    "clip_epsilon",
    "entropy_coefficient",
    "value_coefficient",
    "max_grad_norm",
)


def _initialize_runner(config, init_checkpoint=None):
    network, runner = initialize_mine_expedition_ppo(config)
    if init_checkpoint is None:
        return network, runner, {"kind": "random"}

    source = Path(init_checkpoint).resolve()
    recorded = json.loads((source / "config.json").read_text(encoding="utf-8"))
    source_config = MineExpeditionPPOConfig(**recorded)
    source_config.validate()
    mismatches = {
        name: (getattr(source_config, name), getattr(config, name))
        for name in _TRANSFER_COMPATIBILITY_FIELDS
        if getattr(source_config, name) != getattr(config, name)
    }
    if mismatches:
        raise ValueError(f"initialization checkpoint is incompatible: {mismatches}")
    _, source_template = initialize_mine_expedition_ppo(source_config)
    restored = load_mine_expedition_checkpoint(
        source, source_template, source_config
    )
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("variant") != "fixed"
        or metadata.get("evaluation_start") != "natural"
        or int(metadata.get("global_update", -1)) != source_config.num_updates
    ):
        raise ValueError("initialization must be a final fixed natural-eval checkpoint")
    source_manifest_path = source.parent.parent / "run_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    # Preserve the learned policy, critic, Adam state, and action RNG. Start
    # the new stage with freshly reset target-stage environments and local
    # counters so its experience budget is independently auditable.
    runner = runner.replace(
        train_state=restored.train_state,
        rng=restored.rng,
    )
    return network, runner, {
        "kind": "fixed_checkpoint_transfer",
        "checkpoint": str(source),
        "state_sha256": metadata["state_sha256"],
        "source_execution_code_sha": source_manifest.get("execution_code_sha"),
        "source_config": recorded,
        "preserved": ["policy", "critic", "adam", "action_rng"],
        "reset": [
            "environment_state",
            "environment_rng",
            "episode_counters",
            "stage_update",
            "stage_environment_steps",
        ],
    }


def _checkpoint_dir(destination, update):
    return Path(destination) / "checkpoints" / f"update_{update}"


def _checkpoint_matches(path, update, config):
    if not mine_expedition_checkpoint_files_present(path):
        return False
    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        recorded_config = json.loads(
            (path / "config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("schema_version")
        == "hackrl_mine_expedition_fixed_checkpoint_v1"
        and int(metadata.get("global_update", -1)) == update
        and int(metadata.get("environment_steps", -1)) == update * config.batch_size
        and metadata.get("variant") == "fixed"
        and metadata.get("evaluation_start") == "natural"
        and recorded_config == mine_expedition_config_payload(config)
    )


def _latest_checkpoint(destination, config):
    root = Path(destination) / "checkpoints"
    if not root.is_dir():
        return None
    candidates = []
    for path in root.glob("update_*"):
        suffix = path.name.removeprefix("update_")
        if suffix.isdigit() and int(suffix) <= config.num_updates:
            update = int(suffix)
            if _checkpoint_matches(path, update, config):
                candidates.append(update)
    return max(candidates, default=None)


def _load_checkpoint_evaluations(destination, through_update):
    evaluations = {}
    root = Path(destination) / "checkpoints"
    if not root.is_dir():
        return evaluations
    for path in root.glob("update_*"):
        suffix = path.name.removeprefix("update_")
        evaluation_path = path / "evaluation.json"
        if (
            suffix.isdigit()
            and int(suffix) <= through_update
            and evaluation_path.is_file()
        ):
            evaluations[suffix] = json.loads(
                evaluation_path.read_text(encoding="utf-8")
            )
    return evaluations


def _host_metrics(metrics, update):
    result = {
        key: np.asarray(jax.device_get(value)).tolist()
        for key, value in metrics.items()
    }
    result["update"] = int(update)
    return result


def _evaluate(network, runner, config):
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


def _save_snapshot(destination, update, network, runner, config):
    capacity = _capacity_record(destination, runner, config)
    _append_capacity_record(destination, capacity, f"before_checkpoint_{update}")
    checkpoint = save_mine_expedition_checkpoint(
        _checkpoint_dir(destination, update), runner, config
    )
    evaluation = _evaluate(network, runner, config)
    _write_json(checkpoint / "evaluation.json", evaluation)
    print(
        json.dumps(
            {
                "event": "checkpoint",
                "update": update,
                "environment_steps": int(runner.env_steps),
                "mode_success_rate": evaluation["mode"]["success_rate"],
                "sample_success_rate": evaluation["sample"]["success_rate"],
                "path": str(checkpoint),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return evaluation


def run_cell(config, log_dir=None, init_checkpoint=None):
    config.validate()
    runtime = _runtime_record()
    expected_backend = "gpu" if runtime["requested_device"] == "cuda" else "cpu"
    if runtime["jax_backend"] != expected_backend:
        raise RuntimeError(
            f"HACKRL_DEVICE={runtime['requested_device']} requested but JAX initialized "
            f"{runtime['jax_backend']} instead of {expected_backend}"
        )
    network, runner, initialization = _initialize_runner(config, init_checkpoint)
    update = jax.jit(make_mine_expedition_update(network, config))
    destination = None if log_dir is None else Path(log_dir)
    metrics_history = []
    evaluations = {}
    completed = 0
    capacity = None
    provenance = None

    if destination is not None:
        capacity = _capacity_record(destination, runner, config)
        provenance = _provenance(config, runtime, initialization)
        destination.mkdir(parents=True, exist_ok=True)
        _append_capacity_record(destination, capacity, "start_or_resume")
        manifest_path = destination / "run_manifest.json"
        if manifest_path.is_file():
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if recorded != provenance:
                raise RuntimeError("run manifest differs from the requested committed run")
        else:
            _write_json(manifest_path, provenance)
        latest = _latest_checkpoint(destination, config)
        if latest is not None:
            runner = load_mine_expedition_checkpoint(
                _checkpoint_dir(destination, latest), runner, config
            )
            completed = latest
            evaluations = _load_checkpoint_evaluations(destination, completed)
            metrics_path = destination / "updates.json"
            if metrics_path.is_file():
                metrics_history = [
                    item
                    for item in json.loads(metrics_path.read_text(encoding="utf-8"))
                    if int(item["update"]) <= completed
                ]
            expected_updates = list(range(1, completed + 1))
            recorded_updates = [int(item["update"]) for item in metrics_history]
            if recorded_updates != expected_updates:
                raise RuntimeError(
                    "checkpoint update history is missing or non-contiguous: "
                    f"expected 1..{completed}, found {recorded_updates[:3]}..."
                )
            if completed in config.checkpoint_updates and str(completed) not in evaluations:
                evaluations[str(completed)] = _evaluate(network, runner, config)
                checkpoint = _checkpoint_dir(destination, completed)
                _write_json(
                    checkpoint / "evaluation.json",
                    evaluations[str(completed)],
                )
        elif 0 in config.checkpoint_updates:
            evaluations["0"] = _save_snapshot(
                destination, 0, network, runner, config
            )

    started = time.perf_counter()
    first_update_seconds = None
    for index in range(completed, config.num_updates):
        update_started = time.perf_counter()
        runner, metrics = update(runner)
        jax.block_until_ready(runner.train_state.params)
        elapsed = time.perf_counter() - update_started
        if first_update_seconds is None:
            first_update_seconds = elapsed
        finished = index + 1
        metrics_history.append(_host_metrics(metrics, finished))
        if destination is not None:
            _write_json(destination / "updates.json", metrics_history)
            if finished % 64 == 0:
                periodic_capacity = _capacity_record(destination, runner, config)
                _append_capacity_record(
                    destination,
                    periodic_capacity,
                    f"periodic_update_{finished}",
                )
            if finished in config.checkpoint_updates:
                evaluations[str(finished)] = _save_snapshot(
                    destination, finished, network, runner, config
                )
        print(
            f"[update] {finished}/{config.num_updates} seconds={elapsed:.3f}",
            flush=True,
        )

    training_seconds = time.perf_counter() - started
    final_evaluation = evaluations.get(str(config.num_updates))
    if final_evaluation is None:
        final_evaluation = _evaluate(network, runner, config)
        evaluations[str(config.num_updates)] = final_evaluation
    if destination is not None and not mine_expedition_checkpoint_files_present(
        _checkpoint_dir(destination, config.num_updates)
    ):
        final_capacity = _capacity_record(destination, runner, config)
        _append_capacity_record(
            destination, final_capacity, "before_final_checkpoint"
        )
        save_mine_expedition_checkpoint(
            _checkpoint_dir(destination, config.num_updates), runner, config
        )
        _write_json(
            _checkpoint_dir(destination, config.num_updates) / "evaluation.json",
            final_evaluation,
        )

    steady_seconds = training_seconds - (first_update_seconds or 0.0)
    remaining_transitions = max(config.num_updates - completed - 1, 0) * config.batch_size
    summary = {
        "schema_version": "hackrl_mine_expedition_fixed_result_v1",
        "status": "complete",
        "seed": config.seed,
        "variant": "fixed",
        "training_start": config.training_start,
        "evaluation_start": "natural",
        "updates": config.num_updates,
        "transitions": config.num_updates * config.batch_size,
        "resumed_from_update": completed,
        "parameter_count": mine_expedition_parameter_count(runner.train_state.params),
        "training_seconds_this_invocation": training_seconds,
        "compile_and_first_update_seconds": first_update_seconds,
        "steady_state_transitions_per_second": (
            None
            if remaining_transitions == 0
            else remaining_transitions / max(steady_seconds, 1e-9)
        ),
        "final_evaluation": final_evaluation,
        "checkpoint_evaluations": evaluations,
        "capacity": capacity,
        "execution_code_sha": (
            None if provenance is None else provenance["execution_code_sha"]
        ),
        "runtime": runtime,
        "initialization": initialization,
    }
    if destination is not None:
        _write_json(destination / "summary.json", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--num-updates", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--training-start",
        choices=(
            "curriculum",
            "natural",
            "natural_late",
            "craft_ready",
            "target_ready",
            "return_near",
            "return_path",
            "mine_return",
            "craft_mine_return",
            "natural_return",
        ),
        default="curriculum",
    )
    parser.add_argument("--mode-eval-episodes", type=int, default=1)
    parser.add_argument("--sample-eval-episodes", type=int, default=128)
    parser.add_argument(
        "--checkpoint-updates", default="0,128,512,1024,2048"
    )
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = MineExpeditionPPOConfig(
        seed=args.seed,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_updates=args.num_updates,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        training_start=args.training_start,
        mode_eval_episodes=args.mode_eval_episodes,
        sample_eval_episodes=args.sample_eval_episodes,
        checkpoint_updates=tuple(
            int(item) for item in args.checkpoint_updates.split(",") if item
        ),
    )
    print(
        json.dumps(
            run_cell(config, args.log_dir, args.init_checkpoint),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
