#!/usr/bin/env python3
"""Run the preregistered fixed-normal online-algorithm development matrix."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

if os.environ.get("HACKRL_DEVICE", "cpu") == "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
else:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
from flax import serialization

from hackrl.discrete_sac import (
    init_sd_sac_replay,
    make_sd_sac_replay_updates,
    valid_sd_sac_replay_count,
)
from hackrl.online_algorithm_env import (
    DUAL,
    LEO,
    PQN,
    SD_SAC,
    environment_adapter,
    frozen_evaluation_binding,
    initialize_online_value_runner,
    initialize_sd_sac_runner,
    make_online_value_update,
    make_sd_sac_collection,
)
from hackrl.pack_restore import PackRestoreVariant
from hackrl.pack_restore_gc import (
    PackRestoreGCConfig,
    evaluate_pack_restore_gc_frozen,
)
from hackrl.tick_claim import TickClaimVariant
from hackrl.tick_claim_gc import (
    TickClaimGCConfig,
    evaluate_tick_claim_gc_frozen,
)


REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    REPOSITORY
    / "docs/manifests/online_algorithm_expansion_v1_development.json"
)
METHOD_KEYS = {
    "GC-PQN": PQN,
    "LEO": LEO,
    "Dual LEO(PQN)": DUAL,
    "GC-SD-SAC": SD_SAC,
}
METHOD_SLUGS = {
    PQN: "pqn",
    LEO: "leo",
    DUAL: "dual",
    SD_SAC: "sd-sac",
}
SOURCE_PATHS = (
    "docs/manifests/online_algorithm_expansion_v1.json",
    "docs/manifests/online_algorithm_expansion_v1_implementation.json",
    "docs/manifests/online_algorithm_expansion_v1_smoke.json",
    "docs/manifests/online_algorithm_expansion_v1_development.json",
    "scripts/run_online_algorithm_development.py",
    "scripts/run_online_algorithm_development_gpu_queue.sh",
    "src/hackrl/online_value.py",
    "src/hackrl/discrete_sac.py",
    "src/hackrl/online_algorithm_env.py",
    "src/hackrl/tick_claim.py",
    "src/hackrl/tick_claim_gc.py",
    "src/hackrl/pack_restore.py",
    "src/hackrl/pack_restore_gc.py",
)
PROJECT_MODULES = (
    "hackrl.discrete_sac",
    "hackrl.online_algorithm_env",
    "hackrl.online_value",
    "hackrl.pack_restore",
    "hackrl.pack_restore_gc",
    "hackrl.tick_claim",
    "hackrl.tick_claim_gc",
)


def _require_keys(mapping, keys, context):
    missing = sorted(set(keys) - set(mapping))
    if missing:
        raise RuntimeError(f"{context} is missing required keys: {missing}")


def validate_manifest_contract(manifest):
    _require_keys(
        manifest,
        (
            "common_training",
            "candidate_registry",
            "budget",
            "capacity_preflight",
            "execution",
        ),
        "development manifest",
    )
    common = manifest["common_training"]
    execution = manifest["execution"]
    evaluation = common["evaluation"]
    if common["kernel"] != "fixed" or common["goal_mode"] != "workshop12":
        raise RuntimeError("development must use fixed workshop12 training")
    if evaluation["split"] != "validation":
        raise RuntimeError("development evaluation split must be validation")
    if (
        evaluation["natural_start_episodes"] != 32
        or evaluation["common_setup_diagnostic_episodes"] != 32
    ):
        raise RuntimeError("development evaluation must use 32 episodes per family")
    if evaluation["gate"].find("29/32") < 0:
        raise RuntimeError("development gate must require 29 of 32 successes")
    if int(common["num_envs"]) * int(common["num_steps"]) != 32768:
        raise RuntimeError("one development update must contain 32,768 transitions")
    transitions_per_job = (
        int(common["num_envs"])
        * int(common["num_steps"])
        * int(common["updates"])
    )
    if transitions_per_job != int(common["transitions_per_job"]):
        raise RuntimeError("development transition arithmetic is inconsistent")
    checkpoint_updates = tuple(execution["checkpoint_updates"])
    if checkpoint_updates[0] != 0 or checkpoint_updates[-1] != common["updates"]:
        raise RuntimeError("checkpoint schedule must retain update 0 and the final update")
    if sorted(set(checkpoint_updates)) != list(checkpoint_updates):
        raise RuntimeError("checkpoint updates must be unique and increasing")
    if METHOD_KEYS != {
        "GC-PQN": "GC-PQN",
        "LEO": "LEO",
        "Dual LEO(PQN)": "Dual LEO(PQN)",
        "GC-SD-SAC": "GC-SD-SAC",
    }:
        raise RuntimeError("imported method constants changed")
    online_required = {
        "id",
        "pqn_hidden_size",
        "leo_hidden_size",
        "learning_rate",
        "epsilon_start",
        "epsilon_finish",
        "epsilon_decay_transitions",
    }
    sac_required = {
        "id",
        "hidden_size",
        "actor_learning_rate",
        "critic_learning_rate",
        "temperature_learning_rate",
        "initial_alpha",
        "beta",
        "q_clip_range",
        "polyak_tau",
        "target_entropy_fraction_of_log_actions",
        "replay_capacity",
        "replay_batch_size",
        "replay_update_iterations_per_rollout",
    }
    for registered_name in METHOD_KEYS:
        candidates = manifest["candidate_registry"].get(registered_name)
        if not isinstance(candidates, list) or len(candidates) != 2:
            raise RuntimeError(f"{registered_name} must preregister exactly two candidates")
        required = sac_required if registered_name == "GC-SD-SAC" else online_required
        for candidate in candidates:
            _require_keys(candidate, required, f"{registered_name} candidate")
        identifiers = [candidate.get("id") for candidate in candidates]
        if len(set(identifiers)) != len(identifiers):
            raise RuntimeError(f"{registered_name} candidate IDs must be unique")
    if any(
        candidate.get("acting") != "0.7*Q_PQN + 0.3*Q_LEO"
        for candidate in manifest["candidate_registry"]["Dual LEO(PQN)"]
    ):
        raise RuntimeError("Dual LEO(PQN) acting mixture changed")
    expected_jobs = len(common["learner_seeds"]) * 2 * len(METHOD_KEYS) * 2
    if int(manifest["budget"]["jobs"]) != expected_jobs:
        raise RuntimeError("development job-count budget is inconsistent")
    if int(manifest["budget"]["training_transitions"]) != (
        expected_jobs * transitions_per_job
    ):
        raise RuntimeError("development transition budget is inconsistent")


def _require_smoke_passed(manifest):
    smoke_path = REPOSITORY / manifest["smoke_manifest"]
    smoke = _read_json(smoke_path)
    if smoke.get("status") != "passed":
        raise RuntimeError("the CPU/GPU smoke prerequisite has not passed")
    results = smoke.get("results", {})
    if results.get("cpu", {}).get("cells_passed") != 8:
        raise RuntimeError("CPU smoke result is not 8/8")
    if results.get("gpu", {}).get("cells_passed") != 8:
        raise RuntimeError("GPU smoke result is not 8/8")
    if results.get("gpu", {}).get("checkpoint_written") is not False:
        raise RuntimeError("GPU smoke checkpoint contract changed")


def _runtime_provenance():
    packages = {}
    for package in ("jax", "jaxlib", "flax", "optax", "numpy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "packages": packages,
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "project_module_origins": {
            name: str(Path(sys.modules[name].__file__).resolve())
            for name in PROJECT_MODULES
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "HACKRL_DEVICE",
                "JAX_PLATFORMS",
                "CUDA_VISIBLE_DEVICES",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "HACKRL_DUMMY_KEEPALIVE_CLEARED",
                "PYTHONPATH",
            )
        },
    }


def _require_execution_runtime(manifest, run_root):
    execution = manifest["execution"]
    expected_python = Path(execution["python"]).resolve()
    expected_worktree = Path(execution["clean_worktree_root"]).resolve()
    expected_run_root = Path(execution["run_root"]).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError("development must use the preregistered Python interpreter")
    if REPOSITORY.resolve() != expected_worktree:
        raise RuntimeError("development must run from the preregistered clean worktree")
    if Path(run_root).resolve() != expected_run_root:
        raise RuntimeError("run-root overrides are forbidden for this experiment")
    run_root = Path(run_root)
    if run_root.is_symlink():
        raise RuntimeError("development run root must not be a symlink")
    run_root.mkdir(parents=True, exist_ok=True)
    if os.stat(run_root).st_dev != os.stat("/raid/ext_csv").st_dev:
        raise RuntimeError("development run root is not on the RAID filesystem")
    required_environment = execution["device_environment"]
    for key, expected in required_environment.items():
        if os.environ.get(key) != expected:
            raise RuntimeError(f"{key} must be {expected!r} for development training")
    if jax.default_backend() != "gpu":
        raise RuntimeError("development training requires the JAX GPU backend")
    if not jax.devices() or all(device.platform != "gpu" for device in jax.devices()):
        raise RuntimeError("no JAX GPU device is visible")
    for name in PROJECT_MODULES:
        origin = Path(sys.modules[name].__file__).resolve()
        if not origin.is_relative_to(REPOSITORY.resolve()):
            raise RuntimeError(f"imported project module is outside the worktree: {name}")


def _acquire_runner_lock(lock_path):
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError("another development runner already owns the run root") from error
    handle.seek(0)
    handle.truncate()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    handle.write(f"pid={os.getpid()} started_utc={started}\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_write_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_write_json(path, payload):
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path):
    return _sha256_bytes(Path(path).read_bytes())


def execution_source_hashes():
    return {path: _sha256_file(REPOSITORY / path) for path in SOURCE_PATHS}


def _git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()


def _require_clean_execution_sources():
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *SOURCE_PATHS],
        cwd=REPOSITORY,
        text=True,
    ).strip()
    if output:
        raise RuntimeError(
            "execution sources must be committed in the clean run worktree:\n"
            + output
        )


def build_jobs(manifest):
    jobs = []
    seeds = manifest["common_training"]["learner_seeds"]
    registry = manifest["candidate_registry"]
    for seed in seeds:
        for environment in ("tick", "pack"):
            for registered_name, method in METHOD_KEYS.items():
                for candidate in registry[registered_name]:
                    jobs.append(
                        {
                            "id": (
                                f"{environment}-{METHOD_SLUGS[method]}-"
                                f"{candidate['id']}-s{seed}"
                            ),
                            "environment": environment,
                            "method": method,
                            "registered_name": registered_name,
                            "candidate_id": candidate["id"],
                            "candidate": candidate,
                            "seed": int(seed),
                        }
                    )
    return jobs


def _tree_bytes(tree):
    return int(
        sum(np.asarray(leaf).nbytes for leaf in jax.tree.leaves(jax.device_get(tree)))
    )


def _tree_is_finite(tree):
    return all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree.leaves(jax.device_get(tree))
    )


def _scalarize(value):
    if isinstance(value, dict):
        return {key: _scalarize(item) for key, item in value.items()}
    array = np.asarray(jax.device_get(value))
    if array.ndim == 0:
        return array.item()
    return {"shape": list(array.shape), "finite": bool(np.all(np.isfinite(array)))}


def _parameter_counts(method, runner):
    state = runner.train_state

    def count(tree):
        return int(sum(np.asarray(leaf).size for leaf in jax.tree.leaves(tree)))

    if method == PQN:
        return {"pqn": count(state.params)}
    if method == LEO:
        return {"leo": count(state.params)}
    if method == DUAL:
        return {"pqn": count(state.pqn.params), "leo": count(state.leo.params)}
    return {
        "actor": count(state.actor.params),
        "critic_1": count(state.critic_1.params),
        "critic_2": count(state.critic_2.params),
        "target_critic_1": count(state.critic_1.target_params),
        "target_critic_2": count(state.critic_2.target_params),
        "temperature": count(state.temperature.params),
    }


def environment_config(job, manifest):
    common = manifest["common_training"]
    if common["kernel"] != "fixed" or common["goal_mode"] != "workshop12":
        raise RuntimeError("unexpected development environment contract")
    kwargs = {
        "variant": common["kernel"],
        "seed": job["seed"],
        "num_envs": common["num_envs"],
        "num_steps": common["num_steps"],
        "num_updates": common["updates"],
        "update_epochs": 1,
        "minibatch_size": 1024,
        "learning_rate": 2e-4,
        "gamma": common["gamma"],
        "max_grad_norm": common["max_grad_norm"],
        "goal_mode": common["goal_mode"],
        "evaluation_split": common["evaluation"]["split"],
        "mode_repeats_per_state": 1,
        "sample_repeats_per_state": 1,
    }
    if job["environment"] == "tick":
        return TickClaimGCConfig(hidden_size=8, **kwargs)
    return PackRestoreGCConfig(
        policy_hidden_size=8, teacher_hidden_size=8, **kwargs
    )


def initialize_job(job, config):
    adapter = environment_adapter(job["environment"])
    candidate = job["candidate"]
    method = job["method"]
    if method == SD_SAC:
        networks, runner = initialize_sd_sac_runner(
            adapter,
            config,
            hidden_size=candidate["hidden_size"],
            actor_learning_rate=candidate["actor_learning_rate"],
            critic_learning_rate=candidate["critic_learning_rate"],
            temperature_learning_rate=candidate["temperature_learning_rate"],
            initial_alpha=candidate["initial_alpha"],
            max_grad_norm=config.max_grad_norm,
        )
        inputs = adapter.batch_inputs(runner.env_state, runner.current_goal)
        replay = init_sd_sac_replay(
            candidate["replay_capacity"],
            tuple(inputs[0].shape[1:]),
            tuple(inputs[1].shape[1:]),
            (adapter.num_goals,),
        )
        return adapter, networks, runner, replay
    networks, runner = initialize_online_value_runner(
        adapter,
        config,
        method=method,
        pqn_hidden_size=candidate["pqn_hidden_size"],
        leo_hidden_size=candidate["leo_hidden_size"],
        learning_rate=candidate["learning_rate"],
        max_grad_norm=config.max_grad_norm,
    )
    return adapter, networks, runner, None


def _payload(runner, replay):
    if replay is None:
        return {"runner": runner}
    return {"runner": runner, "replay": replay}


def _run_tree_bytes(root):
    root = Path(root)
    if not root.exists():
        return 0
    total = 0
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            try:
                total += (Path(directory) / filename).stat().st_size
            except FileNotFoundError:
                pass
    return total


def capacity_snapshot(run_root, manifest, reason):
    home = shutil.disk_usage("/home/ext_csv")
    raid = shutil.disk_usage("/raid/ext_csv")
    forecast = manifest["capacity_preflight"]["forecast"]
    run_bytes = _run_tree_bytes(run_root)
    required_now = int(forecast["required_free_bytes"])
    snapshot = {
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": reason,
        "home": {
            "total_bytes": home.total,
            "used_bytes": home.used,
            "free_bytes": home.free,
            "use_fraction": (home.total - home.free) / home.total,
        },
        "raid": {
            "total_bytes": raid.total,
            "used_bytes": raid.used,
            "free_bytes": raid.free,
            "use_fraction": (raid.total - raid.free) / raid.total,
        },
        "run_tree_bytes": run_bytes,
        "projected_remaining_writes_bytes": forecast[
            "projected_peak_write_bytes"
        ],
        "safety_reserve_bytes": forecast["safety_reserve_bytes"],
        "required_free_bytes_now": required_now,
        "passed": raid.free >= required_now,
    }
    if not snapshot["passed"]:
        raise RuntimeError(
            "RAID capacity no longer preserves projected writes plus reserve"
        )
    return snapshot


def _checkpoint_dir(cell, update):
    return Path(cell) / "checkpoints" / f"update_{int(update):06d}"


def _checkpoint_complete(path, job, update, source_hashes):
    path = Path(path)
    state_path = path / "state.msgpack"
    metadata_path = path / "metadata.json"
    if not state_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = _read_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return False
    expected_steps = int(update) * 32768
    return (
        metadata.get("schema_version")
        == "hackrl_online_algorithm_development_checkpoint_v1"
        and metadata.get("job_id") == job["id"]
        and metadata.get("method") == job["method"]
        and metadata.get("environment") == job["environment"]
        and metadata.get("candidate_id") == job["candidate_id"]
        and metadata.get("seed") == job["seed"]
        and metadata.get("update") == int(update)
        and metadata.get("environment_steps") == expected_steps
        and metadata.get("state_sha256") == _sha256_file(state_path)
        and metadata.get("execution_source_hashes") == source_hashes
    )


def save_checkpoint(
    cell,
    job,
    config,
    runner,
    replay,
    source_hashes,
    run_root,
    manifest,
):
    update = int(runner.global_update)
    path = _checkpoint_dir(cell, update)
    if _checkpoint_complete(path, job, update, source_hashes):
        return path
    if path.exists():
        raise RuntimeError(f"preserving invalid existing checkpoint: {path}")
    snapshot = capacity_snapshot(run_root, manifest, f"before_checkpoint:{job['id']}:{update}")
    _append_jsonl(Path(run_root) / "capacity_checks.jsonl", snapshot)
    payload = _payload(runner, replay)
    encoded = serialization.to_bytes(payload)
    restored = serialization.from_bytes(payload, encoded)
    if int(restored["runner"].global_update) != update:
        raise RuntimeError("checkpoint round-trip changed the update counter")
    path.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(path / "state.msgpack", encoded)
    state_hash = _sha256_bytes(encoded)
    metadata = {
        "schema_version": "hackrl_online_algorithm_development_checkpoint_v1",
        "job_id": job["id"],
        "method": job["method"],
        "environment": job["environment"],
        "candidate_id": job["candidate_id"],
        "seed": job["seed"],
        "update": update,
        "environment_steps": int(runner.env_steps),
        "state_bytes": len(encoded),
        "state_sha256": state_hash,
        "execution_source_hashes": source_hashes,
        "parameter_counts": _parameter_counts(job["method"], runner),
        "device_tree_bytes": _tree_bytes(payload),
        "config": asdict(config),
    }
    _atomic_write_json(path / "metadata.json", metadata)
    if not _checkpoint_complete(path, job, update, source_hashes):
        raise RuntimeError("checkpoint verification failed after write")
    return path


def latest_checkpoint(cell, job, source_hashes, checkpoint_updates):
    for update in reversed(tuple(checkpoint_updates)):
        path = _checkpoint_dir(cell, update)
        if path.exists() and not _checkpoint_complete(
            path, job, update, source_hashes
        ):
            raise RuntimeError(f"invalid existing checkpoint must be preserved: {path}")
        if _checkpoint_complete(path, job, update, source_hashes):
            return path, update
    return None, None


def load_checkpoint(path, template):
    return serialization.from_bytes(
        template, (Path(path) / "state.msgpack").read_bytes()
    )


def _validate_evaluation(evaluation, manifest):
    contract = manifest["common_training"]["evaluation"]
    records = evaluation.get("episode_records")
    expected_per_family = int(contract["natural_start_episodes"])
    if not isinstance(records, list) or len(records) != 2 * expected_per_family:
        return False
    for family in ("natural_reset", "common_setup"):
        view = evaluation.get(family, {})
        family_rows = [row for row in records if row.get("family") == family]
        if len(family_rows) != expected_per_family:
            return False
        if view.get("episodes") != expected_per_family:
            return False
        if view.get("success_count") != sum(
            int(bool(row.get("success"))) for row in family_rows
        ):
            return False
        if sorted(row.get("state_index") for row in family_rows) != list(
            range(expected_per_family)
        ):
            return False
        if sorted({row.get("layout") for row in family_rows}) != contract["layouts"]:
            return False
        if sorted({row.get("phase") for row in family_rows}) != contract["phases"]:
            return False
        if any(row.get("repeat") != 0 for row in family_rows):
            return False
    return (
        evaluation.get("kernel") == "fixed"
        and evaluation.get("split") == contract["split"]
        and evaluation.get("mutant_or_bug_metric_persisted") is False
        and evaluation.get("learner_state_immutable") is True
    )


def _normal_evaluation(job, adapter, networks, runner, config, manifest):
    if job["method"] == SD_SAC:
        policy, state = frozen_evaluation_binding(
            method=SD_SAC,
            actor_network=networks[0],
            state=runner.train_state,
        )
    else:
        policy, state = frozen_evaluation_binding(
            method=job["method"], networks=networks, state=runner.train_state
        )
    before = serialization.to_bytes(runner.train_state)
    arguments = {
        "variant": (
            TickClaimVariant.FIXED
            if job["environment"] == "tick"
            else PackRestoreVariant.FIXED
        ),
        "stochastic": False,
        "repeats_per_state": 1,
        "seed_base": 880000,
        "learner_seed": job["seed"],
        "record_episodes": True,
    }
    if job["environment"] == "tick":
        raw = evaluate_tick_claim_gc_frozen(policy, state, **arguments)
    else:
        raw = evaluate_pack_restore_gc_frozen(
            policy,
            state,
            source_growth_period=config.source_growth_period,
            **arguments,
        )
    if before != serialization.to_bytes(runner.train_state):
        raise RuntimeError("frozen evaluation mutated learner state")

    def sanitize_view(view):
        return {
            "episodes": int(view["episodes"]),
            "success_count": int(round(float(view["success_rate"]) * view["episodes"])),
            "success_rate": float(view["success_rate"]),
            "mean_length": float(view["mean_length"]),
            "completed_rate": float(view["completed_rate"]),
        }

    records = [
        {
            key: row[key]
            for key in (
                "family",
                "state_index",
                "layout",
                "phase",
                "repeat",
                "success",
                "length",
            )
        }
        for row in raw["episode_records"]
    ]
    for family in ("natural_reset", "common_setup"):
        family_rows = [row for row in records if row["family"] == family]
        discounted = [
            (config.gamma ** (row["length"] - 1)) if row["success"] else 0.0
            for row in family_rows
        ]
        raw_view = sanitize_view(raw[family])
        raw_view["mean_discounted_return"] = float(np.mean(discounted))
        if family == "natural_reset":
            natural = raw_view
        else:
            common = raw_view
    evaluation = {
        "kernel": "fixed",
        "split": "validation",
        "action_selection": "deterministic greedy/mode",
        "natural_reset": natural,
        "common_setup": common,
        "episode_records": records,
        "mutant_or_bug_metric_persisted": False,
        "learner_state_immutable": True,
    }
    if not _validate_evaluation(evaluation, manifest):
        raise RuntimeError("fixed-normal evaluation violated the frozen contract")
    return evaluation


def _truncate_update_log(path, through_update):
    path = Path(path)
    if not path.is_file():
        return
    by_update = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        update = int(row["update"])
        if update <= int(through_update):
            by_update[update] = row
    payload = b"".join(
        (
            json.dumps(by_update[update], sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        for update in sorted(by_update)
    )
    _atomic_write_bytes(path, payload)


def _summary_complete(path, job, source_hashes, manifest):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        summary = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    final_update = int(manifest["common_training"]["updates"])
    final_checkpoint = _checkpoint_dir(path.parent, final_update)
    expected_steps = int(manifest["common_training"]["transitions_per_job"])
    try:
        return (
            summary.get("schema_version")
            == "hackrl_online_algorithm_development_summary_v1"
            and summary.get("execution_complete") is True
            and summary.get("job") == job["id"]
            and summary.get("method") == job["method"]
            and summary.get("environment") == job["environment"]
            and summary.get("candidate_id") == job["candidate_id"]
            and summary.get("seed") == job["seed"]
            and summary.get("global_update") == final_update
            and summary.get("environment_steps") == expected_steps
            and summary.get("execution_source_hashes") == source_hashes
            and summary.get("checkpoint") == str(final_checkpoint.resolve())
            and summary.get("checkpoint_state_sha256")
            == _sha256_file(final_checkpoint / "state.msgpack")
            and _checkpoint_complete(
                final_checkpoint, job, final_update, source_hashes
            )
            and _validate_evaluation(summary.get("evaluation", {}), manifest)
        )
    except OSError:
        return False


def _compiled_updates(job, adapter, networks, config):
    candidate = job["candidate"]
    if job["method"] != SD_SAC:
        update = jax.jit(
            make_online_value_update(
                adapter,
                config,
                networks,
                method=job["method"],
                epsilon_start=candidate["epsilon_start"],
                epsilon_finish=candidate["epsilon_finish"],
                epsilon_decay_transitions=candidate[
                    "epsilon_decay_transitions"
                ],
            )
        )
        return update, None
    collect = jax.jit(make_sd_sac_collection(adapter, config, networks[0]))
    learn = jax.jit(
        make_sd_sac_replay_updates(
            networks[0],
            networks[1],
            networks[2],
            batch_size=candidate["replay_batch_size"],
            update_iterations=candidate[
                "replay_update_iterations_per_rollout"
            ],
            gamma=config.gamma,
            beta=candidate["beta"],
            clip_range=candidate["q_clip_range"],
            tau=candidate["polyak_tau"],
            target_entropy=(
                candidate["target_entropy_fraction_of_log_actions"]
                * math.log(adapter.num_actions)
            ),
        )
    )
    return collect, learn


def _one_update(job, updates, runner, replay):
    if job["method"] != SD_SAC:
        runner, metrics = updates[0](runner)
        return runner, replay, metrics
    collect, learn = updates
    runner, replay, collection = collect(runner, replay)
    if int(valid_sd_sac_replay_count(replay)) <= 0:
        raise RuntimeError("SD-SAC replay has no valid transition")
    state, rng, learning = learn(runner.train_state, replay, runner.rng)
    runner = runner.replace(
        train_state=state,
        rng=rng,
        global_update=runner.global_update + 1,
    )
    return runner, replay, {"collection": collection, "learning": learning}


def _cell_path(run_root, job):
    return (
        Path(run_root)
        / METHOD_SLUGS[job["method"]]
        / job["candidate_id"]
        / job["environment"]
        / f"seed{job['seed']}"
    )


def run_job(run_root, manifest, job, source_hashes):
    cell = _cell_path(run_root, job)
    summary_path = cell / "summary.json"
    if summary_path.is_file():
        if _summary_complete(summary_path, job, source_hashes, manifest):
            return _read_json(summary_path)
        raise RuntimeError(f"invalid existing summary must be preserved: {summary_path}")
    snapshot = capacity_snapshot(run_root, manifest, f"before_job:{job['id']}")
    _append_jsonl(Path(run_root) / "capacity_checks.jsonl", snapshot)
    config = environment_config(job, manifest)
    config.validate()
    adapter, networks, runner, replay = initialize_job(job, config)
    template = _payload(runner, replay)
    checkpoint_updates = tuple(manifest["execution"]["checkpoint_updates"])
    log_interval = int(manifest["common_training"]["log_interval_updates"])
    checkpoint, start_update = latest_checkpoint(
        cell, job, source_hashes, checkpoint_updates
    )
    if checkpoint is None:
        save_checkpoint(
            cell,
            job,
            config,
            runner,
            replay,
            source_hashes,
            run_root,
            manifest,
        )
        start_update = 0
    else:
        restored = load_checkpoint(checkpoint, template)
        runner = restored["runner"]
        replay = restored.get("replay")
        if int(runner.global_update) != start_update:
            raise RuntimeError("resume checkpoint update mismatch")
    _truncate_update_log(cell / "updates.jsonl", start_update)
    updates = _compiled_updates(job, adapter, networks, config)
    job_started = time.monotonic()
    for update_index in range(int(start_update), config.num_updates):
        step_started = time.monotonic()
        runner, replay, metrics = _one_update(job, updates, runner, replay)
        jax.block_until_ready(runner.global_update)
        completed = update_index + 1
        if not _tree_is_finite(metrics):
            raise RuntimeError("non-finite learner metric")
        if completed == 1 or completed % log_interval == 0:
            if not _tree_is_finite(runner.train_state):
                raise RuntimeError("non-finite learner state")
            row = {
                "recorded_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "job_id": job["id"],
                "update": completed,
                "environment_steps": int(runner.env_steps),
                "wall_time_seconds": time.monotonic() - step_started,
                "metrics": _scalarize(metrics),
            }
            _append_jsonl(cell / "updates.jsonl", row)
            print(
                f"[online-development] {job['id']} "
                f"update={completed}/{config.num_updates} "
                f"env_steps={int(runner.env_steps)}",
                flush=True,
            )
        if completed in checkpoint_updates:
            save_checkpoint(
                cell,
                job,
                config,
                runner,
                replay,
                source_hashes,
                run_root,
                manifest,
            )
    if int(runner.env_steps) != manifest["common_training"]["transitions_per_job"]:
        raise RuntimeError("physical-transition budget mismatch")
    evaluation = _normal_evaluation(
        job, adapter, networks, runner, config, manifest
    )
    _atomic_write_json(cell / "evaluation.json", evaluation)
    final_checkpoint = _checkpoint_dir(cell, config.num_updates)
    summary = {
        "schema_version": "hackrl_online_algorithm_development_summary_v1",
        "execution_complete": True,
        "job": job["id"],
        "method": job["method"],
        "environment": job["environment"],
        "candidate_id": job["candidate_id"],
        "seed": job["seed"],
        "global_update": int(runner.global_update),
        "environment_steps": int(runner.env_steps),
        "parameter_counts": _parameter_counts(job["method"], runner),
        "replay_size": int(replay.size) if replay is not None else 0,
        "replay_total_inserted": (
            int(replay.total_inserted) if replay is not None else 0
        ),
        "wall_time_seconds": time.monotonic() - job_started,
        "evaluation": evaluation,
        "checkpoint": str(final_checkpoint.resolve()),
        "checkpoint_state_sha256": _sha256_file(
            final_checkpoint / "state.msgpack"
        ),
        "execution_source_hashes": source_hashes,
    }
    _atomic_write_json(summary_path, summary)
    if not _summary_complete(summary_path, job, source_hashes, manifest):
        raise RuntimeError("summary verification failed after write")
    return summary


def adjudicate(run_root, manifest, jobs, source_hashes):
    threshold = float(
        manifest["common_training"]["evaluation"]["threshold"]
    )
    minimum_successes = 29
    rows = []
    for job in jobs:
        summary_path = _cell_path(run_root, job) / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing development summary: {summary_path}")
        if not _summary_complete(summary_path, job, source_hashes, manifest):
            raise RuntimeError(f"invalid development summary: {summary_path}")
        summary = _read_json(summary_path)
        evaluation = summary["evaluation"]["natural_reset"]
        rows.append(
            {
                "method": job["method"],
                "environment": job["environment"],
                "candidate_id": job["candidate_id"],
                "seed": job["seed"],
                "success_count": evaluation["success_count"],
                "success_rate": evaluation["success_rate"],
                "mean_discounted_return": evaluation[
                    "mean_discounted_return"
                ],
            }
        )
    cells = []
    for method in METHOD_KEYS.values():
        for environment in ("tick", "pack"):
            candidates = []
            candidate_ids = [
                candidate["id"]
                for candidate in manifest["candidate_registry"][method]
            ]
            for registry_index, candidate_id in enumerate(candidate_ids):
                points = [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["environment"] == environment
                    and row["candidate_id"] == candidate_id
                ]
                if len(points) != len(manifest["common_training"]["learner_seeds"]):
                    raise RuntimeError("candidate adjudication is missing a seed point")
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "registry_index": registry_index,
                        "mean_success_rate": float(
                            np.mean([row["success_rate"] for row in points])
                        ),
                        "mean_discounted_return": float(
                            np.mean(
                                [
                                    row["mean_discounted_return"]
                                    for row in points
                                ]
                            )
                        ),
                        "seed_points": points,
                    }
                )
            selected = max(
                candidates,
                key=lambda item: (
                    item["mean_success_rate"],
                    item["mean_discounted_return"],
                    -item["registry_index"],
                ),
            )
            passed = all(
                point["success_count"] >= minimum_successes
                for point in selected["seed_points"]
            )
            cells.append(
                {
                    "method": method,
                    "environment": environment,
                    "selected_candidate_id": selected["candidate_id"],
                    "passed": passed,
                    "threshold": threshold,
                    "minimum_successes_per_32": minimum_successes,
                    "ranking": sorted(
                        candidates,
                        key=lambda item: (
                            item["mean_success_rate"],
                            item["mean_discounted_return"],
                            -item["registry_index"],
                        ),
                        reverse=True,
                    ),
                }
            )
    result = {
        "schema_version": "hackrl_online_algorithm_development_gate_v1",
        "execution_complete": True,
        "selection_used_only_fixed_normal_natural_start": True,
        "bug_or_mutant_metric_used": False,
        "cells": cells,
        "passed_cells": sum(int(cell["passed"]) for cell in cells),
        "total_cells": len(cells),
        "main_automatically_launched": False,
    }
    _atomic_write_json(Path(run_root) / "development_gate.json", result)
    return result


def prepare_run(run_root, manifest, jobs, source_hashes):
    validate_manifest_contract(manifest)
    _require_smoke_passed(manifest)
    _require_execution_runtime(manifest, run_root)
    _require_clean_execution_sources()
    run_root = Path(run_root)
    snapshot = capacity_snapshot(run_root, manifest, "run_start")
    run_root.mkdir(parents=True, exist_ok=True)
    _append_jsonl(run_root / "capacity_checks.jsonl", snapshot)
    contract = {
        "schema_version": "hackrl_online_algorithm_development_run_contract_v1",
        "run_id": manifest["execution"]["run_id"],
        "run_root": str(run_root.resolve()),
        "git_head": _git_head(),
        "execution_source_hashes": source_hashes,
        "jobs": jobs,
        "budget": manifest["budget"],
        "checkpoint_updates": list(manifest["execution"]["checkpoint_updates"]),
        "storage_preflight": snapshot,
        "runtime_provenance": _runtime_provenance(),
    }
    contract_path = run_root / "run_contract.json"
    if contract_path.is_file():
        recorded = _read_json(contract_path)
        mismatched = [
            key
            for key, value in contract.items()
            if key != "storage_preflight" and recorded.get(key) != value
        ]
        if mismatched:
            raise RuntimeError(
                "refusing resume under changed run contract: "
                + ", ".join(mismatched)
            )
    else:
        _atomic_write_json(contract_path, contract)
    return run_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir")
    parser.add_argument("--job")
    parser.add_argument("--worker", default="manual")
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--adjudicate", action="store_true")
    arguments = parser.parse_args()
    manifest = _read_json(MANIFEST_PATH)
    validate_manifest_contract(manifest)
    jobs = build_jobs(manifest)
    if arguments.list_jobs:
        print(json.dumps({"jobs": jobs}, indent=2, sort_keys=True))
        return
    if arguments.job is not None:
        jobs = [job for job in jobs if job["id"] == arguments.job]
        if not jobs:
            raise ValueError(f"unknown development job: {arguments.job}")
    declared_run_root = Path(manifest["execution"]["run_root"])
    if (
        arguments.log_dir is not None
        and Path(arguments.log_dir).resolve() != declared_run_root.resolve()
    ):
        raise RuntimeError("run-root overrides are forbidden")
    run_root = declared_run_root
    source_hashes = execution_source_hashes()
    all_jobs = build_jobs(manifest)
    runner_lock = _acquire_runner_lock(manifest["execution"]["runner_lock"])
    run_root = prepare_run(run_root, manifest, all_jobs, source_hashes)
    if arguments.adjudicate:
        result = adjudicate(run_root, manifest, all_jobs, source_hashes)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    summaries = []
    for job in jobs:
        summaries.append(run_job(run_root, manifest, job, source_hashes))
    result = None
    if arguments.job is None:
        result = adjudicate(run_root, manifest, all_jobs, source_hashes)
    print(
        json.dumps(
            {
                "worker": arguments.worker,
                "jobs_completed": len(summaries),
                "job_ids": [summary["job"] for summary in summaries],
                "development_gate": result,
            },
            indent=2,
            sort_keys=True,
        )
    )
    runner_lock.close()


if __name__ == "__main__":
    main()
