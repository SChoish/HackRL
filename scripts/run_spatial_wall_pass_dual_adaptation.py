#!/usr/bin/env python3
"""Run the bounded Dual fixed/mutant SPATIAL-WALL-PASS adaptation comparison."""

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
from collections import Counter
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
    BC_HORIZON_UPDATES,
    LEO_EPOCHS,
    LEO_MINIBATCH_SIZE,
    bc_policy_coefficient,
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
    _BRANCH_LOCKED_FIELDS,
    _batch_inputs,
    command_delivery,
    config_from_spatial_wall_pass_gc_payload,
    evaluate_spatial_wall_pass_gc_frozen,
    initialize_spatial_wall_pass_gc,
    spatial_wall_pass_gc_config_payload,
    spatial_wall_pass_gc_parameter_count,
    spatial_wall_pass_outcome,
)

REPOSITORY = Path(__file__).resolve().parents[1]
RUN_ID = "spatial_wall_pass_dual_adaptation_v1"
RUN_ROOT = Path("/raid/ext_csv/HackRL/runs") / RUN_ID
PRETRAIN_ROOT = (
    Path("/raid/ext_csv/HackRL/runs")
    / "spatial_wall_pass_fixed_learnability_v1"
)
GATE_ADJUDICATION = PRETRAIN_ROOT / "gate_adjudication.json"
PRETRAIN_UPDATES = 512
ADAPT_UPDATES = 4096
SCIENCE_UPDATES = (0, 32, 128, 256, 512, 1024, 2048, 4096)
SEEDS = (130, 131)
VARIANTS = ("fixed", "mutant")
EXPECTED_SOURCE_EXECUTION_SHA = "6c476ea0f0c38d5c32a12d48b77200507b444b2c"
EXPECTED_SOURCE_STATE_SHA256 = {
    130: "a84951adbac8a4df878d1729b4aa2d107a763133cabe5e906e37874f8f589fb5",
    131: "f36df5a4a86af54bdc67d5590f53818e6ce105cae2d4fba11d177852eab375cc",
}
GIB = 1024**3
MINIMUM_RESERVE_BYTES = 8 * GIB
PER_CELL_EVALUATION_LOG_ALLOWANCE_BYTES = 512 * 1024**2
EXECUTION_SOURCES = (
    Path("scripts/run_spatial_wall_pass_dual_adaptation.py"),
    Path("scripts/run_spatial_wall_pass_dual_adaptation_gpu.sh"),
    Path("scripts/stop_identity_checked_gpu_keepalive.py"),
    Path("scripts/guard_identity_checked_gpu_keepalive.py"),
    Path("src/hackrl/spatial_wall_pass.py"),
    Path("src/hackrl/spatial_wall_pass_oracle.py"),
    Path("src/hackrl/spatial_wall_pass_gc.py"),
    Path("src/hackrl/dual_leo.py"),
    Path("docs/manifests/spatial_wall_pass_v1.json"),
    Path("docs/manifests/spatial_wall_pass_fixed_learnability_v1.json"),
    Path("docs/manifests/spatial_wall_pass_dual_adaptation_v1.json"),
    Path("docs/manifests/spatial_wall_pass_dual_adaptation_v1_validation.json"),
    Path("tests/test_spatial_wall_pass.py"),
    Path("tests/test_spatial_wall_pass_gc.py"),
    Path("tests/test_spatial_wall_pass_dual_adaptation.py"),
)
REPLICATION_EXECUTION_SOURCES = EXECUTION_SOURCES + (
    Path("scripts/run_spatial_wall_pass_fixed_gate.py"),
    Path("scripts/run_spatial_wall_pass_dual_replication.py"),
    Path("scripts/run_spatial_wall_pass_dual_replication_gpu.sh"),
    Path("docs/manifests/spatial_wall_pass_dual_replication_v1.json"),
    Path("tests/test_spatial_wall_pass_dual_replication.py"),
)
REPLICATION_ALL_SEEDS = (132, 133, 134, 135, 136)
IS_REPLICATION = False


def _git(*arguments):
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def configure_replication(gate_path):
    global RUN_ID
    global RUN_ROOT
    global PRETRAIN_ROOT
    global GATE_ADJUDICATION
    global SEEDS
    global EXPECTED_SOURCE_EXECUTION_SHA
    global EXPECTED_SOURCE_STATE_SHA256
    global EXECUTION_SOURCES
    global IS_REPLICATION

    gate_path = Path(gate_path).resolve()
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("schema_version") != (
        "hackrl_spatial_wall_pass_dual_replication_gate_v1"
    ):
        raise RuntimeError("replication gate schema mismatch")
    if gate.get("status") != "complete":
        raise RuntimeError("replication pretraining gate is incomplete")
    all_seeds = tuple(int(seed) for seed in gate.get("all_seeds", ()))
    passing = tuple(int(seed) for seed in gate.get("passing_seeds", ()))
    if all_seeds != REPLICATION_ALL_SEEDS:
        raise RuntimeError("replication all-seed set changed")
    if not passing or any(seed not in all_seeds for seed in passing):
        raise RuntimeError("replication passing-seed set is invalid")
    expected_root = (
        Path("/raid/ext_csv/HackRL/runs")
        / "spatial_wall_pass_dual_replication_v1"
    )
    pretrain_root = Path(gate["pretrain_root"]).resolve()
    adaptation_root = Path(gate["adaptation_root"]).resolve()
    if pretrain_root != (expected_root / "pretraining").resolve():
        raise RuntimeError("replication pretraining root changed")
    if adaptation_root != (expected_root / "adaptation").resolve():
        raise RuntimeError("replication adaptation root changed")
    state_hashes = {
        int(seed): value
        for seed, value in gate.get(
            "final_checkpoint_state_sha256", {}
        ).items()
    }
    if set(state_hashes) != set(passing):
        raise RuntimeError("replication checkpoint hash set is incomplete")

    RUN_ID = "spatial_wall_pass_dual_replication_v1"
    RUN_ROOT = adaptation_root
    PRETRAIN_ROOT = pretrain_root
    GATE_ADJUDICATION = gate_path
    SEEDS = passing
    EXPECTED_SOURCE_EXECUTION_SHA = gate["execution_code_sha"]
    EXPECTED_SOURCE_STATE_SHA256 = state_hashes
    EXECUTION_SOURCES = REPLICATION_EXECUTION_SOURCES
    IS_REPLICATION = True
    return gate


def jobs():
    return [
        {
            "id": f"dual_{variant}_seed_{seed}",
            "seed": seed,
            "variant": variant,
            "condition": (
                "fixed_continue" if variant == "fixed" else "mutant_adapt"
            ),
        }
        for seed in SEEDS
        for variant in VARIANTS
    ]


def _job_config(job):
    return SpatialWallPassGCConfig(
        variant=job["variant"],
        seed=int(job["seed"]),
        num_envs=512,
        num_steps=64,
        num_updates=ADAPT_UPDATES,
        update_epochs=1,
        minibatch_size=1024,
        policy_hidden_size=512,
        teacher_hidden_size=512,
        learning_rate=2e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coefficient=0.005,
        value_coefficient=0.5,
        max_grad_norm=1.0,
        goal_mode="delivery",
        evaluation_split=SpatialWallPassSplit.VALIDATION.value,
        mode_repeats_per_state=1,
        sample_repeats_per_state=64,
        checkpoint_updates=SCIENCE_UPDATES,
    )


def _source_checkpoint(seed):
    return PRETRAIN_ROOT / f"dual_seed_{int(seed)}" / "checkpoints" / "update_512"


def _cell(job):
    return RUN_ROOT / f"{job['condition']}_seed_{job['seed']}"


def _checkpoint(job, adaptation_update):
    return _cell(job) / "checkpoints" / f"adapt_{int(adaptation_update)}"


def _checkpoint_valid(job, adaptation_update):
    path = _checkpoint(job, adaptation_update)
    if not all(
        (path / name).is_file()
        for name in ("state.msgpack", "config.json", "metadata.json")
    ):
        return False
    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        recorded = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("schema_version") == "hackrl_dual_leo_checkpoint_v1"
        and int(metadata.get("global_update", -1))
        == PRETRAIN_UPDATES + int(adaptation_update)
        and recorded == spatial_wall_pass_gc_config_payload(_job_config(job))
    )


def _latest_checkpoint(job):
    return max(
        (
            update
            for update in SCIENCE_UPDATES
            if _checkpoint_valid(job, update)
        ),
        default=None,
    )


def _compact_evaluation(evaluation):
    compact = {
        "adaptation_update": int(evaluation["adaptation_update"]),
        "global_update": int(evaluation["global_update"]),
        "runner_state_immutable": bool(evaluation["runner_state_immutable"]),
    }
    for kernel in VARIANTS:
        compact[kernel] = {}
        for selection in ("mode", "sample"):
            compact[kernel][selection] = {
                key: value
                for key, value in evaluation[kernel][selection].items()
                if key != "episode_records"
            }
    return compact


def _evaluation_complete(path, adaptation_update):
    if not Path(path).is_file():
        return False
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        int(record.get("adaptation_update", -1)) == int(adaptation_update)
        and all(
            "episode_records" in record.get(kernel, {}).get(selection, {})
            for kernel in VARIANTS
            for selection in ("mode", "sample")
        )
    )


def _validate_source_checkpoint(seed):
    source = _source_checkpoint(seed)
    if _sha256_file(source / "state.msgpack") != EXPECTED_SOURCE_STATE_SHA256[seed]:
        raise RuntimeError(f"source checkpoint hash changed for seed {seed}")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != "hackrl_dual_leo_checkpoint_v1"
        or int(metadata.get("global_update", -1)) != PRETRAIN_UPDATES
        or int(metadata.get("bc_horizon_updates", -1)) != BC_HORIZON_UPDATES
    ):
        raise RuntimeError(f"source checkpoint metadata invalid for seed {seed}")
    recorded = json.loads((source / "config.json").read_text(encoding="utf-8"))
    origin = config_from_spatial_wall_pass_gc_payload(recorded)
    if origin.variant != "fixed" or origin.goal_mode != "normal12":
        raise RuntimeError(f"source checkpoint contract invalid for seed {seed}")
    return {
        "path": str(source.resolve()),
        "state_sha256": EXPECTED_SOURCE_STATE_SHA256[seed],
        "global_update": PRETRAIN_UPDATES,
        "bc_policy_coefficient_at_branch": float(
            bc_policy_coefficient(PRETRAIN_UPDATES)
        ),
    }


def validate_contract():
    gate = json.loads(GATE_ADJUDICATION.read_text(encoding="utf-8"))
    started_at_gate = (
        gate.get("decision", {}).get(
            "mutant_training_started_at_gate_time"
        )
        if IS_REPLICATION
        else gate.get("decision", {}).get("mutant_training_started")
    )
    if (
        gate.get("status") != "complete"
        or gate.get("execution_code_sha") != EXPECTED_SOURCE_EXECUTION_SHA
        or not gate.get("method_gate", {})
        .get("dual", {})
        .get("eligible_for_future_adaptation")
        or started_at_gate is not False
    ):
        raise RuntimeError("fixed gate does not authorize Dual adaptation")
    if (
        IS_REPLICATION
        and _git("rev-parse", "HEAD") != EXPECTED_SOURCE_EXECUTION_SHA
    ):
        raise RuntimeError(
            "adaptation code commit differs from replication pretraining"
        )
    source_records = {
        str(seed): _validate_source_checkpoint(seed) for seed in SEEDS
    }
    planned = jobs()
    expected_jobs = 2 * len(SEEDS)
    if len(planned) != expected_jobs:
        raise RuntimeError(
            f"adaptation contract must contain {expected_jobs} jobs"
        )
    total_transitions = len(planned) * ADAPT_UPDATES * 512 * 64
    if total_transitions != expected_jobs * 134_217_728:
        raise RuntimeError("transition budget mismatch")
    return {
        "run_id": RUN_ID,
        "jobs": planned,
        "source_checkpoints": source_records,
        "pretraining_updates": PRETRAIN_UPDATES,
        "adaptation_updates": ADAPT_UPDATES,
        "checkpoint_updates": list(SCIENCE_UPDATES),
        "total_transitions": total_transitions,
        "goal_mode": "delivery",
        "learn_teacher": True,
        "imitate_teacher": True,
        "same_policy_two_kernel_evaluation": True,
        "gc_adaptation_included": False,
    }


def _provenance(runtime, contract):
    relative = [str(path) for path in EXECUTION_SOURCES]
    dirty = _git("status", "--porcelain", "--", *relative)
    if dirty:
        raise RuntimeError(
            "execution sources must be committed before launch:\n" + dirty
        )
    tracked = set(_git("ls-files", "--", *relative).splitlines())
    missing = sorted(set(relative) - tracked)
    if missing:
        raise RuntimeError(f"untracked execution sources: {missing}")
    return {
        "schema_version": "hackrl_spatial_wall_pass_dual_adaptation_run_v1",
        "run_id": RUN_ID,
        "status": "started",
        "understood_as": (
            (
                "For the preregistered independent seeds that passed the fixed "
                "normal gate, branch each exact update-512 checkpoint into "
                "fixed continuation and mutant adaptation; keep the locked "
                "4096-update budget and evaluation schedule, and evaluate "
                "every frozen policy in both kernels."
            )
            if IS_REPLICATION
            else (
                "Branch Dual seeds 130 and 131 from their exact passing "
                "update-512 checkpoints into fixed continuation and mutant "
                "adaptation, retain the locked 4096-update budget and "
                "evaluation schedule, evaluate every policy in both kernels, "
                "and do not launch GC adaptation."
            )
        ),
        "execution_code_sha": _git("rev-parse", "HEAD"),
        "execution_source_sha256": {
            name: _sha256_file(REPOSITORY / name) for name in relative
        },
        "contract": contract,
        "runtime": runtime,
        "storage": {
            "chosen_filesystem": "/raid/ext_csv",
            "home_excluded_because_utilization_at_prelaunch_was_0.90": True,
            "minimum_safety_reserve_bytes": MINIMUM_RESERVE_BYTES,
            "per_cell_evaluation_log_allowance_bytes": (
                PER_CELL_EVALUATION_LOG_ALLOWANCE_BYTES
            ),
        },
    }


def _projected_capacity():
    state_bytes = max(
        (_source_checkpoint(seed) / "state.msgpack").stat().st_size
        for seed in SEEDS
    )
    pending = sum(
        not _checkpoint_valid(job, update)
        for job in jobs()
        for update in SCIENCE_UPDATES
    )
    overlap = 2 if pending else 0
    projected = (
        state_bytes * (pending + overlap)
        + len(jobs()) * PER_CELL_EVALUATION_LOG_ALLOWANCE_BYTES
    )
    reserve = max(MINIMUM_RESERVE_BYTES, int(np.ceil(0.2 * projected)))
    return {
        "serialized_checkpoint_bytes": state_bytes,
        "retained_checkpoint_count": len(jobs()) * len(SCIENCE_UPDATES),
        "pending_checkpoint_count": pending,
        "active_predecessor_successor_overlap_count": overlap,
        "evaluation_log_allowance_bytes": (
            len(jobs()) * PER_CELL_EVALUATION_LOG_ALLOWANCE_BYTES
        ),
        "projected_remaining_write_bytes": projected,
        "safety_reserve_bytes": reserve,
        "required_free_bytes": projected + reserve,
    }


def record_capacity(event):
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    forecast = _projected_capacity()
    record = {
        "event": event,
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "existing_run_bytes": _directory_bytes(RUN_ROOT),
        **forecast,
        "chosen_filesystem": "/raid/ext_csv",
        "home_filesystem": _filesystem_record("/home/ext_csv"),
        "raid_filesystem": _filesystem_record("/raid/ext_csv"),
    }
    if record["raid_filesystem"]["free_bytes"] < record["required_free_bytes"]:
        raise RuntimeError("RAID lacks projected writes plus safety reserve")
    history_path = RUN_ROOT / "capacity_checks.json"
    history = []
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    history.append(record)
    _write_json(history_path, history)
    return record


def _initialize_templates(config):
    network, template = initialize_spatial_wall_pass_gc(config)
    inputs = _batch_inputs(template.env_state, template.current_goal)
    teacher, leo_template, teacher_minibatch = init_dual_leo_teacher(
        config, inputs[0], inputs[1], NUM_GOALS, NUM_ACTIONS
    )
    return network, template, teacher, leo_template, teacher_minibatch


def _validate_branch_config(source, branch):
    recorded = json.loads((source / "config.json").read_text(encoding="utf-8"))
    origin = config_from_spatial_wall_pass_gc_payload(recorded)
    if origin.goal_mode != "normal12" or branch.goal_mode != "delivery":
        raise RuntimeError("adaptation must branch normal12 to delivery")
    for name in _BRANCH_LOCKED_FIELDS:
        if getattr(origin, name) != getattr(branch, name):
            raise RuntimeError(f"adaptation changed locked field {name}")


def _command_only_digest(runner, leo):
    return serialization.to_bytes(
        {
            "train_state": runner.train_state,
            "env_state": runner.env_state,
            "env_keys": runner.env_keys,
            "sampler_keys": runner.sampler_keys,
            "rng": runner.rng,
            "seen_goals": runner.seen_goals,
            "global_update": runner.global_update,
            "env_steps": runner.env_steps,
            "rollout_cursor": runner.rollout_cursor,
            "leo": leo,
        }
    )


def _load_adaptation_start(job, config, template, leo_template):
    source = _source_checkpoint(job["seed"])
    _validate_branch_config(source, config)
    runner, leo = load_dual_checkpoint(source, template, leo_template)
    if int(runner.global_update) != PRETRAIN_UPDATES:
        raise RuntimeError("pretraining checkpoint is not update 512")
    before = _command_only_digest(runner, leo)
    runner = command_delivery(runner)
    after = _command_only_digest(runner, leo)
    if before != after:
        raise RuntimeError("delivery command switch changed inherited state")
    return runner, leo


def _origin_record(job, runner, leo):
    source = _source_checkpoint(job["seed"])
    return {
        "schema_version": "hackrl_spatial_wall_pass_adaptation_origin_v1",
        "seed": int(job["seed"]),
        "condition": job["condition"],
        "training_variant": job["variant"],
        "source_checkpoint": str(source.resolve()),
        "source_state_sha256": _sha256_file(source / "state.msgpack"),
        "source_global_update": PRETRAIN_UPDATES,
        "adaptation_zero_state_sha256": hashlib.sha256(
            serialization.to_bytes({"gc": runner, "leo": leo})
        ).hexdigest(),
        "bc_policy_coefficient_at_adaptation_start": float(
            bc_policy_coefficient(runner.global_update)
        ),
        "teacher_applied_grad_steps_at_start": int(leo.step),
        "ppo_applied_grad_steps_at_start": int(runner.train_state.step),
    }


def _write_or_check(path, payload, label):
    path = Path(path)
    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded != payload:
            raise RuntimeError(f"{label} changed: {path}")
    else:
        _write_json(path, payload)


def _evaluate(network, runner, config, adaptation_update):
    before = serialization.to_bytes(runner)
    result = {}
    for kernel in VARIANTS:
        result[kernel] = {}
        for stochastic, label, repeats, base in (
            (False, "mode", 1, 60000),
            (True, "sample", 64, 61000),
        ):
            result[kernel][label] = evaluate_spatial_wall_pass_gc_frozen(
                network,
                runner.train_state.params,
                variant=kernel,
                stochastic=stochastic,
                repeats_per_state=repeats,
                seed_base=base,
                learner_seed=config.seed,
                split=SpatialWallPassSplit.VALIDATION.value,
                record_episodes=True,
            )
    result["adaptation_update"] = int(adaptation_update)
    result["global_update"] = int(runner.global_update)
    result["runner_state_immutable"] = (
        before == serialization.to_bytes(runner)
    )
    return result


def _save_checkpoint_and_evaluate(
    job, config, network, runner, leo, adaptation_update
):
    record_capacity(
        f"before_{job['id']}_checkpoint_{int(adaptation_update)}"
    )
    destination = _checkpoint(job, adaptation_update)
    save_dual_checkpoint(
        destination,
        runner,
        leo,
        spatial_wall_pass_gc_config_payload(config),
    )
    evaluation_path = destination / "evaluation.json"
    if not _evaluation_complete(evaluation_path, adaptation_update):
        _write_json(
            evaluation_path,
            _evaluate(network, runner, config, adaptation_update),
        )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    mode = evaluation["mutant"]["mode"]
    print(
        json.dumps(
            {
                "event": "checkpoint",
                "job": job["id"],
                "adaptation_update": int(adaptation_update),
                "global_update": int(runner.global_update),
                "mutant_mode_success_rate": mode["success_rate"],
                "mutant_mode_wall_pass_rate": mode["wall_pass_rate"],
                "mutant_mode_beneficial_use_rate": mode["beneficial_use_rate"],
                "mutant_mode_discounted_return": mode[
                    "mean_discounted_return"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return evaluation


def _load_metrics(cell, completed):
    path = Path(cell) / "updates.json"
    if not path.is_file():
        if completed:
            raise RuntimeError("resumable checkpoint lacks update metrics")
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in rows
        if int(row.get("adaptation_update", -1)) <= int(completed)
    ]
    if [int(row["adaptation_update"]) for row in rows] != list(
        range(1, int(completed) + 1)
    ):
        raise RuntimeError("adaptation metrics are missing or non-contiguous")
    _write_json(path, rows)
    return rows


def _host_metrics(metrics, adaptation_update, seconds):
    result = {
        key: np.asarray(jax.device_get(value)).tolist()
        for key, value in metrics.items()
    }
    result["adaptation_update"] = int(adaptation_update)
    result["seconds"] = float(seconds)
    return result


def run_cell(job):
    config = _job_config(job)
    config.validate()
    cell = _cell(job)
    cell.mkdir(parents=True, exist_ok=True)
    summary_path = cell / "summary.json"
    final_evaluation_path = (
        _checkpoint(job, ADAPT_UPDATES) / "evaluation.json"
    )
    if (
        summary_path.is_file()
        and _checkpoint_valid(job, ADAPT_UPDATES)
        and _evaluation_complete(final_evaluation_path, ADAPT_UPDATES)
    ):
        recorded = json.loads(summary_path.read_text(encoding="utf-8"))
        if recorded.get("status") == "complete":
            print(f"[skip] {job['id']} complete", flush=True)
            return recorded

    (
        network,
        template,
        teacher,
        leo_template,
        teacher_minibatch,
    ) = _initialize_templates(config)
    update_fn = jax.jit(
        make_dual_leo_update(
            network,
            teacher,
            config,
            spatial_wall_pass_outcome,
            _batch_inputs,
            teacher_minibatch,
        )
    )
    saved = _latest_checkpoint(job)
    if saved is None:
        runner, leo = _load_adaptation_start(
            job, config, template, leo_template
        )
        _write_or_check(
            cell / "origin.json",
            _origin_record(job, runner, leo),
            "adaptation origin",
        )
        _save_checkpoint_and_evaluate(
            job, config, network, runner, leo, 0
        )
        saved = 0
    else:
        runner, leo = load_dual_checkpoint(
            _checkpoint(job, saved), template, leo_template
        )
        evaluation_path = _checkpoint(job, saved) / "evaluation.json"
        if not _evaluation_complete(evaluation_path, saved):
            _save_checkpoint_and_evaluate(
                job, config, network, runner, leo, saved
            )
    origin = json.loads((cell / "origin.json").read_text(encoding="utf-8"))
    metrics_history = _load_metrics(cell, saved)
    started = time.perf_counter()
    step_seconds = []

    for _ in range(saved, ADAPT_UPDATES):
        update_started = time.perf_counter()
        runner, leo, metrics = update_fn(runner, leo)
        jax.block_until_ready((runner.global_update, leo.step))
        seconds = time.perf_counter() - update_started
        step_seconds.append(seconds)
        finished = int(runner.global_update) - PRETRAIN_UPDATES
        if finished <= saved:
            raise RuntimeError("adaptation update did not advance")
        metrics_history.append(_host_metrics(metrics, finished, seconds))
        if finished % 32 == 0 or finished in SCIENCE_UPDATES:
            _write_json(cell / "updates.json", metrics_history)
        if finished % 64 == 0:
            record_capacity(f"periodic_{job['id']}_{finished}")
        if finished in SCIENCE_UPDATES:
            _save_checkpoint_and_evaluate(
                job, config, network, runner, leo, finished
            )
        if finished % 32 == 0 or finished in SCIENCE_UPDATES:
            host = metrics_history[-1]
            print(
                json.dumps(
                    {
                        "event": "update",
                        "job": job["id"],
                        "adaptation_update": finished,
                        "updates_total": ADAPT_UPDATES,
                        "seconds": seconds,
                        "recent_transitions_per_second": (
                            config.batch_size
                            / float(np.mean(step_seconds[-8:]))
                        ),
                        "bc_policy_coef": host["bc_policy_coef"],
                        "goal_successes": host["goal_successes"],
                        "wall_pass_count": host.get("wall_pass_count"),
                        "beneficial_delivery_count": host.get(
                            "beneficial_delivery_count"
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    final_evaluation = json.loads(
        (
            _checkpoint(job, ADAPT_UPDATES) / "evaluation.json"
        ).read_text(encoding="utf-8")
    )
    checkpoint_curve = {}
    for update in SCIENCE_UPDATES:
        evaluation = json.loads(
            (_checkpoint(job, update) / "evaluation.json").read_text(
                encoding="utf-8"
            )
        )
        checkpoint_curve[str(update)] = _compact_evaluation(evaluation)
    summary = {
        "schema_version": "hackrl_spatial_wall_pass_dual_adaptation_result_v1",
        "status": "complete",
        "job": job["id"],
        "method": "dual",
        "condition": job["condition"],
        "training_variant": job["variant"],
        "seed": int(job["seed"]),
        "pretraining_updates": PRETRAIN_UPDATES,
        "adaptation_updates": ADAPT_UPDATES,
        "global_update": int(runner.global_update),
        "transitions": ADAPT_UPDATES * config.batch_size,
        "resumed_from_adaptation_update": int(saved),
        "training_seconds_this_invocation": time.perf_counter() - started,
        "steady_state_transitions_per_second": (
            None
            if not step_seconds
            else config.batch_size / float(np.mean(step_seconds))
        ),
        "ppo_parameters": spatial_wall_pass_gc_parameter_count(
            runner.train_state.params
        ),
        "teacher_parameters": teacher_parameter_count(leo),
        "ppo_applied_grad_steps": int(runner.train_state.step),
        "teacher_applied_grad_steps": int(leo.step),
        "learn_teacher": True,
        "imitate_teacher": True,
        "origin": origin,
        "final_evaluation": _compact_evaluation(final_evaluation),
        "checkpoint_curve": checkpoint_curve,
    }
    _write_json(summary_path, summary)
    print(f"[done] {job['id']}", flush=True)
    return summary


def _difference(left, right):
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _policy_kernel_delta(evaluation, selection):
    fixed = evaluation["fixed"][selection]
    mutant = evaluation["mutant"][selection]
    return {
        "success_rate_mutant_minus_fixed": _difference(
            mutant["success_rate"], fixed["success_rate"]
        ),
        "discounted_return_mutant_minus_fixed": _difference(
            mutant["mean_discounted_return"],
            fixed["mean_discounted_return"],
        ),
        "successful_length_mutant_minus_fixed": _difference(
            mutant["mean_success_length"], fixed["mean_success_length"]
        ),
    }


def _route_summary(evaluation, selection):
    records = evaluation["mutant"][selection]["episode_records"]
    routes = Counter()
    for record in records:
        if record["first_wall_pass_step"] < 0:
            continue
        signature = (
            tuple(record["first_wall_pass_from"]),
            record["first_wall_pass_action_name"],
            tuple(record["first_wall_pass_destination"]),
            bool(record["beneficial_use"]),
        )
        routes[signature] += 1
    return [
        {
            "from": list(signature[0]),
            "action": signature[1],
            "destination": list(signature[2]),
            "beneficial_use": signature[3],
            "episodes": count,
        }
        for signature, count in sorted(
            routes.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def summarize():
    cells = {}
    for job in jobs():
        summary_path = _cell(job) / "summary.json"
        final_path = _checkpoint(job, ADAPT_UPDATES) / "evaluation.json"
        if not summary_path.is_file() or not _evaluation_complete(
            final_path, ADAPT_UPDATES
        ):
            raise RuntimeError(f"incomplete cell: {job['id']}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        evaluation = json.loads(final_path.read_text(encoding="utf-8"))
        cells[job["id"]] = {
            "summary": summary,
            "evaluation": evaluation,
        }

    pairs = []
    for seed in SEEDS:
        fixed_job = next(
            job
            for job in jobs()
            if job["seed"] == seed and job["variant"] == "fixed"
        )
        mutant_job = next(
            job
            for job in jobs()
            if job["seed"] == seed and job["variant"] == "mutant"
        )
        fixed_eval = cells[fixed_job["id"]]["evaluation"]
        mutant_eval = cells[mutant_job["id"]]["evaluation"]
        record = {"seed": seed}
        for selection in ("mode", "sample"):
            fixed_on_mutant = fixed_eval["mutant"][selection]
            mutant_on_mutant = mutant_eval["mutant"][selection]
            record[selection] = {
                "mutant_adapt_minus_fixed_continue_on_mutant": {
                    key: _difference(
                        mutant_on_mutant[key], fixed_on_mutant[key]
                    )
                    for key in (
                        "success_rate",
                        "wall_pass_rate",
                        "beneficial_use_rate",
                        "mean_discounted_return",
                        "mean_success_length",
                    )
                },
                "fixed_continue_same_policy_kernel_delta": (
                    _policy_kernel_delta(fixed_eval, selection)
                ),
                "mutant_adapt_same_policy_kernel_delta": (
                    _policy_kernel_delta(mutant_eval, selection)
                ),
                "mutant_adapt_wall_pass_routes": _route_summary(
                    mutant_eval, selection
                ),
            }
        pairs.append(record)

    zero_hashes = {}
    for seed in SEEDS:
        zero_hashes[str(seed)] = {
            condition: _sha256_file(
                _checkpoint(
                    next(
                        job
                        for job in jobs()
                        if job["seed"] == seed
                        and job["condition"] == condition
                    ),
                    0,
                )
                / "state.msgpack"
            )
            for condition in ("fixed_continue", "mutant_adapt")
        }
        if len(set(zero_hashes[str(seed)].values())) != 1:
            raise RuntimeError(
                f"seed {seed} branches did not start from identical state"
            )

    result = {
        "schema_version": "hackrl_spatial_wall_pass_dual_adaptation_aggregate_v1",
        "status": "complete",
        "run_id": RUN_ID,
        "execution_code_sha": _git("rev-parse", "HEAD"),
        "jobs_completed": len(cells),
        "total_transitions": len(cells) * ADAPT_UPDATES * 512 * 64,
        "adaptation_zero_state_sha256": zero_hashes,
        "paired_results": pairs,
        "cells": {
            job_id: {
                "condition": item["summary"]["condition"],
                "seed": item["summary"]["seed"],
                "final_evaluation": item["summary"]["final_evaluation"],
            }
            for job_id, item in cells.items()
        },
        "interpretation_guards": [
            "Wall-pass triggering is separate from beneficial delivery.",
            "Mean successful length is interpreted only with success rate.",
            (
                "These are preregistered independent learner seeds; the "
                "validation layouts remain transforms of one fixture family."
                if IS_REPLICATION
                else (
                    "These two development seeds are first evidence, not an "
                    "independent replication."
                )
            ),
            "GC adaptation is not part of this Dual-only replication.",
        ],
    }
    _write_json(RUN_ROOT / "aggregate_result.json", result)
    return result


def smoke():
    validate_contract()
    outputs = {}
    start_hashes = {}
    adaptation_zero_evaluation = None
    for variant in VARIANTS:
        job = next(
            item
            for item in jobs()
            if item["seed"] == SEEDS[0] and item["variant"] == variant
        )
        config = _job_config(job)
        (
            network,
            template,
            teacher,
            leo_template,
            teacher_minibatch,
        ) = _initialize_templates(config)
        runner, leo = _load_adaptation_start(
            job, config, template, leo_template
        )
        start_hashes[variant] = hashlib.sha256(
            serialization.to_bytes({"gc": runner, "leo": leo})
        ).hexdigest()
        if adaptation_zero_evaluation is None:
            adaptation_zero_evaluation = _compact_evaluation(
                _evaluate(network, runner, config, 0)
            )
        update = jax.jit(
            make_dual_leo_update(
                network,
                teacher,
                config,
                spatial_wall_pass_outcome,
                _batch_inputs,
                teacher_minibatch,
            )
        )
        started = time.perf_counter()
        updated, updated_leo, metrics = update(runner, leo)
        jax.block_until_ready(
            (updated.global_update, updated_leo.step, metrics["loss"])
        )
        outputs[variant] = {
            "seconds_compile_and_update": time.perf_counter() - started,
            "global_update_before": int(runner.global_update),
            "global_update_after": int(updated.global_update),
            "teacher_step_before": int(leo.step),
            "teacher_step_after": int(updated_leo.step),
            "bc_policy_coef": float(metrics["bc_policy_coef"]),
            "valid_transitions": int(metrics["valid_transitions"]),
        }
    if len(set(start_hashes.values())) != 1:
        raise RuntimeError("fixed and mutant smoke branches differ at start")
    return {
        "status": "pass",
        "shape": "512x64",
        "seed": SEEDS[0],
        "identical_start_state_sha256": start_hashes,
        "adaptation_zero_evaluation": adaptation_zero_evaluation,
        "variants": outputs,
    }


def run_queue():
    contract = validate_contract()
    runtime = _runtime_record()
    if runtime["jax_backend"] != ("gpu" if REQUESTED_DEVICE == "cuda" else "cpu"):
        raise RuntimeError(
            f"requested {REQUESTED_DEVICE}, got {runtime['jax_backend']}"
        )
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    provenance = _provenance(runtime, contract)
    _write_or_check(
        RUN_ROOT / "run_manifest.json", provenance, "run manifest"
    )
    record_capacity("prelaunch_or_resume")
    for job in jobs():
        run_cell(job)
    aggregate = summarize()
    record_capacity("complete")
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-queue", action="store_true")
    parser.add_argument("--replication-gate")
    args = parser.parse_args()
    if args.replication_gate:
        configure_replication(args.replication_gate)
    selected = sum((args.validate_only, args.smoke, args.run_queue))
    if selected != 1:
        raise SystemExit("choose exactly one of --validate-only, --smoke, --run-queue")
    if args.validate_only:
        print(json.dumps(validate_contract(), indent=2, sort_keys=True))
    elif args.smoke:
        print(json.dumps(smoke(), indent=2, sort_keys=True))
    else:
        print(json.dumps(run_queue(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
