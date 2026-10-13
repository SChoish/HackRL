#!/usr/bin/env python3
"""Run the preregistered PACK inventory-overwrite PPO adaptation comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback
from dataclasses import replace
from pathlib import Path

if os.environ.get("HACKRL_DEVICE", "cpu") != "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from hackrl.dual_leo import init_dual_leo_teacher, load_dual_checkpoint
from hackrl.pack_restore import PackRestoreVariant
from hackrl.pack_restore_gc import (
    NUM_ACTIONS,
    NUM_GOALS,
    _FROZEN_EVAL_CACHE,
    _batch_inputs,
    config_from_pack_restore_gc_payload,
    evaluate_pack_restore_gc_frozen,
    initialize_pack_restore_gc,
    load_pack_restore_gc_checkpoint,
    make_pack_restore_gc_update,
    pack_restore_gc_parameter_count,
    save_pack_restore_gc_checkpoint,
    step_pack_restore_gc_workers,
)
from hackrl.tick_claim import WORLD_HORIZON


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_ppo_v1.json"
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
RUN_ROOT = Path(SPEC["run_root"])
SOURCE_ROOT = Path(SPEC["source_root"])
SEEDS = tuple(int(seed) for seed in SPEC["seeds"])
CONDITIONS = ("frozen_harmful", "normal_continue", "harmful_adapt")
LEARNING_CONDITIONS = ("normal_continue", "harmful_adapt")
SCIENCE_UPDATES = tuple(int(value) for value in SPEC["evaluation"]["adaptation_updates"])
EVAL_SEED_BASE = 31000
EXECUTION_SOURCES = (
    Path("docs/manifests/pack_inventory_overwrite_ppo_v1.json"),
    Path("scripts/run_pack_inventory_overwrite_ppo.py"),
    Path("scripts/summarize_pack_inventory_overwrite_ppo.py"),
    Path("src/hackrl/pack_restore.py"),
    Path("src/hackrl/pack_restore_gc.py"),
    Path("src/hackrl/dual_leo.py"),
)


def _write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(tree) -> str:
    digest = hashlib.sha256()
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = np.ascontiguousarray(jax.device_get(value))
        digest.update(jax.tree_util.keystr(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _train_state_sha256(train_state) -> str:
    """Hash learned state while normalizing JIT's equal int scalar dtypes."""

    learned = _tree_sha256((train_state.params, train_state.opt_state))
    return hashlib.sha256(f"{learned}:{int(train_state.step)}".encode("ascii")).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY, text=True
    ).strip()


def _source_checkpoint(seed: int) -> Path:
    return (
        SOURCE_ROOT
        / f"seed{seed}"
        / "checkpoints"
        / str(SPEC["source_checkpoint"])
    )


def _source_checkpoint_hashes() -> dict[str, str]:
    return {
        str(_source_checkpoint(seed) / name): _sha256(
            _source_checkpoint(seed) / name
        )
        for seed in SEEDS
        for name in ("state.msgpack", "config.json", "metadata.json", "arm.json")
    }


def build_jobs() -> list[dict]:
    return [
        {
            "id": f"{condition}-seed{seed}",
            "condition": condition,
            "seed": seed,
            "learning": condition in LEARNING_CONDITIONS,
        }
        for seed in SEEDS
        for condition in CONDITIONS
    ]


def _capacity(existing_root: Path = RUN_ROOT) -> dict:
    existing = (
        sum(path.stat().st_size for path in existing_root.rglob("*") if path.is_file())
        if existing_root.exists()
        else 0
    )
    home = shutil.disk_usage("/home/ext_csv")
    raid = shutil.disk_usage("/raid/ext_csv")
    storage = SPEC["checkpoint_storage"]
    required = int(storage["projected_peak_writes_bytes"]) + int(
        storage["safety_reserve_bytes"]
    )
    if raid.free <= required:
        raise RuntimeError(
            f"RAID free bytes {raid.free} do not cover peak writes plus reserve {required}"
        )
    return {
        "measured_unix": time.time(),
        "home": home._asdict(),
        "raid": raid._asdict(),
        "existing_run_bytes": existing,
        "projected_additional_bytes": int(storage["projected_additional_bytes"]),
        "projected_peak_writes_bytes": int(storage["projected_peak_writes_bytes"]),
        "safety_reserve_bytes": int(storage["safety_reserve_bytes"]),
        "minimum_required_free_bytes": required,
        "chosen_filesystem": "/raid/ext_csv",
    }


def _validate_prerequisite() -> None:
    path = Path(SPEC["prerequisite_gate"]["result"])
    if _sha256(path) != SPEC["prerequisite_gate"]["result_sha256"]:
        raise RuntimeError("the no-learning prerequisite result hash changed")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != SPEC["prerequisite_gate"]["required_status"]:
        raise RuntimeError("the no-learning prerequisite gate is not complete")


def _source_config(seed: int):
    checkpoint = _source_checkpoint(seed)
    config = config_from_pack_restore_gc_payload(
        json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    )
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    arm = json.loads((checkpoint / "arm.json").read_text(encoding="utf-8"))
    expected = SPEC["optimization"]
    if (
        config.seed != seed
        or config.variant != PackRestoreVariant.FIXED.value
        or config.goal_mode != "deliver_3"
        or config.num_envs != int(expected["num_envs"])
        or config.num_steps != int(expected["num_steps"])
        or config.minibatch_size != int(expected["minibatch_size"])
        or metadata.get("global_update") != int(SPEC["source_global_update"])
        or arm != {"imitate_teacher": True, "learn_teacher": True}
    ):
        raise RuntimeError(f"source checkpoint contract failed for seed {seed}")
    return config


def _branch_config(seed: int, condition: str):
    variant = (
        PackRestoreVariant.FIXED.value
        if condition == "normal_continue"
        else PackRestoreVariant.INVENTORY_OVERWRITE.value
    )
    config = replace(
        _source_config(seed),
        variant=variant,
        num_updates=int(SPEC["optimization"]["updates"]),
        checkpoint_updates=(),
    )
    config.validate()
    return config


def _fresh_runner(seed: int, condition: str):
    """Restore only actor-critic and Adam into a fresh paired worker state."""

    config = _branch_config(seed, condition)
    network, fresh = initialize_pack_restore_gc(config)
    inputs = _batch_inputs(fresh.env_state, fresh.current_goal)
    _, teacher_template, _ = init_dual_leo_teacher(
        config,
        inputs[0],
        inputs[1],
        NUM_GOALS,
        NUM_ACTIONS,
    )
    source_runner, _ = load_dual_checkpoint(
        _source_checkpoint(seed), fresh, teacher_template
    )
    if int(source_runner.global_update) != int(SPEC["source_global_update"]):
        raise RuntimeError("source runner global update changed")
    source_train_state_hash = _train_state_sha256(source_runner.train_state)
    worker_initial_hash = _tree_sha256(
        (
            fresh.env_state,
            fresh.current_goal,
            fresh.command_active,
            fresh.env_keys,
            fresh.sampler_keys,
            fresh.rng,
        )
    )
    runner = fresh.replace(
        train_state=source_runner.train_state,
        global_update=source_runner.global_update,
    )
    if _train_state_sha256(runner.train_state) != source_train_state_hash:
        raise AssertionError("fresh branch changed the restored train state")
    return network, runner, config, {
        "source_checkpoint": str(_source_checkpoint(seed)),
        "source_train_state_sha256": source_train_state_hash,
        "branch_train_state_sha256": _train_state_sha256(runner.train_state),
        "worker_initial_sha256": worker_initial_hash,
        "source_optimizer_step": int(source_runner.train_state.step),
        "source_global_update": int(source_runner.global_update),
        "fresh_environment_steps": int(runner.env_steps),
        "fresh_rollout_cursor": int(runner.rollout_cursor),
        "teacher_loaded_for_deserialization_only": True,
        "teacher_used_for_actions_or_updates": False,
    }


def _empty_tracker(num_envs: int) -> dict:
    return {
        "episode_loss_seen": [False] * int(num_envs),
        "collector_blocks": 0,
        "collector_slots": 0,
        "physical_transitions": 0,
        "valid_command_transitions": 0,
        "valid_ppo_samples": 0,
        "optimizer_applications": 0,
        "empty_minibatches": 0,
        "completed_world_episodes": 0,
        "loss_episodes": 0,
        "cumulative_lost_grain": 0,
        "physical_loss_events": 0,
        "overwrite_trigger_events": 0,
        "goal_successes": 0,
        "harvest_events": 0,
        "storage_events": 0,
        "delivery_events": 0,
    }


def update_exposure_tracker(
    tracker: dict,
    metrics: dict,
    *,
    learning: bool,
    optimizer_applications: int,
) -> dict:
    output = dict(tracker)
    seen = np.asarray(output["episode_loss_seen"], dtype=bool)
    loss_workers = np.asarray(jax.device_get(metrics["loss_worker_mask"]), dtype=bool)
    world_done = np.asarray(
        jax.device_get(metrics["world_done_worker_mask"]), dtype=bool
    )
    world_resets = int(np.asarray(jax.device_get(metrics["world_resets"])))
    if world_resets != int(np.sum(world_done)):
        raise AssertionError(
            "a 64-step block contained more than one world reset for a worker"
        )
    seen = np.logical_or(seen, loss_workers)
    output["loss_episodes"] += int(np.sum(np.logical_and(seen, world_done)))
    output["completed_world_episodes"] += int(np.sum(world_done))
    seen = np.where(world_done, False, seen)
    output["episode_loss_seen"] = seen.tolist()
    output["collector_blocks"] += 1
    scalar_fields = {
        "collector_slots": "transitions",
        "physical_transitions": "transitions",
        "valid_command_transitions": "valid_transitions",
        "empty_minibatches": "empty_minibatches",
        "cumulative_lost_grain": "lost_grain",
        "physical_loss_events": "physical_loss_events",
        "overwrite_trigger_events": "overwrite_trigger_events",
        "goal_successes": "goal_successes",
        "harvest_events": "harvest_events",
        "storage_events": "storage_events",
        "delivery_events": "delivery_events",
    }
    for destination, source in scalar_fields.items():
        output[destination] += int(np.asarray(jax.device_get(metrics[source])))
    if learning:
        output["valid_ppo_samples"] += int(
            np.asarray(jax.device_get(metrics["valid_transitions"]))
        )
    output["optimizer_applications"] += int(optimizer_applications)
    completed = output["completed_world_episodes"]
    output["loss_episode_rate"] = (
        output["loss_episodes"] / completed if completed else None
    )
    return output


def _scalar_metrics(metrics: dict) -> dict:
    return {
        key: np.asarray(jax.device_get(value)).tolist()
        for key, value in metrics.items()
        if key not in {"loss_worker_mask", "world_done_worker_mask"}
    }


def make_frozen_collection(network, config):
    def rollout_step(runner, _):
        inputs = _batch_inputs(runner.env_state, runner.current_goal)
        policy, _ = network.apply(runner.train_state.params, *inputs)
        rng, action_rng = jax.random.split(runner.rng)
        actions = policy.sample(seed=action_rng)
        runner = runner.replace(rng=rng)
        runner, event = step_pack_restore_gc_workers(runner, actions, config)
        return runner, event

    def collect(runner):
        start_steps = runner.env_steps
        runner, trajectory = jax.lax.scan(
            rollout_step, runner, None, length=config.num_steps
        )
        runner = runner.replace(rollout_cursor=jnp.asarray(0, dtype=jnp.int32))
        return runner, {
            "transitions": runner.env_steps - start_steps,
            "valid_transitions": jnp.sum(trajectory.valid),
            "empty_minibatches": jnp.asarray(0, dtype=jnp.int32),
            "goal_successes": jnp.sum(trajectory.goal_done),
            "world_resets": jnp.sum(trajectory.world_done),
            "physical_loss_events": jnp.sum(trajectory.physical_loss),
            "lost_grain": jnp.sum(trajectory.lost_grain),
            "overwrite_trigger_events": jnp.sum(trajectory.overwrite_trigger),
            "harvest_events": jnp.sum(trajectory.harvest_event),
            "storage_events": jnp.sum(trajectory.storage_event),
            "delivery_events": jnp.sum(trajectory.delivered_amount > 0),
            "loss_worker_mask": jnp.any(trajectory.physical_loss, axis=0),
            "world_done_worker_mask": jnp.any(trajectory.world_done, axis=0),
        }

    return collect


def _evaluate(network, parameters, seed: int) -> dict:
    output = {}
    for variant in (
        PackRestoreVariant.FIXED.value,
        PackRestoreVariant.INVENTORY_OVERWRITE.value,
    ):
        output[variant] = {}
        for name, stochastic, repeats in (
            ("mode", False, int(SPEC["evaluation"]["mode_repeats_per_state"])),
            ("sample", True, int(SPEC["evaluation"]["sample_repeats_per_state"])),
        ):
            result = evaluate_pack_restore_gc_frozen(
                network,
                parameters,
                variant=variant,
                split=SPEC["evaluation"]["split"],
                stochastic=stochastic,
                repeats_per_state=repeats,
                seed_base=EVAL_SEED_BASE,
                learner_seed=seed,
                source_growth_period=_source_config(seed).source_growth_period,
                record_episodes=True,
                natural_only=True,
            )
            expected = int(SPEC["evaluation"]["states"]) * repeats
            if result["overall"]["episodes"] != expected:
                raise AssertionError("evaluation panel size changed")
            output[variant][name] = result
    return output


def _cell(condition: str, seed: int) -> Path:
    return RUN_ROOT / condition / f"seed{seed}"


def _evaluation_path(cell: Path, update: int) -> Path:
    return cell / "evaluations" / f"adapt_{update}.json"


def _checkpoint_complete(path: Path) -> bool:
    return all(
        (path / name).is_file()
        for name in (
            "state.msgpack",
            "config.json",
            "metadata.json",
            "tracker.json",
            "receipt.json",
        )
    )


def _save_checkpoint_atomic(
    destination: Path,
    runner,
    config,
    tracker: dict,
) -> None:
    if _checkpoint_complete(destination):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    save_pack_restore_gc_checkpoint(temporary, runner, config)
    _write_json(temporary / "tracker.json", tracker)
    receipt = {
        "adaptation_update": int(runner.global_update) - int(SPEC["source_global_update"]),
        "global_update": int(runner.global_update),
        "environment_steps": int(runner.env_steps),
        "optimizer_step": int(runner.train_state.step),
        "train_state_sha256": _train_state_sha256(runner.train_state),
        "state_sha256": _sha256(temporary / "state.msgpack"),
    }
    _write_json(temporary / "receipt.json", receipt)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)


def _checkpoint_for_update(cell: Path, update: int) -> Path:
    if update in SCIENCE_UPDATES:
        return cell / "checkpoints" / f"adapt_{update}"
    return cell / "resume" / f"adapt_{update}"


def _latest_checkpoint(cell: Path) -> tuple[int, Path] | None:
    found = []
    for root in (cell / "checkpoints", cell / "resume"):
        if not root.is_dir():
            continue
        for path in root.glob("adapt_*"):
            suffix = path.name.removeprefix("adapt_")
            if suffix.isdigit() and _checkpoint_complete(path):
                found.append((int(suffix), path))
    return max(found, default=None, key=lambda item: item[0])


def _prune_resume(cell: Path, keep: Path | None = None) -> None:
    root = cell / "resume"
    if not root.is_dir():
        return
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith("adapt_"):
            continue
        if keep is not None and path.resolve() == keep.resolve():
            continue
        if path.parent.resolve() != root.resolve():
            raise RuntimeError(f"refusing to prune outside resume root: {path}")
        shutil.rmtree(path)


def _write_origin(cell: Path, condition: str, seed: int, provenance: dict) -> None:
    document = {
        "condition": condition,
        "seed": seed,
        "kernel": SPEC["conditions"][condition]["kernel"],
        **provenance,
    }
    path = cell / "origin.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != document:
        raise RuntimeError(f"origin changed: {path}")
    _write_json(path, document)


def _write_evaluation(cell: Path, update: int, network, runner, seed: int) -> None:
    path = _evaluation_path(cell, update)
    if path.is_file():
        return
    _write_json(
        path,
        {
            "schema_version": "hackrl_pack_inventory_overwrite_eval_v1",
            "seed": seed,
            "adaptation_update": update,
            "collector_slots": update * int(SPEC["optimization"]["collector_slots_per_block"]),
            "train_state_sha256": _train_state_sha256(runner.train_state),
            "kernels": _evaluate(network, runner.train_state.params, seed),
        },
    )
    print(f"[evaluation] {path}", flush=True)


def _update_record_path(cell: Path, update: int) -> Path:
    return cell / "updates" / f"update_{update}.json"


def _load_tracker(checkpoint: Path) -> dict:
    return json.loads((checkpoint / "tracker.json").read_text(encoding="utf-8"))


def _summarize_cell(
    cell: Path,
    condition: str,
    seed: int,
    runner,
    tracker: dict,
    origin: dict,
) -> dict:
    updates = int(SPEC["optimization"]["updates"])
    records = [
        json.loads(_update_record_path(cell, update).read_text(encoding="utf-8"))
        for update in range(1, updates + 1)
    ]
    if tracker["collector_slots"] != int(
        SPEC["optimization"]["collector_slots_per_condition_seed"]
    ):
        raise AssertionError("collector budget did not finish")
    pending_loss_workers = int(np.sum(np.asarray(tracker["episode_loss_seen"])))
    if tracker["physical_transitions"] != tracker["collector_slots"]:
        raise AssertionError("physical transitions differ from collector slots")
    expected_episodes = tracker["physical_transitions"] // WORLD_HORIZON
    if tracker["completed_world_episodes"] != expected_episodes:
        raise AssertionError("completed world-episode accounting changed")
    if pending_loss_workers:
        raise AssertionError("the fixed budget ended with partial loss episodes")
    optimizer_delta = int(runner.train_state.step) - int(
        origin["source_optimizer_step"]
    )
    scheduled_optimizer_applications = (
        int(SPEC["optimization"]["updates"])
        * int(SPEC["optimization"]["num_envs"])
        * int(SPEC["optimization"]["num_steps"])
        // int(SPEC["optimization"]["minibatch_size"])
    )
    if condition in LEARNING_CONDITIONS:
        if optimizer_delta != tracker["optimizer_applications"]:
            raise AssertionError("optimizer step delta differs from recorded applications")
        if optimizer_delta + tracker["empty_minibatches"] != scheduled_optimizer_applications:
            raise AssertionError("applied plus empty PPO minibatches changed")
    elif optimizer_delta != 0 or tracker["optimizer_applications"] != 0:
        raise AssertionError("frozen deployment changed optimizer state")
    return {
        "schema_version": "hackrl_pack_inventory_overwrite_cell_v1",
        "status": "complete",
        "condition": condition,
        "seed": seed,
        "kernel": SPEC["conditions"][condition]["kernel"],
        "learning": condition in LEARNING_CONDITIONS,
        "adaptation_updates": updates,
        "collector_slots": tracker["collector_slots"],
        "physical_transitions": tracker["physical_transitions"],
        "valid_command_transitions": tracker["valid_command_transitions"],
        "valid_ppo_samples": tracker["valid_ppo_samples"],
        "optimizer_applications": tracker["optimizer_applications"],
        "scheduled_optimizer_applications": (
            scheduled_optimizer_applications
            if condition in LEARNING_CONDITIONS
            else 0
        ),
        "empty_minibatches": tracker["empty_minibatches"],
        "completed_world_episodes": tracker["completed_world_episodes"],
        "loss_episodes": tracker["loss_episodes"],
        "loss_episode_rate": tracker["loss_episode_rate"],
        "pending_partial_episode_loss_workers": pending_loss_workers,
        "cumulative_lost_grain": tracker["cumulative_lost_grain"],
        "physical_loss_events": tracker["physical_loss_events"],
        "overwrite_trigger_events": tracker["overwrite_trigger_events"],
        "goal_successes_during_collection": tracker["goal_successes"],
        "harvest_events": tracker["harvest_events"],
        "storage_events": tracker["storage_events"],
        "delivery_events": tracker["delivery_events"],
        "source_optimizer_step": origin["source_optimizer_step"],
        "final_optimizer_step": int(runner.train_state.step),
        "initial_train_state_sha256": origin["source_train_state_sha256"],
        "final_train_state_sha256": _train_state_sha256(runner.train_state),
        "ppo_parameter_count": pack_restore_gc_parameter_count(runner.train_state.params),
        "teacher_updates": 0,
        "bc_updates": 0,
        "update_records": len(records),
        "evaluation_updates": list(SCIENCE_UPDATES),
        "final_evaluation": str(_evaluation_path(cell, updates)),
    }


def run_learning(condition: str, seed: int) -> None:
    cell = _cell(condition, seed)
    summary_path = cell / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "complete":
            print(f"[skip] {condition} seed={seed}", flush=True)
            return
    network, initial_runner, config, provenance = _fresh_runner(seed, condition)
    _write_origin(cell, condition, seed, provenance)
    latest = _latest_checkpoint(cell)
    if latest is None:
        runner = initial_runner
        tracker = _empty_tracker(config.num_envs)
        checkpoint = _checkpoint_for_update(cell, 0)
        _save_checkpoint_atomic(checkpoint, runner, config, tracker)
        _write_evaluation(cell, 0, network, runner, seed)
        completed = 0
    else:
        completed, checkpoint = latest
        _, template = initialize_pack_restore_gc(config)
        runner = load_pack_restore_gc_checkpoint(checkpoint, template, config)
        tracker = _load_tracker(checkpoint)
        if int(runner.global_update) - int(SPEC["source_global_update"]) != completed:
            raise RuntimeError("resume checkpoint update mismatch")
        if completed in SCIENCE_UPDATES:
            _write_evaluation(cell, completed, network, runner, seed)
    update_fn = jax.jit(make_pack_restore_gc_update(network, config))
    while completed < int(SPEC["optimization"]["updates"]):
        started = time.perf_counter()
        optimizer_before = int(runner.train_state.step)
        runner, metrics = update_fn(runner)
        jax.block_until_ready(runner.global_update)
        completed += 1
        optimizer_after = int(runner.train_state.step)
        applications = optimizer_after - optimizer_before
        tracker = update_exposure_tracker(
            tracker,
            metrics,
            learning=True,
            optimizer_applications=applications,
        )
        record = {
            "adaptation_update": completed,
            "collector_slots_this_update": int(
                np.asarray(jax.device_get(metrics["transitions"]))
            ),
            "optimizer_applications_this_update": applications,
            "seconds": time.perf_counter() - started,
            "metrics": _scalar_metrics(metrics),
            "cumulative": {
                key: value
                for key, value in tracker.items()
                if key != "episode_loss_seen"
            },
        }
        _write_json(_update_record_path(cell, completed), record)
        destination = _checkpoint_for_update(cell, completed)
        _save_checkpoint_atomic(destination, runner, config, tracker)
        if completed in SCIENCE_UPDATES:
            _write_evaluation(cell, completed, network, runner, seed)
            _prune_resume(cell)
        else:
            _prune_resume(cell, keep=destination)
        print(
            f"[update] {condition} seed={seed} {completed}/32 "
            f"valid={record['metrics']['valid_transitions']} "
            f"lost={tracker['cumulative_lost_grain']} "
            f"successes={tracker['goal_successes']} "
            f"seconds={record['seconds']:.3f}",
            flush=True,
        )
    origin = json.loads((cell / "origin.json").read_text(encoding="utf-8"))
    summary = _summarize_cell(cell, condition, seed, runner, tracker, origin)
    _write_json(summary_path, summary)
    print(f"[done] {condition} seed={seed}", flush=True)


def run_frozen(seed: int) -> None:
    condition = "frozen_harmful"
    cell = _cell(condition, seed)
    summary_path = cell / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "complete":
            print(f"[skip] {condition} seed={seed}", flush=True)
            return
    network, runner, config, provenance = _fresh_runner(seed, condition)
    _write_origin(cell, condition, seed, provenance)
    tracker = _empty_tracker(config.num_envs)
    before_hash = _train_state_sha256(runner.train_state)
    optimizer_before = int(runner.train_state.step)
    _write_evaluation(cell, 0, network, runner, seed)
    collect = jax.jit(make_frozen_collection(network, config))
    for completed in range(1, int(SPEC["optimization"]["updates"]) + 1):
        started = time.perf_counter()
        runner, metrics = collect(runner)
        jax.block_until_ready(runner.env_steps)
        tracker = update_exposure_tracker(
            tracker,
            metrics,
            learning=False,
            optimizer_applications=0,
        )
        record = {
            "adaptation_update": completed,
            "collector_slots_this_update": int(
                np.asarray(jax.device_get(metrics["transitions"]))
            ),
            "optimizer_applications_this_update": 0,
            "seconds": time.perf_counter() - started,
            "metrics": _scalar_metrics(metrics),
            "cumulative": {
                key: value
                for key, value in tracker.items()
                if key != "episode_loss_seen"
            },
        }
        _write_json(_update_record_path(cell, completed), record)
        if completed in SCIENCE_UPDATES:
            _write_json(cell / "exposure" / f"adapt_{completed}.json", tracker)
            _write_evaluation(cell, completed, network, runner, seed)
        print(
            f"[frozen] seed={seed} {completed}/32 "
            f"lost={tracker['cumulative_lost_grain']} "
            f"loss_episodes={tracker['loss_episodes']} "
            f"seconds={record['seconds']:.3f}",
            flush=True,
        )
    after_hash = _train_state_sha256(runner.train_state)
    if before_hash != after_hash or int(runner.train_state.step) != optimizer_before:
        raise AssertionError("frozen collection changed actor-critic or Adam state")
    origin = json.loads((cell / "origin.json").read_text(encoding="utf-8"))
    summary = _summarize_cell(cell, condition, seed, runner, tracker, origin)
    if summary["optimizer_applications"] != 0 or summary["valid_ppo_samples"] != 0:
        raise AssertionError("frozen condition performed learning")
    _write_json(summary_path, summary)
    print(f"[done] {condition} seed={seed}", flush=True)


def _run_manifest() -> dict:
    relative = [str(path) for path in EXECUTION_SOURCES]
    dirty = _git("status", "--porcelain", "--", *relative)
    if dirty:
        raise RuntimeError("execution sources must be committed before launch:\n" + dirty)
    tracked = set(_git("ls-files", "--", *relative).splitlines())
    if set(relative) - tracked:
        raise RuntimeError(f"execution sources are untracked: {sorted(set(relative) - tracked)}")
    return {
        "schema_version": "hackrl_pack_inventory_overwrite_ppo_run_v1",
        "manifest_id": SPEC["manifest_id"],
        "execution_head": _git("rev-parse", "HEAD"),
        "manifest_sha256": _sha256(SPEC_PATH),
        "execution_source_sha256": {
            name: _sha256(REPOSITORY / name) for name in relative
        },
        "source_checkpoint_sha256": _source_checkpoint_hashes(),
        "jobs": build_jobs(),
        "planned_collector_slots": int(SPEC["budget"]["total_collector_slots"]),
        "automatic_extension": False,
    }


def validate_contract(*, require_committed: bool = False) -> dict:
    _validate_prerequisite()
    if len(build_jobs()) != int(SPEC["budget"]["total_jobs"]):
        raise AssertionError("job count mismatch")
    block = int(SPEC["optimization"]["collector_slots_per_block"])
    updates = int(SPEC["optimization"]["updates"])
    if block != int(SPEC["optimization"]["num_envs"]) * int(
        SPEC["optimization"]["num_steps"]
    ):
        raise AssertionError("collector block arithmetic changed")
    if (
        len(build_jobs()) * block * updates
        != int(SPEC["budget"]["total_collector_slots"])
    ):
        raise AssertionError("total collector budget arithmetic changed")
    for seed in SEEDS:
        _source_config(seed)
    capacity = _capacity()
    result = {
        "status": "contract_valid",
        "jobs": len(build_jobs()),
        "collector_slots": int(SPEC["budget"]["total_collector_slots"]),
        "capacity": capacity,
    }
    if require_committed:
        result["run_manifest"] = _run_manifest()
    return result


def _prepare_run() -> dict:
    validation = validate_contract(require_committed=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = validation["run_manifest"]
    path = RUN_ROOT / "run_manifest.json"
    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded != manifest:
            raise RuntimeError("run manifest differs from the existing execution")
    else:
        _write_json(path, manifest)
        _write_json(RUN_ROOT / "capacity_start.json", validation["capacity"])
    return manifest


def _job_complete(job: dict) -> bool:
    path = _cell(job["condition"], int(job["seed"])) / "summary.json"
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get(
        "status"
    ) == "complete"


def _capacity_checkpoint() -> None:
    measurement = _capacity()
    path = RUN_ROOT / "capacity_checks.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(measurement, sort_keys=True) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-contract", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    arguments = parser.parse_args()
    if arguments.validate_contract:
        print(json.dumps(validate_contract(), indent=2, sort_keys=True))
        return
    if not arguments.all and (arguments.condition is None or arguments.seed is None):
        raise SystemExit("use --all or provide both --condition and --seed")
    manifest = _prepare_run()
    print(f"[provenance] execution_head={manifest['execution_head']}", flush=True)
    jobs = build_jobs() if arguments.all else [
        {
            "condition": arguments.condition,
            "seed": arguments.seed,
            "learning": arguments.condition in LEARNING_CONDITIONS,
        }
    ]
    for job in jobs:
        condition = str(job["condition"])
        seed = int(job["seed"])
        if _job_complete(job):
            print(f"[skip] {condition} seed={seed}", flush=True)
            continue
        try:
            if condition == "frozen_harmful":
                run_frozen(seed)
            else:
                run_learning(condition, seed)
            _capacity_checkpoint()
        except Exception:
            failed = _cell(condition, seed) / "failed.json"
            _write_json(
                failed,
                {"failed_unix": time.time(), "traceback": traceback.format_exc()},
            )
            raise
        finally:
            _FROZEN_EVAL_CACHE.clear()
            jax.clear_caches()
    if arguments.all and all(_job_complete(job) for job in build_jobs()):
        _write_json(
            RUN_ROOT / "training_complete.json",
            {
                "status": "all_15_jobs_complete_pending_aggregation",
                "completed_unix": time.time(),
                "execution_head": manifest["execution_head"],
            },
        )


if __name__ == "__main__":
    main()
