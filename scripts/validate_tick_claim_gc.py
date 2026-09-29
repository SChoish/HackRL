#!/usr/bin/env python3
"""Validate GC-PPO wiring and materialize its resolved run gate."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from flax import serialization

from hackrl.tick_claim import (
    TickClaimAction,
    TickClaimPhase,
    TickClaimStart,
    TickClaimVariant,
    make_tick_claim_state,
    observe_tick_claim,
    tick_claim_goal_vector,
)
from hackrl.tick_claim_gc import (
    TickClaimGCConfig,
    evaluate_tick_claim_gc_frozen,
    initialize_tick_claim_gc,
    make_tick_claim_gc_update,
    step_tick_claim_gc_workers,
    tick_claim_gc_parameter_count,
    validate_tick_claim_gc_checkpoint_resume,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "manifests"
PILOT_MANIFEST = OUTPUT_DIR / "tick_claim_gc_pilot_v1.json"
VALIDATION_PATH = OUTPUT_DIR / "tick_claim_gc_v1_validation.json"
RESOLVED_PATH = OUTPUT_DIR / "tick_claim_gc_v1_resolved.json"
SOURCE_PATHS = (
    ROOT / "src" / "hackrl" / "tick_claim.py",
    ROOT / "src" / "hackrl" / "tick_claim_oracle.py",
    ROOT / "src" / "hackrl" / "tick_claim_gc.py",
    ROOT / "scripts" / "run_tick_claim_gc.py",
    ROOT / "scripts" / "summarize_tick_claim_gc_calibration.py",
    ROOT / "scripts" / "validate_tick_claim_gc.py",
    ROOT / "tests" / "test_tick_claim.py",
    ROOT / "tests" / "test_tick_claim_gc.py",
    PILOT_MANIFEST,
)


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _batched(tree, count):
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (count,) + value.shape), tree
    )


def _termination_checks():
    config = TickClaimGCConfig(
        num_envs=2,
        num_steps=2,
        num_updates=1,
        minibatch_size=2,
        hidden_size=16,
    )
    _, runner = initialize_tick_claim_gc(config)
    initial = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    delivery = initial.replace(
        player_position=initial.delivery_position + jnp.asarray((-1, 0)),
        grain=jnp.asarray(3, dtype=jnp.int32),
        tick=jnp.asarray(10, dtype=jnp.int32),
    )
    runner = runner.replace(env_state=_batched(delivery, 2))
    goal_next, goal_event = step_tick_claim_gc_workers(
        runner,
        jnp.full((2,), TickClaimAction.DELIVER, dtype=jnp.int32),
        config,
    )
    goal_without_reset = bool(
        jnp.all(goal_event.goal_done)
        and not jnp.any(goal_event.world_done)
        and jnp.all(goal_next.env_state.tick == 11)
        and not jnp.any(goal_next.command_active)
    )

    timeout = initial.replace(tick=jnp.asarray(127, dtype=jnp.int32))
    runner = runner.replace(env_state=_batched(timeout, 2))
    world_next, world_event = step_tick_claim_gc_workers(
        runner,
        jnp.full((2,), TickClaimAction.NOOP, dtype=jnp.int32),
        config,
    )
    world_reset_once = bool(
        not jnp.any(world_event.goal_done)
        and jnp.all(world_event.world_done)
        and jnp.all(world_event.reset_count == 1)
        and jnp.all(world_next.env_state.tick == 0)
    )

    both = delivery.replace(tick=jnp.asarray(127, dtype=jnp.int32))
    runner = runner.replace(env_state=_batched(both, 2))
    both_next, both_event = step_tick_claim_gc_workers(
        runner,
        jnp.full((2,), TickClaimAction.DELIVER, dtype=jnp.int32),
        config,
    )
    simultaneous = bool(
        jnp.all(both_event.goal_done)
        and jnp.all(both_event.world_done)
        and jnp.all(both_event.reward == 1)
        and jnp.all(both_event.reset_count == 1)
        and jnp.all(both_next.env_state.tick == 0)
    )
    return {
        "goal_done_preserves_world": goal_without_reset,
        "world_done_resets_exactly_once": world_reset_once,
        "simultaneous_goal_world_records_reward_then_resets": simultaneous,
    }


def _checkpoint_and_frozen_eval_checks():
    config = TickClaimGCConfig(
        num_envs=8,
        num_steps=4,
        num_updates=1,
        minibatch_size=8,
        hidden_size=32,
        sample_repeats_per_state=1,
    )
    with tempfile.TemporaryDirectory(prefix="tick-claim-gc-checkpoint-") as temp:
        checkpoint = validate_tick_claim_gc_checkpoint_resume(temp, config)
    network, runner = initialize_tick_claim_gc(config)
    update = jax.jit(make_tick_claim_gc_update(network, config))
    runner, _ = update(runner)
    jax.block_until_ready(runner.global_update)
    before = serialization.to_bytes(runner)
    evaluation = evaluate_tick_claim_gc_frozen(
        network,
        runner.train_state.params,
        variant=TickClaimVariant.FIXED,
        stochastic=False,
        repeats_per_state=1,
        seed_base=20_000,
        learner_seed=0,
    )
    return checkpoint, {
        "learner_state_byte_identical": before == serialization.to_bytes(runner),
        "episodes": evaluation["overall"]["episodes"],
        "expected_episodes": 64,
    }


def _full_shape_throughput():
    config = TickClaimGCConfig(num_updates=2)
    network, runner = initialize_tick_claim_gc(config)
    update = jax.jit(make_tick_claim_gc_update(network, config))
    durations = []
    for _ in range(2):
        started = time.perf_counter()
        runner, metrics = update(runner)
        jax.block_until_ready(metrics["loss"])
        durations.append(time.perf_counter() - started)
    return {
        "device_platform": jax.devices()[0].platform,
        "batch_size": config.batch_size,
        "compile_and_first_update_seconds": durations[0],
        "steady_second_update_seconds": durations[1],
        "steady_transitions_per_second": config.batch_size / durations[1],
        "updates_executed": 2,
        "final_environment_steps": int(runner.env_steps),
    }


def main():
    fixed_network, fixed = initialize_tick_claim_gc(TickClaimGCConfig())
    _, mutant = initialize_tick_claim_gc(
        TickClaimGCConfig(variant=TickClaimVariant.MUTANT.value)
    )
    parameter_count = tick_claim_gc_parameter_count(fixed.train_state.params)
    paired_parameters = bool(
        jax.tree_util.tree_all(
            jax.tree.map(
                jnp.array_equal,
                fixed.train_state.params,
                mutant.train_state.params,
            )
        )
    )
    paired_environment = bool(
        jax.tree_util.tree_all(
            jax.tree.map(jnp.array_equal, fixed.env_state, mutant.env_state)
        )
    )
    initial_goal_vectors = jax.vmap(
        lambda state: tick_claim_goal_vector(observe_tick_claim(state))
    )(fixed.env_state)
    false_seen = jnp.logical_and(
        fixed.seen_goals[None, :], jnp.logical_not(initial_goal_vectors)
    )
    every_worker_has_false_seen = bool(jnp.all(jnp.any(false_seen, axis=1)))
    del fixed_network
    termination = _termination_checks()
    checkpoint, frozen = _checkpoint_and_frozen_eval_checks()
    throughput = _full_shape_throughput()
    checks = {
        "parameter_count_matches_manifest": parameter_count == 2_640_181,
        "fixed_mutant_initial_parameters_equal": paired_parameters,
        "fixed_mutant_initial_environment_equal": paired_environment,
        "initial_seen_goal_count": int(jnp.sum(fixed.seen_goals)),
        "every_initial_worker_has_false_seen_goal": every_worker_has_false_seen,
        **termination,
        "checkpoint_resume_all_equal": all(checkpoint.values()),
        "frozen_evaluation_state_immutable": frozen[
            "learner_state_byte_identical"
        ],
        "frozen_evaluation_state_count_matches": frozen["episodes"]
        == frozen["expected_episodes"],
        "full_shape_two_updates_completed": throughput[
            "final_environment_steps"
        ]
        == 2 * TickClaimGCConfig().batch_size,
    }
    passed = all(
        value is True
        for key, value in checks.items()
        if key != "initial_seen_goal_count"
    ) and checks["initial_seen_goal_count"] == 3
    validation = {
        "schema_version": "hackrl_gc_validation_v1",
        "passed": passed,
        "checks": checks,
        "checkpoint_resume": checkpoint,
        "frozen_evaluation": frozen,
        "full_shape_throughput": throughput,
        "parameter_count": parameter_count,
    }
    _write_json(VALIDATION_PATH, validation)

    revision = _git_sha()
    source_hashes = {
        str(path.relative_to(ROOT)): _sha256(path) for path in SOURCE_PATHS
    }
    dirty_source = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            "HEAD",
            "--",
            *(str(path.relative_to(ROOT)) for path in SOURCE_PATHS),
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    resolved = {
        "schema_version": "hackrl_gc_resolved_v1",
        "candidate_id": "TICK-CLAIM",
        "profile": "hackrl_gc_v1",
        "implementation_code_sha": revision,
        "pilot_manifest": PILOT_MANIFEST.name,
        "pilot_manifest_sha256": _sha256(PILOT_MANIFEST),
        "validation": VALIDATION_PATH.name,
        "validation_sha256": _sha256(VALIDATION_PATH),
        "source_hashes": source_hashes,
        "source_paths_clean_at_validation": not dirty_source,
        "dirty_source_paths": dirty_source,
        "actual_parameter_count": parameter_count,
        "runtime": {
            "python": os.sys.version.split()[0],
            "jax": importlib.metadata.version("jax"),
            "flax": importlib.metadata.version("flax"),
            "optax": importlib.metadata.version("optax"),
            "distrax": importlib.metadata.version("distrax"),
            "craftax": importlib.metadata.version("craftax"),
            "device_platform": throughput["device_platform"],
        },
        "runnable": passed and not dirty_source,
        "queue_gate": (
            "open_for_frozen_six_cell_calibration"
            if passed and not dirty_source
            else "closed"
        ),
        "design_manifest_status_preserved": "design_only_and_runnable_false",
        "pack_restore_status": "deferred_pending_payoff_gate",
    }
    _write_json(RESOLVED_PATH, resolved)
    print(json.dumps(resolved, indent=2, sort_keys=True))
    if not resolved["runnable"]:
        raise SystemExit("GC validation gate did not open")


if __name__ == "__main__":
    main()
