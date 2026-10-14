#!/usr/bin/env python3
"""Validate the behavior-anchor PPO implementation without launching training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from hackrl.dual_leo import init_dual_leo_teacher, load_dual_checkpoint
from hackrl.pack_behavior_anchor import (
    behavior_anchor_kl,
    build_pack_behavior_anchor,
    make_behavior_anchored_pack_restore_update,
)
from hackrl.pack_restore_gc import (
    NUM_ACTIONS,
    NUM_GOALS,
    _batch_inputs,
    categorical_source_kl,
    config_from_pack_restore_gc_payload,
    initialize_pack_restore_gc,
)


REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    REPOSITORY
    / "docs/manifests/pack_inventory_overwrite_behavior_anchor_ppo_v1.json"
)
OUTPUT_PATH = (
    REPOSITORY
    / "docs/manifests"
    / "pack_inventory_overwrite_behavior_anchor_ppo_v1_implementation_validation.json"
)
AUDIT_PATH = (
    REPOSITORY
    / "docs/manifests"
    / "pack_inventory_overwrite_behavior_anchor_ppo_v1_implementation_computation.json"
)
IMPLEMENTATION_SOURCES = (
    Path("docs/manifests/pack_inventory_overwrite_behavior_anchor_ppo_v1.json"),
    Path("src/hackrl/pack_restore.py"),
    Path("src/hackrl/pack_restore_gc.py"),
    Path("src/hackrl/pack_behavior_anchor.py"),
    Path("src/hackrl/dual_leo.py"),
    Path("scripts/validate_pack_inventory_overwrite_behavior_anchor_ppo.py"),
    Path("tests/test_pack_behavior_anchor.py"),
    Path("tests/test_pack_inventory_overwrite_behavior_anchor_ppo_design.py"),
)


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


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY, text=True
    ).strip()


def _checkpoint(manifest: dict, seed: int) -> Path:
    return (
        Path(manifest["source_root"])
        / f"seed{seed}"
        / "checkpoints"
        / manifest["source_checkpoint"]
    )


def _verify_inputs(manifest: dict) -> dict:
    checks = {}
    prerequisites = manifest["prerequisites"]
    for name in (
        "manual_gate_closeout",
        "learning_rate_result",
        "learning_rate_audit",
    ):
        path = REPOSITORY / prerequisites[name]
        checks[f"{name}_sha256"] = _sha256(path) == prerequisites[f"{name}_sha256"]
    for seed in manifest["seeds"]:
        checkpoint = _checkpoint(manifest, int(seed))
        for filename, expected in manifest["source_checkpoint_sha256"][str(seed)].items():
            checks[f"seed{seed}_{filename}_sha256"] = (
                _sha256(checkpoint / filename) == expected
            )
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise AssertionError(f"input verification failed: {failed}")
    return {"passed": True, "checks": checks}


def _load_source(manifest: dict, seed: int):
    checkpoint = _checkpoint(manifest, seed)
    config = config_from_pack_restore_gc_payload(
        json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    )
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if config.seed != seed or int(metadata["global_update"]) != int(
        manifest["source_global_update"]
    ):
        raise AssertionError(f"source identity mismatch for seed {seed}")
    network, template = initialize_pack_restore_gc(config)
    model_inputs = _batch_inputs(template.env_state, template.current_goal)
    _, teacher_template, _ = init_dual_leo_teacher(
        config,
        model_inputs[0],
        model_inputs[1],
        NUM_GOALS,
        NUM_ACTIONS,
    )
    runner, teacher = load_dual_checkpoint(checkpoint, template, teacher_template)
    return checkpoint, config, network, runner, teacher


def _maximum_absolute(tree) -> float:
    leaves = [np.asarray(jax.device_get(value)) for value in jax.tree.leaves(tree)]
    return max(float(np.max(np.abs(value))) for value in leaves)


def _anchor_contract(manifest: dict, seed: int) -> dict:
    checkpoint, config, network, runner, teacher = _load_source(manifest, seed)
    frozen_before = _tree_sha256((runner.train_state, teacher))
    source_parameters_before = _tree_sha256(runner.train_state.params)
    anchor, metadata = build_pack_behavior_anchor(
        network,
        runner.train_state.params,
        source_growth_period=config.source_growth_period,
        horizon_ticks=int(manifest["evaluation"]["horizon_ticks"]),
    )
    source_parameters_after = _tree_sha256(runner.train_state.params)
    frozen_after = _tree_sha256((runner.train_state, teacher))

    initial_kl = behavior_anchor_kl(network, runner.train_state.params, anchor)
    parameter_gradient = jax.grad(
        lambda parameters: behavior_anchor_kl(network, parameters, anchor)
    )(runner.train_state.params)
    source_logit_gradient = jax.grad(
        lambda logits: jnp.mean(
            categorical_source_kl(logits, anchor.source_logits)
        )
    )(anchor.source_logits)
    current_logit_gradient = jax.grad(
        lambda logits: jnp.mean(
            categorical_source_kl(anchor.source_logits, logits)
        )
    )(anchor.source_logits)

    shift = jnp.linspace(-1.0, 1.0, anchor.source_logits.shape[-1])
    shifted = anchor.source_logits + shift
    objective = lambda logits: jnp.mean(
        categorical_source_kl(anchor.source_logits, logits)
    )
    shifted_before = objective(shifted)
    shifted_gradient = jax.grad(objective)(shifted)
    shifted_after = objective(shifted - 0.5 * shifted_gradient)

    checks = {
        "source_actor_and_teacher_unchanged": frozen_before == frozen_after,
        "source_parameter_hash_unchanged": (
            source_parameters_before == source_parameters_after
        ),
        "all_32_train_starts_present": metadata.start_count == 32,
        "complete_action_logits_present": anchor.source_logits.shape[-1] == NUM_ACTIONS,
        "source_parameters_not_retained": metadata.source_parameters_retained is False,
        "actor_has_no_mutable_normalization_collection": (
            metadata.source_has_mutable_normalization_statistics is False
        ),
        "initial_kl_near_zero": abs(float(initial_kl)) <= 1e-6,
        "initial_parameter_gradient_near_zero": (
            _maximum_absolute(parameter_gradient) <= 1e-6
        ),
        "source_logit_gradient_zero": _maximum_absolute(source_logit_gradient) == 0.0,
        "current_logit_gradient_near_zero": (
            _maximum_absolute(current_logit_gradient) <= 1e-6
        ),
        "changed_output_step_reduces_kl": float(shifted_after) < float(shifted_before),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise AssertionError(f"seed {seed} anchor contract failed: {failed}")
    return {
        "seed": seed,
        "checkpoint": str(checkpoint),
        "source_parameter_sha256": source_parameters_before,
        "anchor_sha256": _tree_sha256(anchor),
        "metadata": asdict(metadata),
        "initial_kl": float(initial_kl),
        "initial_parameter_gradient_max_abs": _maximum_absolute(parameter_gradient),
        "source_logit_gradient_max_abs": _maximum_absolute(source_logit_gradient),
        "current_logit_gradient_max_abs": _maximum_absolute(current_logit_gradient),
        "changed_output_kl_before": float(shifted_before),
        "changed_output_kl_after_one_step": float(shifted_after),
        "checks": checks,
    }


def _capacity(manifest: dict) -> dict:
    run_root = Path(manifest["run_root"])
    existing = (
        sum(path.stat().st_size for path in run_root.rglob("*") if path.is_file())
        if run_root.exists()
        else 0
    )
    home = shutil.disk_usage("/home/ext_csv")
    raid = shutil.disk_usage("/raid/ext_csv")
    storage = manifest["checkpoint_storage"]
    minimum = int(storage["minimum_required_free_bytes"])
    if raid.free <= minimum:
        raise RuntimeError("RAID free space is below projected peak plus reserve")
    return {
        "measured_unix": time.time(),
        "home": home._asdict(),
        "raid": raid._asdict(),
        "existing_run_bytes": existing,
        "projected_peak_writes_bytes": int(storage["projected_peak_writes_bytes"]),
        "safety_reserve_bytes": int(storage["safety_reserve_bytes"]),
        "minimum_required_free_bytes": minimum,
        "chosen_filesystem": storage["chosen_filesystem"],
        "checkpoint_writes": 0,
    }


def validate_contract() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["status"] != "implementation_ready_training_not_authorized":
        raise AssertionError("manifest does not preserve the training approval boundary")
    signature = inspect.signature(make_behavior_anchored_pack_restore_update)
    if "source_parameters" in signature.parameters:
        raise AssertionError("updater API must not accept source parameters")
    inputs = _verify_inputs(manifest)
    anchors = [_anchor_contract(manifest, int(seed)) for seed in manifest["seeds"]]
    return {
        "schema_version": (
            "hackrl_pack_inventory_overwrite_behavior_anchor_ppo_implementation_validation_v1"
        ),
        "status": "implementation_contract_valid_training_not_started",
        "manifest_id": manifest["manifest_id"],
        "execution_head": _git("rev-parse", "HEAD"),
        "source_sha256": {
            str(path): _sha256(REPOSITORY / path) for path in IMPLEMENTATION_SOURCES
        },
        "inputs": inputs,
        "capacity": _capacity(manifest),
        "anchor_contracts": anchors,
        "fairness_contract": {
            "beta_zero_uses_legacy_updater_directly": True,
            "same_actor_critic_and_adam_required": True,
            "same_collector_budget_and_evaluation_states_required": True,
            "anchor_does_not_update_normalization_statistics": True,
            "anchor_does_not_add_optimizer_applications": True,
            "frozen_source_baseline_requires_no_training_job": True,
            "unit_test": "tests/test_pack_behavior_anchor.py",
        },
        "evaluation_contract": manifest["evaluation"],
        "compute_accounting": {
            "collector_slots_changed_by_anchor": False,
            "optimizer_schedule_changed_by_anchor": False,
            "extra_metric": "behavior_anchor_forward_state_evaluations",
            "wall_clock_seconds_must_be_recorded_per_update_if_training_is_authorized": True,
        },
        "training_authorized": False,
        "training_jobs_started": 0,
        "checkpoint_writes": 0,
        "validation_limits": [
            "This is a structural and finite implementation check, not evidence "
            "that anchored PPO improves task performance.",
            "The validator and implementation share an author and KL "
            "implementation; manual-formula and legacy-path unit tests reduce "
            "but do not eliminate verifier-designer dependence.",
            "Low KL on the TRAIN anchor does not establish normal-task "
            "preservation outside those source-visited states.",
            "VALIDATION and TEST are development panels, not untouched confirmation data.",
        ],
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "created_unix": time.time(),
    }


def _run_unit_tests() -> dict:
    command = [
        "/home/ext_csv/miniconda3/envs/offrl/bin/python",
        "-m",
        "pytest",
        "-q",
        "tests/test_pack_behavior_anchor.py",
        "tests/test_pack_inventory_overwrite_behavior_anchor_ppo_design.py",
        "tests/test_pack_restore_gc.py",
        "tests/test_pack_restore_gc_stochastic_overwrite.py",
        "tests/test_pack_inventory_overwrite_ppo.py",
    ]
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONPATH"] = "src:scripts"
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "contract tests failed:\n" + completed.stdout + "\n" + completed.stderr
        )
    return {
        "argv": command,
        "exit_status": completed.returncode,
        "stdout": completed.stdout.strip(),
    }


def _computation_audit(result: dict, started_at: str, runtime_seconds: float) -> dict:
    return {
        "schema_version": 1,
        "claim_id": "pack_inventory_overwrite_behavior_anchor_ppo_v1_implementation",
        "repository": {"commit": result["execution_head"], "dirty": False},
        "command": (
            "JAX_PLATFORMS=cpu PYTHONPATH=src:scripts "
            "/home/ext_csv/miniconda3/envs/offrl/bin/python "
            "scripts/validate_pack_inventory_overwrite_behavior_anchor_ppo.py "
            "--write-validation"
        ),
        "environment": {
            "software": [
                f"Python {os.sys.version.split()[0]}",
                f"JAX {result['jax_version']}",
                f"NumPy {result['numpy_version']}",
            ],
            "hardware": ", ".join(result["jax_devices"]),
        },
        "mathematics": {
            "assertion_tested": (
                "For seeds 30--34, the deterministic full-logit TRAIN anchors "
                "are constructed without mutating source state; KL(source || "
                "current) has a stopped source gradient, is stationary at equal "
                "outputs within 1e-6, and decreases under the tested changed-logit "
                "descent step."
            ),
            "coefficient_domain": "IEEE-754 float32 logits and probabilities",
            "conventions": (
                "Categorical KL sums over all PACK actions per state and then "
                "averages source-trajectory states equally."
            ),
            "inputs": [
                "Pinned PACK Dual fixed-kernel adapt_4096 checkpoints for seeds 30--34",
                "All 32 natural TRAIN starts per seed",
                "128-tick trajectory cap",
            ],
            "bounds": {"learner_seeds": 5, "starts_per_seed": 32, "horizon_ticks": 128},
            "non_claims": result["validation_limits"],
        },
        "randomness": {"used": False, "generator": "", "seed": None},
        "run": {
            "started_at": started_at,
            "runtime_seconds": runtime_seconds,
            "exit_status": 0,
        },
        "outputs": [
            {
                "path": str(OUTPUT_PATH.relative_to(REPOSITORY)),
                "sha256": _sha256(OUTPUT_PATH),
            }
        ],
        "checks": [
            "Pinned prerequisite and checkpoint hashes matched.",
            "All source actor-critic, Adam, and teacher hashes were unchanged.",
            "Full action logits and all 32 TRAIN starts were retained for every seed.",
            "Manual KL, gradient direction, beta-zero parity, empty-minibatch, "
            "and PACK regression tests passed.",
            "RAID free space exceeded projected peak writes plus the 8 GiB "
            "reserve; no checkpoints were written.",
        ],
        "result": (
            "Implementation and finite contract assertions verified for the five "
            "pinned source checkpoints; training was not started."
        ),
        "residual_risks": result["validation_limits"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate-contract", action="store_true")
    action.add_argument("--write-validation", action="store_true")
    arguments = parser.parse_args()
    if arguments.write_validation and _git("status", "--short"):
        raise RuntimeError("commit implementation and use a clean tree before writing validation")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    test_result = _run_unit_tests() if arguments.write_validation else None
    result = validate_contract()
    result["unit_tests"] = test_result
    if arguments.write_validation:
        _write_json(OUTPUT_PATH, result)
        audit = _computation_audit(result, started_at, time.perf_counter() - started)
        _write_json(AUDIT_PATH, audit)
        print(OUTPUT_PATH)
        print(AUDIT_PATH)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
