#!/usr/bin/env python3
"""Validate the preregistered no-learning rare-overwrite regime map."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import jax
import numpy as np

from hackrl.dual_leo import init_dual_leo_teacher, load_dual_checkpoint
from hackrl.pack_inventory_overwrite_gate import validate_reference_paths
from hackrl.pack_inventory_overwrite_rare import (
    build_risk_profile,
    combine_risk_profiles,
    evaluate_regime_point,
    find_return_crossover,
)
from hackrl.pack_inventory_overwrite_rare_kernel import (
    validate_stochastic_overwrite_kernel_vectorized,
)
from hackrl.pack_restore import PackRestoreVariant
from hackrl.pack_restore_gc import (
    _FROZEN_EVAL_CACHE,
    evaluate_pack_restore_gc_frozen,
)
from run_dual_leo_compare import ENVS


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_rare_v1.json"
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
RUN_ROOT = Path(SPEC["run_root"])
SOURCE_ROOT = Path(SPEC["source_policies"]["root"])
SOURCE_FILES = (
    Path("docs/manifests/pack_inventory_overwrite_rare_v1.json"),
    Path("scripts/validate_pack_inventory_overwrite_rare.py"),
    Path("src/hackrl/pack_inventory_overwrite_gate.py"),
    Path("src/hackrl/pack_inventory_overwrite_rare.py"),
    Path("src/hackrl/pack_inventory_overwrite_rare_kernel.py"),
    Path("src/hackrl/pack_restore.py"),
    Path("src/hackrl/pack_restore_gc.py"),
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


def _checkpoint(seed: int) -> Path:
    return (
        SOURCE_ROOT
        / f"seed{seed}"
        / "checkpoints"
        / SPEC["source_policies"]["checkpoint"]
    )


def _checkpoint_hashes() -> dict[str, str]:
    return {
        str(_checkpoint(seed) / name): _sha256(_checkpoint(seed) / name)
        for seed in SPEC["source_policies"]["seeds"]
        for name in ("state.msgpack", "config.json", "metadata.json", "arm.json")
    }


def _capacity() -> dict:
    existing = (
        sum(path.stat().st_size for path in RUN_ROOT.rglob("*") if path.is_file())
        if RUN_ROOT.exists()
        else 0
    )
    checkpoint_paths = [Path(path) for path in _checkpoint_hashes()]
    source_bytes = sum(path.stat().st_size for path in checkpoint_paths)
    projected = int(SPEC["storage"]["projected_additional_bytes"])
    reserve = int(SPEC["storage"]["safety_reserve_bytes"])
    home = shutil.disk_usage("/home/ext_csv")
    raid = shutil.disk_usage("/raid/ext_csv")
    if raid.free <= projected + reserve:
        raise RuntimeError("RAID free space is below projected output plus reserve")
    return {
        "measured_unix": time.time(),
        "home": home._asdict(),
        "raid": raid._asdict(),
        "existing_run_bytes": existing,
        "source_checkpoint_bytes": source_bytes,
        "projected_additional_bytes": projected,
        "safety_reserve_bytes": reserve,
        "chosen_filesystem": SPEC["storage"]["chosen_filesystem"],
        "checkpoint_writes": 0,
    }


def _source_manifest() -> dict:
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=REPOSITORY, text=True
    ).splitlines()
    return {
        "execution_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip(),
        "git_status": status,
        "source_sha256": {
            str(path): _sha256(REPOSITORY / path) for path in SOURCE_FILES
        },
        "checkpoint_sha256": _checkpoint_hashes(),
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
    }


def _verify_prerequisites() -> dict:
    prerequisites = SPEC["prerequisites"]
    result_path = Path(prerequisites["deterministic_gate_result"])
    reference_path = Path(prerequisites["reference_paths"])
    checks = {
        "deterministic_result_hash": (
            _sha256(result_path) == prerequisites["deterministic_gate_result_sha256"]
        ),
        "reference_hash": (
            _sha256(reference_path) == prerequisites["reference_paths_sha256"]
        ),
        "deterministic_status": (
            json.loads(result_path.read_text(encoding="utf-8"))["status"]
            == prerequisites["required_gate_status"]
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"prerequisite verification failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "deterministic_result": str(result_path),
        "reference_paths": str(reference_path),
    }


def _load_seed(seed: int):
    env_spec = ENVS["pack"]
    checkpoint = _checkpoint(seed)
    config = env_spec["from_payload"](
        json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    )
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    arm = json.loads((checkpoint / "arm.json").read_text(encoding="utf-8"))
    expected_update = int(SPEC["source_policies"]["global_update"])
    if (
        config.seed != seed
        or config.variant != PackRestoreVariant.FIXED.value
        or config.policy_hidden_size != 512
        or config.teacher_hidden_size != 512
        or config.num_envs != 512
        or config.num_steps != 64
        or metadata.get("global_update") != expected_update
        or arm != {"imitate_teacher": True, "learn_teacher": True}
    ):
        raise RuntimeError(f"source checkpoint contract failed for seed {seed}")

    network, runner = env_spec["initialize"](config)
    model_inputs = env_spec["inputs"](runner.env_state, runner.current_goal)
    _, teacher, _ = init_dual_leo_teacher(
        config,
        model_inputs[0],
        model_inputs[1],
        env_spec["goals"],
        env_spec["actions"],
    )
    runner, teacher = load_dual_checkpoint(checkpoint, runner, teacher)
    return checkpoint, config, network, runner, teacher


def _close(left: float, right: float) -> bool:
    tolerance = float(SPEC["exact_computation"]["endpoint_tolerance"])
    return abs(float(left) - float(right)) <= tolerance


def _evaluate_seed(seed: int, reference: dict) -> tuple[dict, dict]:
    checkpoint, config, network, runner, teacher = _load_seed(seed)
    before = _tree_sha256((runner, teacher))
    evaluations = {}
    for variant in (
        PackRestoreVariant.FIXED,
        PackRestoreVariant.INVENTORY_OVERWRITE,
    ):
        evaluations[variant.value] = evaluate_pack_restore_gc_frozen(
            network,
            runner.train_state.params,
            variant=variant,
            split="validation",
            stochastic=False,
            repeats_per_state=1,
            seed_base=57_000,
            learner_seed=seed,
            source_growth_period=config.source_growth_period,
            record_episodes=True,
            natural_only=True,
        )
        if evaluations[variant.value]["overall"]["episodes"] != 32:
            raise AssertionError("frozen panel did not contain 32 natural starts")
    after = _tree_sha256((runner, teacher))
    if before != after:
        raise AssertionError("frozen evaluation mutated the restored learner")

    fixed = evaluations[PackRestoreVariant.FIXED.value]
    harmful = evaluations[PackRestoreVariant.INVENTORY_OVERWRITE.value]
    profile = build_risk_profile(fixed["episode_records"], reference["episodes"])
    p0 = evaluate_regime_point(profile, 0.0)
    p1 = evaluate_regime_point(profile, 1.0)
    endpoint_checks = {
        "fixed_success": _close(
            p0["existing_success_rate"], fixed["overall"]["success_rate"]
        ),
        "fixed_return": _close(
            p0["existing_expected_discounted_return"],
            fixed["overall"]["mean_discounted_return"],
        ),
        "deterministic_success": _close(
            p1["existing_success_rate"], harmful["overall"]["success_rate"]
        ),
        "deterministic_return": _close(
            p1["existing_expected_discounted_return"],
            harmful["overall"]["mean_discounted_return"],
        ),
        "all_fixed_starts_succeed": fixed["overall"]["success_rate"] == 1.0,
    }
    if not all(endpoint_checks.values()):
        raise AssertionError(f"formula endpoint failed for seed {seed}: {endpoint_checks}")

    grid = [
        evaluate_regime_point(profile, float(probability))
        for probability in SPEC["exact_computation"]["report_probabilities"]
    ]
    for point in grid:
        difference = float(point["existing_minus_reference_return"])
        tolerance = float(SPEC["exact_computation"]["endpoint_tolerance"])
        point["higher_expected_return"] = (
            "existing_policy"
            if difference > tolerance
            else "zero_loss_reference"
            if difference < -tolerance
            else "equal_within_tolerance"
        )
    row = {
        "seed": seed,
        "checkpoint": str(checkpoint),
        "frozen_state_sha256_before": before,
        "frozen_state_sha256_after": after,
        "optimizer_updates": 0,
        "collector_slots": 0,
        "checkpoint_writes": 0,
        "evaluations": evaluations,
        "endpoint_checks": endpoint_checks,
        "regime_map": grid,
        "crossover": find_return_crossover(profile),
    }
    return row, profile


def _aggregate_direct(rows: list[dict], variant: str, metric: str) -> float:
    return float(
        np.mean([row["evaluations"][variant]["overall"][metric] for row in rows])
    )


def main() -> None:
    started = time.time()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(RUN_ROOT / "capacity_start.json", _capacity())
    source_before = _source_manifest()
    if source_before["git_status"]:
        raise RuntimeError("no-learning run requires a clean committed worktree")
    _write_json(RUN_ROOT / "run_manifest.json", source_before)

    prerequisite = _verify_prerequisites()
    _write_json(RUN_ROOT / "prerequisites.json", prerequisite)
    kernel = validate_stochastic_overwrite_kernel_vectorized()
    _write_json(RUN_ROOT / "kernel_validation.json", kernel)

    reference = validate_reference_paths()
    pinned_reference = json.loads(
        Path(SPEC["prerequisites"]["reference_paths"]).read_text(encoding="utf-8")
    )
    reference_matches = reference == pinned_reference
    if not reference_matches:
        raise AssertionError("fresh reference replay differs from pinned prior artifact")
    _write_json(RUN_ROOT / "reference_paths.json", reference)

    rows = []
    profiles = []
    for seed in SPEC["source_policies"]["seeds"]:
        row, profile = _evaluate_seed(int(seed), reference)
        rows.append(row)
        profiles.append(profile)
        _write_json(RUN_ROOT / f"seed{seed}.json", row)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "risk_count_distribution": profile["risk_count_distribution"],
                    "crossover": row["crossover"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _FROZEN_EVAL_CACHE.clear()
        jax.clear_caches()

    aggregate_profile = combine_risk_profiles(profiles)
    aggregate_grid = [
        evaluate_regime_point(aggregate_profile, float(probability))
        for probability in SPEC["exact_computation"]["report_probabilities"]
    ]
    tolerance = float(SPEC["exact_computation"]["endpoint_tolerance"])
    for point in aggregate_grid:
        difference = float(point["existing_minus_reference_return"])
        point["higher_expected_return"] = (
            "existing_policy"
            if difference > tolerance
            else "zero_loss_reference"
            if difference < -tolerance
            else "equal_within_tolerance"
        )
    aggregate_crossover = find_return_crossover(aggregate_profile)
    risk_profiles = {
        "per_seed": {
            str(seed): profile
            for seed, profile in zip(SPEC["source_policies"]["seeds"], profiles)
        },
        "aggregate": aggregate_profile,
    }
    _write_json(RUN_ROOT / "risk_profiles.json", risk_profiles)

    p0 = aggregate_grid[0]
    p1 = aggregate_grid[-1]
    aggregate_endpoint_checks = {
        "fixed_success": _close(
            p0["existing_success_rate"],
            _aggregate_direct(rows, "fixed", "success_rate"),
        ),
        "fixed_return": _close(
            p0["existing_expected_discounted_return"],
            _aggregate_direct(rows, "fixed", "mean_discounted_return"),
        ),
        "deterministic_success": _close(
            p1["existing_success_rate"],
            _aggregate_direct(rows, "inventory_overwrite", "success_rate"),
        ),
        "deterministic_return": _close(
            p1["existing_expected_discounted_return"],
            _aggregate_direct(rows, "inventory_overwrite", "mean_discounted_return"),
        ),
    }
    if not all(aggregate_endpoint_checks.values()):
        raise AssertionError(
            f"aggregate formula endpoint failed: {aggregate_endpoint_checks}"
        )

    source_after = _source_manifest()
    if source_after["source_sha256"] != source_before["source_sha256"]:
        raise AssertionError("execution source changed during validation")
    if source_after["checkpoint_sha256"] != source_before["checkpoint_sha256"]:
        raise AssertionError("source checkpoint changed during validation")

    gates = {
        "prerequisites": prerequisite["passed"],
        "kernel_locality_and_endpoints": bool(kernel["passed"]),
        "reference_replay": bool(reference["passed"] and reference_matches),
        "all_source_policies_fixed_success": all(
            row["endpoint_checks"]["all_fixed_starts_succeed"] for row in rows
        ),
        "per_seed_formula_endpoints": all(
            all(row["endpoint_checks"].values()) for row in rows
        ),
        "aggregate_formula_endpoints": all(aggregate_endpoint_checks.values()),
        "source_integrity": True,
    }
    result = {
        "schema_version": "hackrl_pack_inventory_overwrite_rare_result_v1",
        "status": (
            "all_no_learning_regime_gates_passed_training_not_started"
            if all(gates.values())
            else "no_learning_regime_gate_failed_training_prohibited"
        ),
        "gates": gates,
        "kernel_validation": kernel,
        "reference_matches_pinned_artifact": reference_matches,
        "rows": rows,
        "aggregate": {
            "risk_count_distribution": aggregate_profile[
                "risk_count_distribution"
            ],
            "success_polynomial_by_survival_power": aggregate_profile[
                "success_polynomial_by_survival_power"
            ],
            "return_polynomial_by_survival_power": aggregate_profile[
                "return_polynomial_by_survival_power"
            ],
            "reference_success_rate": aggregate_profile["reference_success_rate"],
            "reference_mean_discounted_return": aggregate_profile[
                "reference_mean_discounted_return"
            ],
            "regime_map": aggregate_grid,
            "crossover": aggregate_crossover,
            "endpoint_checks": aggregate_endpoint_checks,
        },
        "policy_episodes": len(rows) * 2 * 32,
        "reference_episodes": len(reference["episodes"]),
        "optimizer_updates": 0,
        "collector_slots": 0,
        "checkpoint_writes": 0,
        "training_started": False,
        "source_hashes_unchanged": True,
        "reference_paths_entered_learning": False,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(RUN_ROOT / "result.json", result)
    _write_json(
        RUN_ROOT / "verified_complete.json",
        {
            "result_sha256": _sha256(RUN_ROOT / "result.json"),
            "completed_unix": time.time(),
            "status": result["status"],
        },
    )
    _write_json(RUN_ROOT / "capacity_end.json", _capacity())
    print(json.dumps(result["aggregate"], sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        _write_json(
            RUN_ROOT / "failure.json",
            {"failed_unix": time.time(), "traceback": traceback.format_exc()},
        )
        raise
