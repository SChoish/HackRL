#!/usr/bin/env python3
"""Evaluate frozen learned policies under the exact stochastic-overwrite model."""

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
from hackrl.pack_restore import PackRestoreVariant
from hackrl.pack_restore_gc import (
    _FROZEN_EVAL_CACHE,
    evaluate_pack_restore_gc_frozen,
    load_pack_restore_gc_checkpoint,
)
from run_dual_leo_compare import ENVS


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_policy_panel_v1.json"
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
RUN_ROOT = Path(SPEC["run_root"])
POLICY_NAMES = ("original", "normal_continue", "p1_adapt")
SOURCE_FILES = (
    Path("docs/manifests/pack_inventory_overwrite_policy_panel_v1.json"),
    Path("scripts/validate_pack_inventory_overwrite_policy_panel.py"),
    Path("src/hackrl/pack_inventory_overwrite_gate.py"),
    Path("src/hackrl/pack_inventory_overwrite_rare.py"),
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


def _checkpoint(policy: str, seed: int) -> Path:
    policy_spec = SPEC["policies"][policy]
    root = Path(policy_spec["root"])
    if policy == "original":
        return root / f"seed{seed}" / "checkpoints" / policy_spec["checkpoint"]
    return root / f"seed{seed}" / "checkpoints" / policy_spec["checkpoint"]


def _checkpoint_file_names(policy: str) -> tuple[str, ...]:
    if policy == "original":
        return ("state.msgpack", "config.json", "metadata.json", "arm.json")
    return (
        "state.msgpack",
        "config.json",
        "metadata.json",
        "receipt.json",
        "tracker.json",
    )


def _checkpoint_hashes() -> dict[str, str]:
    return {
        str(_checkpoint(policy, int(seed)) / name): _sha256(
            _checkpoint(policy, int(seed)) / name
        )
        for policy in POLICY_NAMES
        for seed in SPEC["seeds"]
        for name in _checkpoint_file_names(policy)
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
        "input_checkpoint_bytes": source_bytes,
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
    cases = (
        (
            "rare_regime",
            Path(prerequisites["rare_regime_result"]),
            prerequisites["rare_regime_result_sha256"],
            prerequisites["rare_regime_required_status"],
        ),
        (
            "deterministic_ppo",
            Path(prerequisites["deterministic_ppo_result"]),
            prerequisites["deterministic_ppo_result_sha256"],
            prerequisites["deterministic_ppo_required_status"],
        ),
    )
    checks = {}
    for name, path, expected_hash, expected_status in cases:
        result = json.loads(path.read_text(encoding="utf-8"))
        checks[f"{name}_hash"] = _sha256(path) == expected_hash
        checks[f"{name}_status"] = result.get("status") == expected_status
    reference = Path(prerequisites["reference_paths"])
    checks["reference_hash"] = (
        _sha256(reference) == prerequisites["reference_paths_sha256"]
    )
    if not all(checks.values()):
        raise AssertionError(f"prerequisite verification failed: {checks}")
    return {"passed": True, "checks": checks}


def _load_policy(policy: str, seed: int):
    env_spec = ENVS["pack"]
    checkpoint = _checkpoint(policy, seed)
    config = env_spec["from_payload"](
        json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    )
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    policy_spec = SPEC["policies"][policy]
    if config.seed != seed or metadata.get("global_update") != int(
        policy_spec["global_update"]
    ):
        raise RuntimeError(f"checkpoint identity failed for {policy} seed {seed}")
    expected_variant = (
        PackRestoreVariant.FIXED.value
        if policy != "p1_adapt"
        else PackRestoreVariant.INVENTORY_OVERWRITE.value
    )
    if config.variant != expected_variant:
        raise RuntimeError(f"checkpoint variant failed for {policy} seed {seed}")

    network, template = env_spec["initialize"](config)
    if policy == "original":
        model_inputs = env_spec["inputs"](template.env_state, template.current_goal)
        _, teacher, _ = init_dual_leo_teacher(
            config,
            model_inputs[0],
            model_inputs[1],
            env_spec["goals"],
            env_spec["actions"],
        )
        runner, teacher = load_dual_checkpoint(checkpoint, template, teacher)
        frozen_tree = (runner, teacher)
    else:
        runner = load_pack_restore_gc_checkpoint(checkpoint, template, config)
        frozen_tree = runner
    return checkpoint, config, network, runner, frozen_tree


def _close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-12


def _evaluate_policy_seed(
    policy: str,
    seed: int,
    reference: dict,
) -> tuple[dict, dict]:
    checkpoint, config, network, runner, frozen_tree = _load_policy(policy, seed)
    before = _tree_sha256(frozen_tree)
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
            seed_base=61_000,
            learner_seed=seed,
            source_growth_period=config.source_growth_period,
            record_episodes=True,
            natural_only=True,
        )
    after = _tree_sha256(frozen_tree)
    if before != after:
        raise AssertionError(f"frozen evaluation mutated {policy} seed {seed}")

    fixed = evaluations[PackRestoreVariant.FIXED.value]
    harmful = evaluations[PackRestoreVariant.INVENTORY_OVERWRITE.value]
    profile = build_risk_profile(fixed["episode_records"], reference["episodes"])
    p0 = evaluate_regime_point(profile, 0.0)
    p1 = evaluate_regime_point(profile, 1.0)
    checks = {
        "all_fixed_starts_succeed": fixed["overall"]["success_rate"] == 1.0,
        "fixed_success_endpoint": _close(
            p0["existing_success_rate"], fixed["overall"]["success_rate"]
        ),
        "fixed_return_endpoint": _close(
            p0["existing_expected_discounted_return"],
            fixed["overall"]["mean_discounted_return"],
        ),
        "p1_success_endpoint": _close(
            p1["existing_success_rate"], harmful["overall"]["success_rate"]
        ),
        "p1_return_endpoint": _close(
            p1["existing_expected_discounted_return"],
            harmful["overall"]["mean_discounted_return"],
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"endpoint check failed for {policy} seed {seed}: {checks}")
    grid = [
        evaluate_regime_point(profile, float(probability))
        for probability in SPEC["exact_computation"]["report_probabilities"]
    ]
    return {
        "policy": policy,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "frozen_state_sha256_before": before,
        "frozen_state_sha256_after": after,
        "evaluations": evaluations,
        "endpoint_checks": checks,
        "regime_map": grid,
        "reference_crossover": find_return_crossover(profile),
        "optimizer_updates": 0,
        "collector_slots": 0,
        "checkpoint_writes": 0,
    }, profile


def _aggregate_family(policy: str, rows: list[dict], profiles: list[dict]) -> dict:
    profile = combine_risk_profiles(profiles)
    grid = [
        evaluate_regime_point(profile, float(probability))
        for probability in SPEC["exact_computation"]["report_probabilities"]
    ]
    fixed_success = float(
        np.mean(
            [row["evaluations"]["fixed"]["overall"]["success_rate"] for row in rows]
        )
    )
    fixed_return = float(
        np.mean(
            [
                row["evaluations"]["fixed"]["overall"]["mean_discounted_return"]
                for row in rows
            ]
        )
    )
    harmful_success = float(
        np.mean(
            [
                row["evaluations"]["inventory_overwrite"]["overall"][
                    "success_rate"
                ]
                for row in rows
            ]
        )
    )
    harmful_return = float(
        np.mean(
            [
                row["evaluations"]["inventory_overwrite"]["overall"][
                    "mean_discounted_return"
                ]
                for row in rows
            ]
        )
    )
    endpoint_checks = {
        "fixed_success": _close(grid[0]["existing_success_rate"], fixed_success),
        "fixed_return": _close(
            grid[0]["existing_expected_discounted_return"], fixed_return
        ),
        "p1_success": _close(grid[-1]["existing_success_rate"], harmful_success),
        "p1_return": _close(
            grid[-1]["existing_expected_discounted_return"], harmful_return
        ),
    }
    if not all(endpoint_checks.values()):
        raise AssertionError(f"aggregate endpoint failed for {policy}")
    lengths = [episode["fixed_length"] for episode in profile["episodes"]]
    return {
        "policy": policy,
        "episode_count": profile["episode_count"],
        "risk_count_distribution": profile["risk_count_distribution"],
        "success_polynomial_by_survival_power": profile[
            "success_polynomial_by_survival_power"
        ],
        "return_polynomial_by_survival_power": profile[
            "return_polynomial_by_survival_power"
        ],
        "fixed_path_mean_length": float(np.mean(lengths)),
        "fixed_path_length_range": [int(min(lengths)), int(max(lengths))],
        "reference_mean_discounted_return": profile[
            "reference_mean_discounted_return"
        ],
        "reference_crossover": find_return_crossover(profile),
        "regime_map": grid,
        "endpoint_checks": endpoint_checks,
    }


def _probability_comparison(families: dict, probability: float) -> dict:
    points = {
        policy: evaluate_regime_point(profile, probability)
        for policy, profile in families.items()
    }
    returns = {
        policy: point["existing_expected_discounted_return"]
        for policy, point in points.items()
    }
    reference = next(iter(points.values()))["reference_expected_discounted_return"]
    returns["zero_loss_reference"] = reference
    ordered = sorted(returns, key=returns.get, reverse=True)
    pairwise = {
        f"{left}_minus_{right}": returns[left] - returns[right]
        for left in returns
        for right in returns
        if left < right
    }
    return {
        "overwrite_probability": probability,
        "learned_policy_points": points,
        "reference_expected_discounted_return": reference,
        "expected_return_ranking": ordered,
        "expected_returns": returns,
        "pairwise_return_differences": pairwise,
    }


def main() -> None:
    started = time.time()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(RUN_ROOT / "capacity_start.json", _capacity())
    source_before = _source_manifest()
    if source_before["git_status"]:
        raise RuntimeError("policy-panel run requires a clean committed worktree")
    _write_json(RUN_ROOT / "run_manifest.json", source_before)
    prerequisites = _verify_prerequisites()
    _write_json(RUN_ROOT / "prerequisites.json", prerequisites)

    reference = validate_reference_paths()
    pinned_reference = json.loads(
        Path(SPEC["prerequisites"]["reference_paths"]).read_text(encoding="utf-8")
    )
    if reference != pinned_reference:
        raise AssertionError("fresh reference replay differs from pinned artifact")
    _write_json(RUN_ROOT / "reference_paths.json", reference)

    rows_by_policy = {}
    profiles_by_policy = {}
    aggregates = {}
    for policy in POLICY_NAMES:
        rows = []
        profiles = []
        for seed in SPEC["seeds"]:
            row, profile = _evaluate_policy_seed(
                policy, int(seed), reference
            )
            rows.append(row)
            profiles.append(profile)
            _write_json(RUN_ROOT / policy / f"seed{seed}.json", row)
            print(
                json.dumps(
                    {
                        "policy": policy,
                        "seed": seed,
                        "risk_count_distribution": profile[
                            "risk_count_distribution"
                        ],
                        "reference_crossover": row["reference_crossover"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            _FROZEN_EVAL_CACHE.clear()
            jax.clear_caches()
        rows_by_policy[policy] = rows
        profiles_by_policy[policy] = combine_risk_profiles(profiles)
        aggregates[policy] = _aggregate_family(policy, rows, profiles)

    comparisons = [
        _probability_comparison(
            profiles_by_policy,
            float(probability),
        )
        for probability in SPEC["exact_computation"][
            "learning_probabilities_of_interest"
        ]
    ]
    _write_json(
        RUN_ROOT / "risk_profiles.json",
        {
            policy: profiles_by_policy[policy]
            for policy in POLICY_NAMES
        },
    )

    source_after = _source_manifest()
    if source_after["source_sha256"] != source_before["source_sha256"]:
        raise AssertionError("execution source changed during policy-panel run")
    if source_after["checkpoint_sha256"] != source_before["checkpoint_sha256"]:
        raise AssertionError("input checkpoint changed during policy-panel run")
    gates = {
        "prerequisites": prerequisites["passed"],
        "reference_replay": bool(reference["passed"]),
        "all_fixed_starts_succeed": all(
            row["endpoint_checks"]["all_fixed_starts_succeed"]
            for rows in rows_by_policy.values()
            for row in rows
        ),
        "per_seed_formula_endpoints": all(
            all(row["endpoint_checks"].values())
            for rows in rows_by_policy.values()
            for row in rows
        ),
        "family_formula_endpoints": all(
            all(aggregate["endpoint_checks"].values())
            for aggregate in aggregates.values()
        ),
        "source_and_checkpoint_integrity": True,
    }
    result = {
        "schema_version": "hackrl_pack_inventory_overwrite_policy_panel_result_v1",
        "status": (
            "all_policy_panel_gates_passed_training_not_started"
            if all(gates.values())
            else "policy_panel_gate_failed_training_prohibited"
        ),
        "gates": gates,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "rows": rows_by_policy,
        "learned_policy_endpoint_episodes": 3 * 5 * 2 * 32,
        "reference_episodes": len(reference["episodes"]),
        "optimizer_updates": 0,
        "collector_slots": 0,
        "checkpoint_writes": 0,
        "training_started": False,
        "source_hashes_unchanged": True,
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
    print(json.dumps(comparisons, sort_keys=True), flush=True)


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
