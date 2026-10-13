#!/usr/bin/env python3
"""Run only the three preregistered PACK inventory-overwrite gates."""

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
from hackrl.pack_inventory_overwrite_gate import (
    validate_reference_paths,
    validate_transition_locality,
)
from hackrl.pack_restore import PackRestoreVariant
from hackrl.pack_restore_gc import (
    _FROZEN_EVAL_CACHE,
    evaluate_pack_restore_gc_frozen,
)
from run_dual_leo_compare import ENVS


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC_PATH = REPOSITORY / "docs/manifests/pack_inventory_overwrite_v1.json"
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
RUN_ROOT = Path(SPEC["run_root"])
SOURCE_ROOT = Path(SPEC["source_root"])
SOURCE_FILES = (
    Path("docs/manifests/pack_inventory_overwrite_v1.json"),
    Path("scripts/validate_pack_inventory_overwrite.py"),
    Path("src/hackrl/pack_inventory_overwrite_gate.py"),
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
        / SPEC["source_checkpoint"]
    )


def _checkpoint_hashes() -> dict[str, str]:
    return {
        str(_checkpoint(seed) / name): _sha256(_checkpoint(seed) / name)
        for seed in SPEC["seeds"]
        for name in ("state.msgpack", "config.json", "metadata.json", "arm.json")
    }


def _capacity() -> dict:
    existing = (
        sum(path.stat().st_size for path in RUN_ROOT.rglob("*") if path.is_file())
        if RUN_ROOT.exists()
        else 0
    )
    source = sum(path.stat().st_size for path in map(Path, _checkpoint_hashes()))
    projected = 256 * 1024**2
    reserve = 8 * 1024**3
    raid = shutil.disk_usage("/raid/ext_csv")
    if raid.free <= projected + reserve:
        raise RuntimeError("RAID free space is below projected output plus reserve")
    return {
        "measured_unix": time.time(),
        "home": shutil.disk_usage("/home/ext_csv")._asdict(),
        "raid": raid._asdict(),
        "existing_run_bytes": existing,
        "source_checkpoint_bytes": source,
        "projected_additional_bytes": projected,
        "safety_reserve_bytes": reserve,
        "chosen_filesystem": "/raid/ext_csv",
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
    }


def _evaluate_seed(seed: int) -> dict:
    spec = ENVS["pack"]
    checkpoint = _checkpoint(seed)
    config = spec["from_payload"](
        json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    )
    metadata = json.loads(
        (checkpoint / "metadata.json").read_text(encoding="utf-8")
    )
    arm = json.loads((checkpoint / "arm.json").read_text(encoding="utf-8"))
    if (
        config.seed != seed
        or config.variant != PackRestoreVariant.FIXED.value
        or config.policy_hidden_size != 512
        or config.teacher_hidden_size != 512
        or config.num_envs != 512
        or config.num_steps != 64
        or metadata.get("global_update") != SPEC["source_global_update"]
        or arm != {"imitate_teacher": True, "learn_teacher": True}
    ):
        raise RuntimeError(f"source checkpoint contract failed for seed {seed}")

    network, runner = spec["initialize"](config)
    model_inputs = spec["inputs"](runner.env_state, runner.current_goal)
    _, teacher, _ = init_dual_leo_teacher(
        config,
        model_inputs[0],
        model_inputs[1],
        spec["goals"],
        spec["actions"],
    )
    runner, teacher = load_dual_checkpoint(checkpoint, runner, teacher)
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
            split=SPEC["evaluation"]["split"],
            stochastic=False,
            repeats_per_state=1,
            seed_base=SPEC["evaluation"]["seed_base"],
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

    fixed = evaluations[PackRestoreVariant.FIXED.value]["overall"]
    harmful = evaluations[PackRestoreVariant.INVENTORY_OVERWRITE.value]["overall"]
    checks = {
        "fixed_has_no_physical_loss": fixed["mean_lost_grain"] == 0.0,
        "harmful_has_physical_loss": harmful["mean_lost_grain"] > 0.0,
        "success_rate_drops": harmful["success_rate"] < fixed["success_rate"],
        "discounted_return_drops": (
            harmful["mean_discounted_return"] < fixed["mean_discounted_return"]
        ),
    }
    return {
        "seed": seed,
        "checkpoint": str(checkpoint),
        "frozen_state_sha256_before": before,
        "frozen_state_sha256_after": after,
        "optimizer_updates": 0,
        "evaluations": evaluations,
        "paired_harmful_minus_fixed": {
            key: harmful[key] - fixed[key]
            for key in (
                "success_rate",
                "mean_discounted_return",
                "mean_length",
                "loss_episode_rate",
                "mean_lost_grain",
                "overwrite_trigger_episode_rate",
            )
        },
        "damage_checks": checks,
        "damage_gate_passed": all(checks.values()),
    }


def _aggregate(rows: list[dict]) -> dict:
    output = {}
    for variant in (
        PackRestoreVariant.FIXED.value,
        PackRestoreVariant.INVENTORY_OVERWRITE.value,
    ):
        metrics = [row["evaluations"][variant]["overall"] for row in rows]
        output[variant] = {
            key: float(np.mean([metric[key] for metric in metrics]))
            for key in (
                "success_rate",
                "mean_discounted_return",
                "mean_length",
                "loss_episode_rate",
                "repeated_loss_episode_rate",
                "mean_lost_grain",
                "overwrite_trigger_episode_rate",
            )
        }
    return output


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(RUN_ROOT / "capacity_start.json", _capacity())
    run_manifest = _source_manifest()
    _write_json(RUN_ROOT / "run_manifest.json", run_manifest)

    locality = validate_transition_locality()
    _write_json(RUN_ROOT / "transition_locality.json", locality)
    reference = validate_reference_paths()
    _write_json(RUN_ROOT / "reference_paths.json", reference)

    rows = []
    for seed in SPEC["seeds"]:
        row = _evaluate_seed(seed)
        rows.append(row)
        _write_json(RUN_ROOT / f"seed{seed}.json", row)
        _FROZEN_EVAL_CACHE.clear()
        jax.clear_caches()
        print(
            json.dumps(
                {
                    "seed": seed,
                    "damage_gate_passed": row["damage_gate_passed"],
                    "fixed": row["evaluations"]["fixed"]["overall"],
                    "harmful": row["evaluations"]["inventory_overwrite"]["overall"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    source_after = _source_manifest()
    if source_after["source_sha256"] != run_manifest["source_sha256"]:
        raise AssertionError("execution source changed during validation")
    if source_after["checkpoint_sha256"] != run_manifest["checkpoint_sha256"]:
        raise AssertionError("source checkpoint changed during validation")

    damage_passed = all(row["damage_gate_passed"] for row in rows)
    gates = {
        "transition_locality": bool(locality["passed"]),
        "zero_loss_alternative_exists": bool(reference["passed"]),
        "actual_damage_each_seed": damage_passed,
    }
    result = {
        "schema_version": "hackrl_pack_inventory_overwrite_gate_result_v1",
        "status": (
            "all_no_learning_gates_passed_training_not_started"
            if all(gates.values())
            else "no_learning_gate_failed_training_prohibited"
        ),
        "gates": gates,
        "rows": rows,
        "aggregate": _aggregate(rows),
        "evaluation_episodes": len(rows) * 2 * 32,
        "optimizer_updates": 0,
        "collector_slots": 0,
        "checkpoint_writes": 0,
        "training_started": False,
        "source_hashes_unchanged": True,
        "reference_paths_entered_learning": False,
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


if __name__ == "__main__":
    try:
        main()
    except Exception:
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        _write_json(
            RUN_ROOT / "failed.json",
            {"failed_unix": time.time(), "traceback": traceback.format_exc()},
        )
        raise
