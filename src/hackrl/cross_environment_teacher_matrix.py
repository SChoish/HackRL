"""Frozen job construction for cross-environment teacher matrix v1.

The JSON matrix is the sole source of condition, environment, seed, and budget
membership.  This module translates that frozen design into the existing Dual
adaptation engine without inventing extra cells.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hackrl.craft_remain_env import GOAL_IDS as CRAFT_GOAL_IDS
from hackrl.pack_restore import GOAL_IDS as PACK_GOAL_IDS
from hackrl.tick_claim import GOAL_IDS as TICK_GOAL_IDS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = ROOT / "docs/manifests/cross_environment_teacher_matrix_v1.json"
ENVIRONMENT_TO_ENGINE = {
    "tick_claim": "tick",
    "craft_remain": "craft",
}
GOAL_CATALOGS = {
    "tick_claim": TICK_GOAL_IDS,
    "pack_restore": PACK_GOAL_IDS,
    "craft_remain": CRAFT_GOAL_IDS,
}
CONDITION_METHODS = {
    "D-task": "dual_teacher_task",
    "D-frozen": "dual_teacher_frozen",
    "D-off": "dual_bc_off",
}
EXPECTED_JOB_COUNTS = {
    ("tick_claim", "D-task"): 10,
    ("tick_claim", "D-frozen"): 10,
    ("craft_remain", "D-task"): 10,
    ("craft_remain", "D-frozen"): 10,
    ("craft_remain", "D-off"): 10,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_matrix(path: Path = DEFAULT_MATRIX) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_checkpoint(environment: str, seed: int) -> Path:
    if environment == "tick_claim":
        return (
            ROOT
            / "runs/dual_leo_compare_v1/tick/dual/pretrain"
            / f"seed{seed}/checkpoints/update_512"
        )
    if environment == "craft_remain":
        return (
            ROOT
            / "runs/craft_remain_compare_v1/dual/pretrain"
            / f"seed{seed}/checkpoints/update_512"
        )
    raise ValueError(f"no new learning jobs are permitted for {environment}")


def _condition_arm(condition: str, goal_index: int) -> dict:
    if condition == "D-task":
        return {
            "learn_teacher": True,
            "imitate_teacher": True,
            "teacher_goal_indices": [int(goal_index)],
        }
    if condition == "D-frozen":
        return {"learn_teacher": False, "imitate_teacher": True}
    if condition == "D-off":
        return {"learn_teacher": True, "imitate_teacher": False}
    raise ValueError(f"unsupported missing condition: {condition}")


def build_jobs(matrix: dict | None = None) -> list[dict]:
    """Build exactly the missing 50 adaptation jobs from the frozen matrix."""

    matrix = load_matrix() if matrix is None else matrix
    jobs: list[dict] = []
    task_mapping = matrix["shared_learning_contract"]["task_goal_mapping"]
    for environment, environment_spec in matrix["environment_matrix"].items():
        for condition, condition_spec in environment_spec["conditions"].items():
            requested = int(condition_spec["new_learning_jobs"])
            if requested == 0:
                continue
            if environment not in ENVIRONMENT_TO_ENGINE:
                raise ValueError(f"matrix requests an unsupported new job: {environment}")
            mapping = task_mapping[environment]
            catalog = GOAL_CATALOGS[environment]
            goal_index = catalog.index(mapping["goal_id"])
            if goal_index != int(mapping["goal_index"]):
                raise ValueError(f"goal mapping drift for {environment}")
            seeds = tuple(int(seed) for seed in environment_spec["learner_seeds"])
            cells = [(seed, variant) for seed in seeds for variant in ("fixed", "mutant")]
            if len(cells) != requested:
                raise ValueError(
                    f"{environment}/{condition} matrix requests {requested} jobs, "
                    f"but its seeds imply {len(cells)}"
                )
            for seed, variant in cells:
                jobs.append(
                    {
                        "id": (
                            f"{environment}-{condition.lower()}-s{seed}-{variant}"
                            .replace("_", "-")
                        ),
                        "kind": "adapt",
                        "environment": environment,
                        "env": ENVIRONMENT_TO_ENGINE[environment],
                        "method": CONDITION_METHODS[condition],
                        "condition": condition,
                        "seed": seed,
                        "variant": variant,
                        "depends_on": [],
                        "pretrain_checkpoint": str(source_checkpoint(environment, seed)),
                        "origin_arm": {
                            "learn_teacher": True,
                            "imitate_teacher": True,
                        },
                        "record_checkpoint_fingerprint": True,
                        "persist_source_fingerprint": False,
                        "rolling_checkpoints": True,
                        "policy_hidden_size": 512,
                        "teacher_hidden_size": 512,
                        **_condition_arm(condition, goal_index),
                    }
                )
    validate_jobs(matrix, jobs)
    return jobs


def validate_jobs(matrix: dict, jobs: list[dict]) -> None:
    budget = matrix["missing_learning_budget"]
    if len(jobs) != int(budget["total_jobs"]) or len(jobs) != 50:
        raise ValueError(f"job count is {len(jobs)}, expected 50")
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("job identifiers are not unique")
    observed: dict[tuple[str, str], int] = {}
    for job in jobs:
        key = (job["environment"], job["condition"])
        observed[key] = observed.get(key, 0) + 1
        if job["environment"] == "pack_restore":
            raise ValueError("PACK must not receive new matrix training")
        source = Path(job["pretrain_checkpoint"])
        for name in ("state.msgpack", "config.json", "metadata.json"):
            if not (source / name).is_file():
                raise FileNotFoundError(source / name)
    if observed != EXPECTED_JOB_COUNTS:
        raise ValueError(f"job matrix drift: {observed}")
    slots = len(jobs) * int(budget["collector_slots_per_job"])
    if slots != int(budget["total_collector_slots"]) or slots != 6_710_886_400:
        raise ValueError(f"collection budget drift: {slots}")


def job_by_id(job_id: str, matrix: dict | None = None) -> dict:
    matches = [job for job in build_jobs(matrix) if job["id"] == job_id]
    if len(matches) != 1:
        raise KeyError(job_id)
    return matches[0]
