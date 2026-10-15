from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from hackrl.craft_remain_env import GOAL_IDS as CRAFT_GOALS
from hackrl.cross_environment_teacher_matrix import build_jobs, load_matrix
from hackrl.tick_claim import GOAL_IDS as TICK_GOALS


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_dual_leo_compare as engine  # noqa: E402
import run_cross_environment_teacher_matrix as matrix_runner  # noqa: E402


def test_builder_is_exactly_the_frozen_50_job_matrix() -> None:
    jobs = build_jobs()
    assert len(jobs) == 50
    assert Counter((job["environment"], job["condition"]) for job in jobs) == {
        ("tick_claim", "D-task"): 10,
        ("tick_claim", "D-frozen"): 10,
        ("craft_remain", "D-task"): 10,
        ("craft_remain", "D-frozen"): 10,
        ("craft_remain", "D-off"): 10,
    }
    assert not any(job["environment"] == "pack_restore" for job in jobs)
    assert all(job["kind"] == "adapt" and job["depends_on"] == [] for job in jobs)


def test_environment_specific_task_goal_catalogs_drive_D_task() -> None:
    matrix = load_matrix()
    catalogs = {"tick_claim": TICK_GOALS, "craft_remain": CRAFT_GOALS}
    for job in build_jobs(matrix):
        if job["condition"] != "D-task":
            continue
        mapping = matrix["shared_learning_contract"]["task_goal_mapping"][job["environment"]]
        assert job["teacher_goal_indices"] == [catalogs[job["environment"]].index(mapping["goal_id"])]
        assert job["teacher_goal_indices"] == [mapping["goal_index"]]


def test_condition_arms_encode_real_update_toggles() -> None:
    examples = {job["condition"]: job for job in build_jobs()}
    assert engine._arm(examples["D-task"]) == {
        "learn_teacher": True,
        "imitate_teacher": True,
        "teacher_goal_indices": [11],
    }
    assert engine._arm(examples["D-frozen"]) == {
        "learn_teacher": False,
        "imitate_teacher": True,
    }
    assert engine._arm(examples["D-off"]) == {
        "learn_teacher": True,
        "imitate_teacher": False,
    }


def test_adapt_one_is_a_resumable_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints/adapt_1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.msgpack").write_bytes(b"state")
    (checkpoint / "metadata.json").write_text(json.dumps({"global_update": 513}))
    assert engine._latest_adapt(tmp_path) == 1


def test_runner_has_no_queue_mode_and_requires_explicit_main_authorization() -> None:
    text = (ROOT / "scripts/run_cross_environment_teacher_matrix.py").read_text()
    assert 'group.add_argument("--run-job")' in text
    assert 'parser.add_argument("--authorize-main-training"' in text
    assert '"--worker"' not in text


def test_update_zero_curve_is_reused_with_explicit_provenance(tmp_path: Path) -> None:
    job = next(
        item for item in build_jobs()
        if item["environment"] == "tick_claim"
        and item["condition"] == "D-task"
        and item["seed"] == 20
        and item["variant"] == "fixed"
    )
    provenance = matrix_runner._reuse_update_zero_curve(job, tmp_path)
    cell = engine._cell_dir(tmp_path, job)
    assert provenance["origin"] == "reused_result"
    assert provenance["new_evaluation_episodes"] == 0
    assert (cell / "curve/adapt_0.json").is_file()
    assert (cell / "curve/adapt_0.provenance.json").is_file()
    assert provenance == matrix_runner._reuse_update_zero_curve(job, tmp_path)
