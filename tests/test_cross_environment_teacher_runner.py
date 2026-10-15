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
import run_cross_environment_teacher_evaluation as evaluation_runner  # noqa: E402


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
    assert '"diff", "--cached", "--quiet"' in text


def test_all_ten_source_checkpoints_are_bound_to_frozen_evidence() -> None:
    matrix = load_matrix()
    jobs = build_jobs(matrix)
    contract = matrix_runner._source_checkpoint_contract(matrix, jobs)
    assert len(contract) == 10
    sample = Path(jobs[0]["pretrain_checkpoint"]).resolve()
    assert contract[str(sample)] == engine._checkpoint_fingerprint(sample)["sha256"]


def test_common_evaluation_inventory_is_frozen_to_180_policies_and_1440_tasks(tmp_path: Path) -> None:
    policies = evaluation_runner.build_policy_inventory(load_matrix(), tmp_path)
    assert len(policies) == 180
    assert Counter(policy["environment"] for policy in policies) == {
        "tick_claim": 45, "pack_restore": 90, "craft_remain": 45
    }
    tasks = evaluation_runner.build_tasks(policies)
    assert len(tasks) == 1440
    assert len({task["id"] for task in tasks}) == 1440
    text = (ROOT / "scripts/run_cross_environment_teacher_evaluation.py").read_text()
    assert "do not auto re-evaluate" in text
    assert "automatic_re_evaluation" in text
    assert '"diff", "--cached", "--quiet"' in text


def test_smoke_and_main_roots_are_distinct_and_main_manifest_records_budget(tmp_path: Path) -> None:
    assert matrix_runner.DEFAULT_SMOKE_RUN_ROOT != matrix_runner.DEFAULT_MAIN_RUN_ROOT
    matrix = load_matrix()
    jobs = build_jobs(matrix)
    provenance = {
        "git_commit": "a" * 40,
        "scientific_source_sha256": {"frozen": "b" * 64},
    }
    receipt = matrix_runner._ensure_main_manifest(matrix, tmp_path, jobs, provenance)
    assert receipt["main_jobs"] == 50
    assert receipt["updates_per_job"] == 4096
    assert receipt["planned_main_collection_slots"] == 6_710_886_400
    assert receipt["completed_smoke_collection_slots"] == 163_840
    assert receipt["planned_total_including_smoke"] == 6_711_050_240
    assert receipt == matrix_runner._ensure_main_manifest(matrix, tmp_path, jobs, provenance)


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
