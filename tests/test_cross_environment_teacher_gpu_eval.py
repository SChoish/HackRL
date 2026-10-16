from __future__ import annotations

import json
from pathlib import Path

import pytest

import eval_cross_environment_teacher_traces_gpu as gpu_eval
import run_cross_environment_teacher_evaluation_gpu as gpu_runner
from hackrl.cross_environment_teacher_matrix import load_matrix
from hackrl.cross_environment_teacher_trace import public_record


def test_public_at_matches_frozen_public_record() -> None:
    state, _ = gpu_eval._stack_initial("pack_restore", "natural_reset", 1)
    first = gpu_eval._public_at(state, 0)
    expected = public_record(
        gpu_eval.jax.tree.map(lambda value: value[0], state)
    )
    assert first == expected


def test_scientific_evaluation_requires_policy_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="policy_id"):
        gpu_eval.evaluate(
            environment="craft_remain",
            checkpoint=tmp_path / "not-read",
            checkpoint_origin="reused_result",
            learner_seed=20,
            variant="fixed",
            view="mode",
            family="natural_reset",
            output_root=tmp_path,
            scientific=True,
        )


def test_four_policy_shards_are_disjoint_and_complete() -> None:
    matrix = load_matrix(gpu_runner.DEFAULT_MATRIX)
    policies = gpu_runner.build_policy_inventory(
        matrix, gpu_runner.DEFAULT_MAIN_RUN_ROOT
    )
    tasks = gpu_runner.build_tasks(policies)
    shards = [
        {policy["id"] for index, policy in enumerate(policies) if index % 4 == shard}
        for shard in range(4)
    ]
    assert all(len(ids) == 45 for ids in shards)
    assert set.union(*shards) == {policy["id"] for policy in policies}
    assert sum(len(ids) for ids in shards) == len(set.union(*shards))
    assert len(tasks) == 1440


def test_validation_receipt_matches_current_evaluators() -> None:
    receipt = gpu_runner._assert_validation_receipt()
    assert Path(receipt["path"]).is_file()
    payload = json.loads(Path(receipt["path"]).read_text())
    assert payload["passed"] is True
    assert all(
        row["actual_defect_trigger_observed"]
        for row in payload["environments"].values()
    )
