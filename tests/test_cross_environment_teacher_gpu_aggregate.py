import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "aggregate_cross_environment_teacher_gpu_evaluation.py"
SPEC = importlib.util.spec_from_file_location("teacher_aggregate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(condition, trained_variant, seed, exploit, success=1.0):
    return {
        "environment": "tick_claim",
        "condition": condition,
        "trained_variant": trained_variant,
        "checkpoint_origin": "new_matrix_result",
        "seed": seed,
        "kernel": "mutant",
        "policy_view": "sample",
        "family": "natural_reset",
        "success_rate": success,
        "exploit_rate": exploit,
        "trigger_rate": exploit,
        "mean_discounted_return": exploit,
    }


def test_paired_effect_and_condition_contrast_are_seed_paired():
    rows = []
    for seed, task_fixed, task_mutant, frozen_fixed, frozen_mutant in (
        (20, 0.1, 0.9, 0.2, 0.5),
        (21, 0.0, 0.6, 0.1, 0.3),
    ):
        rows.extend(
            [
                _row("D-task", "fixed", seed, task_fixed),
                _row("D-task", "mutant", seed, task_mutant),
                _row("D-frozen", "fixed", seed, frozen_fixed),
                _row("D-frozen", "mutant", seed, frozen_mutant),
            ]
        )
    effects = MODULE._paired_adaptation_effects(rows)
    task = next(row for row in effects if row["condition"] == "D-task")
    frozen = next(row for row in effects if row["condition"] == "D-frozen")
    assert task["metrics"]["exploit_rate"]["mean"] == 0.7
    assert frozen["metrics"]["exploit_rate"]["mean"] == 0.25
    contrast = MODULE._condition_contrasts(effects)
    row = next(
        row
        for row in contrast
        if row["left_condition"] == "D-task"
        and row["right_condition"] == "D-frozen"
    )
    assert row["metrics"]["exploit_rate"]["mean"] == 0.45
    assert row["evidence"]["class"] == "new_same_code_and_initialization"


def test_group_summary_rejects_duplicate_seed_cells():
    row = _row("D-task", "fixed", 20, 0.0)
    try:
        MODULE._group_summary([row, row.copy()])
    except RuntimeError as error:
        assert "duplicate seed" in str(error)
    else:
        raise AssertionError("duplicate seed was accepted")

def test_manuscript_section_names_the_estimand_and_replication_unit():
    result = MODULE.aggregate(MODULE.DEFAULT_ROOT)
    report = MODULE.render_report(result)
    assert "직접 활용률 차이가 아니라 seed별 `ΔU_s" in report
    assert "(64.61−0.31)−(5.00−0.00)=59.30pp" in report
    assert "TICK-CLAIM | D-task − D-frozen | +11.56pp | +25.16pp" in report
    assert "CRAFT-REMAIN | D-task − D-frozen | +60.00pp | +60.00pp" in report
    assert "PACK-RESTORE | D-on − D-off | +59.30pp | +59.06pp" in report
    assert "88,200개 평가 episode" in report
    assert "독립 learner 수를 늘리지 않는다" in report
