import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "evaluate_dual_teacher_greedy",
        REPOSITORY / "scripts" / "evaluate_dual_teacher_greedy.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load()


def test_teacher_greedy_job_matrix_covers_both_defects_and_saved_stages():
    jobs = EVALUATOR.build_evaluations()
    assert len(jobs) == 30
    assert {job["env"] for job in jobs} == {"tick", "pack"}
    assert {job["seed"] for job in jobs} == set(range(20, 25))
    assert {job["stage"] for job in jobs} == set(EVALUATOR.STAGES)
    assert all(job["checkpoint"].name in {"update_512", "adapt_4096"} for job in jobs)


class _Teacher:
    def apply(self, variables, maps, numeric, *, train):
        assert train is False
        batch = maps.shape[0]
        values = jnp.zeros((batch, 3, 4), dtype=jnp.float32)
        values = values.at[:, 2, 1].set(5.0)
        return values


def test_teacher_policy_acts_from_commanded_goal_head():
    policy = EVALUATOR.TeacherGreedyPolicy(_Teacher())
    distribution, value = policy.apply(
        {"params": {}, "batch_stats": {}},
        jnp.zeros((2, 3, 3, 1)),
        jnp.zeros((2, 1)),
        jnp.asarray([[0, 0, 1], [0, 0, 1]], dtype=jnp.float32),
    )
    np.testing.assert_array_equal(jnp.argmax(distribution.logits, axis=-1), [1, 1])
    np.testing.assert_array_equal(value, [0.0, 0.0])


def test_teacher_aggregate_preserves_each_rows_actual_seed():
    metric = {
        "success_rate": 1.0,
        "violation_delivery_rate": 0.0,
        "mean_discounted_return": 0.5,
    }
    rows = [
        {
            "env": "tick",
            "stage": "pretrain",
            "seed": seed,
            "kernels": {
                kernel: {"natural_reset": metric}
                for kernel in EVALUATOR.KERNELS
            },
        }
        for seed in (24, 20)
    ]
    aggregate = EVALUATOR._aggregate(rows)
    assert all(
        [point["seed"] for point in group["seed_points"]] == [24, 20]
        for group in aggregate
    )
