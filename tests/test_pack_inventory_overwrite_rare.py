import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from hackrl.pack_inventory_overwrite_rare import (
    evaluate_polynomial,
    find_return_crossover,
)
from hackrl.pack_restore import (
    PackRestoreAction,
    PackRestorePhase,
    PackRestoreSplit,
    PackRestoreStart,
    PackRestoreVariant,
    make_pack_restore_state,
    pack_restore_step,
    pack_restore_step_inventory_overwrite_event,
    pack_restore_step_inventory_overwrite_stochastic,
    physical_grain_total,
)


def _ready(carried=2, remaining=2):
    state = make_pack_restore_state(
        0,
        int(PackRestorePhase.EMPTY),
        split=PackRestoreSplit.VALIDATION,
        start=PackRestoreStart.PATH_CHECK,
        source_growth_period=8,
    )
    state = pack_restore_step(
        state, int(PackRestoreAction.RIGHT), PackRestoreVariant.FIXED
    )
    return state.replace(
        carried_grain=jnp.asarray(carried, dtype=jnp.int32),
        source_grain=jnp.asarray(remaining, dtype=jnp.int32),
        source_age=jnp.asarray(8, dtype=jnp.int32),
    )


def _equal(left, right):
    return all(
        np.array_equal(np.asarray(getattr(left, name)), np.asarray(getattr(right, name)))
        for name in left.__dataclass_fields__
    )


def test_forced_overwrite_event_selects_only_the_declared_transition():
    state = _ready(carried=2)
    action = int(PackRestoreAction.DO)
    fixed = pack_restore_step(state, action, PackRestoreVariant.FIXED)
    harmful = pack_restore_step(
        state, action, PackRestoreVariant.INVENTORY_OVERWRITE
    )
    forced_false, false_fired = pack_restore_step_inventory_overwrite_event(
        state, action, False
    )
    forced_true, true_fired = pack_restore_step_inventory_overwrite_event(
        state, action, True
    )
    assert not bool(false_fired)
    assert bool(true_fired)
    assert _equal(forced_false, fixed)
    assert _equal(forced_true, harmful)
    assert int(physical_grain_total(state)) - int(physical_grain_total(forced_true)) == 2

    outside = _ready(carried=0)
    outside_fixed = pack_restore_step(outside, action, PackRestoreVariant.FIXED)
    outside_true, outside_fired = pack_restore_step_inventory_overwrite_event(
        outside, action, True
    )
    assert not bool(outside_fired)
    assert _equal(outside_true, outside_fixed)


def test_stochastic_overwrite_has_exact_endpoints_and_same_key_reproducibility():
    state = _ready(carried=2)
    action = int(PackRestoreAction.DO)
    key = jax.random.PRNGKey(19)
    fixed = pack_restore_step(state, action, PackRestoreVariant.FIXED)
    harmful = pack_restore_step(
        state, action, PackRestoreVariant.INVENTORY_OVERWRITE
    )
    p0, p0_fired = pack_restore_step_inventory_overwrite_stochastic(
        state, action, key, 0.0
    )
    p1, p1_fired = pack_restore_step_inventory_overwrite_stochastic(
        state, action, key, 1.0
    )
    first, first_fired = pack_restore_step_inventory_overwrite_stochastic(
        state, action, key, 0.37
    )
    second, second_fired = pack_restore_step_inventory_overwrite_stochastic(
        state, action, key, 0.37
    )
    assert not bool(p0_fired)
    assert bool(p1_fired)
    assert _equal(p0, fixed)
    assert _equal(p1, harmful)
    assert bool(first_fired) == bool(second_fired)
    assert _equal(first, second)


def test_exact_polynomial_and_crossover_are_computed_without_sampling():
    coefficients = {"0": 0.25, "1": 0.50, "2": 0.25}
    assert evaluate_polynomial(coefficients, 0.0) == 1.0
    assert evaluate_polynomial(coefficients, 1.0) == 0.25
    assert evaluate_polynomial(coefficients, 0.5) == 0.5625

    profile = {
        "return_polynomial_by_survival_power": {"1": 1.0},
        "reference_mean_discounted_return": 0.5,
    }
    crossing = find_return_crossover(profile)
    assert crossing["bracketed"]
    assert abs(crossing["probability"] - 0.5) <= 1e-12
    assert crossing["interval_width"] <= 1e-12


def test_rare_manifest_freezes_no_learning_panel_and_probability_grid():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads(
        (root / "docs/manifests/pack_inventory_overwrite_rare_v1.json").read_text()
    )
    assert spec["source_policies"]["seeds"] == [30, 31, 32, 33, 34]
    assert spec["exact_computation"]["report_probabilities"] == [
        0.0,
        0.001,
        0.0025,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
        1.0,
    ]
    assert spec["execution_budget"]["optimizer_updates"] == 0
    assert spec["execution_budget"]["collector_slots"] == 0
    assert spec["execution_budget"]["checkpoint_writes"] == 0
    assert spec["deferred_training"]["status"] == (
        "not_started_even_if_all_gates_pass"
    )
