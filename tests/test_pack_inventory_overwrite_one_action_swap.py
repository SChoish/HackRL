import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]


def _load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load_script(
    "evaluate_pack_inventory_overwrite_one_action_swap_test",
    "scripts/evaluate_pack_inventory_overwrite_one_action_swap.py",
)
MANIFEST = json.loads(EVALUATOR.SPEC_PATH.read_text(encoding="utf-8"))


def test_manifest_locks_single_no_learning_mode_and_160_paths():
    assert MANIFEST["status"] == "frozen_before_execution"
    assert MANIFEST["budget"]["episodes"] == 160
    assert MANIFEST["budget"]["maximum_environment_steps"] == 160 * 128
    assert MANIFEST["budget"]["learning_updates"] == 0
    assert MANIFEST["budget"]["new_execution_modes"] == 1
    assert MANIFEST["stop_rule"].startswith("Run D for all five seeds")


def test_manifest_pins_the_four_prior_b_failure_c_success_cases():
    assert MANIFEST["diagnostic_panel"]["expected_count"] == 4
    assert MANIFEST["diagnostic_panel"]["expected_seed_state_indices"] == [
        [35, 4],
        [35, 12],
        [35, 16],
        [35, 24],
    ]
    b_raw, c_raw = EVALUATOR.pinned_inputs()
    assert EVALUATOR.diagnostic_keys(b_raw, c_raw) == [
        (35, 4),
        (35, 12),
        (35, 16),
        (35, 24),
    ]


def test_choose_one_action_swap_state_machine():
    source = jnp.asarray([3, 3, 3, 3, 3, 3])
    adapted = jnp.asarray([23, 23, 23, 3, 23, 23])
    carried = jnp.asarray([0, 1, 1, 0, 0, 0])
    active = jnp.asarray([True, True, True, True, True, True])
    phase = jnp.asarray([0, 0, 1, 1, 1, 2], dtype=jnp.int32)

    chosen, use_adapted, handoff, swap, next_phase = EVALUATOR.choose_one_action_swap(
        source, adapted, carried, active, phase
    )

    np.testing.assert_array_equal(np.asarray(chosen), [3, 23, 23, 3, 3, 23])
    np.testing.assert_array_equal(
        np.asarray(use_adapted), [False, True, True, True, False, True]
    )
    np.testing.assert_array_equal(
        np.asarray(handoff), [False, True, False, False, False, False]
    )
    np.testing.assert_array_equal(
        np.asarray(swap), [False, False, False, False, True, False]
    )
    np.testing.assert_array_equal(np.asarray(next_phase), [0, 1, 1, 1, 2, 2])


def test_inactive_path_cannot_handoff_or_swap():
    chosen, use_adapted, handoff, swap, next_phase = EVALUATOR.choose_one_action_swap(
        jnp.asarray([3]),
        jnp.asarray([23]),
        jnp.asarray([0]),
        jnp.asarray([False]),
        jnp.asarray([1], dtype=jnp.int32),
    )
    np.testing.assert_array_equal(np.asarray(chosen), [3])
    np.testing.assert_array_equal(np.asarray(use_adapted), [False])
    np.testing.assert_array_equal(np.asarray(handoff), [False])
    np.testing.assert_array_equal(np.asarray(swap), [False])
    np.testing.assert_array_equal(np.asarray(next_phase), [1])


def _step(chosen, source=3, adapted=23, carried=0):
    return {
        "position": [9, 8],
        "carried_grain": carried,
        "source_grain": 2,
        "delivered_total": 1,
        "source_action": source,
        "adapted_action": adapted,
        "chosen_action": chosen,
        "used_adapted": chosen == adapted,
        "switch_event": False,
        "switch_exit": False,
        "source_risk_proposal": False,
        "executed_risky_harvest": False,
        "overwrite_fired": False,
        "lost_grain": 0,
        "step": 1,
    }


def test_first_common_disagreement_is_first_empty_action_difference():
    common = _step(1, source=1, adapted=1, carried=1)
    b = {"steps": [common, _step(23)]}
    c = {"steps": [common, _step(3)]}
    assert EVALUATOR.first_common_disagreement(b, c) == 2


def test_decorate_episode_records_prebranch_risk():
    first = _step(23, source=23, adapted=23, carried=1)
    first["executed_risky_harvest"] = True
    branch = _step(3)
    branch["step"] = 2
    branch["switch_exit"] = True
    episode = {"steps": [first, branch]}
    decorated = EVALUATOR.decorate_episode(episode)
    assert decorated["one_action_swap_count"] == 1
    assert decorated["one_action_swap_step"] == 2
    assert decorated["risk_count_before_swap"] == 1
