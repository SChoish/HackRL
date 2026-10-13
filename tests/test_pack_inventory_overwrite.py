import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from hackrl.pack_inventory_overwrite_gate import (
    independent_trigger_oracle,
    validate_reference_paths,
    validate_transition_locality,
)
from hackrl.pack_restore import (
    PackRestoreAction,
    PackRestorePhase,
    PackRestoreSplit,
    PackRestoreStart,
    PackRestoreVariant,
    inventory_overwrite_trigger,
    make_pack_restore_state,
    pack_restore_step,
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


def _changed_fields(left, right):
    return {
        name
        for name in left.__dataclass_fields__
        if not np.array_equal(np.asarray(getattr(left, name)), np.asarray(getattr(right, name)))
    }


def test_inventory_overwrite_trigger_loses_exactly_preharvest_inventory():
    for carried in (1, 2, 3):
        before = _ready(carried=carried)
        action = int(PackRestoreAction.DO)
        fixed = pack_restore_step(before, action, PackRestoreVariant.FIXED)
        harmful = pack_restore_step(
            before, action, PackRestoreVariant.INVENTORY_OVERWRITE
        )
        assert independent_trigger_oracle(before, action)
        assert bool(inventory_overwrite_trigger(before, action))
        assert _changed_fields(fixed, harmful) == {"carried_grain"}
        assert int(fixed.carried_grain) == carried + 1
        assert int(harmful.carried_grain) == 1
        assert int(fixed.source_grain) == int(before.source_grain) - 1
        assert int(harmful.source_grain) == int(before.source_grain) - 1
        assert int(physical_grain_total(fixed)) == int(physical_grain_total(before))
        assert (
            int(physical_grain_total(before))
            - int(physical_grain_total(harmful))
            == carried
        )


def test_inventory_overwrite_matches_fixed_outside_trigger_and_on_rebuild():
    controls = (
        _ready(carried=0),
        _ready(carried=2).replace(source_age=jnp.asarray(7, dtype=jnp.int32)),
        _ready(carried=2).replace(source_grain=jnp.asarray(0, dtype=jnp.int32)),
    )
    for state in controls:
        for action in PackRestoreAction:
            fixed = pack_restore_step(state, int(action), PackRestoreVariant.FIXED)
            harmful = pack_restore_step(
                state, int(action), PackRestoreVariant.INVENTORY_OVERWRITE
            )
            if not independent_trigger_oracle(state, int(action)):
                assert not _changed_fields(fixed, harmful)

    rebuild = make_pack_restore_state(
        0,
        int(PackRestorePhase.LOADED),
        split=PackRestoreSplit.VALIDATION,
        start=PackRestoreStart.PATH_CHECK,
        source_growth_period=8,
    )
    for action in (PackRestoreAction.MAKE_RECORD, PackRestoreAction.PACK_STORAGE):
        rebuild = pack_restore_step(rebuild, int(action), PackRestoreVariant.FIXED)
    fixed = pack_restore_step(
        rebuild, int(PackRestoreAction.REBUILD_EMPTY), PackRestoreVariant.FIXED
    )
    harmful = pack_restore_step(
        rebuild,
        int(PackRestoreAction.REBUILD_EMPTY),
        PackRestoreVariant.INVENTORY_OVERWRITE,
    )
    assert not _changed_fields(fixed, harmful)
    assert int(harmful.anchor_grain) == 0


def test_locality_and_reference_gates_cover_declared_panels():
    locality = validate_transition_locality()
    assert locality["passed"]
    assert locality["triggered_transitions"] > 0
    assert locality["outside_trigger_transitions"] > locality["triggered_transitions"]

    reference = validate_reference_paths()
    assert reference["passed"]
    assert reference["natural_starts"] == 32
    assert reference["successful_starts"] == 32
    assert reference["zero_loss_starts"] == 32
    assert reference["max_path_length"] <= 128
    assert all(row["trigger_count"] == 0 for row in reference["episodes"])


def test_manifest_freezes_no_learning_scope_and_all_five_seeds():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads(
        (root / "docs/manifests/pack_inventory_overwrite_v1.json").read_text()
    )
    assert spec["seeds"] == [30, 31, 32, 33, 34]
    assert spec["kernel_contract"]["variant"] == "inventory_overwrite"
    assert spec["kernel_contract"]["defect_indicator_added"] is False
    assert spec["reference_path_firewall"] == {
        "policy_initialization": False,
        "training_data": False,
        "behavior_cloning": False,
        "reward_shaping": False,
        "action_mask": False,
        "policy_selection": False,
    }
    assert spec["execution_budget"] == {
        "optimizer_updates": 0,
        "collector_slots": 0,
        "checkpoint_writes": 0,
        "learning_branches": 0,
        "automatic_extension": False,
    }
    assert spec["deferred_training_design"]["status"] == (
        "not_started_even_if_all_gates_pass"
    )
