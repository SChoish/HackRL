from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from hackrl.craft_remain_env import (
    CraftRemainAction,
    CraftRemainPhase,
    CraftRemainStart,
    make_craft_remain_state,
)
from hackrl.cross_environment_teacher_trace import (
    action_has_defect_effect,
    actual_step,
    audit_actual_step,
    build_panel_manifest,
    initial_episode_oracle,
    legal_trigger_actions,
    legacy_opportunity_reached,
)
from hackrl.pack_restore import (
    PackRestoreAction,
    PackRestorePhase,
    PackRestoreSplit,
    PackRestoreStart,
    make_pack_restore_state,
)
from hackrl.tick_claim import (
    TickClaimAction,
    TickClaimPhase,
    TickClaimSplit,
    TickClaimStart,
    make_tick_claim_state,
)


ROOT = Path(__file__).resolve().parents[1]


def _tick_trigger_state():
    state = make_tick_claim_state(
        0,
        TickClaimPhase.RIPE,
        split=TickClaimSplit.VALIDATION,
        start=TickClaimStart.PATH_CHECK,
    )
    return state.replace(
        reservation_present=jnp.asarray(True),
        reservation_due_tick=state.tick,
        reservation_object_id=state.crop_object_id,
        reservation_generation=state.crop_generation,
        reservation_cycle_id=state.crop_cycle_id,
    )


def _pack_trigger_state(*, packed: bool):
    state = make_pack_restore_state(
        0,
        PackRestorePhase.LOADED,
        split=PackRestoreSplit.VALIDATION,
        start=PackRestoreStart.PATH_CHECK,
        source_growth_period=8,
    )
    return state.replace(
        player_position=state.anchor_position + jnp.asarray((1, 0)),
        anchor_present=jnp.asarray(False),
        anchor_grain=jnp.asarray(0, dtype=jnp.int32),
        record_present=jnp.asarray(True),
        record_grain_preview=jnp.asarray(1, dtype=jnp.int32),
        empty_frames=jnp.asarray(1, dtype=jnp.int32),
        packed_present=jnp.asarray(packed),
        packed_grain=jnp.asarray(1 if packed else 0, dtype=jnp.int32),
    )


def _craft_trigger_state():
    state = make_craft_remain_state(CraftRemainPhase.RIPE, CraftRemainStart.PATH_CHECK)
    return state.replace(slot_a=jnp.asarray(1), slot_b=jnp.asarray(1))


def test_frozen_panel_is_generated_exactly_and_caps_are_explicit() -> None:
    frozen = json.loads(
        (ROOT / "docs/manifests/cross_environment_teacher_trace_panel_v1.json").read_text()
    )
    assert frozen == build_panel_manifest()
    assert frozen["families"]["tick_claim"]["natural_reset"]["count"] == 32
    assert frozen["families"]["pack_restore"]["common_setup"]["count"] == 32
    assert frozen["families"]["craft_remain"]["path_check"]["count"] == 2
    assert frozen["evaluation_upper_bound"]["total_episodes"] == 88_200
    assert frozen["evaluation_upper_bound"]["total_steps"] == 11_289_600


def test_independent_oracle_detects_each_real_mutant_transition() -> None:
    cases = (
        ("tick_claim", _tick_trigger_state(), int(TickClaimAction.DO)),
        ("pack_restore", _pack_trigger_state(packed=True), int(PackRestoreAction.REBUILD_EMPTY)),
        ("craft_remain", _craft_trigger_state(), int(CraftRemainAction.CRAFT)),
    )
    for environment, state, action in cases:
        assert action_has_defect_effect(environment, state, action)
        assert action in legal_trigger_actions(environment, state)
        fixed, fixed_aux = actual_step(environment, state, action, "fixed")
        mutant, mutant_aux = actual_step(environment, state, action, "mutant")
        _, fixed_event = audit_actual_step(
            environment, state, action, fixed, fixed_aux, "fixed", initial_episode_oracle(environment)
        )
        _, mutant_event = audit_actual_step(
            environment, state, action, mutant, mutant_aux, "mutant", initial_episode_oracle(environment)
        )
        assert fixed_event["actual_defect_trigger"] is False
        assert mutant_event["actual_defect_trigger"] is True
        assert mutant_event["physical_resource_after"] > fixed_event["physical_resource_after"]


def test_exhaustive_legal_opportunity_does_not_replace_pack_legacy_metric() -> None:
    state = _pack_trigger_state(packed=False)
    assert int(PackRestoreAction.REBUILD_EMPTY) in legal_trigger_actions("pack_restore", state)
    assert legacy_opportunity_reached("pack_restore", state) is False
    panel = build_panel_manifest()
    assert "never overwritten" in panel["legacy_metric_rule"]


def test_public_state_has_no_oracle_or_defect_identity_input() -> None:
    state = _craft_trigger_state()
    field_names = set(state.__dataclass_fields__)
    assert "variant" not in field_names
    assert "oracle" not in field_names
    assert "legal_trigger_action_ids_exhaustive" not in field_names
