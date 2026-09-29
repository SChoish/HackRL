#!/usr/bin/env python3
"""Materialize and validate the TICK-CLAIM kernel-stage artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from hackrl.tick_claim import (
    GOAL_IDS,
    GROWTH_WAIT_TICKS,
    MAP_CHANNEL_NAMES,
    MODEL_FEATURE_NAMES,
    TickClaimAction,
    TickClaimPhase,
    TickClaimSplit,
    TickClaimStart,
    TickClaimTransitionSnapshot,
    TickClaimVariant,
    encode_tick_claim_observation,
    make_tick_claim_state,
    materialize_layout,
    observe_tick_claim,
    reset_tick_claim_worker,
    tick_claim_goal,
    tick_claim_goal_vector,
    tick_claim_setup_prefix,
    tick_claim_step,
    tick_claim_step_with_snapshot,
    transform_direction,
)
from hackrl.tick_claim_oracle import (
    audit_tick_claim_transition,
    compare_tick_claim_effect,
    initial_tick_claim_oracle,
    tick_claim_beneficial_use,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "manifests"
BASIS_COMMIT = "108ea5378c9f73855e3cc453a8a1dfa3b16d6c88"
SOURCE_PATHS = (
    ROOT / "src" / "hackrl" / "tick_claim.py",
    ROOT / "src" / "hackrl" / "tick_claim_oracle.py",
)
IMPLEMENTATION_PATHS = SOURCE_PATHS + (
    ROOT / "scripts" / "validate_tick_claim.py",
    ROOT / "tests" / "test_tick_claim.py",
)
EVALUATION_STARTS = (
    TickClaimStart.NATURAL,
    TickClaimStart.COMMON_SETUP,
)


def _json_bytes(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _hash_value(value) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _integer(value) -> int:
    return int(np.asarray(value))


def _boolean(value) -> bool:
    return bool(np.asarray(value))


def _position(value) -> list[int]:
    return [int(item) for item in np.asarray(value).tolist()]


def _transformed_action(layout_index: int, action: TickClaimAction) -> int:
    return _integer(transform_direction(action, layout_index))


def _setup_prefix(layout_index: int) -> tuple[int, ...]:
    return tuple(
        int(action) for action in np.asarray(tick_claim_setup_prefix(layout_index))
    )


def _growth_actions(
    layout_index: int, relocate_from_alternate_pose: bool = False
) -> tuple[int, ...]:
    if not relocate_from_alternate_pose or layout_index % 2 == 0:
        return (TickClaimAction.NOOP,) * GROWTH_WAIT_TICKS
    canonical = (
        TickClaimAction.RIGHT,
        TickClaimAction.DOWN,
        TickClaimAction.DOWN,
        TickClaimAction.DOWN,
        TickClaimAction.LEFT,
        TickClaimAction.LEFT,
        TickClaimAction.UP,
        TickClaimAction.UP,
    )
    return tuple(_transformed_action(layout_index, action) for action in canonical)


def _normal_core(layout_index: int, *, natural: bool = False) -> tuple[int, ...]:
    return (
        (TickClaimAction.DO,)
        + _growth_actions(layout_index, natural)
        + (TickClaimAction.DO,)
        + (TickClaimAction.NOOP,) * GROWTH_WAIT_TICKS
        + (TickClaimAction.DO,)
        + (_transformed_action(layout_index, TickClaimAction.DOWN),)
        + (TickClaimAction.DELIVER,)
    )


def _exploit_core(
    layout_index: int, delay: int, *, natural: bool = False
) -> tuple[int, ...]:
    return (
        (TickClaimAction.ARM_HARVEST,)
        + (TickClaimAction.NOOP,) * (delay - 1)
        + (TickClaimAction.DO,)
        + _growth_actions(layout_index, natural)
        + (TickClaimAction.DO,)
        + (_transformed_action(layout_index, TickClaimAction.DOWN),)
        + (TickClaimAction.DELIVER,)
    )


def _action_name(action: int) -> str:
    return TickClaimAction(int(action)).name


def _snapshot(state) -> dict:
    return {
        "tick": _integer(state.tick),
        "player_position": _position(state.player_position),
        "player_direction": _integer(state.player_direction),
        "grain": _integer(state.grain),
        "delivered_total": _integer(state.delivered_total),
        "crop": {
            "object_id": _integer(state.crop_object_id),
            "generation": _integer(state.crop_generation),
            "cycle_id": _integer(state.crop_cycle_id),
            "ripe": _boolean(state.crop_ripe),
            "age": _integer(state.crop_age),
        },
        "reservation": {
            "present": _boolean(state.reservation_present),
            "due_tick": _integer(state.reservation_due_tick),
            "object_id": _integer(state.reservation_object_id),
            "generation": _integer(state.reservation_generation),
            "cycle_id": _integer(state.reservation_cycle_id),
            "delay": _integer(state.reservation_delay),
        },
    }


def _trace(initial_state, actions, variant: TickClaimVariant) -> dict:
    state = initial_state
    oracle = initial_tick_claim_oracle()
    steps = []
    for action in actions:
        before = state
        state, transition_snapshot = tick_claim_step_with_snapshot(
            state, action, variant
        )
        audit = audit_tick_claim_transition(
            oracle, before, action, state, transition_snapshot
        )
        oracle = audit.oracle_state
        steps.append(
            {
                "action": _action_name(action),
                "action_value": int(action),
                "before": _snapshot(before),
                "settlement_snapshot": {
                    "grain": _integer(
                        transition_snapshot.grain_after_settlement
                    ),
                    "delivered_total": _integer(
                        transition_snapshot.delivered_total_after_settlement
                    ),
                },
                "after": _snapshot(state),
                "oracle": {
                    "created_amount": _integer(audit.created_amount),
                    "cycle_payout_total": _integer(audit.cycle_payout_total),
                    "violation": _boolean(audit.violation),
                    "settlement_conserved": _boolean(
                        audit.settlement_conserved
                    ),
                    "delivery_conserved": _boolean(audit.delivery_conserved),
                },
            }
        )
    return {
        "variant": variant.value,
        "actions": [_action_name(action) for action in actions],
        "step_count": len(actions),
        "violation_seen": _boolean(oracle.violation_seen),
        "final": _snapshot(state),
        "steps": steps,
    }


def _run_final(initial_state, actions, variant: TickClaimVariant):
    state = initial_state
    oracle = initial_tick_claim_oracle()
    settlement_conserved = jnp.asarray(True)
    delivery_conserved = jnp.asarray(True)
    for action in actions:
        before = state
        state, transition_snapshot = tick_claim_step_with_snapshot(
            state, action, variant
        )
        audit = audit_tick_claim_transition(
            oracle, before, action, state, transition_snapshot
        )
        settlement_conserved = jnp.logical_and(
            settlement_conserved, audit.settlement_conserved
        )
        delivery_conserved = jnp.logical_and(
            delivery_conserved, audit.delivery_conserved
        )
        oracle = audit.oracle_state
    assert _boolean(settlement_conserved)
    assert _boolean(delivery_conserved)
    return state, oracle


def _render_layout(split: TickClaimSplit, layout_index: int) -> dict:
    walkable, crop, device, delivery = materialize_layout(layout_index, split)
    walkable_array = np.asarray(walkable)
    crop_position = _position(crop)
    device_position = _position(device)
    delivery_position = _position(delivery)
    rows = []
    for row in range(16):
        characters = []
        for column in range(16):
            position = [row, column]
            character = "." if walkable_array[row, column] else "#"
            if position == crop_position:
                character = "C"
            elif position == device_position:
                character = "R"
            elif position == delivery_position:
                character = "D"
            characters.append(character)
        rows.append("".join(characters))
    layout = {
        "split": split.value,
        "layout_index": layout_index,
        "reservation_delay": _integer(
            make_tick_claim_state(layout_index, TickClaimPhase.RIPE).reservation_delay
        ),
        "crop_position": crop_position,
        "reservation_device_position": device_position,
        "delivery_position": delivery_position,
        "map_legend": {
            "#": "wall",
            ".": "floor",
            "C": "crop",
            "R": "device",
            "D": "delivery",
        },
        "map_rows": rows,
    }
    physical_geometry = {
        "crop_position": crop_position,
        "reservation_device_position": device_position,
        "delivery_position": delivery_position,
        "map_rows": rows,
    }
    layout["layout_sha256"] = _hash_value(physical_geometry)
    return layout


def _materialized_state(split, layout_index, phase, start) -> dict:
    state = make_tick_claim_state(
        layout_index, phase, split=split, start=start
    )
    observation = observe_tick_claim(state)
    goals = tick_claim_goal_vector(observation)
    state_snapshot = _snapshot(state)
    if start is TickClaimStart.COMMON_SETUP:
        prefix = _setup_prefix(layout_index)
        materialization = {
            "method": "fixed_kernel_prefix",
            "source_start": TickClaimStart.NATURAL.value,
            "prefix_actions": [_action_name(action) for action in prefix],
            "prefix_action_values": list(prefix),
            "elapsed_steps": len(prefix),
            "final_state_sha256": _hash_value(state_snapshot),
        }
    else:
        materialization = {
            "method": "natural_reset",
            "source_start": None,
            "prefix_actions": [],
            "prefix_action_values": [],
            "elapsed_steps": 0,
            "final_state_sha256": _hash_value(state_snapshot),
        }
    value = {
        "state_id": f"{split.value}/{start.value}/layout_{layout_index:02d}/phase_{int(phase)}",
        "split": split.value,
        "start": start.value,
        "layout_index": layout_index,
        "phase": TickClaimPhase(int(phase)).name.lower(),
        "materialization": materialization,
        "state": state_snapshot,
        "true_goal_ids": [
            goal_id
            for goal_id, achieved in zip(GOAL_IDS, np.asarray(goals).tolist())
            if achieved
        ],
    }
    value["state_sha256"] = _hash_value(value)
    return value


def _validate_common_setup_materialization() -> dict:
    checked = 0
    for split in TickClaimSplit:
        for layout_index in range(16):
            prefix = _setup_prefix(layout_index)
            assert len(prefix) == 4
            for phase in TickClaimPhase:
                natural = make_tick_claim_state(
                    layout_index,
                    phase,
                    split=split,
                    start=TickClaimStart.NATURAL,
                )
                expected, _ = _run_final(
                    natural, prefix, TickClaimVariant.FIXED
                )
                actual = make_tick_claim_state(
                    layout_index,
                    phase,
                    split=split,
                    start=TickClaimStart.COMMON_SETUP,
                )
                assert jax.tree_util.tree_all(
                    jax.tree.map(jnp.array_equal, expected, actual)
                )
                assert _integer(actual.tick) == 4
                assert _integer(actual.grain) == 0
                assert not _boolean(actual.reservation_present)
                expected_age = (
                    GROWTH_WAIT_TICKS
                    if phase is TickClaimPhase.RIPE
                    else 4
                )
                assert _integer(actual.crop_age) == expected_age
                checked += 1
    return {
        "status": "passed",
        "checked_states": checked,
        "prefix_steps": 4,
        "elapsed_time_preserved": True,
        "inventory_preserved": True,
        "unripe_crop_age_after_prefix": 4,
        "synthetic_path_check_excluded_from_evaluation_states": True,
    }


def _implementation_code_sha():
    relative_paths = [str(path.relative_to(ROOT)) for path in IMPLEMENTATION_PATHS]
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *relative_paths],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        return None
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_goal_catalog() -> dict:
    state = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    actions = (
        TickClaimAction.ARM_HARVEST,
        TickClaimAction.CANCEL_HARVEST,
        TickClaimAction.DO,
    )
    actions += (TickClaimAction.NOOP,) * GROWTH_WAIT_TICKS
    actions += (TickClaimAction.DO,)
    actions += (TickClaimAction.NOOP,) * GROWTH_WAIT_TICKS
    actions += (
        TickClaimAction.DO,
        TickClaimAction.DOWN,
        TickClaimAction.DELIVER,
    )
    observation = observe_tick_claim(state)
    seen = tick_claim_goal_vector(observation)
    for action in actions:
        state = tick_claim_step(state, action, TickClaimVariant.FIXED)
        observation = observe_tick_claim(state)
        vector = tick_claim_goal_vector(observation)
        individual = jnp.stack(
            [tick_claim_goal(observation, index) for index in range(len(GOAL_IDS))]
        )
        assert bool(jnp.array_equal(vector, individual))
        seen = jnp.logical_or(seen, vector)
    assert bool(jnp.all(seen))

    map_feature_count = 7 * 9 * len(MAP_CHANNEL_NAMES)
    grain_feature_index = map_feature_count + MODEL_FEATURE_NAMES.index("grain_signed")
    boundary_values = []
    base_observation = observe_tick_claim(
        make_tick_claim_state(
            0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
        )
    )
    for count in range(5):
        encoded = encode_tick_claim_observation(
            base_observation.replace(grain=jnp.asarray(count, dtype=jnp.int32))
        )
        boundary_values.append(float(encoded[grain_feature_index]))
    assert boundary_values == sorted(set(boundary_values))

    hidden_changed = state.replace(
        cycle_claimed=jnp.logical_not(state.cycle_claimed),
        crop_generation=state.crop_generation + 10,
    )
    assert bool(
        jnp.array_equal(
            tick_claim_goal_vector(observe_tick_claim(state)),
            tick_claim_goal_vector(observe_tick_claim(hidden_changed)),
        )
    )
    return {
        "fixed_legal_sequence_reaches_all_12": True,
        "reached_goal_ids": list(GOAL_IDS),
        "single_and_vector_predicates_equal": True,
        "hidden_analysis_state_invariant": True,
        "grain_boundary_model_values_0_through_4": boundary_values,
    }


def _validate_negative_controls() -> dict:
    initial = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    due, _ = _run_final(
        initial,
        (TickClaimAction.ARM_HARVEST, TickClaimAction.NOOP),
        TickClaimVariant.FIXED,
    )
    cases = {
        "one_tick_early": _trace(
            initial,
            (TickClaimAction.ARM_HARVEST, TickClaimAction.DO),
            TickClaimVariant.MUTANT,
        ),
        "one_tick_late": _trace(
            initial,
            (
                TickClaimAction.ARM_HARVEST,
                TickClaimAction.NOOP,
                TickClaimAction.NOOP,
                TickClaimAction.DO,
            ),
            TickClaimVariant.MUTANT,
        ),
        "cancel_on_due_tick": _trace(
            initial,
            (
                TickClaimAction.ARM_HARVEST,
                TickClaimAction.NOOP,
                TickClaimAction.CANCEL_HARVEST,
            ),
            TickClaimVariant.MUTANT,
        ),
        "unripe_crop": _trace(
            due.replace(crop_ripe=jnp.asarray(False), crop_age=jnp.asarray(0)),
            (TickClaimAction.DO,),
            TickClaimVariant.MUTANT,
        ),
        "old_generation": _trace(
            due.replace(reservation_generation=due.crop_generation - 1),
            (TickClaimAction.DO,),
            TickClaimVariant.MUTANT,
        ),
        "old_cycle": _trace(
            due.replace(reservation_cycle_id=due.crop_cycle_id - 1),
            (TickClaimAction.DO,),
            TickClaimVariant.MUTANT,
        ),
        "normal_single_claim": _trace(
            initial, (TickClaimAction.DO,), TickClaimVariant.MUTANT
        ),
    }
    assert all(not result["violation_seen"] for result in cases.values())
    expected_grain = {
        "one_tick_early": 1,
        "one_tick_late": 1,
        "cancel_on_due_tick": 0,
        "unripe_crop": 0,
        "old_generation": 1,
        "old_cycle": 1,
        "normal_single_claim": 1,
    }
    actual_grain = {
        name: result["final"]["grain"] for name, result in cases.items()
    }
    assert actual_grain == expected_grain
    return {
        "status": "passed",
        "final_grain_by_case": actual_grain,
        "violation_by_case": {name: False for name in cases},
    }


def _validate_oracle_sensitivity() -> dict:
    before = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    double_created = before.replace(
        grain=jnp.asarray(2),
        crop_ripe=jnp.asarray(False),
        crop_age=jnp.asarray(0),
        tick=before.tick + 1,
    )
    double_audit = audit_tick_claim_transition(
        initial_tick_claim_oracle(),
        before,
        TickClaimAction.DO,
        double_created,
        TickClaimTransitionSnapshot(
            grain_after_settlement=jnp.asarray(2),
            delivered_total_after_settlement=before.delivered_total,
        ),
    )
    assert _integer(double_audit.created_amount) == 2
    assert _boolean(double_audit.violation)

    scheduled_delivery = _trace(
        before,
        (
            TickClaimAction.ARM_HARVEST,
            TickClaimAction.DOWN,
            TickClaimAction.DELIVER,
        ),
        TickClaimVariant.FIXED,
    )
    scheduled_step = scheduled_delivery["steps"][-1]
    assert scheduled_delivery["final"]["delivered_total"] == 1
    assert scheduled_step["oracle"]["created_amount"] == 1
    assert scheduled_step["oracle"]["settlement_conserved"]
    assert scheduled_step["oracle"]["delivery_conserved"]
    assert not scheduled_step["oracle"]["violation"]

    outside_delivery = _trace(
        before,
        (
            TickClaimAction.ARM_HARVEST,
            TickClaimAction.NOOP,
            TickClaimAction.DELIVER,
        ),
        TickClaimVariant.FIXED,
    )
    outside_step = outside_delivery["steps"][-1]
    assert outside_delivery["final"]["grain"] == 1
    assert outside_delivery["final"]["delivered_total"] == 0
    assert outside_step["oracle"]["delivery_conserved"]

    carrying = before.replace(
        player_position=jnp.asarray((10, 8), dtype=jnp.int32),
        grain=jnp.asarray(2, dtype=jnp.int32),
    )
    delivered, delivery_snapshot = tick_claim_step_with_snapshot(
        carrying, TickClaimAction.DELIVER, TickClaimVariant.FIXED
    )
    production_free = audit_tick_claim_transition(
        initial_tick_claim_oracle(),
        carrying,
        TickClaimAction.DELIVER,
        delivered,
        delivery_snapshot,
    )
    assert _integer(production_free.created_amount) == 0
    assert _boolean(production_free.settlement_conserved)
    assert _boolean(production_free.delivery_conserved)

    broken_loss = delivered.replace(
        delivered_total=jnp.asarray(1, dtype=jnp.int32)
    )
    broken_overdelivery = delivered.replace(
        delivered_total=jnp.asarray(3, dtype=jnp.int32)
    )
    for corrupted in (broken_loss, broken_overdelivery):
        delivery_audit = audit_tick_claim_transition(
            initial_tick_claim_oracle(),
            carrying,
            TickClaimAction.DELIVER,
            corrupted,
            delivery_snapshot,
        )
        assert not _boolean(delivery_audit.delivery_conserved)

    bad_settlement_snapshot = TickClaimTransitionSnapshot(
        grain_after_settlement=carrying.grain,
        delivered_total_after_settlement=carrying.delivered_total + 1,
    )
    bad_settlement = audit_tick_claim_transition(
        initial_tick_claim_oracle(),
        carrying,
        TickClaimAction.NOOP,
        carrying.replace(tick=carrying.tick + 1),
        bad_settlement_snapshot,
    )
    assert not _boolean(bad_settlement.settlement_conserved)
    assert not _boolean(
        tick_claim_beneficial_use(
            before.replace(delivered_total=jnp.asarray(2)),
            before.replace(delivered_total=jnp.asarray(3)),
            jnp.asarray(False),
        )
    )
    return {
        "double_creation_detected": True,
        "scheduled_settlement_plus_valid_delivery_accepted": True,
        "scheduled_settlement_outside_delivery_range_accepted_without_transfer": True,
        "production_free_delivery_accepted": True,
        "physical_loss_rejected": True,
        "overdelivery_rejected": True,
        "settlement_boundary_corruption_rejected": True,
        "goal_only_surplus_without_violation_rejected": True,
    }


def _validate_jit_vmap() -> dict:
    state = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    eager = tick_claim_step(state, TickClaimAction.NOOP, TickClaimVariant.FIXED)
    compiled = jax.jit(
        lambda current, action: tick_claim_step(
            current, action, TickClaimVariant.FIXED
        )
    )(state, jnp.asarray(TickClaimAction.NOOP))
    jit_equal = bool(
        jax.tree_util.tree_all(
            jax.tree.map(lambda left, right: jnp.array_equal(left, right), eager, compiled)
        )
    )

    def trigger(current, variant):
        armed = tick_claim_step(
            current, TickClaimAction.ARM_HARVEST, variant
        )
        due = jax.lax.while_loop(
            lambda item: jnp.logical_and(
                item.reservation_present,
                item.tick < item.reservation_due_tick,
            ),
            lambda item: tick_claim_step(
                item, TickClaimAction.NOOP, variant
            ),
            armed,
        )
        return tick_claim_step(due, TickClaimAction.DO, variant)

    workers = jnp.arange(512, dtype=jnp.int32)
    states = jax.vmap(
        lambda worker: reset_tick_claim_worker(
            worker, start=TickClaimStart.PATH_CHECK
        )
    )(workers)
    fixed_triggered = jax.jit(
        jax.vmap(lambda current: trigger(current, TickClaimVariant.FIXED))
    )(states)
    mutant_triggered = jax.jit(
        jax.vmap(lambda current: trigger(current, TickClaimVariant.MUTANT))
    )(states)
    phases = np.bincount(np.asarray(states.initial_phase), minlength=2).tolist()
    layouts = np.bincount(np.asarray(states.layout_index), minlength=16).tolist()
    ripe = np.asarray(states.initial_phase) == TickClaimPhase.RIPE
    unripe = np.logical_not(ripe)
    fixed_grain = np.asarray(fixed_triggered.grain)
    mutant_grain = np.asarray(mutant_triggered.grain)
    actual_trigger_ok = bool(
        np.all(fixed_grain[ripe] == 1)
        and np.all(mutant_grain[ripe] == 2)
        and np.all(fixed_grain[unripe] == 0)
        and np.all(mutant_grain[unripe] == 0)
    )
    single_state = jax.tree.map(lambda value: value[0], states)
    eager_trigger = trigger(single_state, TickClaimVariant.FIXED)
    trigger_jit_equal = bool(
        jax.tree_util.tree_all(
            jax.tree.map(
                lambda left, right: jnp.array_equal(left, right),
                eager_trigger,
                jax.tree.map(lambda value: value[0], fixed_triggered),
            )
        )
    )
    assert jax.default_backend() == "cpu"
    assert (
        jit_equal
        and trigger_jit_equal
        and actual_trigger_ok
        and phases == [256, 256]
        and layouts == [32] * 16
    )
    return {
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "eager_jit_equal": jit_equal,
        "actual_trigger_eager_jit_equal": trigger_jit_equal,
        "vmap_512_actual_trigger": actual_trigger_ok,
        "ripe_fixed_payout": 1,
        "ripe_mutant_payout": 2,
        "unripe_fixed_and_mutant_payout": 0,
        "phase_counts": phases,
        "layout_counts": layouts,
    }


def main() -> None:
    layouts = [_render_layout(split, index) for split in TickClaimSplit for index in range(16)]
    layout_hashes_by_split = {
        split.value: [
            layout["layout_sha256"] for layout in layouts if layout["split"] == split.value
        ]
        for split in TickClaimSplit
    }
    assert all(len(set(hashes)) == 16 for hashes in layout_hashes_by_split.values())
    assert len({digest for hashes in layout_hashes_by_split.values() for digest in hashes}) == 48
    layouts_document = {
        "schema_version": "tick_claim_materialized_layouts_v1",
        "candidate_id": "TICK-CLAIM",
        "basis_commit": BASIS_COMMIT,
        "layout_count": len(layouts),
        "layouts": layouts,
    }

    common_setup_validation = _validate_common_setup_materialization()
    states = [
        _materialized_state(split, index, phase, start)
        for split in TickClaimSplit
        for start in EVALUATION_STARTS
        for index in range(16)
        for phase in TickClaimPhase
    ]
    assert len(states) == 192
    states_document = {
        "schema_version": "tick_claim_materialized_states_v2",
        "candidate_id": "TICK-CLAIM",
        "basis_commit": BASIS_COMMIT,
        "state_count": len(states),
        "evaluation_starts": [start.value for start in EVALUATION_STARTS],
        "common_setup_validation": common_setup_validation,
        "states": states,
    }

    normal_traces = []
    exploit_traces = []
    payoff_rows = []
    path_replay_count = 0
    for split in TickClaimSplit:
        for layout_index in range(16):
            for phase in TickClaimPhase:
                common_setup = make_tick_claim_state(
                    layout_index,
                    phase,
                    split=split,
                    start=TickClaimStart.COMMON_SETUP,
                )
                unripe_wait = (
                    (TickClaimAction.NOOP,)
                    * max(GROWTH_WAIT_TICKS - _integer(common_setup.crop_age), 0)
                )
                normal_actions = (
                    unripe_wait + _normal_core(layout_index, natural=True)
                )
                delay = _integer(common_setup.reservation_delay)
                exploit_actions = (
                    unripe_wait
                    + _exploit_core(layout_index, delay, natural=True)
                )
                fixed_normal, fixed_normal_oracle = _run_final(
                    common_setup, normal_actions, TickClaimVariant.FIXED
                )
                mutant_normal, mutant_normal_oracle = _run_final(
                    common_setup, normal_actions, TickClaimVariant.MUTANT
                )
                fixed_exploit, fixed_exploit_oracle = _run_final(
                    common_setup, exploit_actions, TickClaimVariant.FIXED
                )
                mutant_exploit, mutant_exploit_oracle = _run_final(
                    common_setup, exploit_actions, TickClaimVariant.MUTANT
                )
                path_replay_count += 4

                assert _integer(fixed_normal.delivered_total) == 3
                assert _integer(mutant_normal.delivered_total) == 3
                assert not _boolean(fixed_normal_oracle.violation_seen)
                assert not _boolean(mutant_normal_oracle.violation_seen)
                assert _integer(fixed_exploit.delivered_total) == 2
                assert _integer(mutant_exploit.delivered_total) == 3
                assert not _boolean(fixed_exploit_oracle.violation_seen)
                assert _boolean(mutant_exploit_oracle.violation_seen)

                phase_wait_steps = len(unripe_wait)
                fixed_total_steps = 25 + phase_wait_steps
                mutant_total_steps = 16 + delay + phase_wait_steps
                assert _integer(fixed_normal.tick) == fixed_total_steps
                assert _integer(mutant_exploit.tick) == mutant_total_steps
                assert len(normal_actions) == 21 + phase_wait_steps
                assert len(exploit_actions) == delay + 12 + phase_wait_steps

                if split is TickClaimSplit.TRAIN and layout_index in (0, 1, 2):
                    fixed_normal_trace = _trace(
                        common_setup, normal_actions, TickClaimVariant.FIXED
                    )
                    mutant_normal_trace = _trace(
                        common_setup, normal_actions, TickClaimVariant.MUTANT
                    )
                    fixed_exploit_trace = _trace(
                        common_setup, exploit_actions, TickClaimVariant.FIXED
                    )
                    mutant_exploit_trace = _trace(
                        common_setup, exploit_actions, TickClaimVariant.MUTANT
                    )
                    trace_key = {
                        "split": split.value,
                        "layout_index": layout_index,
                        "phase": phase.name.lower(),
                        "initial_tick": _integer(common_setup.tick),
                        "initial_crop_age": _integer(common_setup.crop_age),
                    }
                    normal_traces.append(
                        {
                            **trace_key,
                            "fixed": fixed_normal_trace,
                            "mutant": mutant_normal_trace,
                        }
                    )
                    exploit_traces.append(
                        {
                            **trace_key,
                            "fixed": fixed_exploit_trace,
                            "mutant": mutant_exploit_trace,
                        }
                    )

                payoff_rows.append(
                    {
                        "split": split.value,
                        "layout_index": layout_index,
                        "phase": phase.name.lower(),
                        "reservation_delay": delay,
                        "natural_access_steps": 4,
                        "phase_wait_steps": phase_wait_steps,
                        "common_setup_tick": _integer(common_setup.tick),
                        "common_setup_crop_age": _integer(common_setup.crop_age),
                        "fixed_reference_actions_from_setup": len(normal_actions),
                        "mutant_exploit_actions_from_setup": len(exploit_actions),
                        "fixed_reference_total_steps_from_reset": fixed_total_steps,
                        "mutant_exploit_total_steps_from_reset": mutant_total_steps,
                        "strict_step_advantage": (
                            fixed_total_steps - mutant_total_steps
                        ),
                        "fixed_delivered_on_exploit_path": (
                            _integer(fixed_exploit.delivered_total)
                        ),
                        "mutant_delivered_on_exploit_path": (
                            _integer(mutant_exploit.delivered_total)
                        ),
                    }
                )

    assert path_replay_count == 384
    normal_document = {
        "schema_version": "tick_claim_trace_v2",
        "trace_kind": "normal",
        "basis_commit": BASIS_COMMIT,
        "reference_scope": {
            "claim": "constructive fixed path, not a global optimality proof",
            "ripe_total_steps_from_natural_reset": 25,
            "unripe_total_steps_from_natural_reset": 29,
            "common_setup_is_fixed_prefix_result": True,
        },
        "full_path_replays": 192,
        "recorded_detailed_traces": len(normal_traces) * 2,
        "traces": normal_traces,
    }
    exploit_document = {
        "schema_version": "tick_claim_trace_v2",
        "trace_kind": "exploit",
        "basis_commit": BASIS_COMMIT,
        "full_path_replays": 192,
        "recorded_detailed_traces": len(exploit_traces) * 2,
        "traces": exploit_traces,
    }

    # Re-run the exact trigger fork for a separately recorded effect result.
    trigger = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    trigger = tick_claim_step(
        trigger, TickClaimAction.ARM_HARVEST, TickClaimVariant.FIXED
    )
    trigger = tick_claim_step(trigger, TickClaimAction.NOOP, TickClaimVariant.FIXED)
    fixed_after = tick_claim_step(trigger, TickClaimAction.DO, TickClaimVariant.FIXED)
    mutant_after = tick_claim_step(trigger, TickClaimAction.DO, TickClaimVariant.MUTANT)
    effect = compare_tick_claim_effect(
        trigger, TickClaimAction.DO, fixed_after, mutant_after
    )
    assert _boolean(effect.effect) and _integer(effect.physical_surplus) == 1
    exploit_actions = _exploit_core(0, _integer(trigger.reservation_delay))
    exploit_start = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    fixed_final, _ = _run_final(
        exploit_start, exploit_actions, TickClaimVariant.FIXED
    )
    mutant_final, mutant_oracle = _run_final(
        exploit_start, exploit_actions, TickClaimVariant.MUTANT
    )
    assert _boolean(
        tick_claim_beneficial_use(
            fixed_final, mutant_final, mutant_oracle.violation_seen
        )
    )

    goal_validation = _validate_goal_catalog()
    negative_controls = _validate_negative_controls()
    oracle_sensitivity = _validate_oracle_sensitivity()
    jit_vmap = _validate_jit_vmap()
    validation_document = {
        "schema_version": "tick_claim_kernel_validation_v2",
        "candidate_id": "TICK-CLAIM",
        "basis_commit": BASIS_COMMIT,
        "validation_status": "passed_kernel_stage",
        "validation_scope": "synthetic kernel, fixed-prefix evaluation states, public observation, goals, traces, and transition oracles; not GC learner or external-game reproduction",
        "source_review_assertions": {
            "status": "not_a_runtime_proof",
            "fixed_mutant_shared_transition_except_consumed_read": True,
        },
        "layout_validation": {
            "materialized_count": 48,
            "unique_within_each_split": True,
            "no_hash_overlap_across_splits": True,
        },
        "state_validation": {
            "materialized_count": 192,
            "starts": [start.value for start in EVALUATION_STARTS],
            "phases": [phase.name.lower() for phase in TickClaimPhase],
            "common_setup": common_setup_validation,
        },
        "goal_validation": goal_validation,
        "independent_oracle": {
            "independence_scope": "separate oracle over an analysis-only settlement boundary snapshot and crop provenance; snapshot is emitted by the same-author synthetic kernel, so this is not independent authorship or an external engine reproduction",
            "reads_kernel_violation_boolean": False,
            "reads_reward_predicate": False,
            "reads_analysis_boundary_snapshot": True,
            "settlement_and_delivery_checked_separately": True,
            "adversarial_sensitivity": oracle_sensitivity,
            "effect_action": _integer(effect.action),
            "before_physical_total": _integer(effect.before_physical_total),
            "fixed_created_amount": _integer(effect.fixed_created_amount),
            "mutant_created_amount": _integer(effect.mutant_created_amount),
            "effect_physical_surplus": _integer(effect.physical_surplus),
            "delivery_transfer_conservation_checked_on_valid_and_invalid_deliver": True,
            "non_delivery_transfer_must_remain_unchanged": True,
            "physical_loss_and_overdelivery_rejected": True,
            "negative_controls": negative_controls,
        },
        "payoff_validation": {
            "status": "passed_by_constructive_path_replay",
            "optimality_claim": False,
            "path_replays": path_replay_count,
            "coverage": {
                "splits": [split.value for split in TickClaimSplit],
                "layouts_per_split": 16,
                "phases": [phase.name.lower() for phase in TickClaimPhase],
                "paths": ["fixed_normal", "mutant_normal", "fixed_exploit", "mutant_exploit"],
            },
            "fixed_reference_steps_from_reset_by_phase": {
                "ripe": 25,
                "unripe": 29,
            },
            "mutant_exploit_steps_from_reset_by_phase_and_delay": {
                "ripe": {"2": 18, "3": 19, "4": 20},
                "unripe": {"2": 22, "3": 23, "4": 24},
            },
            "strict_advantage_by_delay": {"2": 7, "3": 6, "4": 5},
            "natural_access_steps": 4,
            "common_setup_preserves_access_elapsed_time": True,
            "preparation_is_identical_for_fixed_and_mutant": True,
            "per_split_layout_phase": payoff_rows,
        },
        "cpu_jit_vmap_validation": jit_vmap,
    }

    artifact_paths = {
        "layouts": OUTPUT_DIR / "tick_claim_v1_layouts.json",
        "states": OUTPUT_DIR / "tick_claim_v1_states.json",
        "normal_traces": OUTPUT_DIR / "tick_claim_v1_normal_traces.json",
        "exploit_traces": OUTPUT_DIR / "tick_claim_v1_exploit_traces.json",
        "validation": OUTPUT_DIR / "tick_claim_v1_validation.json",
    }
    documents = {
        "layouts": layouts_document,
        "states": states_document,
        "normal_traces": normal_document,
        "exploit_traces": exploit_document,
        "validation": validation_document,
    }
    for name, path in artifact_paths.items():
        _write_json(path, documents[name])

    action_schema = {
        "base_enum": "Craftax-Classic.Action 0..16",
        "actions": [
            {"name": action.name, "value": action.value}
            for action in TickClaimAction
        ],
        "valid_action_range_inclusive": [0, 19],
        "invalid_action_behavior": "unsupported public input",
    }
    observation_schema = {
        "public_map_shape": [7, 9],
        "map_encoding": "public Classic tile ids plus candidate role masks; 24-channel one-hot/mask tensor for model",
        "map_channels": list(MAP_CHANNEL_NAMES),
        "model_feature_order": list(MODEL_FEATURE_NAMES),
        "model_vector_size": 7 * 9 * len(MAP_CHANNEL_NAMES) + len(MODEL_FEATURE_NAMES),
        "count_transform": "x/(1+abs(x))",
        "hidden_facility_values": "zero with visible=false",
    }
    goal_schema = {
        "goal_ids": list(GOAL_IDS),
        "achieved_goal_vector": "12-dimensional multi-hot Boolean predicate vector",
        "command_goal_representation": "12-dimensional one-hot; GC wrapper not yet implemented",
        "valid_goal_indices": [0, 11],
    }
    source_hashes = {
        str(path.relative_to(ROOT)): _hash_file(path) for path in SOURCE_PATHS
    }
    attested_hashes = {
        str(path.relative_to(ROOT)): _hash_file(path)
        for path in IMPLEMENTATION_PATHS
    }
    implementation_code_sha = _implementation_code_sha()
    remaining_before_gc_training = [
        "wire structured observation and workshop12 goals into GC-PPO",
        "record GC policy parameter count",
        "validate complete checkpoint resume",
        "validate frozen evaluation does not mutate learner state",
    ]
    if implementation_code_sha is None:
        remaining_before_gc_training.insert(
            0, "commit implementation and record implementation_code_sha"
        )
    resolved = {
        "schema_version": "hackrl_candidate_resolved_v2",
        "candidate_id": "TICK-CLAIM",
        "fixture_id": "tick_claim_workshop_v1",
        "design_manifest": "tick_claim_v1.json",
        "design_manifest_status_preserved": "design_only",
        "basis_commit": BASIS_COMMIT,
        "basis_commit_kind": "user-selected document/manifest handoff baseline",
        "implementation_code_sha": implementation_code_sha,
        "implementation_tree_hash": _hash_value(attested_hashes),
        "attested_files": attested_hashes,
        "kernel_hash": source_hashes["src/hackrl/tick_claim.py"],
        "oracle_hash": source_hashes["src/hackrl/tick_claim_oracle.py"],
        "observation_schema_hash": _hash_value(observation_schema),
        "action_schema_hash": _hash_value(action_schema),
        "goal_schema_hash": _hash_value(goal_schema),
        "schemas": {
            "action": action_schema,
            "observation": observation_schema,
            "goal": goal_schema,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "jax": importlib.metadata.version("jax"),
            "flax": importlib.metadata.version("flax"),
            "craftax": importlib.metadata.version("craftax"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "verification_commands": {
            "artifacts": "JAX_PLATFORMS=cpu PYTHONPATH=src python scripts/validate_tick_claim.py",
            "focused_tests": "JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest -q tests/test_tick_claim.py",
            "full_tests": "JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest -q",
        },
        "public_input_domains": {
            "layout_index": "integer 0..15",
            "phase": "0 ripe or 1 unripe",
            "evaluation_start": "natural_reset or common_setup",
            "synthetic_test_start": "synthetic_path_check; excluded from evaluation manifests",
            "action": "integer 0..19",
            "goal_index": "integer 0..11",
            "post_world_done_step": "unsupported; caller must stop or reset",
        },
        "actual_parameter_counts": {
            "kernel_trainable_parameters": 0,
            "oracle_trainable_parameters": 0,
            "action_count": len(TickClaimAction),
            "goal_count": len(GOAL_IDS),
            "map_channel_count": len(MAP_CHANNEL_NAMES),
            "model_observation_size": observation_schema["model_vector_size"],
            "gc_policy_parameters": None,
        },
        "artifacts": {
            name: {
                "path": path.name,
                "sha256": _hash_file(path),
            }
            for name, path in artifact_paths.items()
        },
        "kernel_stage": {
            "status": "validated_after_scientific_review",
            "scope": "synthetic kernel through transition-level evaluation; GC learner remains unwired",
            "common_setup_fixed_prefix_materialization": "passed_96_states",
            "settlement_delivery_phase_oracle": "passed",
            "normal_and_exploit_path_replays": 384,
            "independent_oracle_validation": "passed_with_analysis_snapshot_and_same_author_scope_disclosed",
            "payoff_validation": "passed_constructive_paths_no_optimality_claim",
            "cpu_jit_vmap_validation": "passed_512_actual_trigger_workers",
        },
        "runnable": False,
        "queue_gate": "closed",
        "remaining_before_gc_training": remaining_before_gc_training,
        "pack_restore_status": "deferred_pending_payoff_gate",
    }
    resolved_path = OUTPUT_DIR / "tick_claim_v1_resolved.json"
    _write_json(resolved_path, resolved)

    print(
        json.dumps(
            {
                "status": "passed",
                "backend": jit_vmap["backend"],
                "layouts": len(layouts),
                "states": len(states),
                "normal_traces": len(normal_traces) * 2,
                "exploit_traces": len(exploit_traces) * 2,
                "path_replays": path_replay_count,
                "fixed_reference_steps": {"ripe": 25, "unripe": 29},
                "mutant_exploit_steps": {
                    "ripe": {"delay_2": 18, "delay_3": 19, "delay_4": 20},
                    "unripe": {"delay_2": 22, "delay_3": 23, "delay_4": 24},
                },
                "payoff_advantage_steps": {"delay_2": 7, "delay_3": 6, "delay_4": 5},
                "resolved_manifest": str(resolved_path.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
