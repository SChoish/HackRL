"""No-learning gates for the PACK inventory-overwrite defect.

This module contains validation-only reference paths and an oracle that is
independent of reward, goal success, and the kernel's variant branch. Nothing
here is imported by PPO collection or optimization.
"""

from __future__ import annotations

from dataclasses import fields

import jax.numpy as jnp
import numpy as np

from hackrl.pack_restore import (
    GOAL_IDS,
    INITIAL_GRAIN_TOTAL,
    TRAINING_SOURCE_GROWTH_PERIOD,
    PackRestoreAction,
    PackRestorePhase,
    PackRestoreSplit,
    PackRestoreStart,
    PackRestoreState,
    PackRestoreVariant,
    inventory_overwrite_trigger,
    make_pack_restore_state,
    pack_restore_goal_vector,
    pack_restore_setup_prefix,
    pack_restore_step,
    physical_grain_total,
)
from hackrl.tick_claim import transform_direction


_DIRECTION_DELTAS = np.asarray(
    ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)), dtype=np.int32
)
_DELIVER_3 = GOAL_IDS.index("delivery/count_ge_3")


def _movement(action: PackRestoreAction, layout: int) -> int:
    return int(np.asarray(transform_direction(int(action), layout)))


def inventory_safe_reference_actions(
    layout: int,
    phase: int,
    source_growth_period: int = TRAINING_SOURCE_GROWTH_PERIOD,
) -> tuple[int, ...]:
    """Return one non-optimal path that never harvests while carrying grain."""

    if source_growth_period < 0:
        raise ValueError("source_growth_period must be non-negative")
    actions = [int(value) for value in np.asarray(pack_restore_setup_prefix(layout))]
    if int(phase) == int(PackRestorePhase.LOADED):
        actions.append(int(PackRestoreAction.WITHDRAW_ONE))

    down = _movement(PackRestoreAction.DOWN, layout)
    up = _movement(PackRestoreAction.UP, layout)
    right = _movement(PackRestoreAction.RIGHT, layout)
    deliver_one = [down, int(PackRestoreAction.DELIVER), up, right]

    actions.extend(deliver_one)
    actions.extend([int(PackRestoreAction.NOOP)] * source_growth_period)
    actions.append(int(PackRestoreAction.DO))
    actions.extend(deliver_one)
    actions.extend([int(PackRestoreAction.NOOP)] * source_growth_period)
    actions.extend(
        [
            int(PackRestoreAction.DO),
            down,
            int(PackRestoreAction.DELIVER),
        ]
    )
    if len(actions) > 128:
        raise ValueError("reference path exceeds the unchanged 128-tick horizon")
    return tuple(actions)


def independent_trigger_oracle(state: PackRestoreState, action: int) -> bool:
    """Reward-blind trigger B: a valid harvest with carried inventory c > 0."""

    direction = int(np.asarray(state.player_direction))
    front = np.asarray(state.player_position) + _DIRECTION_DELTAS[direction]
    ripe = (
        int(np.asarray(state.source_growth_period)) == 0
        or int(np.asarray(state.source_age))
        >= int(np.asarray(state.source_growth_period))
    )
    return bool(
        int(action) == int(PackRestoreAction.DO)
        and int(np.asarray(state.carried_grain)) > 0
        and np.array_equal(front, np.asarray(state.source_position))
        and int(np.asarray(state.source_grain)) > 0
        and ripe
    )


def _changed_fields(left: PackRestoreState, right: PackRestoreState) -> tuple[str, ...]:
    changed = []
    for field in fields(PackRestoreState):
        if not np.array_equal(
            np.asarray(getattr(left, field.name)),
            np.asarray(getattr(right, field.name)),
        ):
            changed.append(field.name)
    return tuple(changed)


def _audit_pair(state: PackRestoreState, action: int) -> dict:
    fixed = pack_restore_step(state, action, PackRestoreVariant.FIXED)
    harmful = pack_restore_step(
        state, action, PackRestoreVariant.INVENTORY_OVERWRITE
    )
    trigger = independent_trigger_oracle(state, action)
    kernel_oracle = bool(np.asarray(inventory_overwrite_trigger(state, action)))
    if trigger != kernel_oracle:
        raise AssertionError("independent and kernel trigger predicates disagree")

    changed = _changed_fields(fixed, harmful)
    before_total = int(np.asarray(physical_grain_total(state)))
    fixed_total = int(np.asarray(physical_grain_total(fixed)))
    harmful_total = int(np.asarray(physical_grain_total(harmful)))
    carried = int(np.asarray(state.carried_grain))
    if trigger:
        if changed != ("carried_grain",):
            raise AssertionError(f"trigger changed unexpected fields: {changed}")
        if fixed_total != before_total:
            raise AssertionError("fixed harvest did not conserve physical grain")
        if before_total - harmful_total != carried:
            raise AssertionError("harmful harvest loss does not equal pre-harvest c")
        if int(np.asarray(fixed.carried_grain)) != carried + 1:
            raise AssertionError("fixed harvest did not increment inventory")
        if int(np.asarray(harmful.carried_grain)) != 1:
            raise AssertionError("harmful harvest did not overwrite inventory to one")
        if int(np.asarray(fixed.source_grain)) != int(
            np.asarray(state.source_grain)
        ) - 1:
            raise AssertionError("fixed source decrement is wrong")
        if int(np.asarray(harmful.source_grain)) != int(
            np.asarray(state.source_grain)
        ) - 1:
            raise AssertionError("harmful source decrement is wrong")
    elif changed:
        raise AssertionError(f"kernels differ outside B: {changed}")

    return {
        "trigger": trigger,
        "lost_grain": before_total - harmful_total,
    }


def validate_transition_locality() -> dict:
    """Enumerate trigger and non-trigger controls, including rebuild semantics."""

    checked = 0
    triggered = 0
    lost = 0
    period = TRAINING_SOURCE_GROWTH_PERIOD

    base = make_pack_restore_state(
        0,
        int(PackRestorePhase.EMPTY),
        split=PackRestoreSplit.VALIDATION,
        start=PackRestoreStart.PATH_CHECK,
        source_growth_period=period,
    )
    facing = pack_restore_step(
        base, int(PackRestoreAction.RIGHT), PackRestoreVariant.FIXED
    )
    for carried in range(4):
        for remaining in range(3):
            for age in (0, period - 1, period):
                for faces_source in (False, True):
                    state = facing.replace(
                        carried_grain=jnp.asarray(carried, dtype=jnp.int32),
                        source_grain=jnp.asarray(remaining, dtype=jnp.int32),
                        source_age=jnp.asarray(age, dtype=jnp.int32),
                        player_direction=(
                            facing.player_direction
                            if faces_source
                            else base.player_direction
                        ),
                    )
                    for action in PackRestoreAction:
                        row = _audit_pair(state, int(action))
                        checked += 1
                        triggered += int(row["trigger"])
                        lost += row["lost_grain"]

    rebuild = make_pack_restore_state(
        0,
        int(PackRestorePhase.LOADED),
        split=PackRestoreSplit.VALIDATION,
        start=PackRestoreStart.PATH_CHECK,
        source_growth_period=period,
    )
    for action in (PackRestoreAction.MAKE_RECORD, PackRestoreAction.PACK_STORAGE):
        rebuild = pack_restore_step(rebuild, int(action), PackRestoreVariant.FIXED)
    for action in PackRestoreAction:
        row = _audit_pair(rebuild, int(action))
        checked += 1
        triggered += int(row["trigger"])
        lost += row["lost_grain"]

    if triggered == 0 or lost == 0:
        raise AssertionError("locality enumeration did not exercise the defect")
    return {
        "passed": True,
        "enumerated_transitions": checked,
        "triggered_transitions": triggered,
        "outside_trigger_transitions": checked - triggered,
        "summed_trigger_loss": lost,
        "rebuild_actions_checked": len(PackRestoreAction),
        "oracle": "pre/post physical total and independent harvest preconditions",
    }


def validate_reference_paths() -> dict:
    """Replay one zero-loss delivery path from every natural validation start."""

    episodes = []
    for phase in PackRestorePhase:
        for layout in range(16):
            fixed = make_pack_restore_state(
                layout,
                int(phase),
                split=PackRestoreSplit.VALIDATION,
                start=PackRestoreStart.NATURAL,
                source_growth_period=TRAINING_SOURCE_GROWTH_PERIOD,
            )
            harmful = fixed
            if int(np.asarray(physical_grain_total(harmful))) != INITIAL_GRAIN_TOTAL:
                raise AssertionError("natural start does not contain exactly three grains")
            actions = inventory_safe_reference_actions(layout, int(phase))
            episode_loss = 0
            trigger_count = 0
            for action in actions:
                trigger_count += int(independent_trigger_oracle(harmful, action))
                before_total = int(np.asarray(physical_grain_total(harmful)))
                fixed = pack_restore_step(fixed, action, PackRestoreVariant.FIXED)
                harmful = pack_restore_step(
                    harmful, action, PackRestoreVariant.INVENTORY_OVERWRITE
                )
                after_total = int(np.asarray(physical_grain_total(harmful)))
                episode_loss += max(before_total - after_total, 0)
                if _changed_fields(fixed, harmful):
                    raise AssertionError("zero-loss reference path diverged across kernels")

            success = bool(np.asarray(pack_restore_goal_vector(harmful)[_DELIVER_3]))
            if not success or int(np.asarray(harmful.delivered_total)) != 3:
                raise AssertionError("reference path failed to deliver exactly three")
            if episode_loss or trigger_count:
                raise AssertionError("reference path entered the overwrite trigger")
            episodes.append(
                {
                    "layout": layout,
                    "phase": int(phase),
                    "actions": list(actions),
                    "length": len(actions),
                    "delivered": int(np.asarray(harmful.delivered_total)),
                    "lost_grain": episode_loss,
                    "trigger_count": trigger_count,
                    "success": success,
                }
            )

    return {
        "passed": True,
        "natural_starts": len(episodes),
        "successful_starts": sum(int(row["success"]) for row in episodes),
        "zero_loss_starts": sum(int(row["lost_grain"] == 0) for row in episodes),
        "max_path_length": max(row["length"] for row in episodes),
        "horizon": 128,
        "episodes": episodes,
        "use_restriction": "feasibility evidence only; excluded from initialization, data, reward, and optimization",
    }
