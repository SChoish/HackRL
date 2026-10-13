"""Vectorized kernel audit for stochastic PACK inventory overwrite events."""

from __future__ import annotations

from dataclasses import fields

import jax
import jax.numpy as jnp
import numpy as np

from hackrl.pack_inventory_overwrite_gate import independent_trigger_oracle
from hackrl.pack_restore import (
    TRAINING_SOURCE_GROWTH_PERIOD,
    PackRestoreAction,
    PackRestorePhase,
    PackRestoreSplit,
    PackRestoreStart,
    PackRestoreState,
    PackRestoreVariant,
    inventory_overwrite_trigger,
    make_pack_restore_state,
    pack_restore_step,
    pack_restore_step_inventory_overwrite_event,
    pack_restore_step_inventory_overwrite_stochastic,
    physical_grain_total,
)


def _enumerated_states() -> list[PackRestoreState]:
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
    states = []
    for carried in range(4):
        for remaining in range(3):
            for age in (0, period - 1, period):
                for faces_source in (False, True):
                    states.append(
                        facing.replace(
                            carried_grain=jnp.asarray(carried, dtype=jnp.int32),
                            source_grain=jnp.asarray(remaining, dtype=jnp.int32),
                            source_age=jnp.asarray(age, dtype=jnp.int32),
                            player_direction=(
                                facing.player_direction
                                if faces_source
                                else base.player_direction
                            ),
                        )
                    )

    rebuild = make_pack_restore_state(
        0,
        int(PackRestorePhase.LOADED),
        split=PackRestoreSplit.VALIDATION,
        start=PackRestoreStart.PATH_CHECK,
        source_growth_period=period,
    )
    for action in (PackRestoreAction.MAKE_RECORD, PackRestoreAction.PACK_STORAGE):
        rebuild = pack_restore_step(rebuild, int(action), PackRestoreVariant.FIXED)
    states.append(rebuild)
    return states


def _equal_rows(left: PackRestoreState, right: PackRestoreState, count: int):
    equal = np.ones((count,), dtype=bool)
    for field in fields(PackRestoreState):
        left_value = np.asarray(getattr(left, field.name))
        right_value = np.asarray(getattr(right, field.name))
        equal &= np.all(
            left_value.reshape(count, -1) == right_value.reshape(count, -1),
            axis=1,
        )
    return equal


def _batched_kernel(state_batch, action_batch, key_batch):
    fixed = jax.vmap(
        lambda state, action: pack_restore_step(
            state, action, PackRestoreVariant.FIXED
        )
    )(state_batch, action_batch)
    harmful = jax.vmap(
        lambda state, action: pack_restore_step(
            state, action, PackRestoreVariant.INVENTORY_OVERWRITE
        )
    )(state_batch, action_batch)
    forced_false, false_fired = jax.vmap(
        lambda state, action: pack_restore_step_inventory_overwrite_event(
            state, action, False
        )
    )(state_batch, action_batch)
    forced_true, true_fired = jax.vmap(
        lambda state, action: pack_restore_step_inventory_overwrite_event(
            state, action, True
        )
    )(state_batch, action_batch)

    def stochastic(probability):
        return jax.vmap(
            lambda state, action, key: (
                pack_restore_step_inventory_overwrite_stochastic(
                    state, action, key, probability
                )
            )
        )(state_batch, action_batch, key_batch)

    probability_zero, zero_fired = stochastic(0.0)
    probability_one, one_fired = stochastic(1.0)
    first, first_fired = stochastic(0.37)
    second, second_fired = stochastic(0.37)
    kernel_trigger = jax.vmap(inventory_overwrite_trigger)(state_batch, action_batch)
    before_total = jax.vmap(physical_grain_total)(state_batch)
    after_total = jax.vmap(physical_grain_total)(forced_true)
    return (
        fixed,
        harmful,
        forced_false,
        false_fired,
        forced_true,
        true_fired,
        probability_zero,
        zero_fired,
        probability_one,
        one_fired,
        first,
        first_fired,
        second,
        second_fired,
        kernel_trigger,
        before_total,
        after_total,
    )


def validate_stochastic_overwrite_kernel_vectorized() -> dict:
    """Audit the full declared transition panel in one compiled batch."""

    states = _enumerated_states()
    action_values = np.tile(
        np.asarray([int(action) for action in PackRestoreAction], dtype=np.int32),
        len(states),
    )
    expanded_states = [state for state in states for _ in PackRestoreAction]
    checked = len(expanded_states)
    batched_states = jax.tree.map(lambda *values: jnp.stack(values), *expanded_states)
    batched_actions = jnp.asarray(action_values)
    state_indices = np.repeat(
        np.arange(len(states), dtype=np.int32), len(PackRestoreAction)
    )
    key_seeds = 100_000 + 31 * state_indices + action_values
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(key_seeds))

    outputs = jax.device_get(
        jax.jit(_batched_kernel)(batched_states, batched_actions, keys)
    )
    (
        fixed,
        harmful,
        forced_false,
        false_fired,
        forced_true,
        true_fired,
        probability_zero,
        zero_fired,
        probability_one,
        one_fired,
        first,
        first_fired,
        second,
        second_fired,
        kernel_trigger,
        before_total,
        after_total,
    ) = outputs

    host_states = jax.device_get(batched_states)
    independent = np.asarray(
        [
            independent_trigger_oracle(
                jax.tree.map(lambda value: value[index], host_states), int(action)
            )
            for index, action in enumerate(action_values)
        ],
        dtype=bool,
    )
    expected_true = jax.tree.map(
        lambda harmful_value, fixed_value: np.where(
            independent.reshape((checked,) + (1,) * (harmful_value.ndim - 1)),
            harmful_value,
            fixed_value,
        ),
        harmful,
        fixed,
    )

    mismatches = {
        "oracle": int(np.sum(independent != np.asarray(kernel_trigger))),
        "forced_false": int(
            np.sum(np.asarray(false_fired) | ~_equal_rows(forced_false, fixed, checked))
        ),
        "forced_true": int(
            np.sum(
                (np.asarray(true_fired) != independent)
                | ~_equal_rows(forced_true, expected_true, checked)
            )
        ),
        "probability_zero": int(
            np.sum(
                np.asarray(zero_fired)
                | ~_equal_rows(probability_zero, fixed, checked)
            )
        ),
        "probability_one": int(
            np.sum(
                (np.asarray(one_fired) != independent)
                | ~_equal_rows(probability_one, expected_true, checked)
            )
        ),
        "same_key_reproducibility": int(
            np.sum(
                (np.asarray(first_fired) != np.asarray(second_fired))
                | ~_equal_rows(first, second, checked)
            )
        ),
    }
    losses = np.asarray(before_total) - np.asarray(after_total)
    carried = np.asarray(batched_states.carried_grain)
    if np.any(losses[independent] != carried[independent]):
        raise AssertionError("forced overwrite loss is not pre-harvest c")
    triggered = int(np.sum(independent))
    summed_trigger_loss = int(np.sum(losses[independent]))
    if triggered == 0 or summed_trigger_loss == 0 or any(mismatches.values()):
        raise AssertionError(f"stochastic overwrite validation failed: {mismatches}")
    return {
        "passed": True,
        "execution": "one jitted vectorized batch",
        "enumerated_states": len(states),
        "enumerated_transitions": checked,
        "triggered_transitions": triggered,
        "outside_trigger_transitions": checked - triggered,
        "summed_trigger_loss": summed_trigger_loss,
        "mismatches": mismatches,
        "rng_check": "same explicit JAX key at p=0.37 gives the same event and state",
        "probability_claim": (
            "Only exact p=0 and p=1 endpoints are tested; no empirical "
            "calibration claim is made."
        ),
    }
