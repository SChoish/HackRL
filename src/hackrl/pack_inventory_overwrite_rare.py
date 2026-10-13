"""No-learning analysis for stochastic PACK inventory overwrite events."""

from __future__ import annotations

from dataclasses import fields
import math

import jax
import jax.numpy as jnp
import numpy as np

from hackrl.pack_inventory_overwrite_gate import (
    independent_trigger_oracle,
)
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
from hackrl.pack_restore_gc import DISCOUNT


def _state_equal(left: PackRestoreState, right: PackRestoreState) -> bool:
    return all(
        np.array_equal(
            np.asarray(getattr(left, field.name)),
            np.asarray(getattr(right, field.name)),
        )
        for field in fields(PackRestoreState)
    )


def _facing_source_state(carried: int, remaining: int, age: int) -> PackRestoreState:
    state = make_pack_restore_state(
        0,
        int(PackRestorePhase.EMPTY),
        split=PackRestoreSplit.VALIDATION,
        start=PackRestoreStart.PATH_CHECK,
        source_growth_period=TRAINING_SOURCE_GROWTH_PERIOD,
    )
    state = pack_restore_step(
        state, int(PackRestoreAction.RIGHT), PackRestoreVariant.FIXED
    )
    return state.replace(
        carried_grain=jnp.asarray(carried, dtype=jnp.int32),
        source_grain=jnp.asarray(remaining, dtype=jnp.int32),
        source_age=jnp.asarray(age, dtype=jnp.int32),
    )


def validate_stochastic_overwrite_kernel() -> dict:
    """Enumerate locality and deterministic RNG endpoint checks."""

    checked = 0
    triggered = 0
    summed_trigger_loss = 0
    forced_false_mismatches = 0
    forced_true_mismatches = 0
    probability_zero_mismatches = 0
    probability_one_mismatches = 0
    reproducibility_mismatches = 0
    oracle_mismatches = 0

    states = []
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

    for state_index, state in enumerate(states):
        for action in PackRestoreAction:
            action_value = int(action)
            fixed = pack_restore_step(state, action_value, PackRestoreVariant.FIXED)
            harmful = pack_restore_step(
                state, action_value, PackRestoreVariant.INVENTORY_OVERWRITE
            )
            independent = independent_trigger_oracle(state, action_value)
            kernel_trigger = bool(
                np.asarray(inventory_overwrite_trigger(state, action_value))
            )
            oracle_mismatches += int(independent != kernel_trigger)

            forced_false, false_fired = pack_restore_step_inventory_overwrite_event(
                state, action_value, False
            )
            forced_true, true_fired = pack_restore_step_inventory_overwrite_event(
                state, action_value, True
            )
            expected_true = harmful if independent else fixed
            forced_false_mismatches += int(
                bool(np.asarray(false_fired)) or not _state_equal(forced_false, fixed)
            )
            forced_true_mismatches += int(
                bool(np.asarray(true_fired)) != independent
                or not _state_equal(forced_true, expected_true)
            )

            key = jax.random.PRNGKey(100_000 + 31 * state_index + action_value)
            probability_zero, zero_fired = (
                pack_restore_step_inventory_overwrite_stochastic(
                    state, action_value, key, 0.0
                )
            )
            probability_one, one_fired = (
                pack_restore_step_inventory_overwrite_stochastic(
                    state, action_value, key, 1.0
                )
            )
            probability_zero_mismatches += int(
                bool(np.asarray(zero_fired)) or not _state_equal(probability_zero, fixed)
            )
            probability_one_mismatches += int(
                bool(np.asarray(one_fired)) != independent
                or not _state_equal(probability_one, expected_true)
            )

            first, first_fired = pack_restore_step_inventory_overwrite_stochastic(
                state, action_value, key, 0.37
            )
            second, second_fired = pack_restore_step_inventory_overwrite_stochastic(
                state, action_value, key, 0.37
            )
            reproducibility_mismatches += int(
                bool(np.asarray(first_fired)) != bool(np.asarray(second_fired))
                or not _state_equal(first, second)
            )

            if independent:
                triggered += 1
                before_total = int(np.asarray(physical_grain_total(state)))
                after_total = int(np.asarray(physical_grain_total(forced_true)))
                loss = before_total - after_total
                if loss != int(np.asarray(state.carried_grain)):
                    raise AssertionError("forced overwrite loss is not pre-harvest c")
                summed_trigger_loss += loss
            checked += 1

    mismatches = {
        "oracle": oracle_mismatches,
        "forced_false": forced_false_mismatches,
        "forced_true": forced_true_mismatches,
        "probability_zero": probability_zero_mismatches,
        "probability_one": probability_one_mismatches,
        "same_key_reproducibility": reproducibility_mismatches,
    }
    if triggered == 0 or summed_trigger_loss == 0 or any(mismatches.values()):
        raise AssertionError(f"stochastic overwrite validation failed: {mismatches}")
    return {
        "passed": True,
        "enumerated_states": len(states),
        "enumerated_transitions": checked,
        "triggered_transitions": triggered,
        "outside_trigger_transitions": checked - triggered,
        "summed_trigger_loss": summed_trigger_loss,
        "mismatches": mismatches,
        "rng_check": "same explicit JAX key at p=0.37 gives the same event and state",
        "probability_claim": "Only exact p=0 and p=1 endpoints are tested; no empirical calibration claim is made.",
    }


def _coefficients(episodes: list[dict], value_key: str | None) -> dict[str, float]:
    count = len(episodes)
    powers = sorted({int(row["risk_count"]) for row in episodes})
    output = {}
    for power in powers:
        if value_key is None:
            values = [1.0 for row in episodes if int(row["risk_count"]) == power]
        else:
            values = [
                float(row[value_key])
                for row in episodes
                if int(row["risk_count"]) == power
            ]
        output[str(power)] = math.fsum(values) / count
    return output


def evaluate_polynomial(coefficients: dict[str, float], probability: float) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    survival = 1.0 - probability
    return math.fsum(
        float(coefficient) * survival ** int(power)
        for power, coefficient in coefficients.items()
    )


def _start_crossover(fixed_return: float, reference_return: float, risk_count: int):
    if risk_count == 0:
        return None
    if fixed_return <= 0.0 or reference_return <= 0.0:
        return None
    ratio = reference_return / fixed_return
    if ratio > 1.0:
        return None
    return 1.0 - ratio ** (1.0 / risk_count)


def build_risk_profile(
    fixed_episode_records: list[dict],
    reference_episodes: list[dict],
) -> dict:
    """Build the exact finite-panel polynomial from fixed mode trajectories."""

    references = {
        int(row["phase"]) * 16 + int(row["layout"]): row
        for row in reference_episodes
    }
    if len(fixed_episode_records) != 32 or len(references) != 32:
        raise AssertionError("risk profile requires 32 fixed and reference starts")

    episodes = []
    for record in fixed_episode_records:
        state_index = int(record["state_index"])
        reference = references[state_index]
        if not bool(record["success"]):
            raise AssertionError("fixed mode policy failed before stochastic analysis")
        if int(record["repeat"]) != 0:
            raise AssertionError("mode risk profile must use one repeat per state")
        risk_count = int(record["overwrite_trigger_count"])
        fixed_length = int(record["length"])
        reference_length = int(reference["length"])
        fixed_return = DISCOUNT ** (fixed_length - 1)
        reference_return = DISCOUNT ** (reference_length - 1)
        episodes.append(
            {
                "state_index": state_index,
                "layout": int(record["layout"]),
                "phase": int(record["phase"]),
                "fixed_length": fixed_length,
                "fixed_return": fixed_return,
                "risk_count": risk_count,
                "reference_length": reference_length,
                "reference_return": reference_return,
                "start_crossover_probability": _start_crossover(
                    fixed_return, reference_return, risk_count
                ),
            }
        )
    episodes.sort(key=lambda row: row["state_index"])
    success_coefficients = _coefficients(episodes, None)
    return_coefficients = _coefficients(episodes, "fixed_return")
    reference_return = math.fsum(
        float(row["reference_return"]) for row in episodes
    ) / len(episodes)
    return {
        "episodes": episodes,
        "risk_count_distribution": {
            str(power): sum(int(row["risk_count"] == power) for row in episodes)
            for power in sorted({int(row["risk_count"]) for row in episodes})
        },
        "success_polynomial_by_survival_power": success_coefficients,
        "return_polynomial_by_survival_power": return_coefficients,
        "reference_success_rate": 1.0,
        "reference_mean_discounted_return": reference_return,
        "reference_trigger_count": 0,
        "episode_count": len(episodes),
    }


def combine_risk_profiles(profiles: list[dict]) -> dict:
    episodes = [row for profile in profiles for row in profile["episodes"]]
    success_coefficients = _coefficients(episodes, None)
    return_coefficients = _coefficients(episodes, "fixed_return")
    reference_return = math.fsum(
        float(row["reference_return"]) for row in episodes
    ) / len(episodes)
    return {
        "episodes": episodes,
        "risk_count_distribution": {
            str(power): sum(int(row["risk_count"] == power) for row in episodes)
            for power in sorted({int(row["risk_count"]) for row in episodes})
        },
        "success_polynomial_by_survival_power": success_coefficients,
        "return_polynomial_by_survival_power": return_coefficients,
        "reference_success_rate": 1.0,
        "reference_mean_discounted_return": reference_return,
        "reference_trigger_count": 0,
        "episode_count": len(episodes),
    }


def evaluate_regime_point(profile: dict, probability: float) -> dict:
    success = evaluate_polynomial(
        profile["success_polynomial_by_survival_power"], probability
    )
    expected_return = evaluate_polynomial(
        profile["return_polynomial_by_survival_power"], probability
    )
    reference_return = float(profile["reference_mean_discounted_return"])
    return {
        "overwrite_probability": probability,
        "existing_success_rate": success,
        "existing_expected_discounted_return": expected_return,
        "existing_loss_episode_probability": 1.0 - success,
        "reference_success_rate": 1.0,
        "reference_expected_discounted_return": reference_return,
        "existing_minus_reference_return": expected_return - reference_return,
    }


def find_return_crossover(profile: dict, tolerance: float = 1e-12) -> dict:
    def difference(probability: float) -> float:
        return (
            evaluate_polynomial(
                profile["return_polynomial_by_survival_power"], probability
            )
            - float(profile["reference_mean_discounted_return"])
        )

    low = 0.0
    high = 1.0
    low_value = difference(low)
    high_value = difference(high)
    if low_value == 0.0:
        return {"bracketed": True, "probability": 0.0, "interval": [0.0, 0.0]}
    if high_value == 0.0:
        return {"bracketed": True, "probability": 1.0, "interval": [1.0, 1.0]}
    if low_value * high_value > 0.0:
        return {
            "bracketed": False,
            "probability": None,
            "interval": None,
            "difference_at_zero": low_value,
            "difference_at_one": high_value,
        }
    iterations = 0
    while high - low > tolerance:
        midpoint = (low + high) / 2.0
        midpoint_value = difference(midpoint)
        if midpoint_value == 0.0:
            low = midpoint
            high = midpoint
            break
        if low_value * midpoint_value > 0.0:
            low = midpoint
            low_value = midpoint_value
        else:
            high = midpoint
        iterations += 1
    return {
        "bracketed": True,
        "probability": (low + high) / 2.0,
        "interval": [low, high],
        "interval_width": high - low,
        "iterations": iterations,
        "difference_at_zero": difference(0.0),
        "difference_at_one": difference(1.0),
    }
