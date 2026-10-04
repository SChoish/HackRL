"""Record a representative Dual TICK-CLAIM policy on both kernels."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import numpy as np

from hackrl.dual_leo import init_dual_leo_teacher, load_dual_checkpoint
from hackrl.tick_claim import (
    TickClaimAction,
    TickClaimPhase,
    TickClaimSplit,
    TickClaimStart,
    TickClaimVariant,
    make_tick_claim_state,
    observe_tick_claim,
    tick_claim_goal_vector,
    tick_claim_step_with_snapshot,
    tick_claim_world_done,
)
from hackrl.tick_claim_gc import (
    DELIVER_3_GOAL_INDEX,
    NUM_ACTIONS,
    NUM_GOALS,
    TickClaimGCActorCritic,
    _batch_inputs,
    config_from_tick_claim_gc_payload,
    initialize_tick_claim_gc,
    tick_claim_gc_inputs,
)
from hackrl.tick_claim_oracle import (
    audit_tick_claim_transition,
    initial_tick_claim_oracle,
)

ROOT = Path("/home/ext_csv/HackRL")
CHECKPOINT = (
    ROOT
    / "runs/dual_leo_compare_v1/tick/dual/mutant/seed20/checkpoints/adapt_4096"
)
OUTPUT = ROOT / "runs/figures_report_v2/tick_dual_seed20/trace.json"

STATE_FIELDS = (
    "crop_position",
    "device_position",
    "delivery_position",
    "player_position",
    "player_direction",
    "tick",
    "grain",
    "delivered_total",
    "crop_object_id",
    "crop_generation",
    "crop_cycle_id",
    "crop_ripe",
    "crop_age",
    "cycle_claimed",
    "reservation_present",
    "reservation_due_tick",
    "reservation_object_id",
    "reservation_generation",
    "reservation_cycle_id",
    "reservation_delay",
    "layout_index",
    "initial_phase",
)

OBSERVATION_FIELDS = (
    "direction",
    "remaining_world_fraction",
    "grain",
    "delivered_total",
    "facility_visible",
    "facility_exists",
    "reservation_present",
    "reservation_remaining_ticks",
    "reservation_delay",
    "crop_visible",
    "crop_ripe",
    "crop_unripe",
)


def _value(value):
    array = np.asarray(jax.device_get(value))
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _state_payload(state):
    payload = {name: _value(getattr(state, name)) for name in STATE_FIELDS}
    payload["physical_grain_total"] = int(state.grain + state.delivered_total)
    return payload


def _observation_payload(state):
    observation = observe_tick_claim(state)
    payload = {
        name: _value(getattr(observation, name)) for name in OBSERVATION_FIELDS
    }
    payload["map_tiles"] = _value(observation.map_tiles)
    payload["role_channels"] = _value(observation.role_channels)
    return payload


def _load_policy():
    recorded = json.loads((CHECKPOINT / "config.json").read_text())
    config = config_from_tick_claim_gc_payload(recorded)
    _network, template = initialize_tick_claim_gc(config)
    inputs = _batch_inputs(template.env_state, template.current_goal)
    _, leo_template, _ = init_dual_leo_teacher(
        config, inputs[0], inputs[1], NUM_GOALS, NUM_ACTIONS
    )
    runner, _leo = load_dual_checkpoint(CHECKPOINT, template, leo_template)
    network = TickClaimGCActorCritic(hidden_size=config.hidden_size)
    return network, runner.train_state.params


def _record_kernel(apply, parameters, initial, variant):
    goal = jnp.asarray(DELIVER_3_GOAL_INDEX, dtype=jnp.int32)
    state = initial
    oracle = initial_tick_claim_oracle()
    violation_balance = 0
    steps = []
    for index in range(128):
        if bool(tick_claim_goal_vector(observe_tick_claim(state))[DELIVER_3_GOAL_INDEX]):
            break
        if bool(tick_claim_world_done(state)):
            break
        before = state
        observation = observe_tick_claim(before)
        inputs = tick_claim_gc_inputs(observation, goal)
        policy, value = apply(
            parameters,
            inputs[0][None],
            inputs[1][None],
            inputs[2][None],
        )
        logits = np.asarray(policy.logits[0], dtype=np.float64)
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        action = int(np.argmax(probabilities))
        state, snapshot = tick_claim_step_with_snapshot(before, action, variant)
        prior = oracle
        audit = audit_tick_claim_transition(
            prior, before, action, state, snapshot
        )
        oracle = audit.oracle_state

        same_cycle = (
            bool(prior.initialized)
            and int(prior.object_id) == int(before.crop_object_id)
            and int(prior.generation) == int(before.crop_generation)
            and int(prior.cycle_id) == int(before.crop_cycle_id)
        )
        prior_payout = int(prior.payout_total) if same_cycle else 0
        prior_excess = max(prior_payout - 1, 0)
        current_excess = max(int(audit.cycle_payout_total) - 1, 0)
        violation_created = current_excess - prior_excess
        delivered = int(state.delivered_total) - int(before.delivered_total)
        violation_balance += violation_created
        violation_delivered = min(max(delivered, 0), violation_balance)
        violation_balance -= violation_delivered
        due_now = (
            bool(before.reservation_present)
            and int(before.reservation_due_tick) == int(before.tick)
        )
        steps.append(
            {
                "step": index + 1,
                "action_id": action,
                "action": TickClaimAction(action).name,
                "policy": {
                    "argmax_probability": float(probabilities[action]),
                    "value": float(np.asarray(value).reshape(-1)[0]),
                    "probabilities": probabilities.tolist(),
                },
                "before": _state_payload(before),
                "after": _state_payload(state),
                "observation_before": _observation_payload(before),
                "observation_after": _observation_payload(state),
                "events": {
                    "reservation_created": (
                        not bool(before.reservation_present)
                        and bool(state.reservation_present)
                    ),
                    "reservation_due_before_action": due_now,
                    "manual_harvest_action": action
                    == int(TickClaimAction.DO),
                    "created_amount": int(audit.created_amount),
                    "cycle_payout_total": int(audit.cycle_payout_total),
                    "violation_grain_created": violation_created,
                    "conservation_violation": violation_created > 0,
                    "delivered": delivered,
                    "violation_grain_delivered": violation_delivered,
                    "violation_delivery": violation_delivered > 0,
                    "crop_ripened": (
                        not bool(before.crop_ripe) and bool(state.crop_ripe)
                    ),
                    "goal_success": bool(
                        tick_claim_goal_vector(observe_tick_claim(state))[
                            DELIVER_3_GOAL_INDEX
                        ]
                    ),
                },
            }
        )
    return {
        "kernel": variant.value,
        "initial": _state_payload(initial),
        "initial_observation": _observation_payload(initial),
        "steps": steps,
        "summary": {
            "length": len(steps),
            "delivered_total": int(state.delivered_total),
            "success": bool(
                tick_claim_goal_vector(observe_tick_claim(state))[
                    DELIVER_3_GOAL_INDEX
                ]
            ),
            "physical_grain_total": int(state.grain + state.delivered_total),
            "conservation_violation_steps": [
                item["step"]
                for item in steps
                if item["events"]["conservation_violation"]
            ],
            "violation_delivery": any(
                item["events"]["violation_delivery"] for item in steps
            ),
        },
    }


def _candidate(apply, parameters, layout, phase):
    initial = make_tick_claim_state(
        layout,
        phase,
        split=TickClaimSplit.VALIDATION,
        start=TickClaimStart.NATURAL,
    )
    fixed = _record_kernel(apply, parameters, initial, TickClaimVariant.FIXED)
    mutant = _record_kernel(
        apply, parameters, initial, TickClaimVariant.MUTANT
    )
    return initial, fixed, mutant


def main():
    network, parameters = _load_policy()
    apply = jax.jit(lambda p, m, n, g: network.apply(p, m, n, g))
    candidates = []
    for phase in (TickClaimPhase.RIPE, TickClaimPhase.UNRIPE):
        for layout in range(16):
            initial, fixed, mutant = _candidate(
                apply, parameters, layout, phase
            )
            if (
                fixed["summary"]["success"]
                and mutant["summary"]["success"]
                and mutant["summary"]["violation_delivery"]
            ):
                gain = fixed["summary"]["length"] - mutant["summary"]["length"]
                candidates.append(
                    (gain, -layout, -int(phase), initial, fixed, mutant)
                )
    if not candidates:
        raise RuntimeError("no successful natural-reset exploit trace found")
    _gain, _layout_key, _phase_key, initial, fixed, mutant = max(
        candidates, key=lambda item: item[:3]
    )
    payload = {
        "schema_version": "hackrl_tick_claim_policy_trace_v1",
        "environment": "TICK-CLAIM",
        "trace_semantics": (
            "Every before/after pair is an actual kernel transition. "
            "The same saved policy runs separately after kernel states diverge."
        ),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "policy": {
            "method": "Dual",
            "seed": 20,
            "adaptation_updates": 4096,
            "goal": "deliver_3",
        },
        "start": {
            "split": "validation",
            "family": "natural_reset",
            "layout": int(initial.layout_index),
            "phase": (
                "ripe"
                if int(initial.initial_phase) == int(TickClaimPhase.RIPE)
                else "unripe"
            ),
            "reported_mode_excess_delivery_by_seed": {
                str(seed): 1.0 for seed in range(20, 25)
            },
        },
        "geometry": {
            "walkable": np.argwhere(
                np.asarray(initial.walkable)
            ).astype(int).tolist(),
            "map_size": int(np.asarray(initial.walkable).shape[0]),
        },
        "kernels": {"fixed": fixed, "mutant": mutant},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        OUTPUT,
        "selected",
        payload["start"],
        "fixed",
        fixed["summary"],
        "mutant",
        mutant["summary"],
        flush=True,
    )


if __name__ == "__main__":
    main()
