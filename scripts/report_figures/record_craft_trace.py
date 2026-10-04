"""Record a selected maximum-gain Dual CRAFT-REMAIN example on both kernels."""

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

from hackrl.craft_remain_env import (
    DELIVERY,
    GRID,
    SOURCE,
    CraftRemainAction,
    CraftRemainPhase,
    CraftRemainStart,
    CraftRemainVariant,
    craft_remain_goal_vector,
    craft_remain_step,
    craft_remain_world_done,
    make_craft_remain_state,
    observe_craft_remain,
    physical_total,
    transition_oracle,
)
from hackrl.craft_remain_gc import (
    DELIVER_3_GOAL_INDEX,
    NUM_ACTIONS,
    NUM_GOALS,
    CraftRemainGCActorCritic,
    _batch_inputs,
    config_from_craft_remain_gc_payload,
    craft_remain_gc_inputs,
    initialize_craft_remain_gc,
)
from hackrl.dual_leo import init_dual_leo_teacher, load_dual_checkpoint

from report_figures.common import ROOT, provenance
CHECKPOINT = (
    ROOT
    / "runs/craft_remain_compare_v1/dual/mutant/seed21/checkpoints/adapt_4096"
)
OUTPUT = ROOT / "runs/figures_report_v2/craft_dual_seed21/trace.json"

STATE_FIELDS = (
    "player_position",
    "player_direction",
    "tick",
    "carried_grain",
    "slot_a",
    "slot_b",
    "output_present",
    "held_parcel",
    "source_grain",
    "source_age",
    "delivered_total",
    "harvested_count",
    "episode_triggered",
    "retained_taken",
)

OBSERVATION_FIELDS = (
    "direction",
    "remaining_world_fraction",
    "grain",
    "delivered_total",
    "grid_visible",
    "slot_a",
    "slot_b",
    "output_present",
    "held_parcel",
    "source_available",
)


def _value(value):
    array = np.asarray(jax.device_get(value))
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _state_payload(state):
    payload = {name: _value(getattr(state, name)) for name in STATE_FIELDS}
    payload["physical_total"] = int(physical_total(state))
    return payload


def _observation_payload(state):
    observation = observe_craft_remain(state)
    payload = {
        name: _value(getattr(observation, name)) for name in OBSERVATION_FIELDS
    }
    payload["map_tiles"] = _value(observation.map_tiles)
    payload["role_channels"] = _value(observation.role_channels)
    return payload


def _load_policy():
    recorded = json.loads((CHECKPOINT / "config.json").read_text())
    config = config_from_craft_remain_gc_payload(recorded)
    _network, template = initialize_craft_remain_gc(config)
    inputs = _batch_inputs(template.env_state, template.current_goal)
    _, leo_template, _ = init_dual_leo_teacher(
        config, inputs[0], inputs[1], NUM_GOALS, NUM_ACTIONS
    )
    runner, _leo = load_dual_checkpoint(CHECKPOINT, template, leo_template)
    network = CraftRemainGCActorCritic(hidden_size=config.hidden_size)
    return network, runner.train_state.params


def _record_kernel(apply, parameters, initial, variant, *, max_steps=128):
    goal = jnp.asarray(DELIVER_3_GOAL_INDEX, dtype=jnp.int32)
    state = initial
    steps = []
    for index in range(max_steps):
        if bool(craft_remain_goal_vector(state)[DELIVER_3_GOAL_INDEX]):
            break
        if bool(craft_remain_world_done(state)):
            break
        before = state
        observation = observe_craft_remain(before)
        inputs = craft_remain_gc_inputs(observation, goal)
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
        state = craft_remain_step(before, action, variant)
        oracle = {
            key: _value(value)
            for key, value in transition_oracle(before, state).items()
        }
        steps.append(
            {
                "step": index + 1,
                "action_id": action,
                "action": CraftRemainAction(action).name,
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
                    "harvested": int(state.harvested_count)
                    > int(before.harvested_count),
                    "filled_a": int(state.slot_a) > int(before.slot_a)
                    and action == int(CraftRemainAction.FILL_A),
                    "filled_b": int(state.slot_b) > int(before.slot_b)
                    and action == int(CraftRemainAction.FILL_B),
                    "crafted": bool(state.output_present)
                    and not bool(before.output_present),
                    "triggered": bool(oracle["triggered"]),
                    "conservation_violation_amount": int(
                        oracle["conservation_violation_amount"]
                    ),
                    "retained_recovered": bool(oracle["retained_recovered"]),
                    "output_taken": bool(before.output_present)
                    and not bool(state.output_present)
                    and bool(state.held_parcel),
                    "delivered": int(state.delivered_total)
                    - int(before.delivered_total),
                    "excess_delivery": bool(oracle["excess_delivery"]),
                    "goal_success": bool(
                        craft_remain_goal_vector(state)[DELIVER_3_GOAL_INDEX]
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
                craft_remain_goal_vector(state)[DELIVER_3_GOAL_INDEX]
            ),
            "physical_total": int(physical_total(state)),
            "triggered": bool(state.episode_triggered),
            "retained_taken": int(state.retained_taken),
            "exploit": (
                bool(state.episode_triggered)
                and int(state.retained_taken) >= 1
                and int(state.delivered_total) >= 3
            ),
            "conservation_violation_steps": [
                item["step"]
                for item in steps
                if item["events"]["conservation_violation_amount"] > 0
            ],
        },
    }


def main():
    network, parameters = _load_policy()
    apply = jax.jit(lambda p, m, n, g: network.apply(p, m, n, g))
    candidates = []
    for phase in (CraftRemainPhase.RIPE, CraftRemainPhase.UNRIPE):
        initial = make_craft_remain_state(
            phase, CraftRemainStart.NATURAL
        )
        fixed = _record_kernel(
            apply, parameters, initial, CraftRemainVariant.FIXED
        )
        mutant = _record_kernel(
            apply, parameters, initial, CraftRemainVariant.MUTANT
        )
        if (
            fixed["summary"]["success"]
            and mutant["summary"]["success"]
            and mutant["summary"]["exploit"]
        ):
            gain = fixed["summary"]["length"] - mutant["summary"]["length"]
            candidates.append((gain, -int(phase), initial, fixed, mutant))
    if not candidates:
        raise RuntimeError("no successful natural-reset exploit trace found")
    _gain, _phase_key, initial, fixed, mutant = max(
        candidates, key=lambda item: item[:2]
    )
    payload = {
        "schema_version": "hackrl_craft_remain_policy_trace_v1",
        "provenance": provenance(CHECKPOINT),
        "environment": "CRAFT-REMAIN",
        "trace_semantics": (
            "Every before/after pair is an actual kernel transition. "
            "The same saved policy runs separately after kernel states diverge."
        ),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "policy": {
            "method": "Dual",
            "seed": 21,
            "adaptation_updates": 4096,
            "goal": "deliver_3",
        },
        "selection": {
            "rule": "max length gain among successful exploit pairs",
            "candidate_count": 2,
            "eligible_count": len(candidates),
            "selected_length_gain": int(_gain),
            "tie_break": "lower phase",
            "purpose": "illustrative selected example, not an unbiased estimate",
        },
        "start": {
            "family": "natural_reset",
            "phase": (
                "ripe"
                if int(initial.source_grain) == 1
                else "unripe"
            ),
            "growth_period": 16,
        },
        "geometry": {
            "map_size": 16,
            "walkable": [
                [row, col] for row in range(16) for col in range(16)
                if (row, col) not in {tuple(_value(p)) for p in (SOURCE, GRID, DELIVERY)}
            ],
            "source": _value(SOURCE),
            "grid": _value(GRID),
            "delivery": _value(DELIVERY),
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
