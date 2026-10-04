"""Record one saved PACK-RESTORE policy rollout without rendering it.

The JSON is the source for both presentation and analysis renderers. Every
stored frame is an actual pre- or post-transition state; renderers must not
advance the environment or invent intermediate states.
"""

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
from hackrl.pack_restore import (
    PackRestoreAction,
    PackRestorePhase,
    PackRestoreSplit,
    PackRestoreStart,
    PackRestoreVariant,
    make_pack_restore_state,
    observe_pack_restore,
    pack_restore_goal_vector,
    pack_restore_step,
    pack_restore_world_done,
    physical_grain_total,
)
from hackrl.pack_restore_gc import (
    DELIVER_3_GOAL_INDEX,
    NUM_ACTIONS,
    NUM_GOALS,
    PackRestoreGCActorCritic,
    _batch_inputs,
    config_from_pack_restore_gc_payload,
    initialize_pack_restore_gc,
    pack_restore_gc_inputs,
)

from report_figures.common import ROOT, provenance
CHECKPOINT = (
    ROOT
    / "runs/dual_leo_compare_v1/pack/dual/mutant/seed20/checkpoints/adapt_4096"
)
OUTPUT = (
    ROOT
    / "runs/figures_report_v2/pack_dual_seed20/trace.json"
)

STATE_FIELDS = (
    "anchor_position",
    "source_position",
    "delivery_position",
    "unpack_position",
    "player_position",
    "player_direction",
    "tick",
    "carried_grain",
    "source_grain",
    "source_age",
    "source_growth_period",
    "anchor_present",
    "anchor_grain",
    "unpack_present",
    "unpack_grain",
    "empty_frames",
    "packed_present",
    "packed_grain",
    "record_present",
    "record_grain_preview",
    "delivered_total",
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
    "storage_grain",
    "empty_frames",
    "packed_items_count",
    "packed_grain_total",
    "structure_record_present",
    "record_grain_preview",
    "source_ripe",
    "source_age",
    "source_growth_period",
)


def _value(value):
    array = np.asarray(jax.device_get(value))
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _state_payload(state):
    payload = {name: _value(getattr(state, name)) for name in STATE_FIELDS}
    payload["physical_grain_total"] = int(physical_grain_total(state))
    return payload


def _observation_payload(state):
    observation = observe_pack_restore(state)
    payload = {
        name: _value(getattr(observation, name)) for name in OBSERVATION_FIELDS
    }
    payload["map_tiles"] = _value(observation.map_tiles)
    payload["role_channels"] = _value(observation.role_channels)
    return payload


def _load_policy():
    recorded = json.loads((CHECKPOINT / "config.json").read_text())
    config = config_from_pack_restore_gc_payload(recorded)
    _network, template = initialize_pack_restore_gc(config)
    inputs = _batch_inputs(template.env_state, template.current_goal)
    _, leo_template, _ = init_dual_leo_teacher(
        config, inputs[0], inputs[1], NUM_GOALS, NUM_ACTIONS
    )
    runner, _leo = load_dual_checkpoint(CHECKPOINT, template, leo_template)
    network = PackRestoreGCActorCritic(hidden_size=config.policy_hidden_size)
    return network, runner.train_state.params


def _record_kernel(network, parameters, initial, variant, *, max_steps=128):
    apply = jax.jit(lambda p, m, n, g: network.apply(p, m, n, g))
    goal = jnp.asarray(DELIVER_3_GOAL_INDEX, dtype=jnp.int32)
    state = initial
    steps = []
    violation_balance = 0
    for index in range(max_steps):
        if bool(pack_restore_goal_vector(state)[DELIVER_3_GOAL_INDEX]):
            break
        if bool(pack_restore_world_done(state)):
            break
        before = state
        observation = observe_pack_restore(before)
        inputs = pack_restore_gc_inputs(observation, goal)
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
        state = pack_restore_step(before, action, variant)

        physical_before = int(physical_grain_total(before))
        physical_after = int(physical_grain_total(state))
        created = max(physical_after - physical_before, 0)
        delivered = int(state.delivered_total) - int(before.delivered_total)
        violation_balance += created
        violation_delivered = min(max(delivered, 0), violation_balance)
        violation_balance -= violation_delivered

        record_created = (
            not bool(before.record_present) and bool(state.record_present)
        )
        storage_packed = (
            action == int(PackRestoreAction.PACK_STORAGE)
            and bool(before.anchor_present)
            and not bool(state.anchor_present)
        )
        storage_rebuilt = (
            not bool(before.anchor_present) and bool(state.anchor_present)
        )
        steps.append(
            {
                "step": index + 1,
                "action_id": action,
                "action": PackRestoreAction(action).name,
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
                    "record_created": record_created,
                    "storage_packed": storage_packed,
                    "storage_packed_grain": (
                        int(state.packed_grain) if storage_packed else 0
                    ),
                    "empty_frame_created": (
                        storage_packed
                        and int(state.empty_frames) > int(before.empty_frames)
                    ),
                    "storage_rebuilt": storage_rebuilt,
                    "physical_created": created,
                    "conservation_violation": created > 0,
                    "delivered": delivered,
                    "violation_grain_delivered": violation_delivered,
                    "violation_delivery": violation_delivered > 0,
                    "goal_success": bool(
                        pack_restore_goal_vector(state)[DELIVER_3_GOAL_INDEX]
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
                pack_restore_goal_vector(state)[DELIVER_3_GOAL_INDEX]
            ),
            "physical_grain_total": int(physical_grain_total(state)),
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


def main():
    network, parameters = _load_policy()
    initial = make_pack_restore_state(
        0,
        int(PackRestorePhase.LOADED),
        split=PackRestoreSplit.VALIDATION,
        start=PackRestoreStart.NATURAL,
        source_growth_period=8,
    )
    fixed = _record_kernel(
        network, parameters, initial, PackRestoreVariant.FIXED
    )
    mutant = _record_kernel(
        network, parameters, initial, PackRestoreVariant.MUTANT
    )
    payload = {
        "schema_version": "hackrl_pack_restore_policy_trace_v1",
        "provenance": provenance(CHECKPOINT),
        "environment": "PACK-RESTORE",
        "trace_semantics": (
            "Every before/after pair is an actual kernel transition. "
            "Animation frames may hold these states but must not interpolate "
            "unrecorded game states."
        ),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "policy": {
            "method": "Dual",
            "seed": 20,
            "adaptation_updates": 4096,
            "goal": "deliver_3",
        },
        "selection": {
            "rule": "fixed example: validation layout 0, loaded",
            "candidate_count": 1,
        },
        "start": {
            "split": "validation",
            "family": "natural_reset",
            "layout": 0,
            "phase": "loaded",
            "source_growth_period": 8,
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
        "fixed",
        fixed["summary"],
        "mutant",
        mutant["summary"],
        flush=True,
    )


if __name__ == "__main__":
    main()
