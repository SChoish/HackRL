#!/usr/bin/env python3
"""Frozen diagnostic for three legal reservation timings.

The states are reached by fixed-kernel traces from a validation reset.
They are not training starts. Every policy is given deliver_3.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

if os.environ.get("HACKRL_DEVICE", "cpu") != "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from hackrl.tick_claim import (
    TickClaimAction,
    TickClaimPhase,
    TickClaimSplit,
    TickClaimStart,
    TickClaimVariant,
    make_tick_claim_state,
    observe_tick_claim,
    tick_claim_step_with_snapshot,
)
from hackrl.tick_claim_gc import (
    DELIVER_3_GOAL_INDEX,
    TickClaimGCActorCritic,
    TickClaimGCConfig,
    initialize_tick_claim_gc,
    tick_claim_gc_inputs,
)

HISTORY_ROOT = Path("/home/ext_csv/HackRL/runs/tick_claim_gc_history_v1")
PROBE_PATH = (
    TickClaimAction.LEFT,
    TickClaimAction.LEFT,
    TickClaimAction.UP,
    TickClaimAction.UP,
)
PROBE_NAMES = ("no_reservation", "due_next", "due_now")
ORIGINS = ("none", "deliver3", "workshop12", "adam_reset")
UPDATES = (0, 512, 4096)


def build_reservation_probe_states():
    """Legal validation trace: face a ripe crop, arm, then wait one tick."""

    state = make_tick_claim_state(
        0,
        int(TickClaimPhase.RIPE),
        split=TickClaimSplit.VALIDATION,
        start=TickClaimStart.NATURAL,
    )
    for action in PROBE_PATH:
        state, _ = tick_claim_step_with_snapshot(
            state, int(action), TickClaimVariant.FIXED
        )
    no_reservation = state
    due_next, _ = tick_claim_step_with_snapshot(
        state, int(TickClaimAction.ARM_HARVEST), TickClaimVariant.FIXED
    )
    due_now, _ = tick_claim_step_with_snapshot(
        due_next, int(TickClaimAction.NOOP), TickClaimVariant.FIXED
    )
    return {
        "no_reservation": no_reservation,
        "due_next": due_next,
        "due_now": due_now,
    }


def _payout(before, after):
    return int(after.grain) - int(before.grain)


def probe_environment_contract(states):
    """Mutant pays twice only when a due reservation and a manual harvest coincide."""

    report = {}
    for name, state in states.items():
        report[name] = {}
        for variant in (TickClaimVariant.FIXED, TickClaimVariant.MUTANT):
            after, _ = tick_claim_step_with_snapshot(
                state, int(TickClaimAction.DO), variant
            )
            report[name][variant.value] = _payout(state, after)
    return report


def _batched_states(states):
    ordered = [states[name] for name in PROBE_NAMES]
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *ordered)


def _mode_and_do(network, params, batched):
    observations = jax.vmap(observe_tick_claim)(batched)
    goals = jnp.full((len(PROBE_NAMES),), DELIVER_3_GOAL_INDEX, dtype=jnp.int32)
    model_inputs = jax.vmap(tick_claim_gc_inputs)(observations, goals)
    policy, _ = network.apply(params, *model_inputs)
    probabilities = np.asarray(jax.device_get(policy.probs))
    modes = np.asarray(jax.device_get(policy.mode()))
    do_index = int(TickClaimAction.DO)
    return modes, probabilities[:, do_index], probabilities


def _execute(state, action, variant):
    after, _ = tick_claim_step_with_snapshot(state, int(action), variant)
    return {
        "payout": _payout(state, after),
        "crop_ripe": bool(after.crop_ripe),
        "reservation_present": bool(after.reservation_present),
    }


def evaluate_policy(network, params, states):
    modes, do_probs, _ = _mode_and_do(network, params, _batched_states(states))
    rows = {}
    for index, name in enumerate(PROBE_NAMES):
        action = int(modes[index])
        rows[name] = {
            "do_probability": float(do_probs[index]),
            "mode_action": action,
            "mode_is_do": action == int(TickClaimAction.DO),
            "mutant": _execute(states[name], action, TickClaimVariant.MUTANT),
            "fixed": _execute(states[name], action, TickClaimVariant.FIXED),
        }
    return rows


def _checkpoint(origin, variant, seed, update):
    return (
        HISTORY_ROOT
        / origin
        / f"{variant}_seed{seed}"
        / "checkpoints"
        / f"adapt_{update}"
    )


def evaluate_history(log_path):
    states = build_reservation_probe_states()
    contract = probe_environment_contract(states)
    if contract["due_now"] != {"fixed": 1, "mutant": 2}:
        raise RuntimeError(f"due-now probe is not a duplicate payout: {contract}")
    if contract["due_next"] != {"fixed": 1, "mutant": 1}:
        raise RuntimeError(f"one-tick probe is not a single manual payout: {contract}")
    if contract["no_reservation"] != {"fixed": 1, "mutant": 1}:
        raise RuntimeError(f"manual probe is not a single payout: {contract}")
    recorded = json.loads(
        (_checkpoint("workshop12", "fixed", 5, 0) / "config.json").read_text()
    )
    names = {item.name for item in __import__("dataclasses").fields(TickClaimGCConfig)}
    config = TickClaimGCConfig(**{name: recorded[name] for name in names})
    _, template = initialize_tick_claim_gc(config)
    network = TickClaimGCActorCritic(hidden_size=512)
    rows = []
    for origin in ORIGINS:
        for seed in range(5, 10):
            for variant in ("fixed", "mutant"):
                for update in UPDATES:
                    directory = _checkpoint(origin, variant, seed, update)
                    runner = serialization.from_bytes(
                        template, (directory / "state.msgpack").read_bytes()
                    )
                    evaluated = evaluate_policy(
                        network, runner.train_state.params, states
                    )
                    rows.append(
                        {
                            "origin": origin,
                            "seed": seed,
                            "trained_variant": variant,
                            "adaptation_update": update,
                            "probes": evaluated,
                        }
                    )
                    print(
                        f"[probe] {origin} {variant} seed {seed} update {update}",
                        flush=True,
                    )
    document = {
        "schema_version": "tick_claim_reservation_probe_v1",
        "split": "validation",
        "layout_index": 0,
        "phase": "ripe",
        "goal": "deliver_3",
        "trace_actions": [int(action) for action in PROBE_PATH]
        + [int(TickClaimAction.ARM_HARVEST), int(TickClaimAction.NOOP)],
        "environment_contract_do_payout": contract,
        "rows": rows,
    }
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    return document


def main():
    evaluate_history(HISTORY_ROOT / "reservation_probe.json")


if __name__ == "__main__":
    main()
