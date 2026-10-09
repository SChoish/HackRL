#!/usr/bin/env python3
"""Validate SPATIAL-WALL-PASS and optionally write deterministic artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import jax
import numpy as np

from hackrl.spatial_wall_pass import (
    DISCOUNT,
    SpatialWallPassAction,
    SpatialWallPassSplit,
    SpatialWallPassVariant,
    exploit_reference_actions,
    make_spatial_wall_pass_state,
    materialize_layout,
    normal_reference_actions,
    spatial_wall_pass_step_with_transition,
)
from hackrl.spatial_wall_pass_oracle import (
    audit_spatial_wall_pass_transition,
    discounted_delivery_return,
    initial_spatial_wall_pass_oracle,
)


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    raise TypeError(type(value).__name__)


def _sha256_bytes(*values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def _run(split, layout_index, actions, variant):
    state = make_spatial_wall_pass_state(layout_index, split=split)
    oracle = initial_spatial_wall_pass_oracle()
    trace = []
    for tick, action in enumerate(actions, start=1):
        before = state
        state, transition = spatial_wall_pass_step_with_transition(
            state, int(action), variant
        )
        audit = audit_spatial_wall_pass_transition(
            oracle, before, int(action), state
        )
        oracle = audit.oracle_state
        assert bool(audit.item_conserved)
        assert bool(audit.immutable_layout)
        trace.append(
            {
                "tick": tick,
                "action": SpatialWallPassAction(int(action)).name,
                "player": _plain(state.player_position),
                "item_present": bool(state.item_present),
                "carrying_item": bool(state.carrying_item),
                "delivered_total": int(state.delivered_total),
                "wall_pass": bool(audit.wall_pass),
                "kernel_crossed_wall": bool(transition.crossed_wall),
            }
        )
    success = int(state.delivered_total) >= 1
    return {
        "split": split.value,
        "layout_index": layout_index,
        "variant": variant.value,
        "length": len(actions),
        "success": success,
        "delivered_total": int(state.delivered_total),
        "wall_pass_count": sum(row["wall_pass"] for row in trace),
        "item_collected_after_pass": bool(oracle.item_collected_after_pass),
        "beneficial_delivery": bool(oracle.beneficial_delivery_seen),
        "discounted_return": discounted_delivery_return(
            len(actions), DISCOUNT
        )
        if success
        else 0.0,
        "trace": trace,
    }


def _physical_key(state):
    return (
        tuple(int(value) for value in np.asarray(state.player_position)),
        bool(state.item_present),
        bool(state.carrying_item),
        int(state.delivered_total),
    )


def _shortest_delivery(variant):
    """Exact BFS on the finite canonical physical-state projection."""

    initial = make_spatial_wall_pass_state(0)
    queue = deque(((initial, ()),))
    seen = {_physical_key(initial)}
    while queue:
        state, actions = queue.popleft()
        if int(state.delivered_total) >= 1:
            return {
                "length": len(actions),
                "actions": [
                    SpatialWallPassAction(value).name for value in actions
                ],
                "explored_physical_states": len(seen),
            }
        for action in SpatialWallPassAction:
            next_state, _ = spatial_wall_pass_step_with_transition(
                state, int(action), variant
            )
            key = _physical_key(next_state)
            if key not in seen:
                seen.add(key)
                queue.append((next_state, actions + (int(action),)))
    raise AssertionError("delivery is unreachable")


def validate():
    fixed_shortest = _shortest_delivery(SpatialWallPassVariant.FIXED)
    mutant_shortest = _shortest_delivery(SpatialWallPassVariant.MUTANT)
    assert fixed_shortest["length"] == 14
    assert mutant_shortest["length"] == 8
    layouts = []
    normal = []
    exploit = []
    all_hashes = set()
    for split in SpatialWallPassSplit:
        split_hashes = set()
        for layout_index in range(16):
            walkable, wall_mask, camp, item, player = materialize_layout(
                layout_index, split
            )
            layout_hash = _sha256_bytes(
                walkable, wall_mask, camp, item, player
            )
            assert layout_hash not in split_hashes
            assert layout_hash not in all_hashes
            split_hashes.add(layout_hash)
            all_hashes.add(layout_hash)
            layouts.append(
                {
                    "split": split.value,
                    "layout_index": layout_index,
                    "sha256": layout_hash,
                    "walkable_cells": int(np.asarray(walkable).sum()),
                    "barrier_cells": int(np.asarray(wall_mask).sum()),
                    "camp": _plain(camp),
                    "item": _plain(item),
                    "player": _plain(player),
                }
            )

            normal_actions = tuple(
                int(action)
                for action in np.asarray(normal_reference_actions(layout_index))
            )
            fixed_normal = _run(
                split,
                layout_index,
                normal_actions,
                SpatialWallPassVariant.FIXED,
            )
            mutant_normal = _run(
                split,
                layout_index,
                normal_actions,
                SpatialWallPassVariant.MUTANT,
            )
            assert fixed_normal["success"] and mutant_normal["success"]
            assert fixed_normal["wall_pass_count"] == 0
            assert mutant_normal["wall_pass_count"] == 0
            assert fixed_normal["trace"] == mutant_normal["trace"]
            normal.extend((fixed_normal, mutant_normal))

            exploit_actions = tuple(
                int(action)
                for action in np.asarray(exploit_reference_actions(layout_index))
            )
            fixed_exploit = _run(
                split,
                layout_index,
                exploit_actions,
                SpatialWallPassVariant.FIXED,
            )
            mutant_exploit = _run(
                split,
                layout_index,
                exploit_actions,
                SpatialWallPassVariant.MUTANT,
            )
            assert not fixed_exploit["success"]
            assert fixed_exploit["wall_pass_count"] == 0
            assert mutant_exploit["success"]
            assert mutant_exploit["wall_pass_count"] == 2
            assert mutant_exploit["beneficial_delivery"]
            assert (
                mutant_exploit["discounted_return"]
                > fixed_normal["discounted_return"]
            )
            exploit.extend((fixed_exploit, mutant_exploit))

    validation = {
        "schema_version": "hackrl_spatial_wall_pass_validation_v1",
        "candidate_id": "SPATIAL-WALL-PASS",
        "kernel_stage": "transition_and_reference_path_validation",
        "training_started": False,
        "queue_gate": "closed",
        "layout_count": len(layouts),
        "normal_replays": len(normal),
        "exploit_replays": len(exploit),
        "normal_reference_length": 14,
        "mutant_exploit_reference_length": 8,
        "fixed_shortest_path": fixed_shortest,
        "mutant_shortest_path": mutant_shortest,
        "normal_reference_is_shortest_path_claim": True,
        "mutant_exploit_is_shortest_path_claim": True,
        "normal_successes": sum(row["success"] for row in normal),
        "fixed_counterfactual_successes": sum(
            row["success"]
            for row in exploit
            if row["variant"] == SpatialWallPassVariant.FIXED.value
        ),
        "mutant_exploit_successes": sum(
            row["success"]
            for row in exploit
            if row["variant"] == SpatialWallPassVariant.MUTANT.value
        ),
        "independence_boundary": (
            "The oracle is a separate state-difference implementation but was "
            "written in the same repository and author scope as the kernel; "
            "this is internal consistency evidence, not external ground truth."
        ),
        "learning_gate_remaining": [
            "wire the structured observation and 12 normal goals into the validated PPO and Dual learner",
            "confirm fixed natural-start normal-task learning before mutant adaptation",
            "freeze evaluation episodes, action selection, seeds, and transition budgets before viewing exploit outcomes",
        ],
    }
    assert validation["layout_count"] == 48
    assert validation["normal_successes"] == 96
    assert validation["fixed_counterfactual_successes"] == 0
    assert validation["mutant_exploit_successes"] == 48
    return layouts, normal, exploit, validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-dir",
        type=Path,
        help="write deterministic JSON artifacts; omit for validation only",
    )
    args = parser.parse_args()
    layouts, normal, exploit, validation = validate()
    artifacts = {
        "spatial_wall_pass_v1_layouts.json": layouts,
        "spatial_wall_pass_v1_normal_traces.json": normal,
        "spatial_wall_pass_v1_exploit_traces.json": exploit,
        "spatial_wall_pass_v1_validation.json": validation,
    }
    if args.write_dir is not None:
        args.write_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in artifacts.items():
            path = args.write_dir / name
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=_plain) + "\n"
            )
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
