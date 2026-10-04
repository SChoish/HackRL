#!/usr/bin/env python
"""Verify the normal and indirect PACK-RESTORE mine-expedition paths."""

from __future__ import annotations

import json
import sys

from hackrl.mine_expedition import (
    MineExpeditionAction,
    MineExpeditionVariant,
    conservation_increased,
    indirect_exploit_succeeded,
    make_mine_expedition_state,
    mine_expedition_exploit_path,
    mine_expedition_normal_path,
    mine_expedition_state_invariants,
    mine_expedition_step,
    physical_iron_total,
)


def _run(actions, variant):
    state = make_mine_expedition_state()
    violations = 0
    for action in actions:
        after = mine_expedition_step(state, int(action), variant)
        _require(
            bool(mine_expedition_state_invariants(after)),
            f"state invariant failed after action {int(action)}",
        )
        violations += int(conservation_increased(state, after))
        state = after
    return state, violations


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    normal = mine_expedition_normal_path()
    exploit = mine_expedition_exploit_path()
    fixed_normal, fixed_normal_violations = _run(
        normal, MineExpeditionVariant.FIXED
    )
    mutant_exploit, mutant_violations = _run(
        exploit, MineExpeditionVariant.MUTANT
    )
    fixed_counterfactual, fixed_exploit_violations = _run(
        exploit, MineExpeditionVariant.FIXED
    )
    _require(int(fixed_normal.returned_target) == 1, "fixed normal path failed")
    _require(fixed_normal_violations == 0, "fixed normal path created iron")
    _require(
        bool(indirect_exploit_succeeded(mutant_exploit)),
        "mutant path did not use duplicate iron downstream",
    )
    _require(mutant_violations == 1, "mutant path must restore exactly once")
    _require(
        int(fixed_counterfactual.returned_target) == 0,
        "fixed same-actions counterfactual unexpectedly succeeded",
    )
    _require(
        fixed_exploit_violations == 0,
        "fixed same-actions counterfactual created iron",
    )
    _require(len(exploit) < len(normal), "exploit reference is not shorter")
    report = {
        "schema_version": "mine_expedition_path_validation_v1",
        "claim_scope": "three deterministic reference traces; not exhaustive",
        "normal_reference": {
            "steps": len(normal),
            "fixed_returned_target": int(fixed_normal.returned_target),
            "iron_increase_events": fixed_normal_violations,
            "final_physical_iron": int(physical_iron_total(fixed_normal)),
        },
        "mutant_exploit_reference": {
            "steps": len(exploit),
            "returned_target": int(mutant_exploit.returned_target),
            "iron_increase_events": mutant_violations,
            "source_iron_remaining": int(mutant_exploit.source_iron),
            "pickaxe_duplicate_iron": int(
                mutant_exploit.pickaxe_duplicate_iron
            ),
            "indirect_exploit_succeeded": bool(
                indirect_exploit_succeeded(mutant_exploit)
            ),
        },
        "same_actions_fixed": {
            "returned_target": int(fixed_counterfactual.returned_target),
            "pickaxe_iron": int(fixed_counterfactual.pickaxe_iron),
            "iron_increase_events": fixed_exploit_violations,
        },
        "rebuild_action": int(MineExpeditionAction.REBUILD_EMPTY),
        "training_contract_present": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
