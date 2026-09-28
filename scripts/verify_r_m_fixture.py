#!/usr/bin/env python
"""Check R-M: same R-E start, shared diamond segment, both crafts remain valid."""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import BlockType

from hackrl import HackRLEasySymbolicEnvNoAutoReset, MediumTask
from hackrl.scripted_paths import (
    R_M_DIAMOND_SEGMENT,
    R_M_EXPLOIT_PATH,
    R_M_NORMAL_PATH,
)
from hackrl.tasks import R_M_DIAMOND


def _rollout(env, actions, seed=0):
    params = env.default_params
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    _, state = env.reset(reset_key, params)
    history = []
    for action in actions:
        key, step_key = jax.random.split(key)
        _, state, reward, done, info = env.step(
            step_key, state, action, params
        )
        history.append((state, reward, done, info))
        if bool(done):
            break
    return history


def _env(mutant):
    return HackRLEasySymbolicEnvNoAutoReset(MediumTask.R_M, mutant=mutant)


def check_r_e_start_plus_diamond():
    env = _env(False)
    assert env.spec.horizon == 512
    _, state = env.reset(jax.random.PRNGKey(0), env.default_params)
    assert tuple(map(int, state.player_position)) == (8, 8)
    assert int(state.inventory.wood) == 1
    assert int(state.inventory.stone) == 1
    assert int(state.inventory.coal) == 1
    assert int(state.inventory.iron) == 0
    assert int(state.inventory.wood_pickaxe) == 1
    assert int(state.inventory.stone_pickaxe) == 1
    assert int(state.inventory.diamond) == 0
    assert int(state.map[8, 7]) == BlockType.CRAFTING_TABLE.value
    assert int(state.map[8, 9]) == BlockType.FURNACE.value
    assert int(state.map[6, 8]) == BlockType.IRON.value
    assert int(state.map[R_M_DIAMOND]) == BlockType.DIAMOND.value
    return "R-M keeps the R-E start inventory and workshop, plus diamond at (10, 8)"


def check_shared_diamond_segment():
    assert R_M_NORMAL_PATH[-2:] == R_M_DIAMOND_SEGMENT
    assert R_M_EXPLOIT_PATH[1:] == R_M_DIAMOND_SEGMENT
    return "normal and exploit crafts share DOWN, DO onto (10, 8)"


def check_normal_path_still_works():
    for mutant in (False, True):
        history = _rollout(_env(mutant), R_M_NORMAL_PATH)
        crafted = history[3][0]
        assert int(crafted.inventory.iron_pickaxe) >= 1
        assert int(crafted.inventory.diamond) == 0
        assert len(history) == 6
        assert float(history[-1][1]) == 1.0
        assert bool(history[-1][2])
        assert int(history[-1][0].inventory.diamond) >= 1
    return "mine-iron then craft then the shared diamond segment succeeds on both kernels"


def check_exploit_uses_same_segment():
    mutant = _rollout(_env(True), R_M_EXPLOIT_PATH)
    assert int(mutant[0][0].inventory.iron_pickaxe) >= 1
    assert int(mutant[0][0].inventory.diamond) == 0
    assert float(mutant[-1][1]) == 1.0
    assert bool(mutant[-1][2])
    fixed = _rollout(_env(False), R_M_EXPLOIT_PATH)
    assert int(fixed[0][0].inventory.iron_pickaxe) == 0
    assert float(fixed[-1][1]) == 0.0
    assert int(fixed[-1][0].inventory.diamond) == 0
    return "iron-free craft then the same diamond segment succeeds only on mutant"


def check_identical_start():
    key = jax.random.PRNGKey(3)
    fixed = _env(False)
    mutant = _env(True)
    _, fixed_state = fixed.reset(key, fixed.default_params)
    _, mutant_state = mutant.reset(key, mutant.default_params)
    comparisons = jax.tree_util.tree_map(
        jnp.array_equal, fixed_state, mutant_state
    )
    assert all(bool(value) for value in jax.tree_util.tree_leaves(comparisons))
    return "fixed and mutant share the R-M start state"


def main():
    checks = (
        check_r_e_start_plus_diamond,
        check_shared_diamond_segment,
        check_normal_path_still_works,
        check_exploit_uses_same_segment,
        check_identical_start,
    )
    failed = False
    for check in checks:
        try:
            message = check()
        except Exception as error:  # noqa: BLE001
            failed = True
            print(f"FAIL {check.__name__}: {error}")
        else:
            print(f"PASS {check.__name__}: {message}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
