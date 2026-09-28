#!/usr/bin/env python
"""Check the R-E replenish fixture against the three required contracts."""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import BlockType

from hackrl import EasyTask, FixtureVersion, HackRLEasySymbolicEnvNoAutoReset
from hackrl.scripted_paths import (
    EASY_EXPLOIT_PATHS,
    EASY_NORMAL_PATHS,
    R_E_WOOD_WASTE_RECOVERY_PATH,
)
from hackrl.tasks import (
    R_E_REPLENISH_COALS,
    R_E_REPLENISH_STONES,
    R_E_REPLENISH_TREES,
)


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


def _env(mutant, fixture=FixtureVersion.R_E_REPLENISH):
    return HackRLEasySymbolicEnvNoAutoReset(
        EasyTask.R_E, mutant=mutant, fixture=fixture
    )


def check_shortest_paths():
    for mutant in (False, True):
        history = _rollout(_env(mutant), EASY_NORMAL_PATHS[EasyTask.R_E])
        assert len(history) == 3
        assert float(history[-1][1]) == 1.0
        assert bool(history[-1][2])
        exploit = _rollout(_env(mutant), EASY_EXPLOIT_PATHS[EasyTask.R_E])
        succeeded = float(exploit[-1][1]) == 1.0 and bool(exploit[-1][2])
        assert succeeded is mutant
    return "normal 3-step and mutant 1-step paths are unchanged"


def check_recovery_after_wasted_wood():
    history = _rollout(_env(False), R_E_WOOD_WASTE_RECOVERY_PATH)
    wasted = history[0][0]
    harvested = history[3][0]
    final = history[-1]
    assert int(wasted.inventory.wood) == 0
    assert int(harvested.inventory.wood) >= 1
    assert int(harvested.map[9, 10]) == BlockType.GRASS.value
    assert float(final[1]) == 1.0
    assert bool(final[2])
    assert int(final[0].inventory.iron_pickaxe) >= 1
    return "wasted wood can be mined and the iron pickaxe crafted"


def check_identical_fixed_mutant_resources():
    key = jax.random.PRNGKey(0)
    fixed = _env(False)
    mutant = _env(True)
    _, fixed_state = fixed.reset(key, fixed.default_params)
    _, mutant_state = mutant.reset(key, mutant.default_params)
    comparisons = jax.tree_util.tree_map(
        jnp.array_equal, fixed_state, mutant_state
    )
    assert all(bool(value) for value in jax.tree_util.tree_leaves(comparisons))
    for row, col in R_E_REPLENISH_TREES:
        assert int(fixed_state.map[row, col]) == BlockType.TREE.value
    for row, col in R_E_REPLENISH_STONES:
        assert int(fixed_state.map[row, col]) == BlockType.STONE.value
    for row, col in R_E_REPLENISH_COALS:
        assert int(fixed_state.map[row, col]) == BlockType.COAL.value
    return "fixed and mutant share the start state and replenish blocks"


def main():
    checks = (
        check_shortest_paths,
        check_recovery_after_wasted_wood,
        check_identical_fixed_mutant_resources,
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
