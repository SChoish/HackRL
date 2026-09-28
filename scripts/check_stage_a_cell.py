#!/usr/bin/env python
"""Abort only on save failure, NaNs, or patched-fixture mob leakage."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import jax
from craftax.craftax_classic.constants import Action

from hackrl import FixtureDynamics, HackRLEasySymbolicEnvNoAutoReset, MediumTask


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def check_summary(cell_dir: Path):
    summary_path = cell_dir / "summary.json"
    params_path = cell_dir / "params.msgpack"
    if not summary_path.is_file():
        raise SystemExit(f"missing {summary_path}")
    if not params_path.is_file():
        raise SystemExit(f"missing {params_path}")
    summary = json.loads(summary_path.read_text())
    for key in ("loss", "entropy", "eval_mean_return"):
        if not _finite(summary.get(key)):
            raise SystemExit(f"non-finite {key} in {summary_path}")
    return summary


def check_patched_mobs(dynamics: str):
    if FixtureDynamics(dynamics) is not FixtureDynamics.PATCHED:
        return
    env = HackRLEasySymbolicEnvNoAutoReset(
        MediumTask.R_M, mutant=False, dynamics=FixtureDynamics.PATCHED
    )
    params = env.default_params
    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    _, state = env.reset(reset_key, params)
    for _ in range(100):
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step(step_key, state, Action.DO.value, params)
        if bool(state.mob_map.any()) or bool(state.zombies.mask.any()):
            raise SystemExit("patched fixture leaked a mob")


def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: check_stage_a_cell.py CELL_DIR")
    cell_dir = Path(argv[1])
    summary = check_summary(cell_dir)
    check_patched_mobs(summary.get("dynamics", "patched"))
    success = summary.get("eval_success_rate")
    print(f"ok {cell_dir.name} eval_success_rate={success}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
