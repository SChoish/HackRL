#!/usr/bin/env python3
"""Run one pre-registered TICK-CLAIM GC-PPO calibration cell."""

from __future__ import annotations

import argparse
import json
import os

if os.environ.get("HACKRL_DEVICE", "cpu") != "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from hackrl.tick_claim_gc import (
    TickClaimGCConfig,
    run_tick_claim_gc_calibration_cell,
)


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("fixed", "mutant"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--num-updates", type=int, default=32)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--mode-repeats", type=int, default=1)
    parser.add_argument("--sample-repeats", type=int, default=4)
    parser.add_argument(
        "--goal-mode",
        choices=("deliver_3", "workshop12"),
        default="deliver_3",
    )
    parser.add_argument(
        "--checkpoint-updates",
        default="",
        help="comma-separated update indices to snapshot, including 0",
    )
    return parser.parse_args()


def main():
    arguments = _arguments()
    config = TickClaimGCConfig(
        variant=arguments.variant,
        seed=arguments.seed,
        num_envs=arguments.num_envs,
        num_steps=arguments.num_steps,
        num_updates=arguments.num_updates,
        minibatch_size=arguments.minibatch_size,
        hidden_size=arguments.hidden_size,
        mode_repeats_per_state=arguments.mode_repeats,
        sample_repeats_per_state=arguments.sample_repeats,
        goal_mode=arguments.goal_mode,
        checkpoint_updates=tuple(
            int(item)
            for item in arguments.checkpoint_updates.split(",")
            if item
        ),
    )
    summary = run_tick_claim_gc_calibration_cell(
        config, arguments.log_dir
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
