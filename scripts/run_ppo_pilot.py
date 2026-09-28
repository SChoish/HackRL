#!/usr/bin/env python
"""Run bounded HackRL PPO pilots and print JSON metrics."""

import argparse
import json

from hackrl.ppo import PPOConfig, run_ppo_pilot
from hackrl.tasks import EasyTask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=[task.value for task in EasyTask],
        default=EasyTask.R_E.value,
    )
    parser.add_argument(
        "--variant",
        choices=["fixed", "mutant"],
        default="mutant",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="run all three Easy tasks in fixed and mutant variants",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--num-updates", type=int, default=2)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=2)
    parser.add_argument("--layer-size", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=8)
    return parser.parse_args()


def make_config(args, task, variant):
    return PPOConfig(
        task=task,
        mutant=variant == "mutant",
        seed=args.seed,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_updates=args.num_updates,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        layer_size=args.layer_size,
        eval_episodes=args.eval_episodes,
    )


def main():
    args = parse_args()
    if args.all_pairs:
        selections = [
            (task, variant)
            for task in EasyTask
            for variant in ("fixed", "mutant")
        ]
        result = [
            run_ppo_pilot(make_config(args, task, variant))
            for task, variant in selections
        ]
    else:
        result = run_ppo_pilot(
            make_config(args, EasyTask(args.task), args.variant)
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
