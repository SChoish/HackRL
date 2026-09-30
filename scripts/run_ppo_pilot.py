#!/usr/bin/env python
"""Run bounded HackRL PPO pilots and print JSON metrics."""

import argparse
import json

from hackrl.ppo import PPOConfig, run_ppo_pilot
from hackrl.tasks import EasyTask, FixtureDynamics, MediumTask, parse_task


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=[task.value for task in (*EasyTask, *MediumTask)],
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
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Adam learning rate. Stage A2 keeps 2e-4 fixed.",
    )
    parser.add_argument(
        "--anneal-learning-rate",
        action="store_true",
        help="opt into linear learning-rate annealing (protocol default is fixed)",
    )
    parser.add_argument(
        "--dynamics",
        choices=[dynamics.value for dynamics in FixtureDynamics],
        default=FixtureDynamics.PATCHED.value,
        help="legacy keeps the old mob-slot init; patched wipes empty mobs",
    )
    parser.add_argument(
        "--checkpoint-updates",
        default="",
        help="comma-separated update counts to save params+eval, e.g. 0,32,128,512",
    )
    parser.add_argument(
        "--eval-sample-episodes",
        type=int,
        default=None,
        help="sample-eval episodes; defaults to --eval-episodes",
    )
    parser.add_argument(
        "--start-mode",
        choices=["default", "r_e_post_iron", "r_m_d1", "r_m_d2", "r_m_d3"],
        default="default",
    )
    parser.add_argument(
        "--fixture",
        choices=["default", "r_e_replenish"],
        default="default",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="write config, SHA, per-update CSV, and params checkpoint",
    )
    return parser.parse_args()


def make_config(args, task, variant, log_dir=None):
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
        eval_sample_episodes=args.eval_sample_episodes,
        learning_rate=args.learning_rate,
        anneal_learning_rate=args.anneal_learning_rate,
        start_mode=args.start_mode,
        fixture=args.fixture,
        dynamics=args.dynamics,
        log_dir=log_dir,
        checkpoint_updates=tuple(
            int(item) for item in args.checkpoint_updates.split(",") if item
        ),
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
            run_ppo_pilot(
                make_config(
                    args,
                    task,
                    variant,
                    None
                    if args.log_dir is None
                    else f"{args.log_dir}/{task.value}_{variant}",
                )
            )
            for task, variant in selections
        ]
    else:
        result = run_ppo_pilot(
            make_config(
                args, parse_task(args.task), args.variant, args.log_dir
            )
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
