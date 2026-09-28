#!/usr/bin/env python
"""Collect the finished R-M 6-cell table and recount eval terminations."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax.serialization import from_bytes

from hackrl import HackRLEasySymbolicEnvNoAutoReset, MediumTask
from hackrl.ppo import ActorCritic, PPOConfig
from hackrl.rollout import HackRLBatchEnv

ROOT = Path("/home/ext_csv/HackRL/runs/r_m_compare")
CELLS = (
    ("fixed", 0),
    ("fixed", 1),
    ("fixed", 2),
    ("mutant", 0),
    ("mutant", 1),
    ("mutant", 2),
)


def _load_params(env, config: PPOConfig, path: Path):
    network = ActorCritic(
        action_dim=env.action_space(env.default_params).n,
        layer_width=config.layer_size,
        activation=config.activation,
    )
    rng = jax.random.PRNGKey(config.seed)
    dummy, _ = env.reset(rng, env.default_params)
    init_params = network.init(rng, dummy)
    return network, from_bytes(init_params, path.read_bytes())


def _termination_rates(network, params, env, rng, stochastic: bool):
    num_episodes = 32
    vector_env = HackRLBatchEnv(env, num_episodes)

    def run(rng):
        rng, reset_rng = jax.random.split(rng)
        observations, vector_state = vector_env.reset(reset_rng)
        init = (
            observations,
            vector_state,
            jnp.zeros((num_episodes,), dtype=bool),
            jnp.zeros((num_episodes,), dtype=bool),
            jnp.zeros((num_episodes,), dtype=bool),
            jnp.zeros((num_episodes,), dtype=bool),
            jnp.zeros((num_episodes,), dtype=jnp.int32),
            rng,
        )

        def body(carry, _):
            observations, vector_state, recorded, goal, death, timeout, lengths, rng = (
                carry
            )
            rng, action_rng, step_rng = jax.random.split(rng, 3)
            policy, _ = network.apply(params, observations)
            actions = jnp.where(
                stochastic,
                policy.sample(seed=action_rng).astype(jnp.int32),
                policy.mode().astype(jnp.int32),
            )
            observations, vector_state, transition = vector_env.step(
                step_rng, vector_state, actions
            )
            newly = jnp.logical_and(jnp.logical_not(recorded), transition.done)
            info = transition.info
            return (
                observations,
                vector_state,
                jnp.logical_or(recorded, newly),
                jnp.where(newly, info["HackRL/termination_goal"], goal),
                jnp.where(newly, info["HackRL/termination_death"], death),
                jnp.where(newly, info["HackRL/termination_timeout"], timeout),
                jnp.where(newly, transition.episode.episode_length, lengths),
                rng,
            ), None

        (_, _, recorded, goal, death, timeout, lengths, _), _ = jax.lax.scan(
            body, init, None, length=env.spec.horizon
        )
        return recorded, goal, death, timeout, lengths

    recorded, goal, death, timeout, lengths = jax.device_get(jax.jit(run)(rng))
    return {
        "completion_rate": float(recorded.mean()),
        "goal_rate": float(goal.mean()),
        "death_rate": float(death.mean()),
        "timeout_rate": float(timeout.mean()),
        "mean_length": float(lengths.mean()),
        "unique_lengths": sorted({int(value) for value in lengths}),
    }


def main():
    cells = []
    for variant, seed in CELLS:
        cell = f"{variant}_s{seed}"
        summary = json.loads((ROOT / cell / "summary.json").read_text())
        config = PPOConfig(
            task=MediumTask.R_M,
            mutant=variant == "mutant",
            seed=seed,
            layer_size=256,
        )
        env = HackRLEasySymbolicEnvNoAutoReset(
            MediumTask.R_M, mutant=config.mutant
        )
        network, params = _load_params(env, config, ROOT / cell / "params.msgpack")
        rng = jax.random.PRNGKey(10_000 + seed + (100 if variant == "mutant" else 0))
        rng, mode_rng, sample_rng = jax.random.split(rng, 3)
        recount_mode = _termination_rates(network, params, env, mode_rng, False)
        recount_sample = _termination_rates(
            network, params, env, sample_rng, True
        )
        cells.append(
            {
                "cell": cell,
                "variant": variant,
                "seed": seed,
                "git_sha": summary["git_sha"],
                "working_tree": "uncommitted R-M fixture on 3dfccf0",
                "transitions": int(summary["transitions"]),
                "horizon": 512,
                "completed_episodes": int(summary["completed_episodes"]),
                "successful_episodes": int(summary["successful_episodes"]),
                "completed_success_rate": summary["completed_success_rate"],
                "first_success_update": summary["first_success_update"],
                "eval_success_rate": summary["eval_success_rate"],
                "eval_sample_success_rate": summary["eval_sample_success_rate"],
                "eval_mean_length": summary["eval_mean_length"],
                "eval_sample_mean_length": summary["eval_sample_mean_length"],
                "completed_iron_acquisition_rate": summary.get(
                    "completed_iron_acquisition_rate",
                    summary.get("completed_iron_acquire_rate"),
                ),
                "metrics_schema_version": summary.get(
                    "metrics_schema_version", 1
                ),
                "fixture_dynamics_version": summary.get(
                    "fixture_dynamics_version", 1
                ),
                "completed_wood_depletion_rate": summary.get(
                    "completed_wood_depletion_rate",
                    summary.get("completed_wood_exhausted_rate"),
                ),
                "eval_violation_episode_rate": summary[
                    "eval_violation_episode_rate"
                ],
                "posthoc_mode": recount_mode,
                "posthoc_sample": recount_sample,
            }
        )
        print(cell, "mode", recount_mode, "sample", recount_sample)

    payload = {
        "budget_transitions": 262144,
        "task": "R-M",
        "note": (
            "Training summaries keep the original 32-episode eval. "
            "posthoc_* recounts termination_goal/death/timeout from saved params."
        ),
        "cells": cells,
    }
    destination = Path("/home/ext_csv/HackRL/docs/r_m_compare_6run.json")
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
