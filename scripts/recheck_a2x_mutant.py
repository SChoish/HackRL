#!/usr/bin/env python
"""Reload a2x mutant checkpoints and compare train-tail vs eval-path rates."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax.serialization import from_bytes

from craftax.craftax_classic.constants import Action

from hackrl import FixtureDynamics, HackRLEasySymbolicEnvNoAutoReset, MediumTask
from hackrl.evaluation import evaluate_policy
from hackrl.ppo import ActorCritic, PPOConfig
from hackrl.rollout import HackRLBatchEnv
from hackrl.scripted_paths import R_M_NORMAL_PATH

ROOT = Path("/home/ext_csv/HackRL/runs/overnight_stage_a")
CKPTS = (1_048_576, 4_194_304, 16_777_216)
SEEDS = (0, 1, 2)
MODE_N = 32
SAMPLE_N = 256
TRAIN_N = 32
SHARED_SEED = 20260929


def _tail_rate(rows, n):
    tail = rows[-n:]
    success = sum(float(row["successful_episodes"]) for row in tail)
    done = sum(float(row["completed_episodes"]) for row in tail)
    return {
        "successful_episodes": success,
        "completed_episodes": done,
        "success_rate": (success / done) if done else None,
    }


def _load_network(env, seed, path: Path):
    config = PPOConfig(task=MediumTask.R_M, layer_size=256, seed=seed)
    network = ActorCritic(
        action_dim=env.action_space(env.default_params).n,
        layer_width=config.layer_size,
        activation=config.activation,
    )
    rng = jax.random.PRNGKey(seed)
    dummy, _ = env.reset(rng, env.default_params)
    params = from_bytes(network.init(rng, dummy), path.read_bytes())
    return network, params


def _eval(network, params, env, rng, n, stochastic):
    metrics = jax.device_get(
        jax.jit(
            lambda key: evaluate_policy(
                network, params, env, key, n, stochastic=stochastic
            )
        )(rng)
    )
    return {
        "success_rate": float(metrics["eval_success_rate"]),
        "mean_length": float(metrics["eval_mean_length"]),
        "completion_rate": float(metrics["eval_completion_rate"]),
        "termination_death_rate": float(metrics["eval_termination_death_rate"]),
        "termination_goal_rate": float(metrics["eval_termination_goal_rate"]),
    }


def _train_path_first_episode(network, params, env, rng, n, stochastic):
    vector_env = HackRLBatchEnv(env, n)

    def run(rng):
        rng, reset_rng = jax.random.split(rng)
        observations, vector_state = vector_env.reset(reset_rng)
        recorded = jnp.zeros((n,), dtype=bool)
        success = jnp.zeros((n,), dtype=bool)
        lengths = jnp.zeros((n,), dtype=jnp.int32)

        def body(carry, _):
            observations, vector_state, recorded, success, lengths, rng = carry
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
            newly = jnp.logical_and(~recorded, transition.done)
            return (
                observations,
                vector_state,
                recorded | newly,
                jnp.where(newly, transition.episode.goal_success, success),
                jnp.where(newly, transition.episode.episode_length, lengths),
                rng,
            ), None

        (*_, recorded, success, lengths, _), _ = jax.lax.scan(
            body,
            (observations, vector_state, recorded, success, lengths, rng),
            None,
            length=env.spec.horizon,
        )
        return recorded, success, lengths

    recorded, success, lengths = jax.device_get(jax.jit(run)(rng))
    return {
        "first_episode_completion_rate": float(recorded.mean()),
        "first_episode_success_rate": float(success.mean()),
        "first_episode_mean_length": float(lengths.mean()),
    }


def _roll_to(env, actions, seed):
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    observation, state = env.reset(reset_key, env.default_params)
    for action in actions:
        key, step_key = jax.random.split(key)
        observation, state, *_ = env.step(
            step_key, state, action, env.default_params
        )
    return observation, state


def _policy_probe(network, params, observation):
    policy, value = network.apply(params, observation[None])
    mode = int(jax.device_get(policy.mode())[0])
    value = jax.device_get(value)
    value = float(value.reshape(-1)[0])
    probs = [float(item) for item in jax.device_get(policy.probs)[0]]
    return {
        "mode": mode,
        "mode_name": getattr(Action(mode), "name", str(mode)),
        "value": value,
        "prob_make_iron_pickaxe": probs[Action.MAKE_IRON_PICKAXE.value],
        "prob_do": probs[Action.DO.value],
        "prob_down": probs[Action.DOWN.value],
        "prob_up": probs[Action.UP.value],
        "prob_noop": probs[Action.NOOP.value],
    }


def _state_snapshot(state):
    return {
        "position": [int(state.player_position[0]), int(state.player_position[1])],
        "direction": int(state.player_direction),
        "timestep": int(state.timestep),
        "health": int(state.player_health),
        "iron": int(state.inventory.iron),
        "iron_pickaxe": int(state.inventory.iron_pickaxe),
        "diamond": int(state.inventory.diamond),
        "wood": int(state.inventory.wood),
    }


def main():
    env = HackRLEasySymbolicEnvNoAutoReset(
        MediumTask.R_M, mutant=True, dynamics=FixtureDynamics.PATCHED
    )
    report = {"shared_seed": SHARED_SEED, "cells": []}
    for seed in SEEDS:
        cell = ROOT / f"a2x_patched_mutant_s{seed}"
        rows = list(csv.DictReader((cell / "updates.csv").open()))
        stored = json.loads((cell / "summary.json").read_text())
        item = {
            "cell": cell.name,
            "n_updates": len(rows),
            "tail_32": _tail_rate(rows, 32),
            "tail_128": _tail_rate(rows, 128),
            "tail_8": _tail_rate(rows, 8),
            "stored_eval_mode": stored["eval_success_rate"],
            "stored_eval_sample": stored["eval_sample_success_rate"],
            "checkpoints": {},
        }
        rng = jax.random.PRNGKey(SHARED_SEED + seed)
        for transitions in CKPTS:
            ckpt = cell / "checkpoints" / f"transitions_{transitions}" / "params.msgpack"
            network, params = _load_network(env, seed, ckpt)
            rng, mode_rng, sample_rng, train_mode_rng, train_sample_rng = (
                jax.random.split(rng, 5)
            )
            stored_eval_path = ckpt.parent / "eval.json"
            stored_eval = (
                json.loads(stored_eval_path.read_text())
                if stored_eval_path.is_file()
                else None
            )
            probes = {}
            for name, prefix in (
                ("reset", ()),
                ("post_craft", R_M_NORMAL_PATH[:4]),
                ("pre_diamond", R_M_NORMAL_PATH[:5]),
            ):
                observation, state = _roll_to(env, prefix, SHARED_SEED)
                probes[name] = {
                    "state": _state_snapshot(state),
                    "policy": _policy_probe(network, params, observation),
                }
            item["checkpoints"][str(transitions)] = {
                "stored_eval": None
                if stored_eval is None
                else {
                    "eval_success_rate": stored_eval.get("eval_success_rate"),
                    "eval_sample_success_rate": stored_eval.get(
                        "eval_sample_success_rate"
                    ),
                    "eval_mean_length": stored_eval.get("eval_mean_length"),
                },
                "reload_mode_32": _eval(network, params, env, mode_rng, MODE_N, False),
                "reload_sample_256": _eval(
                    network, params, env, sample_rng, SAMPLE_N, True
                ),
                "train_path_mode_first_32": _train_path_first_episode(
                    network, params, env, train_mode_rng, TRAIN_N, False
                ),
                "train_path_sample_first_32": _train_path_first_episode(
                    network, params, env, train_sample_rng, TRAIN_N, True
                ),
                "probes": probes,
            }
        report["cells"].append(item)
        print(json.dumps(item, indent=2))
    destination = ROOT / "a2x_mutant_recheck.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
