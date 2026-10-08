#!/usr/bin/env python3
"""Run one checkpoint-free online-algorithm smoke cell."""

from __future__ import annotations

import argparse
import json
import math
import time

import jax
import numpy as np

from hackrl.discrete_sac import (
    init_sd_sac_replay,
    make_sd_sac_replay_updates,
    valid_sd_sac_replay_count,
)
from hackrl.online_algorithm_env import (
    DUAL,
    LEO,
    PQN,
    SD_SAC,
    environment_adapter,
    initialize_online_value_runner,
    initialize_sd_sac_runner,
    make_online_value_update,
    make_sd_sac_collection,
)
from hackrl.pack_restore_gc import PackRestoreGCConfig
from hackrl.tick_claim_gc import TickClaimGCConfig


METHODS = {"pqn": PQN, "leo": LEO, "dual": DUAL, "sd-sac": SD_SAC}


def tier_settings(tier):
    if tier == "cpu":
        return {
            "num_envs": 2,
            "num_steps": 4,
            "pqn_hidden": 8,
            "leo_hidden": 8,
            "sd_hidden": 8,
            "replay_minibatch": 8,
        }
    if tier == "gpu":
        return {
            "num_envs": 512,
            "num_steps": 64,
            "pqn_hidden": 1024,
            "leo_hidden": 512,
            "sd_hidden": 1024,
            "replay_minibatch": 1024,
        }
    raise ValueError(f"unknown smoke tier: {tier!r}")


def environment_config(environment, settings, seed):
    batch_size = settings["num_envs"] * settings["num_steps"]
    common = {
        "seed": int(seed),
        "num_envs": settings["num_envs"],
        "num_steps": settings["num_steps"],
        "num_updates": 1,
        "update_epochs": 1,
        "minibatch_size": batch_size,
        "learning_rate": 2e-4,
        "max_grad_norm": 1.0,
        "goal_mode": "workshop12",
    }
    if environment == "tick":
        return TickClaimGCConfig(hidden_size=8, **common)
    if environment == "pack":
        return PackRestoreGCConfig(
            policy_hidden_size=8, teacher_hidden_size=8, **common
        )
    raise ValueError(f"unknown smoke environment: {environment!r}")


def block_tree(tree):
    return jax.tree.map(
        lambda leaf: leaf.block_until_ready()
        if hasattr(leaf, "block_until_ready")
        else leaf,
        tree,
    )


def tree_is_finite(tree):
    return all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree.leaves(jax.device_get(tree))
    )


def scalarize(value):
    if isinstance(value, dict):
        return {key: scalarize(item) for key, item in value.items()}
    array = np.asarray(jax.device_get(value))
    if array.ndim == 0:
        return array.item()
    return {"shape": list(array.shape), "finite": bool(np.all(np.isfinite(array)))}


def parameter_count(tree):
    return int(sum(np.asarray(leaf).size for leaf in jax.tree.leaves(tree)))


def run_value_cell(adapter, config, method, settings):
    networks, runner = initialize_online_value_runner(
        adapter,
        config,
        method=method,
        pqn_hidden_size=settings["pqn_hidden"],
        leo_hidden_size=settings["leo_hidden"],
        learning_rate=2e-4,
        max_grad_norm=1.0,
    )
    update = jax.jit(
        make_online_value_update(
            adapter,
            config,
            networks,
            method=method,
            epsilon_start=1.0,
            epsilon_finish=0.1,
            epsilon_decay_transitions=16_777_216,
        )
    )
    initial_steps = int(runner.env_steps)
    start = time.monotonic()
    runner, metrics = update(runner)
    block_tree((runner, metrics))
    elapsed = time.monotonic() - start
    if int(runner.env_steps) - initial_steps != config.batch_size:
        raise AssertionError("fixture runner physical-transition count mismatch")
    if method == DUAL:
        components = {
            "pqn": parameter_count(runner.train_state.pqn.params),
            "leo": parameter_count(runner.train_state.leo.params),
        }
    elif method == PQN:
        components = {"pqn": parameter_count(runner.train_state.params)}
    else:
        components = {"leo": parameter_count(runner.train_state.params)}
    if not tree_is_finite((runner.train_state, metrics)):
        raise AssertionError("non-finite online value state or metric")
    return runner, {
        "elapsed_seconds": elapsed,
        "metrics": scalarize(metrics),
        "parameter_counts": components,
    }


def run_sd_sac_cell(adapter, config, settings):
    (actor, critic_1, critic_2), runner = initialize_sd_sac_runner(
        adapter,
        config,
        hidden_size=settings["sd_hidden"],
        actor_learning_rate=2e-4,
        critic_learning_rate=2e-4,
        temperature_learning_rate=2e-4,
        initial_alpha=0.2,
        max_grad_norm=1.0,
    )
    inputs = adapter.batch_inputs(runner.env_state, runner.current_goal)
    replay = init_sd_sac_replay(
        2 * config.batch_size,
        tuple(inputs[0].shape[1:]),
        tuple(inputs[1].shape[1:]),
        (adapter.num_goals,),
    )
    collect = jax.jit(make_sd_sac_collection(adapter, config, actor))
    initial_steps = int(runner.env_steps)
    start = time.monotonic()
    runner, replay, collection_metrics = collect(runner, replay)
    block_tree((runner, replay, collection_metrics))
    valid_replay = int(valid_sd_sac_replay_count(replay))
    if valid_replay <= 0:
        raise AssertionError("SD-SAC smoke collected no valid replay transition")
    replay_update = jax.jit(
        make_sd_sac_replay_updates(
            actor,
            critic_1,
            critic_2,
            batch_size=min(settings["replay_minibatch"], valid_replay),
            update_iterations=1,
            gamma=config.gamma,
            beta=0.1,
            clip_range=0.5,
            tau=0.005,
            target_entropy=0.98 * math.log(adapter.num_actions),
        )
    )
    state, rng, learning_metrics = replay_update(
        runner.train_state, replay, runner.rng
    )
    runner = runner.replace(
        train_state=state, rng=rng, global_update=runner.global_update + 1
    )
    block_tree((runner, learning_metrics))
    elapsed = time.monotonic() - start
    if int(runner.env_steps) - initial_steps != config.batch_size:
        raise AssertionError("fixture runner physical-transition count mismatch")
    if int(runner.train_state.environment_steps) != config.batch_size:
        raise AssertionError("SD-SAC physical-transition accounting mismatch")
    if not tree_is_finite((runner.train_state, learning_metrics)):
        raise AssertionError("non-finite SD-SAC state or metric")
    components = {
        "actor": parameter_count(runner.train_state.actor.params),
        "critic_1": parameter_count(runner.train_state.critic_1.params),
        "critic_2": parameter_count(runner.train_state.critic_2.params),
        "target_critic_1": parameter_count(
            runner.train_state.critic_1.target_params
        ),
        "target_critic_2": parameter_count(
            runner.train_state.critic_2.target_params
        ),
        "temperature": parameter_count(runner.train_state.temperature.params),
    }
    return runner, {
        "elapsed_seconds": elapsed,
        "collection_metrics": scalarize(collection_metrics),
        "learning_metrics": scalarize(learning_metrics),
        "valid_replay_transitions": valid_replay,
        "parameter_counts": components,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--environment", choices=("tick", "pack"), required=True)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--seed", type=int, default=109)
    args = parser.parse_args()

    backend = jax.default_backend()
    if backend != args.tier:
        raise RuntimeError(
            f"requested {args.tier!r} smoke but JAX default backend is {backend!r}"
        )
    settings = tier_settings(args.tier)
    config = environment_config(args.environment, settings, args.seed)
    adapter = environment_adapter(args.environment)
    method = METHODS[args.method]
    started = time.monotonic()
    if method == SD_SAC:
        runner, details = run_sd_sac_cell(adapter, config, settings)
    else:
        runner, details = run_value_cell(adapter, config, method, settings)
    result = {
        "schema_version": "hackrl_online_algorithm_smoke_cell_v1",
        "success": True,
        "tier": args.tier,
        "backend": backend,
        "device": str(jax.devices()[0]),
        "environment": adapter.name,
        "method": method,
        "seed": args.seed,
        "num_envs": config.num_envs,
        "num_steps": config.num_steps,
        "physical_transitions": config.batch_size,
        "runner_environment_steps": int(runner.env_steps),
        "global_update": int(runner.global_update),
        "wall_time_seconds": time.monotonic() - started,
        "details": details,
        "checkpoint_written": False,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
