#!/usr/bin/env python
"""D2 reach/mine split and leftover mutant fail-trajectory dumps."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax.serialization import from_bytes

from hackrl import FixtureDynamics, HackRLEasySymbolicEnvNoAutoReset, MediumTask, StartMode
from hackrl.ppo import ActorCritic

FRONT = jnp.array((9, 8), dtype=jnp.int32)
D2_ROOT = Path("/home/ext_csv/HackRL/runs/post_stage_a_d2")
A2X_ROOT = Path("/home/ext_csv/HackRL/runs/overnight_stage_a")


def _load(env, seed, path):
    network = ActorCritic(
        action_dim=env.action_space(env.default_params).n,
        layer_width=256,
        activation="tanh",
    )
    rng = jax.random.PRNGKey(seed)
    dummy, _ = env.reset(rng, env.default_params)
    return network, from_bytes(network.init(rng, dummy), path.read_bytes())


def _batch_rollout(network, params, env, rng, n, stochastic):
    reset = jax.vmap(lambda key: env.reset(key, env.default_params))
    step = jax.vmap(
        lambda key, state, action: env.step(key, state, action, env.default_params)
    )

    def run(rng):
        rng, reset_rng = jax.random.split(rng)
        observations, env_state = reset(jax.random.split(reset_rng, n))
        recorded = jnp.zeros((n,), dtype=bool)
        success = jnp.zeros((n,), dtype=bool)
        reached = jnp.zeros((n,), dtype=bool)
        mined_after = jnp.zeros((n,), dtype=bool)
        first_action = jnp.zeros((n,), dtype=jnp.int32)
        lengths = jnp.zeros((n,), dtype=jnp.int32)
        death = jnp.zeros((n,), dtype=bool)
        final_pos = jnp.zeros((n, 2), dtype=jnp.int32)
        final_dir = jnp.zeros((n,), dtype=jnp.int32)
        final_iron = jnp.zeros((n,), dtype=jnp.int32)
        final_pickaxe = jnp.zeros((n,), dtype=jnp.int32)
        final_diamond = jnp.zeros((n,), dtype=jnp.int32)

        def body(carry, step_index):
            (
                observations,
                env_state,
                recorded,
                success,
                reached,
                mined_after,
                first_action,
                lengths,
                death,
                final_pos,
                final_dir,
                final_iron,
                final_pickaxe,
                final_diamond,
                rng,
            ) = carry
            rng, action_rng, step_rng = jax.random.split(rng, 3)
            policy, _ = network.apply(params, observations)
            actions = jnp.where(
                stochastic,
                policy.sample(seed=action_rng).astype(jnp.int32),
                policy.mode().astype(jnp.int32),
            )
            first_action = jnp.where(step_index == 0, actions, first_action)
            observations, env_state, rewards, dones, infos = step(
                jax.random.split(step_rng, n), env_state, actions
            )
            at_front = jnp.logical_and(
                jnp.all(env_state.player_position == FRONT, axis=-1),
                env_state.inventory.diamond == 0,
            )
            now_reached = reached | at_front
            newly = jnp.logical_and(~recorded, dones)
            return (
                observations,
                env_state,
                recorded | newly,
                jnp.where(newly, infos["HackRL/goal_success"], success),
                now_reached,
                mined_after
                | jnp.logical_and(now_reached, infos["HackRL/diamond_acquired"]),
                first_action,
                jnp.where(newly, env_state.timestep, lengths),
                jnp.where(newly, infos["HackRL/termination_death"], death),
                jnp.where(newly[:, None], env_state.player_position, final_pos),
                jnp.where(newly, env_state.player_direction, final_dir),
                jnp.where(newly, env_state.inventory.iron, final_iron),
                jnp.where(newly, env_state.inventory.iron_pickaxe, final_pickaxe),
                jnp.where(newly, env_state.inventory.diamond, final_diamond),
                rng,
            ), None

        carry, _ = jax.lax.scan(
            body,
            (
                observations,
                env_state,
                recorded,
                success,
                reached,
                mined_after,
                first_action,
                lengths,
                death,
                final_pos,
                final_dir,
                final_iron,
                final_pickaxe,
                final_diamond,
                rng,
            ),
            jnp.arange(env.spec.horizon),
        )
        return carry[2:14]

    values = jax.device_get(jax.jit(run)(rng))
    recorded, success, reached, mined_after, first_action, lengths, death = values[:7]
    final_pos, final_dir, final_iron, final_pickaxe, final_diamond = values[7:]
    episodes = []
    for i in range(n):
        episodes.append(
            {
                "index": i,
                "success": bool(success[i]),
                "reached_front": bool(reached[i]),
                "mined_after_reach": bool(mined_after[i]),
                "length": int(lengths[i]),
                "first_action": int(first_action[i]),
                "termination_death": bool(death[i]),
                "final": {
                    "position": [int(final_pos[i, 0]), int(final_pos[i, 1])],
                    "direction": int(final_dir[i]),
                    "iron": int(final_iron[i]),
                    "iron_pickaxe": int(final_pickaxe[i]),
                    "diamond": int(final_diamond[i]),
                },
            }
        )
    return episodes


def _summarize(episodes):
    n = len(episodes)
    reached = sum(item["reached_front"] for item in episodes)
    mined = sum(item["mined_after_reach"] for item in episodes)
    success = sum(item["success"] for item in episodes)
    return {
        "n": n,
        "success_rate": success / n if n else None,
        "reach_front_rate": reached / n if n else None,
        "mine_after_reach_rate": (mined / reached) if reached else None,
        "reached_but_no_mine": reached - mined,
        "success": success,
        "reached": reached,
        "mined_after_reach": mined,
    }


def diagnose_d2():
    env = HackRLEasySymbolicEnvNoAutoReset(
        MediumTask.R_M,
        mutant=False,
        start_mode=StartMode.R_M_D2,
        dynamics=FixtureDynamics.PATCHED,
    )
    report = []
    for seed in (0, 1, 2):
        cell = D2_ROOT / f"d2_patched_fixed_s{seed}"
        item = {"cell": cell.name, "checkpoints": {}}
        rng = jax.random.PRNGKey(20260929 + seed)
        for tag in ("transitions_0", "transitions_1048576"):
            network, params = _load(
                env, seed, cell / "checkpoints" / tag / "params.msgpack"
            )
            rng, mode_rng, sample_rng = jax.random.split(rng, 3)
            mode = _batch_rollout(network, params, env, mode_rng, 32, False)
            sample = _batch_rollout(network, params, env, sample_rng, 64, True)
            item["checkpoints"][tag] = {
                "mode_32": _summarize(mode),
                "sample_64": _summarize(sample),
                "mode_fail_examples": [ep for ep in mode if not ep["success"]][:4],
                "sample_fail_examples": [ep for ep in sample if not ep["success"]][:4],
            }
        report.append(item)
        print(
            json.dumps(
                {
                    "cell": item["cell"],
                    "checkpoints": {
                        tag: {
                            "mode_32": data["mode_32"],
                            "sample_64": data["sample_64"],
                        }
                        for tag, data in item["checkpoints"].items()
                    },
                },
                indent=2,
            )
        )
    return report


def diagnose_mutant_fails():
    env = HackRLEasySymbolicEnvNoAutoReset(
        MediumTask.R_M, mutant=True, dynamics=FixtureDynamics.PATCHED
    )
    report = []
    for seed in (0, 1, 2):
        cell = A2X_ROOT / f"a2x_patched_mutant_s{seed}"
        path = cell / "checkpoints" / "transitions_16777216" / "params.msgpack"
        network, params = _load(env, seed, path)
        episodes = _batch_rollout(
            network, params, env, jax.random.PRNGKey(20260929 + 100 + seed), 8, False
        )
        item = {
            "cell": cell.name,
            "checkpoint": "transitions_16777216",
            "mode_8": _summarize(episodes),
            "fail_trajectories": [ep for ep in episodes if not ep["success"]],
        }
        report.append(item)
        print(json.dumps({"cell": item["cell"], "mode_8": item["mode_8"]}, indent=2))
    return report


def main():
    payload = {
        "d2_reach_mine": diagnose_d2(),
        "mutant_fail_trajectories": diagnose_mutant_fails(),
    }
    destination = D2_ROOT / "d2_reach_mine.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    mutant_path = A2X_ROOT / "a2x_mutant_fail_trajectories.json"
    mutant_path.write_text(
        json.dumps(payload["mutant_fail_trajectories"], indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {destination}")
    print(f"wrote {mutant_path}")


if __name__ == "__main__":
    main()
