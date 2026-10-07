"""Goal-conditioned Double DQN shared by the two controlled bug fixtures.

The agent reuses the Dual teacher's multi-head Q architecture so that the
comparison changes the learning/execution algorithm, not the Q backbone.
Unlike Dual, this state is initialized independently, chooses actions directly
from Q, has no PPO policy or behavior-cloning loss, and learns from replay with
a hard target network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization, struct
from flax.training.train_state import TrainState

from hackrl.dual_leo import DualLeoQ


class DoubleDQNTrainState(TrainState):
    batch_stats: Any = None
    target_params: Any = None
    target_batch_stats: Any = None


class ReplayBuffer(struct.PyTreeNode):
    map_channels: jax.Array
    numeric_features: jax.Array
    goal_index: jax.Array
    action: jax.Array
    reward: jax.Array
    next_map_channels: jax.Array
    next_numeric_features: jax.Array
    next_goal_index: jax.Array
    done: jax.Array
    valid: jax.Array
    cursor: jax.Array
    size: jax.Array
    total_inserted: jax.Array


class ReplayTransition(struct.PyTreeNode):
    map_channels: jax.Array
    numeric_features: jax.Array
    goal_index: jax.Array
    action: jax.Array
    reward: jax.Array
    next_map_channels: jax.Array
    next_numeric_features: jax.Array
    next_goal_index: jax.Array
    done: jax.Array
    valid: jax.Array
    explored: jax.Array


def select_goal_q(q_values, goal_index):
    """Select [batch, action] values from [batch, goal, action]."""

    return jnp.take_along_axis(
        q_values,
        goal_index[..., None, None],
        axis=-2,
    )[..., 0, :]


def double_dqn_targets(
    online_next_q,
    target_next_q,
    reward,
    done,
    gamma,
):
    """Online selection with target-network evaluation."""

    next_action = jnp.argmax(online_next_q, axis=-1)
    bootstrap = jnp.take_along_axis(
        target_next_q, next_action[..., None], axis=-1
    )[..., 0]
    return reward + gamma * bootstrap * (1.0 - done.astype(jnp.float32))


def exploration_epsilon(
    environment_steps,
    *,
    start: float,
    end: float,
    decay_transitions: int,
):
    fraction = jnp.minimum(
        jnp.asarray(environment_steps, dtype=jnp.float32)
        / float(max(int(decay_transitions), 1)),
        1.0,
    )
    return float(start) + fraction * (float(end) - float(start))


def init_double_dqn(
    key,
    map_example,
    numeric_example,
    *,
    num_goals: int,
    num_actions: int,
    hidden_size: int,
    learning_rate: float,
    max_grad_norm: float,
):
    network = DualLeoQ(
        num_goals=int(num_goals),
        num_actions=int(num_actions),
        hidden_size=int(hidden_size),
    )
    variables = network.init(
        key,
        map_example[:1],
        numeric_example[:1],
        train=False,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(max_grad_norm)),
        optax.adam(float(learning_rate), eps=1e-5),
    )
    state = DoubleDQNTrainState.create(
        apply_fn=network.apply,
        params=variables["params"],
        tx=optimizer,
        batch_stats=variables["batch_stats"],
        target_params=variables["params"],
        target_batch_stats=variables["batch_stats"],
    )
    return network, state


def init_replay_buffer(capacity, map_shape, numeric_shape):
    capacity = int(capacity)
    if capacity <= 0:
        raise ValueError("replay capacity must be positive")
    return ReplayBuffer(
        map_channels=jnp.zeros((capacity,) + tuple(map_shape), dtype=jnp.bfloat16),
        numeric_features=jnp.zeros(
            (capacity,) + tuple(numeric_shape), dtype=jnp.float32
        ),
        goal_index=jnp.zeros((capacity,), dtype=jnp.int32),
        action=jnp.zeros((capacity,), dtype=jnp.int32),
        reward=jnp.zeros((capacity,), dtype=jnp.float32),
        next_map_channels=jnp.zeros(
            (capacity,) + tuple(map_shape), dtype=jnp.bfloat16
        ),
        next_numeric_features=jnp.zeros(
            (capacity,) + tuple(numeric_shape), dtype=jnp.float32
        ),
        next_goal_index=jnp.zeros((capacity,), dtype=jnp.int32),
        done=jnp.zeros((capacity,), dtype=jnp.bool_),
        valid=jnp.zeros((capacity,), dtype=jnp.bool_),
        cursor=jnp.asarray(0, dtype=jnp.int32),
        size=jnp.asarray(0, dtype=jnp.int32),
        total_inserted=jnp.asarray(0, dtype=jnp.int32),
    )


def append_replay(replay, transition):
    count = int(transition.action.shape[0])
    capacity = int(replay.action.shape[0])
    if count > capacity:
        raise ValueError("one replay append cannot exceed capacity")
    positions = (
        replay.cursor + jnp.arange(count, dtype=jnp.int32)
    ) % capacity
    return replay.replace(
        map_channels=replay.map_channels.at[positions].set(
            transition.map_channels.astype(jnp.bfloat16)
        ),
        numeric_features=replay.numeric_features.at[positions].set(
            transition.numeric_features
        ),
        goal_index=replay.goal_index.at[positions].set(transition.goal_index),
        action=replay.action.at[positions].set(transition.action),
        reward=replay.reward.at[positions].set(transition.reward),
        next_map_channels=replay.next_map_channels.at[positions].set(
            transition.next_map_channels.astype(jnp.bfloat16)
        ),
        next_numeric_features=replay.next_numeric_features.at[positions].set(
            transition.next_numeric_features
        ),
        next_goal_index=replay.next_goal_index.at[positions].set(
            transition.next_goal_index
        ),
        done=replay.done.at[positions].set(transition.done),
        valid=replay.valid.at[positions].set(transition.valid),
        cursor=(replay.cursor + count) % capacity,
        size=jnp.minimum(capacity, replay.size + count),
        total_inserted=replay.total_inserted + count,
    )


def _sample_replay(replay, indices):
    return {
        "map_channels": replay.map_channels[indices].astype(jnp.float32),
        "numeric_features": replay.numeric_features[indices],
        "goal_index": replay.goal_index[indices],
        "action": replay.action[indices],
        "reward": replay.reward[indices],
        "next_map_channels": replay.next_map_channels[indices].astype(jnp.float32),
        "next_numeric_features": replay.next_numeric_features[indices],
        "next_goal_index": replay.next_goal_index[indices],
        "done": replay.done[indices],
        "valid": replay.valid[indices],
    }


def make_replay_updates(
    network,
    *,
    batch_size: int,
    gradient_steps: int,
    target_update_interval: int,
    gamma: float,
):
    batch_size = int(batch_size)
    gradient_steps = int(gradient_steps)
    target_update_interval = int(target_update_interval)
    if min(batch_size, gradient_steps, target_update_interval) <= 0:
        raise ValueError("DQN update sizes and target interval must be positive")

    def one_step(carry, indices):
        state, replay = carry
        batch = _sample_replay(replay, indices)

        online_next_all = network.apply(
            {"params": state.params, "batch_stats": state.batch_stats},
            batch["next_map_channels"],
            batch["next_numeric_features"],
            train=False,
        )
        target_next_all = network.apply(
            {
                "params": state.target_params,
                "batch_stats": state.target_batch_stats,
            },
            batch["next_map_channels"],
            batch["next_numeric_features"],
            train=False,
        )
        online_next = select_goal_q(online_next_all, batch["next_goal_index"])
        target_next = select_goal_q(target_next_all, batch["next_goal_index"])
        targets = jax.lax.stop_gradient(
            double_dqn_targets(
                online_next,
                target_next,
                batch["reward"],
                batch["done"],
                gamma,
            )
        )
        valid = batch["valid"].astype(jnp.float32)
        valid_count = jnp.sum(valid)

        def loss_fn(params):
            values, mutable = network.apply(
                {"params": params, "batch_stats": state.batch_stats},
                batch["map_channels"],
                batch["numeric_features"],
                train=True,
                mutable=["batch_stats"],
            )
            selected_goal = select_goal_q(values, batch["goal_index"])
            chosen = jnp.take_along_axis(
                selected_goal, batch["action"][:, None], axis=-1
            )[:, 0]
            losses = optax.huber_loss(chosen, targets)
            loss = jnp.sum(losses * valid) / jnp.maximum(valid_count, 1.0)
            auxiliary = (
                mutable["batch_stats"],
                jnp.sum(chosen * valid) / jnp.maximum(valid_count, 1.0),
                jnp.sum(targets * valid) / jnp.maximum(valid_count, 1.0),
            )
            return loss, auxiliary

        (loss, (batch_stats, mean_q, mean_target)), gradients = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        candidate = state.apply_gradients(
            grads=gradients,
            batch_stats=batch_stats,
        )
        applied = valid_count > 0
        state = jax.lax.cond(applied, lambda _: candidate, lambda _: state, None)
        sync = jnp.logical_and(
            applied,
            state.step % target_update_interval == 0,
        )
        target_params = jax.tree.map(
            lambda online, target: jnp.where(sync, online, target),
            state.params,
            state.target_params,
        )
        target_batch_stats = jax.tree.map(
            lambda online, target: jnp.where(sync, online, target),
            state.batch_stats,
            state.target_batch_stats,
        )
        state = state.replace(
            target_params=target_params,
            target_batch_stats=target_batch_stats,
        )
        metrics = {
            "loss": loss,
            "mean_q": mean_q,
            "mean_target": mean_target,
            "valid_samples": valid_count,
            "applied": applied.astype(jnp.int32),
            "target_sync": sync.astype(jnp.int32),
        }
        return (state, replay), metrics

    def update(state, replay, rng):
        rng, sample_rng = jax.random.split(rng)
        positions = jnp.arange(replay.action.shape[0], dtype=jnp.int32)
        eligible = jnp.logical_and(
            positions < replay.size, replay.valid
        ).astype(jnp.float32)
        eligible_count = jnp.sum(eligible)
        probabilities = jnp.where(
            eligible_count > 0,
            eligible / jnp.maximum(eligible_count, 1.0),
            (positions == 0).astype(jnp.float32),
        )
        indices = jax.random.choice(
            sample_rng,
            replay.action.shape[0],
            (gradient_steps, batch_size),
            replace=True,
            p=probabilities,
        )
        (state, _), rows = jax.lax.scan(one_step, (state, replay), indices)
        metrics = {
            "loss": jnp.mean(rows["loss"]),
            "mean_q": jnp.mean(rows["mean_q"]),
            "mean_target": jnp.mean(rows["mean_target"]),
            "sampled_valid_transitions": jnp.sum(rows["valid_samples"]),
            "applied_gradient_steps": jnp.sum(rows["applied"]),
            "target_syncs": jnp.sum(rows["target_sync"]),
        }
        return state, rng, metrics

    return update


def flatten_rollout(trajectory):
    return jax.tree.map(
        lambda value: value.reshape((-1,) + value.shape[2:]),
        trajectory,
    )


def make_double_dqn_update(
    network,
    env_config,
    batch_inputs: Callable,
    step_outcome: Callable,
    *,
    num_actions: int,
    replay_batch_size: int,
    gradient_steps_per_rollout: int,
    target_update_interval: int,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_transitions: int,
):
    replay_update = make_replay_updates(
        network,
        batch_size=replay_batch_size,
        gradient_steps=gradient_steps_per_rollout,
        target_update_interval=target_update_interval,
        gamma=gamma,
    )

    def update(runner, state, replay):
        def rollout_step(current_runner, _):
            maps, numeric, _ = batch_inputs(
                current_runner.env_state, current_runner.current_goal
            )
            q_values = network.apply(
                {"params": state.params, "batch_stats": state.batch_stats},
                maps,
                numeric,
                train=False,
            )
            selected = select_goal_q(q_values, current_runner.current_goal)
            greedy = jnp.argmax(selected, axis=-1).astype(jnp.int32)
            rng, explore_rng, random_action_rng = jax.random.split(
                current_runner.rng, 3
            )
            epsilon = exploration_epsilon(
                current_runner.env_steps,
                start=epsilon_start,
                end=epsilon_end,
                decay_transitions=epsilon_decay_transitions,
            )
            explore = jax.random.uniform(
                explore_rng, (env_config.num_envs,)
            ) < epsilon
            random_action = jax.random.randint(
                random_action_rng,
                (env_config.num_envs,),
                minval=0,
                maxval=num_actions,
                dtype=jnp.int32,
            )
            action = jnp.where(explore, random_action, greedy)
            current_runner = current_runner.replace(rng=rng)
            next_runner, done, valid, reward = step_outcome(
                current_runner, action, env_config
            )
            next_maps, next_numeric, _ = batch_inputs(
                next_runner.env_state, next_runner.current_goal
            )
            transition = ReplayTransition(
                map_channels=maps,
                numeric_features=numeric,
                goal_index=current_runner.current_goal,
                action=action,
                reward=reward,
                next_map_channels=next_maps,
                next_numeric_features=next_numeric,
                next_goal_index=next_runner.current_goal,
                done=done,
                valid=valid,
                explored=explore,
            )
            return next_runner, transition

        runner, trajectory = jax.lax.scan(
            rollout_step, runner, None, length=env_config.num_steps
        )
        runner = runner.replace(global_update=runner.global_update + 1)
        replay = append_replay(replay, flatten_rollout(trajectory))
        state, rng, learning = replay_update(state, replay, runner.rng)
        runner = runner.replace(rng=rng)
        metrics = {
            **learning,
            "epsilon": exploration_epsilon(
                runner.env_steps,
                start=epsilon_start,
                end=epsilon_end,
                decay_transitions=epsilon_decay_transitions,
            ),
            "explored_transitions": jnp.sum(trajectory.explored),
            "valid_environment_transitions": jnp.sum(trajectory.valid),
            "goal_completions": jnp.sum(trajectory.reward),
            "replay_size": replay.size,
            "replay_total_inserted": replay.total_inserted,
            "environment_steps": runner.env_steps,
            "global_update": runner.global_update,
            "gradient_steps": state.step,
        }
        return runner, state, replay, metrics

    return update


def parameter_count(state) -> int:
    return int(
        sum(np.asarray(value).size for value in jax.tree.leaves(state.params))
    )


def _write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_checkpoint(
    directory,
    *,
    runner,
    train_state,
    replay,
    environment_config,
    algorithm_config,
):
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    blob = {"runner": runner, "train_state": train_state, "replay": replay}
    temporary = destination / "state.msgpack.tmp"
    temporary.write_bytes(serialization.to_bytes(blob))
    temporary.replace(destination / "state.msgpack")
    _write_json_atomic(
        destination / "config.json",
        {
            "environment": environment_config,
            "algorithm": algorithm_config,
        },
    )
    _write_json_atomic(
        destination / "metadata.json",
        {
            "schema_version": "hackrl_gc_double_dqn_checkpoint_v1",
            "global_update": int(runner.global_update),
            "environment_steps": int(runner.env_steps),
            "gradient_steps": int(train_state.step),
            "replay_size": int(replay.size),
            "replay_total_inserted": int(replay.total_inserted),
            "parameters": parameter_count(train_state),
        },
    )
    return destination


def load_checkpoint(directory, *, runner, train_state, replay):
    source = Path(directory)
    return serialization.from_bytes(
        {"runner": runner, "train_state": train_state, "replay": replay},
        (source / "state.msgpack").read_bytes(),
    )
