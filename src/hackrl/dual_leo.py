"""Dual LEO teacher for the existing GC-PPO update.

The PPO loss stays on the commanded goal's valid on-policy samples. The teacher
learns all 12 goals from those same samples. A goal head bootstraps unless that
goal's terminal predicate is true or the world actually resets. Policy imitation
is 0.1 at update 0 and decays linearly across the fixed 4608-update horizon.
Value imitation stays 0. The schedule continues through adaptation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import serialization, struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from hackrl.batch_renorm import BatchRenorm

BC_POLICY_COEF = 0.1
BC_VALUE_COEF = 0.0
BC_HORIZON_UPDATES = 512 + 4096
LEO_EPOCHS = 2
LEO_MINIBATCH_SIZE = 512
LEO_LR = 2e-4
LEO_HIDDEN_SIZE = 512
LEO_CONV_FEATURES = 16


def bc_policy_coefficient(update_index):
    """Linear decay over the locked pretrain-plus-adaptation horizon."""

    remaining = 1.0 - jnp.asarray(update_index, dtype=jnp.float32) / BC_HORIZON_UPDATES
    return BC_POLICY_COEF * jnp.maximum(0.0, remaining)


def dual_leo_q_targets(achieved, max_next_q, world_done, gamma):
    """Per-goal bootstrap. Command switching is not a terminal for other heads."""

    done = jnp.logical_or(achieved, world_done[..., None])
    reward = achieved.astype(jnp.float32)
    return reward + gamma * max_next_q * (1.0 - done.astype(jnp.float32))


class DualLeoState(TrainState):
    """PPO teacher state. batch_stats are the input BatchRenorm running values."""

    batch_stats: Any = None


class DualLeoQ(nn.Module):
    """All-goal Q head. Inputs use BatchRenorm; hidden layers use layer norm.

    Official LEO raises if input normalization is off. This module always
    applies it. `train=False` reads the running statistics and does not update them.
    """

    num_goals: int
    num_actions: int
    hidden_size: int = LEO_HIDDEN_SIZE

    @nn.compact
    def __call__(self, map_channels, numeric_features, *, train: bool):
        map_channels = BatchRenorm(use_running_average=not train, name="map_input_renorm")(
            map_channels
        )
        numeric_features = BatchRenorm(
            use_running_average=not train, name="numeric_input_renorm"
        )(numeric_features)
        spatial = nn.Conv(
            features=LEO_CONV_FEATURES,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=constant(0.0),
            name="map_conv",
        )(map_channels)
        spatial = nn.LayerNorm(name="map_norm")(spatial)
        spatial = nn.relu(spatial).reshape((spatial.shape[0], -1))
        hidden = jnp.concatenate((spatial, numeric_features), axis=-1)
        for index in range(4):
            hidden = nn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name=f"q_hidden_{index}",
            )(hidden)
            hidden = nn.LayerNorm(name=f"q_norm_{index}")(hidden)
            hidden = nn.relu(hidden)
        logits = nn.Dense(
            self.num_goals * self.num_actions,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="q_output",
        )(hidden)
        values = logits.reshape((logits.shape[0], self.num_goals, self.num_actions))
        return jax.nn.sigmoid(values)


class _DualTransition(struct.PyTreeNode):
    done: jax.Array
    valid: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    map_channels: jax.Array
    numeric_features: jax.Array
    goal_one_hot: jax.Array
    goal_index: jax.Array
    q_values: jax.Array
    terminal_goals: jax.Array
    world_done: jax.Array
    goal_done: jax.Array


class _PpoBatch(struct.PyTreeNode):
    transition: _DualTransition
    advantages: jax.Array
    targets: jax.Array


def _weighted_mean(value, mask):
    mask = mask.astype(jnp.float32)
    return jnp.sum(value * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _gae(trajectory, last_value, gamma, gae_lambda):
    def backward(carry, transition):
        gae, next_value = carry
        nonterminal = 1.0 - transition.done.astype(jnp.float32)
        delta = transition.reward + gamma * next_value * nonterminal - transition.value
        gae = (delta + gamma * gae_lambda * nonterminal * gae) * transition.valid.astype(
            jnp.float32
        )
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        backward, (jnp.zeros_like(last_value), last_value), trajectory, reverse=True
    )
    return advantages, advantages + trajectory.value


def init_dual_leo_teacher(config, map_example, numeric_example, num_goals, num_actions):
    """Initialize the teacher without consuming the PPO initialization RNG."""

    teacher_hidden_size = int(getattr(config, "teacher_hidden_size", LEO_HIDDEN_SIZE))
    network = DualLeoQ(
        num_goals=num_goals,
        num_actions=num_actions,
        hidden_size=teacher_hidden_size,
    )
    variables = network.init(
        jax.random.fold_in(jax.random.PRNGKey(int(config.seed)), 17),
        map_example[:1],
        numeric_example[:1],
        train=False,
    )
    batch = int(config.num_envs) * int(config.num_steps)
    if batch % LEO_MINIBATCH_SIZE and batch < LEO_MINIBATCH_SIZE:
        minibatch = batch
    else:
        minibatch = LEO_MINIBATCH_SIZE
    if batch % minibatch:
        raise ValueError("teacher minibatch must divide the rollout batch")
    minibatches = batch // minibatch
    schedule = optax.linear_schedule(
        init_value=LEO_LR,
        end_value=1e-20,
        transition_steps=BC_HORIZON_UPDATES * minibatches * LEO_EPOCHS,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(schedule, eps=1e-5),
    )
    return (
        network,
        DualLeoState.create(
            apply_fn=network.apply,
            params=variables["params"],
            tx=tx,
            batch_stats=variables["batch_stats"],
        ),
        minibatch,
    )


def _teacher_q(network, leo_state, map_channels, numeric_features, *, train):
    variables = {"params": leo_state.params, "batch_stats": leo_state.batch_stats}
    if train:
        predicted, updates = network.apply(
            variables, map_channels, numeric_features, train=True, mutable=["batch_stats"]
        )
        return predicted, updates["batch_stats"]
    return network.apply(variables, map_channels, numeric_features, train=False), None


def teacher_parameter_count(leo_state) -> int:
    return int(sum(np.asarray(value).size for value in jax.tree.leaves(leo_state.params)))


def make_dual_leo_update(
    ppo_network,
    teacher_network,
    config,
    step_outcome,
    batch_inputs,
    minibatch_size,
    *,
    learn_teacher=True,
    imitate_teacher=True,
):
    """One PPO update plus the all-goal teacher update.

    step_outcome(runner, actions, config) returns
    runner, done, valid, reward, terminal_goals, world_done, goal_done.

    learn_teacher=False keeps the teacher parameters, Adam state, and input
    BatchRenorm statistics at their current values. The teacher shuffle still
    consumes gc_runner.rng, so turning learning off does not change the policy
    random stream. imitate_teacher=False multiplies the policy BC term by 0.
    """

    num_goals = teacher_network.num_goals
    leo_epochs = LEO_EPOCHS
    bc_horizon = BC_HORIZON_UPDATES

    def update(gc_runner, leo_state):
        def rollout_step(carry, _):
            runner, leo = carry
            command_goal = runner.current_goal
            model_inputs = batch_inputs(runner.env_state, command_goal)
            policy, value = ppo_network.apply(runner.train_state.params, *model_inputs)
            q_values, _ = _teacher_q(
                teacher_network, leo, model_inputs[0], model_inputs[1], train=False
            )
            q_values = jax.lax.stop_gradient(q_values)
            rng, action_rng = jax.random.split(runner.rng)
            actions = policy.sample(seed=action_rng)
            log_prob = policy.log_prob(actions)
            runner = runner.replace(rng=rng)
            (
                runner,
                done,
                valid,
                reward,
                terminal_goals,
                world_done,
                goal_done,
                observed_goals,
            ) = step_outcome(runner, actions, config)
            transition = _DualTransition(
                done=done,
                valid=valid,
                action=actions,
                value=value,
                reward=reward,
                log_prob=log_prob,
                map_channels=model_inputs[0],
                numeric_features=model_inputs[1],
                goal_one_hot=jax.nn.one_hot(command_goal, num_goals, dtype=jnp.float32),
                goal_index=command_goal,
                q_values=q_values,
                terminal_goals=terminal_goals,
                world_done=world_done,
                goal_done=goal_done,
            )
            return (runner, leo), (transition, observed_goals)

        (gc_runner, leo_state), (trajectory, observed_goals) = jax.lax.scan(
            rollout_step, (gc_runner, leo_state), None, length=config.num_steps
        )
        seen_goals = jnp.logical_or(
            gc_runner.seen_goals, jnp.any(observed_goals, axis=(0, 1))
        )
        last_inputs = batch_inputs(gc_runner.env_state, gc_runner.current_goal)
        _, last_value = ppo_network.apply(gc_runner.train_state.params, *last_inputs)
        last_value = jnp.where(gc_runner.command_active, last_value, 0.0)
        advantages, targets = _gae(trajectory, last_value, config.gamma, config.gae_lambda)
        last_q, _ = _teacher_q(
            teacher_network, leo_state, last_inputs[0], last_inputs[1], train=False
        )
        max_next_q = jnp.concatenate((trajectory.q_values[1:], last_q[None]), axis=0).max(
            axis=-1
        )
        q_targets = dual_leo_q_targets(
            trajectory.terminal_goals, max_next_q, trajectory.world_done, config.gamma
        )
        bc_coef = bc_policy_coefficient(gc_runner.global_update)
        if not imitate_teacher:
            bc_coef = jnp.asarray(0.0, dtype=jnp.float32)

        def update_minibatch(train_state, batch):
            def loss_fn(parameters):
                policy, value = ppo_network.apply(
                    parameters,
                    batch.transition.map_channels,
                    batch.transition.numeric_features,
                    batch.transition.goal_one_hot,
                )
                log_prob = policy.log_prob(batch.transition.action)
                valid = batch.transition.valid
                advantage_mean = _weighted_mean(batch.advantages, valid)
                advantage_variance = _weighted_mean(
                    jnp.square(batch.advantages - advantage_mean), valid
                )
                normalized = (batch.advantages - advantage_mean) / jnp.sqrt(
                    advantage_variance + 1e-8
                )
                clipped_value = batch.transition.value + (value - batch.transition.value).clip(
                    -config.clip_epsilon, config.clip_epsilon
                )
                value_loss = 0.5 * _weighted_mean(
                    jnp.maximum(
                        jnp.square(value - batch.targets),
                        jnp.square(clipped_value - batch.targets),
                    ),
                    valid,
                )
                ratio = jnp.exp(log_prob - batch.transition.log_prob)
                actor_loss = -_weighted_mean(
                    jnp.minimum(
                        ratio * normalized,
                        jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
                        * normalized,
                    ),
                    valid,
                )
                entropy = _weighted_mean(policy.entropy(), valid)
                q_values = batch.transition.q_values
                goal_selector = jax.nn.one_hot(
                    batch.transition.goal_index, q_values.shape[1], dtype=q_values.dtype
                )
                q_acting = jax.lax.stop_gradient(
                    jnp.sum(q_values * goal_selector[:, :, None], axis=1)
                )
                teacher_action = jnp.argmax(q_acting, axis=-1)
                bc_policy_loss = _weighted_mean(-policy.log_prob(teacher_action), valid)
                bc_value_loss = 0.5 * _weighted_mean(
                    jnp.square(value - jnp.max(q_acting, axis=-1)), valid
                )
                total = (
                    actor_loss
                    + config.value_coefficient * value_loss
                    - config.entropy_coefficient * entropy
                    + bc_coef * bc_policy_loss
                    + BC_VALUE_COEF * bc_value_loss
                )
                return total, (value_loss, actor_loss, entropy, bc_policy_loss, bc_value_loss)

            valid_count = jnp.sum(batch.transition.valid.astype(jnp.int32))

            def apply_update(state):
                (loss, auxiliary), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
                    state.params
                )
                state = state.apply_gradients(grads=gradients)
                value_loss, policy_loss, entropy, bc_policy_loss, bc_value_loss = auxiliary
                return state, {
                    "loss": loss.astype(jnp.float32),
                    "value_loss": value_loss.astype(jnp.float32),
                    "policy_loss": policy_loss.astype(jnp.float32),
                    "entropy": entropy.astype(jnp.float32),
                    "bc_policy_loss": bc_policy_loss.astype(jnp.float32),
                    "bc_value_loss": bc_value_loss.astype(jnp.float32),
                    "empty_minibatch": jnp.asarray(0, dtype=jnp.int32),
                }

            def skip_update(state):
                zero = jnp.asarray(0.0, dtype=jnp.float32)
                return state, {
                    "loss": zero,
                    "value_loss": zero,
                    "policy_loss": zero,
                    "entropy": zero,
                    "bc_policy_loss": zero,
                    "bc_value_loss": zero,
                    "empty_minibatch": jnp.asarray(1, dtype=jnp.int32),
                }

            return jax.lax.cond(valid_count > 0, apply_update, skip_update, train_state)

        def update_epoch(epoch_state, _):
            train_state, flat_batch, rng = epoch_state
            rng, permutation_rng = jax.random.split(rng)
            permutation = jax.random.permutation(permutation_rng, config.batch_size)
            shuffled = jax.tree.map(
                lambda value: jnp.take(value, permutation, axis=0), flat_batch
            )
            minibatches = jax.tree.map(
                lambda value: value.reshape(
                    (config.num_minibatches, config.minibatch_size) + value.shape[1:]
                ),
                shuffled,
            )
            train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
            return (train_state, flat_batch, rng), losses

        flat_transition = jax.tree.map(
            lambda value: value.reshape((config.batch_size,) + value.shape[2:]), trajectory
        )
        flat_advantages = advantages.reshape((config.batch_size,))
        flat_targets = targets.reshape((config.batch_size,))
        flat_q_targets = q_targets.reshape((config.batch_size,) + q_targets.shape[2:])
        flat_batch = _PpoBatch(flat_transition, flat_advantages, flat_targets)
        (train_state, _, rng), losses = jax.lax.scan(
            update_epoch,
            (gc_runner.train_state, flat_batch, gc_runner.rng),
            None,
            length=config.update_epochs,
        )
        gc_runner = gc_runner.replace(
            train_state=train_state,
            rng=rng,
            seen_goals=seen_goals,
            global_update=gc_runner.global_update + 1,
            rollout_cursor=jnp.asarray(0, dtype=jnp.int32),
        )

        def teacher_minibatch(leo, batch):
            if not learn_teacher:
                del batch
                zero = jnp.asarray(0.0, dtype=jnp.float32)
                return leo, (zero, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
            sample, target = batch

            def loss_fn(params):
                predicted, next_stats = teacher_network.apply(
                    {"params": params, "batch_stats": leo.batch_stats},
                    sample.map_channels,
                    sample.numeric_features,
                    train=True,
                    mutable=["batch_stats"],
                )
                action_selector = jax.nn.one_hot(
                    sample.action, predicted.shape[-1], dtype=predicted.dtype
                )
                chosen = jnp.sum(predicted * action_selector[:, None, :], axis=-1)
                error = jnp.square(chosen - target)
                mask = sample.valid.astype(jnp.float32)[:, None]
                loss = 0.5 * jnp.sum(error * mask) / jnp.maximum(jnp.sum(mask), 1.0)
                return loss, next_stats["batch_stats"]

            valid_count = jnp.sum(sample.valid.astype(jnp.int32))

            def apply_update(state):
                (loss, next_stats), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
                    state.params
                )
                state = state.apply_gradients(grads=gradients).replace(batch_stats=next_stats)
                return state, (loss, jnp.asarray(1, dtype=jnp.int32), valid_count)

            def skip_update(state):
                zero = jnp.asarray(0.0, dtype=jnp.float32)
                return state, (zero, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))

            return jax.lax.cond(valid_count > 0, apply_update, skip_update, leo)

        def teacher_epoch(carry, _):
            leo, rng = carry
            rng, permutation_rng = jax.random.split(rng)
            permutation = jax.random.permutation(permutation_rng, config.batch_size)
            samples = jax.tree.map(
                lambda value: jnp.take(value, permutation, axis=0), flat_transition
            )
            targets_q = jnp.take(flat_q_targets, permutation, axis=0)
            samples = jax.tree.map(
                lambda value: value.reshape(
                    (config.batch_size // minibatch_size, minibatch_size) + value.shape[1:]
                ),
                samples,
            )
            targets_q = targets_q.reshape(
                (config.batch_size // minibatch_size, minibatch_size) + targets_q.shape[1:]
            )
            leo, teacher_info = jax.lax.scan(teacher_minibatch, leo, (samples, targets_q))
            return (leo, rng), teacher_info

        (leo_state, rng), teacher_info = jax.lax.scan(
            teacher_epoch, (leo_state, gc_runner.rng), None, length=leo_epochs
        )
        td_loss, teacher_applied, teacher_valid_samples = teacher_info
        gc_runner = gc_runner.replace(rng=rng)
        occupied = 1.0 - losses["empty_minibatch"].astype(jnp.float32)

        def occupied_mean(value):
            return jnp.sum(value * occupied) / jnp.maximum(jnp.sum(occupied), 1.0)

        metrics = {
            key: occupied_mean(value)
            for key, value in losses.items()
            if key != "empty_minibatch"
        }
        ppo_scheduled = jnp.asarray(
            config.num_minibatches * config.update_epochs, dtype=jnp.int32
        )
        teacher_scheduled = jnp.asarray(
            (config.batch_size // minibatch_size) * leo_epochs, dtype=jnp.int32
        )
        ppo_valid = jnp.sum(trajectory.valid)
        metrics.update(
            {
                "empty_minibatches": jnp.sum(losses["empty_minibatch"]),
                "valid_transitions": ppo_valid,
                "ppo_valid_transitions": ppo_valid,
                "teacher_valid_samples": jnp.sum(teacher_valid_samples),
                "goal_successes": jnp.sum(trajectory.goal_done),
                "bc_policy_coef": bc_coef.astype(jnp.float32),
                "bc_value_coef": jnp.asarray(BC_VALUE_COEF, dtype=jnp.float32),
                "teacher_td_loss": jnp.mean(td_loss),
                "teacher_grad_steps": leo_state.step.astype(jnp.int32),
                "ppo_scheduled_minibatches": ppo_scheduled,
                "ppo_applied_minibatches": ppo_scheduled - jnp.sum(losses["empty_minibatch"]),
                "teacher_scheduled_minibatches": teacher_scheduled,
                "teacher_applied_minibatches": jnp.sum(teacher_applied),
                "ppo_applied_grad_steps": gc_runner.train_state.step.astype(jnp.int32),
                "ppo_scheduled_grad_steps": (
                    gc_runner.global_update * ppo_scheduled
                ).astype(jnp.int32),
                "teacher_applied_grad_steps": leo_state.step.astype(jnp.int32),
                "teacher_scheduled_grad_steps": (
                    gc_runner.global_update * teacher_scheduled
                ).astype(jnp.int32),
                "learn_teacher": jnp.asarray(int(learn_teacher), dtype=jnp.int32),
                "imitate_teacher": jnp.asarray(int(imitate_teacher), dtype=jnp.int32),
            }
        )
        return gc_runner, leo_state, metrics

    return update


def save_dual_checkpoint(directory, gc_runner, leo_state, config_payload):
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    blob = {"gc": gc_runner, "leo": leo_state}
    (destination / "state.msgpack").write_bytes(serialization.to_bytes(blob))
    (destination / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "hackrl_dual_leo_checkpoint_v1",
                "global_update": int(gc_runner.global_update),
                "teacher_applied_grad_steps": int(leo_state.step),
                "teacher_parameters": teacher_parameter_count(leo_state),
                "input_batch_renorm": True,
                "bc_horizon_updates": BC_HORIZON_UPDATES,
                "bc_policy_coef_at_update": float(bc_policy_coefficient(gc_runner.global_update)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def load_dual_checkpoint(directory, gc_template, leo_template):
    source = Path(directory)
    restored = serialization.from_bytes(
        {"gc": gc_template, "leo": leo_template}, (source / "state.msgpack").read_bytes()
    )
    return restored["gc"], restored["leo"]
