"""Independent goal-conditioned Stable Discrete SAC implementation.

This module is specified from Soft Actor-Critic for Discrete Action Settings
(arXiv:1910.07207v2) and Revisiting Discrete Soft Actor-Critic
(arXiv:2209.10081v4).  No implementation text from the unlicensed
``coldsummerday/SD-SAC`` repository was consulted or used.

The functions are intentionally explicit so the categorical expectations,
entropy penalty, double-average target, Q-clip, and terminal mask can be
checked against small hand-computed examples before any environment run.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn, struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from hackrl.batch_renorm import BatchRenorm
from hackrl.online_value import GoalConditionedQ


class ActorTrainState(TrainState):
    batch_stats: Any = None


class CriticTrainState(TrainState):
    batch_stats: Any = None
    target_params: Any = None
    target_batch_stats: Any = None


class TemperatureTrainState(TrainState):
    pass


class SDSACState(struct.PyTreeNode):
    actor: ActorTrainState
    critic_1: CriticTrainState
    critic_2: CriticTrainState
    temperature: TemperatureTrainState
    environment_steps: jax.Array
    update_steps: jax.Array
    gradient_steps: jax.Array


class SDSACBatch(struct.PyTreeNode):
    map_channels: jax.Array
    numeric_features: jax.Array
    goal_one_hot: jax.Array
    action: jax.Array
    reward: jax.Array
    next_map_channels: jax.Array
    next_numeric_features: jax.Array
    next_goal_one_hot: jax.Array
    done: jax.Array
    valid: jax.Array
    behavior_entropy: jax.Array


class SDSACReplay(struct.PyTreeNode):
    """Fixed-capacity replay with explicit behavior-policy entropy."""

    map_channels: jax.Array
    numeric_features: jax.Array
    goal_one_hot: jax.Array
    action: jax.Array
    reward: jax.Array
    next_map_channels: jax.Array
    next_numeric_features: jax.Array
    next_goal_one_hot: jax.Array
    done: jax.Array
    valid: jax.Array
    behavior_entropy: jax.Array
    cursor: jax.Array
    size: jax.Array
    total_inserted: jax.Array


class CategoricalActor(nn.Module):
    num_actions: int
    hidden_size: int = 1024
    dense_layers: int = 4
    conv_features: int = 16

    @nn.compact
    def __call__(
        self,
        map_channels,
        numeric_features,
        goal_one_hot,
        *,
        train: bool,
    ):
        spatial = BatchRenorm(
            use_running_average=not train, name="map_input_renorm"
        )(map_channels)
        numeric = BatchRenorm(
            use_running_average=not train, name="numeric_input_renorm"
        )(numeric_features)
        spatial = nn.Conv(
            features=self.conv_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=constant(0.0),
            name="map_conv",
        )(spatial)
        spatial = nn.LayerNorm(name="map_norm")(spatial)
        spatial = nn.relu(spatial).reshape((spatial.shape[0], -1))
        hidden = jnp.concatenate((spatial, numeric, goal_one_hot), axis=-1)
        for index in range(self.dense_layers):
            hidden = nn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
                name=f"actor_hidden_{index}",
            )(hidden)
            hidden = nn.LayerNorm(name=f"actor_norm_{index}")(hidden)
            hidden = nn.relu(hidden)
        return nn.Dense(
            self.num_actions,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_output",
        )(hidden)


def categorical_terms(logits):
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    probabilities = jnp.exp(log_probabilities)
    entropy = -jnp.sum(probabilities * log_probabilities, axis=-1)
    return probabilities, log_probabilities, entropy


def double_average_soft_value(logits, q_1, q_2, alpha):
    probabilities, log_probabilities, _ = categorical_terms(logits)
    average_q = 0.5 * (q_1 + q_2)
    return jnp.sum(
        probabilities * (average_q - alpha * log_probabilities), axis=-1
    )


def sd_sac_critic_target(reward, done, next_logits, target_q_1, target_q_2, alpha, gamma):
    next_value = double_average_soft_value(
        next_logits, target_q_1, target_q_2, alpha
    )
    return reward + float(gamma) * (
        1.0 - done.astype(jnp.float32)
    ) * next_value


def _weighted_mean(values, valid):
    valid = valid.astype(jnp.float32)
    return jnp.sum(values * valid) / jnp.maximum(jnp.sum(valid), 1.0)


def q_clip_loss(current_q, target_reference_q, td_target, valid, clip_range):
    """Published maximum-of-unclipped-and-clipped squared Q error."""

    clip_range = float(clip_range)
    if clip_range <= 0.0:
        raise ValueError("clip_range must be positive")
    clipped_q = target_reference_q + jnp.clip(
        current_q - target_reference_q, -clip_range, clip_range
    )
    unclipped_error = jnp.square(current_q - td_target)
    clipped_error = jnp.square(clipped_q - td_target)
    return 0.5 * _weighted_mean(
        jnp.maximum(unclipped_error, clipped_error), valid
    )


def entropy_penalty(behavior_entropy, current_entropy, valid, beta):
    return 0.5 * float(beta) * _weighted_mean(
        jnp.square(behavior_entropy - current_entropy), valid
    )


def sd_sac_actor_loss(
    logits,
    q_1,
    q_2,
    behavior_entropy,
    valid,
    *,
    alpha,
    beta,
):
    probabilities, log_probabilities, entropy = categorical_terms(logits)
    average_q = 0.5 * (q_1 + q_2)
    base = jnp.sum(
        probabilities * (alpha * log_probabilities - average_q), axis=-1
    )
    base_loss = _weighted_mean(base, valid)
    penalty = entropy_penalty(behavior_entropy, entropy, valid, beta)
    return base_loss + penalty, {
        "base_loss": base_loss,
        "entropy_penalty": penalty,
        "entropy": _weighted_mean(entropy, valid),
    }


def temperature_loss(log_alpha, entropy, valid, target_entropy):
    """Minimization increases alpha when entropy falls below its target."""

    alpha = jnp.exp(log_alpha)
    difference = jax.lax.stop_gradient(entropy - float(target_entropy))
    return alpha * _weighted_mean(difference, valid)


def _optimizer(learning_rate, max_grad_norm):
    return optax.chain(
        optax.clip_by_global_norm(float(max_grad_norm)),
        optax.adam(float(learning_rate), eps=1e-5),
    )


def init_sd_sac(
    key,
    map_example,
    numeric_example,
    goal_example,
    *,
    num_actions: int,
    hidden_size: int,
    actor_learning_rate: float,
    critic_learning_rate: float,
    temperature_learning_rate: float,
    initial_alpha: float,
    max_grad_norm: float,
):
    if initial_alpha <= 0.0:
        raise ValueError("initial_alpha must be positive")
    actor_key, critic_1_key, critic_2_key = jax.random.split(key, 3)
    actor_network = CategoricalActor(
        num_actions=int(num_actions), hidden_size=int(hidden_size)
    )
    critic_1_network = GoalConditionedQ(
        num_actions=int(num_actions),
        hidden_size=int(hidden_size),
        sigmoid_outputs=False,
    )
    critic_2_network = GoalConditionedQ(
        num_actions=int(num_actions),
        hidden_size=int(hidden_size),
        sigmoid_outputs=False,
    )
    actor_variables = actor_network.init(
        actor_key,
        map_example[:1],
        numeric_example[:1],
        goal_example[:1],
        train=False,
    )
    critic_1_variables = critic_1_network.init(
        critic_1_key,
        map_example[:1],
        numeric_example[:1],
        goal_example[:1],
        train=False,
    )
    critic_2_variables = critic_2_network.init(
        critic_2_key,
        map_example[:1],
        numeric_example[:1],
        goal_example[:1],
        train=False,
    )
    actor = ActorTrainState.create(
        apply_fn=actor_network.apply,
        params=actor_variables["params"],
        tx=_optimizer(actor_learning_rate, max_grad_norm),
        batch_stats=actor_variables["batch_stats"],
    )

    def critic_state(network, variables):
        return CriticTrainState.create(
            apply_fn=network.apply,
            params=variables["params"],
            tx=_optimizer(critic_learning_rate, max_grad_norm),
            batch_stats=variables["batch_stats"],
            target_params=variables["params"],
            target_batch_stats=variables["batch_stats"],
        )

    critic_1 = critic_state(critic_1_network, critic_1_variables)
    critic_2 = critic_state(critic_2_network, critic_2_variables)
    temperature = TemperatureTrainState.create(
        apply_fn=lambda variables: variables,
        params={"log_alpha": jnp.log(jnp.asarray(initial_alpha, dtype=jnp.float32))},
        tx=optax.adam(float(temperature_learning_rate), eps=1e-5),
    )
    return (
        actor_network,
        critic_1_network,
        critic_2_network,
        SDSACState(
            actor=actor,
            critic_1=critic_1,
            critic_2=critic_2,
            temperature=temperature,
            environment_steps=jnp.asarray(0, dtype=jnp.int32),
            update_steps=jnp.asarray(0, dtype=jnp.int32),
            gradient_steps=jnp.asarray(0, dtype=jnp.int32),
        ),
    )


def init_sd_sac_replay(capacity, map_shape, numeric_shape, goal_shape):
    capacity = int(capacity)
    if capacity <= 0:
        raise ValueError("replay capacity must be positive")
    return SDSACReplay(
        map_channels=jnp.zeros((capacity,) + tuple(map_shape), dtype=jnp.bfloat16),
        numeric_features=jnp.zeros(
            (capacity,) + tuple(numeric_shape), dtype=jnp.float32
        ),
        goal_one_hot=jnp.zeros(
            (capacity,) + tuple(goal_shape), dtype=jnp.float32
        ),
        action=jnp.zeros((capacity,), dtype=jnp.int32),
        reward=jnp.zeros((capacity,), dtype=jnp.float32),
        next_map_channels=jnp.zeros(
            (capacity,) + tuple(map_shape), dtype=jnp.bfloat16
        ),
        next_numeric_features=jnp.zeros(
            (capacity,) + tuple(numeric_shape), dtype=jnp.float32
        ),
        next_goal_one_hot=jnp.zeros(
            (capacity,) + tuple(goal_shape), dtype=jnp.float32
        ),
        done=jnp.zeros((capacity,), dtype=jnp.bool_),
        valid=jnp.zeros((capacity,), dtype=jnp.bool_),
        behavior_entropy=jnp.zeros((capacity,), dtype=jnp.float32),
        cursor=jnp.asarray(0, dtype=jnp.int32),
        size=jnp.asarray(0, dtype=jnp.int32),
        total_inserted=jnp.asarray(0, dtype=jnp.int32),
    )


def append_sd_sac_replay(replay: SDSACReplay, batch: SDSACBatch):
    count = int(batch.action.shape[0])
    capacity = int(replay.action.shape[0])
    if count > capacity:
        raise ValueError("one replay append cannot exceed capacity")
    positions = (replay.cursor + jnp.arange(count, dtype=jnp.int32)) % capacity
    return replay.replace(
        map_channels=replay.map_channels.at[positions].set(
            batch.map_channels.astype(jnp.bfloat16)
        ),
        numeric_features=replay.numeric_features.at[positions].set(
            batch.numeric_features
        ),
        goal_one_hot=replay.goal_one_hot.at[positions].set(batch.goal_one_hot),
        action=replay.action.at[positions].set(batch.action),
        reward=replay.reward.at[positions].set(batch.reward),
        next_map_channels=replay.next_map_channels.at[positions].set(
            batch.next_map_channels.astype(jnp.bfloat16)
        ),
        next_numeric_features=replay.next_numeric_features.at[positions].set(
            batch.next_numeric_features
        ),
        next_goal_one_hot=replay.next_goal_one_hot.at[positions].set(
            batch.next_goal_one_hot
        ),
        done=replay.done.at[positions].set(batch.done),
        valid=replay.valid.at[positions].set(batch.valid),
        behavior_entropy=replay.behavior_entropy.at[positions].set(
            batch.behavior_entropy
        ),
        cursor=(replay.cursor + count) % capacity,
        size=jnp.minimum(capacity, replay.size + count),
        total_inserted=replay.total_inserted + count,
    )


def sample_sd_sac_replay(replay: SDSACReplay, indices):
    return SDSACBatch(
        map_channels=replay.map_channels[indices].astype(jnp.float32),
        numeric_features=replay.numeric_features[indices],
        goal_one_hot=replay.goal_one_hot[indices],
        action=replay.action[indices],
        reward=replay.reward[indices],
        next_map_channels=replay.next_map_channels[indices].astype(jnp.float32),
        next_numeric_features=replay.next_numeric_features[indices],
        next_goal_one_hot=replay.next_goal_one_hot[indices],
        done=replay.done[indices],
        valid=replay.valid[indices],
        behavior_entropy=replay.behavior_entropy[indices],
    )


def valid_sd_sac_replay_count(replay: SDSACReplay):
    positions = jnp.arange(replay.action.shape[0], dtype=jnp.int32)
    eligible = jnp.logical_and(positions < replay.size, replay.valid)
    return jnp.sum(eligible.astype(jnp.int32))


def sample_valid_sd_sac_replay(key, replay: SDSACReplay, batch_size: int):
    """Uniformly sample valid stored transitions with replacement.

    Callers must establish ``valid_sd_sac_replay_count(replay) > 0`` before a
    compiled call.  Empty replay is an ineligible update, not a zero-gradient
    update.
    """

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("replay batch size must be positive")
    positions = jnp.arange(replay.action.shape[0], dtype=jnp.int32)
    eligible = jnp.logical_and(positions < replay.size, replay.valid)
    logits = jnp.where(eligible, 0.0, -jnp.inf)
    indices = jax.random.categorical(
        key, logits, shape=(batch_size,)
    ).astype(jnp.int32)
    return sample_sd_sac_replay(replay, indices), indices


def record_sd_sac_environment_steps(state: SDSACState, count: int):
    """Account for newly collected transitions, never replay samples."""

    count = int(count)
    if count < 0:
        raise ValueError("environment transition count cannot be negative")
    return state.replace(environment_steps=state.environment_steps + count)


def _actor_apply(network, state, batch, *, next_state=False, train=False):
    prefix = "next_" if next_state else ""
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    arguments = (
        getattr(batch, f"{prefix}map_channels"),
        getattr(batch, f"{prefix}numeric_features"),
        getattr(batch, f"{prefix}goal_one_hot"),
    )
    if train:
        output, updates = network.apply(
            variables, *arguments, train=True, mutable=["batch_stats"]
        )
        return output, updates["batch_stats"]
    return network.apply(variables, *arguments, train=False)


def _critic_apply(
    network,
    state,
    batch,
    *,
    next_state=False,
    target=False,
    train=False,
    params=None,
):
    prefix = "next_" if next_state else ""
    selected_params = state.target_params if target else state.params
    selected_stats = state.target_batch_stats if target else state.batch_stats
    if params is not None:
        selected_params = params
    variables = {"params": selected_params, "batch_stats": selected_stats}
    arguments = (
        getattr(batch, f"{prefix}map_channels"),
        getattr(batch, f"{prefix}numeric_features"),
        getattr(batch, f"{prefix}goal_one_hot"),
    )
    if train:
        output, updates = network.apply(
            variables, *arguments, train=True, mutable=["batch_stats"]
        )
        return output, updates["batch_stats"]
    return network.apply(variables, *arguments, train=False)


def _polyak_target(state: CriticTrainState, tau: float):
    tau = float(tau)
    if tau <= 0.0 or tau > 1.0:
        raise ValueError("tau must lie in (0, 1]")
    target_params = jax.tree.map(
        lambda online, target: tau * online + (1.0 - tau) * target,
        state.params,
        state.target_params,
    )
    return state.replace(
        target_params=target_params,
        target_batch_stats=state.batch_stats,
    )


def update_sd_sac(
    actor_network,
    critic_1_network,
    critic_2_network,
    state: SDSACState,
    batch: SDSACBatch,
    *,
    gamma: float,
    beta: float,
    clip_range: float,
    tau: float,
    target_entropy: float,
):
    """One exact-expectation SD-SAC gradient update on a replay minibatch."""

    valid = batch.valid.astype(jnp.float32)
    alpha = jnp.exp(state.temperature.params["log_alpha"])
    next_logits = _actor_apply(
        actor_network, state.actor, batch, next_state=True, train=False
    )
    next_q_1 = _critic_apply(
        critic_1_network, state.critic_1, batch, next_state=True, target=True
    )
    next_q_2 = _critic_apply(
        critic_2_network, state.critic_2, batch, next_state=True, target=True
    )
    td_target = jax.lax.stop_gradient(
        sd_sac_critic_target(
            batch.reward,
            batch.done,
            next_logits,
            next_q_1,
            next_q_2,
            alpha,
            gamma,
        )
    )

    def update_critic(network, critic):
        reference_all = _critic_apply(
            network, critic, batch, target=True, train=False
        )
        reference = jnp.take_along_axis(
            reference_all, batch.action[..., None], axis=-1
        )[..., 0]

        def loss_fn(params):
            values, batch_stats = _critic_apply(
                network, critic, batch, train=True, params=params
            )
            chosen = jnp.take_along_axis(
                values, batch.action[..., None], axis=-1
            )[..., 0]
            loss = q_clip_loss(
                chosen, reference, td_target, valid, clip_range
            )
            return loss, (batch_stats, chosen)

        (loss, (batch_stats, chosen)), gradients = jax.value_and_grad(
            loss_fn, has_aux=True
        )(critic.params)
        critic = critic.apply_gradients(grads=gradients).replace(
            batch_stats=batch_stats
        )
        return critic, loss, chosen

    critic_1, critic_1_loss, chosen_1 = update_critic(
        critic_1_network, state.critic_1
    )
    critic_2, critic_2_loss, chosen_2 = update_critic(
        critic_2_network, state.critic_2
    )

    q_1 = jax.lax.stop_gradient(
        _critic_apply(critic_1_network, critic_1, batch, train=False)
    )
    q_2 = jax.lax.stop_gradient(
        _critic_apply(critic_2_network, critic_2, batch, train=False)
    )

    def actor_loss_fn(params):
        logits, batch_stats = actor_network.apply(
            {"params": params, "batch_stats": state.actor.batch_stats},
            batch.map_channels,
            batch.numeric_features,
            batch.goal_one_hot,
            train=True,
            mutable=["batch_stats"],
        )
        loss, metrics = sd_sac_actor_loss(
            logits,
            q_1,
            q_2,
            batch.behavior_entropy,
            valid,
            alpha=alpha,
            beta=beta,
        )
        return loss, (batch_stats["batch_stats"], metrics)

    (actor_loss, (actor_batch_stats, actor_metrics)), actor_gradients = (
        jax.value_and_grad(actor_loss_fn, has_aux=True)(state.actor.params)
    )
    actor = state.actor.apply_gradients(grads=actor_gradients).replace(
        batch_stats=actor_batch_stats
    )
    updated_logits = _actor_apply(actor_network, actor, batch, train=False)
    _, _, updated_entropy = categorical_terms(updated_logits)

    def alpha_loss_fn(params):
        return temperature_loss(
            params["log_alpha"], updated_entropy, valid, target_entropy
        )

    alpha_loss, alpha_gradients = jax.value_and_grad(alpha_loss_fn)(
        state.temperature.params
    )
    temperature = state.temperature.apply_gradients(grads=alpha_gradients)
    critic_1 = _polyak_target(critic_1, tau)
    critic_2 = _polyak_target(critic_2, tau)
    batch_size = int(batch.action.shape[0])
    state = SDSACState(
        actor=actor,
        critic_1=critic_1,
        critic_2=critic_2,
        temperature=temperature,
        environment_steps=state.environment_steps,
        update_steps=state.update_steps + 1,
        gradient_steps=state.gradient_steps + 4,
    )
    metrics = {
        "critic_1_loss": critic_1_loss,
        "critic_2_loss": critic_2_loss,
        "actor_loss": actor_loss,
        "actor_base_loss": actor_metrics["base_loss"],
        "entropy_penalty": actor_metrics["entropy_penalty"],
        "entropy": actor_metrics["entropy"],
        "alpha_loss": alpha_loss,
        "alpha": jnp.exp(temperature.params["log_alpha"]),
        "target_mean": _weighted_mean(td_target, valid),
        "q_1_mean": _weighted_mean(chosen_1, valid),
        "q_2_mean": _weighted_mean(chosen_2, valid),
        "valid_transitions": jnp.sum(batch.valid.astype(jnp.int32)),
        "sampled_transitions": jnp.asarray(batch_size, dtype=jnp.int32),
        "applied_gradient_steps": jnp.asarray(4, dtype=jnp.int32),
    }
    return state, metrics


def make_sd_sac_replay_updates(
    actor_network,
    critic_1_network,
    critic_2_network,
    *,
    batch_size: int,
    update_iterations: int,
    gamma: float,
    beta: float,
    clip_range: float,
    tau: float,
    target_entropy: float,
):
    """Build fixed-count replay updates without changing environment steps."""

    batch_size = int(batch_size)
    update_iterations = int(update_iterations)
    if batch_size <= 0 or update_iterations <= 0:
        raise ValueError("replay batch size and update iterations must be positive")

    def one_step(carry, _):
        state, replay, rng = carry
        rng, sample_key = jax.random.split(rng)
        batch, _ = sample_valid_sd_sac_replay(sample_key, replay, batch_size)
        state, metrics = update_sd_sac(
            actor_network,
            critic_1_network,
            critic_2_network,
            state,
            batch,
            gamma=gamma,
            beta=beta,
            clip_range=clip_range,
            tau=tau,
            target_entropy=target_entropy,
        )
        return (state, replay, rng), metrics

    def update(state, replay, rng):
        (state, _, rng), rows = jax.lax.scan(
            one_step, (state, replay, rng), None, length=update_iterations
        )
        metrics = {
            "last": jax.tree.map(lambda value: value[-1], rows),
            "valid_replay_transitions": valid_sd_sac_replay_count(replay),
            "sampled_learning_transitions": jnp.asarray(
                batch_size * update_iterations, dtype=jnp.int32
            ),
            "scheduled_update_iterations": jnp.asarray(
                update_iterations, dtype=jnp.int32
            ),
            "applied_optimizer_steps": jnp.asarray(
                4 * update_iterations, dtype=jnp.int32
            ),
        }
        return state, rng, metrics

    return update
