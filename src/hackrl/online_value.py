"""Online value-learning cores for the controlled HackRL fixtures.

The PQN and LEO contracts are adapted from the MIT-licensed purejaxgcrl
implementation at eafccd995ac2736df5fbc05ac4bb6b5d58cfcce7 and the
Apache-2.0 PureJaxQL implementation at
47af6d7b35c89ddfe633aaf7341bdb8964cb7cce.  No source file is vendored.

The important boundary is structural: PQN and LEO are online learners.  They
have no replay buffer and no target network.  Targets use the current online
network with stop-gradient.  The first comparison uses the official one-step
default and intentionally does not expose Q(lambda) as a candidate dimension.
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
from hackrl.dual_leo import DualLeoQ


class OnlineQTrainState(TrainState):
    """Complete train state for an online Q learner.

    There are deliberately no ``target_params`` or replay fields.  Lifetime
    environment steps and phase-local steps are separate so an adaptation
    branch can restart only its exploration schedule without falsifying total
    experience accounting.
    """

    batch_stats: Any = None
    environment_steps: Any = 0
    phase_steps: Any = 0
    update_steps: Any = 0
    gradient_steps: Any = 0


class GoalConditionedQ(nn.Module):
    """Commanded-goal Q network with the official PQN-style online head."""

    num_actions: int
    hidden_size: int = 1024
    dense_layers: int = 4
    conv_features: int = 16
    sigmoid_outputs: bool = True

    @nn.compact
    def __call__(
        self,
        map_channels: jax.Array,
        numeric_features: jax.Array,
        goal_one_hot: jax.Array,
        *,
        train: bool,
    ) -> jax.Array:
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
                name=f"q_hidden_{index}",
            )(hidden)
            hidden = nn.LayerNorm(name=f"q_norm_{index}")(hidden)
            hidden = nn.relu(hidden)
        q_values = nn.Dense(
            self.num_actions,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="q_output",
        )(hidden)
        if self.sigmoid_outputs:
            q_values = jax.nn.sigmoid(q_values)
        return q_values


class GoalQTransition(struct.PyTreeNode):
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


class AllGoalTransition(struct.PyTreeNode):
    map_channels: jax.Array
    numeric_features: jax.Array
    action: jax.Array
    terminal_goals: jax.Array
    next_map_channels: jax.Array
    next_numeric_features: jax.Array
    world_done: jax.Array
    valid: jax.Array


class DualOnlineQState(struct.PyTreeNode):
    pqn: OnlineQTrainState
    leo: OnlineQTrainState


def _optimizer(learning_rate: float, max_grad_norm: float):
    return optax.chain(
        optax.clip_by_global_norm(float(max_grad_norm)),
        optax.adam(float(learning_rate), eps=1e-5),
    )


def init_goal_q(
    key,
    map_example,
    numeric_example,
    goal_example,
    *,
    num_actions: int,
    hidden_size: int,
    learning_rate: float,
    max_grad_norm: float,
):
    network = GoalConditionedQ(
        num_actions=int(num_actions), hidden_size=int(hidden_size)
    )
    variables = network.init(
        key,
        map_example[:1],
        numeric_example[:1],
        goal_example[:1],
        train=False,
    )
    state = OnlineQTrainState.create(
        apply_fn=network.apply,
        params=variables["params"],
        tx=_optimizer(learning_rate, max_grad_norm),
        batch_stats=variables["batch_stats"],
        environment_steps=jnp.asarray(0, dtype=jnp.int32),
        phase_steps=jnp.asarray(0, dtype=jnp.int32),
        update_steps=jnp.asarray(0, dtype=jnp.int32),
        gradient_steps=jnp.asarray(0, dtype=jnp.int32),
    )
    return network, state


def init_all_goal_q(
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
        key, map_example[:1], numeric_example[:1], train=False
    )
    state = OnlineQTrainState.create(
        apply_fn=network.apply,
        params=variables["params"],
        tx=_optimizer(learning_rate, max_grad_norm),
        batch_stats=variables["batch_stats"],
        environment_steps=jnp.asarray(0, dtype=jnp.int32),
        phase_steps=jnp.asarray(0, dtype=jnp.int32),
        update_steps=jnp.asarray(0, dtype=jnp.int32),
        gradient_steps=jnp.asarray(0, dtype=jnp.int32),
    )
    return network, state


def goal_q_apply(network, state, map_channels, numeric_features, goal_one_hot):
    return network.apply(
        {"params": state.params, "batch_stats": state.batch_stats},
        map_channels,
        numeric_features,
        goal_one_hot,
        train=False,
    )


def all_goal_q_apply(network, state, map_channels, numeric_features):
    return network.apply(
        {"params": state.params, "batch_stats": state.batch_stats},
        map_channels,
        numeric_features,
        train=False,
    )


def select_goal_q(all_goal_q, goal_index):
    """Select [batch, action] values from [batch, goal, action]."""

    return jnp.take_along_axis(
        all_goal_q, goal_index[..., None, None], axis=-2
    )[..., 0, :]


def online_one_step_targets(reward, done, next_q, gamma):
    """One-step online target with a stopped current-network bootstrap."""

    bootstrap = jnp.max(jax.lax.stop_gradient(next_q), axis=-1)
    return reward + float(gamma) * bootstrap * (
        1.0 - done.astype(jnp.float32)
    )


def leo_one_step_targets(terminal_goals, world_done, next_all_goal_q, gamma):
    """All-goal one-step target with per-goal pseudo-termination."""

    bootstrap = jnp.max(jax.lax.stop_gradient(next_all_goal_q), axis=-1)
    done = jnp.logical_or(terminal_goals, world_done[..., None])
    reward = terminal_goals.astype(jnp.float32)
    return reward + float(gamma) * bootstrap * (
        1.0 - done.astype(jnp.float32)
    )


def weighted_td_loss(chosen_q, target, valid):
    valid = valid.astype(jnp.float32)
    numerator = 0.5 * jnp.sum(jnp.square(chosen_q - target) * valid)
    return numerator / jnp.maximum(jnp.sum(valid), 1.0)


def all_goal_td_loss(chosen_q, target, valid):
    """Sum goal-head errors and divide by valid physical transitions."""

    valid = valid.astype(jnp.float32)[..., None]
    numerator = 0.5 * jnp.sum(jnp.square(chosen_q - target) * valid)
    return numerator / jnp.maximum(jnp.sum(valid), 1.0)


def epsilon_at_phase_step(
    phase_steps,
    *,
    start: float,
    finish: float,
    decay_transitions: int,
):
    if int(decay_transitions) <= 0:
        raise ValueError("decay_transitions must be positive")
    fraction = jnp.minimum(
        jnp.asarray(phase_steps, dtype=jnp.float32) / float(decay_transitions),
        1.0,
    )
    return float(start) + fraction * (float(finish) - float(start))


def epsilon_greedy_actions(key, q_values, epsilon):
    """Consume both random-action and explore-decision streams every call."""

    action_key, explore_key = jax.random.split(key)
    random_actions = jax.random.randint(
        action_key,
        shape=q_values.shape[:-1],
        minval=0,
        maxval=q_values.shape[-1],
    )
    explore = jax.random.uniform(explore_key, shape=q_values.shape[:-1]) < epsilon
    return jnp.where(explore, random_actions, jnp.argmax(q_values, axis=-1))


def combine_dual_q(pqn_q, leo_q, *, leo_weight: float = 0.3):
    weight = float(leo_weight)
    if weight < 0.0 or weight > 1.0:
        raise ValueError("leo_weight must lie in [0, 1]")
    return (1.0 - weight) * pqn_q + weight * leo_q


def reset_phase_counter(state: OnlineQTrainState):
    """Restart exploration only; preserve lifetime experience and optimizer."""

    return state.replace(phase_steps=jnp.asarray(0, dtype=state.phase_steps.dtype))


def update_goal_q(network, state, transition: GoalQTransition, *, gamma: float):
    """Apply one current-network PQN gradient step to a flat rollout batch."""

    batch_size = int(transition.action.shape[0])

    def loss_fn(params):
        maps = jnp.concatenate(
            (transition.map_channels, transition.next_map_channels), axis=0
        )
        numeric = jnp.concatenate(
            (transition.numeric_features, transition.next_numeric_features), axis=0
        )
        goals = jnp.concatenate(
            (transition.goal_one_hot, transition.next_goal_one_hot), axis=0
        )
        q_all, updates = network.apply(
            {"params": params, "batch_stats": state.batch_stats},
            maps,
            numeric,
            goals,
            train=True,
            mutable=["batch_stats"],
        )
        q_values, next_q = jnp.split(q_all, (batch_size,), axis=0)
        target = online_one_step_targets(
            transition.reward, transition.done, next_q, gamma
        )
        chosen = jnp.take_along_axis(
            q_values, transition.action[..., None], axis=-1
        )[..., 0]
        loss = weighted_td_loss(chosen, target, transition.valid)
        return loss, (updates["batch_stats"], chosen, target)

    (loss, (batch_stats, chosen, target)), gradients = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)
    state = state.apply_gradients(grads=gradients).replace(
        batch_stats=batch_stats,
        environment_steps=state.environment_steps + batch_size,
        phase_steps=state.phase_steps + batch_size,
        update_steps=state.update_steps + 1,
        gradient_steps=state.gradient_steps + 1,
    )
    metrics = {
        "loss": loss,
        "chosen_q_mean": jnp.mean(chosen),
        "target_mean": jnp.mean(target),
        "valid_transitions": jnp.sum(transition.valid.astype(jnp.int32)),
        "physical_transitions": jnp.asarray(batch_size, dtype=jnp.int32),
        "applied_gradient_steps": jnp.asarray(1, dtype=jnp.int32),
    }
    return state, metrics


def update_all_goal_q(
    network,
    state,
    transition: AllGoalTransition,
    *,
    gamma: float,
):
    """Apply one LEO all-goal gradient step to a flat rollout batch."""

    batch_size = int(transition.action.shape[0])

    def loss_fn(params):
        maps = jnp.concatenate(
            (transition.map_channels, transition.next_map_channels), axis=0
        )
        numeric = jnp.concatenate(
            (transition.numeric_features, transition.next_numeric_features), axis=0
        )
        q_all, updates = network.apply(
            {"params": params, "batch_stats": state.batch_stats},
            maps,
            numeric,
            train=True,
            mutable=["batch_stats"],
        )
        q_values, next_q = jnp.split(q_all, (batch_size,), axis=0)
        target = leo_one_step_targets(
            transition.terminal_goals,
            transition.world_done,
            next_q,
            gamma,
        )
        action = transition.action[..., None, None]
        chosen = jnp.take_along_axis(q_values, action, axis=-1)[..., 0]
        loss = all_goal_td_loss(chosen, target, transition.valid)
        return loss, (updates["batch_stats"], chosen, target)

    (loss, (batch_stats, chosen, target)), gradients = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)
    state = state.apply_gradients(grads=gradients).replace(
        batch_stats=batch_stats,
        environment_steps=state.environment_steps + batch_size,
        phase_steps=state.phase_steps + batch_size,
        update_steps=state.update_steps + 1,
        gradient_steps=state.gradient_steps + 1,
    )
    metrics = {
        "loss": loss,
        "chosen_q_mean": jnp.mean(chosen),
        "target_mean": jnp.mean(target),
        "valid_transitions": jnp.sum(transition.valid.astype(jnp.int32)),
        "td_targets": jnp.sum(transition.valid.astype(jnp.int32))
        * int(transition.terminal_goals.shape[-1]),
        "physical_transitions": jnp.asarray(batch_size, dtype=jnp.int32),
        "applied_gradient_steps": jnp.asarray(1, dtype=jnp.int32),
    }
    return state, metrics


def update_dual_q(
    pqn_network,
    leo_network,
    state: DualOnlineQState,
    pqn_transition: GoalQTransition,
    leo_transition: AllGoalTransition,
    *,
    gamma: float,
):
    """Update both TD learners without an imitation or cross-loss term."""

    pqn, pqn_metrics = update_goal_q(
        pqn_network, state.pqn, pqn_transition, gamma=gamma
    )
    leo, leo_metrics = update_all_goal_q(
        leo_network, state.leo, leo_transition, gamma=gamma
    )
    return DualOnlineQState(pqn=pqn, leo=leo), {
        "pqn": pqn_metrics,
        "leo": leo_metrics,
    }


def parameter_count(state: OnlineQTrainState) -> int:
    return int(sum(np.asarray(value).size for value in jax.tree.leaves(state.params)))
