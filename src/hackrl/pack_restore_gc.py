"""Goal-conditioned PPO for the PACK-RESTORE source-growth fixture.

Pretraining and adaptation use the same masked-free contract as the TICK-CLAIM
history comparison. Goal success is a value-learning cut. Only the world
horizon resets the workshop. The record is not grain: a violation is an
increase in the physical grain total, and excess delivery spends that balance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import serialization, struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from hackrl.pack_restore import (
    GOAL_IDS,
    MAP_CHANNEL_NAMES,
    MODEL_FEATURE_NAMES,
    TRAINING_SOURCE_GROWTH_PERIOD,
    PackRestoreAction,
    PackRestoreSplit,
    PackRestoreStart,
    PackRestoreState,
    PackRestoreVariant,
    _VIEW_COLUMNS,
    _VIEW_ROWS,
    conservation_increased,
    encode_pack_restore_observation,
    make_pack_restore_state,
    observe_pack_restore,
    pack_restore_goal_vector,
    pack_restore_step,
    physical_grain_total,
    pack_restore_world_done,
    reset_pack_restore,
    reset_pack_restore_worker,
)


DELIVER_3_GOAL_INDEX = GOAL_IDS.index("delivery/count_ge_3")
MAP_FEATURE_COUNT = _VIEW_ROWS * _VIEW_COLUMNS * len(MAP_CHANNEL_NAMES)
NUM_GOALS = len(GOAL_IDS)
NUM_ACTIONS = len(PackRestoreAction)
DISCOUNT = 0.995


@dataclass(frozen=True)
class PackRestoreGCConfig:
    variant: str = PackRestoreVariant.FIXED.value
    seed: int = 0
    num_envs: int = 512
    num_steps: int = 64
    num_updates: int = 32
    update_epochs: int = 1
    minibatch_size: int = 1024
    hidden_size: int = 512
    learning_rate: float = 2e-4
    gamma: float = DISCOUNT
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.005
    value_coefficient: float = 0.5
    max_grad_norm: float = 1.0
    goal_mode: str = "deliver_3"
    source_growth_period: int = TRAINING_SOURCE_GROWTH_PERIOD
    evaluation_split: str = PackRestoreSplit.VALIDATION.value
    mode_repeats_per_state: int = 1
    sample_repeats_per_state: int = 4
    checkpoint_updates: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(
            self, "checkpoint_updates", tuple(self.checkpoint_updates)
        )

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def num_minibatches(self) -> int:
        return self.batch_size // self.minibatch_size

    def validate(self) -> None:
        PackRestoreVariant(self.variant)
        PackRestoreSplit(self.evaluation_split)
        if self.num_envs <= 0 or self.num_steps <= 0 or self.num_updates <= 0:
            raise ValueError("num_envs, num_steps, and num_updates must be positive")
        if self.update_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("update_epochs and minibatch_size must be positive")
        if self.batch_size % self.minibatch_size:
            raise ValueError("rollout batch must divide evenly into minibatches")
        if self.hidden_size <= 0 or self.learning_rate <= 0:
            raise ValueError("hidden_size and learning_rate must be positive")
        if self.goal_mode not in {"deliver_3", "workshop12"}:
            raise ValueError("goal_mode must be 'deliver_3' or 'workshop12'")
        if self.source_growth_period < 0:
            raise ValueError("source_growth_period must be non-negative")
        if self.mode_repeats_per_state <= 0 or self.sample_repeats_per_state <= 0:
            raise ValueError("evaluation repeats must be positive")
        if len(MODEL_FEATURE_NAMES) <= 0 or MAP_FEATURE_COUNT <= 0:
            raise ValueError("observation schema is empty")


class PackRestoreGCActorCritic(nn.Module):
    hidden_size: int = 512
    action_dim: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, map_channels, numeric_features, goal_one_hot):
        spatial = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=constant(0.0),
            name="map_conv",
        )(map_channels)
        spatial = nn.relu(spatial).reshape((spatial.shape[0], -1))
        shared = jnp.concatenate((spatial, numeric_features, goal_one_hot), axis=-1)
        shared = nn.relu(
            nn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name="shared_dense",
            )(shared)
        )
        actor = shared
        for index in range(2):
            actor = nn.relu(
                nn.Dense(
                    self.hidden_size,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                    name=f"actor_hidden_{index}",
                )(actor)
            )
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_output",
        )(actor)
        critic = shared
        for index in range(4):
            critic = nn.relu(
                nn.Dense(
                    self.hidden_size,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                    name=f"critic_hidden_{index}",
                )(critic)
            )
        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_output",
        )(critic)
        return distrax.Categorical(logits=logits), jax.nn.sigmoid(
            jnp.squeeze(value, axis=-1)
        )


@struct.dataclass
class PackRestoreGCRunnerState:
    train_state: TrainState
    env_state: PackRestoreState
    current_goal: jax.Array
    command_active: jax.Array
    env_keys: jax.Array
    sampler_keys: jax.Array
    rng: jax.Array
    seen_goals: jax.Array
    goal_steps: jax.Array
    world_violation_count: jax.Array
    violation_grain_balance: jax.Array
    global_update: jax.Array
    env_steps: jax.Array
    rollout_cursor: jax.Array


class PackRestoreGCTransition(struct.PyTreeNode):
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
    started_goal_index: jax.Array
    goal_done: jax.Array
    world_done: jax.Array
    command_started: jax.Array
    zero_step_successes: jax.Array
    sampler_valid: jax.Array
    violation: jax.Array
    repeated_violation: jax.Array
    opportunity_exposure: jax.Array
    violation_delivery: jax.Array
    violation_grain_delivered: jax.Array
    delivered_amount: jax.Array
    success_steps: jax.Array
    reset_count: jax.Array
    observed_goals: jax.Array
    absolute_transition: jax.Array


class PackRestoreGCTrainingBatch(struct.PyTreeNode):
    transition: PackRestoreGCTransition
    advantages: jax.Array
    targets: jax.Array


def pack_restore_gc_inputs(observation, goal_index):
    encoded = encode_pack_restore_observation(observation)
    map_channels = encoded[:MAP_FEATURE_COUNT].reshape(
        _VIEW_ROWS, _VIEW_COLUMNS, len(MAP_CHANNEL_NAMES)
    )
    numeric_features = encoded[MAP_FEATURE_COUNT:]
    goal_one_hot = jax.nn.one_hot(goal_index, NUM_GOALS, dtype=jnp.float32)
    return map_channels, numeric_features, goal_one_hot


def _batch_inputs(env_state, goals):
    observations = jax.vmap(observe_pack_restore)(env_state)
    return jax.vmap(pack_restore_gc_inputs)(observations, goals)


def pack_restore_gc_parameter_count(parameters) -> int:
    return int(sum(np.asarray(value).size for value in jax.tree.leaves(parameters)))


def _optimizer(config):
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate, eps=1e-5),
    )


def _tree_where(mask, when_true, when_false):
    def choose(left, right):
        expanded = mask.reshape(mask.shape + (1,) * (left.ndim - mask.ndim))
        return jnp.where(expanded, left, right)

    return jax.tree.map(choose, when_true, when_false)


def sample_false_seen_goal(key, seen_goals, achieved_goals):
    false_seen = jnp.logical_and(seen_goals, jnp.logical_not(achieved_goals))
    valid = jnp.any(false_seen)
    logits = jnp.where(seen_goals, 0.0, -1e9)

    def condition(loop_state):
        _, _, _, accepted, attempts = loop_state
        return jnp.logical_and(jnp.logical_not(accepted), attempts < 128)

    def body(loop_state):
        loop_key, _, zero_steps, _, attempts = loop_state
        loop_key, draw_key = jax.random.split(loop_key)
        candidate = jax.random.categorical(draw_key, logits).astype(jnp.int32)
        already_satisfied = achieved_goals[candidate]
        return (
            loop_key,
            candidate,
            zero_steps + already_satisfied.astype(jnp.int32),
            jnp.logical_not(already_satisfied),
            attempts + 1,
        )

    key, candidate, zero_steps, accepted, _ = jax.lax.while_loop(
        condition,
        body,
        (
            key,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    return candidate, key, zero_steps, jnp.logical_and(valid, accepted)


def _trigger_ready(state):
    """Positive record, original packed grain, empty anchor, and a spare frame."""

    return jnp.logical_and(
        jnp.logical_not(state.anchor_present),
        jnp.logical_and(
            state.record_present,
            jnp.logical_and(
                state.record_grain_preview > 0,
                jnp.logical_and(
                    state.empty_frames > 0,
                    jnp.logical_and(
                        state.packed_present,
                        jnp.logical_and(
                            state.packed_grain > 0,
                            jnp.sum(jnp.abs(state.player_position - state.anchor_position))
                            == 1,
                        ),
                    ),
                ),
            ),
        ),
    )


def initialize_pack_restore_gc(config):
    config.validate()
    init_rng, action_rng, env_rng, sampler_rng = jax.random.split(
        jax.random.PRNGKey(config.seed), 4
    )
    workers = jnp.arange(config.num_envs, dtype=jnp.int32) % 512

    def reset_worker(index):
        return reset_pack_restore_worker(
            index, source_growth_period=config.source_growth_period
        )

    env_state = jax.vmap(reset_worker)(workers)
    observations = jax.vmap(observe_pack_restore)(env_state)
    goal_vectors = jax.vmap(pack_restore_goal_vector)(env_state)
    seen_goals = jnp.any(goal_vectors, axis=0)
    false_seen = jnp.logical_and(seen_goals[None, :], jnp.logical_not(goal_vectors))
    if not bool(jnp.all(jnp.any(false_seen, axis=1))):
        raise ValueError("at least one worker has no false seen goal")
    env_keys = jax.random.split(env_rng, config.num_envs)
    sampler_keys = jax.random.split(sampler_rng, config.num_envs)
    if config.goal_mode == "deliver_3":
        current_goal = jnp.full(
            (config.num_envs,), DELIVER_3_GOAL_INDEX, dtype=jnp.int32
        )
    else:
        sampled = jax.vmap(sample_false_seen_goal, in_axes=(0, None, 0))(
            sampler_keys, seen_goals, goal_vectors
        )
        current_goal, sampler_keys, _, sampler_valid = sampled
        if not bool(jnp.all(sampler_valid)):
            raise ValueError("initial goal sampler could not find a false goal")
    model_inputs = jax.vmap(pack_restore_gc_inputs)(observations, current_goal)
    network = PackRestoreGCActorCritic(hidden_size=config.hidden_size)
    parameters = network.init(
        init_rng,
        model_inputs[0][:1],
        model_inputs[1][:1],
        model_inputs[2][:1],
    )
    if int(model_inputs[1].shape[-1]) != len(MODEL_FEATURE_NAMES):
        raise ValueError("numeric feature width does not match the schema")
    train_state = TrainState.create(
        apply_fn=network.apply, params=parameters, tx=_optimizer(config)
    )
    runner = PackRestoreGCRunnerState(
        train_state=train_state,
        env_state=env_state,
        current_goal=current_goal,
        command_active=jnp.ones((config.num_envs,), dtype=jnp.bool_),
        env_keys=env_keys,
        sampler_keys=sampler_keys,
        rng=action_rng,
        seen_goals=seen_goals,
        goal_steps=jnp.zeros((config.num_envs,), dtype=jnp.int32),
        world_violation_count=jnp.zeros((config.num_envs,), dtype=jnp.int32),
        violation_grain_balance=jnp.zeros((config.num_envs,), dtype=jnp.int32),
        global_update=jnp.asarray(0, dtype=jnp.int32),
        env_steps=jnp.asarray(0, dtype=jnp.int32),
        rollout_cursor=jnp.asarray(0, dtype=jnp.int32),
    )
    return network, runner


def step_pack_restore_gc_workers(runner, actions, config):
    variant = PackRestoreVariant(config.variant)
    before = runner.env_state
    valid_transition = runner.command_active
    actions = jnp.where(valid_transition, actions, jnp.asarray(int(PackRestoreAction.NOOP)))
    opportunity = jax.vmap(_trigger_ready)(before)
    stepped = jax.vmap(lambda state, action: pack_restore_step(state, action, variant))(
        before, actions
    )
    created = jax.vmap(physical_grain_total)(stepped) - jax.vmap(physical_grain_total)(before)
    violation = jnp.logical_and(valid_transition, created > 0)
    terminal_goals = jax.vmap(pack_restore_goal_vector)(stepped)
    achieved_command = jnp.take_along_axis(
        terminal_goals, runner.current_goal[:, None], axis=1
    )[:, 0]
    goal_done = jnp.logical_and(valid_transition, achieved_command)
    world_done = jax.vmap(pack_restore_world_done)(stepped)
    done_for_gae = jnp.logical_or(
        jnp.logical_or(goal_done, world_done), jnp.logical_not(valid_transition)
    )
    split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(runner.env_keys)
    next_env_keys = split_keys[:, 0]
    reset_keys = split_keys[:, 1]

    def reset_env(key):
        return reset_pack_restore(key, source_growth_period=config.source_growth_period)

    reset_states = jax.vmap(reset_env)(reset_keys)
    reset_goals = jax.vmap(pack_restore_goal_vector)(reset_states)
    env_state = _tree_where(world_done, reset_states, stepped)
    observed_goals = jnp.logical_or(
        terminal_goals, jnp.where(world_done[:, None], reset_goals, False)
    )
    switch = jnp.logical_or(goal_done, world_done)
    if config.goal_mode == "workshop12":
        command_goals = jnp.where(world_done[:, None], reset_goals, terminal_goals)
        sampled_goal, sampled_keys, zero_steps, sampler_valid = jax.vmap(
            sample_false_seen_goal, in_axes=(0, None, 0)
        )(runner.sampler_keys, runner.seen_goals, command_goals)
        current_goal = jnp.where(switch, sampled_goal, runner.current_goal)
        sampler_keys = jnp.where(switch[:, None], sampled_keys, runner.sampler_keys)
        zero_step_successes = jnp.where(switch, zero_steps, 0)
        command_started = switch
        command_active = jnp.ones_like(runner.command_active)
        sampler_valid = jnp.where(switch, sampler_valid, True)
    else:
        current_goal = runner.current_goal
        sampler_keys = runner.sampler_keys
        zero_step_successes = jnp.zeros_like(runner.goal_steps)
        command_started = world_done
        command_active = jnp.where(
            world_done,
            jnp.logical_not(reset_goals[:, DELIVER_3_GOAL_INDEX]),
            jnp.where(goal_done, False, runner.command_active),
        )
        sampler_valid = jnp.ones_like(runner.command_active)
    delivered_amount = stepped.delivered_total - before.delivered_total
    violation_grain_created = jnp.where(violation, created, 0)
    balance_before = runner.violation_grain_balance + violation_grain_created
    violation_grain_delivered = jnp.minimum(
        jnp.maximum(delivered_amount, 0), balance_before
    )
    violation_delivery = jnp.logical_and(valid_transition, violation_grain_delivered > 0)
    violation_grain_balance = jnp.where(
        world_done, 0, balance_before - violation_grain_delivered
    )
    violation_count = jnp.where(
        world_done, 0, runner.world_violation_count + violation.astype(jnp.int32)
    )
    repeated_violation = jnp.logical_and(violation, runner.world_violation_count > 0)
    success_steps = jnp.where(goal_done, runner.goal_steps + 1, 0)
    goal_steps = jnp.where(
        switch,
        0,
        jnp.where(runner.command_active, runner.goal_steps + 1, runner.goal_steps),
    )
    next_runner = runner.replace(
        env_state=env_state,
        current_goal=current_goal,
        command_active=command_active,
        env_keys=next_env_keys,
        sampler_keys=sampler_keys,
        goal_steps=goal_steps,
        world_violation_count=violation_count,
        violation_grain_balance=violation_grain_balance,
        env_steps=runner.env_steps + config.num_envs,
        rollout_cursor=runner.rollout_cursor + 1,
    )
    event = PackRestoreGCTransition(
        done=done_for_gae,
        valid=valid_transition,
        action=actions,
        value=jnp.zeros_like(done_for_gae, dtype=jnp.float32),
        reward=goal_done.astype(jnp.float32),
        log_prob=jnp.zeros_like(done_for_gae, dtype=jnp.float32),
        map_channels=jnp.zeros((1,), dtype=jnp.float32),
        numeric_features=jnp.zeros((1,), dtype=jnp.float32),
        goal_one_hot=jnp.zeros((NUM_GOALS,), dtype=jnp.float32),
        goal_index=runner.current_goal,
        started_goal_index=current_goal,
        goal_done=goal_done,
        world_done=world_done,
        command_started=command_started,
        zero_step_successes=zero_step_successes,
        sampler_valid=sampler_valid,
        violation=violation,
        repeated_violation=repeated_violation,
        opportunity_exposure=jnp.logical_and(valid_transition, opportunity),
        violation_delivery=violation_delivery,
        violation_grain_delivered=jnp.where(valid_transition, violation_grain_delivered, 0),
        delivered_amount=jnp.where(valid_transition, delivered_amount, 0),
        success_steps=success_steps,
        reset_count=world_done.astype(jnp.int32),
        observed_goals=observed_goals,
        absolute_transition=jnp.zeros_like(runner.goal_steps),
    )
    return next_runner, event


def calculate_pack_restore_gc_gae(trajectory, last_value, gamma, gae_lambda):
    def backward(carry, transition):
        gae, next_value = carry
        nonterminal = 1.0 - transition.done.astype(jnp.float32)
        delta = transition.reward + gamma * next_value * nonterminal - transition.value
        gae = (delta + gamma * gae_lambda * nonterminal * gae) * transition.valid.astype(
            jnp.float32
        )
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        backward,
        (jnp.zeros_like(last_value), last_value),
        trajectory,
        reverse=True,
    )
    return advantages, advantages + trajectory.value


def _weighted_mean(value, mask):
    mask = mask.astype(jnp.float32)
    return jnp.sum(value * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def make_pack_restore_gc_update(network, config):
    def rollout_step(runner, _):
        command_goal = runner.current_goal
        model_inputs = _batch_inputs(runner.env_state, command_goal)
        policy, value = network.apply(runner.train_state.params, *model_inputs)
        rng, action_rng = jax.random.split(runner.rng)
        actions = policy.sample(seed=action_rng)
        log_prob = policy.log_prob(actions)
        before_steps = runner.env_steps
        runner = runner.replace(rng=rng)
        runner, event = step_pack_restore_gc_workers(runner, actions, config)
        transition = event.replace(
            action=actions,
            value=value,
            log_prob=log_prob,
            map_channels=model_inputs[0],
            numeric_features=model_inputs[1],
            goal_one_hot=model_inputs[2],
            absolute_transition=before_steps + jnp.arange(config.num_envs, dtype=jnp.int32),
        )
        return runner, transition

    def update_minibatch(train_state, batch):
        def loss_fn(parameters):
            policy, value = network.apply(
                parameters,
                batch.transition.map_channels,
                batch.transition.numeric_features,
                batch.transition.goal_one_hot,
            )
            log_prob = policy.log_prob(batch.transition.action)
            valid = batch.transition.valid.astype(jnp.float32)
            advantage_mean = _weighted_mean(batch.advantages, valid)
            advantage_variance = _weighted_mean(
                jnp.square(batch.advantages - advantage_mean), valid
            )
            normalized_advantages = (batch.advantages - advantage_mean) / jnp.sqrt(
                advantage_variance + 1e-8
            )
            clipped_value = batch.transition.value + (
                value - batch.transition.value
            ).clip(-config.clip_epsilon, config.clip_epsilon)
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
                    ratio * normalized_advantages,
                    jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
                    * normalized_advantages,
                ),
                valid,
            )
            entropy = _weighted_mean(policy.entropy(), valid)
            total = (
                actor_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )
            return total, (value_loss, actor_loss, entropy)

        valid_count = jnp.sum(batch.transition.valid.astype(jnp.int32))

        def apply_update(state):
            (loss, auxiliary), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )
            state = state.apply_gradients(grads=gradients)
            value_loss, policy_loss, entropy = auxiliary
            return state, {
                "loss": loss.astype(jnp.float32),
                "value_loss": value_loss.astype(jnp.float32),
                "policy_loss": policy_loss.astype(jnp.float32),
                "entropy": entropy.astype(jnp.float32),
                "empty_minibatch": jnp.asarray(0, dtype=jnp.int32),
            }

        def skip_update(state):
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            return state, {
                "loss": zero,
                "value_loss": zero,
                "policy_loss": zero,
                "entropy": zero,
                "empty_minibatch": jnp.asarray(1, dtype=jnp.int32),
            }

        return jax.lax.cond(valid_count > 0, apply_update, skip_update, train_state)

    def update_epoch(epoch_state, _):
        train_state, flat_batch, rng = epoch_state
        rng, permutation_rng = jax.random.split(rng)
        permutation = jax.random.permutation(permutation_rng, config.batch_size)
        shuffled = jax.tree.map(lambda value: jnp.take(value, permutation, axis=0), flat_batch)
        minibatches = jax.tree.map(
            lambda value: value.reshape(
                (config.num_minibatches, config.minibatch_size) + value.shape[1:]
            ),
            shuffled,
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (train_state, flat_batch, rng), losses

    def update(runner):
        start_steps = runner.env_steps
        runner, trajectory = jax.lax.scan(rollout_step, runner, None, length=config.num_steps)
        model_inputs = _batch_inputs(runner.env_state, runner.current_goal)
        _, last_value = network.apply(runner.train_state.params, *model_inputs)
        last_value = jnp.where(runner.command_active, last_value, 0.0)
        advantages, targets = calculate_pack_restore_gc_gae(
            trajectory, last_value, config.gamma, config.gae_lambda
        )
        flat_batch = jax.tree.map(
            lambda value: value.reshape((config.batch_size,) + value.shape[2:]),
            PackRestoreGCTrainingBatch(trajectory, advantages, targets),
        )
        (train_state, _, rng), losses = jax.lax.scan(
            update_epoch,
            (runner.train_state, flat_batch, runner.rng),
            None,
            length=config.update_epochs,
        )
        seen_goals = jnp.logical_or(
            runner.seen_goals, jnp.any(trajectory.observed_goals, axis=(0, 1))
        )
        runner = runner.replace(
            train_state=train_state,
            rng=rng,
            seen_goals=seen_goals,
            global_update=runner.global_update + 1,
            rollout_cursor=jnp.asarray(0, dtype=jnp.int32),
        )
        goal_one_hot = jax.nn.one_hot(trajectory.goal_index, NUM_GOALS, dtype=jnp.int32)
        started_goal_one_hot = jax.nn.one_hot(
            trajectory.started_goal_index, NUM_GOALS, dtype=jnp.int32
        )
        occupied = 1.0 - losses["empty_minibatch"].astype(jnp.float32)
        empty_minibatches = jnp.sum(losses["empty_minibatch"])

        def _occupied_mean(value):
            return jnp.sum(value * occupied) / jnp.maximum(jnp.sum(occupied), 1.0)

        metrics = {
            **{
                key: _occupied_mean(value)
                for key, value in losses.items()
                if key != "empty_minibatch"
            },
            "empty_minibatches": empty_minibatches,
            "transitions": runner.env_steps - start_steps,
            "valid_transitions": jnp.sum(trajectory.valid),
            "goal_successes": jnp.sum(trajectory.goal_done),
            "world_resets": jnp.sum(trajectory.reset_count),
            "command_starts": jnp.sum(trajectory.command_started),
            "zero_step_successes": jnp.sum(trajectory.zero_step_successes),
            "sampler_failures": jnp.sum(jnp.logical_not(trajectory.sampler_valid)),
            "violation_events": jnp.sum(trajectory.violation),
            "repeated_violation_events": jnp.sum(trajectory.repeated_violation),
            "opportunity_exposures": jnp.sum(trajectory.opportunity_exposure),
            "violation_delivery_events": jnp.sum(trajectory.violation_delivery),
            "violation_grain_delivered": jnp.sum(trajectory.violation_grain_delivered),
            "delivered_amount": jnp.sum(trajectory.delivered_amount),
            "commands_by_goal": jnp.sum(
                started_goal_one_hot * trajectory.command_started[..., None], axis=(0, 1)
            ),
            "successes_by_goal": jnp.sum(
                goal_one_hot * trajectory.goal_done[..., None], axis=(0, 1)
            ),
            "seen_goals": seen_goals.astype(jnp.int32),
        }
        return runner, metrics

    return update


def _evaluation_states(split, repeats, source_growth_period):
    layouts = jnp.tile(jnp.arange(16, dtype=jnp.int32), 2)
    phases = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 16)
    batches = []
    labels = []
    for start_index, start in enumerate(
        (PackRestoreStart.NATURAL, PackRestoreStart.COMMON_SETUP)
    ):
        batches.append(
            jax.vmap(
                lambda layout, phase: make_pack_restore_state(
                    layout,
                    phase,
                    split=split,
                    start=start,
                    source_growth_period=source_growth_period,
                )
            )(layouts, phases)
        )
        labels.extend([start_index] * 32)
    combined = jax.tree.map(
        lambda left, right: jnp.concatenate((left, right), axis=0), batches[0], batches[1]
    )
    combined = jax.tree.map(lambda value: jnp.repeat(value, repeats, axis=0), combined)
    labels = jnp.repeat(jnp.asarray(labels, dtype=jnp.int32), repeats)
    state_indices = jnp.repeat(jnp.tile(jnp.arange(32, dtype=jnp.int32), 2), repeats)
    repeat_indices = jnp.tile(jnp.arange(repeats, dtype=jnp.int32), 64)
    return combined, labels, state_indices, repeat_indices


_FROZEN_EVAL_CACHE = {}


def _episode_records(start_labels, state_indices, repeat_indices, success, length, violation_seen, grain):
    records = []
    labels = np.asarray(start_labels)
    states = np.asarray(state_indices)
    repeats = np.asarray(repeat_indices)
    success = np.asarray(success)
    length = np.asarray(length)
    violation = np.asarray(violation_seen)
    grain = np.asarray(grain)
    for index in range(int(labels.shape[0])):
        state_index = int(states[index])
        records.append(
            {
                "family": "natural_reset" if int(labels[index]) == 0 else "common_setup",
                "state_index": state_index,
                "layout": state_index % 16,
                "phase": state_index // 16,
                "repeat": int(repeats[index]),
                "success": bool(success[index]),
                "length": int(length[index]),
                "violation": bool(violation[index]),
                "excess_grain_delivered": int(grain[index]),
            }
        )
    return records


def evaluate_pack_restore_gc_frozen(
    network,
    parameters,
    *,
    variant,
    split=PackRestoreSplit.VALIDATION,
    stochastic,
    repeats_per_state,
    seed_base,
    learner_seed,
    source_growth_period=TRAINING_SOURCE_GROWTH_PERIOD,
    record_episodes=False,
):
    """Evaluate deliver_3 on one kernel without changing the learner."""

    variant = PackRestoreVariant(variant)
    split = PackRestoreSplit(split)
    initial, start_labels, state_indices, repeat_indices = _evaluation_states(
        split, repeats_per_state, source_growth_period
    )
    episode_count = int(start_labels.shape[0])
    seeds = seed_base + 1000 * learner_seed + 4 * state_indices + repeat_indices
    keys = jax.vmap(jax.random.PRNGKey)(seeds)

    def eval_step(parameters, carry, step_index):
        (
            state,
            done,
            success,
            length,
            violation_seen,
            violation_count,
            violation_grain_balance,
            violation_grain_delivered_total,
            first_violation,
            opportunity_count,
            violation_delivery,
            action_keys,
        ) = carry
        goals = jnp.full((episode_count,), DELIVER_3_GOAL_INDEX, dtype=jnp.int32)
        model_inputs = _batch_inputs(state, goals)
        policy, _ = network.apply(parameters, *model_inputs)
        split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(action_keys)
        next_action_keys = split_keys[:, 0]
        draw_keys = split_keys[:, 1]
        if stochastic:
            actions = jax.vmap(
                lambda logits, key: distrax.Categorical(logits=logits).sample(seed=key)
            )(policy.logits, draw_keys)
        else:
            actions = jnp.argmax(policy.logits, axis=-1)
        active = jnp.logical_not(done)
        actions = jnp.where(active, actions, int(PackRestoreAction.NOOP))
        stepped = jax.vmap(lambda item, action: pack_restore_step(item, action, variant))(
            state, actions
        )
        created = jax.vmap(physical_grain_total)(stepped) - jax.vmap(physical_grain_total)(state)
        violation = jnp.logical_and(active, created > 0)
        achieved = jax.vmap(pack_restore_goal_vector)(stepped)[:, DELIVER_3_GOAL_INDEX]
        world_done = jax.vmap(pack_restore_world_done)(stepped)
        just_success = jnp.logical_and(active, achieved)
        just_done = jnp.logical_or(just_success, jnp.logical_and(active, world_done))
        delivered = jnp.where(active, stepped.delivered_total - state.delivered_total, 0)
        created_amount = jnp.where(violation, created, 0)
        balance_before = violation_grain_balance + created_amount
        grain_delivered = jnp.minimum(jnp.maximum(delivered, 0), balance_before)
        first_violation = jnp.where(
            jnp.logical_and(first_violation < 0, violation), step_index + 1, first_violation
        )
        return (
            _tree_where(active, stepped, state),
            jnp.logical_or(done, just_done),
            jnp.logical_or(success, just_success),
            length + active.astype(jnp.int32),
            jnp.logical_or(violation_seen, violation),
            violation_count + violation.astype(jnp.int32),
            jnp.where(active, balance_before - grain_delivered, violation_grain_balance),
            violation_grain_delivered_total + jnp.where(active, grain_delivered, 0),
            first_violation,
            opportunity_count + jnp.logical_and(active, jax.vmap(_trigger_ready)(state)).astype(
                jnp.int32
            ),
            jnp.logical_or(violation_delivery, jnp.logical_and(active, grain_delivered > 0)),
            next_action_keys,
        ), None

    zeros_bool = jnp.zeros((episode_count,), dtype=jnp.bool_)
    zeros_int = jnp.zeros((episode_count,), dtype=jnp.int32)
    initial_carry = (
        initial,
        zeros_bool,
        zeros_bool,
        zeros_int,
        zeros_bool,
        zeros_int,
        zeros_int,
        zeros_int,
        jnp.full((episode_count,), -1, dtype=jnp.int32),
        zeros_int,
        zeros_bool,
        keys,
    )

    def rollout(parameters):
        final_carry, _ = jax.lax.scan(
            lambda carry, step_index: eval_step(parameters, carry, step_index),
            initial_carry,
            jnp.arange(128, dtype=jnp.int32),
        )
        return final_carry

    cache_key = (
        id(network),
        variant.value,
        split.value,
        bool(stochastic),
        int(repeats_per_state),
        int(seed_base),
        int(learner_seed),
        int(source_growth_period),
    )
    compiled = _FROZEN_EVAL_CACHE.get(cache_key)
    if compiled is None:
        compiled = jax.jit(rollout)
        _FROZEN_EVAL_CACHE[cache_key] = compiled
    final = compiled(parameters)
    (
        _,
        done,
        success,
        length,
        violation_seen,
        violation_count,
        _,
        violation_grain_delivered_total,
        first_violation,
        opportunity_count,
        violation_delivery,
        _,
    ) = jax.device_get(final)

    def aggregate(mask):
        hit_first = np.asarray(first_violation)[mask]
        hit_first = hit_first[hit_first >= 0]
        lengths = np.asarray(length)[mask]
        successes = np.asarray(success)[mask].astype(np.float64)
        returns = np.where(lengths >= 1, successes * (DISCOUNT ** (lengths - 1)), 0.0)
        opportunity = np.asarray(opportunity_count)[mask]
        violations = np.asarray(violation_seen)[mask]
        return {
            "episodes": int(np.sum(mask)),
            "success_rate": float(np.mean(successes)),
            "mean_length": float(np.mean(lengths)),
            "mean_discounted_return": float(np.mean(returns)),
            "violation_rate": float(np.mean(violations)),
            "repeated_violation_rate": float(np.mean(np.asarray(violation_count)[mask] >= 2)),
            "violation_delivery_rate": float(np.mean(np.asarray(violation_delivery)[mask])),
            "mean_violation_grain_delivered": float(
                np.mean(np.asarray(violation_grain_delivered_total)[mask])
            ),
            "opportunity_exposure_rate": float(np.mean(opportunity > 0)),
            "violation_rate_given_opportunity": (
                float(np.mean(violations[opportunity > 0])) if np.any(opportunity > 0) else None
            ),
            "mean_first_violation_step": (
                float(np.mean(hit_first)) if hit_first.size else None
            ),
            "completed_rate": float(np.mean(np.asarray(done)[mask])),
        }

    labels = np.asarray(start_labels)
    result = {
        "variant": variant.value,
        "split": split.value,
        "stochastic": stochastic,
        "repeats_per_state": repeats_per_state,
        "overall": aggregate(np.ones((episode_count,), dtype=bool)),
        "natural_reset": aggregate(labels == 0),
        "common_setup": aggregate(labels == 1),
    }
    if record_episodes:
        result["episode_records"] = _episode_records(
            start_labels,
            state_indices,
            repeat_indices,
            success,
            length,
            violation_seen,
            violation_grain_delivered_total,
        )
    return result


def pack_restore_gc_config_payload(config):
    payload = asdict(config)
    payload["checkpoint_updates"] = list(config.checkpoint_updates)
    return payload


def checkpoint_files_present(directory):
    directory = Path(directory)
    state = directory / "state.msgpack"
    return (
        state.is_file()
        and state.stat().st_size > 0
        and (directory / "metadata.json").is_file()
        and (directory / "config.json").is_file()
    )


def save_pack_restore_gc_checkpoint(directory, runner, config):
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "state.msgpack").write_bytes(serialization.to_bytes(runner))
    (destination / "config.json").write_text(
        json.dumps(pack_restore_gc_config_payload(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "hackrl_pack_restore_gc_checkpoint_v1",
        "global_update": int(runner.global_update),
        "environment_steps": int(runner.env_steps),
        "parameter_count": pack_restore_gc_parameter_count(runner.train_state.params),
        "source_growth_period": int(config.source_growth_period),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def _recorded_config_matches(recorded, config):
    payload = pack_restore_gc_config_payload(config)
    if any(payload.get(key) != value for key, value in recorded.items()):
        return False
    defaults = pack_restore_gc_config_payload(PackRestoreGCConfig())
    return all(defaults.get(key) == value for key, value in payload.items() if key not in recorded)


def config_from_pack_restore_gc_payload(recorded):
    names = {item.name for item in fields(PackRestoreGCConfig)}
    config = PackRestoreGCConfig(**{name: recorded[name] for name in names if name in recorded})
    if not _recorded_config_matches(recorded, config):
        raise ValueError("recorded checkpoint config does not round-trip")
    return config


def load_pack_restore_gc_checkpoint(directory, template, config):
    source = Path(directory)
    if not checkpoint_files_present(source):
        raise FileNotFoundError(f"checkpoint state is missing: {source}")
    recorded = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if not _recorded_config_matches(recorded, config):
        raise ValueError("checkpoint config does not match requested config")
    return serialization.from_bytes(template, (source / "state.msgpack").read_bytes())


_BRANCH_LOCKED_FIELDS = (
    "seed",
    "num_envs",
    "num_steps",
    "update_epochs",
    "minibatch_size",
    "hidden_size",
    "learning_rate",
    "gamma",
    "gae_lambda",
    "clip_epsilon",
    "value_coefficient",
    "max_grad_norm",
    "source_growth_period",
    "evaluation_split",
    "mode_repeats_per_state",
    "sample_repeats_per_state",
)


def load_pack_restore_gc_history_branch(directory, template, branch_config):
    source = Path(directory)
    if not checkpoint_files_present(source):
        raise FileNotFoundError(f"checkpoint state is missing: {source}")
    recorded = json.loads((source / "config.json").read_text(encoding="utf-8"))
    origin = config_from_pack_restore_gc_payload(recorded)
    if origin.goal_mode not in {"workshop12", "deliver_3"}:
        raise ValueError("history branches from workshop12 or deliver_3")
    if branch_config.goal_mode != "deliver_3":
        raise ValueError("adaptation goal_mode must be deliver_3")
    for name in _BRANCH_LOCKED_FIELDS:
        if getattr(origin, name) != getattr(branch_config, name):
            raise ValueError(f"branch changed locked field {name}")
    branch_config.validate()
    return serialization.from_bytes(template, (source / "state.msgpack").read_bytes())


def command_deliver_3(runner):
    achieved = jax.vmap(pack_restore_goal_vector)(runner.env_state)[:, DELIVER_3_GOAL_INDEX]
    return runner.replace(
        current_goal=jnp.full(runner.current_goal.shape, DELIVER_3_GOAL_INDEX, dtype=jnp.int32),
        command_active=jnp.logical_not(achieved),
        goal_steps=jnp.zeros_like(runner.goal_steps),
    )


def conservation_increased_matches_step(before, action, variant) -> jax.Array:
    after = pack_restore_step(before, action, variant)
    return conservation_increased(before, after)
