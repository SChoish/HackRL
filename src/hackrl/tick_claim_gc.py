"""Goal-conditioned PPO for TICK-CLAIM.

Goal completion is a value-learning pseudo-termination. World termination
resets the workshop. Masked pretraining also resets when every allowed goal
is already true, so a smaller goal set does not wait out the horizon.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import serialization, struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from hackrl.tick_claim import (
    GOAL_IDS,
    MAP_CHANNEL_NAMES,
    VIEW_COLUMNS,
    VIEW_ROWS,
    TickClaimAction,
    TickClaimSplit,
    TickClaimStart,
    TickClaimState,
    TickClaimVariant,
    encode_tick_claim_observation,
    make_tick_claim_state,
    observe_tick_claim,
    manual_harvest_requested,
    reset_tick_claim,
    reset_tick_claim_worker,
    tick_claim_goal_vector,
    tick_claim_step_with_snapshot,
    tick_claim_world_done,
)
from hackrl.tick_claim_oracle import (
    TickClaimOracleState,
    audit_tick_claim_transition,
    initial_tick_claim_oracle,
)


DELIVER_3_GOAL_INDEX = GOAL_IDS.index("delivery/count_ge_3")
MAP_FEATURE_COUNT = VIEW_ROWS * VIEW_COLUMNS * len(MAP_CHANNEL_NAMES)
NUM_GOALS = len(GOAL_IDS)
NUM_ACTIONS = len(TickClaimAction)
_DIRECTION_DELTAS = jnp.asarray(
    ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)), dtype=jnp.int32
)


@dataclass(frozen=True)
class TickClaimGCConfig:
    variant: str = TickClaimVariant.FIXED.value
    seed: int = 0
    num_envs: int = 512
    num_steps: int = 64
    num_updates: int = 32
    update_epochs: int = 1
    minibatch_size: int = 1024
    hidden_size: int = 512
    learning_rate: float = 2e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.005
    value_coefficient: float = 0.5
    max_grad_norm: float = 1.0
    goal_mode: str = "deliver_3"
    allowed_goals: tuple[int, ...] = ()
    goal_sampling_weights: tuple[float, ...] = ()
    evaluation_split: str = TickClaimSplit.VALIDATION.value
    mode_repeats_per_state: int = 1
    sample_repeats_per_state: int = 4
    checkpoint_updates: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(
            self, "checkpoint_updates", tuple(self.checkpoint_updates)
        )
        object.__setattr__(
            self, "allowed_goals", tuple(int(goal) for goal in self.allowed_goals)
        )
        object.__setattr__(
            self,
            "goal_sampling_weights",
            tuple(float(weight) for weight in self.goal_sampling_weights),
        )

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def num_minibatches(self) -> int:
        return self.batch_size // self.minibatch_size

    @property
    def transitions(self) -> int:
        return self.batch_size * self.num_updates

    def validate(self) -> None:
        TickClaimVariant(self.variant)
        TickClaimSplit(self.evaluation_split)
        if self.num_envs <= 0 or self.num_steps <= 0 or self.num_updates <= 0:
            raise ValueError("num_envs, num_steps, and num_updates must be positive")
        if self.update_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("update_epochs and minibatch_size must be positive")
        if self.batch_size % self.minibatch_size:
            raise ValueError("rollout batch must divide evenly into minibatches")
        if self.hidden_size <= 0 or self.learning_rate <= 0:
            raise ValueError("hidden_size and learning_rate must be positive")
        if self.goal_mode not in {"deliver_3", "workshop12", "masked"}:
            raise ValueError(
                "goal_mode must be 'deliver_3', 'workshop12', or 'masked'"
            )
        if self.goal_mode == "masked":
            allowed = self.allowed_goals
            weights = self.goal_sampling_weights
            if not allowed or len(set(allowed)) != len(allowed):
                raise ValueError("masked mode needs distinct allowed goals")
            if any(goal < 0 or goal >= NUM_GOALS for goal in allowed):
                raise ValueError("allowed goals must be workshop goal indices")
            if len(weights) != NUM_GOALS:
                raise ValueError("goal sampling weights must cover every goal")
            allowed_set = set(allowed)
            for index, weight in enumerate(weights):
                if index in allowed_set and weight <= 0:
                    raise ValueError("an allowed goal needs a positive weight")
                if index not in allowed_set and weight != 0:
                    raise ValueError("a disallowed goal must have weight zero")
        elif self.allowed_goals or self.goal_sampling_weights:
            raise ValueError("goal masks belong only to masked pretraining")
        if self.mode_repeats_per_state <= 0 or self.sample_repeats_per_state <= 0:
            raise ValueError("evaluation repeats must be positive")
        if any(
            update < 0 or update > self.num_updates
            for update in self.checkpoint_updates
        ):
            raise ValueError("checkpoint_updates must lie in [0, num_updates]")


class TickClaimGCActorCritic(nn.Module):
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
        shared = jnp.concatenate(
            (spatial, numeric_features, goal_one_hot), axis=-1
        )
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
class TickClaimGCRunnerState:
    train_state: TrainState
    env_state: TickClaimState
    oracle_state: TickClaimOracleState
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
    normalization_count: jax.Array
    normalization_mean: jax.Array
    normalization_m2: jax.Array


@struct.dataclass
class TickClaimGCStepEvent:
    goal_done: jax.Array
    world_done: jax.Array
    done_for_gae: jax.Array
    valid_transition: jax.Array
    reward: jax.Array
    observed_goals: jax.Array
    command_started: jax.Array
    zero_step_successes: jax.Array
    sampler_valid: jax.Array
    violation: jax.Array
    repeated_violation: jax.Array
    opportunity_exposure: jax.Array
    reservation_created: jax.Array
    manual_harvest_attempt: jax.Array
    violation_delivery: jax.Array
    violation_grain_delivered: jax.Array
    delivered_amount: jax.Array
    success_steps: jax.Array
    reset_count: jax.Array
    terminal_goals: jax.Array


class TickClaimGCTransition(NamedTuple):
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
    reservation_created: jax.Array
    manual_harvest_attempt: jax.Array
    violation_delivery: jax.Array
    violation_grain_delivered: jax.Array
    delivered_amount: jax.Array
    success_steps: jax.Array
    reset_count: jax.Array
    observed_goals: jax.Array
    absolute_transition: jax.Array


class TickClaimGCTrainingBatch(NamedTuple):
    transition: TickClaimGCTransition
    advantages: jax.Array
    targets: jax.Array


def tick_claim_gc_inputs(observation, goal_index):
    encoded = encode_tick_claim_observation(observation)
    map_channels = encoded[:MAP_FEATURE_COUNT].reshape(
        VIEW_ROWS, VIEW_COLUMNS, len(MAP_CHANNEL_NAMES)
    )
    numeric_features = encoded[MAP_FEATURE_COUNT:]
    goal_one_hot = jax.nn.one_hot(goal_index, NUM_GOALS, dtype=jnp.float32)
    return map_channels, numeric_features, goal_one_hot


def _batch_inputs(env_state, goals):
    observations = jax.vmap(observe_tick_claim)(env_state)
    return jax.vmap(tick_claim_gc_inputs)(observations, goals)


def tick_claim_gc_parameter_count(parameters) -> int:
    return int(sum(np.asarray(value).size for value in jax.tree.leaves(parameters)))


def _optimizer(config):
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate, eps=1e-5),
    )


def _broadcast_oracle(num_envs):
    base = initial_tick_claim_oracle()
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (num_envs,) + value.shape), base
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


def _fresh_train_state(key):
    state = reset_tick_claim(
        key, split=TickClaimSplit.TRAIN, start=TickClaimStart.NATURAL
    )
    goals = tick_claim_goal_vector(observe_tick_claim(state))
    return state, goals


def _resample_masked_goal(key, env_state, goals, weights):
    """Sample a currently false allowed goal, resetting while none is false."""

    def cond(state):
        attempts, accepted, *_rest = state
        return jnp.logical_and(jnp.logical_not(accepted), attempts < 8)

    def body(state):
        attempts, _accepted, loop_key, current, current_goals, goal, resets = state
        loop_key, draw_key, reset_key = jax.random.split(loop_key, 3)
        eligible = jnp.logical_and(weights > 0, jnp.logical_not(current_goals))
        logits = jnp.where(
            eligible,
            jnp.log(jnp.maximum(weights, 1e-8)),
            jnp.asarray(-1e9, dtype=jnp.float32),
        )
        candidate = jax.random.categorical(draw_key, logits).astype(jnp.int32)
        valid = jnp.any(eligible)

        def keep(_unused):
            return current, current_goals

        def refresh(reset_rng):
            return _fresh_train_state(reset_rng)

        refreshed, refreshed_goals = jax.lax.cond(valid, keep, refresh, reset_key)
        return (
            attempts + 1,
            valid,
            loop_key,
            refreshed,
            refreshed_goals,
            jnp.where(valid, candidate, goal),
            resets + jnp.logical_not(valid).astype(jnp.int32),
        )

    _attempts, accepted, key, env_state, _goals, goal, resets = jax.lax.while_loop(
        cond,
        body,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            key,
            env_state,
            goals,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    return goal, key, env_state, resets, accepted


def initialize_tick_claim_gc(config):
    config.validate()
    init_rng, action_rng, env_rng, sampler_rng = jax.random.split(
        jax.random.PRNGKey(config.seed), 4
    )
    workers = jnp.arange(config.num_envs, dtype=jnp.int32) % 512
    env_state = jax.vmap(reset_tick_claim_worker)(workers)
    observations = jax.vmap(observe_tick_claim)(env_state)
    goal_vectors = jax.vmap(tick_claim_goal_vector)(observations)
    seen_goals = jnp.any(goal_vectors, axis=0)
    false_seen = jnp.logical_and(
        seen_goals[None, :], jnp.logical_not(goal_vectors)
    )
    if not bool(jnp.all(jnp.any(false_seen, axis=1))):
        raise ValueError("at least one worker has no false seen goal")
    env_keys = jax.random.split(env_rng, config.num_envs)
    sampler_keys = jax.random.split(sampler_rng, config.num_envs)
    if config.goal_mode == "deliver_3":
        current_goal = jnp.full(
            (config.num_envs,), DELIVER_3_GOAL_INDEX, dtype=jnp.int32
        )
    elif config.goal_mode == "workshop12":
        sampled = jax.vmap(sample_false_seen_goal, in_axes=(0, None, 0))(
            sampler_keys, seen_goals, goal_vectors
        )
        current_goal, sampler_keys, _, sampler_valid = sampled
        if not bool(jnp.all(sampler_valid)):
            raise ValueError("initial goal sampler could not find a false goal")
    else:
        weights = jnp.asarray(config.goal_sampling_weights, dtype=jnp.float32)
        current_goal, sampler_keys, env_state, _, sampler_valid = jax.vmap(
            _resample_masked_goal, in_axes=(0, 0, 0, None)
        )(sampler_keys, env_state, goal_vectors, weights)
        if not bool(jnp.all(sampler_valid)):
            raise ValueError("masked pretraining could not find a false allowed goal")
        observations = jax.vmap(observe_tick_claim)(env_state)
    model_inputs = jax.vmap(tick_claim_gc_inputs)(
        observations, current_goal
    )
    network = TickClaimGCActorCritic(hidden_size=config.hidden_size)
    parameters = network.init(
        init_rng,
        model_inputs[0][:1],
        model_inputs[1][:1],
        model_inputs[2][:1],
    )
    train_state = TrainState.create(
        apply_fn=network.apply, params=parameters, tx=_optimizer(config)
    )
    runner = TickClaimGCRunnerState(
        train_state=train_state,
        env_state=env_state,
        oracle_state=_broadcast_oracle(config.num_envs),
        current_goal=current_goal,
        command_active=jnp.ones((config.num_envs,), dtype=jnp.bool_),
        env_keys=env_keys,
        sampler_keys=sampler_keys,
        rng=action_rng,
        seen_goals=seen_goals,
        goal_steps=jnp.zeros((config.num_envs,), dtype=jnp.int32),
        world_violation_count=jnp.zeros((config.num_envs,), dtype=jnp.int32),
        violation_grain_balance=jnp.zeros(
            (config.num_envs,), dtype=jnp.int32
        ),
        global_update=jnp.asarray(0, dtype=jnp.int32),
        env_steps=jnp.asarray(0, dtype=jnp.int32),
        rollout_cursor=jnp.asarray(0, dtype=jnp.int32),
        normalization_count=jnp.asarray(0, dtype=jnp.int32),
        normalization_mean=jnp.zeros((0,), dtype=jnp.float32),
        normalization_m2=jnp.zeros((0,), dtype=jnp.float32),
    )
    return network, runner


def _opportunity_exposure(state):
    direction = jnp.clip(state.player_direction, 0, 4)
    front = state.player_position + _DIRECTION_DELTAS[direction]
    reservation_matches = jnp.logical_and(
        state.reservation_object_id == state.crop_object_id,
        jnp.logical_and(
            state.reservation_generation == state.crop_generation,
            state.reservation_cycle_id == state.crop_cycle_id,
        ),
    )
    return jnp.logical_and(
        state.crop_ripe,
        jnp.logical_and(
            state.reservation_present,
            jnp.logical_and(
                state.reservation_due_tick == state.tick,
                jnp.logical_and(
                    reservation_matches,
                    jnp.all(front == state.crop_position),
                ),
            ),
        ),
    )


def step_tick_claim_gc_workers(runner, actions, config):
    """Step workers; goal_done cuts value flow and world_done alone resets."""

    variant = TickClaimVariant(config.variant)
    before = runner.env_state
    valid_transition = runner.command_active
    actions = jnp.where(
        valid_transition, actions, jnp.asarray(TickClaimAction.NOOP)
    )
    opportunity = jax.vmap(_opportunity_exposure)(before)
    stepped, snapshots = jax.vmap(
        lambda state, action: tick_claim_step_with_snapshot(
            state, action, variant
        )
    )(before, actions)
    audits = jax.vmap(audit_tick_claim_transition)(
        runner.oracle_state, before, actions, stepped, snapshots
    )
    terminal_observations = jax.vmap(observe_tick_claim)(stepped)
    terminal_goals = jax.vmap(tick_claim_goal_vector)(terminal_observations)
    achieved_command = jnp.take_along_axis(
        terminal_goals, runner.current_goal[:, None], axis=1
    )[:, 0]
    goal_done = jnp.logical_and(valid_transition, achieved_command)
    world_done = jax.vmap(tick_claim_world_done)(stepped)
    done_for_gae = jnp.logical_or(
        jnp.logical_or(goal_done, world_done),
        jnp.logical_not(valid_transition),
    )

    split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(
        runner.env_keys
    )
    next_env_keys = split_keys[:, 0]
    reset_keys = split_keys[:, 1]
    reset_states = jax.vmap(
        lambda key: reset_tick_claim(
            key, split=TickClaimSplit.TRAIN, start=TickClaimStart.NATURAL
        )
    )(reset_keys)
    reset_observations = jax.vmap(observe_tick_claim)(reset_states)
    reset_goals = jax.vmap(tick_claim_goal_vector)(reset_observations)
    env_state = _tree_where(world_done, reset_states, stepped)
    oracle_state = _tree_where(
        world_done, _broadcast_oracle(config.num_envs), audits.oracle_state
    )
    observed_goals = jnp.logical_or(
        terminal_goals,
        jnp.where(world_done[:, None], reset_goals, False),
    )

    switch = jnp.logical_or(goal_done, world_done)
    if config.goal_mode == "workshop12":
        command_observation_goals = jnp.where(
            world_done[:, None], reset_goals, terminal_goals
        )
        sampled = jax.vmap(sample_false_seen_goal, in_axes=(0, None, 0))(
            runner.sampler_keys,
            runner.seen_goals,
            command_observation_goals,
        )
        sampled_goal, sampled_keys, zero_steps, sampler_valid = sampled
        current_goal = jnp.where(switch, sampled_goal, runner.current_goal)
        sampler_keys = jnp.where(
            switch[:, None], sampled_keys, runner.sampler_keys
        )
        zero_step_successes = jnp.where(switch, zero_steps, 0)
        command_started = switch
        command_active = jnp.ones_like(runner.command_active)
        sampler_valid = jnp.where(switch, sampler_valid, True)
        masked_extra_reset = jnp.zeros_like(world_done)
    elif config.goal_mode == "masked":
        weights = jnp.asarray(config.goal_sampling_weights, dtype=jnp.float32)
        continuation_goals = jnp.where(
            world_done[:, None], reset_goals, terminal_goals
        )
        (
            sampled_goal,
            sampled_keys,
            sampled_env,
            extra_resets,
            sampled_ok,
        ) = jax.vmap(_resample_masked_goal, in_axes=(0, 0, 0, None))(
            runner.sampler_keys, env_state, continuation_goals, weights
        )
        current_goal = jnp.where(switch, sampled_goal, runner.current_goal)
        sampler_keys = jnp.where(
            switch[:, None], sampled_keys, runner.sampler_keys
        )
        env_state = _tree_where(switch, sampled_env, env_state)
        masked_extra_reset = jnp.logical_and(switch, extra_resets > 0)
        oracle_state = _tree_where(
            masked_extra_reset,
            _broadcast_oracle(config.num_envs),
            oracle_state,
        )
        zero_step_successes = jnp.zeros_like(runner.goal_steps)
        command_started = jnp.logical_and(switch, sampled_ok)
        command_active = jnp.where(switch, sampled_ok, runner.command_active)
        sampler_valid = jnp.where(switch, sampled_ok, True)
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
        masked_extra_reset = jnp.zeros_like(world_done)

    reservation_created = jnp.logical_and(
        jnp.logical_not(before.reservation_present),
        stepped.reservation_present,
    )
    manual_harvest_attempt = jax.vmap(manual_harvest_requested)(before, actions)
    delivered_amount = stepped.delivered_total - before.delivered_total
    same_cycle = jnp.logical_and(
        runner.oracle_state.initialized,
        jnp.logical_and(
            runner.oracle_state.object_id == before.crop_object_id,
            jnp.logical_and(
                runner.oracle_state.generation == before.crop_generation,
                runner.oracle_state.cycle_id == before.crop_cycle_id,
            ),
        ),
    )
    prior_payout = jnp.where(
        same_cycle, runner.oracle_state.payout_total, 0
    )
    prior_excess = jnp.maximum(prior_payout - 1, 0)
    current_excess = jnp.maximum(audits.cycle_payout_total - 1, 0)
    violation_grain_created = jnp.where(
        valid_transition, current_excess - prior_excess, 0
    )
    violation = jnp.logical_and(
        audits.violation, violation_grain_created > 0
    )
    repeated_violation = jnp.logical_and(
        violation, runner.world_violation_count > 0
    )
    violation_grain_before_delivery = (
        runner.violation_grain_balance + violation_grain_created
    )
    violation_grain_delivered = jnp.minimum(
        jnp.maximum(delivered_amount, 0), violation_grain_before_delivery
    )
    violation_delivery = violation_grain_delivered > 0
    violation_grain_balance = (
        violation_grain_before_delivery - violation_grain_delivered
    )
    violation_count = runner.world_violation_count + violation.astype(jnp.int32)
    violation_count = jnp.where(
        jnp.logical_or(world_done, masked_extra_reset), 0, violation_count
    )
    violation_grain_balance = jnp.where(
        jnp.logical_or(world_done, masked_extra_reset),
        0,
        violation_grain_balance,
    )
    success_steps = jnp.where(goal_done, runner.goal_steps + 1, 0)
    goal_steps = jnp.where(
        switch,
        0,
        jnp.where(
            runner.command_active,
            runner.goal_steps + 1,
            runner.goal_steps,
        ),
    )
    next_runner = runner.replace(
        env_state=env_state,
        oracle_state=oracle_state,
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
    event = TickClaimGCStepEvent(
        goal_done=goal_done,
        world_done=world_done,
        done_for_gae=done_for_gae,
        valid_transition=valid_transition,
        reward=goal_done.astype(jnp.float32),
        observed_goals=observed_goals,
        command_started=command_started,
        zero_step_successes=zero_step_successes,
        sampler_valid=sampler_valid,
        violation=violation,
        repeated_violation=repeated_violation,
        opportunity_exposure=jnp.logical_and(valid_transition, opportunity),
        reservation_created=jnp.logical_and(
            valid_transition, reservation_created
        ),
        manual_harvest_attempt=jnp.logical_and(
            valid_transition, manual_harvest_attempt
        ),
        violation_delivery=jnp.logical_and(valid_transition, violation_delivery),
        violation_grain_delivered=jnp.where(
            valid_transition, violation_grain_delivered, 0
        ),
        delivered_amount=jnp.where(valid_transition, delivered_amount, 0),
        success_steps=success_steps,
        reset_count=jnp.logical_or(world_done, masked_extra_reset).astype(
            jnp.int32
        ),
        terminal_goals=terminal_goals,
    )
    return next_runner, event


def calculate_tick_claim_gc_gae(
    trajectory, last_value, gamma, gae_lambda
):
    """Calculate GAE across rollout cuts but not goal/world termination."""

    def backward(carry, transition):
        gae, next_value = carry
        nonterminal = 1.0 - transition.done.astype(jnp.float32)
        delta = (
            transition.reward
            + gamma * next_value * nonterminal
            - transition.value
        )
        gae = (
            delta + gamma * gae_lambda * nonterminal * gae
        ) * transition.valid.astype(jnp.float32)
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


def make_tick_claim_gc_update(network, config):
    """Build one JIT-compatible rollout and PPO update."""

    def rollout_step(runner, _):
        command_goal = runner.current_goal
        model_inputs = _batch_inputs(runner.env_state, command_goal)
        policy, value = network.apply(
            runner.train_state.params, *model_inputs
        )
        rng, action_rng = jax.random.split(runner.rng)
        actions = policy.sample(seed=action_rng)
        log_prob = policy.log_prob(actions)
        before_steps = runner.env_steps
        runner = runner.replace(rng=rng)
        runner, event = step_tick_claim_gc_workers(
            runner, actions, config
        )
        transition = TickClaimGCTransition(
            done=event.done_for_gae,
            valid=event.valid_transition,
            action=actions,
            value=value,
            reward=event.reward,
            log_prob=log_prob,
            map_channels=model_inputs[0],
            numeric_features=model_inputs[1],
            goal_one_hot=model_inputs[2],
            goal_index=command_goal,
            started_goal_index=runner.current_goal,
            goal_done=event.goal_done,
            world_done=event.world_done,
            command_started=event.command_started,
            zero_step_successes=event.zero_step_successes,
            sampler_valid=event.sampler_valid,
            violation=event.violation,
            repeated_violation=event.repeated_violation,
            opportunity_exposure=event.opportunity_exposure,
            reservation_created=event.reservation_created,
            manual_harvest_attempt=event.manual_harvest_attempt,
            violation_delivery=event.violation_delivery,
            violation_grain_delivered=event.violation_grain_delivered,
            delivered_amount=event.delivered_amount,
            success_steps=event.success_steps,
            reset_count=event.reset_count,
            observed_goals=event.observed_goals,
            absolute_transition=(
                before_steps
                + jnp.arange(config.num_envs, dtype=jnp.int32)
            ),
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
            normalized_advantages = (
                batch.advantages - advantage_mean
            ) / jnp.sqrt(advantage_variance + 1e-8)
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
                    jnp.clip(
                        ratio,
                        1.0 - config.clip_epsilon,
                        1.0 + config.clip_epsilon,
                    )
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
            (loss, auxiliary), gradients = jax.value_and_grad(
                loss_fn, has_aux=True
            )(state.params)
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
        permutation = jax.random.permutation(
            permutation_rng, config.batch_size
        )
        shuffled = jax.tree.map(
            lambda value: jnp.take(value, permutation, axis=0),
            flat_batch,
        )
        minibatches = jax.tree.map(
            lambda value: value.reshape(
                (config.num_minibatches, config.minibatch_size)
                + value.shape[1:]
            ),
            shuffled,
        )
        train_state, losses = jax.lax.scan(
            update_minibatch, train_state, minibatches
        )
        return (train_state, flat_batch, rng), losses

    def update(runner):
        start_steps = runner.env_steps
        runner, trajectory = jax.lax.scan(
            rollout_step, runner, None, length=config.num_steps
        )
        model_inputs = _batch_inputs(
            runner.env_state, runner.current_goal
        )
        _, last_value = network.apply(
            runner.train_state.params, *model_inputs
        )
        last_value = jnp.where(runner.command_active, last_value, 0.0)
        advantages, targets = calculate_tick_claim_gc_gae(
            trajectory,
            last_value,
            config.gamma,
            config.gae_lambda,
        )
        flat_batch = jax.tree.map(
            lambda value: value.reshape(
                (config.batch_size,) + value.shape[2:]
            ),
            TickClaimGCTrainingBatch(trajectory, advantages, targets),
        )
        (train_state, _, rng), losses = jax.lax.scan(
            update_epoch,
            (runner.train_state, flat_batch, runner.rng),
            None,
            length=config.update_epochs,
        )
        seen_goals = jnp.logical_or(
            runner.seen_goals,
            jnp.any(trajectory.observed_goals, axis=(0, 1)),
        )
        runner = runner.replace(
            train_state=train_state,
            rng=rng,
            seen_goals=seen_goals,
            global_update=runner.global_update + 1,
            rollout_cursor=jnp.asarray(0, dtype=jnp.int32),
        )
        goal_one_hot = jax.nn.one_hot(
            trajectory.goal_index, NUM_GOALS, dtype=jnp.int32
        )
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
            "zero_step_successes": jnp.sum(
                trajectory.zero_step_successes
            ),
            "sampler_failures": jnp.sum(
                jnp.logical_not(trajectory.sampler_valid)
            ),
            "violation_events": jnp.sum(trajectory.violation),
            "repeated_violation_events": jnp.sum(
                trajectory.repeated_violation
            ),
            "opportunity_exposures": jnp.sum(
                trajectory.opportunity_exposure
            ),
            "reservation_creations": jnp.sum(
                trajectory.reservation_created
            ),
            "manual_harvest_attempts": jnp.sum(
                trajectory.manual_harvest_attempt
            ),
            "violation_delivery_events": jnp.sum(
                trajectory.violation_delivery
            ),
            "violation_grain_delivered": jnp.sum(
                trajectory.violation_grain_delivered
            ),
            "delivered_amount": jnp.sum(trajectory.delivered_amount),
            "successful_transition_sum": jnp.sum(
                trajectory.success_steps
            ),
            "commands_by_goal": jnp.sum(
                started_goal_one_hot
                * trajectory.command_started[..., None],
                axis=(0, 1),
            ),
            "valid_by_goal": jnp.sum(
                goal_one_hot
                * trajectory.valid[..., None].astype(jnp.int32),
                axis=(0, 1),
            ),
            "successes_by_goal": jnp.sum(
                goal_one_hot * trajectory.goal_done[..., None],
                axis=(0, 1),
            ),
            "seen_goals": seen_goals.astype(jnp.int32),
            "first_violation_transition": jnp.min(
                jnp.where(
                    trajectory.violation,
                    trajectory.absolute_transition,
                    jnp.iinfo(jnp.int32).max,
                )
            ),
            "first_opportunity_transition": jnp.min(
                jnp.where(
                    trajectory.opportunity_exposure,
                    trajectory.absolute_transition,
                    jnp.iinfo(jnp.int32).max,
                )
            ),
            "first_reservation_transition": jnp.min(
                jnp.where(
                    trajectory.reservation_created,
                    trajectory.absolute_transition,
                    jnp.iinfo(jnp.int32).max,
                )
            ),
            "first_repeated_violation_transition": jnp.min(
                jnp.where(
                    trajectory.repeated_violation,
                    trajectory.absolute_transition,
                    jnp.iinfo(jnp.int32).max,
                )
            ),
            "first_violation_delivery_transition": jnp.min(
                jnp.where(
                    trajectory.violation_delivery,
                    trajectory.absolute_transition,
                    jnp.iinfo(jnp.int32).max,
                )
            ),
            "first_success_transition": jnp.min(
                jnp.where(
                    trajectory.goal_done,
                    trajectory.absolute_transition,
                    jnp.iinfo(jnp.int32).max,
                )
            ),
        }
        return runner, metrics

    return update


def tick_claim_gc_policy_logits(network, parameters, runner):
    model_inputs = _batch_inputs(runner.env_state, runner.current_goal)
    policy, _ = network.apply(parameters, *model_inputs)
    return policy.logits


def _evaluation_states(split, repeats):
    layouts = jnp.tile(jnp.arange(16, dtype=jnp.int32), 2)
    phases = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 16)
    batches = []
    labels = []
    for start_index, start in enumerate(
        (TickClaimStart.NATURAL, TickClaimStart.COMMON_SETUP)
    ):
        batches.append(
            jax.vmap(
                lambda layout, phase: make_tick_claim_state(
                    layout, phase, split=split, start=start
                )
            )(layouts, phases)
        )
        labels.extend([start_index] * 32)
    combined = jax.tree.map(
        lambda left, right: jnp.concatenate((left, right), axis=0),
        batches[0],
        batches[1],
    )
    combined = jax.tree.map(
        lambda value: jnp.repeat(value, repeats, axis=0), combined
    )
    labels = jnp.repeat(jnp.asarray(labels, dtype=jnp.int32), repeats)
    state_indices = jnp.repeat(
        jnp.tile(jnp.arange(32, dtype=jnp.int32), 2), repeats
    )
    repeat_indices = jnp.tile(jnp.arange(repeats, dtype=jnp.int32), 64)
    return combined, labels, state_indices, repeat_indices


_FROZEN_EVAL_CACHE = {}


def _episode_records(
    start_labels, state_indices, repeat_indices, success, length, violation_seen, grain
):
    """One row per eval episode so return and state effects can be recomputed.

    Discounted return is not stored. With a unit reward on the success step it is
    success * gamma ** (length - 1) when length is at least 1.
    """

    labels = np.asarray(start_labels)
    states = np.asarray(state_indices)
    repeats = np.asarray(repeat_indices)
    success = np.asarray(success)
    length = np.asarray(length)
    violation = np.asarray(violation_seen)
    grain = np.asarray(grain)
    records = []
    for index in range(int(labels.shape[0])):
        state_index = int(states[index])
        records.append(
            {
                "family": (
                    "natural_reset" if int(labels[index]) == 0 else "common_setup"
                ),
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


def evaluate_tick_claim_gc_frozen(
    network,
    parameters,
    *,
    variant,
    split=TickClaimSplit.VALIDATION,
    stochastic,
    repeats_per_state,
    seed_base,
    learner_seed,
    record_episodes=False,
):
    """Evaluate deliver_3 without mutating learner or sampler state.

    The rollout is compiled once for a network object and eval settings.
    Later calls with new parameters reuse that compilation.
    """

    variant = TickClaimVariant(variant)
    split = TickClaimSplit(split)
    initial, start_labels, state_indices, repeat_indices = _evaluation_states(
        split, repeats_per_state
    )
    episode_count = int(start_labels.shape[0])
    seeds = (
        seed_base
        + 1000 * learner_seed
        + 4 * state_indices
        + repeat_indices
    )
    keys = jax.vmap(jax.random.PRNGKey)(seeds)
    oracle = _broadcast_oracle(episode_count)

    def eval_step(parameters, carry, step_index):
        (
            state,
            oracle_state,
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
            delivered_total,
            action_keys,
        ) = carry
        goals = jnp.full(
            (episode_count,), DELIVER_3_GOAL_INDEX, dtype=jnp.int32
        )
        model_inputs = _batch_inputs(state, goals)
        policy, _ = network.apply(parameters, *model_inputs)
        split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(
            action_keys
        )
        next_action_keys = split_keys[:, 0]
        draw_keys = split_keys[:, 1]
        if stochastic:
            actions = jax.vmap(
                lambda logits, key: distrax.Categorical(
                    logits=logits
                ).sample(seed=key)
            )(policy.logits, draw_keys)
        else:
            actions = jnp.argmax(policy.logits, axis=-1)
        active = jnp.logical_not(done)
        actions = jnp.where(active, actions, TickClaimAction.NOOP)
        stepped, snapshots = jax.vmap(
            lambda item, action: tick_claim_step_with_snapshot(
                item, action, variant
            )
        )(state, actions)
        audits = jax.vmap(audit_tick_claim_transition)(
            oracle_state, state, actions, stepped, snapshots
        )
        observations = jax.vmap(observe_tick_claim)(stepped)
        achieved = jax.vmap(tick_claim_goal_vector)(observations)[
            :, DELIVER_3_GOAL_INDEX
        ]
        world_done = jax.vmap(tick_claim_world_done)(stepped)
        just_success = jnp.logical_and(active, achieved)
        just_done = jnp.logical_or(
            just_success, jnp.logical_and(active, world_done)
        )
        opportunity = jnp.logical_and(
            active, jax.vmap(_opportunity_exposure)(state)
        )
        delivered = jnp.where(
            active, stepped.delivered_total - state.delivered_total, 0
        )
        same_cycle = jnp.logical_and(
            oracle_state.initialized,
            jnp.logical_and(
                oracle_state.object_id == state.crop_object_id,
                jnp.logical_and(
                    oracle_state.generation == state.crop_generation,
                    oracle_state.cycle_id == state.crop_cycle_id,
                ),
            ),
        )
        prior_payout = jnp.where(same_cycle, oracle_state.payout_total, 0)
        prior_excess = jnp.maximum(prior_payout - 1, 0)
        current_excess = jnp.maximum(
            audits.cycle_payout_total - 1, 0
        )
        violation_grain_created = jnp.where(
            active, current_excess - prior_excess, 0
        )
        violation = jnp.logical_and(
            audits.violation, violation_grain_created > 0
        )
        grain_before_delivery = (
            violation_grain_balance + violation_grain_created
        )
        violation_grain_delivered = jnp.minimum(
            jnp.maximum(delivered, 0), grain_before_delivery
        )
        first_violation = jnp.where(
            jnp.logical_and(first_violation < 0, violation),
            step_index + 1,
            first_violation,
        )
        return (
            _tree_where(active, stepped, state),
            _tree_where(active, audits.oracle_state, oracle_state),
            jnp.logical_or(done, just_done),
            jnp.logical_or(success, just_success),
            length + active.astype(jnp.int32),
            jnp.logical_or(violation_seen, violation),
            violation_count + violation.astype(jnp.int32),
            grain_before_delivery - violation_grain_delivered,
            violation_grain_delivered_total + violation_grain_delivered,
            first_violation,
            opportunity_count + opportunity.astype(jnp.int32),
            jnp.logical_or(
                violation_delivery,
                jnp.logical_and(active, violation_grain_delivered > 0),
            ),
            jnp.where(active, stepped.delivered_total, delivered_total),
            next_action_keys,
        ), None

    zeros_bool = jnp.zeros((episode_count,), dtype=jnp.bool_)
    zeros_int = jnp.zeros((episode_count,), dtype=jnp.int32)
    initial_carry = (
        initial,
        oracle,
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
        zeros_int,
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
    )
    compiled = _FROZEN_EVAL_CACHE.get(cache_key)
    if compiled is None:
        compiled = jax.jit(rollout)
        _FROZEN_EVAL_CACHE[cache_key] = compiled
    final = compiled(parameters)
    (
        _,
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
        delivered_total,
        _,
    ) = jax.device_get(final)

    def aggregate(mask):
        hit_first = np.asarray(first_violation)[mask]
        hit_first = hit_first[hit_first >= 0]
        return {
            "episodes": int(np.sum(mask)),
            "success_rate": float(np.mean(np.asarray(success)[mask])),
            "mean_length": float(np.mean(np.asarray(length)[mask])),
            "violation_rate": float(
                np.mean(np.asarray(violation_seen)[mask])
            ),
            "repeated_violation_rate": float(
                np.mean(np.asarray(violation_count)[mask] >= 2)
            ),
            "violation_delivery_rate": float(
                np.mean(np.asarray(violation_delivery)[mask])
            ),
            "mean_violation_grain_delivered": float(
                np.mean(
                    np.asarray(violation_grain_delivered_total)[mask]
                )
            ),
            "opportunity_exposure_rate": float(
                np.mean(np.asarray(opportunity_count)[mask] > 0)
            ),
            "violation_rate_given_opportunity": (
                float(
                    np.mean(
                        np.asarray(violation_seen)[mask][
                            np.asarray(opportunity_count)[mask] > 0
                        ]
                    )
                )
                if np.any(np.asarray(opportunity_count)[mask] > 0)
                else None
            ),
            "mean_delivered_total": float(
                np.mean(np.asarray(delivered_total)[mask])
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


def tick_claim_gc_config_payload(config):
    payload = asdict(config)
    payload["checkpoint_updates"] = list(config.checkpoint_updates)
    payload["allowed_goals"] = list(config.allowed_goals)
    payload["goal_sampling_weights"] = list(config.goal_sampling_weights)
    payload.update(
        {
            "schema_version": "hackrl_gc_v1",
            "batch_size": config.batch_size,
            "num_minibatches": config.num_minibatches,
            "transitions": config.transitions,
            "action_mask": "none",
            "input_normalization": "none",
            "value_output": "sigmoid",
        }
    )
    return payload


def _git_sha():
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _git_diff():
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        [
            "git",
            "diff",
            "--binary",
            "HEAD",
            "--",
            "src/hackrl/tick_claim_gc.py",
        ],
        cwd=root,
        text=True,
    )


def save_tick_claim_gc_checkpoint(directory, runner, config):
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "state.msgpack").write_bytes(
        serialization.to_bytes(runner)
    )
    (destination / "config.json").write_text(
        json.dumps(
            tick_claim_gc_config_payload(config),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "hackrl_gc_checkpoint_v1",
        "code_revision": _git_sha(),
        "global_update": int(runner.global_update),
        "environment_steps": int(runner.env_steps),
        "rollout_cursor": int(runner.rollout_cursor),
        "normalization": "none",
        "parameter_count": tick_claim_gc_parameter_count(
            runner.train_state.params
        ),
        "contains": [
            "model_parameters",
            "optimizer_state_and_step",
            "worker_environment_state_and_rng",
            "worker_current_goal_and_progress",
            "seen_goals_and_sampler_rng",
            "learner_rng",
            "world_violation_provenance",
            "rollout_cursor",
            "normalization_state",
            "schedule_position",
            "run_config",
        ],
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def checkpoint_files_present(directory):
    """A checkpoint counts only when its parameter state file is on disk."""

    directory = Path(directory)
    state = directory / "state.msgpack"
    return (
        state.is_file()
        and state.stat().st_size > 0
        and (directory / "metadata.json").is_file()
        and (directory / "config.json").is_file()
    )


def _recorded_config_matches(recorded, config):
    """Accept a checkpoint that omits fields still at their default values."""

    payload = tick_claim_gc_config_payload(config)
    if any(payload.get(key) != value for key, value in recorded.items()):
        return False
    defaults = tick_claim_gc_config_payload(TickClaimGCConfig())
    return all(
        defaults.get(key) == value
        for key, value in payload.items()
        if key not in recorded
    )


def load_tick_claim_gc_checkpoint(directory, template, config):
    source = Path(directory)
    if not checkpoint_files_present(source):
        raise FileNotFoundError(f"checkpoint state is missing: {source}")
    recorded = json.loads(
        (source / "config.json").read_text(encoding="utf-8")
    )
    if not _recorded_config_matches(recorded, config):
        raise ValueError("checkpoint config does not match requested config")
    return serialization.from_bytes(
        template, (source / "state.msgpack").read_bytes()
    )


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
    "evaluation_split",
    "mode_repeats_per_state",
    "sample_repeats_per_state",
)


def config_from_tick_claim_gc_payload(recorded):
    names = {item.name for item in fields(TickClaimGCConfig)}
    config = TickClaimGCConfig(
        **{name: recorded[name] for name in names if name in recorded}
    )
    if not _recorded_config_matches(recorded, config):
        raise ValueError("recorded checkpoint config does not round-trip")
    return config


def load_tick_claim_gc_branch(directory, template, branch_config):
    """Load a workshop12 checkpoint for a deliver_3 fixed or mutant branch.

    Architecture, seed, and optimization hyperparameters stay locked. The
    branch may change variant, goal mode, adaptation budget, and the entropy
    coefficient. Entropy is the exploration intervention, so it is not locked.
    """

    source = Path(directory)
    if not checkpoint_files_present(source):
        raise FileNotFoundError(f"checkpoint state is missing: {source}")
    recorded = json.loads((source / "config.json").read_text(encoding="utf-8"))
    origin = config_from_tick_claim_gc_payload(recorded)
    if origin.goal_mode != "workshop12":
        raise ValueError("adaptation branches only from workshop12 checkpoints")
    if branch_config.goal_mode != "deliver_3":
        raise ValueError("adaptation goal_mode must be deliver_3")
    for name in _BRANCH_LOCKED_FIELDS:
        if getattr(origin, name) != getattr(branch_config, name):
            raise ValueError(f"branch changed locked field {name}")
    branch_config.validate()
    return serialization.from_bytes(
        template, (source / "state.msgpack").read_bytes()
    )


def load_tick_claim_gc_history_branch(directory, template, branch_config):
    """Continue a workshop12 or deliver_3 checkpoint into deliver_3 adaptation.

    A workshop12 origin still has its goal switched by command_deliver_3.
    A deliver_3 origin keeps the goal mode and the worker state. Locked
    hyperparameters match the workshop12 branch.
    """

    source = Path(directory)
    if not checkpoint_files_present(source):
        raise FileNotFoundError(f"checkpoint state is missing: {source}")
    recorded = json.loads((source / "config.json").read_text(encoding="utf-8"))
    origin = config_from_tick_claim_gc_payload(recorded)
    if origin.goal_mode not in {"workshop12", "deliver_3"}:
        raise ValueError("history branches from workshop12 or deliver_3")
    if branch_config.goal_mode != "deliver_3":
        raise ValueError("adaptation goal_mode must be deliver_3")
    for name in _BRANCH_LOCKED_FIELDS:
        if getattr(origin, name) != getattr(branch_config, name):
            raise ValueError(f"branch changed locked field {name}")
    branch_config.validate()
    return serialization.from_bytes(
        template, (source / "state.msgpack").read_bytes()
    )


def reinit_tick_claim_gc_adaptation_start(runner, seed):
    """Replace worker env and RNG from the seed. Parameters and optimizer stay.

    The same seed produces the same environment and RNG for every condition.
    Worker layouts use the standard index assignment. The learner RNG is a
    fresh split of that seed, not a continuation of pretraining.
    """

    action_rng, env_rng, sampler_rng = jax.random.split(
        jax.random.PRNGKey(int(seed)), 3
    )
    num_envs = int(runner.current_goal.shape[0])
    workers = jnp.arange(num_envs, dtype=jnp.int32) % 512
    env_state = jax.vmap(reset_tick_claim_worker)(workers)
    return runner.replace(
        env_state=env_state,
        oracle_state=_broadcast_oracle(num_envs),
        env_keys=jax.random.split(env_rng, num_envs),
        sampler_keys=jax.random.split(sampler_rng, num_envs),
        rng=action_rng,
        goal_steps=jnp.zeros((num_envs,), dtype=jnp.int32),
        world_violation_count=jnp.zeros((num_envs,), dtype=jnp.int32),
        violation_grain_balance=jnp.zeros((num_envs,), dtype=jnp.int32),
        global_update=jnp.asarray(0, dtype=jnp.int32),
        env_steps=jnp.asarray(0, dtype=jnp.int32),
        rollout_cursor=jnp.asarray(0, dtype=jnp.int32),
    )


def reset_tick_claim_gc_optimizer(runner, config):
    """Replace Adam moments and the step counter. Parameters and RNG stay."""

    fresh = TrainState.create(
        apply_fn=TickClaimGCActorCritic(hidden_size=config.hidden_size).apply,
        params=runner.train_state.params,
        tx=_optimizer(config),
    )
    return runner.replace(train_state=fresh)


def command_deliver_3(runner):
    """Point every worker at deliver_3 without touching parameters or optimizer."""

    observations = jax.vmap(observe_tick_claim)(runner.env_state)
    achieved = jax.vmap(tick_claim_goal_vector)(observations)[
        :, DELIVER_3_GOAL_INDEX
    ]
    return runner.replace(
        current_goal=jnp.full(
            runner.current_goal.shape, DELIVER_3_GOAL_INDEX, dtype=jnp.int32
        ),
        command_active=jnp.logical_not(achieved),
        goal_steps=jnp.zeros_like(runner.goal_steps),
    )


def _tree_allclose(left, right, *, atol=0.0, rtol=0.0):
    return bool(
        jax.tree_util.tree_all(
            jax.tree.map(
                lambda a, b: jnp.allclose(
                    a, b, atol=atol, rtol=rtol
                ),
                left,
                right,
            )
        )
    )


def validate_tick_claim_gc_checkpoint_resume(directory, config):
    """Compare an identical next update across a full checkpoint round-trip."""

    network, runner = initialize_tick_claim_gc(config)
    update = jax.jit(make_tick_claim_gc_update(network, config))
    runner, _ = update(runner)
    jax.block_until_ready(runner.global_update)
    logits_before = tick_claim_gc_policy_logits(
        network, runner.train_state.params, runner
    )
    save_tick_claim_gc_checkpoint(directory, runner, config)
    restored = load_tick_claim_gc_checkpoint(directory, runner, config)
    logits_after = tick_claim_gc_policy_logits(
        network, restored.train_state.params, restored
    )
    direct_next, direct_metrics = update(runner)
    restored_next, restored_metrics = update(restored)
    return {
        "policy_logits_equal": bool(
            jnp.array_equal(logits_before, logits_after)
        ),
        "next_train_state_equal": _tree_allclose(
            direct_next, restored_next
        ),
        "next_metrics_equal": _tree_allclose(
            direct_metrics, restored_metrics
        ),
        "optimizer_step_restored": int(
            restored.train_state.step
        ) == int(runner.train_state.step),
        "rng_restored": bool(jnp.array_equal(restored.rng, runner.rng)),
        "sampler_restored": bool(
            jnp.array_equal(restored.sampler_keys, runner.sampler_keys)
            and jnp.array_equal(restored.seen_goals, runner.seen_goals)
        ),
        "environment_restored": _tree_allclose(
            restored.env_state, runner.env_state
        ),
        "schedule_restored": (
            int(restored.global_update) == int(runner.global_update)
            and int(restored.env_steps) == int(runner.env_steps)
            and int(restored.rollout_cursor) == int(runner.rollout_cursor)
        ),
    }


def _checkpoint_update_dir(destination, update):
    return Path(destination) / "checkpoints" / f"update_{update}"


def _load_update_metrics(destination, completed_updates):
    path = Path(destination) / "updates.json"
    if not path.is_file():
        return []
    loaded = json.loads(path.read_text(encoding="utf-8"))
    prefix = [
        item
        for item in loaded
        if isinstance(item, dict) and int(item.get("update", -1)) <= completed_updates
    ]
    prefix.sort(key=lambda item: int(item["update"]))
    expected = list(range(1, completed_updates + 1))
    if [int(item["update"]) for item in prefix] != expected:
        return []
    return prefix


def _write_update_metrics(destination, update_metrics):
    (Path(destination) / "updates.json").write_text(
        json.dumps(update_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def cell_checkpoints_complete(destination, config):
    """True when the declared budget and every scheduled checkpoint exist."""

    destination = Path(destination)
    summary_path = destination / "summary.json"
    final_dir = destination / "checkpoint_final"
    final_meta = final_dir / "metadata.json"
    if not summary_path.is_file() or not checkpoint_files_present(final_dir):
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    final = json.loads(final_meta.read_text(encoding="utf-8"))
    if int(summary.get("updates", -1)) != config.num_updates:
        return False
    if summary.get("goal_mode") != config.goal_mode:
        return False
    if summary.get("variant") != config.variant:
        return False
    if int(summary.get("seed", -1)) != config.seed:
        return False
    if int(final.get("global_update", -1)) != config.num_updates:
        return False
    if int(summary.get("transitions", -1)) != config.transitions:
        return False
    final_config = json.loads(
        (final_dir / "config.json").read_text(encoding="utf-8")
    )
    if not _recorded_config_matches(final_config, config):
        return False
    for update in config.checkpoint_updates:
        checkpoint = _checkpoint_update_dir(destination, update)
        if not checkpoint_files_present(checkpoint):
            return False
        meta = json.loads(
            (checkpoint / "metadata.json").read_text(encoding="utf-8")
        )
        saved_config = json.loads(
            (checkpoint / "config.json").read_text(encoding="utf-8")
        )
        if (
            int(meta.get("global_update", -1)) != update
            or int(meta.get("environment_steps", -1))
            != update * config.batch_size
            or saved_config != expected_config
        ):
            return False
    return True


def _latest_checkpoint_update(destination, config):
    best = None
    root = Path(destination) / "checkpoints"
    if not root.is_dir():
        return None
    for path in root.glob("update_*"):
        suffix = path.name[len("update_") :]
        if not suffix.isdigit():
            continue
        update = int(suffix)
        if update > config.num_updates:
            continue
        if not checkpoint_files_present(path):
            continue
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if int(meta.get("global_update", -1)) != update:
            continue
        if best is None or update > best:
            best = update
    return best


def _save_scheduled_checkpoint(destination, update, runner, config):
    path = save_tick_claim_gc_checkpoint(
        _checkpoint_update_dir(destination, update), runner, config
    )
    print(
        json.dumps(
            {
                "event": "checkpoint",
                "update": int(update),
                "environment_steps": int(runner.env_steps),
                "path": str(path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return path


def _write_started_manifest(destination, git_sha, config):
    (destination / "git_sha.txt").write_text(git_sha + "\n", encoding="utf-8")
    (destination / "working_tree.patch").write_text(_git_diff(), encoding="utf-8")
    (destination / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "hackrl_gc_calibration_cell_v1",
                "status": "started",
                "git_sha": git_sha,
                "config": tick_claim_gc_config_payload(config),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_tick_claim_gc_pilot(config, log_dir):
    """Run one bounded GC-PPO pilot cell and persist full-state artifacts."""

    config.validate()
    destination = Path(log_dir)
    if cell_checkpoints_complete(destination, config):
        return json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    latest = _latest_checkpoint_update(destination, config)
    if latest is None and any(destination.iterdir()):
        allowed = {"git_sha.txt", "working_tree.patch", "run_manifest.json"}
        existing = {path.name for path in destination.iterdir()}
        if not existing <= allowed:
            raise FileExistsError(f"run directory is not empty: {destination}")
    git_sha = _git_sha()
    if latest is None:
        _write_started_manifest(destination, git_sha, config)
    else:
        (destination / "resume_git_sha.txt").write_text(
            git_sha + "\n", encoding="utf-8"
        )
    network, runner = initialize_tick_claim_gc(config)
    update = jax.jit(make_tick_claim_gc_update(network, config))
    save_at = set(config.checkpoint_updates)
    if latest is None:
        completed = 0
        update_metrics = []
        if 0 in save_at:
            _save_scheduled_checkpoint(destination, 0, runner, config)
    else:
        runner = load_tick_claim_gc_checkpoint(
            _checkpoint_update_dir(destination, latest), runner, config
        )
        completed = latest
        update_metrics = _load_update_metrics(destination, completed)
    parameter_count = tick_claim_gc_parameter_count(runner.train_state.params)
    compile_seconds = None
    started = time.perf_counter()
    first_local_update = True
    for update_index in range(completed, config.num_updates):
        update_started = time.perf_counter()
        runner, metrics = update(runner)
        jax.block_until_ready(runner.train_state.params)
        elapsed = time.perf_counter() - update_started
        if first_local_update:
            compile_seconds = elapsed
            first_local_update = False
        host = {
            key: np.asarray(jax.device_get(value)).tolist()
            for key, value in metrics.items()
        }
        host["update"] = update_index + 1
        update_metrics.append(host)
        finished = update_index + 1
        if finished in save_at:
            _write_update_metrics(destination, update_metrics)
            _save_scheduled_checkpoint(destination, finished, runner, config)
        print(
            f"[update] {finished}/{config.num_updates} seconds={elapsed:.2f}",
            flush=True,
        )
    training_seconds = time.perf_counter() - started

    state_before_eval = serialization.to_bytes(runner)
    mode = evaluate_tick_claim_gc_frozen(
        network,
        runner.train_state.params,
        variant=TickClaimVariant(config.variant),
        split=TickClaimSplit(config.evaluation_split),
        stochastic=False,
        repeats_per_state=config.mode_repeats_per_state,
        seed_base=20000,
        learner_seed=config.seed,
    )
    sample = evaluate_tick_claim_gc_frozen(
        network,
        runner.train_state.params,
        variant=TickClaimVariant(config.variant),
        split=TickClaimSplit(config.evaluation_split),
        stochastic=True,
        repeats_per_state=config.sample_repeats_per_state,
        seed_base=20000,
        learner_seed=config.seed,
    )
    eval_state_immutable = (
        state_before_eval == serialization.to_bytes(runner)
    )
    checkpoint_dir = save_tick_claim_gc_checkpoint(
        destination / "checkpoint_final", runner, config
    )
    maximum = np.iinfo(np.int32).max
    first_metric_names = (
        "first_opportunity_transition",
        "first_reservation_transition",
        "first_violation_transition",
        "first_repeated_violation_transition",
        "first_violation_delivery_transition",
        "first_success_transition",
    )
    first_metrics = {}
    for name in first_metric_names:
        values = [
            item[name]
            for item in update_metrics
            if item[name] < maximum
        ]
        first_metrics[name] = min(values) if values else None
    steady_seconds = training_seconds - (compile_seconds or 0.0)
    summary = {
        "schema_version": "hackrl_gc_pilot_result_v1",
        "variant": config.variant,
        "seed": config.seed,
        "goal_mode": config.goal_mode,
        "transitions": config.transitions,
        "updates": config.num_updates,
        "resumed_from_update": completed,
        "checkpoint_updates": list(config.checkpoint_updates),
        "parameter_count": parameter_count,
        "initial_commands": config.num_envs,
        "training_seconds": training_seconds,
        "compile_and_first_update_seconds": compile_seconds,
        "steady_state_transitions_per_second": (
            None
            if latest is not None or config.num_updates <= 1
            else (config.transitions - config.batch_size)
            / max(steady_seconds, 1e-9)
        ),
        **first_metrics,
        "total_goal_successes": int(
            sum(item["goal_successes"] for item in update_metrics)
        ),
        "total_violation_events": int(
            sum(item["violation_events"] for item in update_metrics)
        ),
        "total_repeated_violation_events": int(
            sum(
                item["repeated_violation_events"]
                for item in update_metrics
            )
        ),
        "total_opportunity_exposures": int(
            sum(
                item["opportunity_exposures"]
                for item in update_metrics
            )
        ),
        "total_reservation_creations": int(
            sum(
                item["reservation_creations"]
                for item in update_metrics
            )
        ),
        "total_violation_delivery_events": int(
            sum(
                item["violation_delivery_events"]
                for item in update_metrics
            )
        ),
        "total_violation_grain_delivered": int(
            sum(
                item["violation_grain_delivered"]
                for item in update_metrics
            )
        ),
        "mode_evaluation": mode,
        "sample_evaluation": sample,
        "evaluation_state_immutable": eval_state_immutable,
        "checkpoint": str(checkpoint_dir),
    }
    (destination / "config.json").write_text(
        json.dumps(
            tick_claim_gc_config_payload(config),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (destination / "updates.json").write_text(
        json.dumps(update_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "hackrl_gc_calibration_cell_v1",
                "status": "complete",
                "git_sha": git_sha,
                "config": tick_claim_gc_config_payload(config),
                "summary": "summary.json",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def run_tick_claim_gc_calibration_cell(config, log_dir):
    """Run one manifest-declared calibration cell.

    The historical ``pilot`` entry point remains as an implementation alias;
    this name is the public surface used by the six-cell calibration CLI.
    """

    return run_tick_claim_gc_pilot(config, log_dir)
