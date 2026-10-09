"""Goal-conditioned PPO and Dual adapter for SPATIAL-WALL-PASS.

Normal pretraining uses only public, non-defect goals. A wall crossing and a
delivery caused after such a crossing are logged separately. Frozen evaluation
runs the same policy in either kernel without auto-resetting it.
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

from hackrl.spatial_wall_pass import (
    DISCOUNT,
    GOAL_IDS,
    MAP_CHANNEL_NAMES,
    MODEL_FEATURE_NAMES,
    VIEW_COLUMNS,
    VIEW_ROWS,
    SpatialWallPassAction,
    SpatialWallPassSplit,
    SpatialWallPassStart,
    SpatialWallPassState,
    SpatialWallPassVariant,
    encode_spatial_wall_pass_observation,
    make_spatial_wall_pass_state,
    observe_spatial_wall_pass,
    reset_spatial_wall_pass_worker,
    spatial_wall_pass_goal_vector,
    spatial_wall_pass_step_with_transition,
    spatial_wall_pass_world_done,
)


DELIVERY_GOAL_INDEX = GOAL_IDS.index("delivery/count_ge_1")
MAP_FEATURE_COUNT = VIEW_ROWS * VIEW_COLUMNS * len(MAP_CHANNEL_NAMES)
NUM_GOALS = len(GOAL_IDS)
NUM_ACTIONS = len(SpatialWallPassAction)
PRETRAIN_UPDATES = 512
ADAPT_UPDATES = 4096
DEVELOPMENT_SEEDS = (130, 131)
SCIENCE_UPDATES = (0, 32, 128, 256, 512, 1024, 2048, 4096)


@dataclass(frozen=True)
class SpatialWallPassGCConfig:
    variant: str = SpatialWallPassVariant.FIXED.value
    seed: int = 0
    num_envs: int = 512
    num_steps: int = 64
    num_updates: int = 32
    update_epochs: int = 1
    minibatch_size: int = 1024
    policy_hidden_size: int = 512
    teacher_hidden_size: int = 512
    learning_rate: float = 2e-4
    gamma: float = DISCOUNT
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.005
    value_coefficient: float = 0.5
    max_grad_norm: float = 1.0
    goal_mode: str = "normal12"
    evaluation_split: str = SpatialWallPassSplit.VALIDATION.value
    mode_repeats_per_state: int = 1
    sample_repeats_per_state: int = 64
    checkpoint_updates: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "checkpoint_updates", tuple(self.checkpoint_updates))

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def num_minibatches(self) -> int:
        return self.batch_size // self.minibatch_size

    def validate(self) -> None:
        SpatialWallPassVariant(self.variant)
        SpatialWallPassSplit(self.evaluation_split)
        if self.goal_mode not in {"delivery", "normal12"}:
            raise ValueError("goal_mode must be delivery or normal12")
        if self.batch_size % self.minibatch_size:
            raise ValueError("rollout batch must divide evenly into minibatches")
        if min(
            self.num_envs,
            self.num_steps,
            self.num_updates,
            self.update_epochs,
            self.minibatch_size,
            self.mode_repeats_per_state,
            self.sample_repeats_per_state,
        ) <= 0:
            raise ValueError("training and evaluation counts must be positive")
        if self.policy_hidden_size <= 0 or self.teacher_hidden_size <= 0:
            raise ValueError("policy and teacher widths must be positive")
        invalid_checkpoints = [
            update
            for update in self.checkpoint_updates
            if update < 0 or update > self.num_updates
        ]
        if invalid_checkpoints:
            raise ValueError(
                f"checkpoint updates outside 0..num_updates: {invalid_checkpoints}"
            )


class SpatialWallPassGCActorCritic(nn.Module):
    hidden_size: int = 512
    action_dim: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, map_channels, numeric_features, goal_one_hot):
        spatial = nn.Conv(
            features=32, kernel_size=(3, 3), padding="SAME",
            kernel_init=nn.initializers.lecun_normal(), bias_init=constant(0.0), name="map_conv",
        )(map_channels)
        spatial = nn.relu(spatial).reshape((spatial.shape[0], -1))
        shared = jnp.concatenate((spatial, numeric_features, goal_one_hot), axis=-1)
        shared = nn.relu(nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="shared_dense",
        )(shared))
        actor = shared
        for index in range(2):
            actor = nn.relu(nn.Dense(
                self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                name=f"actor_hidden_{index}",
            )(actor))
        logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor_output",
        )(actor)
        critic = shared
        for index in range(4):
            critic = nn.relu(nn.Dense(
                self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                name=f"critic_hidden_{index}",
            )(critic))
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_output")(critic)
        return distrax.Categorical(logits=logits), jax.nn.sigmoid(jnp.squeeze(value, axis=-1))


@struct.dataclass
class SpatialWallPassGCRunnerState:
    train_state: TrainState
    env_state: SpatialWallPassState
    current_goal: jax.Array
    command_active: jax.Array
    env_keys: jax.Array
    sampler_keys: jax.Array
    rng: jax.Array
    seen_goals: jax.Array
    goal_steps: jax.Array
    global_update: jax.Array
    env_steps: jax.Array
    rollout_cursor: jax.Array


@struct.dataclass
class SpatialWallPassGCTransition:
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
    goal_done: jax.Array
    world_done: jax.Array
    observed_goals: jax.Array
    terminal_goals: jax.Array
    dash_requested: jax.Array
    dash_succeeded: jax.Array
    clear_dash: jax.Array
    wall_pass: jax.Array
    item_collected_after_pass: jax.Array
    beneficial_delivery: jax.Array
    delivery: jax.Array
    absolute_transition: jax.Array


@struct.dataclass
class _Batch:
    transition: SpatialWallPassGCTransition
    advantages: jax.Array
    targets: jax.Array


def spatial_wall_pass_gc_inputs(observation, goal_index):
    encoded = encode_spatial_wall_pass_observation(observation)
    map_channels = encoded[:MAP_FEATURE_COUNT].reshape(VIEW_ROWS, VIEW_COLUMNS, len(MAP_CHANNEL_NAMES))
    numeric = encoded[MAP_FEATURE_COUNT:]
    goal = jax.nn.one_hot(goal_index, NUM_GOALS, dtype=jnp.float32)
    return map_channels, numeric, goal


def _batch_inputs(env_state, goals):
    return jax.vmap(spatial_wall_pass_gc_inputs)(jax.vmap(observe_spatial_wall_pass)(env_state), goals)


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
        already = achieved_goals[candidate]
        return loop_key, candidate, zero_steps + already.astype(jnp.int32), jnp.logical_not(already), attempts + 1

    key, candidate, zero_steps, accepted, _ = jax.lax.while_loop(
        condition, body,
        (key, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False), jnp.asarray(0, dtype=jnp.int32)),
    )
    return candidate, key, zero_steps, jnp.logical_and(valid, accepted)


def _goals(state):
    return spatial_wall_pass_goal_vector(observe_spatial_wall_pass(state))


def initialize_spatial_wall_pass_gc(config):
    config.validate()
    init_rng, action_rng, env_rng, sampler_rng = jax.random.split(jax.random.PRNGKey(config.seed), 4)
    del env_rng
    workers = jnp.arange(config.num_envs, dtype=jnp.int32)
    env_state = jax.vmap(reset_spatial_wall_pass_worker)(workers)
    goal_vectors = jax.vmap(_goals)(env_state)
    # All heads are preregistered normal predicates, not defect objectives.
    seen_goals = jnp.ones((NUM_GOALS,), dtype=jnp.bool_)
    false_seen = jnp.logical_and(seen_goals[None, :], jnp.logical_not(goal_vectors))
    if not bool(jnp.all(jnp.any(false_seen, axis=1))):
        raise ValueError("at least one worker has no false seen goal")
    sampler_keys = jax.random.split(sampler_rng, config.num_envs)
    if config.goal_mode == "delivery":
        current_goal = jnp.full((config.num_envs,), DELIVERY_GOAL_INDEX, dtype=jnp.int32)
    else:
        current_goal, sampler_keys, _, sampler_valid = jax.vmap(
            sample_false_seen_goal, in_axes=(0, None, 0)
        )(sampler_keys, seen_goals, goal_vectors)
        if not bool(jnp.all(sampler_valid)):
            raise ValueError("initial goal sampler could not find a false goal")
    model_inputs = _batch_inputs(env_state, current_goal)
    network = SpatialWallPassGCActorCritic(hidden_size=config.policy_hidden_size)
    parameters = network.init(init_rng, model_inputs[0][:1], model_inputs[1][:1], model_inputs[2][:1])
    if int(model_inputs[1].shape[-1]) != len(MODEL_FEATURE_NAMES):
        raise ValueError("numeric feature width does not match the schema")
    runner = SpatialWallPassGCRunnerState(
        train_state=TrainState.create(apply_fn=network.apply, params=parameters, tx=_optimizer(config)),
        env_state=env_state,
        current_goal=current_goal,
        command_active=jnp.ones((config.num_envs,), dtype=jnp.bool_),
        env_keys=jax.random.split(action_rng, config.num_envs),
        sampler_keys=sampler_keys,
        rng=action_rng,
        seen_goals=seen_goals,
        goal_steps=jnp.zeros((config.num_envs,), dtype=jnp.int32),
        global_update=jnp.asarray(0, dtype=jnp.int32),
        env_steps=jnp.asarray(0, dtype=jnp.int32),
        rollout_cursor=jnp.asarray(0, dtype=jnp.int32),
    )
    return network, runner


def step_spatial_wall_pass_gc_workers(runner, actions, config):
    variant = SpatialWallPassVariant(config.variant)
    before = runner.env_state
    valid = runner.command_active
    actions = jnp.where(valid, actions, jnp.asarray(int(SpatialWallPassAction.NOOP)))
    stepped, transitions = jax.vmap(
        lambda state, action: spatial_wall_pass_step_with_transition(
            state, action, variant
        )
    )(before, actions)
    terminal_goals = jax.vmap(_goals)(stepped)
    achieved = jnp.take_along_axis(
        terminal_goals, runner.current_goal[:, None], axis=1
    )[:, 0]
    goal_done = jnp.logical_and(valid, achieved)
    world_done = jax.vmap(spatial_wall_pass_world_done)(stepped)
    done = jnp.logical_or(
        jnp.logical_or(goal_done, world_done), jnp.logical_not(valid)
    )
    reset_states = jax.vmap(reset_spatial_wall_pass_worker)(
        jnp.arange(config.num_envs, dtype=jnp.int32)
    )
    reset_goals = jax.vmap(_goals)(reset_states)
    env_state = _tree_where(world_done, reset_states, stepped)
    observed = jnp.logical_or(
        terminal_goals, jnp.where(world_done[:, None], reset_goals, False)
    )
    switch = jnp.logical_or(goal_done, world_done)
    if config.goal_mode == "normal12":
        command_goals = jnp.where(world_done[:, None], reset_goals, terminal_goals)
        sampled_goal, sampled_keys, _, sampler_valid = jax.vmap(
            sample_false_seen_goal, in_axes=(0, None, 0)
        )(runner.sampler_keys, runner.seen_goals, command_goals)
        current_goal = jnp.where(switch, sampled_goal, runner.current_goal)
        sampler_keys = jnp.where(switch[:, None], sampled_keys, runner.sampler_keys)
        command_active = jnp.ones_like(runner.command_active)
    else:
        current_goal = runner.current_goal
        sampler_keys = runner.sampler_keys
        command_active = jnp.where(
            world_done,
            jnp.logical_not(reset_goals[:, DELIVERY_GOAL_INDEX]),
            jnp.where(goal_done, False, runner.command_active),
        )
    goal_steps = jnp.where(
        switch,
        0,
        jnp.where(runner.command_active, runner.goal_steps + 1, runner.goal_steps),
    )
    next_runner = runner.replace(
        env_state=env_state,
        current_goal=current_goal,
        command_active=command_active,
        sampler_keys=sampler_keys,
        goal_steps=goal_steps,
        env_steps=runner.env_steps + config.num_envs,
    )
    item_collected_after_pass = jnp.logical_and(
        valid,
        jnp.logical_and(
            stepped.item_collected_after_wall_pass,
            jnp.logical_not(before.item_collected_after_wall_pass),
        ),
    )
    beneficial_delivery = jnp.logical_and(
        valid,
        jnp.logical_and(
            stepped.delivered_after_wall_pass,
            jnp.logical_not(before.delivered_after_wall_pass),
        ),
    )
    event = SpatialWallPassGCTransition(
        done=done,
        valid=valid,
        action=actions,
        value=jnp.zeros_like(done, dtype=jnp.float32),
        reward=goal_done.astype(jnp.float32),
        log_prob=jnp.zeros_like(done, dtype=jnp.float32),
        map_channels=jnp.zeros((1,), dtype=jnp.float32),
        numeric_features=jnp.zeros((1,), dtype=jnp.float32),
        goal_one_hot=jnp.zeros((NUM_GOALS,), dtype=jnp.float32),
        goal_index=runner.current_goal,
        goal_done=goal_done,
        world_done=world_done,
        observed_goals=observed,
        terminal_goals=terminal_goals,
        dash_requested=jnp.logical_and(valid, transitions.dash_requested),
        dash_succeeded=jnp.logical_and(valid, transitions.dash_succeeded),
        clear_dash=jnp.logical_and(
            valid,
            jnp.logical_and(
                transitions.dash_succeeded,
                jnp.logical_not(transitions.crossed_wall),
            ),
        ),
        wall_pass=jnp.logical_and(valid, transitions.crossed_wall),
        item_collected_after_pass=item_collected_after_pass,
        beneficial_delivery=beneficial_delivery,
        delivery=jnp.logical_and(valid, transitions.delivered),
        absolute_transition=jnp.zeros_like(runner.goal_steps),
    )
    return next_runner, event


def spatial_wall_pass_outcome(runner, actions, config):
    runner, event = step_spatial_wall_pass_gc_workers(runner, actions, config)
    return (
        runner,
        event.done,
        event.valid,
        event.reward,
        event.terminal_goals,
        event.world_done,
        event.goal_done,
        event.observed_goals,
    )


def _weighted_mean(value, mask):
    mask = mask.astype(jnp.float32)
    return jnp.sum(value * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def make_spatial_wall_pass_gc_update(network, config):
    def rollout_step(runner, _):
        inputs = _batch_inputs(runner.env_state, runner.current_goal)
        policy, value = network.apply(runner.train_state.params, *inputs)
        rng, action_rng = jax.random.split(runner.rng)
        actions = policy.sample(seed=action_rng)
        log_prob = policy.log_prob(actions)
        before_steps = runner.env_steps
        runner = runner.replace(rng=rng)
        runner, event = step_spatial_wall_pass_gc_workers(runner, actions, config)
        event = event.replace(
            action=actions, value=value, log_prob=log_prob,
            map_channels=inputs[0], numeric_features=inputs[1], goal_one_hot=inputs[2],
            absolute_transition=before_steps + jnp.arange(config.num_envs, dtype=jnp.int32),
        )
        return runner, event

    def update_minibatch(train_state, batch):
        def loss_fn(parameters):
            policy, value = network.apply(
                parameters, batch.transition.map_channels, batch.transition.numeric_features, batch.transition.goal_one_hot,
            )
            valid = batch.transition.valid.astype(jnp.float32)
            advantage_mean = _weighted_mean(batch.advantages, valid)
            advantage_variance = _weighted_mean(jnp.square(batch.advantages - advantage_mean), valid)
            advantages = (batch.advantages - advantage_mean) / jnp.sqrt(advantage_variance + 1e-8)
            ratio = jnp.exp(policy.log_prob(batch.transition.action) - batch.transition.log_prob)
            actor_loss = -_weighted_mean(
                jnp.minimum(
                    ratio * advantages,
                    jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * advantages,
                ),
                valid,
            )
            clipped_value = batch.transition.value + (value - batch.transition.value).clip(-config.clip_epsilon, config.clip_epsilon)
            value_loss = 0.5 * _weighted_mean(
                jnp.maximum(jnp.square(value - batch.targets), jnp.square(clipped_value - batch.targets)), valid,
            )
            entropy = _weighted_mean(policy.entropy(), valid)
            total = actor_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy
            return total, (value_loss, actor_loss, entropy)

        valid_count = jnp.sum(batch.transition.valid.astype(jnp.int32))

        def apply_update(state):
            (loss, auxiliary), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=gradients)
            value_loss, policy_loss, entropy = auxiliary
            return state, {
                "loss": loss, "value_loss": value_loss, "policy_loss": policy_loss, "entropy": entropy,
                "valid_count": valid_count,
            }

        return jax.lax.cond(
            valid_count > 0,
            apply_update,
            lambda state: (state, {"loss": 0.0, "value_loss": 0.0, "policy_loss": 0.0, "entropy": 0.0, "valid_count": valid_count}),
            train_state,
        )

    def update(runner):
        runner, trajectory = jax.lax.scan(rollout_step, runner, None, length=config.num_steps)

        def backward(carry, transition):
            gae, next_value = carry
            nonterminal = 1.0 - transition.done.astype(jnp.float32)
            delta = transition.reward + config.gamma * next_value * nonterminal - transition.value
            gae = (delta + config.gamma * config.gae_lambda * nonterminal * gae) * transition.valid.astype(jnp.float32)
            return (gae, transition.value), gae

        last_inputs = _batch_inputs(runner.env_state, runner.current_goal)
        _, last_value = network.apply(runner.train_state.params, *last_inputs)
        _, advantages = jax.lax.scan(
            backward, (jnp.zeros_like(last_value), last_value), trajectory, reverse=True,
        )
        targets = advantages + trajectory.value
        steps = config.batch_size // config.minibatch_size
        flat = jax.tree.map(
            lambda leaf: leaf.reshape((steps, config.minibatch_size) + leaf.shape[2:]), trajectory
        )
        flat_adv = advantages.reshape((steps, config.minibatch_size))
        flat_targets = targets.reshape((steps, config.minibatch_size))

        def epoch(train_state, _):
            def minibatch(state, piece):
                transition, advantage, target = piece
                state, metrics = update_minibatch(state, _Batch(transition, advantage, target))
                return state, metrics
            pieces = (flat, flat_adv, flat_targets)
            return jax.lax.scan(minibatch, train_state, pieces)

        train_state, metrics = jax.lax.scan(epoch, runner.train_state, None, length=config.update_epochs)
        runner = runner.replace(train_state=train_state, global_update=runner.global_update + 1)
        summary = jax.tree.map(lambda leaf: jnp.sum(leaf), metrics)
        summary["valid_transitions"] = summary["valid_count"]
        summary["dash_requested_count"] = jnp.sum(
            trajectory.dash_requested.astype(jnp.int32)
        )
        summary["dash_succeeded_count"] = jnp.sum(
            trajectory.dash_succeeded.astype(jnp.int32)
        )
        summary["clear_dash_count"] = jnp.sum(
            trajectory.clear_dash.astype(jnp.int32)
        )
        summary["wall_pass_count"] = jnp.sum(trajectory.wall_pass.astype(jnp.int32))
        summary["item_collected_after_pass_count"] = jnp.sum(
            trajectory.item_collected_after_pass.astype(jnp.int32)
        )
        summary["beneficial_delivery_count"] = jnp.sum(
            trajectory.beneficial_delivery.astype(jnp.int32)
        )
        summary["delivery_count"] = jnp.sum(trajectory.delivery.astype(jnp.int32))
        summary["goal_successes"] = jnp.sum(trajectory.reward)
        return runner, summary

    return update


def command_delivery(runner):
    achieved = jax.vmap(_goals)(runner.env_state)[:, DELIVERY_GOAL_INDEX]
    return runner.replace(
        current_goal=jnp.full(
            runner.current_goal.shape, DELIVERY_GOAL_INDEX, dtype=jnp.int32
        ),
        command_active=jnp.logical_not(achieved),
        goal_steps=jnp.zeros_like(runner.goal_steps),
    )


_BRANCH_LOCKED_FIELDS = (
    "seed",
    "num_envs",
    "num_steps",
    "update_epochs",
    "minibatch_size",
    "policy_hidden_size",
    "teacher_hidden_size",
    "learning_rate",
    "gamma",
    "gae_lambda",
    "clip_epsilon",
    "entropy_coefficient",
    "value_coefficient",
    "max_grad_norm",
    "evaluation_split",
    "mode_repeats_per_state",
    "sample_repeats_per_state",
)


def spatial_wall_pass_gc_parameter_count(parameters) -> int:
    return int(sum(np.asarray(value).size for value in jax.tree.leaves(parameters)))


def config_from_spatial_wall_pass_gc_payload(recorded):
    names = {item.name for item in fields(SpatialWallPassGCConfig)}
    data = {name: recorded[name] for name in names if name in recorded}
    if "checkpoint_updates" in data:
        data["checkpoint_updates"] = tuple(data["checkpoint_updates"])
    return SpatialWallPassGCConfig(**data)


def load_spatial_wall_pass_gc_history_branch(directory, template, branch_config):
    source = Path(directory)
    recorded = json.loads((source / "config.json").read_text(encoding="utf-8"))
    origin = config_from_spatial_wall_pass_gc_payload(recorded)
    if origin.goal_mode != "normal12" or branch_config.goal_mode != "delivery":
        raise ValueError("adaptation branches from normal12 to delivery")
    for name in _BRANCH_LOCKED_FIELDS:
        if getattr(origin, name) != getattr(branch_config, name):
            raise ValueError(f"branch changed locked field {name}")
    branch_config.validate()
    return serialization.from_bytes(template, (source / "state.msgpack").read_bytes())


def evaluate_spatial_wall_pass_gc_frozen(
    network,
    parameters,
    *,
    variant,
    stochastic,
    repeats_per_state,
    seed_base,
    learner_seed,
    split=SpatialWallPassSplit.VALIDATION.value,
    record_episodes=False,
    horizon=64,
):
    """Frozen natural-start delivery evaluation on one kernel."""

    variant = SpatialWallPassVariant(variant)
    split = SpatialWallPassSplit(split)
    layouts = jnp.arange(16, dtype=jnp.int32)
    base = jax.vmap(
        lambda index: make_spatial_wall_pass_state(
            index, split=split, start=SpatialWallPassStart.NATURAL
        )
    )(layouts)
    initial = jax.tree.map(
        lambda value: jnp.repeat(value, repeats_per_state, axis=0), base
    )
    layout_indices = jnp.repeat(layouts, repeats_per_state)
    repeat_indices = jnp.tile(
        jnp.arange(repeats_per_state, dtype=jnp.int32), 16
    )
    episode_count = int(layout_indices.shape[0])
    seeds = (
        int(seed_base)
        + 1000 * int(learner_seed)
        + repeats_per_state * layout_indices
        + repeat_indices
    )
    keys = jax.vmap(jax.random.PRNGKey)(seeds)

    def rollout(params):
        def eval_step(carry, step_index):
            (
                state,
                done,
                success,
                length,
                dash_seen,
                dash_count,
                clear_dash_seen,
                clear_dash_count,
                wall_pass_seen,
                wall_pass_count,
                first_wall_pass,
                beneficial_use,
                action_keys,
            ) = carry
            goals = jnp.full(
                (episode_count,), DELIVERY_GOAL_INDEX, dtype=jnp.int32
            )
            policy, _ = network.apply(params, *_batch_inputs(state, goals))
            split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(
                action_keys
            )
            if stochastic:
                actions = jax.vmap(
                    lambda logits, key: distrax.Categorical(
                        logits=logits
                    ).sample(seed=key)
                )(policy.logits, split_keys[:, 1])
            else:
                actions = jnp.argmax(policy.logits, axis=-1)
            active = jnp.logical_not(done)
            actions = jnp.where(
                active, actions, int(SpatialWallPassAction.NOOP)
            )
            stepped, transitions = jax.vmap(
                lambda item, action: spatial_wall_pass_step_with_transition(
                    item, action, variant
                )
            )(state, actions)
            achieved = jax.vmap(_goals)(stepped)[:, DELIVERY_GOAL_INDEX]
            world_done = jax.vmap(spatial_wall_pass_world_done)(stepped)
            just_success = jnp.logical_and(active, achieved)
            just_done = jnp.logical_and(
                active, jnp.logical_or(achieved, world_done)
            )
            dash = jnp.logical_and(active, transitions.dash_requested)
            clear_dash = jnp.logical_and(
                active,
                jnp.logical_and(
                    transitions.dash_succeeded,
                    jnp.logical_not(transitions.crossed_wall),
                ),
            )
            wall_pass = jnp.logical_and(active, transitions.crossed_wall)
            first_wall_pass = jnp.where(
                jnp.logical_and(first_wall_pass < 0, wall_pass),
                step_index + 1,
                first_wall_pass,
            )
            beneficial = jnp.logical_and(
                active,
                jnp.logical_and(
                    transitions.delivered, stepped.delivered_after_wall_pass
                ),
            )
            return (
                _tree_where(active, stepped, state),
                jnp.logical_or(done, just_done),
                jnp.logical_or(success, just_success),
                length + active.astype(jnp.int32),
                jnp.logical_or(dash_seen, dash),
                dash_count + dash.astype(jnp.int32),
                jnp.logical_or(clear_dash_seen, clear_dash),
                clear_dash_count + clear_dash.astype(jnp.int32),
                jnp.logical_or(wall_pass_seen, wall_pass),
                wall_pass_count + wall_pass.astype(jnp.int32),
                first_wall_pass,
                jnp.logical_or(beneficial_use, beneficial),
                split_keys[:, 0],
            ), None

        zeros_bool = jnp.zeros((episode_count,), dtype=jnp.bool_)
        zeros_int = jnp.zeros((episode_count,), dtype=jnp.int32)
        final, _ = jax.lax.scan(
            eval_step,
            (
                initial,
                zeros_bool,
                zeros_bool,
                zeros_int,
                zeros_bool,
                zeros_int,
                zeros_bool,
                zeros_int,
                zeros_bool,
                zeros_int,
                jnp.full((episode_count,), -1, dtype=jnp.int32),
                zeros_bool,
                keys,
            ),
            jnp.arange(horizon, dtype=jnp.int32),
        )
        return final

    final = jax.jit(rollout)(parameters)
    (
        _,
        _,
        success,
        length,
        dash_seen,
        dash_count,
        clear_dash_seen,
        clear_dash_count,
        wall_pass_seen,
        wall_pass_count,
        first_wall_pass,
        beneficial_use,
        _,
    ) = jax.device_get(final)
    success = np.asarray(success, dtype=bool)
    length = np.asarray(length, dtype=np.int32)
    dash_seen = np.asarray(dash_seen, dtype=bool)
    dash_count = np.asarray(dash_count, dtype=np.int32)
    clear_dash_seen = np.asarray(clear_dash_seen, dtype=bool)
    clear_dash_count = np.asarray(clear_dash_count, dtype=np.int32)
    wall_pass_seen = np.asarray(wall_pass_seen, dtype=bool)
    wall_pass_count = np.asarray(wall_pass_count, dtype=np.int32)
    first_wall_pass = np.asarray(first_wall_pass, dtype=np.int32)
    beneficial_use = np.asarray(beneficial_use, dtype=bool)
    returns = np.where(
        length >= 1,
        success.astype(np.float64) * (DISCOUNT ** (length - 1)),
        0.0,
    )
    hit_first = first_wall_pass[first_wall_pass >= 0]
    success_lengths = length[success]
    result = {
        "split": split.value,
        "variant": variant.value,
        "action_selection": "sample" if stochastic else "mode",
        "episodes": episode_count,
        "success_rate": float(np.mean(success)),
        "mean_length": float(np.mean(length)),
        "mean_success_length": (
            float(np.mean(success_lengths)) if success_lengths.size else None
        ),
        "mean_discounted_return": float(np.mean(returns)),
        "dash_use_rate": float(np.mean(dash_seen)),
        "mean_dash_count": float(np.mean(dash_count)),
        "clear_dash_rate": float(np.mean(clear_dash_seen)),
        "mean_clear_dash_count": float(np.mean(clear_dash_count)),
        "wall_pass_rate": float(np.mean(wall_pass_seen)),
        "repeated_wall_pass_rate": float(np.mean(wall_pass_count >= 2)),
        "beneficial_use_rate": float(np.mean(beneficial_use)),
        "wall_pass_without_benefit_rate": float(
            np.mean(np.logical_and(wall_pass_seen, np.logical_not(beneficial_use)))
        ),
        "mean_first_wall_pass_step": (
            float(np.mean(hit_first)) if hit_first.size else None
        ),
        "episode_records": [],
    }
    if record_episodes:
        result["episode_records"] = [
            {
                "layout_index": int(layout_indices[index]),
                "repeat": int(repeat_indices[index]),
                "success": bool(success[index]),
                "length": int(length[index]),
                "discounted_return": float(returns[index]),
                "dash_used": bool(dash_seen[index]),
                "dash_count": int(dash_count[index]),
                "clear_dash": bool(clear_dash_seen[index]),
                "clear_dash_count": int(clear_dash_count[index]),
                "wall_pass": bool(wall_pass_seen[index]),
                "wall_pass_count": int(wall_pass_count[index]),
                "first_wall_pass_step": int(first_wall_pass[index]),
                "beneficial_use": bool(beneficial_use[index]),
            }
            for index in range(episode_count)
        ]
    return result


def evaluate_spatial_wall_pass_frozen(
    network,
    params,
    config,
    *,
    adaptation_update,
    stochastic=False,
    repeats_per_state=None,
    record_episodes=False,
):
    repeats = (
        config.sample_repeats_per_state
        if stochastic
        else config.mode_repeats_per_state
    )
    if repeats_per_state is not None:
        repeats = int(repeats_per_state)
    result = evaluate_spatial_wall_pass_gc_frozen(
        network,
        params,
        variant=config.variant,
        stochastic=stochastic,
        repeats_per_state=repeats,
        seed_base=61000 if stochastic else 60000,
        learner_seed=config.seed,
        split=config.evaluation_split,
        record_episodes=record_episodes,
    )
    result["adaptation_update"] = int(adaptation_update)
    return result


def adaptation_zero_record(network, params, config, **kwargs):
    return evaluate_spatial_wall_pass_frozen(
        network, params, config, adaptation_update=0, **kwargs
    )


def spatial_wall_pass_gc_config_payload(config):
    payload = asdict(config)
    payload["checkpoint_updates"] = list(config.checkpoint_updates)
    return payload


def save_spatial_wall_pass_gc_checkpoint(directory, runner, config):
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "state.msgpack").write_bytes(serialization.to_bytes(runner))
    (destination / "config.json").write_text(
        json.dumps(spatial_wall_pass_gc_config_payload(config), indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (destination / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "hackrl_spatial_wall_pass_gc_checkpoint_v1",
                "global_update": int(runner.global_update),
                "environment_steps": int(runner.env_steps),
                "goal_schema": list(GOAL_IDS),
            },
            indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return destination


def load_spatial_wall_pass_gc_checkpoint(directory, template, config):
    source = Path(directory)
    recorded = json.loads((source / "config.json").read_text(encoding="utf-8"))
    payload = spatial_wall_pass_gc_config_payload(config)
    if any(payload.get(key) != value for key, value in recorded.items()):
        raise ValueError("checkpoint config does not match requested config")
    return serialization.from_bytes(template, (source / "state.msgpack").read_bytes())


def compare_contract():
    """Frozen development contract. Calling this function does not launch it."""

    return {
        "launch": False,
        "methods": ("gc", "dual"),
        "development_seeds": DEVELOPMENT_SEEDS,
        "pretrain_updates": PRETRAIN_UPDATES,
        "adapt_updates": ADAPT_UPDATES,
        "science_updates": SCIENCE_UPDATES,
        "goal_set": "normal12",
        "adaptation_command": "delivery/count_ge_1",
        "adaptation_zero_is_control": True,
        "primary_metrics": (
            "success_rate",
            "mean_success_length",
            "mean_discounted_return",
        ),
        "secondary_metrics": (
            "wall_pass_rate",
            "beneficial_use_rate",
        ),
    }
