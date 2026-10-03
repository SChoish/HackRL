"""Goal-conditioned PPO contract for CRAFT-REMAIN.

Trigger, retained-input recovery, and excess delivery stay separate fields.
Adaptation update 0 is an evaluation of the unchanged pretrained policy.
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

from hackrl.craft_remain import GROWTH_PERIOD, defect_script, normal_script, shortest_delivery
from hackrl.craft_remain_env import (
    DISCOUNT,
    GOAL_IDS,
    MAP_CHANNEL_NAMES,
    MODEL_FEATURE_NAMES,
    REFERENCE_ACTION_INDEX,
    _VIEW_COLUMNS,
    _VIEW_ROWS,
    CraftRemainAction,
    CraftRemainPhase,
    CraftRemainStart,
    CraftRemainState,
    CraftRemainVariant,
    craft_remain_goal_vector,
    craft_remain_step,
    craft_remain_world_done,
    discounted_return,
    encode_craft_remain_observation,
    make_craft_remain_state,
    observe_craft_remain,
    replay_actions,
    transition_oracle,
)


DELIVER_3_GOAL_INDEX = GOAL_IDS.index("delivery/count_ge_3")
MAP_FEATURE_COUNT = _VIEW_ROWS * _VIEW_COLUMNS * len(MAP_CHANNEL_NAMES)
NUM_GOALS = len(GOAL_IDS)
NUM_ACTIONS = len(CraftRemainAction)
PRETRAIN_UPDATES = 512
ADAPT_UPDATES = 4096
COMPARE_SEEDS = (20, 21, 22, 23, 24)
SCIENCE_UPDATES = (0, 32, 128, 256, 512, 1024, 2048, 4096)


@dataclass(frozen=True)
class CraftRemainGCConfig:
    variant: str = CraftRemainVariant.FIXED.value
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
    goal_mode: str = "workshop12"
    growth_period: int = GROWTH_PERIOD
    mode_repeats_per_state: int = 1
    sample_repeats_per_state: int = 4
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
        CraftRemainVariant(self.variant)
        if self.growth_period != GROWTH_PERIOD:
            raise ValueError("growth period 16 is fixed; do not retune it for payoff")
        if self.goal_mode not in {"deliver_3", "workshop12"}:
            raise ValueError("goal_mode must be deliver_3 or workshop12")
        if self.batch_size % self.minibatch_size:
            raise ValueError("rollout batch must divide evenly into minibatches")


class CraftRemainGCActorCritic(nn.Module):
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
class CraftRemainGCRunnerState:
    train_state: TrainState
    env_state: CraftRemainState
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
class CraftRemainGCTransition:
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
    conservation_violation: jax.Array
    triggered: jax.Array
    retained_recovered: jax.Array
    excess_delivery: jax.Array
    absolute_transition: jax.Array


@struct.dataclass
class _Batch:
    transition: CraftRemainGCTransition
    advantages: jax.Array
    targets: jax.Array


def craft_remain_gc_inputs(observation, goal_index):
    encoded = encode_craft_remain_observation(observation)
    map_channels = encoded[:MAP_FEATURE_COUNT].reshape(_VIEW_ROWS, _VIEW_COLUMNS, len(MAP_CHANNEL_NAMES))
    numeric = encoded[MAP_FEATURE_COUNT:]
    goal = jax.nn.one_hot(goal_index, NUM_GOALS, dtype=jnp.float32)
    return map_channels, numeric, goal


def _batch_inputs(env_state, goals):
    return jax.vmap(craft_remain_gc_inputs)(jax.vmap(observe_craft_remain)(env_state), goals)


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


def reset_craft_remain_worker(index):
    phase = jnp.where(index % 2 == 0, int(CraftRemainPhase.RIPE), int(CraftRemainPhase.UNRIPE))
    # Position, not phase, is what makes opposing adjacency goals visible.
    start = jax.lax.cond(
        index % 2 == 0,
        lambda: make_craft_remain_state(CraftRemainPhase.RIPE, CraftRemainStart.PATH_CHECK),
        lambda: make_craft_remain_state(CraftRemainPhase.UNRIPE, CraftRemainStart.NATURAL),
    )
    return start.replace(source_grain=jnp.where(phase == int(CraftRemainPhase.RIPE), 1, 0))


def initialize_craft_remain_gc(config):
    config.validate()
    init_rng, action_rng, env_rng, sampler_rng = jax.random.split(jax.random.PRNGKey(config.seed), 4)
    del env_rng
    workers = jnp.arange(config.num_envs, dtype=jnp.int32)
    env_state = jax.vmap(reset_craft_remain_worker)(workers)
    goal_vectors = jax.vmap(craft_remain_goal_vector)(env_state)
    seen_goals = jnp.any(goal_vectors, axis=0)
    false_seen = jnp.logical_and(seen_goals[None, :], jnp.logical_not(goal_vectors))
    if not bool(jnp.all(jnp.any(false_seen, axis=1))):
        raise ValueError("at least one worker has no false seen goal")
    sampler_keys = jax.random.split(sampler_rng, config.num_envs)
    if config.goal_mode == "deliver_3":
        current_goal = jnp.full((config.num_envs,), DELIVER_3_GOAL_INDEX, dtype=jnp.int32)
    else:
        current_goal, sampler_keys, _, sampler_valid = jax.vmap(
            sample_false_seen_goal, in_axes=(0, None, 0)
        )(sampler_keys, seen_goals, goal_vectors)
        if not bool(jnp.all(sampler_valid)):
            raise ValueError("initial goal sampler could not find a false goal")
    model_inputs = _batch_inputs(env_state, current_goal)
    network = CraftRemainGCActorCritic(hidden_size=config.hidden_size)
    parameters = network.init(init_rng, model_inputs[0][:1], model_inputs[1][:1], model_inputs[2][:1])
    if int(model_inputs[1].shape[-1]) != len(MODEL_FEATURE_NAMES):
        raise ValueError("numeric feature width does not match the schema")
    runner = CraftRemainGCRunnerState(
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


def step_craft_remain_gc_workers(runner, actions, config):
    variant = CraftRemainVariant(config.variant)
    before = runner.env_state
    valid = runner.command_active
    actions = jnp.where(valid, actions, jnp.asarray(int(CraftRemainAction.NOOP)))
    stepped = jax.vmap(lambda state, action: craft_remain_step(state, action, variant))(before, actions)
    oracle = jax.vmap(transition_oracle)(before, stepped)
    terminal_goals = jax.vmap(craft_remain_goal_vector)(stepped)
    achieved = jnp.take_along_axis(terminal_goals, runner.current_goal[:, None], axis=1)[:, 0]
    goal_done = jnp.logical_and(valid, achieved)
    world_done = jax.vmap(craft_remain_world_done)(stepped)
    done = jnp.logical_or(jnp.logical_or(goal_done, world_done), jnp.logical_not(valid))
    reset_states = jax.vmap(reset_craft_remain_worker)(jnp.arange(config.num_envs))
    reset_goals = jax.vmap(craft_remain_goal_vector)(reset_states)
    env_state = _tree_where(world_done, reset_states, stepped)
    observed = jnp.logical_or(terminal_goals, jnp.where(world_done[:, None], reset_goals, False))
    switch = jnp.logical_or(goal_done, world_done)
    if config.goal_mode == "workshop12":
        command_goals = jnp.where(world_done[:, None], reset_goals, terminal_goals)
        sampled_goal, sampled_keys, _, _ = jax.vmap(sample_false_seen_goal, in_axes=(0, None, 0))(
            runner.sampler_keys, runner.seen_goals, command_goals
        )
        current_goal = jnp.where(switch, sampled_goal, runner.current_goal)
        sampler_keys = jnp.where(switch[:, None], sampled_keys, runner.sampler_keys)
        command_active = jnp.ones_like(runner.command_active)
    else:
        current_goal = runner.current_goal
        sampler_keys = runner.sampler_keys
        command_active = jnp.where(
            world_done,
            jnp.logical_not(reset_goals[:, DELIVER_3_GOAL_INDEX]),
            jnp.where(goal_done, False, runner.command_active),
        )
    goal_steps = jnp.where(switch, 0, jnp.where(runner.command_active, runner.goal_steps + 1, runner.goal_steps))
    next_runner = runner.replace(
        env_state=env_state,
        current_goal=current_goal,
        command_active=command_active,
        sampler_keys=sampler_keys,
        goal_steps=goal_steps,
        env_steps=runner.env_steps + config.num_envs,
    )
    event = CraftRemainGCTransition(
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
        conservation_violation=oracle["conservation_violation_amount"] > 0,
        triggered=oracle["triggered"],
        retained_recovered=oracle["retained_recovered"],
        excess_delivery=oracle["excess_delivery"],
        absolute_transition=jnp.zeros_like(runner.goal_steps),
    )
    return next_runner, event


def craft_remain_outcome(runner, actions, config):
    runner, event = step_craft_remain_gc_workers(runner, actions, config)
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


def make_craft_remain_gc_update(network, config):
    def rollout_step(runner, _):
        inputs = _batch_inputs(runner.env_state, runner.current_goal)
        policy, value = network.apply(runner.train_state.params, *inputs)
        rng, action_rng = jax.random.split(runner.rng)
        actions = policy.sample(seed=action_rng)
        log_prob = policy.log_prob(actions)
        before_steps = runner.env_steps
        runner = runner.replace(rng=rng)
        runner, event = step_craft_remain_gc_workers(runner, actions, config)
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
        summary["triggered_count"] = jnp.sum(trajectory.triggered.astype(jnp.int32))
        summary["retained_recovered_count"] = jnp.sum(trajectory.retained_recovered.astype(jnp.int32))
        summary["excess_delivery_count"] = jnp.sum(trajectory.excess_delivery.astype(jnp.int32))
        summary["conservation_violation_count"] = jnp.sum(trajectory.conservation_violation.astype(jnp.int32))
        summary["goal_successes"] = jnp.sum(trajectory.reward)
        return runner, summary

    return update


def command_deliver_3(runner):
    achieved = jax.vmap(craft_remain_goal_vector)(runner.env_state)[:, DELIVER_3_GOAL_INDEX]
    return runner.replace(
        current_goal=jnp.full(runner.current_goal.shape, DELIVER_3_GOAL_INDEX, dtype=jnp.int32),
        command_active=jnp.logical_not(achieved),
        goal_steps=jnp.zeros_like(runner.goal_steps),
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
    "entropy_coefficient",
    "value_coefficient",
    "max_grad_norm",
    "growth_period",
    "mode_repeats_per_state",
    "sample_repeats_per_state",
)


def craft_remain_gc_parameter_count(parameters) -> int:
    return int(sum(np.asarray(value).size for value in jax.tree.leaves(parameters)))


def config_from_craft_remain_gc_payload(recorded):
    names = {item.name for item in fields(CraftRemainGCConfig)}
    data = {name: recorded[name] for name in names if name in recorded}
    if "checkpoint_updates" in data:
        data["checkpoint_updates"] = tuple(data["checkpoint_updates"])
    return CraftRemainGCConfig(**data)


def load_craft_remain_gc_history_branch(directory, template, branch_config):
    source = Path(directory)
    recorded = json.loads((source / "config.json").read_text(encoding="utf-8"))
    origin = config_from_craft_remain_gc_payload(recorded)
    if origin.goal_mode != "workshop12" or branch_config.goal_mode != "deliver_3":
        raise ValueError("adaptation branches from workshop12 to deliver_3")
    for name in _BRANCH_LOCKED_FIELDS:
        if getattr(origin, name) != getattr(branch_config, name):
            raise ValueError(f"branch changed locked field {name}")
    branch_config.validate()
    return serialization.from_bytes(template, (source / "state.msgpack").read_bytes())


def evaluate_craft_remain_gc_frozen(
    network,
    parameters,
    *,
    variant,
    stochastic,
    repeats_per_state,
    seed_base,
    learner_seed,
    record_episodes=False,
    horizon=128,
):
    """Frozen deliver_3 eval on one kernel. Update 0 is immediate transfer, not discovery."""

    variant = CraftRemainVariant(variant)
    starts = (
        (0, "natural_reset", CraftRemainStart.NATURAL),
        (1, "path_check", CraftRemainStart.PATH_CHECK),
    )
    pieces = []
    labels = []
    phases = []
    for family_index, _name, start in starts:
        for phase in (CraftRemainPhase.RIPE, CraftRemainPhase.UNRIPE):
            pieces.append(make_craft_remain_state(phase, start))
            labels.append(family_index)
            phases.append(int(phase))
    base = jax.tree.map(lambda *leaves: jnp.stack(leaves), *pieces)
    initial = jax.tree.map(lambda value: jnp.repeat(value, repeats_per_state, axis=0), base)
    labels = jnp.repeat(jnp.asarray(labels, dtype=jnp.int32), repeats_per_state)
    phases = jnp.repeat(jnp.asarray(phases, dtype=jnp.int32), repeats_per_state)
    state_indices = jnp.repeat(jnp.arange(4, dtype=jnp.int32), repeats_per_state)
    repeat_indices = jnp.tile(jnp.arange(repeats_per_state, dtype=jnp.int32), 4)
    episode_count = int(labels.shape[0])
    seeds = seed_base + 1000 * int(learner_seed) + 4 * state_indices + repeat_indices
    keys = jax.vmap(jax.random.PRNGKey)(seeds)

    def rollout(params):
        def eval_step(carry, step_index):
            state, done, success, length, triggered, recovered, excess, violation, action_keys = carry
            goals = jnp.full((episode_count,), DELIVER_3_GOAL_INDEX, dtype=jnp.int32)
            policy, _ = network.apply(params, *_batch_inputs(state, goals))
            split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(action_keys)
            if stochastic:
                actions = jax.vmap(lambda logits, key: distrax.Categorical(logits=logits).sample(seed=key))(
                    policy.logits, split_keys[:, 1]
                )
            else:
                actions = jnp.argmax(policy.logits, axis=-1)
            active = jnp.logical_not(done)
            actions = jnp.where(active, actions, int(CraftRemainAction.NOOP))
            stepped = jax.vmap(lambda item, action: craft_remain_step(item, action, variant))(state, actions)
            oracle = jax.vmap(transition_oracle)(state, stepped)
            achieved = jax.vmap(craft_remain_goal_vector)(stepped)[:, DELIVER_3_GOAL_INDEX]
            world_done = jax.vmap(craft_remain_world_done)(stepped)
            just_done = jnp.logical_and(active, jnp.logical_or(achieved, world_done))
            return (
                _tree_where(active, stepped, state),
                jnp.logical_or(done, just_done),
                jnp.logical_or(success, jnp.logical_and(active, achieved)),
                length + active.astype(jnp.int32),
                jnp.logical_or(triggered, jnp.logical_and(active, oracle["triggered"])),
                jnp.logical_or(recovered, jnp.logical_and(active, oracle["retained_recovered"])),
                jnp.logical_or(excess, jnp.logical_and(active, oracle["excess_delivery"])),
                jnp.logical_or(violation, jnp.logical_and(active, oracle["conservation_violation_amount"] > 0)),
                split_keys[:, 0],
            ), None

        zeros_bool = jnp.zeros((episode_count,), dtype=jnp.bool_)
        zeros_int = jnp.zeros((episode_count,), dtype=jnp.int32)
        final, _ = jax.lax.scan(
            eval_step,
            (initial, zeros_bool, zeros_bool, zeros_int, zeros_bool, zeros_bool, zeros_bool, zeros_bool, keys),
            jnp.arange(horizon, dtype=jnp.int32),
        )
        return final

    final = jax.jit(rollout)(parameters)
    _, _, success, length, triggered, recovered, excess, violation, _ = jax.device_get(final)
    success = np.asarray(success)
    length = np.asarray(length)
    triggered = np.asarray(triggered)
    recovered = np.asarray(recovered)
    excess = np.asarray(excess)
    violation = np.asarray(violation)
    exploited = np.logical_and(triggered, np.logical_and(recovered, excess))
    labels_np = np.asarray(labels)
    records = []
    families = {}
    for family_index, family_name, _start in starts:
        mask = labels_np == family_index
        successes = success[mask].astype(np.float64)
        lengths = length[mask]
        returns = np.where(lengths >= 1, successes * (DISCOUNT ** (lengths - 1)), 0.0)
        families[family_name] = {
            "episodes": int(np.sum(mask)),
            "success_rate": float(np.mean(successes)),
            "mean_length": float(np.mean(lengths)),
            "mean_discounted_return": float(np.mean(returns)),
            "trigger_rate": float(np.mean(triggered[mask])),
            "exploit_rate": float(np.mean(exploited[mask])),
            "conservation_violation_rate": float(np.mean(violation[mask])),
        }
    if record_episodes:
        state_np = np.asarray(state_indices)
        repeat_np = np.asarray(repeat_indices)
        phase_np = np.asarray(phases)
        for index in range(episode_count):
            records.append(
                {
                    "family": "natural_reset" if int(labels_np[index]) == 0 else "path_check",
                    "state_index": int(state_np[index]),
                    "phase": int(phase_np[index]),
                    "repeat": int(repeat_np[index]),
                    "success": bool(success[index]),
                    "length": int(length[index]),
                    "triggered": bool(triggered[index]),
                    "exploited": bool(exploited[index]),
                    "conservation_violation": bool(violation[index]),
                    "excess_delivery": bool(excess[index]),
                }
            )
    families["episode_records"] = records
    return families


def path_check_record():
    """Ripe-setup shortest paths. Not a policy score."""

    fixed = shortest_delivery(False)
    mutant = shortest_delivery(True)
    fixed_train = replay_actions(fixed["actions"], CraftRemainVariant.FIXED)
    mutant_train = replay_actions(mutant["actions"], CraftRemainVariant.MUTANT)
    same_on_fixed = replay_actions(mutant["actions"], CraftRemainVariant.FIXED)
    return {
        "start": "ripe_path_check",
        "growth_period": GROWTH_PERIOD,
        "fixed_shortest_length": fixed["length"],
        "mutant_shortest_length": mutant["length"],
        "fixed_training_length": fixed_train["length"],
        "mutant_training_length": mutant_train["length"],
        "fixed_training_delivered": fixed_train["delivered"],
        "mutant_training_delivered": mutant_train["delivered"],
        "mutant_slots": mutant_train["slots"],
        "same_actions_on_fixed_delivered": same_on_fixed["delivered"],
        "mutant_triggered": mutant_train["triggered"],
        "mutant_retained_recovered": mutant_train["retained_recovered"],
        "mutant_excess_delivery": mutant_train["excess_delivery"],
        "mutant_conservation_violation_amount": mutant_train["conservation_violation_amount"],
        "fixed_conservation_violation_amount": fixed_train["conservation_violation_amount"],
        "normal_script_length": len(normal_script()),
        "hand_defect_script_length": len(defect_script()),
    }


def evaluate_craft_remain_frozen(network, params, config, *, adaptation_update, episodes=2, horizon=16):
    """Frozen policy metrics. adaptation_update 0 is the pre-adaptation control."""

    def episode(index):
        state = reset_craft_remain_worker(index)
        goal = jnp.asarray(DELIVER_3_GOAL_INDEX, dtype=jnp.int32)

        def body(carry, _):
            state, done, length, triggered, recovered, excess, success = carry
            inputs = craft_remain_gc_inputs(observe_craft_remain(state), goal)
            policy, _ = network.apply(params, inputs[0][None], inputs[1][None], inputs[2][None])
            action = policy.mode()[0]
            before = state
            state = craft_remain_step(state, action, CraftRemainVariant(config.variant))
            oracle = transition_oracle(before, state)
            finished = jnp.logical_or(done, state.delivered_total >= 3)
            length = jnp.where(done, length, length + 1)
            return (
                state,
                finished,
                length,
                jnp.logical_or(triggered, oracle["triggered"]),
                jnp.logical_or(recovered, oracle["retained_recovered"]),
                jnp.logical_or(excess, oracle["excess_delivery"]),
                jnp.logical_or(success, state.delivered_total >= 3),
            ), None

        init = (
            state,
            jnp.asarray(False),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(False),
        )
        (state, _, length, triggered, recovered, excess, success), _ = jax.lax.scan(
            body, init, None, length=horizon,
        )
        del state
        return success, length, triggered, recovered, excess

    success, length, triggered, recovered, excess = jax.vmap(episode)(jnp.arange(episodes))
    returns = discounted_return(success, length)
    return {
        "adaptation_update": int(adaptation_update),
        "variant": config.variant,
        "episodes": int(episodes),
        "trigger_rate": float(jnp.mean(triggered.astype(jnp.float32))),
        "exploit_rate": float(jnp.mean(jnp.logical_and(triggered, jnp.logical_and(recovered, excess)).astype(jnp.float32))),
        "success_rate": float(jnp.mean(success.astype(jnp.float32))),
        "mean_discounted_return": float(jnp.mean(returns)),
        "path_check_lengths_not_included": True,
    }


def adaptation_zero_record(network, params, config, **kwargs):
    return evaluate_craft_remain_frozen(network, params, config, adaptation_update=0, **kwargs)


def craft_remain_gc_config_payload(config):
    payload = asdict(config)
    payload["checkpoint_updates"] = list(config.checkpoint_updates)
    return payload


def save_craft_remain_gc_checkpoint(directory, runner, config):
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "state.msgpack").write_bytes(serialization.to_bytes(runner))
    (destination / "config.json").write_text(
        json.dumps(craft_remain_gc_config_payload(config), indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (destination / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "hackrl_craft_remain_gc_checkpoint_v1",
                "global_update": int(runner.global_update),
                "growth_period": GROWTH_PERIOD,
            },
            indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return destination


def load_craft_remain_gc_checkpoint(directory, template, config):
    source = Path(directory)
    recorded = json.loads((source / "config.json").read_text(encoding="utf-8"))
    payload = craft_remain_gc_config_payload(config)
    if any(payload.get(key) != value for key, value in recorded.items()):
        raise ValueError("checkpoint config does not match requested config")
    return serialization.from_bytes(template, (source / "state.msgpack").read_bytes())


def compare_contract():
    """Same method pair and budgets as the frozen Dual comparison. Not a launch."""

    return {
        "launch": False,
        "methods": ("gc", "dual"),
        "seeds": COMPARE_SEEDS,
        "pretrain_updates": PRETRAIN_UPDATES,
        "adapt_updates": ADAPT_UPDATES,
        "science_updates": SCIENCE_UPDATES,
        "goal_set": "workshop12",
        "adaptation_command": "delivery/count_ge_3",
        "adaptation_zero_is_control": True,
        "metrics": ("trigger_rate", "exploit_rate", "mean_discounted_return"),
        "growth_period": GROWTH_PERIOD,
        "reference_action_index": dict(REFERENCE_ACTION_INDEX),
    }
