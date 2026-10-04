"""Fixed-kernel PPO learnability gate for the mine expedition.

Training may reset into public curriculum stages, but evaluation always starts
from the natural initial state and runs for the full 256-tick world horizon.
The policy receives neither the kernel variant nor duplicate provenance.  This
module intentionally exposes no mutant-training option: the fixed normal task
must pass before exploit-learning comparisons are opened.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
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

from hackrl.mine_expedition import (
    ACTION_COUNT,
    WORLD_HORIZON,
    MineExpeditionAction,
    MineExpeditionState,
    MineExpeditionVariant,
)
from hackrl.mine_expedition_env import (
    MAP_CHANNEL_NAMES,
    NUMERIC_FEATURE_NAMES,
    MineExpeditionStart,
    observe_mine_expedition,
    reset_mine_expedition,
    step_mine_expedition_env,
)


TASK_DISCOUNT = 0.995


@dataclass(frozen=True)
class MineExpeditionPPOConfig:
    seed: int = 0
    num_envs: int = 128
    num_steps: int = 64
    num_updates: int = 2048
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
    training_start: str = MineExpeditionStart.CURRICULUM.value
    mode_eval_episodes: int = 1
    sample_eval_episodes: int = 128
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
        start = MineExpeditionStart(self.training_start)
        if self.num_envs <= 0 or self.num_steps <= 0 or self.num_updates <= 0:
            raise ValueError("num_envs, num_steps, and num_updates must be positive")
        if self.update_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("update_epochs and minibatch_size must be positive")
        if self.batch_size % self.minibatch_size:
            raise ValueError("rollout batch must divide evenly into minibatches")
        if self.hidden_size <= 0 or self.learning_rate <= 0:
            raise ValueError("hidden_size and learning_rate must be positive")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must lie in their probability ranges")
        if self.clip_epsilon <= 0 or self.entropy_coefficient < 0:
            raise ValueError("clip_epsilon must be positive and entropy non-negative")
        if self.value_coefficient < 0 or self.max_grad_norm <= 0:
            raise ValueError("value coefficient and max gradient norm are invalid")
        if self.mode_eval_episodes <= 0 or self.sample_eval_episodes <= 0:
            raise ValueError("evaluation episode counts must be positive")
        if start not in {MineExpeditionStart.CURRICULUM, MineExpeditionStart.NATURAL}:
            raise ValueError("training_start must be curriculum or natural")
        if any(update < 0 or update > self.num_updates for update in self.checkpoint_updates):
            raise ValueError("checkpoint updates must lie within the training run")


class MineExpeditionActorCritic(nn.Module):
    hidden_size: int = 512
    action_dim: int = ACTION_COUNT

    @nn.compact
    def __call__(self, map_channels, numeric_features):
        spatial = nn.Conv(
            features=16,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="SAME",
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=constant(0.0),
            name="map_conv_0",
        )(map_channels)
        spatial = nn.relu(spatial)
        spatial = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=constant(0.0),
            name="map_conv_1",
        )(spatial)
        spatial = nn.relu(spatial).reshape((spatial.shape[0], -1))
        shared = jnp.concatenate((spatial, numeric_features), axis=-1)
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
        return distrax.Categorical(logits=logits), jnp.squeeze(value, axis=-1)


@struct.dataclass
class MineExpeditionRunnerState:
    train_state: TrainState
    env_state: MineExpeditionState
    env_keys: jax.Array
    rng: jax.Array
    episode_returns: jax.Array
    episode_lengths: jax.Array
    global_update: jax.Array
    env_steps: jax.Array


class MineExpeditionTransition(struct.PyTreeNode):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    map_channels: jax.Array
    numeric_features: jax.Array
    success: jax.Array
    timeout: jax.Array
    crafted_pickaxe: jax.Array
    mined_target: jax.Array
    returned_target: jax.Array
    iron_increase: jax.Array
    indirect_use: jax.Array
    completed_return: jax.Array
    completed_length: jax.Array
    reset_count: jax.Array


class MineExpeditionTrainingBatch(struct.PyTreeNode):
    transition: MineExpeditionTransition
    advantages: jax.Array
    targets: jax.Array


def _batch_inputs(env_state):
    observation = jax.vmap(observe_mine_expedition)(env_state)
    return observation.map_channels, observation.numeric_features


def mine_expedition_parameter_count(parameters) -> int:
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


def initialize_mine_expedition_ppo(config):
    config.validate()
    init_rng, action_rng, env_rng = jax.random.split(
        jax.random.PRNGKey(config.seed), 3
    )
    worker_keys = jax.random.split(env_rng, config.num_envs)
    split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(worker_keys)
    env_keys = split_keys[:, 0]
    reset_keys = split_keys[:, 1]
    start = MineExpeditionStart(config.training_start)
    env_state = jax.vmap(lambda key: reset_mine_expedition(key, start))(reset_keys)
    map_channels, numeric_features = _batch_inputs(env_state)
    if map_channels.shape[1:] != (32, 32, len(MAP_CHANNEL_NAMES)):
        raise ValueError("map observation does not match the public schema")
    if numeric_features.shape[-1] != len(NUMERIC_FEATURE_NAMES):
        raise ValueError("numeric observation does not match the public schema")
    network = MineExpeditionActorCritic(hidden_size=config.hidden_size)
    parameters = network.init(
        init_rng, map_channels[:1], numeric_features[:1]
    )
    train_state = TrainState.create(
        apply_fn=network.apply, params=parameters, tx=_optimizer(config)
    )
    return network, MineExpeditionRunnerState(
        train_state=train_state,
        env_state=env_state,
        env_keys=env_keys,
        rng=action_rng,
        episode_returns=jnp.zeros((config.num_envs,), dtype=jnp.float32),
        episode_lengths=jnp.zeros((config.num_envs,), dtype=jnp.int32),
        global_update=jnp.asarray(0, dtype=jnp.int32),
        env_steps=jnp.asarray(0, dtype=jnp.int32),
    )


def step_mine_expedition_workers(runner, actions, config):
    stepped = jax.vmap(
        lambda state, action: step_mine_expedition_env(
            state, action, MineExpeditionVariant.FIXED
        )
    )(runner.env_state, actions)
    _, stepped_states, events = stepped
    split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(runner.env_keys)
    next_env_keys = split_keys[:, 0]
    reset_keys = split_keys[:, 1]
    start = MineExpeditionStart(config.training_start)
    reset_states = jax.vmap(lambda key: reset_mine_expedition(key, start))(
        reset_keys
    )
    env_state = _tree_where(events.done, reset_states, stepped_states)
    current_returns = runner.episode_returns + events.reward
    current_lengths = runner.episode_lengths + 1
    next_runner = runner.replace(
        env_state=env_state,
        env_keys=next_env_keys,
        episode_returns=jnp.where(events.done, 0.0, current_returns),
        episode_lengths=jnp.where(events.done, 0, current_lengths),
        env_steps=runner.env_steps + config.num_envs,
    )
    transition = MineExpeditionTransition(
        done=events.done,
        action=actions,
        value=jnp.zeros_like(events.reward),
        reward=events.reward,
        log_prob=jnp.zeros_like(events.reward),
        map_channels=jnp.zeros((1,), dtype=jnp.float32),
        numeric_features=jnp.zeros((1,), dtype=jnp.float32),
        success=events.success,
        timeout=events.timeout,
        crafted_pickaxe=events.crafted_pickaxe,
        mined_target=events.mined_target,
        returned_target=events.returned_target,
        iron_increase=events.iron_increase,
        indirect_use=events.indirect_use,
        completed_return=jnp.where(events.done, current_returns, 0.0),
        completed_length=jnp.where(events.done, current_lengths, 0),
        reset_count=events.done.astype(jnp.int32),
    )
    return next_runner, transition


def calculate_mine_expedition_gae(trajectory, last_value, gamma, gae_lambda):
    def backward(carry, transition):
        gae, next_value = carry
        nonterminal = 1.0 - transition.done.astype(jnp.float32)
        delta = transition.reward + gamma * next_value * nonterminal - transition.value
        gae = delta + gamma * gae_lambda * nonterminal * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        backward,
        (jnp.zeros_like(last_value), last_value),
        trajectory,
        reverse=True,
    )
    return advantages, advantages + trajectory.value


def make_mine_expedition_update(network, config):
    config.validate()

    def rollout_step(runner, _):
        model_inputs = _batch_inputs(runner.env_state)
        policy, value = network.apply(runner.train_state.params, *model_inputs)
        rng, action_rng = jax.random.split(runner.rng)
        actions = policy.sample(seed=action_rng)
        log_prob = policy.log_prob(actions)
        runner = runner.replace(rng=rng)
        runner, event = step_mine_expedition_workers(runner, actions, config)
        transition = event.replace(
            value=value,
            log_prob=log_prob,
            map_channels=model_inputs[0],
            numeric_features=model_inputs[1],
        )
        return runner, transition

    def update_minibatch(train_state, batch):
        def loss_fn(parameters):
            policy, value = network.apply(
                parameters,
                batch.transition.map_channels,
                batch.transition.numeric_features,
            )
            log_prob = policy.log_prob(batch.transition.action)
            normalized_advantages = (
                batch.advantages - jnp.mean(batch.advantages)
            ) / (jnp.std(batch.advantages) + 1e-8)
            clipped_value = batch.transition.value + (
                value - batch.transition.value
            ).clip(-config.clip_epsilon, config.clip_epsilon)
            value_loss = 0.5 * jnp.mean(
                jnp.maximum(
                    jnp.square(value - batch.targets),
                    jnp.square(clipped_value - batch.targets),
                )
            )
            ratio = jnp.exp(log_prob - batch.transition.log_prob)
            policy_loss = -jnp.mean(
                jnp.minimum(
                    ratio * normalized_advantages,
                    jnp.clip(
                        ratio,
                        1.0 - config.clip_epsilon,
                        1.0 + config.clip_epsilon,
                    )
                    * normalized_advantages,
                )
            )
            entropy = jnp.mean(policy.entropy())
            total = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )
            return total, (value_loss, policy_loss, entropy)

        (loss, auxiliary), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            train_state.params
        )
        train_state = train_state.apply_gradients(grads=gradients)
        value_loss, policy_loss, entropy = auxiliary
        return train_state, {
            "loss": loss,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "entropy": entropy,
        }

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
        train_state, losses = jax.lax.scan(
            update_minibatch, train_state, minibatches
        )
        return (train_state, flat_batch, rng), losses

    def update(runner):
        start_steps = runner.env_steps
        runner, trajectory = jax.lax.scan(
            rollout_step, runner, None, length=config.num_steps
        )
        _, last_value = network.apply(
            runner.train_state.params, *_batch_inputs(runner.env_state)
        )
        advantages, targets = calculate_mine_expedition_gae(
            trajectory, last_value, config.gamma, config.gae_lambda
        )
        flat_batch = jax.tree.map(
            lambda value: value.reshape((config.batch_size,) + value.shape[2:]),
            MineExpeditionTrainingBatch(trajectory, advantages, targets),
        )
        (train_state, _, rng), losses = jax.lax.scan(
            update_epoch,
            (runner.train_state, flat_batch, runner.rng),
            None,
            length=config.update_epochs,
        )
        runner = runner.replace(
            train_state=train_state,
            rng=rng,
            global_update=runner.global_update + 1,
        )
        completed = trajectory.reset_count.astype(jnp.float32)
        completed_count = jnp.sum(completed)
        metrics = {
            key: jnp.mean(value) for key, value in losses.items()
        }
        metrics.update(
            {
                "transitions": runner.env_steps - start_steps,
                "completed_episodes": completed_count.astype(jnp.int32),
                "completed_successes": jnp.sum(
                    trajectory.success.astype(jnp.int32)
                ),
                "completed_timeouts": jnp.sum(
                    trajectory.timeout.astype(jnp.int32)
                ),
                "completed_mean_return": jnp.sum(
                    trajectory.completed_return
                )
                / jnp.maximum(completed_count, 1.0),
                "completed_mean_length": jnp.sum(
                    trajectory.completed_length
                )
                / jnp.maximum(completed_count, 1.0),
                "crafted_pickaxes": jnp.sum(trajectory.crafted_pickaxe),
                "mined_targets": jnp.sum(trajectory.mined_target),
                "returned_targets": jnp.sum(trajectory.returned_target),
                "iron_increase_events": jnp.sum(trajectory.iron_increase),
                "indirect_use_events": jnp.sum(trajectory.indirect_use),
            }
        )
        return runner, metrics

    return update


_FROZEN_EVAL_CACHE = {}


def evaluate_mine_expedition_frozen(
    network,
    parameters,
    *,
    stochastic,
    episodes,
    seed_base,
    learner_seed,
    discount=TASK_DISCOUNT,
    record_episodes=False,
):
    """Evaluate natural-start fixed play without changing learner state."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    keys = jax.vmap(jax.random.PRNGKey)(
        seed_base + 1000 * learner_seed + jnp.arange(episodes, dtype=jnp.int32)
    )
    states = jax.vmap(
        lambda key: reset_mine_expedition(key, MineExpeditionStart.NATURAL)
    )(keys)

    def rollout(parameters):
        def eval_step(carry, _):
            (
                state,
                done,
                success,
                length,
                shaped_return,
                crafted,
                mined,
                action_keys,
            ) = carry
            policy, _ = network.apply(parameters, *_batch_inputs(state))
            split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(action_keys)
            next_action_keys = split_keys[:, 0]
            draw_keys = split_keys[:, 1]
            if stochastic:
                actions = jax.vmap(
                    lambda logits, key: distrax.Categorical(logits=logits).sample(
                        seed=key
                    )
                )(policy.logits, draw_keys)
            else:
                actions = jnp.argmax(policy.logits, axis=-1)
            active = jnp.logical_not(done)
            actions = jnp.where(
                active, actions, int(MineExpeditionAction.NOOP)
            )
            _, stepped, events = jax.vmap(
                lambda item, action: step_mine_expedition_env(
                    item, action, MineExpeditionVariant.FIXED
                )
            )(state, actions)
            just_done = jnp.logical_and(active, events.done)
            return (
                _tree_where(active, stepped, state),
                jnp.logical_or(done, just_done),
                jnp.logical_or(success, jnp.logical_and(active, events.success)),
                length + active.astype(jnp.int32),
                shaped_return + jnp.where(active, events.reward, 0.0),
                jnp.logical_or(
                    crafted, jnp.logical_and(active, events.crafted_pickaxe)
                ),
                jnp.logical_or(mined, jnp.logical_and(active, events.mined_target)),
                next_action_keys,
            ), None

        zeros_bool = jnp.zeros((episodes,), dtype=jnp.bool_)
        zeros_int = jnp.zeros((episodes,), dtype=jnp.int32)
        final, _ = jax.lax.scan(
            eval_step,
            (
                states,
                zeros_bool,
                zeros_bool,
                zeros_int,
                jnp.zeros((episodes,), dtype=jnp.float32),
                zeros_bool,
                zeros_bool,
                keys,
            ),
            None,
            length=WORLD_HORIZON,
        )
        return final

    cache_key = (
        id(network),
        bool(stochastic),
        int(episodes),
        int(seed_base),
        int(learner_seed),
        float(discount),
    )
    compiled = _FROZEN_EVAL_CACHE.get(cache_key)
    if compiled is None:
        compiled = jax.jit(rollout)
        _FROZEN_EVAL_CACHE[cache_key] = compiled
    (
        _,
        done,
        success,
        length,
        shaped_return,
        crafted,
        mined,
        _,
    ) = jax.device_get(compiled(parameters))
    success_array = np.asarray(success, dtype=np.float64)
    length_array = np.asarray(length)
    discounted_return = np.where(
        length_array >= 1,
        success_array * (discount ** (length_array - 1)),
        0.0,
    )
    result = {
        "variant": MineExpeditionVariant.FIXED.value,
        "start": MineExpeditionStart.NATURAL.value,
        "stochastic": bool(stochastic),
        "episodes": int(episodes),
        "success_rate": float(np.mean(success_array)),
        "mean_length": float(np.mean(length_array)),
        "mean_shaped_return": float(np.mean(np.asarray(shaped_return))),
        "mean_discounted_task_return": float(np.mean(discounted_return)),
        "completed_rate": float(np.mean(np.asarray(done))),
        "crafted_pickaxe_rate": float(np.mean(np.asarray(crafted))),
        "mined_target_rate": float(np.mean(np.asarray(mined))),
    }
    if record_episodes:
        result["episode_records"] = [
            {
                "episode": index,
                "success": bool(success[index]),
                "length": int(length[index]),
                "shaped_return": float(shaped_return[index]),
                "discounted_task_return": float(discounted_return[index]),
                "crafted_pickaxe": bool(crafted[index]),
                "mined_target": bool(mined[index]),
            }
            for index in range(episodes)
        ]
    return result


def mine_expedition_config_payload(config):
    payload = asdict(config)
    payload["checkpoint_updates"] = list(config.checkpoint_updates)
    return payload


def mine_expedition_checkpoint_files_present(directory):
    directory = Path(directory)
    state = directory / "state.msgpack"
    metadata_path = directory / "metadata.json"
    config_path = directory / "config.json"
    if not (
        state.is_file()
        and state.stat().st_size > 0
        and metadata_path.is_file()
        and config_path.is_file()
    ):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    digest = hashlib.sha256(state.read_bytes()).hexdigest()
    return metadata.get("state_sha256") == digest


def _atomic_write_bytes(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_write_json(path, payload):
    _atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def save_mine_expedition_checkpoint(directory, runner, config):
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    state_bytes = serialization.to_bytes(runner)
    _atomic_write_bytes(destination / "state.msgpack", state_bytes)
    _atomic_write_json(
        destination / "config.json", mine_expedition_config_payload(config)
    )
    metadata = {
        "schema_version": "hackrl_mine_expedition_fixed_checkpoint_v1",
        "global_update": int(runner.global_update),
        "environment_steps": int(runner.env_steps),
        "parameter_count": mine_expedition_parameter_count(
            runner.train_state.params
        ),
        "variant": MineExpeditionVariant.FIXED.value,
        "evaluation_start": MineExpeditionStart.NATURAL.value,
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
    }
    _atomic_write_json(destination / "metadata.json", metadata)
    return destination


def load_mine_expedition_checkpoint(directory, template, config):
    source = Path(directory)
    if not mine_expedition_checkpoint_files_present(source):
        raise FileNotFoundError(f"checkpoint state is missing: {source}")
    recorded = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if recorded != mine_expedition_config_payload(config):
        raise ValueError("checkpoint config does not match requested config")
    return serialization.from_bytes(template, (source / "state.msgpack").read_bytes())
