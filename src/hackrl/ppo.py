"""PPO adapted from Michael Matthews' Craftax_Baselines.

The network, GAE, clipped PPO objective, optimizer, and update ordering follow
Craftax_Baselines commit 7ce36fa05b84a2c9e758012f1e6da402e1e3a891.
HackRL changes only environment construction, terminal-safe rollout handling,
episode metrics, evaluation, and the bounded pilot interface.

Upstream copyright (c) 2024 Michael Matthews, MIT License. See
THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from hackrl.evaluation import evaluate_policy
from hackrl.rollout import HackRLBatchEnv
from hackrl.run_artifacts import (
    first_success_update,
    repo_git_diff,
    repo_git_sha,
    write_run_artifacts,
)
from hackrl.tasks import (
    EasyTask,
    FIXTURE_DYNAMICS_VERSION,
    FixtureVersion,
    HackRLEasySymbolicEnvNoAutoReset,
    MediumTask,
    StartMode,
    parse_task,
)


class ActorCritic(nn.Module):
    """The symbolic actor-critic network from Craftax_Baselines."""

    action_dim: int
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, observation):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        actor = observation
        for _ in range(3):
            actor = nn.Dense(
                self.layer_width,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(actor)
            actor = activation(actor)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)
        policy = distrax.Categorical(logits=logits)

        critic = observation
        for _ in range(3):
            critic = nn.Dense(
                self.layer_width,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(critic)
            critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)
        return policy, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    observation: jax.Array
    terminal_observation: jax.Array
    completed_episode: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    episode_success: jax.Array
    episode_violation: jax.Array
    episode_repeat_harvest: jax.Array
    episode_iron_acquired: jax.Array
    episode_iron_pickaxe_crafted: jax.Array
    episode_diamond_acquired: jax.Array
    episode_wood_depleted: jax.Array
    episode_wood_depleted_before_pickaxe: jax.Array
    episode_wood_replenished_after_depletion: jax.Array
    episode_damage: jax.Array
    episode_damage_taken: jax.Array
    episode_termination_goal: jax.Array
    episode_termination_death: jax.Array
    episode_termination_timeout: jax.Array
    violation_event: jax.Array
    repeat_harvest_event: jax.Array


class TrainingBatch(NamedTuple):
    transition: Transition
    advantages: jax.Array
    targets: jax.Array


@dataclass(frozen=True)
class PPOConfig:
    """Craftax PPO settings plus a bounded HackRL task selection."""

    task: EasyTask | MediumTask | str = EasyTask.R_E
    mutant: bool = True
    seed: int = 0
    num_envs: int = 8
    num_steps: int = 16
    num_updates: int = 2
    update_epochs: int = 4
    num_minibatches: int = 2
    layer_size: int = 64
    learning_rate: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.8
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 1.0
    activation: str = "tanh"
    anneal_learning_rate: bool = False
    eval_episodes: int = 8
    start_mode: str = StartMode.DEFAULT.value
    fixture: str = FixtureVersion.DEFAULT.value
    log_dir: str | None = None

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    def validate(self):
        if self.num_envs <= 0 or self.num_steps <= 0:
            raise ValueError("num_envs and num_steps must be positive")
        if self.num_updates <= 0 or self.update_epochs <= 0:
            raise ValueError("num_updates and update_epochs must be positive")
        if (
            self.num_minibatches <= 0
            or self.batch_size % self.num_minibatches
        ):
            raise ValueError("rollout batch must divide evenly into minibatches")
        if self.layer_size <= 0 or self.eval_episodes <= 0:
            raise ValueError("layer_size and eval_episodes must be positive")
        if self.activation not in {"tanh", "relu"}:
            raise ValueError("activation must be 'tanh' or 'relu'")
        start_mode = StartMode(self.start_mode)
        if (
            start_mode is StartMode.R_E_POST_IRON
            and parse_task(self.task) is not EasyTask.R_E
        ):
            raise ValueError("r_e_post_iron is only defined for R-E")
        fixture = FixtureVersion(self.fixture)
        if (
            fixture is FixtureVersion.R_E_REPLENISH
            and parse_task(self.task) is not EasyTask.R_E
        ):
            raise ValueError("r_e_replenish is only defined for R-E")


def _parameter_norm(parameters):
    squared = [
        jnp.sum(jnp.square(value))
        for value in jax.tree.leaves(parameters)
    ]
    return jnp.sqrt(jnp.sum(jnp.stack(squared)))


def _linear_schedule(config: PPOConfig):
    updates_per_iteration = config.num_minibatches * config.update_epochs

    def schedule(count):
        completed_updates = count // updates_per_iteration
        fraction = 1.0 - completed_updates / config.num_updates
        return config.learning_rate * fraction

    return schedule


def _make_optimizer(config: PPOConfig):
    learning_rate = (
        _linear_schedule(config)
        if config.anneal_learning_rate
        else config.learning_rate
    )
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=learning_rate, eps=1e-5),
    )


def _make_update(vector_env, network, config: PPOConfig):
    def rollout_step(runner_state, _):
        train_state, vector_state, observation, rng = runner_state

        rng, action_rng, step_rng = jax.random.split(rng, 3)
        policy, value = network.apply(train_state.params, observation)
        action = policy.sample(seed=action_rng)
        log_prob = policy.log_prob(action)

        (
            policy_observation,
            vector_state,
            env_transition,
        ) = vector_env.step(step_rng, vector_state, action)
        episode = env_transition.episode
        transition = Transition(
            done=env_transition.done,
            action=action,
            value=value,
            reward=env_transition.reward,
            log_prob=log_prob,
            observation=observation,
            terminal_observation=env_transition.terminal_observation,
            completed_episode=episode.completed,
            episode_return=episode.episode_return,
            episode_length=episode.episode_length,
            episode_success=episode.goal_success,
            episode_violation=episode.violation_count > 0,
            episode_repeat_harvest=episode.repeated_harvest_count > 0,
            episode_iron_acquired=episode.iron_acquired_count > 0,
            episode_iron_pickaxe_crafted=(
                episode.iron_pickaxe_crafted_count > 0
            ),
            episode_diamond_acquired=episode.diamond_acquired_count > 0,
            episode_wood_depleted=episode.wood_depletion_count > 0,
            episode_wood_depleted_before_pickaxe=(
                episode.wood_depleted_before_pickaxe
            ),
            episode_wood_replenished_after_depletion=(
                episode.wood_replenished_after_depletion
            ),
            episode_damage=episode.damage_event_count > 0,
            episode_damage_taken=episode.damage_taken,
            episode_termination_goal=episode.termination_goal,
            episode_termination_death=episode.termination_death,
            episode_termination_timeout=episode.termination_timeout,
            violation_event=env_transition.info["HackRL/violation"],
            repeat_harvest_event=env_transition.repeated_harvest,
        )
        runner_state = (
            train_state,
            vector_state,
            policy_observation,
            rng,
        )
        return runner_state, transition

    def calculate_gae(trajectory, last_value):
        def get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            nonterminal = 1.0 - transition.done.astype(jnp.float32)
            delta = (
                transition.reward
                + config.gamma * next_value * nonterminal
                - transition.value
            )
            gae = (
                delta
                + config.gamma
                * config.gae_lambda
                * nonterminal
                * gae
            )
            return (gae, transition.value), gae

        _, advantages = jax.lax.scan(
            get_advantages,
            (jnp.zeros_like(last_value), last_value),
            trajectory,
            reverse=True,
        )
        return advantages, advantages + trajectory.value

    def update_minibatch(train_state, batch):
        def loss_fn(parameters, transition, advantages, targets):
            policy, value = network.apply(
                parameters, transition.observation
            )
            log_prob = policy.log_prob(transition.action)

            value_pred_clipped = transition.value + (
                value - transition.value
            ).clip(-config.clip_epsilon, config.clip_epsilon)
            value_losses = jnp.square(value - targets)
            value_losses_clipped = jnp.square(
                value_pred_clipped - targets
            )
            value_loss = 0.5 * jnp.maximum(
                value_losses, value_losses_clipped
            ).mean()

            ratio = jnp.exp(log_prob - transition.log_prob)
            normalized_advantages = (
                advantages - advantages.mean()
            ) / (advantages.std() + 1e-8)
            actor_loss_unclipped = ratio * normalized_advantages
            actor_loss_clipped = (
                jnp.clip(
                    ratio,
                    1.0 - config.clip_epsilon,
                    1.0 + config.clip_epsilon,
                )
                * normalized_advantages
            )
            actor_loss = -jnp.minimum(
                actor_loss_unclipped, actor_loss_clipped
            ).mean()
            entropy = policy.entropy().mean()
            total_loss = (
                actor_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )
            return total_loss, (value_loss, actor_loss, entropy)

        gradient_function = jax.value_and_grad(loss_fn, has_aux=True)
        (total_loss, auxiliary), gradients = gradient_function(
            train_state.params,
            batch.transition,
            batch.advantages,
            batch.targets,
        )
        train_state = train_state.apply_gradients(grads=gradients)
        value_loss, actor_loss, entropy = auxiliary
        return train_state, {
            "loss": total_loss,
            "value_loss": value_loss,
            "policy_loss": actor_loss,
            "entropy": entropy,
        }

    def update_epoch(epoch_state, _):
        train_state, trajectory, advantages, targets, rng = epoch_state
        rng, permutation_rng = jax.random.split(rng)
        permutation = jax.random.permutation(
            permutation_rng, config.batch_size
        )
        flattened = jax.tree.map(
            lambda value: value.reshape(
                (config.batch_size,) + value.shape[2:]
            ),
            TrainingBatch(trajectory, advantages, targets),
        )
        shuffled = jax.tree.map(
            lambda value: jnp.take(value, permutation, axis=0),
            flattened,
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
        return (
            train_state,
            trajectory,
            advantages,
            targets,
            rng,
        ), losses

    def update(runner_state, _):
        (
            train_state,
            vector_state,
            observation,
            rng,
        ), trajectory = jax.lax.scan(
            rollout_step,
            runner_state,
            None,
            length=config.num_steps,
        )
        _, last_value = network.apply(train_state.params, observation)
        advantages, targets = calculate_gae(trajectory, last_value)

        update_state = (
            train_state,
            trajectory,
            advantages,
            targets,
            rng,
        )
        update_state, loss_metrics = jax.lax.scan(
            update_epoch,
            update_state,
            None,
            length=config.update_epochs,
        )
        train_state, _, _, _, rng = update_state
        losses = jax.tree.map(jnp.mean, loss_metrics)
        completed = trajectory.completed_episode
        terminated_before_iron = jnp.logical_and(
            ~trajectory.episode_iron_acquired,
            jnp.logical_and(
                ~trajectory.episode_iron_pickaxe_crafted,
                ~trajectory.episode_diamond_acquired,
            ),
        )
        terminated_after_iron_before_pickaxe = jnp.logical_and(
            trajectory.episode_iron_acquired,
            ~trajectory.episode_iron_pickaxe_crafted,
        )
        terminated_after_pickaxe_before_diamond = jnp.logical_and(
            trajectory.episode_iron_pickaxe_crafted,
            ~trajectory.episode_diamond_acquired,
        )
        metrics = {
            **losses,
            "transitions": jnp.asarray(
                config.batch_size, dtype=jnp.int32
            ),
            "completed_episodes": jnp.sum(completed),
            "successful_episodes": jnp.sum(
                jnp.logical_and(completed, trajectory.episode_success)
            ),
            "violation_episodes": jnp.sum(
                jnp.logical_and(completed, trajectory.episode_violation)
            ),
            "repeat_harvest_episodes": jnp.sum(
                jnp.logical_and(
                    completed, trajectory.episode_repeat_harvest
                )
            ),
            "iron_acquisition_episodes": jnp.sum(
                jnp.logical_and(completed, trajectory.episode_iron_acquired)
            ),
            "iron_pickaxe_craft_episodes": jnp.sum(
                jnp.logical_and(
                    completed, trajectory.episode_iron_pickaxe_crafted
                )
            ),
            "diamond_acquisition_episodes": jnp.sum(
                jnp.logical_and(completed, trajectory.episode_diamond_acquired)
            ),
            "wood_depletion_episodes": jnp.sum(
                jnp.logical_and(completed, trajectory.episode_wood_depleted)
            ),
            "wood_depleted_before_pickaxe_episodes": jnp.sum(
                jnp.logical_and(
                    completed, trajectory.episode_wood_depleted_before_pickaxe
                )
            ),
            "wood_replenished_after_depletion_episodes": jnp.sum(
                jnp.logical_and(
                    completed,
                    trajectory.episode_wood_replenished_after_depletion,
                )
            ),
            "damage_episodes": jnp.sum(
                jnp.logical_and(completed, trajectory.episode_damage)
            ),
            "completed_damage_sum": jnp.sum(
                jnp.where(completed, trajectory.episode_damage_taken, 0)
            ),
            "goal_terminations": jnp.sum(
                jnp.logical_and(completed, trajectory.episode_termination_goal)
            ),
            "death_terminations": jnp.sum(
                jnp.logical_and(completed, trajectory.episode_termination_death)
            ),
            "timeout_terminations": jnp.sum(
                jnp.logical_and(completed, trajectory.episode_termination_timeout)
            ),
            "death_before_iron_episodes": jnp.sum(
                jnp.logical_and(
                    completed,
                    jnp.logical_and(
                        terminated_before_iron,
                        trajectory.episode_termination_death,
                    ),
                )
            ),
            "timeout_before_iron_episodes": jnp.sum(
                jnp.logical_and(
                    completed,
                    jnp.logical_and(
                        terminated_before_iron,
                        trajectory.episode_termination_timeout,
                    ),
                )
            ),
            "death_after_iron_before_pickaxe_episodes": jnp.sum(
                jnp.logical_and(
                    completed,
                    jnp.logical_and(
                        terminated_after_iron_before_pickaxe,
                        trajectory.episode_termination_death,
                    ),
                )
            ),
            "timeout_after_iron_before_pickaxe_episodes": jnp.sum(
                jnp.logical_and(
                    completed,
                    jnp.logical_and(
                        terminated_after_iron_before_pickaxe,
                        trajectory.episode_termination_timeout,
                    ),
                )
            ),
            "death_after_pickaxe_before_diamond_episodes": jnp.sum(
                jnp.logical_and(
                    completed,
                    jnp.logical_and(
                        terminated_after_pickaxe_before_diamond,
                        trajectory.episode_termination_death,
                    ),
                )
            ),
            "timeout_after_pickaxe_before_diamond_episodes": jnp.sum(
                jnp.logical_and(
                    completed,
                    jnp.logical_and(
                        terminated_after_pickaxe_before_diamond,
                        trajectory.episode_termination_timeout,
                    ),
                )
            ),
            "violation_events": jnp.sum(trajectory.violation_event),
            "repeat_harvest_events": jnp.sum(
                trajectory.repeat_harvest_event
            ),
            "completed_return_sum": jnp.sum(
                jnp.where(completed, trajectory.episode_return, 0.0)
            ),
            "completed_length_sum": jnp.sum(
                jnp.where(completed, trajectory.episode_length, 0)
            ),
        }
        runner_state = (
            train_state,
            vector_state,
            observation,
            rng,
        )
        return runner_state, metrics

    return update


def run_ppo_pilot(config: PPOConfig):
    """Train and evaluate one bounded Easy-task PPO pilot."""

    config.validate()
    git_sha = repo_git_sha()
    git_diff = repo_git_diff()
    env = HackRLEasySymbolicEnvNoAutoReset(
        parse_task(config.task),
        mutant=config.mutant,
        start_mode=config.start_mode,
        fixture=config.fixture,
    )
    vector_env = HackRLBatchEnv(env, config.num_envs)
    network = ActorCritic(
        action_dim=env.action_space(env.default_params).n,
        layer_width=config.layer_size,
        activation=config.activation,
    )

    rng = jax.random.PRNGKey(config.seed)
    rng, initialization_rng, reset_rng = jax.random.split(rng, 3)
    observation_shape = env.observation_space(
        env.default_params
    ).shape
    network_parameters = network.init(
        initialization_rng,
        jnp.zeros((1, *observation_shape)),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_parameters,
        tx=_make_optimizer(config),
    )
    initial_parameter_norm = _parameter_norm(train_state.params)
    observations, vector_state = vector_env.reset(reset_rng)
    runner_state = (
        train_state,
        vector_state,
        observations,
        rng,
    )

    update = _make_update(vector_env, network, config)

    def train_all_updates(state):
        return jax.lax.scan(
            update, state, None, length=config.num_updates
        )

    runner_state, update_metrics = jax.jit(train_all_updates)(runner_state)
    train_state, _, _, rng = runner_state
    final_parameter_norm = _parameter_norm(train_state.params)

    host_metrics = jax.tree.map(
        lambda value: np.asarray(jax.device_get(value)),
        update_metrics,
    )
    totals = {}
    for key in (
        "transitions",
        "completed_episodes",
        "successful_episodes",
        "violation_episodes",
        "repeat_harvest_episodes",
        "iron_acquisition_episodes",
        "iron_pickaxe_craft_episodes",
        "diamond_acquisition_episodes",
        "wood_depletion_episodes",
        "wood_depleted_before_pickaxe_episodes",
        "wood_replenished_after_depletion_episodes",
        "damage_episodes",
        "completed_damage_sum",
        "goal_terminations",
        "death_terminations",
        "timeout_terminations",
        "death_before_iron_episodes",
        "timeout_before_iron_episodes",
        "death_after_iron_before_pickaxe_episodes",
        "timeout_after_iron_before_pickaxe_episodes",
        "death_after_pickaxe_before_diamond_episodes",
        "timeout_after_pickaxe_before_diamond_episodes",
        "violation_events",
        "repeat_harvest_events",
        "completed_return_sum",
        "completed_length_sum",
    ):
        totals[key] = float(np.sum(host_metrics[key]))
    optimization = {
        key: float(host_metrics[key][-1])
        for key in ("loss", "policy_loss", "value_loss", "entropy")
    }
    first_success = first_success_update(host_metrics["successful_episodes"])

    rng, mode_rng, sample_rng = jax.random.split(rng, 3)

    def _evaluate(key, stochastic):
        return evaluate_policy(
            network,
            train_state.params,
            env,
            key,
            config.eval_episodes,
            stochastic=stochastic,
        )

    evaluation = {
        key: float(jax.device_get(value))
        for key, value in jax.jit(lambda key: _evaluate(key, False))(
            mode_rng
        ).items()
    }
    sample_evaluation = {
        key.replace("eval_", "eval_sample_", 1): float(jax.device_get(value))
        for key, value in jax.jit(lambda key: _evaluate(key, True))(
            sample_rng
        ).items()
    }

    completed = totals["completed_episodes"]
    result = {
        "task": parse_task(config.task).value,
        "variant": "mutant" if config.mutant else "fixed",
        "start_mode": StartMode(config.start_mode).value,
        "fixture": FixtureVersion(config.fixture).value,
        "git_sha": git_sha,
        "working_tree_dirty": bool(git_diff),
        "metrics_schema_version": 2,
        "fixture_dynamics_version": FIXTURE_DYNAMICS_VERSION,
        **totals,
        **optimization,
        **evaluation,
        **sample_evaluation,
        "first_success_update": first_success,
        "completed_success_rate": (
            totals["successful_episodes"] / completed if completed else 0.0
        ),
        "completed_violation_rate": (
            totals["violation_episodes"] / completed if completed else 0.0
        ),
        "completed_repeat_harvest_rate": (
            totals["repeat_harvest_episodes"] / completed
            if completed
            else 0.0
        ),
        "completed_iron_acquisition_rate": (
            totals["iron_acquisition_episodes"] / completed
            if completed
            else 0.0
        ),
        "completed_iron_pickaxe_craft_rate": (
            totals["iron_pickaxe_craft_episodes"] / completed
            if completed
            else 0.0
        ),
        "completed_diamond_acquisition_rate": (
            totals["diamond_acquisition_episodes"] / completed
            if completed
            else 0.0
        ),
        "completed_wood_depletion_rate": (
            totals["wood_depletion_episodes"] / completed
            if completed
            else 0.0
        ),
        "completed_wood_depleted_before_pickaxe_rate": (
            totals["wood_depleted_before_pickaxe_episodes"] / completed
            if completed
            else 0.0
        ),
        "completed_wood_replenished_after_depletion_rate": (
            totals["wood_replenished_after_depletion_episodes"] / completed
            if completed
            else 0.0
        ),
        "completed_damage_episode_rate": (
            totals["damage_episodes"] / completed if completed else 0.0
        ),
        "completed_mean_damage_taken": (
            totals["completed_damage_sum"] / completed if completed else 0.0
        ),
        "completed_goal_termination_rate": (
            totals["goal_terminations"] / completed if completed else 0.0
        ),
        "completed_death_termination_rate": (
            totals["death_terminations"] / completed if completed else 0.0
        ),
        "completed_timeout_termination_rate": (
            totals["timeout_terminations"] / completed if completed else 0.0
        ),
        "completed_death_before_iron_rate": (
            totals["death_before_iron_episodes"] / completed
            if completed
            else 0.0
        ),
        "completed_timeout_before_iron_rate": (
            totals["timeout_before_iron_episodes"] / completed
            if completed
            else 0.0
        ),
        "completed_death_after_iron_before_pickaxe_rate": (
            totals["death_after_iron_before_pickaxe_episodes"] / completed
            if completed
            else 0.0
        ),
        "completed_timeout_after_iron_before_pickaxe_rate": (
            totals["timeout_after_iron_before_pickaxe_episodes"] / completed
            if completed
            else 0.0
        ),
        "completed_death_after_pickaxe_before_diamond_rate": (
            totals["death_after_pickaxe_before_diamond_episodes"] / completed
            if completed
            else 0.0
        ),
        "completed_timeout_after_pickaxe_before_diamond_rate": (
            totals["timeout_after_pickaxe_before_diamond_episodes"] / completed
            if completed
            else 0.0
        ),
        "initial_parameter_norm": float(
            jax.device_get(initial_parameter_norm)
        ),
        "final_parameter_norm": float(
            jax.device_get(final_parameter_norm)
        ),
    }
    if config.log_dir is not None:
        write_run_artifacts(
            config.log_dir,
            config=config,
            git_sha=git_sha,
            git_diff=git_diff,
            train_state=train_state,
            update_metrics=host_metrics,
            summary=result,
        )
        result["log_dir"] = config.log_dir
    return result
