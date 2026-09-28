"""Deterministic policy evaluation for HackRL environments."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from hackrl.rollout import HackRLBatchEnv


@struct.dataclass
class EvaluationEpisodes:
    """First completed episode recorded for each evaluation worker."""

    completed: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    goal_success: jax.Array
    violation_count: jax.Array
    harvest_count: jax.Array
    repeated_harvest_count: jax.Array
    ever_iron: jax.Array
    wood_exhausted_before_goal: jax.Array


def _empty_results(num_episodes: int) -> EvaluationEpisodes:
    zeros_i = jnp.zeros((num_episodes,), dtype=jnp.int32)
    zeros_b = jnp.zeros((num_episodes,), dtype=bool)
    return EvaluationEpisodes(
        completed=zeros_b,
        episode_return=jnp.zeros((num_episodes,), dtype=jnp.float32),
        episode_length=zeros_i,
        goal_success=zeros_b,
        violation_count=zeros_i,
        harvest_count=zeros_i,
        repeated_harvest_count=zeros_i,
        ever_iron=zeros_b,
        wood_exhausted_before_goal=zeros_b,
    )


def _record_first(results, record, newly_completed):
    return EvaluationEpisodes(
        completed=jnp.logical_or(results.completed, newly_completed),
        episode_return=jnp.where(
            newly_completed, record.episode_return, results.episode_return
        ),
        episode_length=jnp.where(
            newly_completed, record.episode_length, results.episode_length
        ),
        goal_success=jnp.where(
            newly_completed, record.goal_success, results.goal_success
        ),
        violation_count=jnp.where(
            newly_completed, record.violation_count, results.violation_count
        ),
        harvest_count=jnp.where(
            newly_completed, record.harvest_count, results.harvest_count
        ),
        repeated_harvest_count=jnp.where(
            newly_completed,
            record.repeated_harvest_count,
            results.repeated_harvest_count,
        ),
        ever_iron=jnp.where(
            newly_completed, record.ever_iron, results.ever_iron
        ),
        wood_exhausted_before_goal=jnp.where(
            newly_completed,
            record.wood_exhausted_before_goal,
            results.wood_exhausted_before_goal,
        ),
    )


def evaluate_policy(
    network,
    parameters,
    env,
    rng,
    num_episodes: int,
    stochastic: bool = False,
):
    """Evaluate a categorical policy by mode or sample."""

    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    vector_env = HackRLBatchEnv(env, num_episodes)
    rng, reset_rng = jax.random.split(rng)
    observations, vector_state = vector_env.reset(reset_rng)
    results = _empty_results(num_episodes)

    def evaluation_step(carry, _):
        observations, vector_state, results, rng = carry
        rng, action_rng, step_rng = jax.random.split(rng, 3)
        policy, _ = network.apply(parameters, observations)
        actions = jnp.where(
            stochastic,
            policy.sample(seed=action_rng).astype(jnp.int32),
            policy.mode().astype(jnp.int32),
        )
        (
            policy_observations,
            vector_state,
            transition,
        ) = vector_env.step(step_rng, vector_state, actions)
        newly_completed = jnp.logical_and(
            jnp.logical_not(results.completed), transition.done
        )
        results = _record_first(
            results, transition.episode, newly_completed
        )
        return (
            policy_observations,
            vector_state,
            results,
            rng,
        ), None

    (_, _, results, _), _ = jax.lax.scan(
        evaluation_step,
        (observations, vector_state, results, rng),
        None,
        length=env.spec.horizon,
    )
    completed_float = results.completed.astype(jnp.float32)
    return {
        "eval_episodes": jnp.asarray(num_episodes, dtype=jnp.int32),
        "eval_completion_rate": jnp.mean(completed_float),
        "eval_success_rate": jnp.mean(results.goal_success.astype(jnp.float32)),
        "eval_violation_episode_rate": jnp.mean(
            (results.violation_count > 0).astype(jnp.float32)
        ),
        "eval_repeat_harvest_episode_rate": jnp.mean(
            (results.repeated_harvest_count > 0).astype(jnp.float32)
        ),
        "eval_iron_acquire_rate": jnp.mean(
            results.ever_iron.astype(jnp.float32)
        ),
        "eval_wood_exhausted_before_goal_rate": jnp.mean(
            results.wood_exhausted_before_goal.astype(jnp.float32)
        ),
        "eval_mean_return": jnp.mean(results.episode_return),
        "eval_mean_length": jnp.mean(
            results.episode_length.astype(jnp.float32)
        ),
    }
