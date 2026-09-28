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
    iron_acquired_count: jax.Array
    iron_pickaxe_crafted_count: jax.Array
    diamond_acquired_count: jax.Array
    wood_depletion_count: jax.Array
    wood_replenishment_count: jax.Array
    wood_depleted_before_pickaxe: jax.Array
    wood_replenished_after_depletion: jax.Array
    damage_event_count: jax.Array
    damage_taken: jax.Array
    first_iron_acquired_timestep: jax.Array
    first_iron_pickaxe_crafted_timestep: jax.Array
    first_diamond_acquired_timestep: jax.Array
    termination_goal: jax.Array
    termination_death: jax.Array
    termination_timeout: jax.Array


def _empty_results(num_episodes: int) -> EvaluationEpisodes:
    zeros_i = jnp.zeros((num_episodes,), dtype=jnp.int32)
    zeros_b = jnp.zeros((num_episodes,), dtype=bool)
    unseen_i = jnp.full((num_episodes,), -1, dtype=jnp.int32)
    return EvaluationEpisodes(
        completed=zeros_b,
        episode_return=jnp.zeros((num_episodes,), dtype=jnp.float32),
        episode_length=zeros_i,
        goal_success=zeros_b,
        violation_count=zeros_i,
        harvest_count=zeros_i,
        repeated_harvest_count=zeros_i,
        iron_acquired_count=zeros_i,
        iron_pickaxe_crafted_count=zeros_i,
        diamond_acquired_count=zeros_i,
        wood_depletion_count=zeros_i,
        wood_replenishment_count=zeros_i,
        wood_depleted_before_pickaxe=zeros_b,
        wood_replenished_after_depletion=zeros_b,
        damage_event_count=zeros_i,
        damage_taken=zeros_i,
        first_iron_acquired_timestep=unseen_i,
        first_iron_pickaxe_crafted_timestep=unseen_i,
        first_diamond_acquired_timestep=unseen_i,
        termination_goal=zeros_b,
        termination_death=zeros_b,
        termination_timeout=zeros_b,
    )


def _record_first(results, record, newly_completed):
    values = {
        field: jnp.where(
            newly_completed,
            getattr(record, field),
            getattr(results, field),
        )
        for field in EvaluationEpisodes.__dataclass_fields__
        if field != "completed"
    }
    return EvaluationEpisodes(
        completed=jnp.logical_or(results.completed, newly_completed),
        **values,
    )


def _conditional_mean(values, mask):
    count = jnp.sum(mask.astype(jnp.int32))
    total = jnp.sum(jnp.where(mask, values, 0))
    return total.astype(jnp.float32) / jnp.maximum(count, 1)


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
    completed = results.completed
    iron_acquired = results.iron_acquired_count > 0
    pickaxe_crafted = results.iron_pickaxe_crafted_count > 0
    diamond_acquired = results.diamond_acquired_count > 0
    terminated_before_iron = jnp.logical_and(
        completed,
        jnp.logical_and(
            ~iron_acquired, jnp.logical_and(~pickaxe_crafted, ~diamond_acquired)
        ),
    )
    terminated_after_iron_before_pickaxe = jnp.logical_and(
        completed, jnp.logical_and(iron_acquired, ~pickaxe_crafted)
    )
    terminated_after_pickaxe_before_diamond = jnp.logical_and(
        completed, jnp.logical_and(pickaxe_crafted, ~diamond_acquired)
    )
    return {
        "eval_episodes": jnp.asarray(num_episodes, dtype=jnp.int32),
        "eval_completion_rate": jnp.mean(completed.astype(jnp.float32)),
        "eval_success_rate": jnp.mean(results.goal_success.astype(jnp.float32)),
        "eval_violation_episode_rate": jnp.mean(
            (results.violation_count > 0).astype(jnp.float32)
        ),
        "eval_repeat_harvest_episode_rate": jnp.mean(
            (results.repeated_harvest_count > 0).astype(jnp.float32)
        ),
        "eval_iron_acquisition_episode_rate": jnp.mean(
            iron_acquired.astype(jnp.float32)
        ),
        "eval_iron_pickaxe_craft_episode_rate": jnp.mean(
            pickaxe_crafted.astype(jnp.float32)
        ),
        "eval_diamond_acquisition_episode_rate": jnp.mean(
            diamond_acquired.astype(jnp.float32)
        ),
        "eval_wood_depletion_episode_rate": jnp.mean(
            (results.wood_depletion_count > 0).astype(jnp.float32)
        ),
        "eval_wood_depleted_before_pickaxe_rate": jnp.mean(
            results.wood_depleted_before_pickaxe.astype(jnp.float32)
        ),
        "eval_wood_replenished_after_depletion_rate": jnp.mean(
            results.wood_replenished_after_depletion.astype(jnp.float32)
        ),
        "eval_damage_episode_rate": jnp.mean(
            (results.damage_event_count > 0).astype(jnp.float32)
        ),
        "eval_mean_damage_taken": jnp.mean(
            results.damage_taken.astype(jnp.float32)
        ),
        "eval_termination_goal_rate": jnp.mean(
            results.termination_goal.astype(jnp.float32)
        ),
        "eval_termination_death_rate": jnp.mean(
            results.termination_death.astype(jnp.float32)
        ),
        "eval_termination_timeout_rate": jnp.mean(
            results.termination_timeout.astype(jnp.float32)
        ),
        "eval_terminated_before_iron_rate": jnp.mean(
            terminated_before_iron.astype(jnp.float32)
        ),
        "eval_terminated_after_iron_before_pickaxe_rate": jnp.mean(
            terminated_after_iron_before_pickaxe.astype(jnp.float32)
        ),
        "eval_terminated_after_pickaxe_before_diamond_rate": jnp.mean(
            terminated_after_pickaxe_before_diamond.astype(jnp.float32)
        ),
        "eval_death_before_iron_rate": jnp.mean(
            jnp.logical_and(
                terminated_before_iron, results.termination_death
            ).astype(jnp.float32)
        ),
        "eval_timeout_before_iron_rate": jnp.mean(
            jnp.logical_and(
                terminated_before_iron, results.termination_timeout
            ).astype(jnp.float32)
        ),
        "eval_death_after_iron_before_pickaxe_rate": jnp.mean(
            jnp.logical_and(
                terminated_after_iron_before_pickaxe,
                results.termination_death,
            ).astype(jnp.float32)
        ),
        "eval_timeout_after_iron_before_pickaxe_rate": jnp.mean(
            jnp.logical_and(
                terminated_after_iron_before_pickaxe,
                results.termination_timeout,
            ).astype(jnp.float32)
        ),
        "eval_death_after_pickaxe_before_diamond_rate": jnp.mean(
            jnp.logical_and(
                terminated_after_pickaxe_before_diamond,
                results.termination_death,
            ).astype(jnp.float32)
        ),
        "eval_timeout_after_pickaxe_before_diamond_rate": jnp.mean(
            jnp.logical_and(
                terminated_after_pickaxe_before_diamond,
                results.termination_timeout,
            ).astype(jnp.float32)
        ),
        "eval_mean_first_iron_acquisition_step": _conditional_mean(
            results.first_iron_acquired_timestep, iron_acquired
        ),
        "eval_mean_first_iron_pickaxe_craft_step": _conditional_mean(
            results.first_iron_pickaxe_crafted_timestep, pickaxe_crafted
        ),
        "eval_mean_first_diamond_acquisition_step": _conditional_mean(
            results.first_diamond_acquired_timestep, diamond_acquired
        ),
        "eval_mean_return": jnp.mean(results.episode_return),
        "eval_mean_length": jnp.mean(
            results.episode_length.astype(jnp.float32)
        ),
    }
