"""JAX-compatible episode metrics for HackRL evaluation and training."""

from __future__ import annotations

import jax.numpy as jnp
from flax import struct


NORMAL_PLANT_REGROWTH_STEPS = 600


@struct.dataclass
class EpisodeTracker:
    """Per-environment event history needed for causal diagnostics."""

    last_harvest_timestep: jnp.ndarray
    episode_return: jnp.ndarray
    episode_length: jnp.ndarray
    violation_count: jnp.ndarray
    harvest_count: jnp.ndarray
    repeated_harvest_count: jnp.ndarray
    goal_success: jnp.ndarray
    iron_acquired_count: jnp.ndarray
    iron_pickaxe_crafted_count: jnp.ndarray
    diamond_acquired_count: jnp.ndarray
    wood_depletion_count: jnp.ndarray
    wood_replenishment_count: jnp.ndarray
    wood_depleted_before_pickaxe: jnp.ndarray
    wood_replenished_after_depletion: jnp.ndarray
    damage_event_count: jnp.ndarray
    damage_taken: jnp.ndarray
    first_iron_acquired_timestep: jnp.ndarray
    first_iron_pickaxe_crafted_timestep: jnp.ndarray
    first_diamond_acquired_timestep: jnp.ndarray
    termination_goal: jnp.ndarray
    termination_death: jnp.ndarray
    termination_timeout: jnp.ndarray


def init_episode_tracker(max_growing_plants: int) -> EpisodeTracker:
    """Create a tracker for one environment episode."""

    zero_i = jnp.asarray(0, dtype=jnp.int32)
    false = jnp.asarray(False)
    unseen = jnp.asarray(-1, dtype=jnp.int32)
    return EpisodeTracker(
        last_harvest_timestep=jnp.full(
            (max_growing_plants,), -1, dtype=jnp.int32
        ),
        episode_return=jnp.asarray(0.0, dtype=jnp.float32),
        episode_length=zero_i,
        violation_count=zero_i,
        harvest_count=zero_i,
        repeated_harvest_count=zero_i,
        goal_success=false,
        iron_acquired_count=zero_i,
        iron_pickaxe_crafted_count=zero_i,
        diamond_acquired_count=zero_i,
        wood_depletion_count=zero_i,
        wood_replenishment_count=zero_i,
        wood_depleted_before_pickaxe=false,
        wood_replenished_after_depletion=false,
        damage_event_count=zero_i,
        damage_taken=zero_i,
        first_iron_acquired_timestep=unseen,
        first_iron_pickaxe_crafted_timestep=unseen,
        first_diamond_acquired_timestep=unseen,
        termination_goal=false,
        termination_death=false,
        termination_timeout=false,
    )


def _first_event_timestep(previous, occurred, timestep):
    first_occurrence = jnp.logical_and(previous < 0, occurred)
    return jnp.where(first_occurrence, timestep, previous)


def update_episode_tracker(
    tracker: EpisodeTracker,
    reward,
    info,
    timestep,
    normal_regrowth_steps: int = NORMAL_PLANT_REGROWTH_STEPS,
):
    """Update one episode from explicit transition and termination events."""

    harvested = info["HackRL/plant_harvested"]
    plant_index = info["HackRL/harvested_plant_index"]
    safe_index = jnp.maximum(plant_index, 0)
    previous_harvest = tracker.last_harvest_timestep[safe_index]
    repeated_harvest = jnp.logical_and(
        harvested,
        jnp.logical_and(
            previous_harvest >= 0,
            timestep - previous_harvest < normal_regrowth_steps,
        ),
    )
    recorded_timestep = jnp.where(harvested, timestep, previous_harvest)
    last_harvest_timestep = tracker.last_harvest_timestep.at[safe_index].set(
        recorded_timestep
    )

    iron_acquired = info["HackRL/iron_acquired"]
    iron_pickaxe_crafted = info["HackRL/iron_pickaxe_crafted"]
    diamond_acquired = info["HackRL/diamond_acquired"]
    wood_depleted = info["HackRL/wood_depleted"]
    wood_replenished = info["HackRL/wood_replenished"]
    damage_taken = info["HackRL/damage_taken"].astype(jnp.int32)
    pickaxe_not_yet_crafted = jnp.logical_and(
        tracker.iron_pickaxe_crafted_count == 0,
        jnp.logical_not(iron_pickaxe_crafted),
    )
    wood_was_depleted = jnp.logical_or(
        tracker.wood_depletion_count > 0,
        wood_depleted,
    )

    updated = tracker.replace(
        last_harvest_timestep=last_harvest_timestep,
        episode_return=tracker.episode_return + reward,
        episode_length=tracker.episode_length + 1,
        violation_count=(
            tracker.violation_count
            + info["HackRL/violation"].astype(jnp.int32)
        ),
        harvest_count=tracker.harvest_count + harvested.astype(jnp.int32),
        repeated_harvest_count=(
            tracker.repeated_harvest_count
            + repeated_harvest.astype(jnp.int32)
        ),
        goal_success=jnp.logical_or(
            tracker.goal_success, info["HackRL/goal_success"]
        ),
        iron_acquired_count=(
            tracker.iron_acquired_count + iron_acquired.astype(jnp.int32)
        ),
        iron_pickaxe_crafted_count=(
            tracker.iron_pickaxe_crafted_count
            + iron_pickaxe_crafted.astype(jnp.int32)
        ),
        diamond_acquired_count=(
            tracker.diamond_acquired_count
            + diamond_acquired.astype(jnp.int32)
        ),
        wood_depletion_count=(
            tracker.wood_depletion_count + wood_depleted.astype(jnp.int32)
        ),
        wood_replenishment_count=(
            tracker.wood_replenishment_count
            + wood_replenished.astype(jnp.int32)
        ),
        wood_depleted_before_pickaxe=jnp.logical_or(
            tracker.wood_depleted_before_pickaxe,
            jnp.logical_and(wood_depleted, pickaxe_not_yet_crafted),
        ),
        wood_replenished_after_depletion=jnp.logical_or(
            tracker.wood_replenished_after_depletion,
            jnp.logical_and(wood_replenished, wood_was_depleted),
        ),
        damage_event_count=(
            tracker.damage_event_count + (damage_taken > 0).astype(jnp.int32)
        ),
        damage_taken=tracker.damage_taken + damage_taken,
        first_iron_acquired_timestep=_first_event_timestep(
            tracker.first_iron_acquired_timestep,
            iron_acquired,
            timestep,
        ),
        first_iron_pickaxe_crafted_timestep=_first_event_timestep(
            tracker.first_iron_pickaxe_crafted_timestep,
            iron_pickaxe_crafted,
            timestep,
        ),
        first_diamond_acquired_timestep=_first_event_timestep(
            tracker.first_diamond_acquired_timestep,
            diamond_acquired,
            timestep,
        ),
        termination_goal=jnp.logical_or(
            tracker.termination_goal, info["HackRL/termination_goal"]
        ),
        termination_death=jnp.logical_or(
            tracker.termination_death, info["HackRL/termination_death"]
        ),
        termination_timeout=jnp.logical_or(
            tracker.termination_timeout, info["HackRL/termination_timeout"]
        ),
    )
    return updated, repeated_harvest


def episode_summary(tracker: EpisodeTracker):
    """Return scalar episode facts suitable for logging."""

    return {
        "episode_return": tracker.episode_return,
        "episode_length": tracker.episode_length,
        "violation_count": tracker.violation_count,
        "harvest_count": tracker.harvest_count,
        "repeated_harvest_count": tracker.repeated_harvest_count,
        "goal_success": tracker.goal_success,
        "used_repeat_harvest": tracker.repeated_harvest_count > 0,
        "iron_acquired_count": tracker.iron_acquired_count,
        "iron_pickaxe_crafted_count": tracker.iron_pickaxe_crafted_count,
        "diamond_acquired_count": tracker.diamond_acquired_count,
        "wood_depletion_count": tracker.wood_depletion_count,
        "wood_replenishment_count": tracker.wood_replenishment_count,
        "wood_depleted_before_pickaxe": tracker.wood_depleted_before_pickaxe,
        "wood_replenished_after_depletion": (
            tracker.wood_replenished_after_depletion
        ),
        "damage_event_count": tracker.damage_event_count,
        "damage_taken": tracker.damage_taken,
        "first_iron_acquired_timestep": tracker.first_iron_acquired_timestep,
        "first_iron_pickaxe_crafted_timestep": (
            tracker.first_iron_pickaxe_crafted_timestep
        ),
        "first_diamond_acquired_timestep": (
            tracker.first_diamond_acquired_timestep
        ),
        "termination_goal": tracker.termination_goal,
        "termination_death": tracker.termination_death,
        "termination_timeout": tracker.termination_timeout,
    }
