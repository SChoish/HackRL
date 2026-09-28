import jax
import jax.numpy as jnp

from hackrl import EasyTask, HackRLEasySymbolicEnvNoAutoReset
from hackrl.metrics import (
    episode_summary,
    init_episode_tracker,
    update_episode_tracker,
)
from hackrl.scripted_paths import EASY_EXPLOIT_PATHS, EASY_NORMAL_PATHS


def _tracked_rollout(mutant, actions):
    env = HackRLEasySymbolicEnvNoAutoReset(EasyTask.L_E, mutant=mutant)
    params = env.default_params
    key = jax.random.PRNGKey(70)
    key, reset_key = jax.random.split(key)
    _, state = env.reset(reset_key, params)
    tracker = init_episode_tracker(env.static_env_params.max_growing_plants)
    repeated_events = []

    for action in actions:
        key, step_key = jax.random.split(key)
        _, state, reward, _, info = env.step(step_key, state, action, params)
        tracker, repeated = update_episode_tracker(
            tracker, reward, info, state.timestep
        )
        repeated_events.append(bool(repeated))

    return tracker, episode_summary(tracker), repeated_events


def _event_info(**overrides):
    info = {
        "HackRL/plant_harvested": jnp.asarray(False),
        "HackRL/harvested_plant_index": jnp.asarray(-1),
        "HackRL/violation": jnp.asarray(False),
        "HackRL/goal_success": jnp.asarray(False),
        "HackRL/iron_acquired": jnp.asarray(False),
        "HackRL/iron_pickaxe_crafted": jnp.asarray(False),
        "HackRL/diamond_acquired": jnp.asarray(False),
        "HackRL/wood_depleted": jnp.asarray(False),
        "HackRL/wood_replenished": jnp.asarray(False),
        "HackRL/damage_taken": jnp.asarray(0, dtype=jnp.int32),
        "HackRL/termination_goal": jnp.asarray(False),
        "HackRL/termination_death": jnp.asarray(False),
        "HackRL/termination_timeout": jnp.asarray(False),
    }
    info.update(overrides)
    return info


def test_l_e_violation_is_distinct_from_repeat_harvest_use():
    tracker, summary, repeated = _tracked_rollout(
        True, EASY_NORMAL_PATHS[EasyTask.L_E]
    )

    assert int(tracker.violation_count) == 2
    assert int(tracker.harvest_count) == 2
    assert int(tracker.repeated_harvest_count) == 0
    assert repeated == [False, False, False]
    assert bool(summary["goal_success"])
    assert not bool(summary["used_repeat_harvest"])


def test_l_e_exploit_path_records_one_repeat_harvest():
    tracker, summary, repeated = _tracked_rollout(
        True, EASY_EXPLOIT_PATHS[EasyTask.L_E]
    )

    assert int(tracker.violation_count) == 2
    assert int(tracker.harvest_count) == 2
    assert int(tracker.repeated_harvest_count) == 1
    assert repeated == [False, True]
    assert bool(summary["goal_success"])
    assert bool(summary["used_repeat_harvest"])


def test_l_e_fixed_normal_path_has_harvests_without_violations_or_repeats():
    tracker, summary, repeated = _tracked_rollout(
        False, EASY_NORMAL_PATHS[EasyTask.L_E]
    )

    assert int(tracker.violation_count) == 0
    assert int(tracker.harvest_count) == 2
    assert int(tracker.repeated_harvest_count) == 0
    assert repeated == [False, False, False]
    assert bool(summary["goal_success"])


def test_first_late_harvest_is_not_mislabeled_as_repeat():
    tracker = init_episode_tracker(10)
    info = _event_info(
        **{
            "HackRL/plant_harvested": jnp.asarray(True),
            "HackRL/harvested_plant_index": jnp.asarray(0),
            "HackRL/violation": jnp.asarray(True),
        }
    )

    tracker, repeated = jax.jit(update_episode_tracker)(
        tracker,
        jnp.asarray(0.0),
        info,
        jnp.asarray(700),
    )

    assert not bool(repeated)
    assert int(tracker.last_harvest_timestep[0]) == 700


def test_tracker_records_events_timestamps_damage_and_termination():
    tracker = init_episode_tracker(10)
    info = _event_info(
        **{
            "HackRL/iron_acquired": jnp.asarray(True),
            "HackRL/wood_depleted": jnp.asarray(True),
            "HackRL/damage_taken": jnp.asarray(2, dtype=jnp.int32),
            "HackRL/termination_death": jnp.asarray(True),
        }
    )

    tracker, _ = update_episode_tracker(
        tracker,
        jnp.asarray(0.0),
        info,
        jnp.asarray(7),
    )
    summary = episode_summary(tracker)

    assert int(tracker.iron_acquired_count) == 1
    assert int(tracker.first_iron_acquired_timestep) == 7
    assert bool(tracker.wood_depleted_before_pickaxe)
    assert int(tracker.damage_event_count) == 1
    assert int(tracker.damage_taken) == 2
    assert bool(summary["termination_death"])


def test_normal_pickaxe_craft_is_not_wood_depletion_before_pickaxe():
    tracker = init_episode_tracker(10)
    info = _event_info(
        **{
            "HackRL/iron_pickaxe_crafted": jnp.asarray(True),
            "HackRL/wood_depleted": jnp.asarray(True),
        }
    )

    tracker, _ = update_episode_tracker(
        tracker,
        jnp.asarray(0.0),
        info,
        jnp.asarray(3),
    )

    assert int(tracker.wood_depletion_count) == 1
    assert int(tracker.iron_pickaxe_crafted_count) == 1
    assert not bool(tracker.wood_depleted_before_pickaxe)
