import jax

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
    info = {
        "HackRL/plant_harvested": jax.numpy.asarray(True),
        "HackRL/harvested_plant_index": jax.numpy.asarray(0),
        "HackRL/violation": jax.numpy.asarray(True),
        "HackRL/goal_success": jax.numpy.asarray(False),
    }

    tracker, repeated = jax.jit(update_episode_tracker)(
        tracker,
        jax.numpy.asarray(0.0),
        info,
        jax.numpy.asarray(700),
    )

    assert not bool(repeated)
    assert int(tracker.last_harvest_timestep[0]) == 700
