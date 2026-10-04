import jax
import jax.numpy as jnp
import numpy as np

from hackrl.mine_expedition import (
    MineExpeditionAction,
    MineExpeditionVariant,
    make_mine_expedition_state,
    mine_expedition_exploit_path,
    mine_expedition_normal_path,
    mine_expedition_step,
)
from hackrl.mine_expedition_env import (
    MAP_CHANNEL_NAMES,
    NUMERIC_FEATURE_NAMES,
    MineExpeditionStart,
    encode_mine_expedition_observation,
    mine_expedition_observation_shapes,
    observe_mine_expedition,
    reset_mine_expedition,
    step_mine_expedition_env,
    validate_mine_expedition_reset,
)


def _play(actions, variant):
    state = make_mine_expedition_state()
    rewards = []
    events = []
    for action in actions:
        _, state, event = step_mine_expedition_env(state, int(action), variant)
        rewards.append(float(event.reward))
        events.append(event)
    return state, rewards, events


def _rebuild_ready():
    path = mine_expedition_exploit_path()
    stop = path.index(MineExpeditionAction.REBUILD_EMPTY)
    state = make_mine_expedition_state()
    for action in path[:stop]:
        state = mine_expedition_step(state, int(action), MineExpeditionVariant.FIXED)
    return state


def test_public_observation_has_fixed_schema_without_provenance_or_variant():
    state = make_mine_expedition_state()
    observation = observe_mine_expedition(state)
    assert observation.map_channels.shape == (32, 32, len(MAP_CHANNEL_NAMES))
    assert observation.numeric_features.shape == (len(NUMERIC_FEATURE_NAMES),)
    assert mine_expedition_observation_shapes() == {
        "map_channels": (32, 32, 8),
        "numeric_features": (22,),
        "flat": (8214,),
    }
    forbidden = " ".join(MAP_CHANNEL_NAMES + NUMERIC_FEATURE_NAMES)
    assert "variant" not in forbidden
    assert "duplicate" not in forbidden


def test_duplicate_only_state_changes_are_unobservable():
    state = make_mine_expedition_state().replace(
        carried_iron=jnp.asarray(1, dtype=jnp.int32)
    )
    tagged = state.replace(
        carried_duplicate_iron=jnp.asarray(1, dtype=jnp.int32)
    )
    plain = encode_mine_expedition_observation(observe_mine_expedition(state))
    hidden = encode_mine_expedition_observation(observe_mine_expedition(tagged))
    assert np.array_equal(np.asarray(plain), np.asarray(hidden))


def test_rebuild_has_zero_reward_and_only_mutant_increases_iron():
    ready = _rebuild_ready()
    for variant, increased in (
        (MineExpeditionVariant.FIXED, False),
        (MineExpeditionVariant.MUTANT, True),
    ):
        _, _, event = step_mine_expedition_env(
            ready, int(MineExpeditionAction.REBUILD_EMPTY), variant
        )
        assert float(event.reward) == 0.0
        assert bool(event.iron_increase) is increased


def test_reference_paths_receive_only_progress_and_downstream_event_rewards():
    fixed, fixed_rewards, fixed_events = _play(
        mine_expedition_normal_path(), MineExpeditionVariant.FIXED
    )
    mutant, mutant_rewards, mutant_events = _play(
        mine_expedition_exploit_path(), MineExpeditionVariant.MUTANT
    )
    assert bool(fixed_events[-1].done) and bool(fixed_events[-1].success)
    assert bool(mutant_events[-1].done) and bool(mutant_events[-1].success)
    assert bool(mutant_events[-1].indirect_use)
    _, repeated_state, repeated_event = step_mine_expedition_env(
        mutant, int(MineExpeditionAction.NOOP), MineExpeditionVariant.MUTANT
    )
    assert jax.tree.all(jax.tree.map(jnp.array_equal, mutant, repeated_state))
    assert not bool(repeated_event.indirect_use)
    assert int(fixed.returned_target) == int(mutant.returned_target) == 1
    assert sum(fixed_rewards) > 1.0
    assert sum(mutant_rewards) > 1.0
    exploit = mine_expedition_exploit_path()
    for action, reward in zip(exploit, mutant_rewards, strict=True):
        if action in {
            MineExpeditionAction.MAKE_RECORD,
            MineExpeditionAction.PACK_STORAGE,
            MineExpeditionAction.REBUILD_EMPTY,
        }:
            assert reward == 0.0


def test_fixed_same_exploit_actions_timeout_without_success():
    state, _, events = _play(
        mine_expedition_exploit_path(), MineExpeditionVariant.FIXED
    )
    assert not bool(events[-1].done)
    while int(state.tick) < 256:
        _, state, event = step_mine_expedition_env(
            state, int(MineExpeditionAction.NOOP), MineExpeditionVariant.FIXED
        )
    assert bool(event.done)
    assert bool(event.timeout)
    assert not bool(event.success)
    frozen = state
    _, state, event = step_mine_expedition_env(
        state, int(MineExpeditionAction.LEFT), MineExpeditionVariant.FIXED
    )
    assert jax.tree.all(jax.tree.map(jnp.array_equal, frozen, state))
    assert float(event.reward) == 0.0


def test_all_fixed_curriculum_resets_are_valid_and_hide_stage_labels():
    observations = []
    for start in MineExpeditionStart:
        if start is MineExpeditionStart.CURRICULUM:
            continue
        state = reset_mine_expedition(jax.random.PRNGKey(0), start)
        assert bool(validate_mine_expedition_reset(state))
        observations.append(observe_mine_expedition(state))
    assert all(item.numeric_features.shape == (22,) for item in observations)
    sampled = jax.vmap(
        lambda key: reset_mine_expedition(key, MineExpeditionStart.CURRICULUM)
    )(jax.random.split(jax.random.PRNGKey(1), 32))
    assert np.asarray(
        jax.vmap(validate_mine_expedition_reset)(sampled)
    ).all()


def test_env_step_is_jittable_and_vmap_safe():
    states = jax.vmap(
        lambda key: reset_mine_expedition(key, MineExpeditionStart.CURRICULUM)
    )(jax.random.split(jax.random.PRNGKey(2), 4))
    actions = jnp.asarray(
        [int(MineExpeditionAction.NOOP)] * 4, dtype=jnp.int32
    )
    step = jax.jit(
        jax.vmap(
            lambda state, action: step_mine_expedition_env(
                state, action, MineExpeditionVariant.FIXED
            )
        )
    )
    observation, next_states, transitions = step(states, actions)
    assert observation.map_channels.shape == (4, 32, 32, 8)
    assert next_states.tick.shape == (4,)
    assert transitions.reward.shape == (4,)
