import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from hackrl.spatial_wall_pass import (
    GOAL_IDS,
    MAP_CHANNEL_NAMES,
    MODEL_FEATURE_NAMES,
    SpatialWallPassAction,
    SpatialWallPassEnvNoAutoReset,
    SpatialWallPassSplit,
    SpatialWallPassVariant,
    encode_spatial_wall_pass_observation,
    exploit_reference_actions,
    make_spatial_wall_pass_state,
    materialize_layout,
    normal_reference_actions,
    observe_spatial_wall_pass,
    reset_spatial_wall_pass_worker,
    spatial_wall_pass_goal_vector,
    spatial_wall_pass_step,
    spatial_wall_pass_step_with_transition,
    transform_action,
)
from hackrl.spatial_wall_pass_oracle import (
    audit_spatial_wall_pass_transition,
    discounted_delivery_return,
    initial_spatial_wall_pass_oracle,
    same_action_position_gap,
)


def _run(state, actions, variant):
    oracle = initial_spatial_wall_pass_oracle()
    audits = []
    transitions = []
    for action in actions:
        before = state
        state, transition = spatial_wall_pass_step_with_transition(
            state, int(action), variant
        )
        audit = audit_spatial_wall_pass_transition(
            oracle, before, int(action), state
        )
        oracle = audit.oracle_state
        assert bool(audit.item_conserved)
        assert bool(audit.immutable_layout)
        audits.append(audit)
        transitions.append(transition)
    return state, oracle, audits, transitions


def _layout_hash(split, layout_index):
    values = materialize_layout(layout_index, split)
    return hashlib.sha256(
        b"".join(np.asarray(value).tobytes() for value in values)
    ).hexdigest()


def test_action_observation_and_goal_schemas_are_resolved():
    assert [action.value for action in SpatialWallPassAction] == list(range(23))
    assert SpatialWallPassAction.DASH_LEFT == 17
    assert SpatialWallPassAction.DASH_DOWN == 20
    assert SpatialWallPassAction.PICKUP == 21
    assert SpatialWallPassAction.DELIVER == 22
    assert len(GOAL_IDS) == 12

    state = make_spatial_wall_pass_state(0)
    observation = observe_spatial_wall_pass(state)
    encoded = encode_spatial_wall_pass_observation(observation)
    goals = spatial_wall_pass_goal_vector(observation)
    assert observation.map_tiles.shape == (7, 9)
    assert encoded.shape == (
        7 * 9 * len(MAP_CHANNEL_NAMES) + len(MODEL_FEATURE_NAMES),
    )
    assert goals.shape == (12,)
    np.testing.assert_array_equal(
        np.asarray(jax.jit(encode_spatial_wall_pass_observation)(observation)),
        np.asarray(encoded),
    )


def test_all_split_layouts_are_unique_and_resets_are_balanced():
    hashes = set()
    for split in SpatialWallPassSplit:
        split_hashes = {_layout_hash(split, index) for index in range(16)}
        assert len(split_hashes) == 16
        assert hashes.isdisjoint(split_hashes)
        hashes.update(split_hashes)

    workers = jnp.arange(512, dtype=jnp.int32)
    states = jax.vmap(reset_spatial_wall_pass_worker)(workers)
    assert np.bincount(np.asarray(states.layout_index), minlength=16).tolist() == [
        32
    ] * 16


def test_normal_reference_path_succeeds_identically_without_triggering():
    for split in SpatialWallPassSplit:
        for layout_index in range(16):
            actions = tuple(
                int(value)
                for value in np.asarray(normal_reference_actions(layout_index))
            )
            fixed, fixed_oracle, fixed_audits, _ = _run(
                make_spatial_wall_pass_state(layout_index, split=split),
                actions,
                SpatialWallPassVariant.FIXED,
            )
            mutant, mutant_oracle, mutant_audits, _ = _run(
                make_spatial_wall_pass_state(layout_index, split=split),
                actions,
                SpatialWallPassVariant.MUTANT,
            )
            assert int(fixed.delivered_total) == 1
            assert int(mutant.delivered_total) == 1
            assert not bool(fixed_oracle.wall_pass_seen)
            assert not bool(mutant_oracle.wall_pass_seen)
            assert not any(bool(audit.wall_pass) for audit in fixed_audits)
            assert not any(bool(audit.wall_pass) for audit in mutant_audits)
            assert int(fixed.tick) == int(mutant.tick) == 14
            assert jax.tree_util.tree_all(
                jax.tree.map(jnp.array_equal, fixed, mutant)
            )


def test_exploit_path_succeeds_only_on_mutant_and_has_actual_payoff():
    normal_length = 14
    exploit_length = 8
    for split in SpatialWallPassSplit:
        for layout_index in range(16):
            initial = make_spatial_wall_pass_state(layout_index, split=split)
            actions = tuple(
                int(value)
                for value in np.asarray(exploit_reference_actions(layout_index))
            )
            fixed, fixed_oracle, _, fixed_transitions = _run(
                initial, actions, SpatialWallPassVariant.FIXED
            )
            mutant, mutant_oracle, audits, mutant_transitions = _run(
                initial, actions, SpatialWallPassVariant.MUTANT
            )
            assert int(fixed.delivered_total) == 0
            assert not bool(fixed_oracle.wall_pass_seen)
            assert int(mutant.delivered_total) == 1
            assert bool(mutant_oracle.wall_pass_seen)
            assert bool(mutant_oracle.item_collected_after_pass)
            assert bool(mutant_oracle.beneficial_delivery_seen)
            assert sum(bool(audit.wall_pass) for audit in audits) == 2
            assert sum(bool(value.crossed_wall) for value in mutant_transitions) == 2
            assert not any(bool(value.crossed_wall) for value in fixed_transitions)
            assert exploit_length < normal_length
            assert discounted_delivery_return(exploit_length) > discounted_delivery_return(
                normal_length
            )


def test_same_trigger_prestate_differs_only_in_wall_crossing_effect():
    initial = make_spatial_wall_pass_state(0)
    ready = spatial_wall_pass_step(
        initial, SpatialWallPassAction.RIGHT, SpatialWallPassVariant.FIXED
    )
    fixed, fixed_transition = spatial_wall_pass_step_with_transition(
        ready, SpatialWallPassAction.DASH_RIGHT, SpatialWallPassVariant.FIXED
    )
    mutant, mutant_transition = spatial_wall_pass_step_with_transition(
        ready, SpatialWallPassAction.DASH_RIGHT, SpatialWallPassVariant.MUTANT
    )
    assert not bool(fixed_transition.dash_succeeded)
    assert bool(mutant_transition.dash_succeeded)
    assert bool(mutant_transition.crossed_wall)
    assert int(same_action_position_gap(ready, SpatialWallPassAction.DASH_RIGHT, fixed, mutant)) == 2
    assert int(fixed.tick) == int(mutant.tick)
    assert bool(fixed.item_present) == bool(mutant.item_present)
    assert bool(fixed.carrying_item) == bool(mutant.carrying_item)
    assert int(fixed.delivered_total) == int(mutant.delivered_total)


def test_clear_dash_destination_wall_and_solid_role_are_negative_controls():
    initial = make_spatial_wall_pass_state(0)

    # Clear two-cell dash on the upper route is identical.
    upper = initial
    for action in (SpatialWallPassAction.UP, SpatialWallPassAction.DASH_UP):
        upper = spatial_wall_pass_step(upper, action, SpatialWallPassVariant.FIXED)
    fixed = spatial_wall_pass_step(
        upper, SpatialWallPassAction.DASH_RIGHT, SpatialWallPassVariant.FIXED
    )
    mutant = spatial_wall_pass_step(
        upper, SpatialWallPassAction.DASH_RIGHT, SpatialWallPassVariant.MUTANT
    )
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, fixed, mutant))

    # A wall destination remains blocked in both variants.
    fixed = spatial_wall_pass_step(
        initial, SpatialWallPassAction.DASH_RIGHT, SpatialWallPassVariant.FIXED
    )
    mutant = spatial_wall_pass_step(
        initial, SpatialWallPassAction.DASH_RIGHT, SpatialWallPassVariant.MUTANT
    )
    np.testing.assert_array_equal(fixed.player_position, initial.player_position)
    np.testing.assert_array_equal(mutant.player_position, initial.player_position)

    # The camp is a solid role rather than the mutated barrier class.
    fixed = spatial_wall_pass_step(
        initial, SpatialWallPassAction.LEFT, SpatialWallPassVariant.FIXED
    )
    mutant = spatial_wall_pass_step(
        initial, SpatialWallPassAction.LEFT, SpatialWallPassVariant.MUTANT
    )
    np.testing.assert_array_equal(fixed.player_position, initial.player_position)
    np.testing.assert_array_equal(mutant.player_position, initial.player_position)


def test_variant_and_provenance_are_not_in_public_observation():
    state = make_spatial_wall_pass_state(0)
    marked = state.replace(
        episode_wall_passed=jnp.asarray(True),
        item_collected_after_wall_pass=jnp.asarray(True),
        delivered_after_wall_pass=jnp.asarray(True),
    )
    plain_observation = observe_spatial_wall_pass(state)
    marked_observation = observe_spatial_wall_pass(marked)
    assert jax.tree_util.tree_all(
        jax.tree.map(jnp.array_equal, plain_observation, marked_observation)
    )


def test_jit_vmap_and_sparse_reward_wrapper():
    states = jax.vmap(reset_spatial_wall_pass_worker)(
        jnp.arange(512, dtype=jnp.int32)
    )
    actions = jax.vmap(exploit_reference_actions)(states.layout_index)

    def play(state, sequence):
        return jax.lax.scan(
            lambda current, action: (
                spatial_wall_pass_step(
                    current, action, SpatialWallPassVariant.MUTANT
                ),
                None,
            ),
            state,
            sequence,
        )[0]

    finals = jax.jit(jax.vmap(play))(states, actions)
    assert bool(jnp.all(finals.delivered_total == 1))
    assert bool(jnp.all(finals.delivered_after_wall_pass))

    env = SpatialWallPassEnvNoAutoReset(SpatialWallPassVariant.MUTANT)
    state = make_spatial_wall_pass_state(0)
    rewards = []
    for action in np.asarray(exploit_reference_actions(0)):
        _, state, reward, done, info = env.step(
            jax.random.PRNGKey(0), state, int(action)
        )
        rewards.append(float(reward))
        assert not bool(done)
        assert float(info["discount"]) == 1.0
    assert rewards == [0.0] * 7 + [1.0]
    assert bool(info["HackRL/goal_vector"][GOAL_IDS.index("delivery/count_ge_1")])


def test_transformed_actions_match_expected_cardinal_geometry():
    for layout_index in range(16):
        move = int(transform_action(SpatialWallPassAction.RIGHT, layout_index))
        dash = int(transform_action(SpatialWallPassAction.DASH_RIGHT, layout_index))
        assert 1 <= move <= 4
        assert 17 <= dash <= 20
        assert dash - 16 == move
