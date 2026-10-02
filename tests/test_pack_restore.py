import jax
import jax.numpy as jnp
import numpy as np

from hackrl.pack_restore import (
    GOAL_IDS,
    MAP_CHANNEL_NAMES,
    MODEL_FEATURE_NAMES,
    INITIAL_GRAIN_TOTAL,
    PackRestoreAction,
    PackRestorePhase,
    PackRestoreSplit,
    PackRestoreStart,
    PackRestoreVariant,
    _VIEW_COLUMNS,
    _VIEW_ROWS,
    _signed_count,
    conservation_increased,
    encode_pack_restore_observation,
    make_pack_restore_state,
    observe_pack_restore,
    pack_restore_goal_vector,
    pack_restore_step,
    physical_grain_total,
    same_action_effect_gap,
)


def _play(state, actions, variant):
    totals = []
    violations = []
    for action in actions:
        after = pack_restore_step(state, int(action), variant)
        totals.append(int(physical_grain_total(after)))
        violations.append(bool(conservation_increased(state, after)))
        state = after
    return state, totals, violations


def _normal_loaded():
    return (
        PackRestoreAction.WITHDRAW_ONE,
        PackRestoreAction.RIGHT,
        PackRestoreAction.DO,
        PackRestoreAction.DO,
        PackRestoreAction.DOWN,
        PackRestoreAction.DELIVER,
    )


def _normal_empty():
    return (
        PackRestoreAction.RIGHT,
        PackRestoreAction.DO,
        PackRestoreAction.DO,
        PackRestoreAction.DOWN,
        PackRestoreAction.DELIVER,
    )


def _rebuild_prefix():
    return (
        PackRestoreAction.MAKE_RECORD,
        PackRestoreAction.PACK_STORAGE,
        PackRestoreAction.REBUILD_EMPTY,
        PackRestoreAction.WITHDRAW_ONE,
        PackRestoreAction.RIGHT,
        PackRestoreAction.DO,
        PackRestoreAction.DO,
        PackRestoreAction.DOWN,
        PackRestoreAction.DELIVER,
    )


def test_reset_conserves_three_grains_and_opposes_the_storage_goals():
    for phase in (PackRestorePhase.EMPTY, PackRestorePhase.LOADED):
        state = make_pack_restore_state(
            0, int(phase), start=PackRestoreStart.PATH_CHECK
        )
        assert int(physical_grain_total(state)) == INITIAL_GRAIN_TOTAL
        goals = np.asarray(pack_restore_goal_vector(state))
        assert goals.shape == (len(GOAL_IDS),)
        empty = bool(goals[GOAL_IDS.index("facility/storage_empty")])
        loaded = bool(goals[GOAL_IDS.index("facility/storage_has_raw_material")])
        assert empty != loaded
        assert not bool(goals[GOAL_IDS.index("delivery/count_ge_3")])


def test_normal_path_delivers_three_without_creating_grain():
    loaded = make_pack_restore_state(
        0, int(PackRestorePhase.LOADED), start=PackRestoreStart.PATH_CHECK
    )
    final, totals, violations = _play(loaded, _normal_loaded(), PackRestoreVariant.FIXED)
    assert totals == [INITIAL_GRAIN_TOTAL] * len(_normal_loaded())
    assert not any(violations)
    assert int(final.delivered_total) == 3
    assert bool(pack_restore_goal_vector(final)[GOAL_IDS.index("delivery/count_ge_3")])
    assert int(final.tick) == 6

    empty = make_pack_restore_state(
        0, int(PackRestorePhase.EMPTY), start=PackRestoreStart.PATH_CHECK
    )
    final, totals, violations = _play(empty, _normal_empty(), PackRestoreVariant.MUTANT)
    assert totals == [INITIAL_GRAIN_TOTAL] * len(_normal_empty())
    assert not any(violations)
    assert int(final.delivered_total) == 3
    assert int(final.tick) == 5


def test_mutant_rebuild_creates_grain_and_the_same_actions_deliver_the_goal():
    start = make_pack_restore_state(
        0, int(PackRestorePhase.LOADED), start=PackRestoreStart.PATH_CHECK
    )
    actions = _rebuild_prefix()
    mutant, mutant_totals, mutant_violations = _play(
        start, actions, PackRestoreVariant.MUTANT
    )
    fixed, fixed_totals, fixed_violations = _play(
        start, actions, PackRestoreVariant.FIXED
    )
    assert mutant_violations == [False, False, True] + [False] * (len(actions) - 3)
    assert not any(fixed_violations)
    assert mutant_totals[2] == INITIAL_GRAIN_TOTAL + 1
    assert fixed_totals == [INITIAL_GRAIN_TOTAL] * len(actions)
    assert int(mutant.delivered_total) == 3
    assert int(mutant.packed_grain) == 1
    assert int(physical_grain_total(mutant)) == 4
    assert bool(pack_restore_goal_vector(mutant)[-1])
    assert int(fixed.delivered_total) == 2
    assert not bool(pack_restore_goal_vector(fixed)[-1])
    assert len(actions) > len(_normal_loaded())


def test_effect_oracle_is_zero_except_on_a_positive_record_rebuild():
    start = make_pack_restore_state(
        0, int(PackRestorePhase.LOADED), start=PackRestoreStart.PATH_CHECK
    )
    ready = start
    for action in (PackRestoreAction.MAKE_RECORD, PackRestoreAction.PACK_STORAGE):
        ready = pack_restore_step(ready, int(action), PackRestoreVariant.FIXED)
    gap = same_action_effect_gap(ready, int(PackRestoreAction.REBUILD_EMPTY))
    assert int(gap) == 1
    for action in (
        PackRestoreAction.DO,
        PackRestoreAction.STORE_ONE,
        PackRestoreAction.WITHDRAW_ONE,
        PackRestoreAction.DELIVER,
        PackRestoreAction.PLACE_PACKED,
    ):
        assert int(same_action_effect_gap(start, int(action))) == 0


def test_negative_controls_do_not_create_grain():
    start = make_pack_restore_state(
        0, int(PackRestorePhase.EMPTY), start=PackRestoreStart.PATH_CHECK
    )
    recorded = pack_restore_step(
        start, int(PackRestoreAction.MAKE_RECORD), PackRestoreVariant.MUTANT
    )
    assert int(recorded.record_grain_preview) == 0
    packed_empty = pack_restore_step(
        recorded, int(PackRestoreAction.PACK_STORAGE), PackRestoreVariant.MUTANT
    )
    assert int(packed_empty.empty_frames) == 2
    assert not bool(packed_empty.anchor_present)
    rebuilt = pack_restore_step(
        packed_empty, int(PackRestoreAction.REBUILD_EMPTY), PackRestoreVariant.MUTANT
    )
    assert int(rebuilt.anchor_grain) == 0
    assert int(physical_grain_total(rebuilt)) == INITIAL_GRAIN_TOTAL

    occupied = rebuilt.replace(unpack_present=jnp.asarray(True), unpack_grain=jnp.asarray(0))
    occupied = occupied.replace(packed_present=jnp.asarray(True), packed_grain=jnp.asarray(1))
    rejected = pack_restore_step(
        occupied, int(PackRestoreAction.PLACE_PACKED), PackRestoreVariant.FIXED
    )
    assert bool(rejected.packed_present)
    assert int(rejected.unpack_grain) == 0
    assert int(physical_grain_total(rejected)) == int(physical_grain_total(occupied))


SOURCE_GROWTH_PERIOD = 8


def _growth_state(phase):
    return make_pack_restore_state(
        0,
        int(phase),
        start=PackRestoreStart.PATH_CHECK,
        source_growth_period=SOURCE_GROWTH_PERIOD,
    )


def _local_duplicate_then_deliver():
    cycle = (
        PackRestoreAction.PACK_STORAGE,
        PackRestoreAction.REBUILD_EMPTY,
        PackRestoreAction.WITHDRAW_ONE,
    )
    return (
        (PackRestoreAction.MAKE_RECORD,)
        + cycle * 3
        + (PackRestoreAction.DOWN, PackRestoreAction.DELIVER)
    )


def _normal_growth_path(phase):
    """Earliest fixed collection: eight ticks of aging before each of two harvests."""

    opening = [PackRestoreAction.NOOP] * 8
    if phase is PackRestorePhase.LOADED:
        opening[0] = PackRestoreAction.WITHDRAW_ONE
    opening[1] = PackRestoreAction.RIGHT
    return tuple(
        opening
        + [PackRestoreAction.DO]
        + [PackRestoreAction.NOOP] * 8
        + [PackRestoreAction.DO, PackRestoreAction.DOWN, PackRestoreAction.DELIVER]
    )


def test_source_growth_makes_duplication_faster_than_the_normal_optimum():
    loaded_actions = _local_duplicate_then_deliver()
    empty_actions = (PackRestoreAction.STORE_ONE,) + loaded_actions
    loaded, loaded_totals, loaded_violations = _play(
        _growth_state(PackRestorePhase.LOADED),
        loaded_actions,
        PackRestoreVariant.MUTANT,
    )
    empty, _, empty_violations = _play(
        _growth_state(PackRestorePhase.EMPTY),
        empty_actions,
        PackRestoreVariant.MUTANT,
    )
    fixed_loaded, _, fixed_violations = _play(
        _growth_state(PackRestorePhase.LOADED),
        loaded_actions,
        PackRestoreVariant.FIXED,
    )
    assert loaded_violations == [
        action is PackRestoreAction.REBUILD_EMPTY for action in loaded_actions
    ]
    assert empty_violations == [
        action is PackRestoreAction.REBUILD_EMPTY for action in empty_actions
    ]
    assert not any(fixed_violations)
    assert int(loaded.delivered_total) == 3
    assert int(empty.delivered_total) == 3
    assert int(fixed_loaded.delivered_total) == 0
    assert bool(pack_restore_goal_vector(loaded)[-1])
    assert not bool(pack_restore_goal_vector(fixed_loaded)[-1])
    assert loaded_totals[-1] == INITIAL_GRAIN_TOTAL + 3
    facing = _growth_state(PackRestorePhase.LOADED)
    for _ in range(7):
        facing = pack_restore_step(
            facing, int(PackRestoreAction.RIGHT), PackRestoreVariant.FIXED
        )
    early = pack_restore_step(facing, int(PackRestoreAction.DO), PackRestoreVariant.FIXED)
    assert int(early.source_grain) == 2
    ripe = pack_restore_step(facing, int(PackRestoreAction.NOOP), PackRestoreVariant.FIXED)
    harvested = pack_restore_step(ripe, int(PackRestoreAction.DO), PackRestoreVariant.FIXED)
    assert int(harvested.source_grain) == 1
    positions = np.argwhere(np.asarray(_growth_state(PackRestorePhase.LOADED).walkable))
    source = np.asarray(loaded.source_position)
    delivery = np.asarray(loaded.delivery_position)
    harvest_cells = [
        cell for cell in positions if np.abs(cell - source).sum() == 1
    ]
    assert harvest_cells
    assert all(np.abs(cell - delivery).sum() != 1 for cell in harvest_cells)
    for phase in (PackRestorePhase.LOADED, PackRestorePhase.EMPTY):
        normal, totals, violations = _play(
            _growth_state(phase), _normal_growth_path(phase), PackRestoreVariant.FIXED
        )
        assert len(_normal_growth_path(phase)) == 20
        assert totals == [INITIAL_GRAIN_TOTAL] * 20
        assert not any(violations)
        assert int(normal.delivered_total) == 3
    assert len(loaded_actions) < 20
    assert len(empty_actions) < 20


def test_common_setup_reaches_the_path_check_pose_without_changing_grain():
    for layout in (0, 1, 4):
        for split in (PackRestoreSplit.TRAIN, PackRestoreSplit.VALIDATION):
            reference = make_pack_restore_state(
                layout,
                int(PackRestorePhase.LOADED),
                split=split,
                start=PackRestoreStart.PATH_CHECK,
                source_growth_period=8,
            )
            arrived = make_pack_restore_state(
                layout,
                int(PackRestorePhase.LOADED),
                split=split,
                start=PackRestoreStart.COMMON_SETUP,
                source_growth_period=8,
            )
            assert np.asarray(arrived.player_position).tolist() == np.asarray(
                reference.player_position
            ).tolist()
            assert int(arrived.player_direction) == int(reference.player_direction)
            assert int(arrived.tick) == 10
            assert int(arrived.carried_grain) == 0
            assert int(arrived.anchor_grain) == 1
            assert int(arrived.source_grain) == 2
            assert int(arrived.delivered_total) == 0
            assert not bool(arrived.record_present)
            assert int(arrived.source_age) == 8


def test_model_features_keep_the_delivery_thresholds_distinct():
    state = make_pack_restore_state(
        0, int(PackRestorePhase.LOADED), start=PackRestoreStart.PATH_CHECK
    )
    encoded = np.asarray(encode_pack_restore_observation(observe_pack_restore(state)))
    assert encoded.shape[-1] == _VIEW_ROWS * _VIEW_COLUMNS * len(MAP_CHANNEL_NAMES) + len(
        MODEL_FEATURE_NAMES
    )
    two = float(_signed_count(2))
    three = float(_signed_count(3))
    assert two != three
    hidden = state.replace(anchor_present=jnp.asarray(False), anchor_grain=jnp.asarray(1))
    goals = np.asarray(pack_restore_goal_vector(hidden))
    assert not bool(goals[GOAL_IDS.index("facility/storage_empty")])
    assert not bool(goals[GOAL_IDS.index("facility/storage_has_raw_material")])
    state = make_pack_restore_state(
        0, int(PackRestorePhase.LOADED), start=PackRestoreStart.PATH_CHECK
    )
    action = jnp.asarray(int(PackRestoreAction.WITHDRAW_ONE))
    eager = pack_restore_step(state, action, PackRestoreVariant.MUTANT)
    compiled = jax.jit(
        lambda item, act: pack_restore_step(item, act, PackRestoreVariant.MUTANT)
    )(state, action)
    assert int(compiled.carried_grain) == int(eager.carried_grain) == 1
    paired = jax.vmap(
        lambda item: pack_restore_step(item, action, PackRestoreVariant.FIXED)
    )(jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), state))
    assert np.asarray(paired.anchor_grain).tolist() == [0, 0]
