import jax
import jax.numpy as jnp
import numpy as np

from hackrl.mine_expedition import (
    CAMP_POSITION,
    INITIAL_PHYSICAL_IRON,
    INITIAL_TARGET_MINERALS,
    IRON_SOURCE_POSITION,
    MAP_SIZE,
    MineExpeditionAction,
    MineExpeditionVariant,
    STORAGE_POSITION,
    TARGET_POSITION,
    UNPACK_POSITION,
    WORKBENCH_POSITION,
    WORLD_HORIZON,
    conservation_increased,
    duplicate_iron_total,
    indirect_exploit_succeeded,
    make_mine_expedition_state,
    mine_expedition_exploit_path,
    mine_expedition_normal_path,
    mine_expedition_state_invariants,
    mine_expedition_step,
    physical_iron_total,
    physical_target_total,
    same_action_effect_gap,
)


def _play(actions, variant, state=None):
    state = make_mine_expedition_state() if state is None else state
    totals = []
    violations = []
    for action in actions:
        after = mine_expedition_step(state, int(action), variant)
        assert bool(mine_expedition_state_invariants(after))
        totals.append(int(physical_iron_total(after)))
        violations.append(bool(conservation_increased(state, after)))
        state = after
    return state, totals, violations


def _prefix_to_storage():
    path = mine_expedition_normal_path()
    index = path.index(MineExpeditionAction.WITHDRAW_ONE)
    return path[:index]


def test_32x32_fixture_roles_are_distinct_floor_cells_with_reference_paths():
    state = make_mine_expedition_state()
    assert state.walkable.shape == (MAP_SIZE, MAP_SIZE) == (32, 32)
    roles = {
        CAMP_POSITION,
        STORAGE_POSITION,
        UNPACK_POSITION,
        WORKBENCH_POSITION,
        IRON_SOURCE_POSITION,
        TARGET_POSITION,
    }
    assert len(roles) == 6
    for row, column in roles:
        assert bool(state.walkable[row, column])
    assert int(physical_iron_total(state)) == INITIAL_PHYSICAL_IRON
    assert int(physical_target_total(state)) == INITIAL_TARGET_MINERALS
    assert len(mine_expedition_normal_path()) < WORLD_HORIZON
    assert len(mine_expedition_exploit_path()) < WORLD_HORIZON


def test_fixed_normal_path_crafts_mines_and_returns_without_violation():
    actions = mine_expedition_normal_path()
    for variant in MineExpeditionVariant:
        final, totals, violations = _play(actions, variant)
        assert totals == [INITIAL_PHYSICAL_IRON] * len(actions)
        assert not any(violations)
        assert int(final.source_iron) == 0
        assert int(final.pickaxe_iron) == 2
        assert int(final.pickaxe_duplicate_iron) == 0
        assert int(final.target_remaining) == 0
        assert int(final.returned_target) == 1
        assert int(physical_target_total(final)) == INITIAL_TARGET_MINERALS
        assert not bool(indirect_exploit_succeeded(final))


def test_mutant_exploit_uses_created_iron_in_downstream_tool():
    exploit = mine_expedition_exploit_path()
    normal = mine_expedition_normal_path()
    final, totals, violations = _play(exploit, MineExpeditionVariant.MUTANT)
    expected_violations = [
        action is MineExpeditionAction.REBUILD_EMPTY for action in exploit
    ]
    assert violations == expected_violations
    assert max(totals) == INITIAL_PHYSICAL_IRON + 1
    assert int(final.source_iron) == 1
    assert int(final.pickaxe_iron) == 2
    assert int(final.pickaxe_duplicate_iron) == 1
    assert int(duplicate_iron_total(final)) == 1
    assert int(final.returned_target) == 1
    assert bool(final.target_mined_with_duplicate_tool)
    assert bool(indirect_exploit_succeeded(final))
    assert len(exploit) < len(normal)


def test_same_exploit_actions_fail_in_fixed_kernel():
    actions = mine_expedition_exploit_path()
    fixed, totals, violations = _play(actions, MineExpeditionVariant.FIXED)
    mutant, _, _ = _play(actions, MineExpeditionVariant.MUTANT)
    assert totals == [INITIAL_PHYSICAL_IRON] * len(actions)
    assert not any(violations)
    assert int(fixed.source_iron) == 1
    assert int(fixed.pickaxe_iron) == 0
    assert int(fixed.target_remaining) == 1
    assert int(fixed.returned_target) == 0
    assert not bool(indirect_exploit_succeeded(fixed))
    assert int(mutant.returned_target) == 1


def test_empty_record_rebuild_is_a_negative_control():
    state, _, _ = _play(_prefix_to_storage(), MineExpeditionVariant.FIXED)
    state = mine_expedition_step(
        state, int(MineExpeditionAction.WITHDRAW_ONE), MineExpeditionVariant.MUTANT
    )
    assert int(state.anchor_iron) == 0
    before = state
    for action in (
        MineExpeditionAction.MAKE_RECORD,
        MineExpeditionAction.PACK_STORAGE,
        MineExpeditionAction.REBUILD_EMPTY,
    ):
        after = mine_expedition_step(state, int(action), MineExpeditionVariant.MUTANT)
        assert not bool(conservation_increased(state, after))
        state = after
    assert bool(state.record_present)
    assert int(state.record_iron_preview) == 0
    assert int(physical_iron_total(state)) == int(physical_iron_total(before))


def test_effect_oracle_fires_on_rebuild_and_not_sampled_control_actions():
    state, _, _ = _play(_prefix_to_storage(), MineExpeditionVariant.FIXED)
    for action in (
        MineExpeditionAction.MAKE_RECORD,
        MineExpeditionAction.PACK_STORAGE,
    ):
        state = mine_expedition_step(state, int(action), MineExpeditionVariant.FIXED)
    assert int(
        same_action_effect_gap(state, int(MineExpeditionAction.REBUILD_EMPTY))
    ) == 1
    for action in (
        MineExpeditionAction.NOOP,
        MineExpeditionAction.WITHDRAW_ONE,
        MineExpeditionAction.PLACE_PACKED,
        MineExpeditionAction.RETURN_TARGET,
    ):
        assert int(same_action_effect_gap(state, int(action))) == 0


def test_step_is_jittable_and_vmap_safe():
    state, _, _ = _play(_prefix_to_storage(), MineExpeditionVariant.FIXED)
    action = jnp.asarray(int(MineExpeditionAction.MAKE_RECORD))
    eager = mine_expedition_step(state, action, MineExpeditionVariant.MUTANT)
    compiled = jax.jit(
        lambda item, act: mine_expedition_step(
            item, act, MineExpeditionVariant.MUTANT
        )
    )(state, action)
    assert int(compiled.record_iron_preview) == int(eager.record_iron_preview) == 1
    batch = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), state)
    paired = jax.vmap(
        lambda item: mine_expedition_step(
            item, action, MineExpeditionVariant.FIXED
        )
    )(batch)
    assert np.asarray(paired.record_iron_preview).tolist() == [1, 1]
