import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hackrl.tick_claim import (
    GOAL_IDS,
    GROWTH_WAIT_TICKS,
    MAP_CHANNEL_NAMES,
    MODEL_FEATURE_NAMES,
    TILE_CROP_RIPE,
    TILE_CROP_UNRIPE,
    TickClaimAction,
    TickClaimEnvNoAutoReset,
    TickClaimPhase,
    TickClaimSplit,
    TickClaimStart,
    TickClaimTransitionSnapshot,
    TickClaimVariant,
    encode_tick_claim_observation,
    make_tick_claim_state,
    materialize_layout,
    observe_tick_claim,
    reset_tick_claim_worker,
    tick_claim_goal,
    tick_claim_goal_vector,
    tick_claim_setup_prefix,
    tick_claim_step,
    tick_claim_step_with_snapshot,
    transform_direction,
)
from hackrl.tick_claim_oracle import (
    audit_tick_claim_transition,
    compare_tick_claim_effect,
    initial_tick_claim_oracle,
    tick_claim_beneficial_use,
)


def _transformed_action(layout_index, action):
    return int(transform_direction(action, layout_index))


def _setup_prefix(layout_index):
    return tuple(
        int(action) for action in np.asarray(tick_claim_setup_prefix(layout_index))
    )


def _growth_actions(layout_index, relocate_from_alternate_pose=False):
    if not relocate_from_alternate_pose or layout_index % 2 == 0:
        return (TickClaimAction.NOOP,) * GROWTH_WAIT_TICKS
    canonical = (
        TickClaimAction.RIGHT,
        TickClaimAction.DOWN,
        TickClaimAction.DOWN,
        TickClaimAction.DOWN,
        TickClaimAction.LEFT,
        TickClaimAction.LEFT,
        TickClaimAction.UP,
        TickClaimAction.UP,
    )
    return tuple(_transformed_action(layout_index, action) for action in canonical)


def _normal_core(layout_index, *, natural=False):
    return (
        (TickClaimAction.DO,)
        + _growth_actions(layout_index, natural)
        + (TickClaimAction.DO,)
        + (TickClaimAction.NOOP,) * GROWTH_WAIT_TICKS
        + (TickClaimAction.DO,)
        + (_transformed_action(layout_index, TickClaimAction.DOWN),)
        + (TickClaimAction.DELIVER,)
    )


def _exploit_core(layout_index, delay, *, natural=False):
    return (
        (TickClaimAction.ARM_HARVEST,)
        + (TickClaimAction.NOOP,) * (delay - 1)
        + (TickClaimAction.DO,)
        + _growth_actions(layout_index, natural)
        + (TickClaimAction.DO,)
        + (_transformed_action(layout_index, TickClaimAction.DOWN),)
        + (TickClaimAction.DELIVER,)
    )


def _run(state, actions, variant):
    oracle = initial_tick_claim_oracle()
    violations = []
    delivery_checks = []
    for action in actions:
        before = state
        state, snapshot = tick_claim_step_with_snapshot(state, action, variant)
        audit = audit_tick_claim_transition(
            oracle, before, action, state, snapshot
        )
        oracle = audit.oracle_state
        assert bool(audit.settlement_conserved)
        violations.append(bool(audit.violation))
        delivery_checks.append(bool(audit.delivery_conserved))
    return state, oracle, violations, delivery_checks


def _layout_hash(split, layout_index):
    walkable, crop, device, delivery = materialize_layout(layout_index, split)
    payload = b"".join(
        np.asarray(value).tobytes()
        for value in (walkable, crop, device, delivery)
    )
    return hashlib.sha256(payload).hexdigest()


def test_action_and_observation_schemas_are_resolved():
    assert [action.value for action in TickClaimAction] == list(range(20))
    assert TickClaimAction.ARM_HARVEST == 17
    assert TickClaimAction.CANCEL_HARVEST == 18
    assert TickClaimAction.DELIVER == 19

    state = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    observation = observe_tick_claim(state)
    encoded = encode_tick_claim_observation(observation)
    assert observation.map_tiles.shape == (7, 9)
    assert encoded.shape == (7 * 9 * len(MAP_CHANNEL_NAMES) + len(MODEL_FEATURE_NAMES),)
    np.testing.assert_array_equal(
        np.asarray(jax.jit(encode_tick_claim_observation)(observation)),
        np.asarray(encoded),
    )


def test_all_materialized_layouts_are_unique_and_normal_prefix_is_traversable():
    all_hashes = set()
    for split in TickClaimSplit:
        split_hashes = {_layout_hash(split, index) for index in range(16)}
        assert len(split_hashes) == 16
        assert all_hashes.isdisjoint(split_hashes)
        all_hashes.update(split_hashes)

        for layout_index in range(16):
            state = make_tick_claim_state(
                layout_index,
                TickClaimPhase.RIPE,
                split=split,
                start=TickClaimStart.NATURAL,
            )
            state, _, _, _ = _run(
                state,
                _setup_prefix(layout_index),
                TickClaimVariant.FIXED,
            )
            harvested = tick_claim_step(
                state, TickClaimAction.DO, TickClaimVariant.FIXED
            )
            armed = tick_claim_step(
                state, TickClaimAction.ARM_HARVEST, TickClaimVariant.FIXED
            )
            assert int(harvested.grain) == 1
            assert bool(armed.reservation_present)


def test_common_setup_is_the_exact_fixed_prefix_result_for_every_state():
    for split in TickClaimSplit:
        for layout_index in range(16):
            for phase in TickClaimPhase:
                natural = make_tick_claim_state(
                    layout_index,
                    phase,
                    split=split,
                    start=TickClaimStart.NATURAL,
                )
                expected, _, _, _ = _run(
                    natural,
                    _setup_prefix(layout_index),
                    TickClaimVariant.FIXED,
                )
                common = make_tick_claim_state(
                    layout_index,
                    phase,
                    split=split,
                    start=TickClaimStart.COMMON_SETUP,
                )
                assert jax.tree_util.tree_all(
                    jax.tree.map(jnp.array_equal, expected, common)
                )
                assert int(common.tick) == 4
                assert int(common.grain) == 0
                assert not bool(common.reservation_present)
                expected_age = (
                    GROWTH_WAIT_TICKS
                    if phase is TickClaimPhase.RIPE
                    else len(_setup_prefix(layout_index))
                )
                assert int(common.crop_age) == expected_age

    synthetic = make_tick_claim_state(
        0,
        TickClaimPhase.UNRIPE,
        start=TickClaimStart.PATH_CHECK,
    )
    assert int(synthetic.tick) == 0
    assert int(synthetic.crop_age) == 0


def test_initial_worker_assignment_balances_layouts_and_opposed_phase_goals():
    workers = jnp.arange(512, dtype=jnp.int32)
    states = jax.vmap(reset_tick_claim_worker)(workers)
    assert np.bincount(np.asarray(states.layout_index), minlength=16).tolist() == [32] * 16
    assert np.bincount(np.asarray(states.initial_phase), minlength=2).tolist() == [256, 256]
    pairs = np.asarray(states.layout_index) * 2 + np.asarray(states.initial_phase)
    assert np.bincount(pairs, minlength=32).tolist() == [16] * 32

    observations = jax.vmap(observe_tick_claim)(states)
    goals = jax.vmap(tick_claim_goal_vector)(observations)
    ripe_index = GOAL_IDS.index("visible/crop_ripe")
    unripe_index = GOAL_IDS.index("visible/crop_unripe")
    assert bool(jnp.any(goals[:, ripe_index]))
    assert bool(jnp.any(goals[:, unripe_index]))
    assert bool(jnp.all(jnp.logical_xor(goals[:, ripe_index], goals[:, unripe_index])))


@pytest.mark.parametrize("layout_index", range(16))
def test_normal_path_is_identical_and_conserving(layout_index):
    initial = make_tick_claim_state(
        layout_index, TickClaimPhase.RIPE, start=TickClaimStart.NATURAL
    )
    actions = _setup_prefix(layout_index) + _normal_core(layout_index, natural=True)
    fixed, fixed_oracle, fixed_violations, fixed_delivery = _run(
        initial, actions, TickClaimVariant.FIXED
    )
    mutant, mutant_oracle, mutant_violations, mutant_delivery = _run(
        initial, actions, TickClaimVariant.MUTANT
    )

    assert int(fixed.delivered_total) == 3
    assert int(mutant.delivered_total) == 3
    assert not any(fixed_violations)
    assert not any(mutant_violations)
    assert not bool(fixed_oracle.violation_seen)
    assert not bool(mutant_oracle.violation_seen)
    assert all(fixed_delivery)
    assert all(mutant_delivery)
    assert int(fixed.tick) == len(actions)
    assert int(mutant.tick) == len(actions)
    assert len(_normal_core(layout_index)) == 21


@pytest.mark.parametrize("layout_index", range(16))
def test_exploit_path_has_violation_effect_and_actual_payoff(layout_index):
    initial = make_tick_claim_state(
        layout_index, TickClaimPhase.RIPE, start=TickClaimStart.NATURAL
    )
    prefix = _setup_prefix(layout_index)
    prepared, _, _, _ = _run(initial, prefix, TickClaimVariant.FIXED)
    delay = int(prepared.reservation_delay)
    exploit = _exploit_core(layout_index, delay, natural=True)
    fixed, fixed_oracle, fixed_violations, fixed_delivery = _run(
        prepared, exploit, TickClaimVariant.FIXED
    )
    mutant, mutant_oracle, mutant_violations, mutant_delivery = _run(
        prepared, exploit, TickClaimVariant.MUTANT
    )

    assert int(fixed.delivered_total) == 2
    assert int(mutant.delivered_total) == 3
    assert not any(fixed_violations)
    assert any(mutant_violations)
    assert not bool(fixed_oracle.violation_seen)
    assert bool(mutant_oracle.violation_seen)
    assert all(fixed_delivery)
    assert all(mutant_delivery)
    assert bool(tick_claim_beneficial_use(fixed, mutant, mutant_oracle.violation_seen))

    # Fixed needs three one-unit crop cycles: 3 harvest actions, 16 subsequent
    # growth ticks, one move, and one delivery = an exact 21-step lower bound.
    fixed_exact_core_steps = 3 + 2 * GROWTH_WAIT_TICKS + 1 + 1
    assert fixed_exact_core_steps == len(_normal_core(layout_index)) == 21
    assert len(prefix) == 4
    assert len(exploit) == delay + 12
    assert len(prefix) + fixed_exact_core_steps == 25
    assert len(exploit) < fixed_exact_core_steps
    assert fixed_exact_core_steps - len(exploit) == 9 - delay


def test_same_trigger_prestate_differs_only_at_scheduled_consumed_check():
    state = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    state, _, _, _ = _run(
        state,
        (TickClaimAction.ARM_HARVEST, TickClaimAction.NOOP),
        TickClaimVariant.FIXED,
    )
    fixed = tick_claim_step(state, TickClaimAction.DO, TickClaimVariant.FIXED)
    mutant = tick_claim_step(state, TickClaimAction.DO, TickClaimVariant.MUTANT)
    effect = compare_tick_claim_effect(
        state, TickClaimAction.DO, fixed, mutant
    )

    assert int(fixed.grain) == 1
    assert int(mutant.grain) == 2
    assert int(effect.physical_surplus) == 1
    assert int(effect.fixed_created_amount) == 1
    assert int(effect.mutant_created_amount) == 2
    assert bool(effect.effect)
    assert int(fixed.tick) == int(mutant.tick)
    assert int(fixed.crop_cycle_id) == int(mutant.crop_cycle_id)
    assert bool(fixed.crop_ripe) == bool(mutant.crop_ripe) is False
    assert not bool(fixed.reservation_present)
    assert not bool(mutant.reservation_present)


def _due_state():
    state = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    state = tick_claim_step(
        state, TickClaimAction.ARM_HARVEST, TickClaimVariant.FIXED
    )
    return tick_claim_step(state, TickClaimAction.NOOP, TickClaimVariant.FIXED)


@pytest.mark.parametrize("variant", tuple(TickClaimVariant))
def test_scheduled_settlement_and_delivery_same_tick_is_conserved(variant):
    state = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    oracle = initial_tick_claim_oracle()
    last_audit = None
    for action in (
        TickClaimAction.ARM_HARVEST,
        TickClaimAction.DOWN,
        TickClaimAction.DELIVER,
    ):
        before = state
        state, snapshot = tick_claim_step_with_snapshot(state, action, variant)
        last_audit = audit_tick_claim_transition(
            oracle, before, action, state, snapshot
        )
        oracle = last_audit.oracle_state

    assert int(state.grain) == 0
    assert int(state.delivered_total) == 1
    assert last_audit is not None
    assert int(last_audit.created_amount) == 1
    assert bool(last_audit.settlement_conserved)
    assert bool(last_audit.delivery_conserved)
    assert not bool(last_audit.violation)
    assert not bool(oracle.violation_seen)


def test_scheduled_settlement_outside_delivery_range_does_not_transfer():
    state = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    oracle = initial_tick_claim_oracle()
    last_audit = None
    for action in (
        TickClaimAction.ARM_HARVEST,
        TickClaimAction.NOOP,
        TickClaimAction.DELIVER,
    ):
        before = state
        state, snapshot = tick_claim_step_with_snapshot(
            state, action, TickClaimVariant.FIXED
        )
        last_audit = audit_tick_claim_transition(
            oracle, before, action, state, snapshot
        )
        oracle = last_audit.oracle_state

    assert int(state.grain) == 1
    assert int(state.delivered_total) == 0
    assert last_audit is not None
    assert int(last_audit.created_amount) == 1
    assert bool(last_audit.settlement_conserved)
    assert bool(last_audit.delivery_conserved)


def test_oracle_separates_production_free_delivery_from_loss_and_overdelivery():
    carrying = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    ).replace(
        player_position=jnp.asarray((10, 8), dtype=jnp.int32),
        grain=jnp.asarray(2, dtype=jnp.int32),
    )
    delivered, snapshot = tick_claim_step_with_snapshot(
        carrying, TickClaimAction.DELIVER, TickClaimVariant.FIXED
    )
    valid = audit_tick_claim_transition(
        initial_tick_claim_oracle(),
        carrying,
        TickClaimAction.DELIVER,
        delivered,
        snapshot,
    )
    assert int(valid.created_amount) == 0
    assert bool(valid.settlement_conserved)
    assert bool(valid.delivery_conserved)

    lost = delivered.replace(delivered_total=jnp.asarray(1, dtype=jnp.int32))
    overdelivered = delivered.replace(
        delivered_total=jnp.asarray(3, dtype=jnp.int32)
    )
    for corrupted in (lost, overdelivered):
        audit = audit_tick_claim_transition(
            initial_tick_claim_oracle(),
            carrying,
            TickClaimAction.DELIVER,
            corrupted,
            snapshot,
        )
        assert not bool(audit.delivery_conserved)


def test_negative_controls_do_not_double_pay():
    initial = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    armed = tick_claim_step(
        initial, TickClaimAction.ARM_HARVEST, TickClaimVariant.MUTANT
    )

    one_tick_early = tick_claim_step(
        armed, TickClaimAction.DO, TickClaimVariant.MUTANT
    )
    assert int(one_tick_early.grain) == 1

    due = _due_state()
    scheduled_only = tick_claim_step(
        due, TickClaimAction.NOOP, TickClaimVariant.MUTANT
    )
    one_tick_late = tick_claim_step(
        scheduled_only, TickClaimAction.DO, TickClaimVariant.MUTANT
    )
    assert int(one_tick_late.grain) == 1

    cancelled = tick_claim_step(
        due, TickClaimAction.CANCEL_HARVEST, TickClaimVariant.MUTANT
    )
    assert int(cancelled.grain) == 0
    assert not bool(cancelled.reservation_present)

    unripe = tick_claim_step(
        due.replace(crop_ripe=jnp.asarray(False), crop_age=jnp.asarray(0)),
        TickClaimAction.DO,
        TickClaimVariant.MUTANT,
    )
    assert int(unripe.grain) == 0

    old_generation = tick_claim_step(
        due.replace(reservation_generation=due.crop_generation - 1),
        TickClaimAction.DO,
        TickClaimVariant.MUTANT,
    )
    old_cycle = tick_claim_step(
        due.replace(reservation_cycle_id=due.crop_cycle_id - 1),
        TickClaimAction.DO,
        TickClaimVariant.MUTANT,
    )
    assert int(old_generation.grain) == 1
    assert int(old_cycle.grain) == 1

    fixed_single = tick_claim_step(
        initial, TickClaimAction.DO, TickClaimVariant.FIXED
    )
    mutant_single = tick_claim_step(
        initial, TickClaimAction.DO, TickClaimVariant.MUTANT
    )
    assert int(fixed_single.grain) == int(mutant_single.grain) == 1


def test_observation_masking_and_goals_use_public_observation_only():
    state = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    ).replace(
        reservation_present=jnp.asarray(True),
        reservation_due_tick=jnp.asarray(7),
    )
    visible = observe_tick_claim(state)
    assert bool(visible.facility_visible)
    assert bool(visible.reservation_present)
    assert int(visible.reservation_remaining_ticks) == 7

    hidden = observe_tick_claim(state.replace(player_position=jnp.asarray((15, 15))))
    assert not bool(hidden.facility_visible)
    assert not bool(hidden.facility_exists)
    assert not bool(hidden.reservation_present)
    assert int(hidden.reservation_remaining_ticks) == 0
    assert int(hidden.reservation_delay) == 0
    assert not bool(hidden.crop_visible)
    goals = tick_claim_goal_vector(hidden)
    assert not bool(goals[GOAL_IDS.index("facility/reservation_present")])
    assert not bool(goals[GOAL_IDS.index("facility/reservation_absent")])
    assert not bool(goals[GOAL_IDS.index("visible/crop_ripe")])
    assert not bool(goals[GOAL_IDS.index("visible/crop_unripe")])

    for count in (0, 1, 2, 3, 4):
        observation = visible.replace(grain=jnp.asarray(count))
        vector = tick_claim_goal_vector(observation)
        for threshold in (1, 2, 3):
            index = GOAL_IDS.index(f"inventory/raw_material_ge_{threshold}")
            assert bool(vector[index]) == (count >= threshold)
            assert bool(tick_claim_goal(observation, index)) == bool(vector[index])


def test_all_workshop12_goals_are_reached_by_fixed_legal_transitions():
    state = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    actions = (
        TickClaimAction.ARM_HARVEST,
        TickClaimAction.CANCEL_HARVEST,
        TickClaimAction.DO,
    )
    actions += (TickClaimAction.NOOP,) * GROWTH_WAIT_TICKS
    actions += (TickClaimAction.DO,)
    actions += (TickClaimAction.NOOP,) * GROWTH_WAIT_TICKS
    actions += (
        TickClaimAction.DO,
        TickClaimAction.DOWN,
        TickClaimAction.DELIVER,
    )
    seen = tick_claim_goal_vector(observe_tick_claim(state))
    for action in actions:
        state = tick_claim_step(state, action, TickClaimVariant.FIXED)
        observation = observe_tick_claim(state)
        vector = tick_claim_goal_vector(observation)
        seen = jnp.logical_or(seen, vector)
        np.testing.assert_array_equal(
            np.asarray(vector),
            np.asarray(
                jnp.stack(
                    [tick_claim_goal(observation, index) for index in range(len(GOAL_IDS))]
                )
            ),
        )
    assert bool(jnp.all(seen))
    assert int(state.delivered_total) == 3

    hidden_changed = state.replace(
        cycle_claimed=jnp.logical_not(state.cycle_claimed),
        crop_generation=state.crop_generation + 10,
    )
    np.testing.assert_array_equal(
        np.asarray(tick_claim_goal_vector(observe_tick_claim(state))),
        np.asarray(tick_claim_goal_vector(observe_tick_claim(hidden_changed))),
    )


def test_goal_completion_does_not_reset_world_and_horizon_uses_terminal_observation():
    env = TickClaimEnvNoAutoReset(
        TickClaimVariant.MUTANT, start=TickClaimStart.PATH_CHECK
    )
    state = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    actions = _exploit_core(0, int(state.reservation_delay))
    state, _, _, _ = _run(state, actions[:-1], TickClaimVariant.MUTANT)
    observation, delivered, reward, world_done, info = env.step(
        jax.random.PRNGKey(0), state, actions[-1]
    )
    assert float(reward) == 1.0
    assert bool(info["HackRL/goal_done"])
    assert not bool(world_done)
    assert int(delivered.delivered_total) == 3
    assert int(observation.delivered_total) == 3

    _, continued, repeated_reward, repeated_done, repeated_info = env.step(
        jax.random.PRNGKey(2), delivered, TickClaimAction.NOOP
    )
    assert float(repeated_reward) == 0.0
    assert not bool(repeated_info["HackRL/goal_done"])
    assert not bool(repeated_done)
    assert int(continued.delivered_total) == 3
    assert int(continued.tick) == int(delivered.tick) + 1

    terminal_start = state.replace(tick=jnp.asarray(127))
    terminal_observation, terminal, _, done, terminal_info = env.step(
        jax.random.PRNGKey(1), terminal_start, TickClaimAction.NOOP
    )
    assert bool(done)
    assert bool(terminal_info["HackRL/world_done"])
    assert int(terminal.tick) == 128
    assert float(terminal_observation.remaining_world_fraction) == 0.0
    np.testing.assert_array_equal(terminal.player_position, terminal_start.player_position)


def test_kernel_oracle_observation_and_predicates_are_jittable_and_vmappable():
    state = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    fixed_step = jax.jit(
        lambda current, action: tick_claim_step(
            current, action, TickClaimVariant.FIXED
        )
    )
    compiled = fixed_step(state, jnp.asarray(TickClaimAction.NOOP))
    eager = tick_claim_step(state, TickClaimAction.NOOP, TickClaimVariant.FIXED)
    assert jax.tree_util.tree_all(
        jax.tree.map(lambda left, right: jnp.array_equal(left, right), compiled, eager)
    )

    workers = jnp.arange(32, dtype=jnp.int32)
    states = jax.vmap(reset_tick_claim_worker)(workers)
    actions = jnp.full((32,), TickClaimAction.NOOP, dtype=jnp.int32)
    stepped = jax.jit(
        jax.vmap(
            lambda current, action: tick_claim_step(
                current, action, TickClaimVariant.MUTANT
            )
        )
    )(states, actions)
    observations = jax.jit(jax.vmap(observe_tick_claim))(stepped)
    goals = jax.jit(jax.vmap(tick_claim_goal_vector))(observations)
    assert goals.shape == (32, 12)
    assert np.asarray(stepped.tick).tolist() == [1] * 32

    oracle = initial_tick_claim_oracle()
    _, snapshot = tick_claim_step_with_snapshot(
        state, TickClaimAction.NOOP, TickClaimVariant.FIXED
    )
    audit = jax.jit(audit_tick_claim_transition)(
        oracle, state, TickClaimAction.NOOP, compiled, snapshot
    )
    assert not bool(audit.violation)
    assert bool(audit.delivery_conserved)



def test_independent_oracle_rejects_adversarial_physical_transitions():
    before = make_tick_claim_state(
        0, TickClaimPhase.RIPE, start=TickClaimStart.PATH_CHECK
    )
    double_created = before.replace(
        grain=jnp.asarray(2),
        crop_ripe=jnp.asarray(False),
        crop_age=jnp.asarray(0),
        tick=before.tick + 1,
    )
    double_audit = audit_tick_claim_transition(
        initial_tick_claim_oracle(),
        before,
        TickClaimAction.DO,
        double_created,
        TickClaimTransitionSnapshot(
            grain_after_settlement=jnp.asarray(2),
            delivered_total_after_settlement=before.delivered_total,
        ),
    )
    assert int(double_audit.created_amount) == 2
    assert bool(double_audit.violation)

    carrying = before.replace(grain=jnp.asarray(2))
    broken_transfer = carrying.replace(
        grain=jnp.asarray(0),
        delivered_total=jnp.asarray(1),
        tick=carrying.tick + 1,
    )
    delivery_audit = audit_tick_claim_transition(
        initial_tick_claim_oracle(),
        carrying,
        TickClaimAction.DELIVER,
        broken_transfer,
        TickClaimTransitionSnapshot(
            grain_after_settlement=carrying.grain,
            delivered_total_after_settlement=carrying.delivered_total,
        ),
    )
    assert not bool(delivery_audit.delivery_conserved)

    fixed_goal = before.replace(delivered_total=jnp.asarray(2))
    mutant_goal = before.replace(delivered_total=jnp.asarray(3))
    assert not bool(
        tick_claim_beneficial_use(fixed_goal, mutant_goal, jnp.asarray(False))
    )


def test_crop_phase_tiles_and_growth_boundary_are_exact():
    state = make_tick_claim_state(
        0, TickClaimPhase.UNRIPE, start=TickClaimStart.PATH_CHECK
    )
    for expected_age in range(1, GROWTH_WAIT_TICKS):
        state = tick_claim_step(state, TickClaimAction.NOOP, TickClaimVariant.FIXED)
        assert int(state.crop_age) == expected_age
        assert not bool(state.crop_ripe)
        assert bool(jnp.any(observe_tick_claim(state).map_tiles == TILE_CROP_UNRIPE))

    state = tick_claim_step(state, TickClaimAction.NOOP, TickClaimVariant.FIXED)
    assert int(state.crop_age) == GROWTH_WAIT_TICKS
    assert bool(state.crop_ripe)
    assert int(state.crop_cycle_id) == 1
    assert bool(jnp.any(observe_tick_claim(state).map_tiles == TILE_CROP_RIPE))
