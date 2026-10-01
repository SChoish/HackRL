import jax
import jax.numpy as jnp
import numpy as np

from hackrl.tick_claim import (
    TickClaimAction,
    TickClaimVariant,
    manual_harvest_requested,
    tick_claim_step_with_snapshot,
)
from hackrl.tick_claim_gc import (
    TickClaimGCConfig,
    initialize_tick_claim_gc,
    make_tick_claim_gc_update,
    step_tick_claim_gc_workers,
)
from eval_tick_claim_reservation_probe import build_reservation_probe_states
from run_tick_claim_gc_goalbundle import (
    ABSENT,
    ALLOWED,
    M_GOALS,
    N_GOALS,
    PRESENT,
    build_jobs,
    sampling_weights,
)


def test_bundle_conditions_exclude_reservation_goals():
    assert set(ALLOWED["D"]) == {11}
    assert set(ALLOWED["DM"]) == {11, *M_GOALS}
    assert set(ALLOWED["DN"]) == {11, *N_GOALS}
    assert set(ALLOWED["DMN"]) == {0, 1, 2, 3, 4, 5, 8, 9, 10, 11}
    assert set(M_GOALS).isdisjoint(N_GOALS)
    jobs = build_jobs()
    assert len([job for job in jobs if job["kind"] == "pretrain"]) == 20
    assert len([job for job in jobs if job["kind"] == "adapt"]) == 40
    for name, allowed in ALLOWED.items():
        assert PRESENT not in allowed and ABSENT not in allowed
        weights = sampling_weights(allowed)
        assert weights[PRESENT] == weights[ABSENT] == 0
        assert sum(weights) == len(allowed)
        assert {index for index, weight in enumerate(weights) if weight} == set(allowed)
        assert name in {job["condition"] for job in jobs}


def test_manual_harvest_attempt_is_logged_and_is_not_the_reward():
    facing = build_reservation_probe_states()["no_reservation"]
    assert bool(manual_harvest_requested(facing, int(TickClaimAction.DO)))
    paid, _snapshot = tick_claim_step_with_snapshot(
        facing, int(TickClaimAction.DO), TickClaimVariant.FIXED
    )
    assert int(paid.grain) - int(facing.grain) == 1

    config = TickClaimGCConfig(
        num_envs=2,
        num_steps=2,
        num_updates=1,
        minibatch_size=2,
        hidden_size=8,
        goal_mode="deliver_3",
    )
    _network, runner = initialize_tick_claim_gc(config)
    facing_batch = jax.tree.map(
        lambda leaf: jnp.stack([leaf, leaf]), facing
    )
    runner = runner.replace(
        env_state=facing_batch,
        command_active=jnp.ones((2,), dtype=jnp.bool_),
        current_goal=jnp.asarray([11, 11], dtype=jnp.int32),
    )
    _next, event = step_tick_claim_gc_workers(
        runner, jnp.asarray([int(TickClaimAction.DO), int(TickClaimAction.DO)]), config
    )
    assert int(np.sum(np.asarray(event.manual_harvest_attempt))) == 2
    assert float(np.sum(np.asarray(event.reward))) == 0.0
    _next, idle = step_tick_claim_gc_workers(
        runner, jnp.asarray([int(TickClaimAction.NOOP), int(TickClaimAction.NOOP)]), config
    )
    assert int(np.sum(np.asarray(idle.manual_harvest_attempt))) == 0

    network, fresh = initialize_tick_claim_gc(config)
    _stepped, metrics = jax.jit(make_tick_claim_gc_update(network, config))(fresh)
    jax.block_until_ready(_stepped.global_update)
    assert int(np.asarray(metrics["manual_harvest_attempts"])) >= 0
