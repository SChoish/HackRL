import jax
import jax.numpy as jnp
import numpy as np

from hackrl.tick_claim import TickClaimPhase, TickClaimSplit, TickClaimStart, make_tick_claim_state
from hackrl.tick_claim_gc import (
    NUM_GOALS,
    TickClaimGCConfig,
    _resample_masked_goal,
    config_from_tick_claim_gc_payload,
    initialize_tick_claim_gc,
    make_tick_claim_gc_update,
    reinit_tick_claim_gc_adaptation_start,
    reset_tick_claim_gc_optimizer,
    tick_claim_gc_config_payload,
)
from run_tick_claim_gc_goalset import (
    ABSENT,
    ADAPT_UPDATES,
    ALLOWED,
    BATCH,
    CONDITIONS,
    PRESENT,
    SEEDS,
    VALID_BUDGET,
    build_jobs,
    sampling_weights,
)


def test_goalset_jobs_are_twenty_pretrains_and_forty_adapt_cells():
    jobs = build_jobs()
    pretrain = [job for job in jobs if job["kind"] == "pretrain"]
    adapt = [job for job in jobs if job["kind"] == "adapt"]
    assert len(pretrain) == 20
    assert len(adapt) == 40
    assert {job["seed"] for job in jobs} == set(SEEDS)
    assert {job["condition"] for job in adapt} == set(CONDITIONS)
    assert all(
        job["depends_on"] == [f"pretrain-{job['condition']}-s{job['seed']}"]
        for job in adapt
    )
    assert ADAPT_UPDATES * BATCH == 134_217_728
    assert VALID_BUDGET == 512 * BATCH


def test_reservation_weights_match_the_uniform_twelve_goal_rate():
    paired = sampling_weights(ALLOWED["AR"])
    full = sampling_weights(ALLOWED["BR"])
    assert paired[PRESENT] == paired[ABSENT] == full[PRESENT] == full[ABSENT] == 1
    assert paired[11] == 10
    assert sum(full) == NUM_GOALS
    assert abs(paired[PRESENT] / sum(paired) - 1 / NUM_GOALS) < 1e-9
    bare = sampling_weights(ALLOWED["B"])
    assert bare[PRESENT] == bare[ABSENT] == 0
    assert sum(weight > 0 for weight in bare) == 10


def test_old_checkpoint_payload_still_round_trips():
    config = TickClaimGCConfig(
        num_envs=2,
        num_steps=2,
        num_updates=1,
        minibatch_size=2,
        hidden_size=8,
    )
    recorded = tick_claim_gc_config_payload(config)
    del recorded["allowed_goals"]
    del recorded["goal_sampling_weights"]
    restored = config_from_tick_claim_gc_payload(recorded)
    assert restored.goal_mode == "deliver_3"
    assert restored.allowed_goals == ()
    assert restored.goal_sampling_weights == ()


def test_masked_resample_resets_only_when_no_allowed_goal_is_false():
    state = make_tick_claim_state(
        0,
        int(TickClaimPhase.RIPE),
        split=TickClaimSplit.TRAIN,
        start=TickClaimStart.NATURAL,
    )
    weights = jnp.zeros((NUM_GOALS,), dtype=jnp.float32).at[11].set(1)
    key = jax.random.PRNGKey(0)
    false_goals = jnp.zeros((NUM_GOALS,), dtype=jnp.bool_)
    goal, _, _, resets, ok = _resample_masked_goal(key, state, false_goals, weights)
    assert bool(ok) and int(resets) == 0 and int(goal) == 11
    true_goals = jnp.ones((NUM_GOALS,), dtype=jnp.bool_)
    goal, _, _, resets, ok = _resample_masked_goal(key, state, true_goals, weights)
    assert bool(ok) and int(resets) >= 1 and int(goal) == 11


def test_masked_update_keeps_the_command_active():
    weights = sampling_weights(ALLOWED["A"])
    config = TickClaimGCConfig(
        num_envs=2,
        num_steps=2,
        num_updates=1,
        minibatch_size=2,
        hidden_size=8,
        goal_mode="masked",
        allowed_goals=ALLOWED["A"],
        goal_sampling_weights=weights,
    )
    network, runner = initialize_tick_claim_gc(config)
    assert set(int(goal) for goal in np.asarray(runner.current_goal)) <= {11}
    update = jax.jit(make_tick_claim_gc_update(network, config))
    stepped, metrics = update(runner)
    jax.block_until_ready(stepped.global_update)
    assert int(np.asarray(metrics["valid_transitions"])) == 4
    assert int(np.asarray(metrics["empty_minibatches"])) == 0
    assert bool(np.all(np.asarray(stepped.command_active)))


def test_adaptation_start_is_shared_by_seed_and_resets_adam():
    config = TickClaimGCConfig(
        num_envs=2,
        num_steps=2,
        num_updates=1,
        minibatch_size=2,
        hidden_size=8,
        seed=10,
    )
    network, runner = initialize_tick_claim_gc(config)
    stepped, _metrics = jax.jit(make_tick_claim_gc_update(network, config))(runner)
    jax.block_until_ready(stepped.global_update)
    params = jax.tree.map(lambda leaf: np.asarray(jax.device_get(leaf)), stepped.train_state.params)
    reset = reset_tick_claim_gc_optimizer(stepped, config)
    started = reinit_tick_claim_gc_adaptation_start(reset, 10)
    again = reinit_tick_claim_gc_adaptation_start(
        reset_tick_claim_gc_optimizer(stepped, config), 10
    )
    assert int(started.train_state.step) == 0
    assert int(started.global_update) == 0
    left = jax.tree.map(lambda leaf: np.asarray(jax.device_get(leaf)), started.train_state.params)
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(left)))
    assert np.array_equal(np.asarray(started.rng), np.asarray(again.rng))
    assert np.array_equal(np.asarray(started.env_keys), np.asarray(again.env_keys))
