import itertools

import jax
import jax.numpy as jnp
import numpy as np

from hackrl.tick_claim import GOAL_IDS, TickClaimSplit, TickClaimVariant
from hackrl.tick_claim_gc import TickClaimGCConfig, _evaluation_states, initialize_tick_claim_gc
from hackrl.tick_claim_gc_goal_eval import (
    _rollout,
    aggregate_goal_episodes,
    tick_claim_gc_goal_eval_action_seed,
)


def test_action_seeds_do_not_collide_inside_the_frozen_grid():
    seeds = [
        tick_claim_gc_goal_eval_action_seed(*item)
        for item in itertools.product(
            range(3), range(12), range(2), range(32), range(4)
        )
    ]
    assert len(seeds) == len(set(seeds))
    assert min(seeds) >= 30000


def test_already_true_episodes_stay_out_of_the_ability_rate():
    success = np.asarray([True, True, False])
    length = np.asarray([1, 4, 128])
    initially_true = np.asarray([True, False, False])
    done = np.asarray([True, True, True])
    false = np.asarray([False, False, False])
    zeros = np.asarray([0, 0, 0])
    row = aggregate_goal_episodes(
        success=success,
        length=length,
        initially_true=initially_true,
        done=done,
        violation_seen=false,
        violation_count=zeros,
        violation_delivery=false,
        mask=np.asarray([True, True, True]),
    )
    assert row["episodes"] == 3
    assert row["eligible_episodes"] == 2
    assert row["already_satisfied_rate"] == 1 / 3
    assert row["success_rate_all"] == 2 / 3
    assert row["success_rate_eligible"] == 0.5
    assert row["mean_success_length_eligible"] == 4
    empty = aggregate_goal_episodes(
        success=success[:0],
        length=length[:0],
        initially_true=initially_true[:0],
        done=done[:0],
        violation_seen=false[:0],
        violation_count=zeros[:0],
        violation_delivery=false[:0],
        mask=np.asarray([], dtype=bool),
    )
    assert empty["eligible_episodes"] == 0
    assert empty["success_rate_eligible"] is None
    assert GOAL_IDS[7] == "facility/reservation_absent"


def test_one_command_rollout_reports_an_eligible_denominator():
    config = TickClaimGCConfig(
        num_envs=2,
        num_steps=2,
        num_updates=1,
        minibatch_size=2,
        hidden_size=8,
        goal_mode="workshop12",
    )
    network, runner = initialize_tick_claim_gc(config)
    states, _labels, _state_indices, _repeats = _evaluation_states(
        TickClaimSplit.VALIDATION, 1
    )
    goals = jnp.zeros((int(_labels.shape[0]),), dtype=jnp.int32)
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(goals.shape[0], dtype=jnp.int32))
    rolled = _rollout(
        network,
        runner.train_state.params,
        states,
        goals,
        variant=TickClaimVariant.FIXED,
        stochastic=False,
        keys=keys,
    )
    initially_true = np.asarray(jax.device_get(rolled["initially_true"]))
    assert initially_true.shape == (64,)
    assert int(np.sum(~initially_true)) == 64
