"""Training kernel agreement with the CRAFT-REMAIN reference engine."""

import jax
import jax.numpy as jnp
import numpy as np

from hackrl.craft_remain import replay, shortest_delivery
from hackrl.craft_remain_env import (
    REFERENCE_ACTION_INDEX,
    CraftRemainAction,
    CraftRemainVariant,
    comparable_fields,
    craft_remain_step,
    make_craft_remain_state,
    replay_actions,
)
from hackrl.craft_remain_gc import (
    CraftRemainGCConfig,
    adaptation_zero_record,
    command_deliver_3,
    compare_contract,
    evaluate_craft_remain_gc_frozen,
    initialize_craft_remain_gc,
    load_craft_remain_gc_checkpoint,
    make_craft_remain_gc_update,
    path_check_record,
    save_craft_remain_gc_checkpoint,
    step_craft_remain_gc_workers,
)
from hackrl.dual_leo import init_dual_leo_teacher, make_dual_leo_update
from hackrl.craft_remain_gc import _batch_inputs, craft_remain_outcome


def _reference_fields(state):
    row, col, _facing, carried, slot_a, slot_b, output, parcel, source, age, delivered = state
    return (row, col, carried, slot_a, slot_b, output, parcel, source, age, delivered)


def test_shortest_paths_match_the_reference_on_both_kernels():
    record = path_check_record()
    assert record["fixed_shortest_length"] == 37
    assert record["mutant_shortest_length"] == 24
    assert record["growth_period"] == 16
    for mutant in (False, True):
        found = shortest_delivery(mutant)
        variant = CraftRemainVariant.MUTANT if mutant else CraftRemainVariant.FIXED
        trained = replay_actions(found["actions"], variant)
        reference = replay(found["actions"], mutant)
        assert trained["length"] == reference["length"] == found["length"]
        assert trained["delivered"] == reference["delivered"]
        assert trained["fields"] == _reference_fields(reference["state"])
        assert trained["slots"] == (reference["state"][4], reference["state"][5])
    assert record["same_actions_on_fixed_delivered"] == 2
    assert record["mutant_training_delivered"] == 3
    assert record["mutant_triggered"] is True
    assert record["mutant_retained_recovered"] is True
    assert record["mutant_excess_delivery"] is True
    assert record["mutant_conservation_violation_amount"] == 2
    assert record["fixed_conservation_violation_amount"] == 0


def test_eager_jit_and_vmap_agree():
    state = make_craft_remain_state()
    action = jnp.asarray(REFERENCE_ACTION_INDEX["DO"])
    eager = craft_remain_step(state, action, CraftRemainVariant.MUTANT)
    compiled = jax.jit(lambda item, act: craft_remain_step(item, act, CraftRemainVariant.MUTANT))(state, action)
    assert comparable_fields(compiled) == comparable_fields(eager)
    paired = jax.vmap(lambda item: craft_remain_step(item, action, CraftRemainVariant.FIXED))(
        jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), state)
    )
    assert np.asarray(paired.carried_grain).tolist() == [1, 1]


def test_oracle_separates_violation_from_recovery_and_excess():
    mutant = replay_actions(shortest_delivery(True)["actions"], CraftRemainVariant.MUTANT)
    fixed = replay_actions(shortest_delivery(True)["actions"], CraftRemainVariant.FIXED)
    assert mutant["conservation_violation_amount"] == 2
    assert mutant["retained_recovered"] is True
    assert mutant["excess_delivery"] is True
    assert fixed["conservation_violation_amount"] == 0
    assert fixed["excess_delivery"] is False
    assert fixed["delivered"] == 2


def test_checkpoint_and_adaptation_zero_record(tmp_path):
    config = CraftRemainGCConfig(
        num_envs=2, num_steps=4, minibatch_size=8, num_updates=1, hidden_size=32, seed=0,
    )
    network, runner = initialize_craft_remain_gc(config)
    before = np.asarray(jax.device_get(jax.tree.leaves(runner.train_state.params)[0]))
    switched = command_deliver_3(runner)
    after = np.asarray(jax.device_get(jax.tree.leaves(switched.train_state.params)[0]))
    assert np.array_equal(before, after)
    updated, metrics = jax.jit(make_craft_remain_gc_update(network, config))(runner)
    assert int(updated.global_update) == 1
    destination = tmp_path / "update_1"
    save_craft_remain_gc_checkpoint(destination, updated, config)
    _, template = initialize_craft_remain_gc(config)
    restored = load_craft_remain_gc_checkpoint(destination, template, config)
    assert int(restored.global_update) == 1
    left = np.asarray(jax.device_get(jax.tree.leaves(updated.train_state.params)[0]))
    right = np.asarray(jax.device_get(jax.tree.leaves(restored.train_state.params)[0]))
    assert np.array_equal(left, right)
    zero = adaptation_zero_record(network, runner.train_state.params, config, episodes=2, horizon=4)
    assert zero["adaptation_update"] == 0
    assert set(("trigger_rate", "exploit_rate", "mean_discounted_return")) <= set(zero)
    assert zero["path_check_lengths_not_included"] is True
    contract = compare_contract()
    assert contract["launch"] is False
    assert contract["science_updates"][0] == 0
    assert "triggered_count" in metrics
    actions = jnp.full((2,), int(CraftRemainAction.CRAFT), dtype=jnp.int32)
    _, event = step_craft_remain_gc_workers(runner, actions, config)
    assert event.conservation_violation.shape == (2,)


def test_frozen_eval_separates_trigger_exploit_and_return():
    config = CraftRemainGCConfig(
        num_envs=2, num_steps=4, minibatch_size=8, num_updates=1, hidden_size=32, seed=1,
    )
    network, runner = initialize_craft_remain_gc(config)
    result = evaluate_craft_remain_gc_frozen(
        network, runner.train_state.params, variant="mutant", stochastic=False,
        repeats_per_state=1, seed_base=20000, learner_seed=20, record_episodes=True, horizon=2,
    )
    assert set(result["natural_reset"]) >= {"success_rate", "mean_length", "mean_discounted_return", "trigger_rate", "exploit_rate"}
    assert {row["family"] for row in result["episode_records"]} == {"natural_reset", "path_check"}
    assert all("exploited" in row and "triggered" in row for row in result["episode_records"])


def test_job_count_matches_the_contract():
    from scripts.run_craft_remain_compare import build_jobs
    jobs = build_jobs()
    assert len(jobs) == 30
    assert sum(job["kind"] == "pretrain" for job in jobs) == 10
    assert sum(job["kind"] == "adapt" for job in jobs) == 20


def test_dual_update_accepts_the_craft_outcome():
    config = CraftRemainGCConfig(
        num_envs=2, num_steps=4, minibatch_size=8, num_updates=1, hidden_size=32, seed=3,
        goal_mode="workshop12",
    )
    network, runner = initialize_craft_remain_gc(config)
    inputs = _batch_inputs(runner.env_state, runner.current_goal)
    teacher, leo_state, minibatch = init_dual_leo_teacher(
        config, inputs[0], inputs[1], network_goals := 12, 24,
    )
    del network_goals
    update = jax.jit(make_dual_leo_update(
        network, teacher, config, craft_remain_outcome, _batch_inputs, minibatch,
    ))
    updated, _leo, metrics = update(runner, leo_state)
    assert int(updated.global_update) == 1
    assert "teacher_applied_grad_steps" in metrics or "ppo_applied_grad_steps" in metrics or len(metrics) > 0
