"""Shared evidence checks for fixed mine-expedition experiments."""

from pathlib import Path

from flax import serialization

from hackrl.mine_expedition_ppo import (
    evaluate_mine_expedition_frozen,
    initialize_mine_expedition_ppo,
    load_mine_expedition_checkpoint,
)


GIB = 1024**3


def validate_evaluation(evaluation, optimizer, location, errors):
    if not isinstance(evaluation, dict):
        errors.append(f"{location}: evaluation is not an object")
        return False
    valid = evaluation.get("runner_state_immutable") is True
    if not valid:
        errors.append(f"{location}: evaluation changed runner state")
    for policy, episodes in (
        ("mode", optimizer["mode_eval_episodes_per_seed"]),
        ("sample", optimizer["sample_eval_episodes_per_seed"]),
    ):
        block = evaluation.get(policy, {})
        expected_stochastic = policy == "sample"
        if not isinstance(block, dict) or any(
            (
                block.get("variant") != "fixed",
                block.get("start") != "natural",
                block.get("episodes") != episodes,
                block.get("stochastic") is not expected_stochastic,
            )
        ):
            errors.append(f"{location}: invalid natural-start {policy} evaluation")
            valid = False
            continue
        rate = block.get("success_rate")
        if not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
            errors.append(f"{location}: invalid {policy} success rate")
            valid = False
    return valid


def validate_capacity(records, destination, phase, seed, errors):
    if not isinstance(records, list):
        errors.append(f"seed {seed}: capacity history is not a list")
        return
    required = {"start_or_resume"}
    required.update(
        f"before_checkpoint_{update}" for update in phase["checkpoint_updates"]
    )
    required.update(
        f"periodic_update_{update}"
        for update in range(64, phase["updates"] + 1, 64)
    )
    observed = set()
    for index, record in enumerate(records):
        try:
            observed.add(record["event"])
            projected = int(record["projected_remaining_write_bytes"])
            reserve = int(record["safety_reserve_bytes"])
            required_free = int(record["required_free_bytes"])
            target_free = int(record["target_filesystem"]["free_bytes"])
            recorded_destination = Path(record["destination"]).resolve()
        except (KeyError, TypeError, ValueError):
            errors.append(f"seed {seed}: malformed capacity record {index}")
            continue
        if (
            recorded_destination != destination.resolve()
            or reserve < 8 * GIB
            or reserve < int(0.2 * projected)
            or required_free != projected + reserve
            or target_free < required_free
        ):
            errors.append(f"seed {seed}: invalid capacity record {index}")
    missing = sorted(required - observed)
    if missing:
        errors.append(f"seed {seed}: missing capacity events: {missing}")


def rollout_window(updates, width):
    window = updates[-width:]
    totals = {
        name: sum(float(row.get(name, 0)) for row in window)
        for name in (
            "completed_episodes",
            "completed_successes",
            "completed_timeouts",
            "crafted_pickaxes",
            "mined_targets",
            "returned_targets",
        )
    }
    totals["success_fraction"] = totals["completed_successes"] / max(
        totals["completed_episodes"], 1.0
    )
    totals["updates"] = width
    return totals


def re_evaluate(checkpoint, config):
    network, template = initialize_mine_expedition_ppo(config)
    runner = load_mine_expedition_checkpoint(checkpoint, template, config)
    state_before = serialization.to_bytes(runner)
    mode = evaluate_mine_expedition_frozen(
        network,
        runner.train_state.params,
        stochastic=False,
        episodes=config.mode_eval_episodes,
        seed_base=30000,
        learner_seed=config.seed,
        discount=config.gamma,
    )
    sample = evaluate_mine_expedition_frozen(
        network,
        runner.train_state.params,
        stochastic=True,
        episodes=config.sample_eval_episodes,
        seed_base=30000,
        learner_seed=config.seed,
        discount=config.gamma,
    )
    return {
        "mode": mode,
        "sample": sample,
        "runner_state_immutable": state_before == serialization.to_bytes(runner),
    }
