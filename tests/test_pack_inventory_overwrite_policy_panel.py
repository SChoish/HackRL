import json
from pathlib import Path

from hackrl.pack_inventory_overwrite_rare import evaluate_regime_point


def test_policy_panel_manifest_freezes_all_baselines_without_learning():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads(
        (root / "docs/manifests/pack_inventory_overwrite_policy_panel_v1.json").read_text()
    )
    assert spec["seeds"] == [30, 31, 32, 33, 34]
    assert set(spec["policies"]) == {
        "original",
        "normal_continue",
        "p1_adapt",
        "zero_loss_reference",
    }
    assert spec["policies"]["p1_adapt"]["checkpoint"] == "adapt_32"
    assert spec["exact_computation"]["learning_probabilities_of_interest"] == [
        0.05,
        0.3,
    ]
    assert spec["execution_budget"] == {
        "learned_policy_endpoint_episodes": 960,
        "reference_episodes": 32,
        "optimizer_updates": 0,
        "collector_slots": 0,
        "checkpoint_writes": 0,
        "training_started": False,
        "automatic_extension": False,
    }


def test_policy_curve_allows_more_than_one_risky_harvest():
    profile = {
        "success_polynomial_by_survival_power": {"0": 0.25, "2": 0.75},
        "return_polynomial_by_survival_power": {"0": 0.20, "2": 0.60},
        "reference_mean_discounted_return": 0.5,
    }
    point = evaluate_regime_point(profile, 0.3)
    assert point["existing_success_rate"] == 0.25 + 0.75 * 0.7**2
    assert point["existing_expected_discounted_return"] == 0.20 + 0.60 * 0.7**2
    assert point["reference_expected_discounted_return"] == 0.5


def test_policy_curve_separates_fixed_failure_from_stochastic_loss():
    profile = {
        "success_polynomial_by_survival_power": {"0": 0.25, "2": 0.50},
        "no_overwrite_polynomial_by_survival_power": {"0": 0.25, "2": 0.75},
        "return_polynomial_by_survival_power": {"0": 0.20, "2": 0.40},
        "reference_mean_discounted_return": 0.5,
    }
    point = evaluate_regime_point(profile, 0.3)
    assert point["existing_failure_probability"] == 1.0 - (
        0.25 + 0.50 * 0.7**2
    )
    assert point["existing_loss_episode_probability"] == 1.0 - (
        0.25 + 0.75 * 0.7**2
    )
    assert (
        point["existing_failure_probability"]
        > point["existing_loss_episode_probability"]
    )


def test_stochastic_ppo_manifest_fixes_only_two_learning_probabilities():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads(
        (
            root
            / "docs/manifests/pack_inventory_overwrite_stochastic_ppo_v1.json"
        ).read_text()
    )
    assert spec["seeds"] == [30, 31, 32, 33, 34]
    assert {
        name: condition["overwrite_probability"]
        for name, condition in spec["conditions"].items()
    } == {"p05_adapt": 0.05, "p30_adapt": 0.30}
    assert spec["optimization"]["updates"] == 32
    assert spec["budget"]["total_jobs"] == 10
    assert spec["budget"]["total_collector_slots"] == 10_485_760
    assert spec["optimization"]["adam_restored"] is True
    assert spec["optimization"]["teacher_updates"] == 0
    assert spec["optimization"]["bc_updates"] == 0
    assert spec["optimization"]["probability_in_observation"] is False
    assert spec["evaluation"]["adaptation_updates"] == [0, 2, 8, 32]
    assert spec["budget"]["automatic_extension"] is False
    assert "is an optimal-avoidance boundary" in spec[
        "probability_selection_interpretation"
    ]
    assert "Neither probability" in spec["probability_selection_interpretation"]
