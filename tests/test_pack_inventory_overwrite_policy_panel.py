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
