import importlib.util
import json
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


def _load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SUMMARY = _load_script(
    "summarize_pack_inventory_overwrite_gate_heldout_test",
    "scripts/summarize_pack_inventory_overwrite_gate_heldout.py",
)


def _inputs():
    raw = json.loads(SUMMARY.RAW_PATH.read_text(encoding="utf-8"))
    design = json.loads(SUMMARY.DESIGN_PATH.read_text(encoding="utf-8"))
    run_manifest = json.loads(SUMMARY.RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
    complete = json.loads(SUMMARY.COMPLETE_PATH.read_text(encoding="utf-8"))
    return raw, design, run_manifest, complete


def test_decimal_audit_reproduces_all_modes_and_probabilities():
    raw, design, _, _ = _inputs()
    for mode in design["conditions"]:
        episodes = [
            episode
            for row in raw["rows"]
            if row["mode"] == mode
            for episode in row["fixed"]["episodes"]
        ]
        for label, probability in SUMMARY.PROBABILITIES:
            audited = SUMMARY.decimal_profile(episodes, probability)
            for metric, value in audited.items():
                assert float(value) == pytest.approx(
                    raw["comparison"][mode]["exact"][label][metric], abs=1e-14
                )


def test_primary_contrast_is_small_positive_and_spans_four_seeds():
    raw, _, _, _ = _inputs()
    primary = SUMMARY.contrast(raw["comparison"], "carry_gated", "source")
    assert primary["mean_difference"] == pytest.approx(0.0028763541226285217)
    assert primary["signs"] == {
        "positive": 4,
        "equal_within_tolerance": 0,
        "negative": 1,
    }
    by_seed = {row["seed"]: row["difference"] for row in primary["per_seed"]}
    assert by_seed[35] == pytest.approx(-0.04404295285260529)
    assert all(by_seed[seed] > 0 for seed in (36, 37, 38, 39))


def test_gate_reduces_risk_but_introduces_two_normal_failures():
    raw, _, _, _ = _inputs()
    source = raw["comparison"]["source"]
    gated = raw["comparison"]["carry_gated"]
    assert source["fixed"]["success_rate"] == 1.0
    assert gated["fixed"]["success_rate"] == 0.9875
    assert source["fixed_risky_start_count"] == 94
    assert gated["fixed_risky_start_count"] == 36
    assert gated["exact"]["p05"]["success_rate"] > source["exact"]["p05"]["success_rate"]
    seed35 = next(row for row in gated["per_seed"] if row["seed"] == 35)
    assert seed35["fixed_success_rate"] == 0.9375


def test_gate_does_not_have_a_seed_consistent_advantage_over_adapted_only():
    raw, _, _, _ = _inputs()
    result = SUMMARY.contrast(raw["comparison"], "carry_gated", "adapted")
    assert result["mean_difference"] == pytest.approx(0.004203554680437094)
    assert result["signs"] == {
        "positive": 1,
        "equal_within_tolerance": 0,
        "negative": 4,
    }


def test_compact_builder_preserves_panel_provenance_and_scope_limits():
    raw, design, run_manifest, complete = _inputs()
    compact = SUMMARY.build_compact(raw, design, run_manifest, complete)
    assert compact["decimal_audit"]["passed"]
    assert compact["provenance"]["raw_result"]["sha256"] == complete["result_sha256"]
    assert compact["protocol"]["panel"]["ordered_full_state_sha256"] == design[
        "heldout_panel"
    ]["ordered_full_state_sha256"]
    assert compact["protocol"]["panel"]["exact_overlap_with_train_natural_states"] == 0
    assert compact["protocol"]["panel"]["exact_overlap_with_validation_natural_states"] == 0
    assert any("does not test generalization" in item for item in compact["non_claims"])
