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
    "summarize_pack_inventory_overwrite_one_action_swap_test",
    "scripts/summarize_pack_inventory_overwrite_one_action_swap.py",
)


def _inputs():
    raw = json.loads(SUMMARY.RAW_PATH.read_text(encoding="utf-8"))
    design = json.loads(SUMMARY.DESIGN_PATH.read_text(encoding="utf-8"))
    run_manifest = json.loads(SUMMARY.RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
    complete = json.loads(SUMMARY.COMPLETE_PATH.read_text(encoding="utf-8"))
    return raw, design, run_manifest, complete


def test_decimal_audit_reproduces_D_at_p0_and_p05():
    raw, _, _, _ = _inputs()
    fixed = [episode for row in raw["rows"] for episode in row["fixed"]["episodes"]]
    for label, probability in (("p00", SUMMARY.Decimal("0")), ("p05", SUMMARY.Decimal("0.05"))):
        audited = SUMMARY.decimal_profile(fixed, probability)
        for metric, value in audited.items():
            assert float(value) == pytest.approx(
                raw["aggregate"]["exact"][label][metric], abs=1e-14
            )


def test_predeclared_diagnostic_has_mixed_recovery():
    raw, _, _, _ = _inputs()
    assert raw["diagnostic"]["recovered_count"] == 2
    assert raw["diagnostic"]["recovered_seed_state_indices"] == [[35, 16], [35, 24]]
    failed = [
        [row["seed"], row["state_index"]]
        for row in raw["diagnostic"]["targets"]
        if not row["D_success"]
    ]
    assert failed == [[35, 4], [35, 12]]
    assert raw["diagnostic"]["new_D_failures_among_B_successes"] == []


def test_D_whole_panel_lies_between_B_and_C_without_risk_count_change():
    raw, _, _, _ = _inputs()
    comparison = raw["comparison"]
    b = comparison["B_carry_persistent"]
    c = comparison["C_carry_gated"]
    d = comparison["D_one_action_swap"]
    assert b["fixed_success_rate"] == 0.975
    assert d["fixed_success_rate"] == 0.9875
    assert c["fixed_success_rate"] == 1.0
    assert (
        b["exact"]["p05"]["expected_discounted_return"]
        < d["exact"]["p05"]["expected_discounted_return"]
        < c["exact"]["p05"]["expected_discounted_return"]
    )
    assert (
        b["fixed_risky_start_count"]
        == c["fixed_risky_start_count"]
        == d["fixed_risky_start_count"]
        == 40
    )


def test_p05_contrasts_preserve_all_seed_directions():
    raw, _, _, _ = _inputs()
    d_minus_b = SUMMARY.contrast(
        raw["comparison"], "D_one_action_swap", "B_carry_persistent"
    )
    c_minus_d = SUMMARY.contrast(
        raw["comparison"], "C_carry_gated", "D_one_action_swap"
    )
    assert d_minus_b["mean_difference"] == pytest.approx(0.01247108857613044)
    assert d_minus_b["signs"] == {
        "positive": 4,
        "equal_within_tolerance": 1,
        "negative": 0,
    }
    assert c_minus_d["mean_difference"] == pytest.approx(0.0117034037636769)
    assert c_minus_d["signs"] == {
        "positive": 2,
        "equal_within_tolerance": 3,
        "negative": 0,
    }


def test_compact_builder_preserves_provenance_and_result_limits():
    raw, design, run_manifest, complete = _inputs()
    compact = SUMMARY.build_compact(raw, design, run_manifest, complete)
    assert compact["diagnostic"]["recovered_count"] == 2
    assert compact["D_execution"]["swap_path_count"] == 72
    assert compact["D_execution"]["no_swap_path_count"] == 88
    assert compact["decimal_audit"]["passed"]
    assert compact["provenance"]["raw_result"]["sha256"] == complete["result_sha256"]
    assert any("not a fresh generalization panel" in item for item in compact["non_claims"])
