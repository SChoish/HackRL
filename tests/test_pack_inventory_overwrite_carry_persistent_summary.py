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
    "summarize_pack_inventory_overwrite_carry_persistent_test",
    "scripts/summarize_pack_inventory_overwrite_carry_persistent.py",
)


def test_decimal_audit_reproduces_b_at_p0_and_p05():
    raw = json.loads(SUMMARY.RAW_PATH.read_text(encoding="utf-8"))
    fixed = [episode for row in raw["rows"] for episode in row["fixed"]["episodes"]]
    for label, probability in (("p00", SUMMARY.Decimal("0")), ("p05", SUMMARY.Decimal("0.05"))):
        audited = SUMMARY.decimal_profile(fixed, probability)
        for metric, value in audited.items():
            assert float(value) == pytest.approx(
                raw["comparison"]["B_carry_persistent"]["exact"][label][metric], abs=1e-14
            )


def test_primary_c_minus_b_and_sensitivity_are_preserved():
    raw = json.loads(SUMMARY.RAW_PATH.read_text(encoding="utf-8"))
    result = SUMMARY.contrast(raw["comparison"], "C_carry_gated", "B_carry_persistent")
    assert result["mean_difference"] == pytest.approx(0.02417449233980711)
    assert result["signs"] == {"positive": 4, "equal_within_tolerance": 1, "negative": 0}
    assert result["seed35_excluded_sensitivity"]["mean_difference"] == pytest.approx(
        0.0008974247435711846
    )


def test_return_rule_changes_retention_not_risky_start_count():
    raw = json.loads(SUMMARY.RAW_PATH.read_text(encoding="utf-8"))
    b = raw["comparison"]["B_carry_persistent"]
    c = raw["comparison"]["C_carry_gated"]
    assert b["fixed_risky_start_count"] == c["fixed_risky_start_count"] == 40
    assert b["fixed_success_rate"] == 0.975
    assert c["fixed_success_rate"] == 1.0
    assert b["adapted_step_fraction"] == pytest.approx(0.8401486988847584)
    assert c["adapted_step_fraction"] == pytest.approx(0.35770871419865935)


def test_secondary_b_minus_a_is_seed35_sensitive():
    raw = json.loads(SUMMARY.RAW_PATH.read_text(encoding="utf-8"))
    result = SUMMARY.contrast(raw["comparison"], "B_carry_persistent", "A_late_risk_persistent")
    assert result["mean_difference"] == pytest.approx(-0.013010291087558846)
    assert result["seed35_excluded_sensitivity"]["mean_difference"] == pytest.approx(
        0.011069043957014163
    )
    assert result["seed35_excluded_sensitivity"]["signs"] == {
        "positive": 4,
        "equal_within_tolerance": 0,
        "negative": 0,
    }
