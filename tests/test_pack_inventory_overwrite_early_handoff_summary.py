import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))


def _load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SUMMARY = _load_script(
    "summarize_pack_inventory_overwrite_early_handoff_test",
    "scripts/summarize_pack_inventory_overwrite_early_handoff.py",
)


def test_decimal_audit_reproduces_all_carry_gate_probability_points():
    raw = json.loads(SUMMARY.RAW_PATH.read_text(encoding="utf-8"))
    fixed = [episode for row in raw["rows"] for episode in row["fixed"]["episodes"]]
    for label, probability in SUMMARY.PROBABILITIES.items():
        audited = SUMMARY.decimal_profile(fixed, probability)
        for metric, value in audited.items():
            assert float(value) == pytest.approx(
                raw["aggregate"]["exact"][label][metric], abs=1e-14
            )


def test_carry_gate_improves_p05_return_for_all_five_seeds():
    raw = json.loads(SUMMARY.RAW_PATH.read_text(encoding="utf-8"))
    carry_source = SUMMARY.paired_contrast(raw["comparison"], "carry_gated", "source")
    carry_late = SUMMARY.paired_contrast(raw["comparison"], "carry_gated", "persistent")
    assert carry_source["signs"] == {
        "positive": 5,
        "equal_within_tolerance": 0,
        "negative": 0,
    }
    assert carry_late["signs"] == {
        "positive": 5,
        "equal_within_tolerance": 0,
        "negative": 0,
    }
    assert carry_source["mean_difference"] == pytest.approx(0.01321999679800877)
    assert carry_late["mean_difference"] == pytest.approx(0.01116420125224833)


def test_carry_gate_retains_task_and_reduces_risky_starts():
    raw = json.loads(SUMMARY.RAW_PATH.read_text(encoding="utf-8"))
    comparison = raw["comparison"]
    assert comparison["carry_gated"]["fixed_success_rate"] == 1.0
    assert comparison["source"]["fixed_risky_start_count"] == 94
    assert comparison["persistent"]["fixed_risky_start_count"] == 82
    assert comparison["carry_gated"]["fixed_risky_start_count"] == 40
    assert comparison["adapted"]["fixed_risky_start_count"] == 6
    diagnostic = SUMMARY.carry_diagnostic(raw)
    assert diagnostic["episodes"] == 160
    assert diagnostic["episodes_with_handoff"] == 160
    assert diagnostic["first_handoff_safe_action_count"] == 154
    assert diagnostic["first_handoff_risky_action_count"] == 6
    assert diagnostic["failed_path_count"] == 0
