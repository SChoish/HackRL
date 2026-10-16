from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_cross_environment_teacher_matrix_figure.py"
RESULTS = ROOT / "docs/manifests/cross_environment_teacher_matrix_v1_results.json"
SPEC = importlib.util.spec_from_file_location("teacher_figure", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_figure_data_keeps_seed_metrics_and_panels_separate() -> None:
    rows = MODULE._rows(json.loads(RESULTS.read_text()))
    assert len(rows) == 120
    assert {row["environment"] for row in rows} == {
        "tick_claim",
        "pack_restore",
        "craft_remain",
    }
    assert {row["panel"] for row in rows} == {
        "exploit_adoption_contrast",
        "normal_success",
        "discounted_return_contrast",
    }
    tick_natural = [
        row
        for row in rows
        if row["environment"] == "tick_claim"
        and row["panel"] == "exploit_adoption_contrast"
        and row["series"] == "natural"
    ]
    assert len(tick_natural) == 5
    assert sum(row["value"] for row in tick_natural) / 5 == 11.5625
