"""Explanation timing and captions must follow the saved events."""
from pathlib import Path
import sys
import pytest

pytest.importorskip("PIL")
pytest.importorskip("imageio")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_figures.explainer import build_timeline, describe_step


def _step(index=1, **events):
    return {"step": index, "action": "REBUILD_EMPTY",
            "before": {"physical_grain_total": 3},
            "after": {"physical_grain_total": 5}, "events": events}


def test_pause_timeline_keeps_every_transition_and_does_not_repeat_held_event():
    fixed = {"initial": {}, "steps": [_step(1), _step(2), _step(3)]}
    mutant = {"initial": {}, "steps": [_step(1, conservation_violation=True, physical_created=2)]}
    trace = {"kernels": {"fixed": fixed, "mutant": mutant}}
    timeline = build_timeline(trace, "pack")
    assert [s["trace_index"] for s in timeline["segments"]] == [0, 1, 2, 3, 3]
    assert [s["duration_seconds"] for s in timeline["segments"]] == [7, 4, 1, 1, 7]
    assert timeline["total_frames"] == 200
    assert timeline["duration_seconds"] == 20
    for left, right in zip(timeline["segments"], timeline["segments"][1:]):
        assert left["end_frame_exclusive"] == right["start_frame"]


def test_explanation_uses_actual_created_amount():
    info = describe_step("pack", _step(conservation_violation=True, physical_created=2))
    assert "3 -> 5" in info["detail"] and "+2" in info["detail"]
    assert info["focus"] == ["anchor_position"]
    assert not describe_step("pack", _step(conservation_violation=True), held=True)["important"]


def test_failed_rebuild_does_not_get_a_duplication_caption():
    info = describe_step("pack", _step())
    assert not info["important"]
    assert "creates extra grain" not in info["title"]


def test_empty_trace_has_intro_and_summary():
    timeline = build_timeline({"kernels": {"fixed": {"steps": []}, "mutant": {"steps": []}}}, "pack")
    assert [s["kind"] for s in timeline["segments"]] == ["intro", "outro"]


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_durations_rejected(duration):
    with pytest.raises(ValueError):
        build_timeline({}, "pack", step_seconds=duration)
