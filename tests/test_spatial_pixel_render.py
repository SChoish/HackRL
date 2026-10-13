"""Ensure pretty rendering does not change spatial experiment evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from report_figures.record_spatial_trace import demo_trace, from_spec
from report_figures.render_spatial_pixel import render_frame, snapshot, event_message


@pytest.fixture(scope="module")
def demo():
    return demo_trace()


def test_reference_paths_and_block_are_real_kernel_outcomes(demo):
    normal,shortcut,recovery = demo["scenes"]
    assert [s["summary"]["length"] for s in demo["scenes"]] == [14,8,17]
    assert all(s["summary"]["success"] for s in demo["scenes"])
    assert normal["summary"]["wall_passes"] == 0
    assert shortcut["summary"]["wall_passes"] == 2
    blocked = recovery["steps"][1]
    assert blocked["events"]["blocked_dash"]
    assert blocked["before"]["player_position"] == blocked["after"]["player_position"]
    assert blocked["events"]["intermediate_position"] in demo["geometry"]["walls"]


def test_rendering_does_not_mutate_trace_and_holds_completed_state(demo):
    original = deepcopy(demo)
    for index in (0,2,17,22):
        assert render_frame(demo,index).size == (1920,1080)
    assert demo == original
    state,step,held = snapshot(demo["scenes"][1],17)
    assert held and state["tick"] == 8
    assert event_message(demo["scenes"][1],state,step,held)[0] == "DELIVERY COMPLETE"


def test_empty_unsuccessful_trace_never_reports_success(demo):
    scene = deepcopy(demo["scenes"][0])
    scene["steps"] = []
    scene["summary"] = {"success":False,"length":0,"end_reason":"trace end"}
    state,step,held = snapshot(scene,10)
    assert held and state["tick"] == 0
    assert event_message(scene,state,step,held)[0] == "EPISODE ENDED"


def write_spec(tmp_path,demo,tamper=False):
    scene = demo["scenes"][1]
    rows = [{"from":s["before"]["player_position"],"to":s["after"]["player_position"],
             "action":s["action"],"blocked_dash":s["events"]["blocked_dash"]}
            for s in scene["steps"]]
    if tamper:
        rows[1]["to"] = rows[1]["from"]
    record = {"layout_index":0,"length":8,"success":True,"path":rows}
    (tmp_path/"evaluation.json").write_text(json.dumps({"mutant":{"mode":{"episode_records":[record]}}}))
    spec = {"split":"validation","layout":0,"selection":"fixture roundtrip, not learned",
            "scenes":[{"file":"evaluation.json","pointer":"/mutant/mode/episode_records/0",
                       "kernel":"mutant","title":"TEST","subtitle":"Recorded path"}]}
    path = tmp_path/"spec.json"
    path.write_text(json.dumps(spec))
    return path


def test_recorded_episode_replays_and_preserves_source_identity(tmp_path,demo):
    trace = from_spec(write_spec(tmp_path,demo))
    assert trace["scenes"][0]["summary"]["wall_passes"] == 2
    assert len(trace["provenance"]["sources"][0]["sha256"]) == 64


def test_wrong_recorded_geometry_is_rejected(tmp_path,demo):
    with pytest.raises(ValueError,match="disagrees with kernel"):
        from_spec(write_spec(tmp_path,demo,tamper=True))
