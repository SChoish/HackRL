import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "analyze_gc_double_dqn_postmortem",
        REPOSITORY / "scripts/analyze_gc_double_dqn_postmortem.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


POSTMORTEM = _load()


def test_tick_probes_are_verified_fixed_suffix_states():
    probes = POSTMORTEM._tick_probes()
    assert len(probes) == 16 * 5
    assert {item["probe"] for item in probes} == {
        "harvest_1",
        "harvest_2",
        "harvest_3",
        "approach_delivery",
        "deliver_3",
    }


def test_pack_probes_cover_both_phases_and_verified_fixed_suffixes():
    probes = POSTMORTEM._pack_probes()
    assert len(probes) == 16 * 9
    assert {item["phase"] for item in probes} == {"empty", "loaded"}
    assert sum(item["probe"] == "withdraw_storage" for item in probes) == 16
    assert sum(item["probe"] == "deliver_3" for item in probes) == 32


def test_write_json_refuses_implicit_overwrite(tmp_path):
    output = tmp_path / "result.json"
    POSTMORTEM._write_json(output, {"version": 1})

    with pytest.raises(FileExistsError):
        POSTMORTEM._write_json(output, {"version": 2})

    assert json.loads(output.read_text(encoding="utf-8")) == {"version": 1}
    POSTMORTEM._write_json(output, {"version": 2}, force=True)
    assert json.loads(output.read_text(encoding="utf-8")) == {"version": 2}


def test_update_log_reports_field_coverage(tmp_path):
    path = tmp_path / "updates.jsonl"
    rows = [
        {
            "goal_completions": 2,
            "sampled_valid_transitions": 3,
            "successes_by_goal": [1, 1],
        },
        {"goal_completions": 5, "sampled_valid_transitions": 7},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = POSTMORTEM._summarize_update_log(path)

    assert result["goal_completions_all_goals"] == 7
    assert result["sampled_valid_transitions_all_goals"] == 10
    assert result["per_goal_success_counts_coverage"] == "some"
    assert result["sampled_goal_indices_coverage"] == "none"


def test_replay_selects_delivery_goals_by_name(monkeypatch):
    monkeypatch.setattr(
        POSTMORTEM,
        "_goal_ids",
        lambda env: (
            "delivery/count_ge_1",
            "inventory/raw_material_ge_1",
            "delivery/count_ge_3",
        ),
    )
    replay = SimpleNamespace(
        size=4,
        valid=np.asarray([True, True, True, False]),
        goal_index=np.asarray([0, 1, 2, 99]),
        reward=np.asarray([1.0, 0.0, 1.0, 0.0]),
        action=np.asarray(
            [
                int(POSTMORTEM.TickClaimAction.DELIVER),
                int(POSTMORTEM.TickClaimAction.NOOP),
                int(POSTMORTEM.TickClaimAction.DELIVER),
                99,
            ]
        ),
    )

    result = POSTMORTEM._summarize_replay("tick", replay)

    assert [
        row["goal_index"] for row in result["delivery_goal_rows"]
    ] == [0, 2]
    assert sum(row["valid_entries"] for row in result["goals"]) == 3
