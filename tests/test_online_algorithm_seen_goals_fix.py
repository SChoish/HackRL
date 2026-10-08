import json
from pathlib import Path


ORIGINAL = Path("docs/manifests/online_algorithm_expansion_v1_development.json")
RERUN = Path("docs/manifests/online_algorithm_expansion_v1_development_seen_goals_fix_v1.json")
SMOKE = Path("docs/manifests/online_algorithm_seen_goals_fix_v1_smoke.json")


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_corrected_rerun_changes_no_candidate_seed_or_budget_contract():
    original = _load(ORIGINAL)
    rerun = _load(RERUN)
    assert rerun["candidate_registry"] == original["candidate_registry"]
    assert rerun["common_training"] == original["common_training"]
    assert rerun["budget"] == original["budget"]
    assert rerun["selection_and_stop"] == original["selection_and_stop"]
    assert rerun["authority"]["main_adaptation"] is False
    assert rerun["execution"]["main_launch"] == "forbidden and never automatic"


def test_corrected_rerun_isolated_on_raid_and_waits_for_fresh_smoke():
    rerun = _load(RERUN)
    execution = rerun["execution"]
    assert rerun["status"] == "authorized_waiting_for_post_fix_smoke"
    assert execution["run_root"].startswith("/raid/ext_csv/HackRL/runs/")
    assert execution["clean_worktree_root"].startswith(
        "/raid/ext_csv/HackRL/worktrees/"
    )
    assert "seen_goals_fix_v1" in execution["run_root"]
    assert execution["run_root"] != _load(ORIGINAL)["execution"]["run_root"]
    assert rerun["smoke_manifest"] == str(SMOKE)


def test_post_fix_smoke_requires_full_shape_and_seen_goal_expansion():
    smoke = _load(SMOKE)
    assert smoke["status"] == "authorized_not_started"
    assert smoke["cells"]["count_per_tier"] == 8
    assert smoke["cells"]["gpu_shape"] == "512x64"
    assert any("expands seen_goals" in item for item in smoke["pass_criteria"])
    assert smoke["authority"]["checkpoint_writes"] is False
    assert smoke["authority"]["main"] is False
