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
    assert rerun["status"] == "authorized_ready_to_execute"
    assert execution["run_root"].startswith("/raid/ext_csv/HackRL/runs/")
    assert execution["clean_worktree_root"].startswith(
        "/raid/ext_csv/HackRL/worktrees/"
    )
    assert "seen_goals_fix_v1" in execution["run_root"]
    assert execution["run_root"] != _load(ORIGINAL)["execution"]["run_root"]
    assert rerun["smoke_manifest"] == str(SMOKE)


def test_post_fix_smoke_requires_full_shape_and_seen_goal_expansion():
    smoke = _load(SMOKE)
    assert smoke["status"] == "passed"
    assert smoke["cells"]["count_per_tier"] == 8
    assert smoke["cells"]["gpu_shape"] == "512x64"
    assert any("online-value cell" in item and "SD-SAC cell" in item for item in smoke["pass_criteria"])
    assert smoke["authority"]["checkpoint_writes"] is False
    assert smoke["authority"]["main"] is False


def test_historical_gate_is_machine_readably_invalidated_and_queue_is_fail_closed():
    historical = _load(
        Path("docs/manifests/online_algorithm_expansion_v1_development_results.json")
    )
    assert "invalidated_by_shared_wiring_defect" in historical["status"]
    queue = Path(
        "scripts/run_online_algorithm_seen_goals_fix_gpu_queue.sh"
    ).read_text(encoding="utf-8")
    assert "stop_identity_checked_gpu_keepalive.py" in queue
    assert "kill -TERM" not in queue
    assert "exit 66" in queue and "exit 75" in queue


def test_development_runner_binds_smoke_sources_and_aggregate_hash():
    runner = Path("scripts/run_online_algorithm_seen_goals_fix.py").read_text(
        encoding="utf-8"
    )
    assert "runtime_source_sha256" in runner
    assert "aggregate_sha256" in runner
    assert "base._require_smoke_passed = _require_bound_seen_goal_smoke" in runner
