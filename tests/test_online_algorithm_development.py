import importlib.util
import json
from pathlib import Path


MANIFEST_PATH = Path(
    "docs/manifests/online_algorithm_expansion_v1_development.json"
)
RUNNER_PATH = Path("scripts/run_online_algorithm_development.py")


def _manifest():
    return json.loads(MANIFEST_PATH.read_text())


def _runner():
    spec = importlib.util.spec_from_file_location(
        "run_online_algorithm_development", RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_development_authority_stops_before_main_and_uses_raid():
    manifest = _manifest()
    authority = manifest["authority"]
    execution = manifest["execution"]
    assert authority["normal_only_development"] is True
    assert authority["checkpoint_writes"] is True
    assert authority["main_adaptation"] is False
    assert authority["dependency_changes"] is False
    assert execution["run_root"].startswith("/raid/ext_csv/HackRL/runs/")
    assert execution["device_environment"]["HACKRL_DEVICE"] == "cuda"
    assert execution["device_environment"]["HACKRL_DUMMY_KEEPALIVE_CLEARED"] == "1"
    assert execution["device_environment"]["PYTHONPATH"].endswith("/src")
    assert execution["main_launch"] == "forbidden and never automatic"


def test_candidate_registry_and_job_budget_are_frozen():
    manifest = _manifest()
    registry = manifest["candidate_registry"]
    methods = ("GC-PQN", "LEO", "Dual LEO(PQN)", "GC-SD-SAC")
    assert registry["frozen_before_any_development_result"] is True
    assert all(len(registry[method]) == 2 for method in methods)
    candidate_ids = [
        candidate["id"]
        for method in methods
        for candidate in registry[method]
    ]
    assert len(candidate_ids) == len(set(candidate_ids)) == 8
    budget = manifest["budget"]
    assert budget["jobs"] == 4 * 2 * 2 * 2 == 32
    assert budget["training_transitions"] == 32 * 16_777_216
    assert budget["automatic_extension"] is False
    assert manifest["execution"]["checkpoint_updates"] == [0, 128, 256, 384, 512]
    assert manifest["common_training"]["log_interval_updates"] == 32
    assert manifest["common_training"]["evaluation"]["threshold"] == 0.9


def test_capacity_forecast_preserves_required_reserve():
    manifest = _manifest()
    capacity = manifest["capacity_preflight"]
    forecast = capacity["forecast"]
    assert capacity["home"]["decision"] == "rejected for this new run"
    assert capacity["raid"]["decision"] == "selected"
    assert forecast["safety_reserve_bytes"] == max(
        8 * 1024**3,
        int(0.2 * forecast["projected_peak_write_bytes"]),
    )
    assert forecast["required_free_bytes"] == (
        forecast["projected_peak_write_bytes"]
        + forecast["safety_reserve_bytes"]
    )
    assert capacity["raid"]["available_bytes"] > forecast["required_free_bytes"]


def test_runner_builds_exact_unique_job_matrix():
    runner = _runner()
    manifest = _manifest()
    runner.validate_manifest_contract(manifest)
    jobs = runner.build_jobs(manifest)
    assert len(jobs) == 32
    assert len({job["id"] for job in jobs}) == 32
    assert {job["seed"] for job in jobs} == {110, 111}
    assert {job["environment"] for job in jobs} == {"tick", "pack"}
    assert {job["method"] for job in jobs} == {
        "GC-PQN",
        "LEO",
        "Dual LEO(PQN)",
        "GC-SD-SAC",
    }


def test_checkpoint_identity_rejects_corrupt_state(tmp_path):
    runner = _runner()
    manifest = _manifest()
    job = runner.build_jobs(manifest)[0]
    source_hashes = {"source": "abc"}
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    state = checkpoint / "state.msgpack"
    state.write_bytes(b"valid-state")
    metadata = {
        "schema_version": "hackrl_online_algorithm_development_checkpoint_v1",
        "job_id": job["id"],
        "method": job["method"],
        "environment": job["environment"],
        "candidate_id": job["candidate_id"],
        "seed": job["seed"],
        "update": 128,
        "environment_steps": 128 * 32768,
        "state_sha256": runner._sha256_file(state),
        "execution_source_hashes": source_hashes,
    }
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
    assert runner._checkpoint_complete(
        checkpoint, job, 128, source_hashes
    )
    state.write_bytes(b"corrupt-state")
    assert not runner._checkpoint_complete(
        checkpoint, job, 128, source_hashes
    )


def test_latest_checkpoint_stops_on_invalid_existing_directory(tmp_path):
    runner = _runner()
    job = runner.build_jobs(_manifest())[0]
    checkpoint = runner._checkpoint_dir(tmp_path, 128)
    checkpoint.mkdir(parents=True)
    try:
        runner.latest_checkpoint(tmp_path, job, {"source": "abc"}, [0, 128])
    except RuntimeError as error:
        assert "must be preserved" in str(error)
    else:
        raise AssertionError("invalid checkpoint was silently skipped")


def test_evaluation_contract_requires_exact_32_by_32_layout_phase_grid():
    runner = _runner()
    manifest = _manifest()
    records = []
    for family in ("natural_reset", "common_setup"):
        for state_index in range(32):
            records.append(
                {
                    "family": family,
                    "state_index": state_index,
                    "layout": state_index % 16,
                    "phase": state_index // 16,
                    "repeat": 0,
                    "success": state_index != 31,
                    "length": 10,
                }
            )
    evaluation = {
        "kernel": "fixed",
        "split": "validation",
        "natural_reset": {"episodes": 32, "success_count": 31},
        "common_setup": {"episodes": 32, "success_count": 31},
        "episode_records": records,
        "mutant_or_bug_metric_persisted": False,
        "learner_state_immutable": True,
    }
    assert runner._validate_evaluation(evaluation, manifest)
    evaluation["episode_records"] = records[:-1]
    assert not runner._validate_evaluation(evaluation, manifest)


def test_resume_log_is_deduplicated_and_truncated(tmp_path):
    runner = _runner()
    path = tmp_path / "updates.jsonl"
    rows = [
        {"update": 1, "value": "old"},
        {"update": 32, "value": "old"},
        {"update": 32, "value": "new"},
        {"update": 64, "value": "discard"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    runner._truncate_update_log(path, 32)
    kept = [json.loads(line) for line in path.read_text().splitlines()]
    assert kept == [
        {"update": 1, "value": "old"},
        {"update": 32, "value": "new"},
    ]


def test_adjudication_uses_only_fixed_normal_success_then_return(
    tmp_path, monkeypatch
):
    runner = _runner()
    manifest = _manifest()
    jobs = runner.build_jobs(manifest)
    for job in jobs:
        first = job["candidate_id"] == manifest["candidate_registry"][
            job["method"]
        ][0]["id"]
        success = 0.90625 if first else 0.875
        discounted_return = 0.5 if first else 0.9
        summary = {
            "execution_complete": True,
            "evaluation": {
                "natural_reset": {
                    "success_count": int(success * 32),
                    "success_rate": success,
                    "mean_discounted_return": discounted_return,
                }
            },
        }
        path = runner._cell_path(tmp_path, job) / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary))
    monkeypatch.setattr(runner, "_summary_complete", lambda *args: True)
    result = runner.adjudicate(tmp_path, manifest, jobs, {"source": "abc"})
    assert result["selection_used_only_fixed_normal_natural_start"] is True
    assert result["bug_or_mutant_metric_used"] is False
    assert result["passed_cells"] == result["total_cells"] == 8
    assert all(cell["passed"] for cell in result["cells"])
    assert all(cell["threshold"] == 0.9 for cell in result["cells"])
    assert all(cell["minimum_successes_per_32"] == 29 for cell in result["cells"])
    assert all(
        cell["selected_candidate_id"].endswith("v1")
        for cell in result["cells"]
    )
    assert result["main_automatically_launched"] is False
