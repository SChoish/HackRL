import importlib.util
import json
import sys
from pathlib import Path
import pytest



REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_gc_double_dqn",
        REPOSITORY / "scripts" / "run_gc_double_dqn.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = _load()


def test_main_job_graph_is_exactly_six_pretrains_and_twelve_branches():
    jobs = RUNNER.build_jobs()
    assert len(jobs) == 18
    assert {job["env"] for job in jobs} == {"tick", "pack"}
    assert {job["seed"] for job in jobs} == {40, 41, 42}
    pretrain = [job for job in jobs if job["kind"] == "pretrain"]
    adapt = [job for job in jobs if job["kind"] == "adapt"]
    assert len(pretrain) == 6
    assert len(adapt) == 12
    assert {job["variant"] for job in adapt} == {"fixed", "mutant"}
    assert all(len(job["depends_on"]) == 1 for job in adapt)


def test_main_transition_budget_matches_declared_171_billion():
    transitions = (
        6
        * RUNNER.PRETRAIN_UPDATES
        * RUNNER.TRANSITIONS_PER_UPDATE
        + 12
        * RUNNER.ADAPT_UPDATES
        * RUNNER.TRANSITIONS_PER_UPDATE
    )
    assert transitions == 1_711_276_032


def test_replay_and_gradient_contract_records_compute_separately():
    assert RUNNER.REPLAY_CAPACITY == 65_536
    assert RUNNER.REPLAY_BATCH_SIZE == 1_024
    assert RUNNER.GRADIENT_STEPS_PER_ROLLOUT == 32
    assert (
        RUNNER.ALGORITHM_CONFIG[
            "sampled_transitions_per_generated_transition"
        ]
        == 1.0
    )
    assert RUNNER.TARGET_UPDATE_INTERVAL == 1_024
    assert RUNNER.EPSILON_DECAY_TRANSITIONS == 13_421_772
    assert RUNNER.ALGORITHM_CONFIG["replay_storage"][
        "invalid_transitions_excluded_from_learning_samples"
    ] is True



def test_storage_forecast_keeps_raid_reserve():
    assert RUNNER.DEFAULT_RUN_ROOT.is_relative_to(Path("/raid/ext_csv"))
    assert RUNNER.RETAINED_FULL_CHECKPOINTS == 18
    assert RUNNER.SAFETY_RESERVE_BYTES >= 8 * 1024**3
    assert (
        RUNNER.REQUIRED_FREE_BYTES
        == RUNNER.PROJECTED_PEAK_WRITE_BYTES
        + RUNNER.SAFETY_RESERVE_BYTES
    )


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _checkpoint(path, update):
    path.mkdir(parents=True, exist_ok=True)
    (path / "state.msgpack").write_bytes(b"state")
    _write_json(path / "config.json", {"algorithm": "test"})
    _write_json(path / "metadata.json", {"global_update": update})


def _view(success, violation, length, discounted_return):
    return {
        "success_rate": success,
        "violation_delivery_rate": violation,
        "mean_length": length,
        "mean_discounted_return": discounted_return,
    }


def test_main_summary_keeps_seed_paired_u_and_same_policy_kernel_effect(tmp_path):
    seed = 40
    for env in ("tick", "pack"):
        pretrain_cell = RUNNER._pretrain_root(tmp_path, env, seed)
        pretrain_checkpoint = pretrain_cell / "checkpoints" / "update_512"
        _checkpoint(pretrain_checkpoint, RUNNER.PRETRAIN_UPDATES)
        fingerprint = {"sha256": f"{env}-pretrain"}
        _write_json(
            pretrain_cell / "summary.json",
            {
                "execution_complete": True,
                "global_update": RUNNER.PRETRAIN_UPDATES,
                "checkpoint": str(pretrain_checkpoint),
                "checkpoint_fingerprint": fingerprint,
            },
        )
        for trained_variant, violation in (("fixed", 0.25), ("mutant", 0.75)):
            cell = RUNNER._adapt_root(tmp_path, env, trained_variant, seed)
            final_update = RUNNER.PRETRAIN_UPDATES + RUNNER.ADAPT_UPDATES
            checkpoint = cell / "checkpoints" / f"update_{final_update}"
            _checkpoint(checkpoint, final_update)
            _write_json(
                cell / "summary.json",
                {
                    "execution_complete": True,
                    "global_update": final_update,
                    "checkpoint": str(checkpoint),
                    "source_checkpoint_fingerprint": fingerprint,
                },
            )
            for update in RUNNER.SCIENCE_UPDATES:
                current_violation = (
                    violation if update == RUNNER.ADAPT_UPDATES else 0.0
                )
                curve = {
                    "adaptation_updates": update,
                    "fixed": {
                        "natural_reset": _view(0.8, 0.0, 20.0, 0.7),
                    },
                    "mutant": {
                        "natural_reset": _view(
                            0.9, current_violation, 15.0, 0.8
                        ),
                    },
                }
                _write_json(
                    cell / "curve" / f"adapt_{update}.json", curve
                )

    result = RUNNER.summarize_main(tmp_path, seeds=(seed,))
    assert result["execution_complete"] is True
    assert all(
        row["mean_paired_u"] == 0.5
        for row in result["primary_exploitation"]
    )
    assert all(
        abs(row["mean_success_difference"] - 0.1) < 1e-12
        for row in result["same_policy_kernel_effects"]
    )
    assert all(
        row["first_saved_adaptation_update"] == RUNNER.ADAPT_UPDATES
        for row in result["first_saved_exploitation"]
    )


def test_development_contract_records_two_jobs_and_rejects_main_reuse(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        RUNNER, "_require_clean_execution_sources", lambda: None
    )
    monkeypatch.setattr(
        RUNNER,
        "_capacity_snapshot",
        lambda run_root: {
            "chosen_run_root": str(Path(run_root).resolve()),
            "required_free_bytes": 1,
        },
    )
    RUNNER._prepare_run_root(tmp_path, mode="development")
    contract = json.loads(
        (tmp_path / "run_contract.json").read_text(encoding="utf-8")
    )
    assert contract["run_id"] == "gc_double_dqn_development_v1"
    assert contract["mode"] == "development"
    assert contract["budget"]["total_jobs"] == 2
    assert contract["budget"]["adaptation_jobs"] == 0
    assert len(contract["jobs"]) == 2
    with pytest.raises(RuntimeError, match="different run contract"):
        RUNNER._prepare_run_root(tmp_path, mode="main")



def test_manifest_budget_and_algorithm_match_runtime_contract():
    manifest = json.loads(
        (
            REPOSITORY
            / "docs/manifests/gc_double_dqn_two_defects_v1.json"
        ).read_text(encoding="utf-8")
    )
    algorithm = manifest["algorithm"]
    assert algorithm["replay"]["capacity"] == RUNNER.REPLAY_CAPACITY
    assert algorithm["replay"]["batch_size"] == RUNNER.REPLAY_BATCH_SIZE
    assert (
        algorithm["replay"][
            "gradient_steps_per_32768_generated_transitions"
        ]
        == RUNNER.GRADIENT_STEPS_PER_ROLLOUT
    )
    assert (
        algorithm["target_network"]["interval_gradient_steps"]
        == RUNNER.TARGET_UPDATE_INTERVAL
    )
    assert (
        manifest["budget"]["total_transitions"]
        == 6
        * RUNNER.PRETRAIN_UPDATES
        * RUNNER.TRANSITIONS_PER_UPDATE
        + 12
        * RUNNER.ADAPT_UPDATES
        * RUNNER.TRANSITIONS_PER_UPDATE
    )
