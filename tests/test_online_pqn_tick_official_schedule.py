import json
from pathlib import Path


MANIFEST = Path("docs/manifests/online_pqn_tick_official_schedule_v1.json")
RUNNER = Path("scripts/run_online_pqn_tick_official_schedule.py")
WRAPPER = Path("scripts/run_online_pqn_tick_official_schedule_gpu.sh")
GUARD = Path("scripts/guard_identity_checked_gpu_keepalive.py")


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_schedule_restoration_is_exactly_two_fixed_tick_pqn_jobs():
    manifest = load_manifest()
    training = manifest["training"]
    assert training["method"] == "GC-PQN"
    assert training["environment"] == "TICK-CLAIM"
    assert training["kernel"] == "fixed"
    assert training["learner_seeds"] == [110, 111]
    assert manifest["budget"]["jobs"] == 2
    assert manifest["budget"]["automatic_extension"] is False
    assert manifest["authority"]["main_matrix"] is False


def test_transition_and_optimizer_arithmetic_matches_upstream_schedule_shape():
    training = load_manifest()["training"]
    assert training["num_envs"] * training["num_steps"] == 1024
    assert 1024 // training["minibatch_size"] == training["num_minibatches"] == 4
    assert training["update_epochs"] == 1
    assert training["updates"] * 1024 == training["transitions_per_job"] == 16_777_216
    assert (
        training["updates"]
        * training["num_minibatches"]
        * training["update_epochs"]
        == training["optimizer_applications_per_job"]
        == 65_536
    )


def test_storage_and_evaluation_scope_are_frozen():
    manifest = load_manifest()
    execution = manifest["execution"]
    assert execution["run_root"].startswith("/raid/ext_csv/HackRL/runs/")
    assert execution["checkpoint_updates"] == [0, 4096, 8192, 12288, 16384]
    assert manifest["capacity_preflight"]["home"]["decision"] == "rejected for checkpoint writes"
    assert manifest["capacity_preflight"]["raid"]["decision"] == "selected"
    assert manifest["evaluation"]["bug_and_mutant_metrics_persisted_or_used"] is False
    assert manifest["evaluation"]["natural_start_episodes"] == 32


def test_normalization_diagnostic_is_hash_bound_and_non_training():
    diagnostic = load_manifest()["normalization_diagnostic"]
    assert len(diagnostic["output_sha256"]) == 64
    assert diagnostic["observed_result"]["checkpoints"] == 4
    assert diagnostic["observed_result"]["all_below_correction_threshold_1000"] is True
    low, high = diagnostic["observed_result"]["greedy_action_disagreement_rate_range"]
    assert 0.84 < low <= high < 0.98
    assert diagnostic["observed_result"]["checkpoint_hashes_unchanged"] is True


def test_runner_and_gpu_guard_bind_minibatches_and_identity_checks():
    runner = RUNNER.read_text(encoding="utf-8")
    wrapper = WRAPPER.read_text(encoding="utf-8")
    guard = GUARD.read_text(encoding="utf-8")
    assert "minibatch_size=training[\"minibatch_size\"]" in runner
    assert "update_epochs=training[\"update_epochs\"]" in runner
    assert "optimizer_applications_per_job" in runner
    assert "stop_identity_checked_gpu_keepalive.py" in wrapper
    assert "--beyondg-gpu-keepalive" not in wrapper
    assert "identity.validate_process" in guard
    assert "signal.pidfd_send_signal(runner[\"pidfd\"], signal.SIGTERM)" in guard
    assert "except BaseException as error" in guard
    assert "wait -n -p FINISHED_PID" in wrapper
    assert "HACKRL_GPU_GUARD_ACTIVE=1" in wrapper
