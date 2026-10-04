import hashlib
import importlib.util
import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_mine_expedition_fixed_gate",
    REPOSITORY / "scripts/summarize_mine_expedition_fixed_gate.py",
)
GATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GATE)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _evaluation(sample_rate):
    def block(stochastic, episodes, success_rate):
        return {
            "variant": "fixed",
            "start": "natural",
            "stochastic": stochastic,
            "episodes": episodes,
            "success_rate": success_rate,
        }

    return {
        "runner_state_immutable": True,
        "mode": block(False, 1, 1.0),
        "sample": block(True, 128, sample_rate),
    }


def _materialize_seed(root, seed, sample_rate):
    manifest = json.loads(GATE.GATE_MANIFEST.read_text(encoding="utf-8"))
    contract = manifest["training_contract"]
    config = GATE._expected_config(contract, seed)
    authorized = manifest["authorized_source_sha256"]
    runtime = {"jax_backend": "gpu"}
    execution_sha = "a" * 40
    destination = root / f"seed{seed}"
    final_evaluation = _evaluation(sample_rate)
    _write(
        destination / "run_manifest.json",
        {
            "config": config,
            "execution_code_sha": execution_sha,
            "runtime": runtime,
            "execution_source_sha256": authorized,
        },
    )
    _write(
        destination / "summary.json",
        {
            "status": "complete",
            "seed": seed,
            "variant": "fixed",
            "evaluation_start": "natural",
            "updates": contract["num_updates"],
            "transitions": contract["transitions_per_seed"],
            "execution_code_sha": execution_sha,
            "runtime": runtime,
            "parameter_count": 2_656_969,
            "final_evaluation": final_evaluation,
        },
    )
    events = ["start_or_resume"]
    events.extend(
        f"before_checkpoint_{update}" for update in contract["checkpoint_updates"]
    )
    events.extend(
        f"periodic_update_{update}"
        for update in range(64, contract["num_updates"] + 1, 64)
    )
    projected = 1024
    reserve = 8 * 1024**3
    _write(
        destination / "capacity_checks.json",
        [
            {
                "event": event,
                "destination": str(destination.resolve()),
                "projected_remaining_write_bytes": projected,
                "safety_reserve_bytes": reserve,
                "required_free_bytes": projected + reserve,
                "target_filesystem": {"free_bytes": projected + reserve},
            }
            for event in events
        ],
    )
    batch_size = contract["num_envs"] * contract["num_steps"]
    for update in contract["checkpoint_updates"]:
        checkpoint = destination / "checkpoints" / f"update_{update}"
        checkpoint.mkdir(parents=True)
        state = f"seed={seed},update={update}".encode()
        (checkpoint / "state.msgpack").write_bytes(state)
        _write(checkpoint / "config.json", config)
        _write(
            checkpoint / "metadata.json",
            {
                "schema_version": "hackrl_mine_expedition_fixed_checkpoint_v1",
                "global_update": update,
                "environment_steps": update * batch_size,
                "variant": "fixed",
                "evaluation_start": "natural",
                "state_sha256": hashlib.sha256(state).hexdigest(),
                "parameter_count": 2_656_969,
            },
        )
        _write(
            checkpoint / "evaluation.json",
            final_evaluation if update == contract["num_updates"] else _evaluation(0.0),
        )


def test_adjudicator_records_a_predeclared_pass(tmp_path):
    for seed in (40, 41, 42):
        _materialize_seed(tmp_path, seed, 0.9)
    result = GATE.adjudicate(tmp_path, reevaluate=False)
    assert result["status"] == "pass"
    assert result["execution_complete"]
    assert result["gate_passed"]


def test_adjudicator_distinguishes_scientific_failure_from_incomplete_execution(tmp_path):
    for seed in (40, 41, 42):
        _materialize_seed(tmp_path, seed, 0.5)
    failed = GATE.adjudicate(tmp_path, reevaluate=False)
    assert failed["status"] == "fail"
    assert failed["execution_complete"]
    assert failed["gate_passed"] is False

    (tmp_path / "seed42" / "summary.json").unlink()
    incomplete = GATE.adjudicate(tmp_path, reevaluate=False)
    assert incomplete["status"] == "incomplete"
    assert not incomplete["execution_complete"]
    assert incomplete["gate_passed"] is None


def test_adjudicator_treats_malformed_evaluation_as_incomplete(tmp_path):
    for seed in (40, 41, 42):
        _materialize_seed(tmp_path, seed, 0.9)
    final_path = tmp_path / "seed40/checkpoints/update_2048/evaluation.json"
    malformed = json.loads(final_path.read_text(encoding="utf-8"))
    del malformed["mode"]
    _write(final_path, malformed)
    summary_path = tmp_path / "seed40/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["final_evaluation"] = malformed
    _write(summary_path, summary)
    result = GATE.adjudicate(tmp_path, reevaluate=False)
    assert result["status"] == "incomplete"
    assert result["gate_passed"] is None
    assert any("invalid mode evaluation" in error for error in result["errors"])
