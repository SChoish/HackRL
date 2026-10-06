import hashlib
import importlib.util
import json
from pathlib import Path

import jax

from hackrl.mine_expedition_env import (
    MineExpeditionStart,
    mine_expedition_start_candidates,
)
from hackrl.mine_expedition_ppo import (
    MineExpeditionPPOConfig,
    initialize_mine_expedition_ppo,
    make_mine_expedition_update,
)


REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST = (
    REPOSITORY
    / "docs/manifests/mine_expedition_fixed_return_curriculum_v1.json"
)
AUTHORIZED_EXECUTION_SHA = "c" * 40
AUTHORIZED_MANIFEST_SHA256 = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
SPEC = importlib.util.spec_from_file_location(
    "summarize_mine_expedition_fixed_return_curriculum",
    REPOSITORY
    / "scripts/summarize_mine_expedition_fixed_return_curriculum.py",
)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DIAGNOSTIC)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _evaluation(sample_rate, mode_rate=1.0):
    def block(stochastic, episodes, rate):
        return {
            "variant": "fixed",
            "start": "natural",
            "stochastic": stochastic,
            "episodes": episodes,
            "success_rate": rate,
        }

    return {
        "runner_state_immutable": True,
        "mode": block(False, 1, mode_rate),
        "sample": block(True, 128, sample_rate),
    }


def _materialize_phase(
    root,
    phase_name,
    seed,
    *,
    rollout_success,
    sample_rate=0.95,
    mode_rate=1.0,
):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    phase = manifest["phases"][phase_name]
    destination = root / phase["directory"] / f"seed{seed}"
    config = DIAGNOSTIC._expected_config(manifest, phase_name, seed)
    execution_sha = AUTHORIZED_EXECUTION_SHA
    previous = DIAGNOSTIC._expected_initialization(
        root.resolve(), manifest, phase_name, seed
    )
    initialization = {"kind": "random"}
    if previous is not None:
        previous_index = manifest["phase_order"].index(phase_name) - 1
        previous_name = manifest["phase_order"][previous_index]
        previous_config = DIAGNOSTIC._expected_config(
            manifest, previous_name, seed
        )
        if (previous / "state.msgpack").is_file():
            source_digest = hashlib.sha256(
                (previous / "state.msgpack").read_bytes()
            ).hexdigest()
            source_execution_sha = json.loads(
                (previous.parent.parent / "run_manifest.json").read_text(
                    encoding="utf-8"
                )
            )["execution_code_sha"]
        else:
            source_state = f"{previous_name}:{seed}".encode()
            source_digest = hashlib.sha256(source_state).hexdigest()
            source_execution_sha = execution_sha
            previous.mkdir(parents=True, exist_ok=True)
            (previous / "state.msgpack").write_bytes(source_state)
            _write(previous / "config.json", previous_config)
            _write(
                previous / "metadata.json",
                {"state_sha256": source_digest},
            )
            _write(
                previous.parent.parent / "run_manifest.json",
                {"execution_code_sha": source_execution_sha},
            )
        initialization = {
            "kind": "fixed_checkpoint_transfer",
            "checkpoint": str(previous),
            "state_sha256": source_digest,
            "source_execution_code_sha": source_execution_sha,
            "source_config": previous_config,
            "preserved": ["policy", "critic", "adam", "action_rng"],
        }
    evaluation = _evaluation(sample_rate, mode_rate)
    execution_sources = {
        **manifest["authorized_source_sha256"],
        str(MANIFEST.relative_to(REPOSITORY)): AUTHORIZED_MANIFEST_SHA256,
    }
    _write(
        destination / "run_manifest.json",
        {
            "config": config,
            "execution_code_sha": execution_sha,
            "execution_source_sha256": execution_sources,
            "initialization": initialization,
        },
    )
    _write(
        destination / "summary.json",
        {
            "status": "complete",
            "seed": seed,
            "variant": "fixed",
            "training_start": phase["training_start"],
            "evaluation_start": "natural",
            "updates": phase["updates"],
            "transitions": phase["transitions_per_seed"],
            "execution_code_sha": execution_sha,
            "final_evaluation": evaluation,
        },
    )
    window = int(phase.get("route_window_updates", 128))
    updates = []
    for update in range(1, phase["updates"] + 1):
        in_window = update > phase["updates"] - window
        successful = rollout_success and in_window
        updates.append(
            {
                "update": update,
                "completed_episodes": int(in_window),
                "completed_successes": int(successful),
                "completed_timeouts": int(in_window and not successful),
                "crafted_pickaxes": int(successful),
                "mined_targets": int(successful),
                "returned_targets": int(successful),
            }
        )
    _write(destination / "updates.json", updates)
    projected = 1024
    reserve = 8 * 1024**3
    events = {"start_or_resume"}
    events.update(f"before_checkpoint_{x}" for x in phase["checkpoint_updates"])
    events.update(
        f"periodic_update_{x}" for x in range(64, phase["updates"] + 1, 64)
    )
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
            for event in sorted(events)
        ],
    )
    checkpoint = destination / "checkpoints" / f"update_{phase['updates']}"
    state = f"{phase_name}:{seed}".encode()
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.msgpack").write_bytes(state)
    _write(checkpoint / "config.json", config)
    _write(
        checkpoint / "metadata.json",
        {
            "global_update": phase["updates"],
            "environment_steps": phase["transitions_per_seed"],
            "variant": "fixed",
            "evaluation_start": "natural",
            "state_sha256": hashlib.sha256(state).hexdigest(),
        },
    )
    _write(checkpoint / "evaluation.json", evaluation)


def test_manifest_budget_and_cumulative_reset_contract_match_code():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert sum(
        phase["updates"] for phase in manifest["phases"].values()
    ) == manifest["budget"]["updates_per_seed"] == 4096
    assert (
        manifest["budget"]["transitions_per_seed"]
        == manifest["budget"]["updates_per_seed"]
        * manifest["optimizer"]["num_envs"]
        * manifest["optimizer"]["num_steps"]
    )
    mapping = {
        "return_near": MineExpeditionStart.RETURN_NEAR,
        "return_path": MineExpeditionStart.RETURN_PATH,
        "mine_return": MineExpeditionStart.MINE_RETURN,
        "craft_mine_return": MineExpeditionStart.CRAFT_MINE_RETURN,
        "natural_full": MineExpeditionStart.NATURAL_RETURN,
    }
    for phase_name in manifest["phase_order"]:
        ticks = tuple(
            int(state.tick)
            for state in mine_expedition_start_candidates(mapping[phase_name])
        )
        assert ticks == tuple(manifest["phases"][phase_name]["candidate_ticks"])


def test_return_near_one_update_runs_and_keeps_fixed_contract():
    config = MineExpeditionPPOConfig(
        seed=7,
        num_envs=2,
        num_steps=4,
        num_updates=1,
        update_epochs=1,
        minibatch_size=8,
        hidden_size=16,
        training_start="return_near",
        mode_eval_episodes=1,
        sample_eval_episodes=2,
    )
    network, runner = initialize_mine_expedition_ppo(config)
    updated, metrics = jax.jit(make_mine_expedition_update(network, config))(runner)
    assert int(updated.global_update) == 1
    assert int(updated.env_steps) == config.batch_size
    assert int(metrics["transitions"]) == config.batch_size
    assert int(metrics["iron_increase_events"]) == 0
    assert int(metrics["indirect_use_events"]) == 0


def test_phase_rejects_wrong_manifest_digest_and_mixed_execution_sha(tmp_path):
    for seed in (40, 41, 42):
        _materialize_phase(
            tmp_path,
            "return_near",
            seed,
            rollout_success=True,
        )

    wrong_manifest = DIAGNOSTIC.summarize_phase(
        tmp_path,
        "return_near",
        expected_execution_sha=AUTHORIZED_EXECUTION_SHA,
        expected_manifest_sha256="0" * 64,
        reevaluate=False,
    )
    assert not wrong_manifest["execution_complete"]
    assert any("externally authorized digest" in x for x in wrong_manifest["errors"])

    seed42 = tmp_path / "phase_1_return_near/seed42"
    for name in ("summary.json", "run_manifest.json"):
        path = seed42 / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["execution_code_sha"] = "d" * 40
        _write(path, payload)
    mixed_execution = DIAGNOSTIC.summarize_phase(
        tmp_path,
        "return_near",
        expected_execution_sha=AUTHORIZED_EXECUTION_SHA,
        expected_manifest_sha256=AUTHORIZED_MANIFEST_SHA256,
        reevaluate=False,
    )
    assert not mixed_execution["execution_complete"]
    assert any(
        "externally authorized SHA" in x for x in mixed_execution["errors"]
    )


def test_early_phase_advances_only_when_all_three_seeds_qualify(tmp_path):
    for seed, passed in zip((40, 41, 42), (True, True, False), strict=True):
        _materialize_phase(
            tmp_path,
            "return_path",
            seed,
            rollout_success=passed,
            sample_rate=0.0,
            mode_rate=0.0,
        )
    result = DIAGNOSTIC.summarize_phase(
        tmp_path,
        "return_path",
        expected_execution_sha=AUTHORIZED_EXECUTION_SHA,
        expected_manifest_sha256=AUTHORIZED_MANIFEST_SHA256,
        reevaluate=False,
    )
    assert result["execution_complete"]
    assert result["status"] == "stop"
    assert result["qualified_seeds"] == [40, 41]
    assert not result["fixed_natural_gate_passed"]
    assert not result["mutant_training_authorized"]


def test_final_gate_requires_mode_and_ninety_percent_for_every_seed(tmp_path):
    for seed, sample_rate, mode_rate in (
        (40, 0.95, 1.0),
        (41, 0.90, 1.0),
        (42, 0.95, 1.0),
    ):
        _materialize_phase(
            tmp_path,
            "natural_full",
            seed,
            rollout_success=False,
            sample_rate=sample_rate,
            mode_rate=mode_rate,
        )
    passed = DIAGNOSTIC.summarize_phase(
        tmp_path,
        "natural_full",
        expected_execution_sha=AUTHORIZED_EXECUTION_SHA,
        expected_manifest_sha256=AUTHORIZED_MANIFEST_SHA256,
        reevaluate=False,
    )
    assert passed["status"] == "pass"
    assert passed["qualified_seeds"] == [40, 41, 42]
    assert passed["fixed_natural_gate_passed"]
    assert not passed["mutant_training_authorized"]

    checkpoint = (
        tmp_path
        / "phase_5_natural_full/seed42/checkpoints/update_2048/evaluation.json"
    )
    failed_evaluation = _evaluation(0.89, 1.0)
    _write(checkpoint, failed_evaluation)
    summary = tmp_path / "phase_5_natural_full/seed42/summary.json"
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    summary_payload["final_evaluation"] = failed_evaluation
    _write(summary, summary_payload)
    failed = DIAGNOSTIC.summarize_phase(
        tmp_path,
        "natural_full",
        expected_execution_sha=AUTHORIZED_EXECUTION_SHA,
        expected_manifest_sha256=AUTHORIZED_MANIFEST_SHA256,
        reevaluate=False,
    )
    assert failed["status"] == "fail"
    assert failed["qualified_seeds"] == [40, 41]
    assert not failed["fixed_natural_gate_passed"]
    assert not failed["mutant_training_authorized"]


def test_only_full_ordered_aggregate_can_authorize_mutant_training(tmp_path):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    phase_results = tmp_path / "phase_results"
    for phase_name in manifest["phase_order"]:
        for seed in manifest["fixed_contract"]["seeds"]:
            _materialize_phase(
                tmp_path,
                phase_name,
                seed,
                rollout_success=True,
                sample_rate=0.95,
                mode_rate=1.0,
            )
        phase_result = DIAGNOSTIC.summarize_phase(
            tmp_path,
            phase_name,
            expected_execution_sha=AUTHORIZED_EXECUTION_SHA,
            expected_manifest_sha256=AUTHORIZED_MANIFEST_SHA256,
            reevaluate=False,
        )
        assert phase_result["phase_passed"]
        assert not phase_result["mutant_training_authorized"]
        _write(phase_results / f"{phase_name}.json", phase_result)

    result = DIAGNOSTIC.summarize_experiment(
        tmp_path,
        expected_execution_sha=AUTHORIZED_EXECUTION_SHA,
        expected_manifest_sha256=AUTHORIZED_MANIFEST_SHA256,
        reevaluate=False,
    )
    assert result["execution_complete"]
    assert result["status"] == "pass"
    assert result["fixed_natural_gate_passed"]
    assert result["mutant_training_authorized"]


def test_aggregate_accepts_predeclared_early_scientific_stop(tmp_path):
    phase_results = tmp_path / "phase_results"
    for seed in (40, 41, 42):
        _materialize_phase(
            tmp_path,
            "return_near",
            seed,
            rollout_success=True,
        )
    return_near = DIAGNOSTIC.summarize_phase(
        tmp_path,
        "return_near",
        expected_execution_sha=AUTHORIZED_EXECUTION_SHA,
        expected_manifest_sha256=AUTHORIZED_MANIFEST_SHA256,
        reevaluate=False,
    )
    _write(phase_results / "return_near.json", return_near)
    for seed, passed in zip((40, 41, 42), (True, True, False), strict=True):
        _materialize_phase(
            tmp_path,
            "return_path",
            seed,
            rollout_success=passed,
        )
    return_path = DIAGNOSTIC.summarize_phase(
        tmp_path,
        "return_path",
        expected_execution_sha=AUTHORIZED_EXECUTION_SHA,
        expected_manifest_sha256=AUTHORIZED_MANIFEST_SHA256,
        reevaluate=False,
    )
    _write(phase_results / "return_path.json", return_path)
    result = DIAGNOSTIC.summarize_experiment(
        tmp_path,
        expected_execution_sha=AUTHORIZED_EXECUTION_SHA,
        expected_manifest_sha256=AUTHORIZED_MANIFEST_SHA256,
        reevaluate=False,
    )
    assert result["execution_complete"]
    assert result["status"] == "scientific_fail_return_path"
    assert result["completed_phases"] == ["return_near", "return_path"]
    assert not result["fixed_natural_gate_passed"]
    assert not result["mutant_training_authorized"]

    tampered = {**return_path, "phase_passed": True, "status": "advance"}
    _write(phase_results / "return_path.json", tampered)
    rejected = DIAGNOSTIC.summarize_experiment(
        tmp_path,
        expected_execution_sha=AUTHORIZED_EXECUTION_SHA,
        expected_manifest_sha256=AUTHORIZED_MANIFEST_SHA256,
        reevaluate=False,
    )
    assert not rejected["execution_complete"]
    assert not rejected["fixed_natural_gate_passed"]
    assert not rejected["mutant_training_authorized"]
    assert "differs from recomputed evidence" in rejected["errors"][0]
