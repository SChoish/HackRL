import hashlib
import importlib.util
import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_mine_expedition_fixed_curriculum_diagnostic",
    REPOSITORY
    / "scripts/summarize_mine_expedition_fixed_curriculum_diagnostic.py",
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
    sample_rate=0.9,
    mode_rate=1.0,
):
    manifest = json.loads(DIAGNOSTIC.MANIFEST.read_text(encoding="utf-8"))
    phase = manifest["phases"][phase_name]
    phase_dir = DIAGNOSTIC.PHASE_DIRECTORIES[phase_name]
    destination = root / phase_dir / f"seed{seed}"
    config = DIAGNOSTIC._expected_config(manifest, phase_name, seed)
    authorized = manifest["authorized_source_sha256"]
    execution_sha = "b" * 40
    initialization = {"kind": "random"}
    if phase_name == "stage_b_natural_late":
        initialization = {
            "kind": "fixed_checkpoint_transfer",
            "checkpoint": str(
                (
                    root
                    / "stage_a"
                    / f"seed{seed}"
                    / "checkpoints/update_512"
                ).resolve()
            ),
        }
    evaluation = _evaluation(sample_rate, mode_rate)
    _write(
        destination / "run_manifest.json",
        {
            "config": config,
            "execution_code_sha": execution_sha,
            "execution_source_sha256": authorized,
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
        qualified = rollout_success and update > phase["updates"] - window
        updates.append(
            {
                "update": update,
                "completed_episodes": 1 if update > phase["updates"] - window else 0,
                "completed_successes": int(qualified),
                "completed_timeouts": int(
                    update > phase["updates"] - window and not qualified
                ),
                "crafted_pickaxes": int(qualified),
                "mined_targets": int(qualified),
                "returned_targets": int(qualified),
            }
        )
    _write(destination / "updates.json", updates)
    projected = 1024
    reserve = 8 * 1024**3
    events = {"start_or_resume"}
    events.update(f"before_checkpoint_{x}" for x in phase["checkpoint_updates"])
    events.update(
        f"periodic_update_{x}"
        for x in range(64, phase["updates"] + 1, 64)
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
            for event in events
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


def test_stage_a_uses_rollouts_only_to_choose_the_predeclared_route(tmp_path):
    for seed, passed in zip((40, 41, 42), (True, True, False), strict=True):
        _materialize_phase(
            tmp_path,
            "stage_a_craft_ready",
            seed,
            rollout_success=passed,
            mode_rate=0.0,
            sample_rate=0.0,
        )
    result = DIAGNOSTIC.summarize(
        tmp_path, "stage_a_craft_ready", reevaluate=False
    )
    assert result["execution_complete"]
    assert result["status"] == "advance"
    assert result["qualified_seeds"] == [40, 41]
    assert result["fixed_natural_gate_passed"] is False
    assert result["mutant_training_authorized"] is False


def test_stage_a_failure_routes_to_target_ready_fallback(tmp_path):
    for seed, passed in zip((40, 41, 42), (True, False, False), strict=True):
        _materialize_phase(
            tmp_path,
            "stage_a_craft_ready",
            seed,
            rollout_success=passed,
        )
    result = DIAGNOSTIC.summarize(
        tmp_path, "stage_a_craft_ready", reevaluate=False
    )
    assert result["status"] == "fallback"
    assert result["phase_passed"] is False


def test_only_stage_b_natural_evaluation_can_open_the_fixed_gate(tmp_path):
    for seed, sample_rate in zip((40, 41, 42), (0.9, 0.8, 0.4), strict=True):
        _materialize_phase(
            tmp_path,
            "stage_b_natural_late",
            seed,
            rollout_success=False,
            sample_rate=sample_rate,
        )
    result = DIAGNOSTIC.summarize(
        tmp_path, "stage_b_natural_late", reevaluate=False
    )
    assert result["status"] == "pass"
    assert result["fixed_natural_gate_passed"]
    assert result["mutant_training_authorized"]


def test_target_ready_localization_never_opens_mutant_training(tmp_path):
    for seed in (40, 41, 42):
        _materialize_phase(
            tmp_path,
            "fallback_target_ready",
            seed,
            rollout_success=True,
        )
    result = DIAGNOSTIC.summarize(
        tmp_path, "fallback_target_ready", reevaluate=False
    )
    assert result["status"] == "localized"
    assert result["fixed_natural_gate_passed"] is False
    assert result["mutant_training_authorized"] is False
