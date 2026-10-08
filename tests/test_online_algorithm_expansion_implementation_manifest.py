import json
from pathlib import Path


MANIFEST = Path("docs/manifests/online_algorithm_expansion_v1_implementation.json")


def _manifest():
    return json.loads(MANIFEST.read_text())


def test_current_stage_authorizes_only_implementation_and_unit_tests():
    authority = _manifest()["authority"]
    assert authority["implementation"]
    assert authority["unit_tests"]
    for key in (
        "cpu_smoke",
        "gpu_smoke",
        "development",
        "main",
        "queue_launch",
        "checkpoint_writes",
        "dependency_changes",
    ):
        assert authority[key] is False


def test_pqn_and_dual_contracts_are_frozen_without_hidden_bc_or_q_lambda():
    contracts = _manifest()["method_contracts"]
    assert contracts["GC-PQN"]["replay"] is False
    assert contracts["GC-PQN"]["target_network"] is False
    assert contracts["GC-PQN"]["q_lambda"] is False
    dual = contracts["Dual LEO(PQN)"]
    assert dual["acting_equation"] == "Q_act = 0.7*Q_PQN + 0.3*Q_LEO"
    assert dual["leo_weight"] == 0.3
    assert dual["anneal"] is False
    assert dual["behavior_cloning"] is False


def test_gate_and_all_evaluation_budget_arithmetic_are_mechanical():
    manifest = _manifest()
    gate = manifest["normal_gate_denominator"]
    assert gate["episodes_per_learner_seed"] == 16 * 2
    assert gate["integer_requirement"] == ">= 29 successes out of 32"
    budget = manifest["evaluation_and_smoke_budget"]
    assert budget["development_per_candidate_seed_cell"]["episodes"] == 32 + 32
    assert budget["development_per_candidate_seed_cell"]["maximum_physical_transitions"] == 64 * 128
    main = budget["main_per_seed_cell"]
    assert main["episodes"] == 2 * 8 * 2 * (64 + 256)
    assert main["maximum_physical_transitions"] == main["episodes"] * 128
    full = budget["full_main"]
    assert full["episodes"] == 24 * main["episodes"]
    assert full["maximum_physical_transitions"] == full["episodes"] * 128
    assert budget["cpu_smoke_future_stage"]["physical_transitions"] == 8 * 8
    assert budget["gpu_smoke_future_stage"]["physical_transitions"] == 8 * 32768
