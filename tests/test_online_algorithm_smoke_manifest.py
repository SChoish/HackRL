import json
from pathlib import Path


MANIFEST = Path("docs/manifests/online_algorithm_expansion_v1_smoke.json")


def _manifest():
    return json.loads(MANIFEST.read_text())


def test_smoke_authority_excludes_training_and_checkpoint_writes():
    authority = _manifest()["authorization"]
    assert authority["cpu_smoke"]
    assert authority["gpu_smoke_after_cpu_pass"]
    assert authority["development"] is False
    assert authority["main"] is False
    assert authority["queue_launch"] is False
    assert authority["checkpoint_writes"] is False
    assert authority["dependency_changes"] is False
    assert authority["implementation_bugfixes"] is False
    assert authority["process_termination"] is True
    assert "--beyondg-gpu-keepalive" in authority["process_termination_scope"]
    assert authority["temporary_logs_under_tmp"] is True


def test_smoke_matrix_and_transition_arithmetic_are_fixed():
    manifest = _manifest()
    cells = manifest["cells"]
    assert cells["count"] == len(cells["methods"]) * len(cells["environments"]) == 8
    cpu = manifest["cpu"]
    gpu = manifest["gpu"]
    assert cpu["physical_transitions_per_cell"] == cpu["num_envs"] * cpu["num_steps"] == 8
    assert cpu["physical_transitions_total"] == cells["count"] * 8 == 64
    assert gpu["physical_transitions_per_cell"] == gpu["num_envs"] * gpu["num_steps"] == 32768
    assert gpu["physical_transitions_total"] == cells["count"] * 32768 == 262144
    assert gpu["checkpoint_retention"] is False


def test_smoke_execution_contract_is_sequential_reproducible_and_noninterfering():
    manifest = _manifest()
    identity = manifest["execution_identity"]
    gpu = manifest["gpu"]
    contract = manifest["execution_contract"]
    assert identity["host"] == "ext_csv-box"
    assert len(identity["base_git_sha"]) == 40
    assert "all eight CPU cells" in identity["result_invalidation"]
    assert gpu["device_index"] == 0
    assert gpu["device_uuid"].startswith("GPU-")
    assert "immediately before every cell" in gpu["fresh_entry_check"]
    assert "never signal" in gpu["noninterference"]
    assert "sequentially" in contract["order"]
    assert contract["success_exit_code"] == 0
    assert contract["timeout_seconds_per_cell"] == 1800
    assert contract["workdir"] == "/home/ext_csv/HackRL"
    assert contract["cpu"]["environment"]["JAX_PLATFORMS"] == "cpu"
    assert contract["gpu"]["environment"]["JAX_PLATFORMS"] == "cuda"
    assert contract["gpu"]["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert "/tmp" in contract["output"]
    assert "transient in-memory replay" in contract["replay_scope"]
    assert "does not authorize source edits" in manifest["failure_rule"]
    assert "additional process termination" in manifest["failure_rule"]


def test_smoke_values_cannot_open_or_populate_development():
    manifest = _manifest()
    boundary = manifest["smoke_only_optimizer_inputs"]["selection_boundary"]
    assert "not development candidates" in boundary
    assert "cannot populate" in boundary
    cpu = manifest["results"]["cpu"]
    gpu = manifest["results"]["gpu"]
    assert cpu["status"] == "passed"
    assert cpu["cells_passed"] == cpu["cells_total"] == 8
    assert cpu["physical_transitions"] == 64
    assert cpu["checkpoint_written"] is False
    assert gpu["status"] == "passed"
    assert gpu["process_terminated"] is True
    assert gpu["terminated_pid"] == 107
    assert gpu["failed_attempts"][0]["physical_transitions"] == 0
    assert gpu["cells_passed"] == gpu["cells_total"] == 8
    assert gpu["physical_transitions"] == 262144
    assert gpu["checkpoint_written"] is False
    assert gpu["finite_state_and_metrics"] is True
