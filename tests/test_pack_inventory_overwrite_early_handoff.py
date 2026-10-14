import importlib.util
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))


def _load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load_script(
    "evaluate_pack_inventory_overwrite_early_handoff_test",
    "scripts/evaluate_pack_inventory_overwrite_early_handoff.py",
)
MANIFEST = json.loads(
    (
        REPOSITORY
        / "docs/manifests/pack_inventory_overwrite_early_handoff_v1.json"
    ).read_text(encoding="utf-8")
)


def test_manifest_locks_one_zero_learning_mode_and_320_episodes():
    assert MANIFEST["status"] == "frozen_before_execution"
    assert MANIFEST["seeds"] == [35, 36, 37, 38, 39]
    assert list(MANIFEST["new_execution_mode"]) == ["carry_gated"]
    assert MANIFEST["comparison_modes_from_prior_run"] == [
        "source",
        "persistent",
        "adapted",
    ]
    assert MANIFEST["evaluation"]["direct_kernels"] == [
        "fixed",
        "inventory_overwrite_p1",
    ]
    assert MANIFEST["budget"]["learning_updates"] == 0
    assert MANIFEST["budget"]["episodes"] == 320
    assert MANIFEST["budget"]["maximum_environment_steps"] == 40960
    assert MANIFEST["budget"]["automatic_extension"] is False


def test_contract_arithmetic_and_raid_storage_are_fixed():
    assert len(EVALUATOR.SEEDS) * len(EVALUATOR.KERNELS) * 32 == 320
    assert 320 * EVALUATOR.HORIZON == 40960
    assert MANIFEST["storage"]["chosen_filesystem"] == "/raid/ext_csv"
    assert MANIFEST["storage"]["home_utilization_percent_at_preflight"] >= 90
    assert MANIFEST["storage"]["checkpoints_written"] == 0


def test_execution_sources_include_prior_evaluator_and_kernel_dependencies():
    assert set(MANIFEST["execution_sources"]) == {
        "docs/manifests/pack_inventory_overwrite_early_handoff_v1.json",
        "scripts/evaluate_pack_inventory_overwrite_early_handoff.py",
        "scripts/evaluate_pack_inventory_overwrite_policy_switch.py",
        "src/hackrl/pack_restore.py",
        "src/hackrl/pack_restore_gc.py",
        "src/hackrl/pack_inventory_overwrite_rare.py",
        "src/hackrl/dual_leo.py",
    }


def test_prerequisite_hashes_remain_pinned():
    records = EVALUATOR.validate_prerequisites()
    assert set(records) == {"prior_design", "prior_result", "prior_complete"}
