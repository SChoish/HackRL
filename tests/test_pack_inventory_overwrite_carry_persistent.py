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
    "evaluate_pack_inventory_overwrite_carry_persistent_test",
    "scripts/evaluate_pack_inventory_overwrite_carry_persistent.py",
)
MANIFEST = json.loads(
    (
        REPOSITORY
        / "docs/manifests/pack_inventory_overwrite_carry_persistent_v1.json"
    ).read_text(encoding="utf-8")
)


def test_manifest_locks_one_zero_learning_mode_and_160_paths():
    assert MANIFEST["status"] == "frozen_before_execution"
    assert MANIFEST["seeds"] == [35, 36, 37, 38, 39]
    assert list(MANIFEST["new_mode"]) == ["carry_persistent"]
    assert MANIFEST["evaluation"]["kernel"] == "fixed"
    assert MANIFEST["evaluation"]["exact_overwrite_probabilities"] == [0.0, 0.05]
    assert MANIFEST["budget"]["learning_updates"] == 0
    assert MANIFEST["budget"]["episodes"] == 160
    assert MANIFEST["budget"]["maximum_environment_steps"] == 20480
    assert MANIFEST["budget"]["automatic_extension"] is False


def test_primary_contrast_and_seed35_sensitivity_are_predeclared():
    assert MANIFEST["analysis"]["primary_metric"] == "C_minus_B_p05_expected_discounted_return"
    assert "seed-35-excluded" in MANIFEST["analysis"]["report"]
    assert MANIFEST["comparison"]["primary"].startswith("At p=0.05, paired C minus B")


def test_contract_arithmetic_and_raid_storage_are_fixed():
    assert len(EVALUATOR.SEEDS) * 32 == 160
    assert 160 * EVALUATOR.HORIZON == 20480
    assert MANIFEST["storage"]["chosen_filesystem"] == "/raid/ext_csv"
    assert MANIFEST["storage"]["home_utilization_percent_at_preflight"] >= 90
    assert MANIFEST["storage"]["checkpoints_written"] == 0


def test_prerequisite_hashes_remain_pinned():
    records = EVALUATOR.validate_prerequisites()
    assert set(records) == {
        "late_result",
        "carry_gated_design",
        "carry_gated_result",
        "carry_gated_complete",
    }
