import importlib.util
import json
from pathlib import Path
import sys

import jax.numpy as jnp


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))


def _load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load_script(
    "evaluate_pack_inventory_overwrite_gate_heldout_test",
    "scripts/evaluate_pack_inventory_overwrite_gate_heldout.py",
)
MANIFEST = json.loads(
    (REPOSITORY / "docs/manifests/pack_inventory_overwrite_gate_heldout_v1.json").read_text(
        encoding="utf-8"
    )
)


def test_manifest_freezes_the_three_condition_zero_learning_test_panel():
    assert MANIFEST["status"] == "frozen_before_execution"
    assert MANIFEST["seeds"] == [35, 36, 37, 38, 39]
    assert list(MANIFEST["conditions"]) == ["source", "adapted", "carry_gated"]
    assert MANIFEST["heldout_panel"]["split"] == "test"
    assert MANIFEST["evaluation"]["states_per_seed"] == 32
    assert MANIFEST["budget"]["learning_updates"] == 0
    assert MANIFEST["budget"]["teacher_updates"] == 0
    assert MANIFEST["budget"]["bc_updates"] == 0
    assert MANIFEST["budget"]["direct_episodes"] == 960
    assert MANIFEST["budget"]["maximum_environment_steps"] == 122880
    assert MANIFEST["budget"]["automatic_extension"] is False


def test_test_panel_hash_uniqueness_and_nonoverlap_are_reproduced():
    panel = EVALUATOR.state_panel_contract(8)
    assert panel["ordered_full_state_sha256"] == MANIFEST["heldout_panel"][
        "ordered_full_state_sha256"
    ]
    assert panel["state_count"] == 32
    assert panel["unique_state_count"] == 32
    assert panel["exact_overlap_with_train_natural_states"] == 0
    assert panel["exact_overlap_with_validation_natural_states"] == 0
    assert panel["state_indices"] == list(range(32))
    assert panel["all_repeat_indices_zero"] is True


def test_policy_selector_implements_only_the_frozen_three_rules():
    source = jnp.asarray([1, 2, 3, 4])
    adapted = jnp.asarray([5, 6, 7, 8])
    carried = jnp.asarray([0, 1, 2, 0])
    active = jnp.asarray([True, True, False, True])

    source_chosen, source_use = EVALUATOR.choose_actions(
        "source", source, adapted, carried, active
    )
    adapted_chosen, adapted_use = EVALUATOR.choose_actions(
        "adapted", source, adapted, carried, active
    )
    gated_chosen, gated_use = EVALUATOR.choose_actions(
        "carry_gated", source, adapted, carried, active
    )

    assert source_chosen.tolist() == [1, 2, 3, 4]
    assert source_use.tolist() == [False, False, False, False]
    assert adapted_chosen.tolist() == [5, 6, 3, 8]
    assert adapted_use.tolist() == [True, True, False, True]
    assert gated_chosen.tolist() == [1, 6, 3, 4]
    assert gated_use.tolist() == [False, True, False, False]


def test_execution_sources_pin_manifest_evaluator_and_semantic_dependencies():
    assert set(MANIFEST["execution_sources"]) == {
        "docs/manifests/pack_inventory_overwrite_gate_heldout_v1.json",
        "scripts/evaluate_pack_inventory_overwrite_gate_heldout.py",
        "scripts/evaluate_pack_inventory_overwrite_policy_switch.py",
        "src/hackrl/pack_restore.py",
        "src/hackrl/pack_restore_gc.py",
        "src/hackrl/pack_inventory_overwrite_rare.py",
        "src/hackrl/dual_leo.py",
    }


def test_budget_arithmetic_and_primary_comparison_are_fixed():
    assert len(EVALUATOR.SEEDS) * len(EVALUATOR.MODES) * len(EVALUATOR.KERNELS) * 32 == 960
    assert 960 * EVALUATOR.HORIZON == 122880
    assert MANIFEST["analysis"]["independent_unit"] == "learner seed"
    assert MANIFEST["analysis"]["primary_comparison"] == "carry_gated minus source at p=0.05"
    assert MANIFEST["storage"]["chosen_filesystem"] == "/raid/ext_csv"
    assert MANIFEST["storage"]["checkpoints_written"] == 0
