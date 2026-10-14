import hashlib
import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    REPOSITORY
    / "docs/manifests/pack_inventory_overwrite_behavior_anchor_ppo_v1.json"
)
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_design_has_no_execution_authority_and_closes_manual_gate_fallback():
    assert MANIFEST["status"] == "design_only_no_execution"
    assert MANIFEST["next_authorized_unit"].startswith("Implement")
    assert "Do not launch training" in MANIFEST["next_authorized_unit"]
    assert MANIFEST["budget"]["manual_gate_fallback"] is False
    closeout = json.loads(
        (
            REPOSITORY / MANIFEST["prerequisites"]["manual_gate_closeout"]
        ).read_text(encoding="utf-8")
    )
    assert closeout["claim_revision"]["manual_gate_phase"].startswith("Closed")


def test_design_fixes_one_probability_one_beta_and_two_conditions():
    assert MANIFEST["seeds"] == [30, 31, 32, 33, 34]
    assert list(MANIFEST["conditions"]) == ["p05_ppo", "p05_anchor_kl_beta01"]
    assert {row["overwrite_probability"] for row in MANIFEST["conditions"].values()} == {
        0.05
    }
    assert [row["behavior_anchor_coefficient"] for row in MANIFEST["conditions"].values()] == [
        0.0,
        0.1,
    ]
    assert MANIFEST["budget"]["coefficient_sweep"] is False
    assert MANIFEST["budget"]["probability_sweep"] is False


def test_budget_arithmetic_and_optimizer_schedule_are_fixed():
    optimization = MANIFEST["optimization"]
    budget = MANIFEST["budget"]
    assert optimization["num_envs"] * optimization["num_steps"] == 32768
    assert 32768 * optimization["updates"] == optimization["collector_slots_per_job"]
    assert budget["jobs"] == len(MANIFEST["seeds"]) * len(MANIFEST["conditions"]) == 10
    assert budget["collector_slots"] == budget["jobs"] * optimization[
        "collector_slots_per_job"
    ] == 10485760
    assert optimization["scheduled_optimizer_applications_per_job"] == 1024
    assert budget["automatic_extension"] is False


def test_anchor_is_train_only_and_both_evaluation_panels_are_declared_development_data():
    anchor = MANIFEST["behavior_anchor"]
    assert anchor["split"] == "train"
    assert anchor["kernel"] == "fixed"
    assert anchor["starts_per_seed"] == 32
    assert anchor["loss_direction"] == "D_KL(stop_gradient(source policy) || current policy)"
    assert anchor["coefficient"] == 0.1
    assert anchor["value_anchor"] is False
    assert set(MANIFEST["evaluation"]["development_panels"]) == {"validation", "test"}
    assert "not a fresh holdout" in MANIFEST["evaluation"]["development_panels"]["test"][
        "role"
    ]


def test_prerequisite_and_source_checkpoint_hashes_are_pinned():
    prerequisites = MANIFEST["prerequisites"]
    for name in ("manual_gate_closeout", "learning_rate_result", "learning_rate_audit"):
        path = REPOSITORY / prerequisites[name]
        assert _sha256(path) == prerequisites[f"{name}_sha256"]
    root = Path(MANIFEST["source_root"])
    checkpoint = MANIFEST["source_checkpoint"]
    for seed in MANIFEST["seeds"]:
        for filename, expected in MANIFEST["source_checkpoint_sha256"][str(seed)].items():
            assert _sha256(root / f"seed{seed}" / "checkpoints" / checkpoint / filename) == expected


def test_storage_forecast_keeps_peak_writes_plus_reserve_on_raid():
    storage = MANIFEST["checkpoint_storage"]
    assert storage["chosen_filesystem"] == "/raid/ext_csv"
    assert storage["minimum_required_free_bytes"] == storage[
        "projected_peak_writes_bytes"
    ] + storage["safety_reserve_bytes"]
    assert storage["raid_free_bytes_at_design"] > storage["minimum_required_free_bytes"]
    assert storage["checkpoints_written_in_this_design_phase"] == 0
