import json
import sys
from pathlib import Path

import numpy as np

from hackrl.pack_restore_gc import (
    PackRestoreGCConfig,
    config_from_pack_restore_gc_payload,
    pack_restore_gc_config_payload,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_dual_leo_compare import (
    SCIENCE_UPDATES,
    _checkpoint_fingerprint,
    _job_config,
    _pretrain_dir,
    _prune_rolling_checkpoints,
)
from run_pack_restore_scale_bc import (
    EXECUTION_SOURCES,
    EXPECTED_TEACHER_PARAMETERS,
    FULL_DUAL_ARM,
    PROJECTED_FINAL_CHECKPOINT_BYTES,
    SEEDS,
    SIZE_SPECS,
    build_jobs,
)
from summarize_pack_restore_scale_bc import (
    _shared_dual_origin_checks,
    compute_outputs,
)


def test_pack_restore_policy_and_teacher_widths_are_independent():
    config = PackRestoreGCConfig(policy_hidden_size=1024, teacher_hidden_size=512)
    assert config.policy_hidden_size == 1024
    assert config.teacher_hidden_size == 512
    payload = pack_restore_gc_config_payload(config)
    assert payload["policy_hidden_size"] == 1024
    assert payload["teacher_hidden_size"] == 512
    assert "hidden_size" not in payload


def test_legacy_pack_restore_config_payload_maps_both_widths_to_512():
    payload = pack_restore_gc_config_payload(PackRestoreGCConfig())
    payload["hidden_size"] = payload.pop("policy_hidden_size")
    payload.pop("teacher_hidden_size")
    restored = config_from_pack_restore_gc_payload(payload)
    assert restored.policy_hidden_size == 512
    assert restored.teacher_hidden_size == 512


def test_scale_job_matrix_and_shared_dual_pretraining_source(tmp_path):
    jobs = build_jobs()
    assert len(jobs) == 240
    assert sum(job["kind"] == "pretrain" for job in jobs) == 60
    assert sum(job["kind"] == "adapt" for job in jobs) == 180
    assert {job["seed"] for job in jobs} == set(SEEDS)
    assert {job["size"] for job in jobs} == {"S", "M", "L"}
    assert sum(job.get("condition") == "G" and job["kind"] == "adapt" for job in jobs) == 60
    assert sum(job.get("condition") == "D-on" for job in jobs) == 60
    assert sum(job.get("condition") == "D-off" for job in jobs) == 60
    assert all(job["rolling_checkpoints"] for job in jobs)
    assert all(job["record_checkpoint_fingerprint"] for job in jobs)

    adaptations = [job for job in jobs if job["kind"] == "adapt"]
    indexed = {
        (job["size"], job["seed"], job["condition"], job["variant"]): job
        for job in adaptations
    }
    for size in ("S", "M", "L"):
        for seed in SEEDS:
            for variant in ("fixed", "mutant"):
                on = indexed[(size, seed, "D-on", variant)]
                off = indexed[(size, seed, "D-off", variant)]
                assert on["depends_on"] == off["depends_on"]
                assert _pretrain_dir(tmp_path, on) == _pretrain_dir(tmp_path, off)
                assert off["learn_teacher"] is True
                assert off["imitate_teacher"] is False
                assert off["origin_arm"] == FULL_DUAL_ARM


def test_scale_manifest_matches_generated_budget():
    manifest = json.loads(
        (ROOT / "docs/manifests/pack_restore_scale_bc_v1.json").read_text(
            encoding="utf-8"
        )
    )
    jobs = build_jobs()
    pretraining = sum(job["kind"] == "pretrain" for job in jobs)
    adaptation = len(jobs) - pretraining
    expected_transitions = (
        pretraining * 512 * 512 * 64 + adaptation * 4096 * 512 * 64
    )
    assert manifest["budget"]["pretraining_jobs"] == pretraining == 60
    assert manifest["budget"]["total_jobs"] == len(jobs) == 240
    assert manifest["budget"]["total_transitions"] == expected_transitions
    assert manifest["seeds"] == list(SEEDS)
    assert manifest["schedule"]["science_evaluations"] == list(SCIENCE_UPDATES)
    assert manifest["teacher"]["parameters"] == EXPECTED_TEACHER_PARAMETERS
    assert (
        manifest["checkpoint_storage"]["projected_all_final_states_bytes"]
        == PROJECTED_FINAL_CHECKPOINT_BYTES
    )
    assert manifest["implementation"]["execution_sources"] == [
        str(path) for path in EXECUTION_SOURCES
    ]
    assert {
        size: {
            "policy_hidden_size": spec["policy_hidden_size"],
            "ppo_actor_critic_parameters": spec["expected_ppo_parameters"],
        }
        for size, spec in ((item["size"], item) for item in SIZE_SPECS)
    } == manifest["model_sizes"]


def test_scale_configs_keep_the_teacher_at_512():
    jobs = build_jobs()
    for spec in SIZE_SPECS:
        job = next(
            item
            for item in jobs
            if item["kind"] == "pretrain"
            and item["method"] == "dual"
            and item["size"] == spec["size"]
        )
        config = _job_config(
            job,
            goal_mode="workshop12",
            variant="fixed",
            updates=512,
        )
        assert config.policy_hidden_size == spec["policy_hidden_size"]
        assert config.teacher_hidden_size == 512
        assert spec["expected_ppo_parameters"] > EXPECTED_TEACHER_PARAMETERS


def _evaluation(exploit, success=1.0, length=10.0, discounted_return=0.95):
    block = {
        "violation_delivery_rate": exploit,
        "success_rate": success,
        "mean_length": length,
        "mean_discounted_return": discounted_return,
    }
    return {
        dynamics: {
            policy: {family: dict(block) for family in ("natural_reset", "common_setup")}
            for policy in ("mode", "sample")
        }
        for dynamics in ("fixed", "mutant")
    }


def test_aggregator_uses_seed_paired_bc_difference():
    curves = {}
    values = {
        ("D-on", "fixed"): 0.1,
        ("D-on", "mutant"): 0.5,
        ("D-off", "fixed"): 0.2,
        ("D-off", "mutant"): 0.3,
    }
    for (condition, variant), exploit in values.items():
        curves[("S", condition, 30, variant, 0)] = _evaluation(exploit)
    output = compute_outputs(curves)
    primary_delta = output["Delta_BC"]["primary_seed_points"]
    point = next(
        row
        for row in primary_delta
        if row["size"] == "S"
        and row["seed"] == 30
        and row["adaptation_update"] == 0
    )
    np.testing.assert_allclose(point["value"], (0.5 - 0.1) - (0.3 - 0.2))


def test_checkpoint_fingerprint_covers_load_bearing_files(tmp_path):
    checkpoint = tmp_path / "update_512"
    checkpoint.mkdir()
    for name, content in {
        "state.msgpack": b"state",
        "config.json": b"{}\n",
        "metadata.json": b'{"global_update": 512}\n',
        "arm.json": b'{"imitate_teacher": true, "learn_teacher": true}\n',
    }.items():
        (checkpoint / name).write_bytes(content)
    first = _checkpoint_fingerprint(checkpoint)
    (checkpoint / "state.msgpack").write_bytes(b"different")
    second = _checkpoint_fingerprint(checkpoint)
    assert first["sha256"] != second["sha256"]
    assert set(first["files"]) == {
        "state.msgpack",
        "config.json",
        "metadata.json",
        "arm.json",
    }


def test_rolling_checkpoint_pruning_keeps_only_successor(tmp_path):
    root = tmp_path / "checkpoints"
    for name in ("adapt_0", "adapt_32", "unrelated"):
        (root / name).mkdir(parents=True)
    _prune_rolling_checkpoints(root, "adapt_", 32)
    assert not (root / "adapt_0").exists()
    assert (root / "adapt_32").is_dir()
    assert (root / "unrelated").is_dir()


def test_shared_dual_origin_gate_requires_all_four_matching_branches():
    origins = {}
    for condition in ("D-on", "D-off"):
        for variant in ("fixed", "mutant"):
            origins[("S", condition, 30, variant)] = {
                "source_checkpoint_sha256": "same-checkpoint"
            }
    checks = _shared_dual_origin_checks(origins)
    target = next(row for row in checks if row["size"] == "S" and row["seed"] == 30)
    assert target["passed"] is True
    origins[("S", "D-off", 30, "mutant")][
        "source_checkpoint_sha256"
    ] = "different-checkpoint"
    checks = _shared_dual_origin_checks(origins)
    target = next(row for row in checks if row["size"] == "S" and row["seed"] == 30)
    assert target["passed"] is False
