import importlib.util
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = _load(
    "run_pack_restore_delivery_only_teacher",
    "scripts/run_pack_restore_delivery_only_teacher.py",
)
SUMMARY = _load(
    "summarize_pack_restore_delivery_only_teacher",
    "scripts/summarize_pack_restore_delivery_only_teacher.py",
)

def test_job_graph_is_exactly_the_authorized_twenty_adaptations():
    jobs = RUNNER.build_jobs()
    assert len(jobs) == 20
    assert {job["seed"] for job in jobs} == set(range(30, 40))
    assert {job["variant"] for job in jobs} == {"fixed", "mutant"}
    assert all(job["kind"] == "adapt" for job in jobs)
    assert all(job["size"] == "S" for job in jobs)
    assert all(job["condition"] == "D-delivery" for job in jobs)
    assert all(job["learn_teacher"] is True for job in jobs)
    assert all(job["imitate_teacher"] is True for job in jobs)
    assert all(job["teacher_goal_indices"] == [RUNNER.DELIVER_3_GOAL_INDEX] for job in jobs)
    assert all(job["origin_arm"] == RUNNER.FULL_DUAL_ARM for job in jobs)
    assert (
        len(jobs) * RUNNER.ADAPT_UPDATES * 512 * 64
        == 2_684_354_560
    )


def test_manifest_matches_job_budget_and_storage_forecast():
    manifest = json.loads(
        (REPOSITORY / RUNNER.MANIFEST_PATH).read_text(encoding="utf-8")
    )
    assert manifest["budget"]["new_total_jobs"] == len(RUNNER.build_jobs())
    assert (
        manifest["budget"]["new_total_transitions"]
        == len(RUNNER.build_jobs()) * RUNNER.ADAPT_UPDATES * 512 * 64
    )
    assert (
        manifest["storage"]["projected_remaining_write_bytes"]
        == RUNNER.PROJECTED_REMAINING_WRITE_BYTES
    )
    assert manifest["storage"]["safety_reserve_bytes"] == 8 * 1024**3
    assert (
        manifest["source_scale_run"]["execution_code_sha"]
        == RUNNER.EXPECTED_SOURCE_EXECUTION_SHA
    )
    assert (
        manifest["source_scale_run"]["manifest_sha256"]
        == RUNNER.EXPECTED_SOURCE_MANIFEST_SHA256
    )
    assert {
        int(seed): digest
        for seed, digest in manifest["source_scale_run"][
            "source_checkpoint_fingerprints"
        ].items()
    } == RUNNER.EXPECTED_SOURCE_FINGERPRINTS
    assert manifest["optimization"]["adaptation_updates"] == RUNNER.ADAPT_UPDATES
    assert tuple(manifest["evaluation"]["conditions"]) == SUMMARY.CONDITIONS
    assert manifest["excluded"] == [
        "new pretraining",
        "new random seeds",
        "M or L policy widths",
        "Transformer policies",
        "8192-update extension",
        "learning-rate sweep",
        "mine curriculum changes",
        "mutant mine training",
    ]


def test_source_checkpoint_paths_and_fingerprints_are_seed_specific():
    paths = [RUNNER.source_checkpoint(seed) for seed in RUNNER.SEEDS]
    assert len(set(paths)) == 10
    assert all(path.name == "update_512" for path in paths)
    assert set(RUNNER.EXPECTED_SOURCE_FINGERPRINTS) == set(range(30, 40))


def test_teacher_hash_is_numeric_and_container_stable():
    left = {"x": np.asarray([1.0, 2.0], dtype=np.float32)}
    right = {"x": jnp.asarray([1.0, 2.0], dtype=np.float32)}
    assert RUNNER._canonical_tree_sha256(left) == RUNNER._canonical_tree_sha256(right)
    right["x"] = right["x"].at[1].set(3.0)
    assert RUNNER._canonical_tree_sha256(left) != RUNNER._canonical_tree_sha256(right)


def test_claim_reclamation_uses_process_start_identity(tmp_path):
    claims = tmp_path / "claims"
    claims.mkdir()
    stale = claims / "stale"
    stale.write_text(
        json.dumps(
            {
                "pid": 1,
                "process_start_ticks": -1,
                "token": "stale",
                "host": "test",
            }
        ),
        encoding="utf-8",
    )
    RUNNER._reclaim_claims(tmp_path)
    assert not stale.exists()


def test_bootstrap_keeps_learner_seed_as_the_input_unit():
    result = SUMMARY._bootstrap([0.0, 0.25, 0.5], __import__("numpy").random.default_rng(0))
    assert result["n_seeds"] == 3
    assert result["mean"] == 0.25
    assert len(result["bootstrap_95_percentile_interval"]) == 2
