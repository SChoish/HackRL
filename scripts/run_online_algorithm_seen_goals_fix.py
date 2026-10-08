#!/usr/bin/env python3
"""Run the seen-goal-fixed copy of the frozen online development matrix."""

import hashlib
from pathlib import Path

import run_online_algorithm_development as base

REPOSITORY = Path(__file__).resolve().parents[1]
base.MANIFEST_PATH = (
    REPOSITORY
    / "docs/manifests/online_algorithm_expansion_v1_development_seen_goals_fix_v1.json"
)
base.SOURCE_PATHS = (
    "docs/manifests/online_algorithm_expansion_v1.json",
    "docs/manifests/online_algorithm_expansion_v1_implementation.json",
    "docs/manifests/online_algorithm_seen_goals_fix_v1_smoke.json",
    "docs/manifests/online_algorithm_expansion_v1_development_seen_goals_fix_v1.json",
    "scripts/run_online_algorithm_development.py",
    "scripts/run_online_algorithm_seen_goals_fix.py",
    "scripts/run_online_algorithm_seen_goals_fix_gpu_queue.sh",
    "scripts/stop_identity_checked_gpu_keepalive.py",
    "src/hackrl/online_value.py",
    "src/hackrl/discrete_sac.py",
    "src/hackrl/online_algorithm_env.py",
    "src/hackrl/tick_claim.py",
    "src/hackrl/tick_claim_gc.py",
    "src/hackrl/pack_restore.py",
    "src/hackrl/pack_restore_gc.py",
)

_base_require_smoke_passed = base._require_smoke_passed


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_bound_seen_goal_smoke(manifest):
    _base_require_smoke_passed(manifest)
    smoke = base._read_json(REPOSITORY / manifest["smoke_manifest"])
    identity = smoke.get("execution_identity", {})
    expected = identity.get("runtime_source_sha256", {})
    if not expected:
        raise RuntimeError("post-fix smoke has no bound runtime source hashes")
    for relative, digest in expected.items():
        if _sha256(REPOSITORY / relative) != digest:
            raise RuntimeError(f"post-fix smoke source mismatch: {relative}")
    artifact = smoke.get("result_artifact", {})
    result_path = Path(artifact.get("aggregate_path", ""))
    if not result_path.is_file():
        raise RuntimeError("post-fix smoke aggregate is missing")
    if _sha256(result_path) != artifact.get("aggregate_sha256"):
        raise RuntimeError("post-fix smoke aggregate hash mismatch")


base._require_smoke_passed = _require_bound_seen_goal_smoke

if __name__ == "__main__":
    base.main()
