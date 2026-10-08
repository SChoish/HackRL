#!/usr/bin/env python3
"""Run the seen-goal-fixed copy of the frozen online development matrix."""

from pathlib import Path

import run_online_algorithm_development as base

REPOSITORY = Path(__file__).resolve().parents[1]
base.MANIFEST_PATH = REPOSITORY / "docs/manifests/online_algorithm_expansion_v1_development_seen_goals_fix_v1.json"
base.SOURCE_PATHS = (
    "docs/manifests/online_algorithm_expansion_v1.json",
    "docs/manifests/online_algorithm_expansion_v1_implementation.json",
    "docs/manifests/online_algorithm_seen_goals_fix_v1_smoke.json",
    "docs/manifests/online_algorithm_expansion_v1_development_seen_goals_fix_v1.json",
    "scripts/run_online_algorithm_development.py",
    "scripts/run_online_algorithm_seen_goals_fix.py",
    "scripts/run_online_algorithm_seen_goals_fix_gpu_queue.sh",
    "src/hackrl/online_value.py",
    "src/hackrl/discrete_sac.py",
    "src/hackrl/online_algorithm_env.py",
    "src/hackrl/tick_claim.py",
    "src/hackrl/tick_claim_gc.py",
    "src/hackrl/pack_restore.py",
    "src/hackrl/pack_restore_gc.py",
)

if __name__ == "__main__":
    base.main()
