#!/usr/bin/env python3
"""Turn off policy imitation only during adaptation.

Starts from the finished full-Dual pretraining checkpoints. Teacher updates
and the teacher shuffle still run. The existing BC-on adaptation branches are
left in runs/dual_leo_compare_v1.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from run_dual_leo_compare import (
    SEEDS,
    _claim,
    _job_complete,
    _reclaim,
    _release,
    run_adapt,
)

PRETRAIN_ROOT = Path("/home/ext_csv/HackRL/runs/dual_leo_compare_v1")
ORIGIN_ARM = {"learn_teacher": True, "imitate_teacher": True}


def build_jobs(pretrain_root=PRETRAIN_ROOT):
    jobs = []
    root = Path(pretrain_root)
    for env in ("tick", "pack"):
        for seed in SEEDS:
            source = root / env / "dual" / "pretrain" / f"seed{seed}" / "checkpoints" / "update_512"
            for variant in ("fixed", "mutant"):
                jobs.append(
                    {
                        "id": f"{env}-adapt-bc-off-s{seed}-{variant}",
                        "kind": "adapt",
                        "env": env,
                        "method": "adapt_bc_off",
                        "seed": seed,
                        "variant": variant,
                        "learn_teacher": True,
                        "imitate_teacher": False,
                        "origin_arm": dict(ORIGIN_ARM),
                        "pretrain_checkpoint": str(source),
                    }
                )
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--pretrain-root", default=str(PRETRAIN_ROOT))
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.pretrain_root)
    if len(jobs) != 20:
        raise RuntimeError(f"expected 20 adaptation jobs, found {len(jobs)}")
    missing = [job["pretrain_checkpoint"] for job in jobs if not Path(job["pretrain_checkpoint"], "state.msgpack").is_file()]
    if missing:
        raise RuntimeError(f"full Dual pretrain checkpoint missing: {missing[0]}")
    print(f"[worker] {args.worker} jobs={len(jobs)}", flush=True)
    while True:
        _reclaim(log_dir)
        chosen = None
        for job in jobs:
            if _job_complete(log_dir, job):
                continue
            if _claim(log_dir, job["id"]):
                chosen = job
                break
        if chosen is None:
            if all(_job_complete(log_dir, job) for job in jobs):
                print(f"[worker] {args.worker} sweep complete", flush=True)
                return
            time.sleep(15)
            continue
        try:
            print(f"[job] {chosen['id']}", flush=True)
            run_adapt(log_dir, chosen)
        finally:
            _release(log_dir, chosen["id"])


if __name__ == "__main__":
    main()
