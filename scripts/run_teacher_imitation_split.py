#!/usr/bin/env python3
"""Add the two missing Dual cells on TICK-CLAIM and PACK-RESTORE.

teacher_only learns the teacher and does not imitate it. frozen_imitation
keeps the initialized teacher parameters and BatchRenorm statistics fixed
while the policy imitates that teacher. Both start at pretraining. The
finished GC-PPO and full Dual runs stay in runs/dual_leo_compare_v1.
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
    run_pretrain,
)

ARMS = (
    ("teacher_only", True, False),
    ("frozen_imitation", False, True),
)


def build_jobs():
    jobs = []
    for env in ("tick", "pack"):
        for method, learn_teacher, imitate_teacher in ARMS:
            for seed in SEEDS:
                pretrain = f"{env}-{method}-pretrain-s{seed}"
                jobs.append(
                    {
                        "id": pretrain,
                        "kind": "pretrain",
                        "env": env,
                        "method": method,
                        "seed": seed,
                        "learn_teacher": learn_teacher,
                        "imitate_teacher": imitate_teacher,
                    }
                )
                for variant in ("fixed", "mutant"):
                    jobs.append(
                        {
                            "id": f"{env}-{method}-s{seed}-{variant}",
                            "kind": "adapt",
                            "env": env,
                            "method": method,
                            "seed": seed,
                            "variant": variant,
                            "learn_teacher": learn_teacher,
                            "imitate_teacher": imitate_teacher,
                            "depends_on": [pretrain],
                        }
                    )
    return jobs


def _ready(log_dir, job, jobs):
    lookup = {item["id"]: item for item in jobs}
    return all(_job_complete(log_dir, lookup[name]) for name in job.get("depends_on", []))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--worker", required=True)
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    if len(jobs) != 60:
        raise RuntimeError(f"expected 60 jobs, found {len(jobs)}")
    print(f"[worker] {args.worker} jobs={len(jobs)}", flush=True)
    while True:
        _reclaim(log_dir)
        chosen = None
        for job in jobs:
            if _job_complete(log_dir, job) or not _ready(log_dir, job, jobs):
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
            if chosen["kind"] == "pretrain":
                run_pretrain(log_dir, chosen)
            else:
                run_adapt(log_dir, chosen)
        finally:
            _release(log_dir, chosen["id"])


if __name__ == "__main__":
    main()
