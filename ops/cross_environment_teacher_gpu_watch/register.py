#!/usr/bin/env python3
"""Register the GPU evaluation wrapper with the external server watcher."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO = Path("/home/ext_csv/HackRL")
OPS = REPO / "ops/cross_environment_teacher_gpu_watch"
OUTPUT = Path(
    "/raid/ext_csv/HackRL/evaluations/"
    "cross_environment_teacher_matrix_v1_gpu"
)
CONFIG = Path("/home/ext_csv/server_management/config/ext_csv.json")

sys.path.insert(0, "/home/ext_csv/server_management/scripts/hosts/beyondg")
from queue_worker import atomic_json  # noqa: E402
from watch_beyondg_lease import load_config  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    manifest = _json(OUTPUT / "evaluation_manifest.json")
    commit = manifest["evaluation_execution_commit"]
    if commit != subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=REPO, text=True
    ).strip():
        raise RuntimeError("prepared manifest and checkout commit differ")
    files = [OPS / name for name in (
        "probe.py", "run_gpu.sh", "run_cpu_wait.sh", "register.py"
    )]
    if not all(path.is_file() for path in files):
        raise RuntimeError("watcher source is incomplete")
    with CONFIG.with_suffix(".json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        load_config(path=CONFIG)
        before = _json(CONFIG)
        after = json.loads(json.dumps(before))
        worker = after["worker"]
        runner = str(REPO / "scripts/run_cross_environment_teacher_evaluation_gpu.py")
        worker["process_patterns"] = [
            runner,
            "scripts/run_cross_environment_teacher_evaluation_gpu.py",
        ]
        probe = [str(Path(sys.executable)), str(OPS / "probe.py")]
        for mode, script in {
            "cpu": OPS / "run_cpu_wait.sh",
            "gpu": OPS / "run_gpu.sh",
        }.items():
            job = worker[mode]
            job.update(
                command=["bash", str(script)],
                cwd=str(REPO),
                queue_patterns=[str(script)],
                log=str(OUTPUT / f"server_watcher_{mode}.log"),
                completion_probe=probe,
            )
            job.pop("env", None)
        usage = shutil.disk_usage("/raid/ext_csv")
        receipt = {
            "schema_version": (
                "hackrl_cross_environment_teacher_gpu_watcher_registration_v1"
            ),
            "recorded_unix": time.time(),
            "evaluation_execution_commit": commit,
            "watcher_files_sha256": {
                str(path.relative_to(REPO)): _sha256(path) for path in files
            },
            "config_before_sha256": _sha256(CONFIG),
            "config_before": before,
            "config_after": after,
            "capacity": {
                "raid_free_bytes": usage.free,
                "target": "/raid/ext_csv",
            },
            "gpu_shards": 4,
            "cpu_fallback": "wait-only; no scientific evaluation on CPU",
            "interruption_contract": (
                "A dangling started task without a passing receipt is terminal "
                "and is not automatically re-evaluated."
            ),
        }
        if args.apply:
            receipt_path = OUTPUT / "server_watcher_registration.json"
            if receipt_path.exists():
                raise RuntimeError(f"registration already exists: {receipt_path}")
            atomic_json(receipt_path, receipt)
            atomic_json(CONFIG, after)
            print("REGISTERED")
        else:
            print("READY; no config changed")
        print(json.dumps({
            "evaluation_execution_commit": commit,
            "raid_free_bytes": usage.free,
            "gpu_shards": 4,
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
