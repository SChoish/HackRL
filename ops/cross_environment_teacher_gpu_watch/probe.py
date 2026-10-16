#!/usr/bin/env python3
"""Read-only completion probe for the GPU common-trace evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


REPO = Path("/home/ext_csv/HackRL")
OUTPUT = Path(
    "/raid/ext_csv/HackRL/evaluations/"
    "cross_environment_teacher_matrix_v1_gpu"
)
EXPECTED_TASKS = 1440
EXPECTED_EPISODES = 88_200
STEP_CAP = 11_289_600


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _process_alive() -> bool:
    uid = os.getuid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            argv = [
                item.decode()
                for item in (entry / "cmdline").read_bytes().split(b"\0")
                if item
            ]
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            continue
        if (
            "--run-shard" in argv
            and any(
                Path(item).name
                == "run_cross_environment_teacher_evaluation_gpu.py"
                for item in argv
            )
        ):
            return True
    return False


def status() -> dict:
    manifest_path = OUTPUT / "evaluation_manifest.json"
    if not manifest_path.is_file():
        return {
            "terminal": False,
            "outcome": "pending",
            "reason": "evaluation_manifest_not_prepared",
            "completed_tasks": 0,
        }
    manifest = _json(manifest_path)
    if (
        manifest.get("task_count") != EXPECTED_TASKS
        or manifest.get("episode_cap") != EXPECTED_EPISODES
        or manifest.get("step_cap") != STEP_CAP
        or manifest.get("partial_cpu_receipts_reused") != 0
    ):
        raise ValueError("evaluation manifest contract drift")
    current_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=REPO, text=True
    ).strip()
    if manifest.get("evaluation_execution_commit") != current_commit:
        raise ValueError("checkout differs from prepared evaluation commit")
    execution_path = OUTPUT / "execution_receipt.json"
    if execution_path.is_file():
        receipt = _json(execution_path)
        task_receipts = list((OUTPUT / "task_receipts").glob("*.json"))
        if not (
            receipt.get("passed") is True
            and receipt.get("completed_tasks") == EXPECTED_TASKS
            and receipt.get("episodes") == EXPECTED_EPISODES
            and 0 < int(receipt.get("steps", 0)) <= STEP_CAP
            and len(task_receipts) == EXPECTED_TASKS
            and receipt.get("evaluation_execution_commit") == current_commit
        ):
            raise ValueError("final execution receipt drift")
        return {
            "terminal": True,
            "outcome": "complete",
            "science_complete": True,
            "completed_tasks": EXPECTED_TASKS,
            "episodes": receipt["episodes"],
            "steps": receipt["steps"],
        }
    rows = []
    for path in sorted((OUTPUT / "shard_ledgers").glob("*.jsonl")):
        rows.extend(_jsonl(path))
    latest = {}
    for row in rows:
        task = row.get("task")
        if isinstance(task, str):
            latest[task] = row
    receipts = {path.stem for path in (OUTPUT / "task_receipts").glob("*.json")}
    failed = sorted(
        task for task, row in latest.items()
        if row.get("status") == "failed" and task not in receipts
    )
    dangling = sorted(
        task for task, row in latest.items()
        if row.get("status") == "started" and task not in receipts
    )
    alive = _process_alive()
    if failed or (dangling and not alive):
        return {
            "terminal": True,
            "outcome": "failed",
            "science_complete": False,
            "reason": (
                "evaluation_failed_no_auto_rerun"
                if failed else "evaluation_interrupted_no_auto_rerun"
            ),
            "failed_tasks": failed,
            "dangling_started_tasks": dangling,
            "completed_tasks": len(receipts),
        }
    return {
        "terminal": False,
        "outcome": "pending",
        "evaluator_alive": alive,
        "completed_tasks": len(receipts),
        "started_tasks": len(latest),
    }


def main() -> int:
    try:
        result = status()
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({
            "terminal": False,
            "error": f"{type(error).__name__}:{error}",
        }, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["terminal"] else 10


if __name__ == "__main__":
    raise SystemExit(main())
