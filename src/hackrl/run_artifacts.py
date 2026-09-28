"""Persist PPO diagnosis artifacts: config, SHA, updates, and params."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from flax.serialization import to_bytes

from hackrl.tasks import FixtureVersion, StartMode, parse_task


def repo_git_sha(repo_root: Path | None = None) -> str:
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def config_payload(config) -> dict:
    payload = asdict(config)
    payload["task"] = parse_task(config.task).value
    payload["start_mode"] = StartMode(config.start_mode).value
    payload["fixture"] = FixtureVersion(config.fixture).value
    payload["variant"] = "mutant" if config.mutant else "fixed"
    return payload


def first_success_update(successful_episodes) -> int | None:
    values = np.asarray(successful_episodes)
    hits = np.flatnonzero(values > 0)
    if hits.size == 0:
        return None
    return int(hits[0])


def write_run_artifacts(
    log_dir,
    *,
    config,
    git_sha: str,
    train_state,
    update_metrics,
    summary: dict,
):
    destination = Path(log_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "config.json").write_text(
        json.dumps(config_payload(config), indent=2, sort_keys=True) + "\n"
    )
    (destination / "git_sha.txt").write_text(f"{git_sha}\n")
    (destination / "params.msgpack").write_bytes(to_bytes(train_state.params))

    metrics = {
        key: np.asarray(value) for key, value in update_metrics.items()
    }
    fieldnames = ["update", *sorted(metrics)]
    with (destination / "updates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        n_updates = int(next(iter(metrics.values())).shape[0])
        for index in range(n_updates):
            row = {"update": index}
            for key, value in metrics.items():
                item = value[index]
                row[key] = (
                    item.item() if getattr(item, "shape", ()) == () else item.tolist()
                )
            writer.writerow(row)

    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return destination
