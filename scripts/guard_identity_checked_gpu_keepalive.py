#!/usr/bin/env python3
"""Protect one owned GPU runner from re-created, identity-checked keepalives."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import signal
import time

import stop_identity_checked_gpu_keepalive as identity


def append(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def runner_identity(pid):
    pidfd = identity.pidfd_open(pid)
    proc = Path("/proc") / str(pid)
    return {
        "pid": pid,
        "pidfd": pidfd,
        "uid": proc.stat().st_uid,
        "executable": str(Path(os.readlink(proc / "exe")).resolve()),
        "start_ticks": int((proc / "stat").read_text().split()[21]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-pid", type=int, required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    runner = runner_identity(args.runner_pid)
    if runner["uid"] != os.getuid():
        raise RuntimeError("runner belongs to a different uid")
    runner_poller = select.poll()
    runner_poller.register(runner["pidfd"], select.POLLIN)
    append(args.log, {
        "event": "guard_started",
        "runner_pid": runner["pid"],
        "runner_start_ticks": runner["start_ticks"],
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    try:
        while not runner_poller.poll(0):
            for pid in identity.gpu_pids():
                if pid == runner["pid"]:
                    continue
                try:
                    item = identity.validate_process(pid)
                except Exception as error:
                    append(args.log, {
                        "event": "unknown_gpu_process_fail_closed",
                        "pid": pid,
                        "error": repr(error),
                        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    })
                    signal.pidfd_send_signal(runner["pidfd"], signal.SIGTERM)
                    raise
                try:
                    signal.pidfd_send_signal(item["pidfd"], signal.SIGTERM)
                    poller = select.poll()
                    poller.register(item["pidfd"], select.POLLIN)
                    if not poller.poll(10_000):
                        signal.pidfd_send_signal(runner["pidfd"], signal.SIGTERM)
                        raise RuntimeError(f"keepalive PID {pid} did not exit")
                    append(args.log, {
                        "event": "validated_keepalive_terminated",
                        "pid": pid,
                        "proc_start_ticks": item["proc_start_ticks"],
                        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    })
                finally:
                    os.close(item["pidfd"])
            time.sleep(args.poll_seconds)
        append(args.log, {
            "event": "runner_exited",
            "runner_pid": runner["pid"],
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    except BaseException as error:
        append(args.log, {
            "event": "guard_failed",
            "runner_pid": runner["pid"],
            "error": repr(error),
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        if not runner_poller.poll(0):
            signal.pidfd_send_signal(runner["pidfd"], signal.SIGTERM)
        raise
    finally:
        os.close(runner["pidfd"])


if __name__ == "__main__":
    main()
