#!/usr/bin/env python3
"""Stop only the exact GPU-0 keepalive process authorized by the run contract."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

DEVICE_INDEX = 0
MARKER = "--beyondg-gpu-keepalive"
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.pidfd_open.argtypes = (ctypes.c_int, ctypes.c_uint)
LIBC.pidfd_open.restype = ctypes.c_int


def pidfd_open(pid):
    descriptor = LIBC.pidfd_open(int(pid), 0)
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), pid)
    return descriptor


def gpu_pids():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(DEVICE_INDEX),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def validate_process(pid):
    pidfd = pidfd_open(pid)
    try:
        proc = Path("/proc") / str(pid)
        stat = proc.stat()
        argv = (proc / "cmdline").read_bytes().split(b"\0")
        argv = [part.decode("utf-8", errors="strict") for part in argv if part]
        executable = Path(os.readlink(proc / "exe")).resolve()
        expected_executable = Path(sys.executable).resolve()
        if stat.st_uid != os.getuid():
            raise RuntimeError(f"GPU process {pid} is owned by another uid")
        if executable != expected_executable:
            raise RuntimeError(
                f"GPU process {pid} executable {executable} != {expected_executable}"
            )
        if MARKER not in argv:
            raise RuntimeError(f"GPU process {pid} lacks the standalone keepalive marker")
        return {
            "pid": pid,
            "pidfd": pidfd,
            "executable": str(executable),
            "argv": argv,
            "proc_start_ticks": int((proc / "stat").read_text().split()[21]),
        }
    except Exception:
        os.close(pidfd)
        raise


def main():
    observed = gpu_pids()
    validated = []
    try:
        for pid in observed:
            validated.append(validate_process(pid))
    except Exception:
        for item in validated:
            os.close(item["pidfd"])
        raise

    terminated = []
    try:
        for item in validated:
            signal.pidfd_send_signal(item["pidfd"], signal.SIGTERM)
        for item in validated:
            poller = select.poll()
            poller.register(item["pidfd"], select.POLLIN)
            if not poller.poll(10_000):
                raise RuntimeError(f"keepalive PID {item['pid']} did not exit after SIGTERM")
            terminated.append(
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"pidfd", "argv"}
                }
            )
    finally:
        for item in validated:
            os.close(item["pidfd"])

    remaining = []
    for _ in range(100):
        remaining = gpu_pids()
        if not remaining:
            break
        time.sleep(0.1)
    if remaining:
        raise RuntimeError(f"GPU 0 still has compute processes after entry check: {remaining}")
    print(
        json.dumps(
            {
                "device_index": DEVICE_INDEX,
                "observed_pids": observed,
                "terminated": terminated,
                "remaining_pids": remaining,
                "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
