#!/usr/bin/env bash
set -euo pipefail
WORKTREE=/home/ext_csv/HackRL/.worktrees/spatial_recovery_probe
ROOT=/raid/ext_csv/HackRL/runs/spatial_recovery_probe_v1
PYTHON=/home/ext_csv/miniconda3/envs/offrl/bin/python
exec 9>/tmp/hackrl-spatial-recovery-probe-v1.lock
flock -n 9 || exit 75
cd "$WORKTREE"
if python3 scripts/check_spatial_recovery_probe_complete.py; then exit 0; else STATUS=$?; [[ "$STATUS" == 3 ]] || exit "$STATUS"; fi
mkdir -p "$ROOT"
export HACKRL_DEVICE=cuda JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$WORKTREE/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PYTHON" scripts/stop_identity_checked_gpu_keepalive.py >>"$ROOT/operations.log" 2>&1
"$PYTHON" scripts/run_spatial_recovery_probe.py >>"$ROOT/launcher.log" 2>&1 &
RUNNER_PID=$!
"$PYTHON" scripts/guard_identity_checked_gpu_keepalive.py --runner-pid "$RUNNER_PID" --log "$ROOT/guard.jsonl" &
GUARD_PID=$!
STATUS=0
wait "$RUNNER_PID" || STATUS=$?
wait "$GUARD_PID" || STATUS=$?
exit "$STATUS"
