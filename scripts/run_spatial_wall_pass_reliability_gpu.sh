#!/usr/bin/env bash
set -euo pipefail
WORKTREE=/home/ext_csv/HackRL/.worktrees/spatial_reliability
RUN_ROOT=/raid/ext_csv/HackRL/runs/spatial_wall_pass_reliability_mix_v1
PYTHON=/home/ext_csv/miniconda3/envs/offrl/bin/python
MODE="${1:?use --smoke or --queue}"
[[ "$MODE" == "--smoke" || "$MODE" == "--queue" ]] || exit 2
exec 9>/tmp/hackrl-spatial-reliability-mix-v1.lock
flock -n 9 || exit 75
cd "$WORKTREE"
mkdir -p "$RUN_ROOT"
export HACKRL_DEVICE=cuda JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$WORKTREE/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PYTHON" scripts/stop_identity_checked_gpu_keepalive.py >>"$RUN_ROOT/operations.log" 2>&1
"$PYTHON" scripts/run_spatial_wall_pass_reliability.py "$MODE" >>"$RUN_ROOT/launcher.log" 2>&1 &
RUNNER_PID=$!
"$PYTHON" scripts/guard_identity_checked_gpu_keepalive.py --runner-pid "$RUNNER_PID" --log "$RUN_ROOT/keepalive_guard.jsonl" &
GUARD_PID=$!
STATUS=0
wait "$RUNNER_PID" || STATUS=$?
wait "$GUARD_PID" || STATUS=$?
exit "$STATUS"
