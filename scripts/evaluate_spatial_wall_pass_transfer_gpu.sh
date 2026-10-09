#!/usr/bin/env bash
# Frozen inference only. Does not invoke an optimizer or save checkpoints.
set -euo pipefail
WORKTREE=/home/ext_csv/HackRL/.worktrees/spatial_transfer
RUN_ROOT=/raid/ext_csv/HackRL/runs/spatial_wall_pass_frozen_transfer_v1
PYTHON=/home/ext_csv/miniconda3/envs/offrl/bin/python
exec 9>/tmp/hackrl-spatial-frozen-transfer-v1.lock
flock -n 9 || exit 75
cd "$WORKTREE"
mkdir -p "$RUN_ROOT"
export HACKRL_DEVICE=cuda JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$WORKTREE/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PYTHON" scripts/stop_identity_checked_gpu_keepalive.py >>"$RUN_ROOT/operations.log" 2>&1
"$PYTHON" scripts/evaluate_spatial_wall_pass_transfer.py >>"$RUN_ROOT/run.log" 2>&1 &
EVALUATOR_PID=$!
"$PYTHON" scripts/guard_identity_checked_gpu_keepalive.py --runner-pid "$EVALUATOR_PID" --log "$RUN_ROOT/keepalive_guard.jsonl" &
GUARD_PID=$!
STATUS=0
wait "$EVALUATOR_PID" || STATUS=$?
wait "$GUARD_PID" || STATUS=$?
exit "$STATUS"
