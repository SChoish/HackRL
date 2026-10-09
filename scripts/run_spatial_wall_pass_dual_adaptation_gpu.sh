#!/usr/bin/env bash
# Identity-checked single-GPU entry for the bounded spatial Dual adaptation run.
set -euo pipefail

WORKTREE=/raid/ext_csv/HackRL/worktrees/spatial_wall_pass_v1
RUN_ROOT=/raid/ext_csv/HackRL/runs/spatial_wall_pass_dual_adaptation_v1
PYTHON=/home/ext_csv/miniconda3/envs/offrl/bin/python
LOCK=/tmp/hackrl-spatial-wall-pass-dual-adaptation-v1-gpu.lock

mkdir -p "$RUN_ROOT"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "spatial Dual adaptation GPU lock is already held" >&2
  exit 75
fi

"$PYTHON" "$WORKTREE/scripts/stop_identity_checked_gpu_keepalive.py"   >>"$RUN_ROOT/operations.log" 2>&1

cd "$WORKTREE"
export HACKRL_DEVICE=cuda
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HACKRL_DUMMY_KEEPALIVE_CLEARED=1
export HACKRL_GPU_GUARD_ACTIVE=1
export PYTHONPATH="$WORKTREE/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export EIGEN_NUM_THREADS=1

"$PYTHON" scripts/run_spatial_wall_pass_dual_adaptation.py --run-queue   >>"$RUN_ROOT/launcher.log" 2>&1 &
RUNNER_PID=$!
"$PYTHON" scripts/guard_identity_checked_gpu_keepalive.py   --runner-pid "$RUNNER_PID" --log "$RUN_ROOT/keepalive_guard.jsonl" &
GUARD_PID=$!

printf '{"event":"launcher_started","runner_pid":%s,"guard_pid":%s,"utc":"%s"}\n'   "$RUNNER_PID" "$GUARD_PID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"   >>"$RUN_ROOT/operations.log"

set +e
wait "$RUNNER_PID"
RUNNER_STATUS=$?
wait "$GUARD_PID"
GUARD_STATUS=$?
set -e

printf '{"event":"launcher_finished","runner_status":%s,"guard_status":%s,"utc":"%s"}\n'   "$RUNNER_STATUS" "$GUARD_STATUS" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"   >>"$RUN_ROOT/operations.log"

if [[ "$GUARD_STATUS" -ne 0 ]]; then
  exit "$GUARD_STATUS"
fi
if [[ "$RUNNER_STATUS" -ne 0 ]]; then
  exit "$RUNNER_STATUS"
fi
exit 0
