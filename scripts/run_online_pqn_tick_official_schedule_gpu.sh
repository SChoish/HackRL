#!/usr/bin/env bash
# Identity-checked single-GPU entry for the bounded PQN TICK schedule gate.
set -euo pipefail

WORKTREE=/raid/ext_csv/HackRL/worktrees/online_pqn_tick_official_schedule_v1
RUN_ROOT=/raid/ext_csv/HackRL/runs/online_pqn_tick_official_schedule_v1
PYTHON=/home/ext_csv/miniconda3/envs/offrl/bin/python
LOCK=/tmp/hackrl-online-pqn-tick-official-schedule-v1-gpu.lock

if [[ ! -f "$WORKTREE/scripts/run_online_pqn_tick_official_schedule.py" ]]; then
  echo "missing schedule-restoration worktree" >&2
  exit 66
fi

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "schedule-restoration GPU lock is already held" >&2
  exit 75
fi

"$PYTHON" "$WORKTREE/scripts/stop_identity_checked_gpu_keepalive.py"

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

"$PYTHON" scripts/run_online_pqn_tick_official_schedule.py --log-dir "$RUN_ROOT" "$@" \
  >>"$RUN_ROOT/launcher.log" 2>&1 &
RUNNER_PID=$!
"$PYTHON" scripts/guard_identity_checked_gpu_keepalive.py \
  --runner-pid "$RUNNER_PID" --log "$RUN_ROOT/keepalive_guard.jsonl" &
GUARD_PID=$!

set +e
FINISHED_PID=
wait -n -p FINISHED_PID "$RUNNER_PID" "$GUARD_PID"
FIRST_STATUS=$?
if [[ "$FINISHED_PID" == "$GUARD_PID" ]]; then
  GUARD_STATUS=$FIRST_STATUS
  if [[ "$GUARD_STATUS" -ne 0 ]]; then
    kill -TERM "$RUNNER_PID" 2>/dev/null
  fi
  wait "$RUNNER_PID"
  RUNNER_STATUS=$?
else
  RUNNER_STATUS=$FIRST_STATUS
  wait "$GUARD_PID"
  GUARD_STATUS=$?
fi
set -e
if [[ "$GUARD_STATUS" -ne 0 ]]; then
  exit "$GUARD_STATUS"
fi
if [[ "$RUNNER_STATUS" -ne 0 ]]; then
  exit "$RUNNER_STATUS"
fi
