#!/usr/bin/env bash
# Resumable single-GPU entry for the seen-goal-fixed development rerun.
set -euo pipefail

WORKTREE=/raid/ext_csv/HackRL/worktrees/online_algorithm_expansion_v1_development_seen_goals_fix_v1
RUN_ROOT=/raid/ext_csv/HackRL/runs/online_algorithm_expansion_v1_development_seen_goals_fix_v1
PYTHON=/home/ext_csv/miniconda3/envs/offrl/bin/python
LOCK=/tmp/hackrl-online-algorithm-seen-goals-fix-v1-gpu.lock

if [[ ! -f "$WORKTREE/scripts/run_online_algorithm_seen_goals_fix.py" ]]; then
  exit 0
fi
if [[ -f "$RUN_ROOT/development_gate.json" ]]; then
  exit 0
fi

exec 9>"$LOCK"
if ! flock -n 9; then
  exit 0
fi

mapfile -t GPU_PIDS < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
for GPU_PID in "${GPU_PIDS[@]}"; do
  GPU_PID="${GPU_PID//[[:space:]]/}"
  [[ -n "$GPU_PID" ]] || continue
  if [[ ! -r "/proc/$GPU_PID/cmdline" ]]; then
    exit 0
  fi
  COMMAND=$(tr '\0' ' ' < "/proc/$GPU_PID/cmdline")
  if [[ "$COMMAND" != *"--beyondg-gpu-keepalive"* ]]; then
    exit 0
  fi
  kill -TERM "$GPU_PID"
  for _ in {1..100}; do
    if ! kill -0 "$GPU_PID" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  if kill -0 "$GPU_PID" 2>/dev/null; then
    exit 1
  fi
done

cd "$WORKTREE"
export HACKRL_DEVICE=cuda
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HACKRL_DUMMY_KEEPALIVE_CLEARED=1
export PYTHONPATH="$WORKTREE/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export EIGEN_NUM_THREADS=1

exec "$PYTHON" scripts/run_online_algorithm_seen_goals_fix.py --log-dir "$RUN_ROOT" --worker beyondg-gpu-watcher
