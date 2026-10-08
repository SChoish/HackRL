#!/usr/bin/env bash
# Resumable single-GPU entry for the seen-goal-fixed development rerun.
set -euo pipefail

WORKTREE=/raid/ext_csv/HackRL/worktrees/online_algorithm_expansion_v1_development_seen_goals_fix_v1
RUN_ROOT=/raid/ext_csv/HackRL/runs/online_algorithm_expansion_v1_development_seen_goals_fix_v1
PYTHON=/home/ext_csv/miniconda3/envs/offrl/bin/python
LOCK=/tmp/hackrl-online-algorithm-seen-goals-fix-v1-gpu.lock

if [[ ! -f "$WORKTREE/scripts/run_online_algorithm_seen_goals_fix.py" ]]; then
  echo "missing corrected development worktree" >&2
  exit 66
fi

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "corrected development GPU lock is already held" >&2
  exit 75
fi

"$PYTHON" "$WORKTREE/scripts/stop_identity_checked_gpu_keepalive.py"

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
