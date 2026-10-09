#!/usr/bin/env bash
# Single-GPU entry for the preregistered independent Dual replication.
set -euo pipefail

WORKTREE=/raid/ext_csv/HackRL/worktrees/spatial_wall_pass_v1
RUN_ROOT=/raid/ext_csv/HackRL/runs/spatial_wall_pass_dual_replication_v1
PYTHON=/home/ext_csv/miniconda3/envs/offrl/bin/python
LOCK=/tmp/hackrl-spatial-wall-pass-dual-replication-v1-gpu.lock

mkdir -p "$RUN_ROOT"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "spatial Dual replication GPU lock is already held" >&2
  exit 75
fi

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

exec "$PYTHON" scripts/run_spatial_wall_pass_dual_replication.py --run-queue
