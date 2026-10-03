#!/usr/bin/env bash
# GC-PPO vs Dual LEO. One worker, no time limit.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HACKRL_DUAL_LEO=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export EIGEN_NUM_THREADS=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_force_compilation_parallelism=1"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
LOG_DIR="${HACKRL_DUAL_LEO_ROOT:-$ROOT/runs/dual_leo_compare_v1}"
WORKER="${HACKRL_SHARD:-all}"
mkdir -p "$LOG_DIR"

echo "[queue] dual-leo worker=$WORKER device=${HACKRL_DEVICE:-cpu} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PYTHON" "$ROOT/scripts/run_dual_leo_compare.py" --log-dir "$LOG_DIR" --worker "$WORKER"
echo "[queue] dual-leo worker=$WORKER stopped $(date -u +%Y-%m-%dT%H:%M:%SZ)"
