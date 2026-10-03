#!/usr/bin/env bash
# Teacher-learning versus frozen-teacher imitation. One worker, no time limit.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HACKRL_TEACHER_IMITATION=1
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
LOG_DIR="${HACKRL_TEACHER_IMITATION_ROOT:-$ROOT/runs/teacher_imitation_split_v1}"
WORKER="${HACKRL_SHARD:-all}"
mkdir -p "$LOG_DIR"

echo "[queue] teacher-imitation worker=$WORKER device=${HACKRL_DEVICE:-cpu} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PYTHON" "$ROOT/scripts/run_teacher_imitation_split.py" --log-dir "$LOG_DIR" --worker "$WORKER"
echo "[queue] teacher-imitation worker=$WORKER stopped $(date -u +%Y-%m-%dT%H:%M:%SZ)"
