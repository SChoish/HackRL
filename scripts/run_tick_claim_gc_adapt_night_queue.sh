#!/usr/bin/env bash
# A then B then C adaptation sweep. Two workers share one claim directory.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HACKRL_ADAPT_NIGHT=1
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
LOG_DIR="${HACKRL_NIGHT_ROOT:-$ROOT/runs/tick_claim_gc_workshop12_adapt_night_v1}"
WORKER="${HACKRL_SHARD:-all}"
mkdir -p "$LOG_DIR"

if ! git -C "$ROOT" diff --quiet -- \
  src/hackrl/tick_claim_gc.py \
  scripts/run_tick_claim_gc_adapt_night.py \
  scripts/run_tick_claim_gc_adapt_night_queue.sh \
  docs/manifests/tick_claim_gc_workshop12_adapt_night_v1.json
then
  echo "refusing to start the night sweep from a dirty source" >&2
  exit 1
fi

echo "[queue] night worker=$WORKER device=${HACKRL_DEVICE:-cpu} sha=$(git -C "$ROOT" rev-parse HEAD)"
"$PYTHON" "$ROOT/scripts/run_tick_claim_gc_adapt_night.py" \
  --log-dir "$LOG_DIR" \
  --worker "$WORKER" \
  --hours "${HACKRL_NIGHT_HOURS:-8}"
echo "[queue] night worker=$WORKER stopped $(date -u +%Y-%m-%dT%H:%M:%SZ)"
