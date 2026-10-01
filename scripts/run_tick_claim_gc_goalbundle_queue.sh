#!/usr/bin/env bash
# Non-reservation goal-bundle comparison. No time limit.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HACKRL_GOALBUNDLE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export EIGEN_NUM_THREADS=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_force_compilation_parallelism=1"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/src:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
LOG_DIR="${HACKRL_GOALBUNDLE_ROOT:-$ROOT/runs/tick_claim_gc_goalbundle_v1}"
WORKER="${HACKRL_SHARD:-all}"
mkdir -p "$LOG_DIR"

if ! git -C "$ROOT" diff --quiet -- \
  src/hackrl/tick_claim.py \
  src/hackrl/tick_claim_gc.py \
  scripts/run_tick_claim_gc_goalbundle.py \
  scripts/run_tick_claim_gc_goalbundle_queue.sh \
  scripts/eval_tick_claim_reservation_probe.py \
  docs/manifests/tick_claim_gc_goalbundle_v1.json
then
  echo "refusing to start the goal-bundle sweep from a dirty source" >&2
  exit 1
fi

echo "[queue] goalbundle worker=$WORKER device=${HACKRL_DEVICE:-cpu} sha=$(git -C "$ROOT" rev-parse HEAD)"
"$PYTHON" "$ROOT/scripts/run_tick_claim_gc_goalbundle.py" \
  --log-dir "$LOG_DIR" \
  --worker "$WORKER"
echo "[queue] goalbundle worker=$WORKER stopped $(date -u +%Y-%m-%dT%H:%M:%SZ)"
