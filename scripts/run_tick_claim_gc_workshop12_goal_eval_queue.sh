#!/usr/bin/env bash
# Frozen 12-goal evaluation of the workshop12 checkpoints. Does not train.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HACKRL_GOAL_EVAL=1
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
RUN_ROOT="${HACKRL_WORKSHOP12_ROOT:-$ROOT/runs/tick_claim_gc_workshop12_base_v1}"
LOG_DIR="${HACKRL_GOAL_EVAL_ROOT:-$ROOT/runs/tick_claim_gc_workshop12_goal_eval_v1}"
DEVICE="${HACKRL_DEVICE:-cpu}"
SHARD="${HACKRL_SHARD:-all}"
mkdir -p "$LOG_DIR"

if ! git -C "$ROOT" diff --quiet -- \
  src/hackrl/tick_claim_gc.py \
  src/hackrl/tick_claim_gc_goal_eval.py \
  scripts/eval_tick_claim_gc_workshop12_goals.py \
  scripts/run_tick_claim_gc_workshop12_goal_eval_queue.sh \
  docs/manifests/tick_claim_gc_workshop12_goal_eval_v1.json
then
  echo "refusing to score with a dirty goal-eval source" >&2
  exit 1
fi

case "$SHARD" in
  gpu0|0) SEEDS="0" ;;
  gpu1|1) SEEDS="1,2" ;;
  all|cpu) SEEDS="0,1,2" ;;
  *) echo "unknown shard $SHARD" >&2; exit 1 ;;
esac

echo "[queue] goal-eval device=$DEVICE shard=$SHARD sha=$(git -C "$ROOT" rev-parse HEAD) seeds=$SEEDS"
"$PYTHON" "$ROOT/scripts/eval_tick_claim_gc_workshop12_goals.py" \
  --run-root "$RUN_ROOT" \
  --log-dir "$LOG_DIR" \
  --seeds "$SEEDS" \
  --updates 0,32,128,256,512
echo "[queue] goal-eval shard=$SHARD complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
