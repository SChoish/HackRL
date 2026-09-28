#!/usr/bin/env bash
# R-M fixed/mutant × seeds 0-2 at the R-E diagnosis budget.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HACKRL_R_M_COMPARE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export EIGEN_NUM_THREADS=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_force_compilation_parallelism=1"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
RUN_ROOT="${HACKRL_R_M_ROOT:-$ROOT/runs/r_m_compare}"
DEVICE="${HACKRL_DEVICE:-cuda}"
SHARD="${HACKRL_SHARD:-all}"
mkdir -p "$RUN_ROOT"

CELLS=(
  fixed:0
  fixed:1
  fixed:2
  mutant:0
  mutant:1
  mutant:2
)

echo "[queue] device=$DEVICE shard=$SHARD root=$RUN_ROOT sha=$(git -C "$ROOT" rev-parse HEAD)"

for spec in "${CELLS[@]}"; do
  IFS=: read -r variant seed <<<"$spec"
  case "$SHARD" in
    fixed) [[ "$variant" == fixed ]] || continue ;;
    mutant) [[ "$variant" == mutant ]] || continue ;;
    all) ;;
    *) echo "unknown HACKRL_SHARD=$SHARD" >&2; exit 1 ;;
  esac
  cell="${variant}_s${seed}"
  if [[ -f "$RUN_ROOT/$cell/summary.json" ]]; then
    echo "[skip] $cell"
    continue
  fi
  echo "[start] $cell $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PYTHON" "$ROOT/scripts/run_ppo_pilot.py" \
    --task R-M \
    --variant "$variant" \
    --seed "$seed" \
    --num-envs 128 \
    --num-steps 64 \
    --num-updates 32 \
    --update-epochs 4 \
    --num-minibatches 8 \
    --layer-size 256 \
    --eval-episodes 32 \
    --log-dir "$RUN_ROOT/$cell"
  echo "[done] $cell $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done
echo "[queue] shard=$SHARD complete"
