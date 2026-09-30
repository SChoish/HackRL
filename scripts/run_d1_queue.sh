#!/usr/bin/env bash
# D1 only: patched fixed, start after the 5-step normal prefix, 3 seeds x 1.05M.
# D2/D3 stay unqueued until D1 shows any learning.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HACKRL_D1=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export EIGEN_NUM_THREADS=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_force_compilation_parallelism=1"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
RUN_ROOT="${HACKRL_D1_ROOT:-$ROOT/runs/post_stage_a_d1}"
DEVICE="${HACKRL_DEVICE:-cuda}"
SHARD="${HACKRL_SHARD:-all}"
mkdir -p "$RUN_ROOT"

COMMON=(
  --task R-M
  --variant fixed
  --dynamics patched
  --start-mode r_m_d1
  --num-envs 128
  --num-steps 64
  --num-updates 128
  --update-epochs 4
  --num-minibatches 8
  --layer-size 256
  --eval-episodes 32
  --eval-sample-episodes 128
  --learning-rate 2e-4
  --checkpoint-updates 0,32,64,128
)

run_cell() {
  local cell="$1"
  local seed="$2"
  if [[ -f "$RUN_ROOT/$cell/summary.json" ]]; then
    echo "[skip] $cell"
    return 0
  fi
  echo "[start] $cell $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PYTHON" "$ROOT/scripts/run_ppo_pilot.py" \
    "${COMMON[@]}" \
    --seed "$seed" \
    --log-dir "$RUN_ROOT/$cell"
  "$PYTHON" "$ROOT/scripts/check_stage_a_cell.py" "$RUN_ROOT/$cell"
  echo "[done] $cell $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

seeds=()
case "$SHARD" in
  gpu0|0) seeds=(0) ;;
  gpu1|1) seeds=(1 2) ;;
  all|fixed) seeds=(0 1 2) ;;
  mutant) echo "[skip] D1 is fixed-only shard=$SHARD"; exit 0 ;;
  *) echo "unknown shard $SHARD" >&2; exit 1 ;;
esac

echo "[queue] D1 device=$DEVICE shard=$SHARD root=$RUN_ROOT sha=$(git -C "$ROOT" rev-parse HEAD) seeds=${seeds[*]}"
for seed in "${seeds[@]}"; do
  run_cell "d1_patched_fixed_s${seed}" "$seed"
done
echo "[queue] D1 complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
