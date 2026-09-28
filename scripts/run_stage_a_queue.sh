#!/usr/bin/env bash
# Overnight Stage A: A1 (legacy vs patched, 262k, annealed LR) then A2 (patched, 4.19M, fixed LR).
# GPU 0=fixed, GPU 1=mutant. Success rate 0 is a valid result.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HACKRL_STAGE_A=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export EIGEN_NUM_THREADS=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_force_compilation_parallelism=1"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
RUN_ROOT="${HACKRL_STAGE_A_ROOT:-$ROOT/runs/overnight_stage_a}"
DEVICE="${HACKRL_DEVICE:-cuda}"
SHARD="${HACKRL_SHARD:-all}"
mkdir -p "$RUN_ROOT"

COMMON=(
  --task R-M
  --num-envs 128
  --num-steps 64
  --update-epochs 4
  --num-minibatches 8
  --layer-size 256
  --eval-episodes 32
  --learning-rate 2e-4
)

run_cell() {
  local cell="$1"
  shift
  if [[ -f "$RUN_ROOT/$cell/summary.json" ]]; then
    echo "[skip] $cell"
    return 0
  fi
  echo "[start] $cell $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PYTHON" "$ROOT/scripts/run_ppo_pilot.py" "$@" --log-dir "$RUN_ROOT/$cell"
  "$PYTHON" "$ROOT/scripts/check_stage_a_cell.py" "$RUN_ROOT/$cell"
  echo "[done] $cell $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

echo "[queue] device=$DEVICE shard=$SHARD root=$RUN_ROOT sha=$(git -C "$ROOT" rev-parse HEAD)"

for variant in fixed mutant; do
  case "$SHARD" in
    fixed) [[ "$variant" == fixed ]] || continue ;;
    mutant) [[ "$variant" == mutant ]] || continue ;;
    all) ;;
    *) echo "unknown HACKRL_SHARD=$SHARD" >&2; exit 1 ;;
  esac

  for dynamics in legacy patched; do
    for seed in 0 1 2; do
      run_cell "a1_${dynamics}_${variant}_s${seed}" \
        "${COMMON[@]}" \
        --variant "$variant" \
        --seed "$seed" \
        --dynamics "$dynamics" \
        --num-updates 32 \
        --anneal-learning-rate
    done
  done

  for seed in 0 1 2; do
    run_cell "a2_patched_${variant}_s${seed}" \
      "${COMMON[@]}" \
      --variant "$variant" \
      --seed "$seed" \
      --dynamics patched \
      --num-updates 512 \
      --checkpoint-updates 0,32,128,512
  done
done

echo "[queue] shard=$SHARD complete"
