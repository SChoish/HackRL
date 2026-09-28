#!/usr/bin/env bash
# 12-cell R-E compare: fixture × variant × seed at the diagnosis budget.
# Skips cells that already have summary.json so CPU/GPU can resume the same tree.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HACKRL_RE_COMPARE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export EIGEN_NUM_THREADS=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_force_compilation_parallelism=1"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
RUN_ROOT="${HACKRL_RE_COMPARE_ROOT:-$ROOT/runs/r_e_replenish_compare}"
DEVICE="${HACKRL_DEVICE:-cuda}"
SHARD="${HACKRL_SHARD:-all}"
mkdir -p "$RUN_ROOT"

CELLS=(
  default:fixed:0
  default:fixed:1
  default:fixed:2
  r_e_replenish:fixed:0
  r_e_replenish:fixed:1
  r_e_replenish:fixed:2
  default:mutant:0
  default:mutant:1
  default:mutant:2
  r_e_replenish:mutant:0
  r_e_replenish:mutant:1
  r_e_replenish:mutant:2
)

cell_done() {
  local summary="$RUN_ROOT/$1/summary.json"
  [[ -f "$summary" ]]
}

selected=()
for spec in "${CELLS[@]}"; do
  IFS=: read -r fixture variant seed <<<"$spec"
  case "$SHARD" in
    fixed) [[ "$variant" == fixed ]] || continue ;;
    mutant) [[ "$variant" == mutant ]] || continue ;;
    all) ;;
    *) echo "unknown HACKRL_SHARD=$SHARD" >&2; exit 1 ;;
  esac
  selected+=("$spec")
done

echo "[queue] device=$DEVICE shard=$SHARD root=$RUN_ROOT sha=$(git -C "$ROOT" rev-parse HEAD)"

for spec in "${selected[@]}"; do
  IFS=: read -r fixture variant seed <<<"$spec"
  cell="${fixture}_${variant}_s${seed}"
  if cell_done "$cell"; then
    echo "[skip] $cell"
    continue
  fi
  echo "[start] $cell $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PYTHON" "$ROOT/scripts/run_ppo_pilot.py" \
    --task R-E \
    --variant "$variant" \
    --fixture "$fixture" \
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
