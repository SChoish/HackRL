#!/usr/bin/env bash
# Overnight Stage A. Short A1/A2 cells are preserved. New long A2 uses a2x_* names.
# Overflow seeds 3-4 start only when a pair can finish 45 minutes before lease end.
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
LEASE="${HACKRL_LEASE:-/home/ext_csv/MPI_sweep/logs/beyondg/lease.json}"
DEVICE="${HACKRL_DEVICE:-cuda}"
SHARD="${HACKRL_SHARD:-all}"
mkdir -p "$RUN_ROOT"

# Measured short A2: ~205s / 4.19M. 8.39M ≈ 410s, plus 45-minute lease buffer.
OVERFLOW_UPDATES=1024
OVERFLOW_MINUTES=7
LEASE_BUFFER_MIN=45

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

remaining_minutes() {
  python3 - "$LEASE" <<'PY'
import json, sys, time
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    print(0)
    raise SystemExit(0)
data = json.loads(path.read_text())
remain_h = float(data.get("remaining_h", 0)) - (time.time() - float(data["ts"])) / 3600
print(max(0, int(remain_h * 60)))
PY
}

can_start_overflow() {
  local remain
  remain="$(remaining_minutes)"
  local need=$((OVERFLOW_MINUTES + LEASE_BUFFER_MIN))
  echo "[lease] remaining_min=$remain need_min=$need for overflow"
  [[ "$remain" -ge "$need" ]]
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

  # Preserved short diagnostic A2 (4.19M). Skip if already complete.
  for seed in 0 1 2; do
    run_cell "a2_patched_${variant}_s${seed}" \
      "${COMMON[@]}" \
      --variant "$variant" \
      --seed "$seed" \
      --dynamics patched \
      --num-updates 512 \
      --checkpoint-updates 0,32,128,512
  done

  for seed in 0 1 2; do
    run_cell "a2x_patched_${variant}_s${seed}" \
      "${COMMON[@]}" \
      --variant "$variant" \
      --seed "$seed" \
      --dynamics patched \
      --num-updates 2048 \
      --checkpoint-updates 0,32,128,512,1024,2048
  done

  for seed in 3 4; do
    if ! can_start_overflow; then
      echo "[overflow] skip ${variant} seed $seed; not enough lease"
      break
    fi
    run_cell "a2o_patched_${variant}_s${seed}" \
      "${COMMON[@]}" \
      --variant "$variant" \
      --seed "$seed" \
      --dynamics patched \
      --num-updates "$OVERFLOW_UPDATES" \
      --checkpoint-updates 0,32,128,512,1024
  done
done

echo "[queue] shard=$SHARD complete"
