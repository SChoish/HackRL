#!/usr/bin/env bash
# Short R-E fixed diagnosis: default reset vs post-iron start, one GPU each.
#
# The 2026-09-28 two-cell run used this budget and start-mode split before
# reset-time ever_iron seeding. Those artifacts are under
# runs/re_fixed_diagnosis_20260928T114712Z/. Current HEAD seeds ever_iron
# from the start inventory so 1-step post-iron crafts are counted.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export EIGEN_NUM_THREADS=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_force_compilation_parallelism=1"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/re_fixed_diagnosis_${STAMP}}"
mkdir -p "$RUN_ROOT"

COMMON=(
  "$PYTHON" "$ROOT/scripts/run_ppo_pilot.py"
  --task R-E
  --variant fixed
  --seed 0
  --num-envs 128
  --num-steps 64
  --num-updates 32
  --update-epochs 4
  --num-minibatches 8
  --layer-size 256
  --eval-episodes 32
)

echo "diagnosis root $RUN_ROOT"
echo "python $PYTHON"

CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 "${COMMON[@]}" \
  --start-mode default \
  --log-dir "$RUN_ROOT/r_e_fixed_default" \
  > "$RUN_ROOT/r_e_fixed_default.log" 2>&1 &
echo $! > "$RUN_ROOT/r_e_fixed_default.pid"

CUDA_VISIBLE_DEVICES=1 taskset -c 16-31 "${COMMON[@]}" \
  --start-mode r_e_post_iron \
  --log-dir "$RUN_ROOT/r_e_fixed_post_iron" \
  > "$RUN_ROOT/r_e_fixed_post_iron.log" 2>&1 &
echo $! > "$RUN_ROOT/r_e_fixed_post_iron.pid"

echo "launched default pid $(cat "$RUN_ROOT/r_e_fixed_default.pid")"
echo "launched post_iron pid $(cat "$RUN_ROOT/r_e_fixed_post_iron.pid")"
echo "$RUN_ROOT" > "$ROOT/logs/re_fixed_diagnosis.latest"
