#!/usr/bin/env bash
# Manual dual-GPU launcher. Prefer the watcher GPU wrapper:
# server_management/scripts/hosts/beyondg/run_queue_hackrl_r_e_replenish_gpu.sh
# Does not overwrite pilot_pe_seed0 or re_fixed_diagnosis_* artifacts.
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
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/r_e_replenish_compare_${STAMP}}"
mkdir -p "$RUN_ROOT"

run_gpu_queue() {
  local gpu="$1"
  local cpus="$2"
  local log="$RUN_ROOT/gpu${gpu}.log"
  shift 2
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    cd "$ROOT"
    for spec in "$@"; do
      IFS=: read -r fixture variant seed <<<"$spec"
      cell="${fixture}_${variant}_s${seed}"
      echo "[start] $cell $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      taskset -c "$cpus" "$PYTHON" "$ROOT/scripts/run_ppo_pilot.py" \
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
  ) >"$log" 2>&1 &
  echo $! >"$RUN_ROOT/gpu${gpu}.pid"
}

# GPU 0: default/replenish × seeds 0-2, fixed. GPU 1: the mutant pair.
run_gpu_queue 0 0-15 \
  default:fixed:0 default:fixed:1 default:fixed:2 \
  r_e_replenish:fixed:0 r_e_replenish:fixed:1 r_e_replenish:fixed:2
run_gpu_queue 1 16-31 \
  default:mutant:0 default:mutant:1 default:mutant:2 \
  r_e_replenish:mutant:0 r_e_replenish:mutant:1 r_e_replenish:mutant:2

echo "compare root $RUN_ROOT"
echo "gpu0 pid $(cat "$RUN_ROOT/gpu0.pid")"
echo "gpu1 pid $(cat "$RUN_ROOT/gpu1.pid")"
echo "$RUN_ROOT" > "$ROOT/logs/r_e_replenish_compare.latest"
