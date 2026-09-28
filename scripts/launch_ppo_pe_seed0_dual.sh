#!/usr/bin/env bash
# P-E PPO seed 0 split across GPU 0 and 1. One process per GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/logs"
PY="${PY:-/home/ext_csv/miniconda3/envs/offrl/bin/python3.12}"
mkdir -p "$LOG"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export EIGEN_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1 --xla_gpu_autotune_level=0 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

run_pairs() {
  local gpu="$1" cpus="$2" tag="$3"
  shift 3
  local log="$LOG/${tag}.log"
  local pidf="$LOG/${tag}.pid"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda \
    taskset -c "$cpus" \
    bash -c '
      set -euo pipefail
      PY="$1"; ROOT="$2"; shift 2
      cd "$ROOT"
      while [[ $# -ge 2 ]]; do
        task="$1"; variant="$2"; shift 2
        echo "[start] task=$task variant=$variant $(date -Iseconds)"
        "$PY" -u scripts/run_ppo_pilot.py \
          --task "$task" \
          --variant "$variant" \
          --seed 0 \
          --num-envs 512 \
          --num-steps 64 \
          --num-updates 256 \
          --update-epochs 4 \
          --num-minibatches 8 \
          --layer-size 256 \
          --eval-episodes 32
        echo "[done] task=$task variant=$variant $(date -Iseconds)"
      done
    ' bash "$PY" "$ROOT" "$@" \
    </dev/null >"$log" 2>&1 &
  local pid=$!
  disown "$pid" 2>/dev/null || true
  echo "$pid" >"$pidf"
  echo "gpu=$gpu pid=$pid tag=$tag log=$log"
}

cd "$ROOT"
run_pairs 0 200-207 ppo_pe_s0_gpu0 R-E mutant B-E mutant L-E mutant
sleep 3
run_pairs 1 208-215 ppo_pe_s0_gpu1 R-E fixed B-E fixed L-E fixed
