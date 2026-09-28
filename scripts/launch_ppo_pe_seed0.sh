#!/usr/bin/env bash
# P-E first PPO seed 0: Easy 3 tasks × fixed/mutant. GPU 0 only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/logs"
PY="${PY:-/home/ext_csv/miniconda3/envs/offrl/bin/python3.12}"
mkdir -p "$LOG"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export EIGEN_NUM_THREADS=1
export D4RL_SUPPRESS_IMPORT_ERROR=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1 --xla_gpu_autotune_level=0 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

log="$LOG/ppo_pe_seed0.log"
pidf="$LOG/ppo_pe_seed0.pid"
cd "$ROOT"
nohup env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
  taskset -c 200-207 \
  "$PY" -u scripts/run_ppo_pilot.py \
    --all-pairs \
    --seed 0 \
    --num-envs 512 \
    --num-steps 64 \
    --num-updates 256 \
    --update-epochs 4 \
    --num-minibatches 8 \
    --layer-size 256 \
    --eval-episodes 32 \
  </dev/null >"$log" 2>&1 &
pid=$!
disown "$pid" 2>/dev/null || true
echo "$pid" >"$pidf"
echo "gpu=0 pid=$pid log=$log"
