#!/usr/bin/env bash
set -euo pipefail
cd /home/ext_csv/HackRL/.worktrees/pack_teacher_fixed_control_v1
export HACKRL_DEVICE=cuda JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD/src"
exec /home/ext_csv/miniconda3/envs/offrl/bin/python -u /home/ext_csv/HackRL/.worktrees/pack_teacher_fixed_control_v1/scripts/run_pack_teacher_fixed_control_v1.py --worker
