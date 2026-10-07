#!/usr/bin/env bash
# Bounded GC Double DQN development gate or 18-job main queue.
set -euo pipefail

ROOT=/home/ext_csv/HackRL
PYTHON=/home/ext_csv/miniconda3/envs/offrl/bin/python
MODE="${1:-main}"

case "$MODE" in
  development)
    RUN_ROOT=/raid/ext_csv/HackRL/runs/gc_double_dqn_development_v1
    EXTRA=(--development)
    ;;
  main)
    RUN_ROOT=/raid/ext_csv/HackRL/runs/gc_double_dqn_two_defects_v1
    EXTRA=()
    ;;
  *)
    printf 'usage: %s [development|main]\n' "$0" >&2
    exit 64
    ;;
esac

SOURCES=(
  docs/manifests/gc_double_dqn_two_defects_v1.json
  scripts/evaluate_dual_teacher_greedy.py
  scripts/run_gc_double_dqn.py
  scripts/run_gc_double_dqn_queue.sh
  scripts/run_dual_leo_compare.py
  src/hackrl/gc_double_dqn.py
  src/hackrl/dual_leo.py
  src/hackrl/tick_claim.py
  src/hackrl/tick_claim_gc.py
  src/hackrl/pack_restore.py
  src/hackrl/pack_restore_gc.py
)

cd "$ROOT"
for source in "${SOURCES[@]}"; do
  git ls-files --error-unmatch "$source" >/dev/null
  if ! git diff --quiet HEAD -- "$source"; then
    printf '[ddqn] execution source differs from HEAD: %s\n' "$source" >&2
    exit 75
  fi
done

mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/.queue.lock"
if ! flock -n 9; then
  printf '[ddqn] another %s queue owns %s\n' "$MODE" "$RUN_ROOT" >&2
  exit 75
fi
exec > >(tee -a "$RUN_ROOT/queue.log") 2>&1

export HACKRL_DEVICE=cuda
unset CUDA_VISIBLE_DEVICES JAX_PLATFORMS JAX_PLATFORM_NAME
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export EIGEN_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/src:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

printf '[ddqn] mode=%s execution_sha=%s run_root=%s start=%s\n' \
  "$MODE" "$(git rev-parse HEAD)" "$RUN_ROOT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$PYTHON" scripts/run_gc_double_dqn.py \
  --log-dir "$RUN_ROOT" \
  --worker "ddqn-${MODE}-gpu" \
  "${EXTRA[@]}"
