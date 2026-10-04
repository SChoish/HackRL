#!/usr/bin/env bash
# Three-seed fixed-kernel mine-expedition learnability gate. No time limit.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
RUN_ROOT="${HACKRL_MINE_FIXED_ROOT:-/raid/ext_csv/HackRL/runs/mine_expedition_fixed_learnability_v1}"
PACK_EXECUTION_ROOT="${HACKRL_PACK_EXECUTION_ROOT:-/raid/ext_csv/HackRL/worktrees/pack_restore_scale_bc_eaf603b}"
PACK_RUN_ROOT="${HACKRL_PACK_RUN_ROOT:-/home/ext_csv/HackRL/runs/pack_restore_scale_bc_v1}"
PACK_EXECUTION_SHA="eaf603b1ab378b34bf2a788927f59446cabdde50"

if pgrep -f '[r]un_pack_restore_scale_bc.py' >/dev/null; then
  printf '[mine-fixed-queue] PACK-RESTORE scale queue is still active; refusing concurrent launch\n' >&2
  exit 75
fi

mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/.queue.lock"
if ! flock -n 9; then
  printf '[mine-fixed-queue] another fixed-gate worker owns %s\n' "$RUN_ROOT" >&2
  exit 75
fi
exec > >(tee -a "$RUN_ROOT/queue.log") 2>&1

export HACKRL_DEVICE="${HACKRL_DEVICE:-cuda}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export EIGEN_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"
actual_pack_sha="$(git -C "$PACK_EXECUTION_ROOT" rev-parse HEAD)"
if [[ "$actual_pack_sha" != "$PACK_EXECUTION_SHA" ]]; then
  printf '[mine-fixed-queue] predecessor checkout SHA mismatch: %s\n' "$actual_pack_sha" >&2
  exit 75
fi
if [[ -n "$(git -C "$PACK_EXECUTION_ROOT" status --porcelain -- \
  scripts/summarize_pack_restore_scale_bc.py \
  scripts/run_pack_restore_scale_bc.py \
  scripts/run_dual_leo_compare.py \
  src/hackrl/pack_restore_gc.py \
  src/hackrl/dual_leo.py)" ]]; then
  printf '[mine-fixed-queue] predecessor validation checkout is dirty\n' >&2
  exit 75
fi
predecessor_result="$RUN_ROOT/pack_restore_predecessor_result.json"
JAX_PLATFORMS=cpu PYTHONPATH="$PACK_EXECUTION_ROOT/src:$PACK_EXECUTION_ROOT/scripts" \
  "$PYTHON" "$PACK_EXECUTION_ROOT/scripts/summarize_pack_restore_scale_bc.py" \
  --run-root "$PACK_RUN_ROOT" \
  --output "$predecessor_result" \
  --require-complete
"$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); actual=p.get("run_provenance", {}).get("execution_code_sha"); expected=sys.argv[2]; raise SystemExit(0 if p.get("status") == "complete" and actual == expected else 1)' \
  "$predecessor_result" "$PACK_EXECUTION_SHA"
if pgrep -f '[r]un_pack_restore_scale_bc.py' >/dev/null; then
  printf '[mine-fixed-queue] PACK-RESTORE restarted after completion validation\n' >&2
  exit 75
fi

JAX_PLATFORMS=cpu "$PYTHON" -m pytest -q \
  tests/test_mine_expedition.py \
  tests/test_mine_expedition_env.py \
  tests/test_mine_expedition_gate.py \
  tests/test_mine_expedition_ppo.py

for seed in 40 41 42; do
  if pgrep -f '[r]un_pack_restore_scale_bc.py' >/dev/null; then
    printf '[mine-fixed-queue] PACK-RESTORE became active before seed=%s\n' "$seed" >&2
    exit 75
  fi
  destination="$RUN_ROOT/seed${seed}"
  printf '[mine-fixed-queue] seed=%s device=%s start=%s\n' \
    "$seed" "$HACKRL_DEVICE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PYTHON" "$ROOT/scripts/run_mine_expedition_fixed_gate.py" \
    --seed "$seed" \
    --log-dir "$destination"
done

"$PYTHON" "$ROOT/scripts/summarize_mine_expedition_fixed_gate.py" \
  --run-root "$RUN_ROOT" \
  --output "$RUN_ROOT/gate_result.json"

printf '[mine-fixed-queue] complete=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
