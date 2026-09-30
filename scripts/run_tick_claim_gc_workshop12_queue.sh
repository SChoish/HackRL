#!/usr/bin/env bash
# TICK-CLAIM fixed workshop12 base ability: seeds 0,1,2 x 512 updates.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HACKRL_WORKSHOP12=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export EIGEN_NUM_THREADS=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_force_compilation_parallelism=1"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
RUN_ROOT="${HACKRL_WORKSHOP12_ROOT:-$ROOT/runs/tick_claim_gc_workshop12_base_v1}"
DEVICE="${HACKRL_DEVICE:-cpu}"
SHARD="${HACKRL_SHARD:-all}"
mkdir -p "$RUN_ROOT"

if ! git -C "$ROOT" diff --quiet -- \
  src/hackrl/tick_claim_gc.py \
  scripts/run_tick_claim_gc.py \
  scripts/run_tick_claim_gc_workshop12_queue.sh \
  docs/manifests/tick_claim_gc_workshop12_base_v1.json
then
  echo "refusing to start with a dirty workshop12 source" >&2
  exit 1
fi

cell_done() {
  local cell="$1"
  "$PYTHON" - "$RUN_ROOT/$cell" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = root / "summary.json"
final_path = root / "checkpoint_final" / "metadata.json"
if not summary_path.is_file() or not final_path.is_file():
    raise SystemExit(1)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
final = json.loads(final_path.read_text(encoding="utf-8"))
if summary.get("goal_mode") != "workshop12" or summary.get("variant") != "fixed":
    raise SystemExit(1)
if int(summary.get("updates", -1)) != 512 or int(final.get("global_update", -1)) != 512:
    raise SystemExit(1)
if int(summary.get("transitions", -1)) != 16_777_216:
    raise SystemExit(1)
for update in (0, 32, 128, 256, 512):
    meta = json.loads(
        (root / "checkpoints" / f"update_{update}" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if int(meta.get("global_update", -1)) != update:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

run_cell() {
  local seed="$1"
  local cell="fixed_seed${seed}"
  if cell_done "$cell"; then
    echo "[skip] $cell"
    return 0
  fi
  echo "[start] $cell device=$DEVICE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PYTHON" "$ROOT/scripts/run_tick_claim_gc.py" \
    --variant fixed \
    --goal-mode workshop12 \
    --seed "$seed" \
    --num-envs 512 \
    --num-steps 64 \
    --num-updates 512 \
    --minibatch-size 1024 \
    --hidden-size 512 \
    --mode-repeats 1 \
    --sample-repeats 4 \
    --checkpoint-updates 0,32,128,256,512 \
    --log-dir "$RUN_ROOT/$cell"
  cell_done "$cell"
  echo "[done] $cell $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

seeds=()
case "$SHARD" in
  gpu0|0) seeds=(0) ;;
  gpu1|1) seeds=(1 2) ;;
  all|cpu) seeds=(0 1 2) ;;
  *) echo "unknown shard $SHARD" >&2; exit 1 ;;
esac

echo "[queue] workshop12 device=$DEVICE shard=$SHARD root=$RUN_ROOT sha=$(git -C "$ROOT" rev-parse HEAD) seeds=${seeds[*]}"
for seed in "${seeds[@]}"; do
  run_cell "$seed"
done
echo "[queue] workshop12 shard=$SHARD complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
