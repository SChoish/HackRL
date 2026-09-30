#!/usr/bin/env bash
# 18-cell deliver_3 adaptation from workshop12 checkpoints 0/128/512.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HACKRL_ADAPT=1
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
LOG_DIR="${HACKRL_ADAPT_ROOT:-$ROOT/runs/tick_claim_gc_workshop12_adapt_v1}"
DEVICE="${HACKRL_DEVICE:-cpu}"
SHARD="${HACKRL_SHARD:-all}"
mkdir -p "$LOG_DIR"

if ! git -C "$ROOT" diff --quiet -- \
  src/hackrl/tick_claim_gc.py \
  scripts/run_tick_claim_gc_adapt.py \
  scripts/run_tick_claim_gc_adapt_queue.sh \
  docs/manifests/tick_claim_gc_workshop12_adapt_v1.json
then
  echo "refusing to adapt from a dirty source" >&2
  exit 1
fi

case "$SHARD" in
  gpu0|0) SEEDS="0" ;;
  gpu1|1) SEEDS="1,2" ;;
  all|cpu) SEEDS="0,1,2" ;;
  *) echo "unknown shard $SHARD" >&2; exit 1 ;;
esac

echo "[queue] adapt device=$DEVICE shard=$SHARD sha=$(git -C "$ROOT" rev-parse HEAD) seeds=$SEEDS"
"$PYTHON" "$ROOT/scripts/run_tick_claim_gc_adapt.py" \
  --run-root "$RUN_ROOT" \
  --log-dir "$LOG_DIR" \
  --seeds "$SEEDS" \
  --updates 0,128,512

"$PYTHON" - "$LOG_DIR" "$SEEDS" <<'PY'
import json
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
seeds = [int(item) for item in sys.argv[2].split(",") if item]
missing = []
for seed in seeds:
    for update in (0, 128, 512):
        pre = log_dir / "pre_adapt" / f"seed{seed}_update{update}.json"
        if not pre.is_file():
            missing.append(str(pre))
        for variant in ("fixed", "mutant"):
            cell = log_dir / f"from_update_{update}" / f"{variant}_seed{seed}"
            for relative in (
                "checkpoints/adapt_0/state.msgpack",
                "checkpoints/adapt_128/state.msgpack",
                "summary.json",
                "mutant_cross_eval.json",
            ):
                path = cell / relative
                if not path.is_file() or path.stat().st_size <= 0:
                    missing.append(str(path))
if missing:
    print("adaptation outputs missing:", *missing, sep="\n", file=sys.stderr)
    raise SystemExit(1)
PY
echo "[queue] adapt shard=$SHARD complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
