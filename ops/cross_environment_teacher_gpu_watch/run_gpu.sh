#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ext_csv/HackRL
OUTPUT=/raid/ext_csv/HackRL/evaluations/cross_environment_teacher_matrix_v1_gpu
PYTHON=/home/ext_csv/miniconda3/envs/offrl/bin/python
RUNNER="$REPO/scripts/run_cross_environment_teacher_evaluation_gpu.py"
PROBE="$REPO/ops/cross_environment_teacher_gpu_watch/probe.py"
SHARDS=4

mkdir -p "$OUTPUT" "$OUTPUT/shard_logs"
cd "$REPO"
exec 9>"$OUTPUT/.server_watcher_gpu.lock"
flock 9

set +e
PROBE_OUTPUT=$("$PYTHON" "$PROBE")
PROBE_STATUS=$?
set -e
echo "[gpu-eval-watch] initial probe status=$PROBE_STATUS $PROBE_OUTPUT"
if [[ "$PROBE_STATUS" -eq 0 ]]; then
  exit 0
elif [[ "$PROBE_STATUS" -ne 10 ]]; then
  exit "$PROBE_STATUS"
fi

if [[ ! -f "$OUTPUT/evaluation_manifest.json" ]]; then
  PYTHONPATH=src:scripts "$PYTHON" "$RUNNER"     --prepare --authorize-common-evaluation
fi

PIDS=()
for SHARD in 0 1 2 3; do
  echo "[gpu-eval-watch] shard=$SHARD start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  env CUDA_VISIBLE_DEVICES=0 HACKRL_DEVICE=cuda JAX_PLATFORMS=cuda     XLA_PYTHON_CLIENT_PREALLOCATE=false     XLA_PYTHON_CLIENT_ALLOCATOR=platform     PYTHONPATH=src:scripts "$PYTHON" "$RUNNER"     --run-shard --shard-index "$SHARD" --shard-count "$SHARDS"     --authorize-common-evaluation     >"$OUTPUT/shard_logs/shard_${SHARD}.log" 2>&1 &
  PIDS+=("$!")
done

STATUS=0
for INDEX in 0 1 2 3; do
  if wait "${PIDS[$INDEX]}"; then
    echo "[gpu-eval-watch] shard=$INDEX passed $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  else
    echo "[gpu-eval-watch] shard=$INDEX failed $(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
    STATUS=1
  fi
done
if [[ "$STATUS" -ne 0 ]]; then
  "$PYTHON" "$PROBE" || true
  exit "$STATUS"
fi

PYTHONPATH=src:scripts "$PYTHON" "$RUNNER"   --finalize --authorize-common-evaluation
"$PYTHON" "$PROBE"
