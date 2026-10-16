#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/ext_csv/miniconda3/envs/offrl/bin/python
PROBE=/home/ext_csv/HackRL/ops/cross_environment_teacher_gpu_watch/probe.py

echo "[gpu-eval-watch] CPU fallback is wait-only $(date -u +%Y-%m-%dT%H:%M:%SZ)"
while true; do
  set +e
  OUTPUT=$("$PYTHON" "$PROBE")
  STATUS=$?
  set -e
  echo "[gpu-eval-watch] CPU probe status=$STATUS $OUTPUT"
  if [[ "$STATUS" -eq 0 ]]; then
    exit 0
  elif [[ "$STATUS" -ne 10 ]]; then
    exit "$STATUS"
  fi
  sleep 30
done
