#!/usr/bin/env bash
# Restart interrupted queue processes; stop on provenance/contract failures.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="${HACKRL_TEACHER_FREEZE_ROOT:-/raid/ext_csv/HackRL/runs/pack_restore_pretrained_teacher_freeze_v1}"
PYTHON="${PYTHON:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
MAX_RESTARTS="${HACKRL_WATCHDOG_MAX_RESTARTS:-6}"
DELAY_SECONDS="${HACKRL_WATCHDOG_DELAY_SECONDS:-30}"
mkdir -p "$RUN_ROOT"
exec 8>"$RUN_ROOT/.watchdog.lock"
if ! flock -n 8; then
  printf '[teacher-freeze-watchdog] another watchdog is active\n' >&2
  exit 75
fi
exec > >(tee -a "$RUN_ROOT/watchdog.log") 2>&1

attempt=0
while true; do
  result="$RUN_ROOT/result.json"
  if [[ -f "$result" ]] && "$PYTHON" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("execution_complete") is True else 1)' "$result"; then
    printf '[teacher-freeze-watchdog] complete=%s result=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$result"
    exit 0
  fi
  attempt=$((attempt + 1))
  printf '[teacher-freeze-watchdog] launch attempt=%s time=%s\n' \
    "$attempt" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  HACKRL_TEACHER_FREEZE_ROOT="$RUN_ROOT" \
    bash "$ROOT/scripts/run_pack_restore_pretrained_teacher_freeze_queue.sh"
  status=$?
  if (( status == 0 )); then
    continue
  fi
  if (( status == 75 )); then
    printf '[teacher-freeze-watchdog] non-retryable contract failure status=%s\n' "$status" >&2
    exit "$status"
  fi
  if (( attempt >= MAX_RESTARTS )); then
    printf '[teacher-freeze-watchdog] retry budget exhausted attempts=%s status=%s\n' \
      "$attempt" "$status" >&2
    exit "$status"
  fi
  printf '[teacher-freeze-watchdog] interrupted status=%s; retrying in %ss\n' \
    "$status" "$DELAY_SECONDS"
  sleep "$DELAY_SECONDS"
done
