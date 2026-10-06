#!/usr/bin/env bash
# Pinned host-managed entry point for the fixed curriculum diagnostic.
set -euo pipefail

MODE="${1:?expected cpu or gpu}"
EXECUTION_ROOT="/raid/ext_csv/HackRL/worktrees/mine_expedition_curriculum_8477e4e"
EXECUTION_SHA="8477e4ea63ada6d7a8e9adfe52086a69f3bd07a2"
RUN_ROOT="/raid/ext_csv/HackRL/runs/mine_expedition_fixed_curriculum_diagnostic_v1"
export PYTHON="/home/ext_csv/miniconda3/envs/offrl/bin/python"

if [[ "$MODE" != "cpu" && "$MODE" != "gpu" ]]; then
  printf '[mine-curriculum-server] invalid mode: %s\n' "$MODE" >&2
  exit 2
fi
if [[ ! -d "$EXECUTION_ROOT" ]] \
  || [[ "$(git -C "$EXECUTION_ROOT" rev-parse HEAD)" != "$EXECUTION_SHA" ]]; then
  printf '[mine-curriculum-server] pinned execution checkout is unavailable or changed\n' >&2
  exit 2
fi
if [[ ! -x "$EXECUTION_ROOT/scripts/run_mine_expedition_fixed_curriculum_diagnostic_queue.sh" ]]; then
  printf '[mine-curriculum-server] pinned queue is unavailable\n' >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
target_free_bytes="$(df -B1 --output=avail "$RUN_ROOT" | tail -n 1 | tr -d ' ')"
home_free_bytes="$(df -B1 --output=avail /home/ext_csv | tail -n 1 | tr -d ' ')"
minimum_free_bytes=$((10 * 1024 * 1024 * 1024))
printf '[mine-curriculum-server] capacity utc=%s mode=%s target_free=%s home_free=%s minimum_free=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODE" "$target_free_bytes" \
  "$home_free_bytes" "$minimum_free_bytes"
if (( target_free_bytes < minimum_free_bytes )); then
  printf '[mine-curriculum-server] refusing start: RAID lacks the conservative run reserve\n' >&2
  exit 75
fi

if [[ "$MODE" == "gpu" ]]; then
  export HACKRL_DEVICE=cuda
else
  export HACKRL_DEVICE=cpu
  export CUDA_VISIBLE_DEVICES=""
fi
export HACKRL_MINE_CURRICULUM_ROOT="$RUN_ROOT"

cd "$EXECUTION_ROOT"
exec bash "$EXECUTION_ROOT/scripts/run_mine_expedition_fixed_curriculum_diagnostic_queue.sh"
