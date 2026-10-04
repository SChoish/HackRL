#!/usr/bin/env bash
# Pinned entry point for the host-managed mine-expedition fixed gate.
set -euo pipefail

MODE="${1:?expected cpu or gpu}"
EXECUTION_ROOT="/raid/ext_csv/HackRL/worktrees/mine_expedition_fixed_734c50c"
EXECUTION_SHA="734c50c581a9a797c8f0fb83b88b6316c8b209b0"
RUN_ROOT="/raid/ext_csv/HackRL/runs/mine_expedition_fixed_learnability_v1"

if [[ "$MODE" != "cpu" && "$MODE" != "gpu" ]]; then
  printf '[mine-fixed-server] invalid mode: %s\n' "$MODE" >&2
  exit 2
fi
if [[ ! -d "$EXECUTION_ROOT" ]]; then
  printf '[mine-fixed-server] pinned execution checkout is unavailable\n' >&2
  exit 2
fi
if [[ "$(git -C "$EXECUTION_ROOT" rev-parse HEAD)" != "$EXECUTION_SHA" ]]; then
  printf '[mine-fixed-server] pinned execution checkout is unavailable or changed\n' >&2
  exit 2
fi
if [[ ! -x "$EXECUTION_ROOT/scripts/run_mine_expedition_fixed_gate_queue.sh" ]]; then
  printf '[mine-fixed-server] pinned queue is unavailable\n' >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
target_free_bytes="$(df -B1 --output=avail "$RUN_ROOT" | tail -n 1 | tr -d ' ')"
home_free_bytes="$(df -B1 --output=avail /home/ext_csv | tail -n 1 | tr -d ' ')"
minimum_free_bytes=$((10 * 1024 * 1024 * 1024))
printf '[mine-fixed-server] capacity utc=%s mode=%s target_free=%s home_free=%s minimum_free=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODE" "$target_free_bytes" \
  "$home_free_bytes" "$minimum_free_bytes"
if (( target_free_bytes < minimum_free_bytes )); then
  printf '[mine-fixed-server] refusing start: RAID lacks the conservative run reserve\n' >&2
  exit 75
fi

if [[ "$MODE" == "gpu" ]]; then
  export HACKRL_DEVICE=cuda
else
  export HACKRL_DEVICE=cpu
  unset CUDA_VISIBLE_DEVICES
fi
export HACKRL_MINE_FIXED_ROOT="$RUN_ROOT"

cd "$EXECUTION_ROOT"
exec bash "$EXECUTION_ROOT/scripts/run_mine_expedition_fixed_gate_queue.sh"
