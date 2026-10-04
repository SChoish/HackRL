#!/usr/bin/env bash
# Shared capacity gate and pinned entry point for the host-managed PACK-RESTORE queue.
set -euo pipefail

MODE="${1:?expected cpu or gpu}"
EXECUTION_ROOT="/raid/ext_csv/HackRL/worktrees/pack_restore_scale_bc_eaf603b"
RUN_ROOT="/home/ext_csv/HackRL/runs/pack_restore_scale_bc_v1"
PROJECTED_FINAL_BYTES=41573442180
SAFETY_MARGIN_BYTES=$((8 * 1024 * 1024 * 1024))

if [[ "$MODE" != "cpu" && "$MODE" != "gpu" ]]; then
  printf '[server-queue] invalid mode: %s\n' "$MODE" >&2
  exit 2
fi
if [[ ! -x "$EXECUTION_ROOT/scripts/run_pack_restore_scale_bc_queue.sh" ]]; then
  printf '[server-queue] pinned queue is unavailable: %s\n' "$EXECUTION_ROOT" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
current_bytes="$(du -sbL "$RUN_ROOT" | awk '{print $1}')"
target_free_bytes="$(df -B1 --output=avail "$RUN_ROOT" | tail -n 1 | tr -d ' ')"
home_free_bytes="$(df -B1 --output=avail /home/ext_csv | tail -n 1 | tr -d ' ')"
remaining_bytes=$((PROJECTED_FINAL_BYTES - current_bytes))
if (( remaining_bytes < 0 )); then
  remaining_bytes=0
fi
required_bytes=$((remaining_bytes + SAFETY_MARGIN_BYTES))
printf '[server-queue] capacity utc=%s mode=%s current=%s remaining=%s required=%s target_free=%s home_free=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODE" "$current_bytes" "$remaining_bytes" \
  "$required_bytes" "$target_free_bytes" "$home_free_bytes"
if (( target_free_bytes < required_bytes )); then
  printf '[server-queue] refusing start: RAID capacity is below projected peak plus reserve\n' >&2
  exit 75
fi

if [[ "$MODE" == "gpu" ]]; then
  export HACKRL_DEVICE="cuda"
else
  export HACKRL_DEVICE="cpu"
fi
export HACKRL_SHARD="server-$MODE"
export HACKRL_PACK_RESTORE_SCALE_BC_ROOT="$RUN_ROOT"

cd "$EXECUTION_ROOT"
exec bash "$EXECUTION_ROOT/scripts/run_pack_restore_scale_bc_queue.sh"
