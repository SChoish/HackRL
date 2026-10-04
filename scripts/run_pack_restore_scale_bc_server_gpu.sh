#!/usr/bin/env bash
# GPU entry point registered with the ext_csv host watcher.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec bash "$ROOT/scripts/run_pack_restore_scale_bc_server_common.sh" gpu
