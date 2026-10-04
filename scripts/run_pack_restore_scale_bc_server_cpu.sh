#!/usr/bin/env bash
# CPU fallback registered with the ext_csv host watcher.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
unset CUDA_VISIBLE_DEVICES
exec bash "$ROOT/scripts/run_pack_restore_scale_bc_server_common.sh" cpu
