#!/usr/bin/env bash
# GPU entry point for the ext_csv host watcher.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0
exec bash "$ROOT/scripts/run_mine_expedition_fixed_curriculum_server_common.sh" gpu
