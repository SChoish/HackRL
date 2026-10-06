#!/usr/bin/env bash
# CPU fallback entry point for the ext_csv host watcher.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/scripts/run_mine_expedition_fixed_curriculum_server_common.sh" cpu
