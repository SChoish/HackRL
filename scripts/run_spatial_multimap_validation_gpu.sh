#!/usr/bin/env bash
set -euo pipefail
export HACKRL_VALIDATE_ONLY=1
exec bash /home/ext_csv/HackRL/.worktrees/spatial_multimap/scripts/run_spatial_multimap_gpu.sh
