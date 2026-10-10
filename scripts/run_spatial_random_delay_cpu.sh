#!/usr/bin/env bash
set -euo pipefail
trap '' HUP
# No CPU science queue; this task is GPU-only.
while true; do sleep 3600; done
