#!/usr/bin/env bash
set -euo pipefail
trap 'exit 0' TERM INT
while true; do sleep 30 & wait $!; done
