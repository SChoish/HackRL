#!/usr/bin/env bash
# Await GPU allocation without changing the registered inference backend.
set -euo pipefail
trap 'exit 0' TERM INT
while true; do sleep 30 & wait "$!"; done
