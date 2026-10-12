#!/usr/bin/env bash
set -euo pipefail
WORKTREE=/home/ext_csv/HackRL/.worktrees/spatial_multimap
ROOT=/raid/ext_csv/HackRL/runs/spatial_multimap_v1
PYTHON=/home/ext_csv/miniconda3/envs/offrl/bin/python
[[ ! -f "$ROOT/verified_complete.json" ]] || exit 0
[[ ! -f "$ROOT/failed.json" ]] || exit 0
if [[ "${HACKRL_VALIDATE_ONLY:-0}" == 1 ]]; then
  [[ ! -f "$ROOT/validation/verified_complete.json" ]] || exit 0
else
  [[ -f "$ROOT/launch_approval.json" ]] || exit 0
fi
if [[ -f "$ROOT/development_gate.json" ]]; then
  "$PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$ROOT/development_gate.json" || exit 0
fi
[[ -f /raid/ext_csv/HackRL/runs/spatial_multimap_v1_panel/maps_locked.json ]] || exit 1
mkdir -p "$ROOT"
exec 9>/tmp/hackrl-spatial-multimap-v1.lock
flock -n 9 || exit 75
cd "$WORKTREE"
export HACKRL_DEVICE=cuda JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$WORKTREE/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PYTHON" "$WORKTREE/scripts/stop_identity_checked_gpu_keepalive.py" >>"$ROOT/operations.log" 2>&1
ARGS=()
if [[ "${HACKRL_VALIDATE_ONLY:-0}" == 1 ]]; then ARGS+=(--validate-only); fi
"$PYTHON" "$WORKTREE/scripts/run_spatial_multimap.py" "${ARGS[@]}" >>"$ROOT/launcher.log" 2>&1 &
RUNNER_PID=$!
"$PYTHON" "$WORKTREE/scripts/guard_identity_checked_gpu_keepalive.py" --runner-pid "$RUNNER_PID" --log "$ROOT/guard.jsonl" &
GUARD_PID=$!
STATUS=0
wait "$RUNNER_PID" || STATUS=$?
wait "$GUARD_PID" || STATUS=$?
exit "$STATUS"
