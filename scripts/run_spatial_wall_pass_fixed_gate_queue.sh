#!/usr/bin/env bash
set -euo pipefail

repo=/raid/ext_csv/HackRL/worktrees/spatial_wall_pass_v1
run_root=/raid/ext_csv/HackRL/runs/spatial_wall_pass_fixed_learnability_v1
python_bin=/home/ext_csv/miniconda3/envs/offrl/bin/python
mkdir -p "$run_root"
exec 9>"$run_root/queue.lock"
if ! flock -n 9; then
  echo "queue already owned" >&2
  exit 1
fi

cd "$repo"
for cell in gc:130 dual:130 gc:131 dual:131; do
  method=${cell%%:*}
  seed=${cell##*:}
  destination="$run_root/${method}_seed_${seed}"
  summary="$destination/summary.json"
  if [[ -f "$summary" ]] && "$python_bin" -c 'import json,sys; raise SystemExit(json.load(open(sys.argv[1])).get("status") != "complete")' "$summary"; then
    echo "[queue] skip complete method=$method seed=$seed"
    continue
  fi
  mkdir -p "$destination"
  echo "[queue] start method=$method seed=$seed utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  HACKRL_DEVICE=cuda PYTHONPATH=src "$python_bin"     scripts/run_spatial_wall_pass_fixed_gate.py     --method "$method"     --seed "$seed"     --log-dir "$destination"     2>&1 | tee -a "$destination/run.log"
  echo "[queue] finish method=$method seed=$seed utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

echo "[queue] all complete utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
