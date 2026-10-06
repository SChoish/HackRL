#!/usr/bin/env bash
# One predeclared fixed-only backward-curriculum diagnostic.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
RUN_ROOT="${HACKRL_MINE_CURRICULUM_ROOT:-/raid/ext_csv/HackRL/runs/mine_expedition_fixed_curriculum_diagnostic_v1}"
PREDECESSOR_ARCHIVE="/raid/ext_csv/HackRL/runs/_archive/mine_expedition_fixed_learnability_v1_initial_evidence_20261005T141101Z"

mkdir -p "$RUN_ROOT"
resolved_run_root="$(realpath "$RUN_ROOT")"
if [[ "$resolved_run_root" != /raid/ext_csv/HackRL/runs/* ]]; then
  printf '[mine-curriculum] refusing non-RAID run root: %s\n' "$resolved_run_root" >&2
  exit 75
fi
exec 9>"$RUN_ROOT/.queue.lock"
if ! flock -n 9; then
  printf '[mine-curriculum] another diagnostic worker owns %s\n' "$RUN_ROOT" >&2
  exit 75
fi

result="$RUN_ROOT/diagnostic_result.json"
current_sha="$(git -C "$ROOT" rev-parse HEAD)"
if [[ -f "$result" ]] && "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); ok=(d.get("schema_version") == "hackrl_mine_expedition_curriculum_phase_result_v1" and d.get("diagnostic_id") == "mine_expedition_fixed_curriculum_diagnostic_v1" and d.get("execution_complete") is True and d.get("phase") in ("stage_b_natural_late", "fallback_target_ready") and d.get("execution_code_shas") == [sys.argv[2]]); raise SystemExit(0 if ok else 1)' "$result" "$current_sha"; then
  printf '[mine-curriculum] already complete result=%s\n' "$result"
  exit 0
fi

exec > >(tee -a "$RUN_ROOT/queue.log") 2>&1
export HACKRL_DEVICE="${HACKRL_DEVICE:-cuda}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export EIGEN_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"
printf '[mine-curriculum] start=%s device=%s run_root=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$HACKRL_DEVICE" "$RUN_ROOT"

target_available_bytes="$(df -B1 --output=avail "$resolved_run_root" | tail -n 1 | tr -d ' ')"
home_available_bytes="$(df -B1 --output=avail /home/ext_csv | tail -n 1 | tr -d ' ')"
archive_bytes="$(du -sb "$PREDECESSOR_ARCHIVE" | cut -f1)"
run_bytes="$(du -sb "$RUN_ROOT" | cut -f1)"
checkpoint_bytes="$(du -sb "$PREDECESSOR_ARCHIVE/seed40/checkpoints/update_2048" | cut -f1)"
# Maximum route is stage A plus stage B: 24 retained checkpoints. Add two
# checkpoints for predecessor/successor overlap and 512 MiB for evaluations,
# logs, summaries, and serialization overhead. The safety reserve is the
# larger of 8 GiB and 20% of the forecast remaining writes.
retained_checkpoints=24
overlap_checkpoints=2
artifact_allowance_bytes=$((512 * 1024 * 1024))
projected_remaining_bytes=$((checkpoint_bytes * (retained_checkpoints + overlap_checkpoints) + artifact_allowance_bytes))
twenty_percent_bytes=$(((projected_remaining_bytes + 4) / 5))
minimum_reserve_bytes=$((8 * 1024 * 1024 * 1024))
if (( twenty_percent_bytes > minimum_reserve_bytes )); then
  safety_reserve_bytes="$twenty_percent_bytes"
else
  safety_reserve_bytes="$minimum_reserve_bytes"
fi
required_available_bytes=$((projected_remaining_bytes + safety_reserve_bytes))
printf '[mine-curriculum] capacity target_available_bytes=%s home_available_bytes=%s archive_bytes=%s run_bytes=%s checkpoint_bytes=%s retained_checkpoints=%s overlap_checkpoints=%s artifact_allowance_bytes=%s projected_remaining_bytes=%s safety_reserve_bytes=%s required_available_bytes=%s chosen_filesystem=/raid/ext_csv\n' \
  "$target_available_bytes" "$home_available_bytes" "$archive_bytes" "$run_bytes" \
  "$checkpoint_bytes" "$retained_checkpoints" "$overlap_checkpoints" \
  "$artifact_allowance_bytes" "$projected_remaining_bytes" "$safety_reserve_bytes" \
  "$required_available_bytes"
if (( target_available_bytes < required_available_bytes )); then
  printf '[mine-curriculum] insufficient RAID capacity: available=%s required=%s\n' \
    "$target_available_bytes" "$required_available_bytes" >&2
  exit 75
fi

if [[ ! -f "$PREDECESSOR_ARCHIVE/gate_result.json" ]]; then
  printf '[mine-curriculum] predecessor evidence archive is missing\n' >&2
  exit 75
fi
"$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); ok=(d.get("schema_version") == "hackrl_mine_expedition_fixed_gate_result_v1" and d.get("gate_id") == "mine_expedition_fixed_learnability_v1" and d.get("execution_complete") is True and d.get("status") == "fail" and not d.get("errors")); raise SystemExit(0 if ok else 1)' "$PREDECESSOR_ARCHIVE/gate_result.json"
for seed in 40 41 42; do
  "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("execution_code_sha") == sys.argv[2] else 1)' \
    "$PREDECESSOR_ARCHIVE/seed$seed/run_manifest.json" \
    b71c185384b0ab52bf5de4940125c576fee5e607
done

"$PYTHON" -c 'import hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]); manifest=json.loads((root / "docs/manifests/mine_expedition_fixed_curriculum_diagnostic_v1.json").read_text()); bad=[name for name, expected in manifest["authorized_source_sha256"].items() if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected]; print("[mine-curriculum] authorized source hashes verified" if not bad else "[mine-curriculum] authorized source hash mismatch: " + ", ".join(bad)); raise SystemExit(1 if bad else 0)' "$ROOT"

JAX_PLATFORMS=cpu "$PYTHON" -m pytest -q \
  tests/test_mine_expedition.py \
  tests/test_mine_expedition_env.py \
  tests/test_mine_expedition_gate.py \
  tests/test_mine_expedition_ppo.py \
  tests/test_mine_expedition_curriculum_diagnostic.py

# The server watcher releases its idle GPU allocation before entering this
# queue. Re-run the stage-transfer contract on that clean GPU before the first
# checkpoint-producing job starts.
HACKRL_DEVICE=cuda "$PYTHON" -m pytest -q \
  tests/test_mine_expedition_ppo.py::test_stage_transfer_preserves_learner_and_resets_stage_counters

run_seed() {
  local phase_dir="$1"
  local seed="$2"
  local training_start="$3"
  local updates="$4"
  local checkpoints="$5"
  local init_checkpoint="${6:-}"
  local destination="$RUN_ROOT/$phase_dir/seed$seed"
  printf '[mine-curriculum] phase=%s seed=%s start=%s training_start=%s\n' \
    "$phase_dir" "$seed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$training_start"
  local args=(
    --seed "$seed"
    --num-updates "$updates"
    --training-start "$training_start"
    --checkpoint-updates "$checkpoints"
    --log-dir "$destination"
  )
  if [[ -n "$init_checkpoint" ]]; then
    args+=(--init-checkpoint "$init_checkpoint")
  fi
  "$PYTHON" "$ROOT/scripts/run_mine_expedition_fixed_gate.py" "${args[@]}"
}

for seed in 40 41 42; do
  run_seed stage_a "$seed" craft_ready 512 0,128,512
done
"$PYTHON" "$ROOT/scripts/summarize_mine_expedition_fixed_curriculum_diagnostic.py" \
  --run-root "$RUN_ROOT" \
  --phase stage_a_craft_ready \
  --output "$RUN_ROOT/stage_a_result.json"

stage_a_status="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "$RUN_ROOT/stage_a_result.json")"
if [[ "$stage_a_status" == advance ]]; then
  printf '[mine-curriculum] route=stage_b_natural_late\n'
  for seed in 40 41 42; do
    run_seed stage_b "$seed" natural_late 2048 0,128,512,1024,2048 \
      "$RUN_ROOT/stage_a/seed$seed/checkpoints/update_512"
  done
  "$PYTHON" "$ROOT/scripts/summarize_mine_expedition_fixed_curriculum_diagnostic.py" \
    --run-root "$RUN_ROOT" \
    --phase stage_b_natural_late \
    --output "$result"
elif [[ "$stage_a_status" == fallback ]]; then
  printf '[mine-curriculum] route=fallback_target_ready\n'
  for seed in 40 41 42; do
    run_seed fallback "$seed" target_ready 512 0,128,512
  done
  "$PYTHON" "$ROOT/scripts/summarize_mine_expedition_fixed_curriculum_diagnostic.py" \
    --run-root "$RUN_ROOT" \
    --phase fallback_target_ready \
    --output "$result"
else
  printf '[mine-curriculum] invalid completed stage-A status: %s\n' "$stage_a_status" >&2
  exit 75
fi

printf '[mine-curriculum] complete=%s result=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$result"
