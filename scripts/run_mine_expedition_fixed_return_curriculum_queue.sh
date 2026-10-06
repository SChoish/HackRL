#!/usr/bin/env bash
# One bounded fixed-only cumulative return curriculum.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
RUN_ROOT="${HACKRL_MINE_RETURN_ROOT:-/raid/ext_csv/HackRL/runs/mine_expedition_fixed_return_curriculum_v1}"
MANIFEST="$ROOT/docs/manifests/mine_expedition_fixed_return_curriculum_v1.json"

mkdir -p "$RUN_ROOT"
resolved_run_root="$(realpath "$RUN_ROOT")"
if [[ "$resolved_run_root" != /raid/ext_csv/HackRL/runs/* ]]; then
  printf '[mine-return] refusing non-RAID run root: %s\n' "$resolved_run_root" >&2
  exit 75
fi
exec 9>"$RUN_ROOT/.queue.lock"
if ! flock -n 9; then
  printf '[mine-return] another worker owns %s\n' "$RUN_ROOT" >&2
  exit 75
fi

result="$RUN_ROOT/diagnostic_result.json"
current_sha="$(git -C "$ROOT" rev-parse HEAD)"
authorized_execution_sha="${HACKRL_AUTHORIZED_EXECUTION_SHA:-}"
authorized_manifest_sha256="${HACKRL_AUTHORIZED_MANIFEST_SHA256:-}"
if [[ -z "$authorized_execution_sha" || -z "$authorized_manifest_sha256" ]]; then
  printf '[mine-return] external execution SHA and manifest digest are required\n' >&2
  exit 75
fi
actual_manifest_sha256="$("$PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$MANIFEST")"
if [[ "$current_sha" != "$authorized_execution_sha" ]]; then
  printf '[mine-return] execution SHA mismatch: actual=%s authorized=%s\n' \
    "$current_sha" "$authorized_execution_sha" >&2
  exit 75
fi
if [[ "$actual_manifest_sha256" != "$authorized_manifest_sha256" ]]; then
  printf '[mine-return] manifest digest mismatch: actual=%s authorized=%s\n' \
    "$actual_manifest_sha256" "$authorized_manifest_sha256" >&2
  exit 75
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
printf '[mine-return] start=%s device=%s run_root=%s execution_sha=%s manifest_sha256=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$HACKRL_DEVICE" "$RUN_ROOT" \
  "$current_sha" "$actual_manifest_sha256"

target_available_bytes="$(df -B1 --output=avail "$resolved_run_root" | tail -n 1 | tr -d ' ')"
home_available_bytes="$(df -B1 --output=avail /home/ext_csv | tail -n 1 | tr -d ' ')"
run_bytes="$(du -sb "$RUN_ROOT" | cut -f1)"
read -r checkpoint_bytes retained_checkpoints overlap_checkpoints artifact_allowance_bytes < <(
  "$PYTHON" -c 'import json,sys; s=json.load(open(sys.argv[1]))["storage"]; print(s["measured_checkpoint_bytes"], s["retained_checkpoint_count"], s["active_overlap_checkpoint_count"], s["log_and_evaluation_allowance_bytes"])' "$MANIFEST"
)
projected_remaining_bytes=$((checkpoint_bytes * (retained_checkpoints + overlap_checkpoints) + artifact_allowance_bytes))
twenty_percent_bytes=$(((projected_remaining_bytes + 4) / 5))
minimum_reserve_bytes=$((8 * 1024 * 1024 * 1024))
if (( twenty_percent_bytes > minimum_reserve_bytes )); then
  safety_reserve_bytes="$twenty_percent_bytes"
else
  safety_reserve_bytes="$minimum_reserve_bytes"
fi
required_available_bytes=$((projected_remaining_bytes + safety_reserve_bytes))
printf '[mine-return] capacity target_available_bytes=%s home_available_bytes=%s run_bytes=%s checkpoint_bytes=%s retained_checkpoints=%s overlap_checkpoints=%s artifact_allowance_bytes=%s projected_remaining_bytes=%s safety_reserve_bytes=%s required_available_bytes=%s chosen_filesystem=/raid/ext_csv\n' \
  "$target_available_bytes" "$home_available_bytes" "$run_bytes" \
  "$checkpoint_bytes" "$retained_checkpoints" "$overlap_checkpoints" \
  "$artifact_allowance_bytes" "$projected_remaining_bytes" \
  "$safety_reserve_bytes" "$required_available_bytes"
if (( target_available_bytes < required_available_bytes )); then
  printf '[mine-return] insufficient RAID capacity: available=%s required=%s\n' \
    "$target_available_bytes" "$required_available_bytes" >&2
  exit 75
fi

"$PYTHON" -c 'import hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]); manifest=json.loads(pathlib.Path(sys.argv[2]).read_text()); authorized=manifest.get("authorized_source_sha256", {}); bad=[name for name, expected in authorized.items() if not (root / name).is_file() or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected]; print(f"[mine-return] authorized source hashes verified count={len(authorized)}" if authorized and not bad else "[mine-return] authorized source hash failure: " + ", ".join(bad or ["empty manifest"])); raise SystemExit(1 if bad or not authorized else 0)' "$ROOT" "$MANIFEST"
"$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); raise SystemExit(0 if m.get("status") == "implementation_validated_execution_authorized" else 1)' "$MANIFEST"

if [[ -f "$result" ]]; then
  completion_recheck="$RUN_ROOT/completion_recheck.json"
  if "$PYTHON" "$ROOT/scripts/summarize_mine_expedition_fixed_return_curriculum.py" \
      --run-root "$RUN_ROOT" --final \
      --expected-execution-sha "$authorized_execution_sha" \
      --expected-manifest-sha256 "$authorized_manifest_sha256" \
      --output "$completion_recheck" \
      && "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); ok=(d.get("schema_version") == "hackrl_mine_expedition_fixed_return_curriculum_result_v1" and d.get("experiment_id") == "mine_expedition_fixed_return_curriculum_v1" and d.get("execution_complete") is True and not d.get("errors") and d.get("execution_code_shas") == [sys.argv[2]] and d.get("authorized_execution_sha") == sys.argv[2] and d.get("authorized_manifest_sha256") == sys.argv[3] and d.get("fixed_natural_gate_passed") is d.get("mutant_training_authorized")); raise SystemExit(0 if ok else 1)' "$completion_recheck" "$authorized_execution_sha" "$authorized_manifest_sha256"; then
    mv "$completion_recheck" "$result"
    printf '[mine-return] already complete and revalidated result=%s\n' "$result"
    exit 0
  fi
fi

JAX_PLATFORMS=cpu "$PYTHON" -m pytest -q \
  tests/test_mine_expedition.py \
  tests/test_mine_expedition_env.py \
  tests/test_mine_expedition_gate.py \
  tests/test_mine_expedition_ppo.py \
  tests/test_mine_expedition_return_curriculum.py
HACKRL_DEVICE=cuda "$PYTHON" -m pytest -q \
  tests/test_mine_expedition_return_curriculum.py::test_return_near_one_update_runs_and_keeps_fixed_contract

run_seed() {
  local phase_name="$1"
  local phase_dir="$2"
  local seed="$3"
  local training_start="$4"
  local updates="$5"
  local checkpoints="$6"
  local init_checkpoint="${7:-}"
  local destination="$RUN_ROOT/$phase_dir/seed$seed"
  printf '[mine-return] phase=%s seed=%s start=%s training_start=%s\n' \
    "$phase_name" "$seed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$training_start"
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

mkdir -p "$RUN_ROOT/phase_results"
mapfile -t phase_names < <(
  "$PYTHON" -c 'import json,sys; print(*json.load(open(sys.argv[1]))["phase_order"], sep="\n")' "$MANIFEST"
)
mapfile -t seeds < <(
  "$PYTHON" -c 'import json,sys; print(*json.load(open(sys.argv[1]))["fixed_contract"]["seeds"], sep="\n")' "$MANIFEST"
)
previous_dir=""
previous_updates=""
for phase_name in "${phase_names[@]}"; do
  read -r phase_dir training_start updates checkpoints < <(
    "$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1]))["phases"][sys.argv[2]]; print(p["directory"], p["training_start"], p["updates"], ",".join(map(str,p["checkpoint_updates"])))' "$MANIFEST" "$phase_name"
  )
  for seed in "${seeds[@]}"; do
    init_checkpoint=""
    if [[ -n "$previous_dir" ]]; then
      init_checkpoint="$RUN_ROOT/$previous_dir/seed$seed/checkpoints/update_$previous_updates"
    fi
    run_seed "$phase_name" "$phase_dir" "$seed" "$training_start" \
      "$updates" "$checkpoints" "$init_checkpoint"
  done
  phase_result="$RUN_ROOT/phase_results/$phase_name.json"
  "$PYTHON" "$ROOT/scripts/summarize_mine_expedition_fixed_return_curriculum.py" \
    --run-root "$RUN_ROOT" \
    --phase "$phase_name" \
    --expected-execution-sha "$authorized_execution_sha" \
    --expected-manifest-sha256 "$authorized_manifest_sha256" \
    --output "$phase_result"
  phase_passed="$("$PYTHON" -c 'import json,sys; print("true" if json.load(open(sys.argv[1])).get("phase_passed") is True else "false")' "$phase_result")"
  if [[ "$phase_passed" != true ]]; then
    printf '[mine-return] predeclared_stop phase=%s\n' "$phase_name"
    break
  fi
  previous_dir="$phase_dir"
  previous_updates="$updates"
done

"$PYTHON" "$ROOT/scripts/summarize_mine_expedition_fixed_return_curriculum.py" \
  --run-root "$RUN_ROOT" \
  --final \
  --expected-execution-sha "$authorized_execution_sha" \
  --expected-manifest-sha256 "$authorized_manifest_sha256" \
  --output "$result"
"$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); ok=(d.get("execution_complete") is True and not d.get("errors") and d.get("execution_code_shas") == [sys.argv[2]] and d.get("authorized_execution_sha") == sys.argv[2] and d.get("authorized_manifest_sha256") == sys.argv[3] and d.get("fixed_natural_gate_passed") is d.get("mutant_training_authorized")); raise SystemExit(0 if ok else 1)' "$result" "$authorized_execution_sha" "$authorized_manifest_sha256"
printf '[mine-return] complete=%s result=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$result"
