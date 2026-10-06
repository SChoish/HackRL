#!/usr/bin/env bash
# One bounded 20-job S-policy pretrained-delivery-only teacher experiment.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
RUN_ROOT="${HACKRL_DELIVERY_ONLY_TEACHER_ROOT:-/raid/ext_csv/HackRL/runs/pack_restore_delivery_only_teacher_v1}"
MANIFEST="$ROOT/docs/manifests/pack_restore_delivery_only_teacher_v1.json"

mkdir -p "$RUN_ROOT"
resolved_run_root="$(realpath "$RUN_ROOT")"
if [[ "$resolved_run_root" != /raid/ext_csv/HackRL/runs/* ]]; then
  printf '[delivery-only] refusing non-RAID run root: %s\n' "$resolved_run_root" >&2
  exit 75
fi
exec 9>"$RUN_ROOT/.queue.lock"
if ! flock -n 9; then
  printf '[delivery-only] another queue owns %s\n' "$RUN_ROOT" >&2
  exit 75
fi

authorized_execution_sha="${HACKRL_AUTHORIZED_EXECUTION_SHA:-}"
authorized_manifest_sha256="${HACKRL_AUTHORIZED_MANIFEST_SHA256:-}"
if [[ -z "$authorized_execution_sha" || -z "$authorized_manifest_sha256" ]]; then
  printf '[delivery-only] external execution SHA and manifest digest are required\n' >&2
  exit 75
fi
current_sha="$(git -C "$ROOT" rev-parse HEAD)"
actual_manifest_sha256="$("$PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$MANIFEST")"
if [[ "$current_sha" != "$authorized_execution_sha" ]]; then
  printf '[delivery-only] execution SHA mismatch actual=%s authorized=%s\n' \
    "$current_sha" "$authorized_execution_sha" >&2
  exit 75
fi
if [[ "$actual_manifest_sha256" != "$authorized_manifest_sha256" ]]; then
  printf '[delivery-only] manifest digest mismatch actual=%s authorized=%s\n' \
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
export PYTHONPATH="$ROOT/src:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"
printf '[delivery-only] start=%s run_root=%s execution_sha=%s manifest_sha256=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_ROOT" "$current_sha" "$actual_manifest_sha256"

"$PYTHON" -c 'import hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]); manifest=json.loads(pathlib.Path(sys.argv[2]).read_text()); authorized=manifest.get("authorized_source_sha256", {}); bad=[name for name, expected in authorized.items() if not (root / name).is_file() or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected]; print(f"[delivery-only] authorized source hashes verified count={len(authorized)}" if authorized and not bad else "[delivery-only] authorized source hash failure: " + ", ".join(bad or ["empty manifest"])); raise SystemExit(1 if bad or not authorized else 0)' "$ROOT" "$MANIFEST"
"$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); raise SystemExit(0 if m.get("status") == "implementation_validated_execution_authorized" else 1)' "$MANIFEST"

JAX_PLATFORMS=cpu "$PYTHON" -m pytest -q \
  tests/test_dual_leo.py \
  tests/test_pack_restore_scale_bc.py \
  tests/test_pack_restore_delivery_only_teacher.py

JAX_PLATFORMS=cpu "$PYTHON" scripts/run_pack_restore_delivery_only_teacher.py \
  --validate-contract

preflight_audit="$RUN_ROOT/preflight_evaluation_audit.json"
HACKRL_DEVICE=cuda "$PYTHON" scripts/audit_pack_restore_teacher_evaluation.py \
  --output "$preflight_audit"

HACKRL_DEVICE=cuda "$PYTHON" scripts/run_pack_restore_delivery_only_teacher.py \
  --smoke

HACKRL_DEVICE=cuda "$PYTHON" scripts/run_pack_restore_delivery_only_teacher.py \
  --log-dir "$RUN_ROOT" \
  --worker "delivery-only-worker"

result="$RUN_ROOT/result.json"
JAX_PLATFORMS=cpu "$PYTHON" scripts/summarize_pack_restore_delivery_only_teacher.py \
  --run-root "$RUN_ROOT" \
  --expected-execution-sha "$authorized_execution_sha" \
  --expected-manifest-sha256 "$authorized_manifest_sha256" \
  --preflight-audit "$preflight_audit" \
  --output "$result"

printf '[delivery-only] complete=%s result=%s preflight_audit=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$result" "$preflight_audit"
