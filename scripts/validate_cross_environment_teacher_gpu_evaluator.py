#!/usr/bin/env python3
"""Compare durable CPU-reference and CUDA-batched trace smoke artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(
    "/raid/ext_csv/HackRL/evaluations/"
    "cross_environment_teacher_matrix_v1_gpu_validation"
)
MAX_POLICY_TOTAL_VARIATION = 1e-3

DEFAULT_OUTPUT = Path(
    "docs/manifests/cross_environment_teacher_gpu_evaluator_validation_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _trace(root: Path, backend: str, environment: str) -> Path:
    matches = list((root / backend).glob(
        f"**/{environment}/**/episodes.jsonl.gz"
    ))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {backend}/{environment} trace, found {len(matches)}"
        )
    return matches[0]


def compare(root: Path) -> dict:
    results = {}
    for environment in ("craft_remain", "tick_claim", "pack_restore"):
        cpu_path = _trace(root, "cpu_reference", environment)
        gpu_path = _trace(root, "gpu", environment)
        cpu = _rows(cpu_path)
        gpu = _rows(gpu_path)
        if len(cpu) != len(gpu):
            raise RuntimeError(f"{environment}: record count differs")
        cpu_header = dict(cpu[0])
        gpu_header = dict(gpu[0])
        gpu_header.pop("evaluation_backend", None)
        gpu_header.pop("policy_id", None)
        if cpu_header != gpu_header:
            raise RuntimeError(f"{environment}: header contract differs")
        max_logit_error = 0.0
        max_policy_total_variation = 0.0
        non_logit_exact = True
        actions_equal = True
        for left, right in zip(cpu[1:], gpu[1:], strict=True):
            left = dict(left)
            right = dict(right)
            if left.get("record_type") == "step":
                left_logits = np.asarray(left.pop("policy_logits"))
                right_logits = np.asarray(right.pop("policy_logits"))
                max_logit_error = max(
                    max_logit_error,
                    float(np.max(np.abs(left_logits - right_logits))),
                )
                left_probability = np.exp(left_logits - np.max(left_logits))
                left_probability /= np.sum(left_probability)
                right_probability = np.exp(right_logits - np.max(right_logits))
                right_probability /= np.sum(right_probability)
                max_policy_total_variation = max(
                    max_policy_total_variation,
                    float(0.5 * np.sum(np.abs(
                        left_probability - right_probability
                    ))),
                )
                actions_equal = actions_equal and left["action"] == right["action"]
            non_logit_exact = non_logit_exact and left == right
        cpu_trigger = any(
            row.get("record_type") == "episode_terminal"
            and row.get("actual_defect_triggered") is True
            for row in cpu
        )
        gpu_trigger = any(
            row.get("record_type") == "episode_terminal"
            and row.get("actual_defect_triggered") is True
            for row in gpu
        )
        cpu_replay = json.loads(
            cpu_path.with_name("independent_replay.json").read_text()
        )
        gpu_replay = json.loads(
            gpu_path.with_name("independent_replay.json").read_text()
        )
        passed = bool(
            non_logit_exact
            and actions_equal
            and max_policy_total_variation <= MAX_POLICY_TOTAL_VARIATION
            and cpu_trigger
            and gpu_trigger
            and cpu_replay["passed"]
            and gpu_replay["passed"]
        )
        results[environment] = {
            "passed": passed,
            "records": len(cpu),
            "steps": cpu_replay["steps_checked"],
            "actions_equal": actions_equal,
            "non_logit_records_exact": non_logit_exact,
            "max_abs_policy_logit_error": max_logit_error,
            "max_policy_total_variation": max_policy_total_variation,
            "max_policy_total_variation_threshold": MAX_POLICY_TOTAL_VARIATION,
            "actual_defect_trigger_observed": cpu_trigger and gpu_trigger,
            "cpu_independent_replay_passed": cpu_replay["passed"],
            "gpu_independent_replay_passed": gpu_replay["passed"],
            "cpu_trace": str(cpu_path),
            "cpu_trace_sha256": _sha256(cpu_path),
            "gpu_trace": str(gpu_path),
            "gpu_trace_sha256": _sha256(gpu_path),
        }
    return {
        "schema_version": (
            "hackrl_cross_environment_teacher_gpu_evaluator_validation_v1"
        ),
        "scientific_performance_result": False,
        "purpose": (
            "Implementation equivalence smoke only; not a policy performance result."
        ),
        "validation_root": str(root),
        "validated_source_sha256": {
            "scripts/eval_cross_environment_teacher_traces.py": _sha256(
                REPO_ROOT / "scripts/eval_cross_environment_teacher_traces.py"
            ),
            "scripts/eval_cross_environment_teacher_traces_gpu.py": _sha256(
                REPO_ROOT / "scripts/eval_cross_environment_teacher_traces_gpu.py"
            ),
        },
        "independence_scope": (
            "The CPU and GPU implementations share checkpoints and environment semantics. "
            "Their action/state trace paths differ, and each trace is checked by the same "
            "state-only replay verifier; this is not an externally authored oracle."
        ),
        "environments": results,
        "passed": all(row["passed"] for row in results.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = compare(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
