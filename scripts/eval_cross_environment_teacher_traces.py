#!/usr/bin/env python3
"""Frozen full-trace evaluator for cross-environment teacher matrix v1."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

if os.environ.get("HACKRL_DEVICE", "cpu") != "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

import run_dual_leo_compare as engine
from hackrl.cross_environment_teacher_matrix import DEFAULT_MATRIX, load_matrix, sha256_file
from hackrl.cross_environment_teacher_trace import (
    ACTION_ENUMS,
    DISCOUNT,
    HORIZON,
    TRACE_SCHEMA_VERSION,
    actual_step,
    audit_actual_step,
    build_panel_manifest,
    content_sha256,
    episode_exploited,
    goal_complete,
    initial_episode_oracle,
    legal_trigger_actions,
    legacy_opportunity_reached,
    observe,
    public_record,
    start_states,
    world_done,
)
from hackrl.dual_leo import load_dual_checkpoint
from run_cross_environment_teacher_matrix import _register_craft


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "docs/manifests/cross_environment_teacher_trace_panel_v1.json"
DEFAULT_OUTPUT = Path("/raid/ext_csv/HackRL/evaluations/cross_environment_teacher_matrix_v1")
ENV_TO_ENGINE = {"tick_claim": "tick", "pack_restore": "pack", "craft_remain": "craft"}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_fingerprint(directory: Path) -> dict:
    names = ["state.msgpack", "config.json", "metadata.json"]
    if (directory / "arm.json").is_file():
        names.append("arm.json")
    files = {name: sha256_file(directory / name) for name in names}
    combined = hashlib.sha256()
    for name, digest in sorted(files.items()):
        combined.update(name.encode() + b"\0" + digest.encode() + b"\n")
    return {"sha256": combined.hexdigest(), "files": files}


def _load_actor(environment: str, checkpoint: Path):
    _register_craft()
    spec = engine.ENVS[ENV_TO_ENGINE[environment]]
    payload = json.loads((checkpoint / "config.json").read_text())
    config = spec["from_payload"](payload)
    network, template = spec["initialize"](config)
    _, leo_template, _ = engine._start_teacher(spec, config, template)
    runner, _ = load_dual_checkpoint(checkpoint, template, leo_template)
    return spec, network, runner.train_state.params


def _actor_action(spec, network, params, state, goal_index, view, key):
    batched = jax.tree.map(lambda value: value[None], state)
    goals = jnp.asarray([goal_index], dtype=jnp.int32)
    policy, _ = network.apply(params, *spec["inputs"](batched, goals))
    key, draw_key = jax.random.split(key)
    if view == "mode":
        action = int(jnp.argmax(policy.logits[0]))
    else:
        action = int(policy.sample(seed=draw_key)[0])
    return action, key, np.asarray(jax.device_get(policy.logits[0])).tolist()


def _validate_panel(path: Path) -> dict:
    frozen = json.loads(path.read_text())
    current = build_panel_manifest()
    if frozen != current:
        raise RuntimeError("frozen evaluation panel differs from current generators")
    return frozen


def evaluate(
    *,
    environment: str,
    checkpoint: Path,
    checkpoint_origin: str,
    learner_seed: int,
    variant: str,
    view: str,
    family: str,
    matrix_path: Path,
    panel_path: Path,
    output_root: Path,
    max_starts: int | None = None,
    horizon: int = HORIZON,
    scientific: bool = True,
) -> dict:
    matrix = load_matrix(matrix_path)
    panel = _validate_panel(panel_path)
    mapping = matrix["shared_learning_contract"]["task_goal_mapping"][environment]
    goal_index = int(mapping["goal_index"])
    repeats = 1 if view == "mode" else 4
    starts = start_states(environment, family)
    if max_starts is not None:
        starts = starts[: int(max_starts)]
    spec, network, params = _load_actor(environment, checkpoint)
    fingerprint = _checkpoint_fingerprint(checkpoint)
    stem = (
        output_root
        / checkpoint_origin
        / environment
        / fingerprint["sha256"]
        / variant
        / view
        / family
    )
    trace_path = stem / "episodes.jsonl.gz"
    summary_path = stem / "summary.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    episode_rows = []
    total_steps = 0
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        header = {
            "record_type": "header",
            "schema_version": TRACE_SCHEMA_VERSION,
            "scientific_performance_result": bool(scientific),
            "checkpoint_origin": checkpoint_origin,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_fingerprint": fingerprint,
            "environment": environment,
            "kernel": variant,
            "policy_view": view,
            "family": family,
            "learner_seed": int(learner_seed),
            "goal_id": mapping["goal_id"],
            "goal_index": goal_index,
            "matrix_sha256": sha256_file(matrix_path),
            "panel_sha256": sha256_file(panel_path),
            "panel_family_sha256": panel["families"][environment][family]["ordered_panel_sha256"],
            "oracle_isolation": "oracle runs after actor action selection and is not passed to actor, teacher, or reward",
        }
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for start in starts:
            for repeat in range(repeats):
                state = start["state"]
                seed = 20000 + 1000 * int(learner_seed) + 4 * int(start["state_index"]) + repeat
                key = jax.random.PRNGKey(seed)
                oracle = initial_episode_oracle(environment)
                success = False
                termination = "horizon"
                length = 0
                for step_index in range(int(horizon)):
                    observation = observe(environment, state)
                    action, key, logits = _actor_action(
                        spec, network, params, state, goal_index, view, key
                    )
                    legal = legal_trigger_actions(environment, state)
                    legacy = legacy_opportunity_reached(environment, state)
                    before = state
                    after, auxiliary = actual_step(environment, before, action, variant)
                    oracle, event = audit_actual_step(
                        environment, before, action, after, auxiliary, variant, oracle
                    )
                    length += 1
                    success = goal_complete(environment, after, goal_index)
                    ended = world_done(environment, after)
                    termination = "goal_complete" if success else ("world_horizon" if ended else "active")
                    row = {
                        "record_type": "step",
                        "episode_id": f"{start['id']}:repeat{repeat}",
                        "start_id": start["id"],
                        "state_index": start["state_index"],
                        "layout": start["layout"],
                        "phase": start["phase"],
                        "repeat": repeat,
                        "rng_seed": seed,
                        "timestep": step_index,
                        "state_before": public_record(before),
                        "public_observation": public_record(observation),
                        "action": action,
                        "action_name": ACTION_ENUMS[environment](action).name,
                        "policy_logits": logits,
                        "legal_trigger_action_ids_exhaustive": legal,
                        "legal_opportunity_any": bool(legal),
                        "selected_legal_trigger_action": action in legal,
                        "legacy_opportunity_reached_narrow": legacy,
                        "state_after": public_record(after),
                        "termination_after_step": termination,
                        "success_after_step": success,
                        **event,
                    }
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    total_steps += 1
                    state = after
                    if success or ended:
                        break
                exploited = episode_exploited(environment, oracle)
                terminal = {
                    "record_type": "episode_terminal",
                    "episode_id": f"{start['id']}:repeat{repeat}",
                    "success": bool(success),
                    "length": length,
                    "discounted_return": float((DISCOUNT ** (length - 1)) if success and length else 0.0),
                    "termination_reason": termination if termination != "active" else "trace_horizon",
                    "actual_defect_triggered": bool(oracle["triggered"]),
                    "actual_exploitation": bool(exploited),
                    "defect_resource_delivered": int(oracle["defect_resource_delivered"]),
                    "final_state": public_record(state),
                }
                handle.write(json.dumps(terminal, sort_keys=True) + "\n")
                episode_rows.append(terminal)
    trace_bytes = trace_path.stat().st_size
    summary = {
        "schema_version": "hackrl_cross_environment_teacher_trace_summary_v1",
        "scientific_performance_result": bool(scientific),
        "checkpoint_origin": checkpoint_origin,
        "environment": environment,
        "kernel": variant,
        "policy_view": view,
        "family": family,
        "episodes": len(episode_rows),
        "steps": total_steps,
        "success_rate": float(np.mean([row["success"] for row in episode_rows])),
        "mean_discounted_return": float(np.mean([row["discounted_return"] for row in episode_rows])),
        "trigger_rate": float(np.mean([row["actual_defect_triggered"] for row in episode_rows])),
        "exploit_rate": float(np.mean([row["actual_exploitation"] for row in episode_rows])),
        "trace": str(trace_path),
        "trace_gzip_bytes": trace_bytes,
        "gzip_bytes_per_step": trace_bytes / max(total_steps, 1),
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--environment", choices=tuple(ENV_TO_ENGINE))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-origin", choices=("reused_result", "new_matrix_result"))
    parser.add_argument("--learner-seed", type=int)
    parser.add_argument("--variant", choices=("fixed", "mutant"))
    parser.add_argument("--view", choices=("mode", "sample"))
    parser.add_argument("--family")
    parser.add_argument("--max-starts", type=int)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--non-scientific-smoke", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()
    if args.print_plan:
        print(json.dumps(_validate_panel(args.panel), indent=2, sort_keys=True))
        return
    required = (args.environment, args.checkpoint, args.checkpoint_origin, args.learner_seed, args.variant, args.view, args.family)
    if any(value is None for value in required):
        raise SystemExit("evaluation requires environment, checkpoint provenance, seed, kernel, view, and family")
    result = evaluate(
        environment=args.environment,
        checkpoint=args.checkpoint,
        checkpoint_origin=args.checkpoint_origin,
        learner_seed=args.learner_seed,
        variant=args.variant,
        view=args.view,
        family=args.family,
        matrix_path=args.matrix,
        panel_path=args.panel,
        output_root=args.output_root,
        max_starts=args.max_starts,
        horizon=args.horizon,
        scientific=not args.non_scientific_smoke,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
