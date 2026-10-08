#!/usr/bin/env python3
"""No-training postmortem for the stopped goal-conditioned Double DQN branch."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import numpy as np

from hackrl.pack_restore import (
    GOAL_IDS as PACK_GOAL_IDS,
    PackRestoreAction,
    PackRestorePhase,
    PackRestoreSplit,
    PackRestoreStart,
    PackRestoreVariant,
    make_pack_restore_state,
    pack_restore_goal_vector,
    pack_restore_step,
)
from hackrl.pack_restore_gc import DELIVER_3_GOAL_INDEX as PACK_DELIVER_3
from hackrl.tick_claim import (
    GOAL_IDS as TICK_GOAL_IDS,
    GROWTH_WAIT_TICKS,
    TickClaimAction,
    TickClaimPhase,
    TickClaimSplit,
    TickClaimStart,
    TickClaimVariant,
    make_tick_claim_state,
    observe_tick_claim,
    tick_claim_goal_vector,
    tick_claim_step,
    transform_direction,
)
from hackrl.tick_claim_gc import DELIVER_3_GOAL_INDEX as TICK_DELIVER_3
from run_gc_double_dqn import (
    ENVS,
    _checkpoint_fingerprint,
    _configure_development_profile,
    _environment_config,
    _load_from_checkpoint,
)


RUNS = {
    "baseline_replay65k": {
        "profile": "baseline",
        "root": Path(
            "/raid/ext_csv/HackRL/runs/gc_double_dqn_development_v1"
        ),
    },
    "replay262k": {
        "profile": "replay262k",
        "root": Path(
            "/raid/ext_csv/HackRL/runs/"
            "gc_double_dqn_development_replay262k_v1"
        ),
    },
}
DEFAULT_OUTPUT = Path(
    "/raid/ext_csv/HackRL/runs/gc_double_dqn_postmortem_v1/result.json"
)
TEACHER_RESULT = Path(
    "/raid/ext_csv/HackRL/runs/dual_teacher_greedy_two_defects_v1/result.json"
)
SOURCE_GROWTH_PERIOD = 8


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value, *, force=False):
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"output already exists; pass --force to replace it: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tick_probes():
    probes = []
    step = jax.jit(
        lambda state, action: tick_claim_step(
            state, action, TickClaimVariant.FIXED
        )
    )
    labels = {
        0: "harvest_1",
        1 + GROWTH_WAIT_TICKS: "harvest_2",
        2 + 2 * GROWTH_WAIT_TICKS: "harvest_3",
        3 + 2 * GROWTH_WAIT_TICKS: "approach_delivery",
        4 + 2 * GROWTH_WAIT_TICKS: "deliver_3",
    }
    for layout in range(16):
        state = make_tick_claim_state(
            layout,
            int(TickClaimPhase.RIPE),
            split=TickClaimSplit.VALIDATION,
            start=TickClaimStart.PATH_CHECK,
        )
        actions = (
            (int(TickClaimAction.DO),)
            + (int(TickClaimAction.NOOP),) * GROWTH_WAIT_TICKS
            + (int(TickClaimAction.DO),)
            + (int(TickClaimAction.NOOP),) * GROWTH_WAIT_TICKS
            + (
                int(TickClaimAction.DO),
                int(transform_direction(int(TickClaimAction.DOWN), layout)),
                int(TickClaimAction.DELIVER),
            )
        )
        for index, action in enumerate(actions):
            if index in labels:
                probes.append(
                    {
                        "layout": layout,
                        "phase": "ripe",
                        "probe": labels[index],
                        "expected_action": action,
                        "state": state,
                    }
                )
            state = step(state, jnp.asarray(action, dtype=jnp.int32))
        if not bool(
            tick_claim_goal_vector(observe_tick_claim(state))[TICK_DELIVER_3]
        ):
            raise RuntimeError(
                f"TICK fixed trace did not deliver three items: layout={layout}"
            )
    return probes


def _pack_actions(layout, phase):
    opening = [int(PackRestoreAction.NOOP)] * SOURCE_GROWTH_PERIOD
    if phase is PackRestorePhase.LOADED:
        opening[0] = int(PackRestoreAction.WITHDRAW_ONE)
    opening[1] = int(
        transform_direction(int(PackRestoreAction.RIGHT), layout)
    )
    return tuple(
        opening
        + [int(PackRestoreAction.DO)]
        + [int(PackRestoreAction.NOOP)] * SOURCE_GROWTH_PERIOD
        + [
            int(PackRestoreAction.DO),
            int(transform_direction(int(PackRestoreAction.DOWN), layout)),
            int(PackRestoreAction.DELIVER),
        ]
    )


def _pack_probes():
    probes = []
    step = jax.jit(
        lambda state, action: pack_restore_step(
            state, action, PackRestoreVariant.FIXED
        )
    )
    for layout in range(16):
        for phase in (PackRestorePhase.EMPTY, PackRestorePhase.LOADED):
            state = make_pack_restore_state(
                layout,
                int(phase),
                split=PackRestoreSplit.VALIDATION,
                start=PackRestoreStart.PATH_CHECK,
                source_growth_period=SOURCE_GROWTH_PERIOD,
            )
            labels = {
                SOURCE_GROWTH_PERIOD: "harvest_1",
                1 + 2 * SOURCE_GROWTH_PERIOD: "harvest_2",
                2 + 2 * SOURCE_GROWTH_PERIOD: "approach_delivery",
                3 + 2 * SOURCE_GROWTH_PERIOD: "deliver_3",
            }
            if phase is PackRestorePhase.LOADED:
                labels[0] = "withdraw_storage"
            actions = _pack_actions(layout, phase)
            for index, action in enumerate(actions):
                if index in labels:
                    probes.append(
                        {
                            "layout": layout,
                            "phase": phase.name.lower(),
                            "probe": labels[index],
                            "expected_action": action,
                            "state": state,
                        }
                    )
                state = step(state, jnp.asarray(action, dtype=jnp.int32))
            if not bool(pack_restore_goal_vector(state)[PACK_DELIVER_3]):
                raise RuntimeError(
                    "PACK fixed trace did not deliver three items: "
                    f"layout={layout}, phase={phase.name.lower()}"
                )
    return probes


def _goal_ids(env):
    return TICK_GOAL_IDS if env == "tick" else PACK_GOAL_IDS


def _action_names(env):
    action_type = TickClaimAction if env == "tick" else PackRestoreAction
    return {int(action): action.name for action in action_type}


def _summarize_replay(env, replay):
    size = int(replay.size)
    valid = np.asarray(replay.valid)[:size]
    goals = np.asarray(replay.goal_index)[:size]
    rewards = np.asarray(replay.reward)[:size]
    actions = np.asarray(replay.action)[:size]
    names = _action_names(env)
    if not (
        np.all((goals[valid] >= 0) & (goals[valid] < len(_goal_ids(env))))
        and np.all(np.isin(actions[valid], tuple(names)))
    ):
        raise ValueError("valid replay entries contain an unknown goal or action")
    rows = []
    for index, goal in enumerate(_goal_ids(env)):
        selected = valid & (goals == index)
        rewarded = selected & (rewards > 0)
        unique, counts = np.unique(actions[rewarded], return_counts=True)
        rows.append(
            {
                "goal_index": index,
                "goal": goal,
                "valid_entries": int(np.sum(selected)),
                "rewarded_entries": int(np.sum(rewarded)),
                "rewarded_action_counts": {
                    names[int(action)]: int(count)
                    for action, count in zip(unique, counts)
                },
            }
        )
    valid_entries = int(np.sum(valid))
    if sum(row["valid_entries"] for row in rows) != valid_entries:
        raise RuntimeError("per-goal replay counts do not reconcile")
    delivery_rows = [
        row for row in rows if row["goal"].startswith("delivery/")
    ]
    if not delivery_rows:
        raise RuntimeError("no delivery goals are defined")
    return {
        "replay_size": size,
        "valid_entries": valid_entries,
        "goals": rows,
        "delivery_goal_rows": delivery_rows,
    }


def _field_coverage(rows, key):
    present = sum(key in row for row in rows)
    if present == len(rows):
        return "all"
    if present:
        return "some"
    return "none"


def _summarize_update_log(path):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"update log is empty: {path}")
    return {
        "updates": len(rows),
        "goal_completions_all_goals": int(
            sum(row["goal_completions"] for row in rows)
        ),
        "sampled_valid_transitions_all_goals": int(
            sum(row["sampled_valid_transitions"] for row in rows)
        ),
        "per_goal_success_counts_coverage": _field_coverage(
            rows, "successes_by_goal"
        ),
        "sampled_goal_indices_coverage": _field_coverage(
            rows, "sampled_goal_indices"
        ),
    }


def _q_probe_rows(env, network, train_state):
    probes = _tick_probes() if env == "tick" else _pack_probes()
    states = jax.tree.map(
        lambda *values: jnp.stack(values),
        *(item["state"] for item in probes),
    )
    delivery = TICK_DELIVER_3 if env == "tick" else PACK_DELIVER_3
    goals = jnp.full((len(probes),), delivery, dtype=jnp.int32)
    maps, numeric, _ = ENVS[env]["inputs"](states, goals)
    q_all = network.apply(
        {"params": train_state.params, "batch_stats": train_state.batch_stats},
        maps,
        numeric,
        train=False,
    )
    q_values = np.asarray(q_all[:, delivery, :], dtype=np.float64)
    if not np.all(np.isfinite(q_values)):
        raise RuntimeError(f"{env} Q probe produced a non-finite value")
    names = _action_names(env)
    rows = []
    for item, values in zip(probes, q_values):
        expected = int(item["expected_action"])
        top = int(np.argmax(values))
        order = np.argsort(-values, kind="stable")
        rank = int(np.flatnonzero(order == expected)[0]) + 1
        rows.append(
            {
                "layout": item["layout"],
                "phase": item["phase"],
                "probe": item["probe"],
                "expected_action": names[expected],
                "top_action": names[top],
                "top1_match": top == expected,
                "expected_action_rank": rank,
                "expected_q": float(values[expected]),
                "top_q": float(values[top]),
                "top_minus_expected_q": float(values[top] - values[expected]),
                "top3": [names[int(action)] for action in order[:3]],
            }
        )
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["phase"], row["probe"])].append(row)
    summary = []
    for (phase, probe), items in sorted(grouped.items()):
        summary.append(
            {
                "phase": phase,
                "probe": probe,
                "states": len(items),
                "top1_matches": sum(item["top1_match"] for item in items),
                "top1_match_rate": float(
                    np.mean([item["top1_match"] for item in items])
                ),
                "median_expected_action_rank": float(
                    np.median(
                        [item["expected_action_rank"] for item in items]
                    )
                ),
            }
        )
    return {
        "goal": "delivery/count_ge_3",
        "kernel": "fixed",
        "split": "validation",
        "start": "path_check",
        "probe_states": len(rows),
        "summary": summary,
        "rows": rows,
    }


def _teacher_summary():
    result = _read_json(TEACHER_RESULT)
    if (
        result.get("execution_complete") is not True
        or result.get("checkpoints_immutable") is not True
        or result.get("evaluation", {}).get("checkpoint_count") != 30
    ):
        raise RuntimeError("Dual teacher result is incomplete or mutable")
    rows = []
    for row in result["natural_reset_means"]:
        rows.append(
            {
                key: row[key]
                for key in (
                    "env",
                    "stage",
                    "kernel",
                    "mean_success_rate",
                    "mean_violation_delivery_rate",
                    "n_learner_seeds",
                )
            }
        )
    return {
        "already_complete": True,
        "training_or_optimizer_updates": 0,
        "checkpoint_count": result["evaluation"]["checkpoint_count"],
        "artifact": str(TEACHER_RESULT),
        "artifact_sha256": _sha256(TEACHER_RESULT),
        "natural_reset_means": rows,
    }


def _normal_gate(run):
    path = run["root"] / "development_gate.json"
    gate = _read_json(path)
    if (
        gate.get("passed") is not False
        or gate.get("selection_used_only_fixed_normal_success") is not True
        or {row.get("env") for row in gate.get("rows", [])}
        != {"tick", "pack"}
    ):
        raise RuntimeError(f"unexpected development gate: {path}")
    return {"artifact": str(path), "artifact_sha256": _sha256(path), **gate}


def _analyze_runs():
    analyses = []
    gates = {}
    for run_name, run in RUNS.items():
        gates[run_name] = _normal_gate(run)
        _configure_development_profile(run["profile"])
        for env in ("tick", "pack"):
            cell = run["root"] / env / "pretrain" / "seed100"
            summary = _read_json(cell / "summary.json")
            checkpoint = Path(summary["checkpoint"])
            if not checkpoint.is_absolute():
                checkpoint = (cell / checkpoint).resolve()
            recorded_fingerprint = summary["checkpoint_fingerprint"]
            observed_fingerprint = _checkpoint_fingerprint(checkpoint)
            if observed_fingerprint != recorded_fingerprint:
                raise RuntimeError(
                    f"checkpoint fingerprint mismatch: {checkpoint}"
                )
            config = _environment_config(
                env,
                seed=100,
                goal_mode="workshop12",
                variant="fixed",
                updates=512,
            )
            network, runner, train_state, replay = _load_from_checkpoint(
                env, config, checkpoint
            )
            analyses.append(
                {
                    "run": run_name,
                    "env": env,
                    "checkpoint": str(checkpoint),
                    "checkpoint_fingerprint": observed_fingerprint,
                    "training_log": _summarize_update_log(
                        cell / "updates.jsonl"
                    ),
                    "final_replay_window": _summarize_replay(env, replay),
                    "normal_trace_q": _q_probe_rows(
                        env, network, train_state
                    ),
                }
            )
            del network, runner, train_state, replay
            jax.clear_caches()
            gc.collect()
    return analyses, gates


def analyze():
    try:
        analyses, gates = _analyze_runs()
    finally:
        _configure_development_profile("baseline")
    source = Path(__file__).resolve()
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=source.parents[1],
        text=True,
    ).strip()
    return {
        "schema_version": "hackrl_gc_double_dqn_postmortem_v1",
        "execution_provenance": {
            "git_head": git_head,
            "source": str(source),
            "source_sha256": _sha256(source),
        },
        "execution_complete": True,
        "training_or_optimizer_updates": 0,
        "main_experiment_started": False,
        "normal_development_gates": gates,
        "analyses": analyses,
        "dual_teacher_greedy_execution": _teacher_summary(),
        "evidence_limits": [
            "Update logs contain total goal completions and total sampled valid transitions, but not their goal identities.",
            "A final replay window proves only what remained in that bounded window; it cannot recover overwritten earlier transitions.",
            "Uniform replay made valid retained entries eligible for sampling, but exact sampled indices were not logged.",
            "Constructive path-check probes test local Q rankings on designer-selected legal suffix states, not natural-start policy execution or generalization.",
            "The fixed kernel and existing fixture tests establish that each scripted trace reaches delivery; they do not make the probe set an independent learned-policy benchmark."
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output"
    )
    arguments = parser.parse_args()
    result = analyze()
    _write_json(arguments.output, result, force=arguments.force)
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "analyses": len(result["analyses"]),
                "training_or_optimizer_updates": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
