#!/usr/bin/env python3
"""Execute saved Dual teachers greedily on both controlled bug fixtures.

This is a read-only checkpoint evaluation. It asks whether the learned teacher
itself is an executable delivery policy; it does not train or mutate any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import distrax
import jax
import jax.numpy as jnp
import numpy as np

from hackrl.dual_leo import init_dual_leo_teacher, load_dual_checkpoint
from run_dual_leo_compare import DISCOUNT, ENVS


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY / "runs" / "dual_leo_compare_v1"
SEEDS = (20, 21, 22, 23, 24)
STAGES = ("pretrain", "fixed_adapted", "mutant_adapted")
KERNELS = ("fixed", "mutant")
FAMILIES = ("natural_reset", "common_setup")
MINIMUM_FREE_BYTES = 8 * 1024**3


class TeacherGreedyPolicy:
    def __init__(self, teacher_network):
        self.teacher_network = teacher_network

    def apply(self, parameters, maps, numeric, goal_one_hot):
        values = self.teacher_network.apply(
            {
                "params": parameters["params"],
                "batch_stats": parameters["batch_stats"],
            },
            maps,
            numeric,
            train=False,
        )
        goal_index = jnp.argmax(goal_one_hot, axis=-1)
        selected = jnp.take_along_axis(
            values, goal_index[:, None, None], axis=1
        )[:, 0, :]
        return distrax.Categorical(logits=selected), jnp.zeros(
            (maps.shape[0],), dtype=jnp.float32
        )


def checkpoint_for(env, seed, stage, source_root=SOURCE_ROOT):
    root = Path(source_root) / env / "dual"
    if stage == "pretrain":
        return root / "pretrain" / f"seed{seed}" / "checkpoints" / "update_512"
    if stage == "fixed_adapted":
        return root / "fixed" / f"seed{seed}" / "checkpoints" / "adapt_4096"
    if stage == "mutant_adapted":
        return root / "mutant" / f"seed{seed}" / "checkpoints" / "adapt_4096"
    raise ValueError(f"unknown teacher stage: {stage}")


def build_evaluations():
    return [
        {
            "env": env,
            "seed": int(seed),
            "stage": stage,
            "checkpoint": checkpoint_for(env, seed, stage),
        }
        for env in ("tick", "pack")
        for seed in SEEDS
        for stage in STAGES
    ]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _family_return(records, family):
    values = []
    for row in records:
        if row["family"] != family:
            continue
        length = int(row["length"])
        success = 1.0 if row["success"] else 0.0
        values.append(
            success * (DISCOUNT ** (length - 1)) if length >= 1 else 0.0
        )
    return float(np.mean(values))


def _evaluate_checkpoint(job):
    spec = ENVS[job["env"]]
    checkpoint = Path(job["checkpoint"])
    config = spec["from_payload"](_read(checkpoint / "config.json"))
    _, runner_template = spec["initialize"](config)
    inputs = spec["inputs"](
        runner_template.env_state, runner_template.current_goal
    )
    teacher_network, teacher_template, _ = init_dual_leo_teacher(
        config,
        inputs[0],
        inputs[1],
        spec["goals"],
        spec["actions"],
    )
    _, teacher = load_dual_checkpoint(
        checkpoint, runner_template, teacher_template
    )
    policy = TeacherGreedyPolicy(teacher_network)
    parameters = {
        "params": teacher.params,
        "batch_stats": teacher.batch_stats,
    }
    kernels = {}
    for kernel in KERNELS:
        evaluation = spec["evaluate"](
            policy,
            parameters,
            variant=kernel,
            stochastic=False,
            repeats_per_state=1,
            seed_base=20000,
            learner_seed=job["seed"],
            record_episodes=True,
        )
        kernels[kernel] = {
            family: {
                **evaluation[family],
                "mean_discounted_return": _family_return(
                    evaluation["episode_records"], family
                ),
            }
            for family in FAMILIES
        }
        kernels[kernel]["episodes"] = evaluation["episode_records"]
    contrasts = {}
    for family in FAMILIES:
        fixed = kernels["fixed"][family]
        mutant = kernels["mutant"][family]
        contrasts[family] = {
            "success_rate_difference": (
                mutant["success_rate"] - fixed["success_rate"]
            ),
            "mean_length_difference": (
                mutant["mean_length"] - fixed["mean_length"]
            ),
            "discounted_return_difference": (
                mutant["mean_discounted_return"]
                - fixed["mean_discounted_return"]
            ),
            "violation_delivery_rate_difference": (
                mutant["violation_delivery_rate"]
                - fixed["violation_delivery_rate"]
            ),
        }
    return {
        "env": job["env"],
        "seed": job["seed"],
        "stage": job["stage"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_state_sha256": _sha256(checkpoint / "state.msgpack"),
        "kernels": kernels,
        "same_teacher_mutant_minus_fixed": contrasts,
    }


def _aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        for kernel in KERNELS:
            block = row["kernels"][kernel]["natural_reset"]
            grouped[(row["env"], row["stage"], kernel)].append(
                (row["seed"], block)
            )
    output = []
    for (env, stage, kernel), seed_blocks in sorted(grouped.items()):
        blocks = [block for _, block in seed_blocks]
        output.append(
            {
                "env": env,
                "stage": stage,
                "kernel": kernel,
                "n_learner_seeds": len(blocks),
                "mean_success_rate": float(
                    np.mean([item["success_rate"] for item in blocks])
                ),
                "mean_violation_delivery_rate": float(
                    np.mean(
                        [item["violation_delivery_rate"] for item in blocks]
                    )
                ),
                "mean_discounted_return": float(
                    np.mean(
                        [item["mean_discounted_return"] for item in blocks]
                    )
                ),
                "seed_points": [
                    {
                        "seed": int(seed),
                        "success_rate": item["success_rate"],
                        "violation_delivery_rate": item[
                            "violation_delivery_rate"
                        ],
                        "mean_discounted_return": item[
                            "mean_discounted_return"
                        ],
                    }
                    for seed, item in seed_blocks
                ],
            }
        )
    return output


def _nearest_existing_parent(path):
    candidate = Path(path).parent
    while not candidate.exists():
        candidate = candidate.parent
    return candidate


def evaluate_all(output):
    output = Path(output)
    capacity_root = _nearest_existing_parent(output)
    capacity = shutil.disk_usage(capacity_root)
    if capacity.free < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            "teacher evaluation output filesystem lacks 8 GiB reserve"
        )
    jobs = build_evaluations()
    missing = [
        str(job["checkpoint"])
        for job in jobs
        if not (Path(job["checkpoint"]) / "state.msgpack").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"teacher checkpoints are missing: {missing}")
    before = {
        str(job["checkpoint"]): _sha256(Path(job["checkpoint"]) / "state.msgpack")
        for job in jobs
    }
    rows = []
    for index, job in enumerate(jobs, start=1):
        rows.append(_evaluate_checkpoint(job))
        print(
            f"[teacher-greedy] {index}/{len(jobs)} "
            f"{job['env']} seed={job['seed']} stage={job['stage']}",
            flush=True,
        )
    after = {
        path: _sha256(Path(path) / "state.msgpack")
        for path in before
    }
    immutable = before == after
    result = {
        "schema_version": "hackrl_dual_teacher_greedy_two_defects_v1",
        "execution_complete": immutable and len(rows) == len(jobs),
        "training_or_optimizer_updates": 0,
        "source_root": str(SOURCE_ROOT.resolve()),
        "seeds": list(SEEDS),
        "stages": list(STAGES),
        "evaluation": {
            "policy": "argmax delivery-head Q",
            "start_families": list(FAMILIES),
            "kernels": list(KERNELS),
            "repeats_per_state": 1,
            "checkpoint_count": len(jobs),
        },
        "capacity": {
            "filesystem": str(capacity_root.resolve()),
            "free_bytes_before_write": int(capacity.free),
            "minimum_reserve_bytes": MINIMUM_FREE_BYTES,
        },
        "checkpoints_immutable": immutable,
        "rows": rows,
        "natural_reset_means": _aggregate(rows),
        "limits": [
            (
                "Teachers learned from PPO-generated experience; failure is "
                "not a test of independently trained Q-learning."
            ),
            "Greedy teacher execution changes visitation relative to the PPO policy.",
            "The learner seed, not the evaluation episode, is the independent unit.",
        ],
    }
    _write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = evaluate_all(arguments.output)
    print(
        json.dumps(
            {
                "execution_complete": result["execution_complete"],
                "rows": len(result["rows"]),
                "output": str(Path(arguments.output).resolve()),
            },
            sort_keys=True,
        )
    )
    if not result["execution_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
