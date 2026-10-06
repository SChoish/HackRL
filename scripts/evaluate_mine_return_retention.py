#!/usr/bin/env python3
"""Close the frozen mine study with one no-learning return-near evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import jax
import numpy as np

from hackrl.mine_expedition_env import MineExpeditionStart
from hackrl.mine_expedition_ppo import (
    MineExpeditionPPOConfig,
    evaluate_mine_expedition_frozen,
    initialize_mine_expedition_ppo,
    load_mine_expedition_checkpoint,
    mine_expedition_checkpoint_files_present,
)


SEEDS = (40, 41, 42)
SOURCE_PHASE = "phase_2_return_path"
SOURCE_UPDATE = 384
SOURCE_RESULT_SHA256 = (
    "7aa01e77a929ed895f9204dac3e9aa550ad12e740f59e1f1408dd7e51c21d243"
)
REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(
    "docs/manifests/pack_restore_pretrained_teacher_freeze_v1.json"
)
EXECUTION_SOURCES = (
    Path("scripts/evaluate_mine_return_retention.py"),
    Path("src/hackrl/mine_expedition.py"),
    Path("src/hackrl/mine_expedition_env.py"),
    Path("src/hackrl/mine_expedition_ppo.py"),
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_snapshot(tree):
    return tuple(
        np.asarray(jax.device_get(value)).tobytes()
        for value in jax.tree.leaves(tree)
    )


def _git(*arguments):
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _provenance():
    sources = [str(path) for path in EXECUTION_SOURCES]
    dirty = _git("status", "--porcelain", "--", *sources)
    if dirty:
        raise RuntimeError("mine evaluation sources must be committed:\n" + dirty)
    tracked = set(_git("ls-files", "--", *sources).splitlines())
    missing = sorted(set(sources) - tracked)
    if missing:
        raise RuntimeError(f"untracked mine evaluation sources: {missing}")
    return {
        "execution_code_sha": _git("rev-parse", "HEAD"),
        "experiment_manifest_sha256": _sha256(REPOSITORY / MANIFEST_PATH),
        "execution_source_sha256": {
            name: _sha256(REPOSITORY / name) for name in sources
        },
    }


def evaluate(run_root):
    run_root = Path(run_root).resolve()
    provenance = _provenance()
    source_result = run_root / "diagnostic_result.json"
    if _sha256(source_result) != SOURCE_RESULT_SHA256:
        raise RuntimeError("source mine result digest does not match the frozen study")
    result_document = _read(source_result)
    if (
        result_document.get("execution_complete") is not True
        or result_document.get("status") != "scientific_fail_return_path"
        or result_document.get("mutant_training_authorized") is not False
    ):
        raise RuntimeError("source mine result is not the declared frozen failure")

    rows = []
    for seed in SEEDS:
        checkpoint = (
            run_root
            / SOURCE_PHASE
            / f"seed{seed}"
            / "checkpoints"
            / f"update_{SOURCE_UPDATE}"
        )
        if not mine_expedition_checkpoint_files_present(checkpoint):
            raise RuntimeError(f"invalid source checkpoint: {checkpoint}")
        state_path = checkpoint / "state.msgpack"
        before_digest = _sha256(state_path)
        config = MineExpeditionPPOConfig(**_read(checkpoint / "config.json"))
        if (
            config.seed != seed
            or config.training_start != MineExpeditionStart.RETURN_PATH.value
            or config.num_updates != SOURCE_UPDATE
        ):
            raise RuntimeError(f"source checkpoint contract mismatch: {checkpoint}")
        network, template = initialize_mine_expedition_ppo(config)
        runner = load_mine_expedition_checkpoint(checkpoint, template, config)
        learner_before = _tree_snapshot(runner)
        mode = evaluate_mine_expedition_frozen(
            network,
            runner.train_state.params,
            stochastic=False,
            episodes=1,
            seed_base=31_000,
            learner_seed=seed,
            start=MineExpeditionStart.RETURN_NEAR,
            record_episodes=True,
        )
        sample = evaluate_mine_expedition_frozen(
            network,
            runner.train_state.params,
            stochastic=True,
            episodes=128,
            seed_base=31_000,
            learner_seed=seed,
            start=MineExpeditionStart.RETURN_NEAR,
            record_episodes=True,
        )
        learner_after = _tree_snapshot(runner)
        after_digest = _sha256(state_path)
        if learner_before != learner_after or before_digest != after_digest:
            raise RuntimeError(f"frozen evaluation mutated learner state: seed {seed}")
        rows.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_state_sha256_before": before_digest,
                "checkpoint_state_sha256_after": after_digest,
                "runner_state_immutable": True,
                "mode": mode,
                "sample": sample,
            }
        )

    return {
        "schema_version": "hackrl_mine_return_retention_evaluation_v1",
        "study_status": "frozen_after_this_no_learning_evaluation",
        "source_study": "mine_expedition_fixed_return_curriculum_v1",
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "source_phase": SOURCE_PHASE,
        "source_update": SOURCE_UPDATE,
        "provenance": provenance,
        "evaluation_start": MineExpeditionStart.RETURN_NEAR.value,
        "mode_episodes_per_seed": 1,
        "sample_episodes_per_seed": 128,
        "training_or_optimizer_updates": 0,
        "seeds": list(SEEDS),
        "results": rows,
        "interpretation_rule": {
            "retained": (
                "High return_near success means the near delivery behavior survived "
                "phase 2; low return_path performance is then a route/state coverage gap."
            ),
            "lost": (
                "Low return_near success means phase-2 training also lost behavior that "
                "phase 1 had learned."
            ),
        },
        "claim_limit": (
            "This is a deterministic-checkpoint diagnostic, not a new learning run or "
            "an independent replication."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        default=(
            "/raid/ext_csv/HackRL/runs/"
            "mine_expedition_fixed_return_curriculum_v1"
        ),
    )
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = evaluate(arguments.run_root)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps({
        "output": str(output.resolve()),
        "sample_success_rate_by_seed": {
            str(row["seed"]): row["sample"]["success_rate"]
            for row in result["results"]
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
