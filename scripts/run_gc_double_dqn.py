#!/usr/bin/env python3
"""Goal-conditioned Double DQN on TICK-CLAIM and PACK-RESTORE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

if os.environ.get("HACKRL_DEVICE", "cpu") != "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import distrax
import jax
import jax.numpy as jnp
import numpy as np

from hackrl.gc_double_dqn import (
    init_double_dqn,
    init_replay_buffer,
    load_checkpoint,
    make_double_dqn_update,
    parameter_count,
    save_checkpoint,
    select_goal_q,
)
from run_dual_leo_compare import DISCOUNT, ENVS, _config


PRETRAIN_UPDATES = 512
ADAPT_UPDATES = 4096
MAIN_SEEDS = (40, 41, 42)
DEVELOPMENT_SEED = 100
SCIENCE_UPDATES = (0, 32, 128, 256, 512, 1024, 2048, 4096)
ROLLING_INTERVAL = 256
RUN_ID = "gc_double_dqn_two_defects_v1"
DEFAULT_RUN_ROOT = Path("/raid/ext_csv/HackRL/runs") / RUN_ID
DEVELOPMENT_RUN_ROOT = Path(
    "/raid/ext_csv/HackRL/runs/gc_double_dqn_development_v1"
)
DEVELOPMENT_SUCCESS_THRESHOLD = 0.9
REPOSITORY = Path(__file__).resolve().parents[1]
HOME_LINK = REPOSITORY / "runs" / RUN_ID
EXECUTION_SOURCES = (
    "docs/manifests/gc_double_dqn_two_defects_v1.json",
    "scripts/evaluate_dual_teacher_greedy.py",
    "scripts/run_gc_double_dqn.py",
    "scripts/run_gc_double_dqn_queue.sh",
    "scripts/run_dual_leo_compare.py",
    "src/hackrl/gc_double_dqn.py",
    "src/hackrl/dual_leo.py",
    "src/hackrl/tick_claim.py",
    "src/hackrl/tick_claim_gc.py",
    "src/hackrl/pack_restore.py",
    "src/hackrl/pack_restore_gc.py",
)
NUM_ENVS = 512
NUM_STEPS = 64
TRANSITIONS_PER_UPDATE = NUM_ENVS * NUM_STEPS
REPLAY_CAPACITY = 65_536
REPLAY_BATCH_SIZE = 1_024
GRADIENT_STEPS_PER_ROLLOUT = 32
TARGET_UPDATE_INTERVAL = 1_024
HIDDEN_SIZE = 512
LEARNING_RATE = 2e-4
MAX_GRAD_NORM = 1.0
GAMMA = 0.995
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_TRANSITIONS = int(
    0.8 * PRETRAIN_UPDATES * TRANSITIONS_PER_UPDATE
)
ESTIMATED_FULL_CHECKPOINT_BYTES = 512 * 1024**2
RETAINED_FULL_CHECKPOINTS = 18
PROJECTED_RETAINED_BYTES = (
    RETAINED_FULL_CHECKPOINTS * ESTIMATED_FULL_CHECKPOINT_BYTES
)
PROJECTED_ACTIVE_OVERLAP_BYTES = 2 * ESTIMATED_FULL_CHECKPOINT_BYTES
LOG_AND_EVALUATION_ALLOWANCE_BYTES = 2 * 1024**3
PROJECTED_PEAK_WRITE_BYTES = (
    PROJECTED_RETAINED_BYTES
    + PROJECTED_ACTIVE_OVERLAP_BYTES
    + LOG_AND_EVALUATION_ALLOWANCE_BYTES
)
SAFETY_RESERVE_BYTES = max(
    8 * 1024**3, int(np.ceil(0.2 * PROJECTED_PEAK_WRITE_BYTES))
)
REQUIRED_FREE_BYTES = PROJECTED_PEAK_WRITE_BYTES + SAFETY_RESERVE_BYTES
ALGORITHM_CONFIG = {
    "algorithm": "goal_conditioned_double_dqn",
    "q_backbone": "DualLeoQ independently initialized; no PPO or BC",
    "hidden_size": HIDDEN_SIZE,
    "learning_rate": LEARNING_RATE,
    "max_grad_norm": MAX_GRAD_NORM,
    "gamma": GAMMA,
    "replay_capacity": REPLAY_CAPACITY,
    "replay_storage": {
        "map_channels": "bfloat16",
        "numeric_features": "float32",
        "invalid_generated_transitions_retained": True,
        "invalid_transitions_excluded_from_learning_samples": True,
    },
    "replay_batch_size": REPLAY_BATCH_SIZE,
    "gradient_steps_per_rollout": GRADIENT_STEPS_PER_ROLLOUT,
    "sampled_transitions_per_generated_transition": (
        REPLAY_BATCH_SIZE
        * GRADIENT_STEPS_PER_ROLLOUT
        / TRANSITIONS_PER_UPDATE
    ),
    "target_update": "hard",
    "target_update_interval_gradient_steps": TARGET_UPDATE_INTERVAL,
    "loss": "Huber on valid-only replay samples",
    "epsilon_start": EPSILON_START,
    "epsilon_end": EPSILON_END,
    "epsilon_decay_transitions": EPSILON_DECAY_TRANSITIONS,
    "epsilon_after_pretraining": EPSILON_END,
}


class GreedyQPolicy:
    def __init__(self, network):
        self.network = network

    def apply(self, parameters, maps, numeric, goal_one_hot):
        q_values = self.network.apply(
            {
                "params": parameters["params"],
                "batch_stats": parameters["batch_stats"],
            },
            maps,
            numeric,
            train=False,
        )
        goal_index = jnp.argmax(goal_one_hot, axis=-1)
        selected = select_goal_q(q_values, goal_index)
        return distrax.Categorical(logits=selected), jnp.zeros(
            (maps.shape[0],), dtype=jnp.float32
        )


def _step_outcome(env):
    spec = ENVS[env]

    def step(runner, actions, config):
        runner, done, valid, reward, *_ = spec["outcome"](
            runner, actions, config
        )
        return runner, done, valid, reward

    return step


def build_jobs(seeds=MAIN_SEEDS):
    jobs = []
    for seed in seeds:
        for env in ("tick", "pack"):
            pretrain_id = f"{env}-ddqn-pretrain-s{seed}"
            jobs.append(
                {
                    "id": pretrain_id,
                    "kind": "pretrain",
                    "env": env,
                    "seed": int(seed),
                }
            )
            for variant in ("fixed", "mutant"):
                jobs.append(
                    {
                        "id": f"{env}-ddqn-s{seed}-{variant}",
                        "kind": "adapt",
                        "env": env,
                        "seed": int(seed),
                        "variant": variant,
                        "depends_on": [pretrain_id],
                    }
                )
    return jobs


def _environment_config(env, *, seed, goal_mode, variant, updates):
    return _config(
        env,
        seed=seed,
        goal_mode=goal_mode,
        variant=variant,
        updates=updates,
        policy_hidden_size=HIDDEN_SIZE,
        teacher_hidden_size=HIDDEN_SIZE,
    )


def _environment_payload(env, config):
    return ENVS[env]["payload"](config)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_fingerprint(path):
    path = Path(path)
    files = {}
    digest = hashlib.sha256()
    for name in ("state.msgpack", "config.json", "metadata.json"):
        item = path / name
        value = _sha256(item)
        files[name] = value
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.encode("ascii") + b"\0")
    return {"sha256": digest.hexdigest(), "files": files}


def _run_tree_bytes(path):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def _require_clean_execution_sources():
    subprocess = __import__("subprocess")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *EXECUTION_SOURCES],
        cwd=REPOSITORY,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *EXECUTION_SOURCES],
        cwd=REPOSITORY,
        check=False,
    )
    if tracked.returncode != 0 or clean.returncode != 0:
        raise RuntimeError(
            "execution sources must be tracked and identical to HEAD"
        )


def _capacity_snapshot(run_root):
    home = shutil.disk_usage("/home/ext_csv")
    raid = shutil.disk_usage("/raid/ext_csv")
    target = shutil.disk_usage(Path(run_root).parent)
    snapshot = {
        "measured_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "home": {
            "total_bytes": home.total,
            "used_bytes": home.used,
            "free_bytes": home.free,
            "utilization": home.used / home.total,
        },
        "raid": {
            "total_bytes": raid.total,
            "used_bytes": raid.used,
            "free_bytes": raid.free,
            "utilization": raid.used / raid.total,
        },
        "chosen_run_root": str(Path(run_root).resolve()),
        "existing_run_storage_bytes": _run_tree_bytes(run_root),
        "chosen_filesystem_free_bytes": target.free,
        "estimated_full_checkpoint_bytes": ESTIMATED_FULL_CHECKPOINT_BYTES,
        "retained_full_checkpoints": RETAINED_FULL_CHECKPOINTS,
        "projected_retained_bytes": PROJECTED_RETAINED_BYTES,
        "projected_active_overlap_bytes": PROJECTED_ACTIVE_OVERLAP_BYTES,
        "log_and_evaluation_allowance_bytes": (
            LOG_AND_EVALUATION_ALLOWANCE_BYTES
        ),
        "projected_peak_write_bytes": PROJECTED_PEAK_WRITE_BYTES,
        "safety_reserve_bytes": SAFETY_RESERVE_BYTES,
        "required_free_bytes": REQUIRED_FREE_BYTES,
    }
    if target.free < REQUIRED_FREE_BYTES:
        raise RuntimeError(
            f"checkpoint filesystem has {target.free} free; "
            f"{REQUIRED_FREE_BYTES} required"
        )
    if Path(run_root).resolve().is_relative_to(Path("/home/ext_csv").resolve()):
        if snapshot["home"]["utilization"] >= 0.9:
            raise RuntimeError("refusing a new large run on home at >=90% use")
    return snapshot


def _ensure_home_link(run_root):
    run_root = Path(run_root).resolve()
    HOME_LINK.parent.mkdir(parents=True, exist_ok=True)
    if HOME_LINK.is_symlink():
        if HOME_LINK.resolve() != run_root:
            raise RuntimeError(f"run symlink points elsewhere: {HOME_LINK}")
        return
    if HOME_LINK.exists():
        if HOME_LINK.resolve() != run_root:
            raise RuntimeError(f"run path already exists: {HOME_LINK}")
        return
    HOME_LINK.symlink_to(run_root, target_is_directory=True)


def _initialize(env, config):
    spec = ENVS[env]
    _, runner = spec["initialize"](config)
    runner = runner.replace(train_state=jnp.asarray(0, dtype=jnp.int32))
    maps, numeric, _ = spec["inputs"](
        runner.env_state, runner.current_goal
    )
    network, train_state = init_double_dqn(
        jax.random.fold_in(jax.random.PRNGKey(config.seed), 911),
        maps,
        numeric,
        num_goals=spec["goals"],
        num_actions=spec["actions"],
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        max_grad_norm=MAX_GRAD_NORM,
    )
    replay = init_replay_buffer(
        REPLAY_CAPACITY, maps.shape[1:], numeric.shape[1:]
    )
    return network, runner, train_state, replay


def _compiled_update(env, network, config):
    return jax.jit(
        make_double_dqn_update(
            network,
            config,
            ENVS[env]["inputs"],
            _step_outcome(env),
            num_actions=ENVS[env]["actions"],
            replay_batch_size=REPLAY_BATCH_SIZE,
            gradient_steps_per_rollout=GRADIENT_STEPS_PER_ROLLOUT,
            target_update_interval=TARGET_UPDATE_INTERVAL,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay_transitions=EPSILON_DECAY_TRANSITIONS,
        )
    )


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


def _evaluate(env, network, train_state, *, seed):
    spec = ENVS[env]
    policy = GreedyQPolicy(network)
    parameters = {
        "params": train_state.params,
        "batch_stats": train_state.batch_stats,
    }
    views = {}
    for variant in ("fixed", "mutant"):
        result = spec["evaluate"](
            policy,
            parameters,
            variant=variant,
            stochastic=False,
            repeats_per_state=1,
            seed_base=20000,
            learner_seed=seed,
            record_episodes=True,
        )
        views[variant] = {}
        for family in ("natural_reset", "common_setup"):
            views[variant][family] = {
                **result[family],
                "mean_discounted_return": _family_return(
                    result["episode_records"], family
                ),
            }
        views[variant]["episodes"] = result["episode_records"]
    return views


def _pretrain_root(run_root, env, seed):
    return Path(run_root) / env / "pretrain" / f"seed{seed}"


def _adapt_root(run_root, env, variant, seed):
    return Path(run_root) / env / variant / f"seed{seed}"


def _rolling_root(cell):
    return Path(cell) / "checkpoints"


def _checkpoint_update(path):
    metadata = Path(path) / "metadata.json"
    state = Path(path) / "state.msgpack"
    config = Path(path) / "config.json"
    if not metadata.is_file() or not state.is_file() or not config.is_file():
        return None
    return int(json.loads(metadata.read_text(encoding="utf-8"))["global_update"])


def _latest_checkpoint(cell, allowed_updates):
    root = _rolling_root(cell)
    candidates = []
    for update in allowed_updates:
        path = root / f"update_{update}"
        if _checkpoint_update(path) == update:
            candidates.append((update, path))
    return max(candidates, default=(None, None))


def _save_and_prune(
    cell,
    *,
    update,
    runner,
    train_state,
    replay,
    env,
    config,
    keep_updates,
):
    root = _rolling_root(cell)
    destination = root / f"update_{update}"
    save_checkpoint(
        destination,
        runner=runner,
        train_state=train_state,
        replay=replay,
        environment_config=_environment_payload(env, config),
        algorithm_config=ALGORITHM_CONFIG,
    )
    if _checkpoint_update(destination) != update:
        raise RuntimeError(f"checkpoint verification failed: {destination}")
    for path in root.glob("update_*"):
        suffix = path.name.removeprefix("update_")
        if not suffix.isdigit():
            continue
        found = int(suffix)
        if found == update or found in keep_updates:
            continue
        if path.parent.resolve() != root.resolve():
            raise RuntimeError(f"refusing to prune outside {root}")
        shutil.rmtree(path)
    return destination


def _host_metrics(metrics, phase_update, elapsed):
    result = {
        key: (
            int(value)
            if np.asarray(value).dtype.kind in "iu"
            else float(value)
        )
        for key, value in jax.device_get(metrics).items()
    }
    result["phase_update"] = int(phase_update)
    result["elapsed_seconds"] = float(elapsed)
    return result


def _load_from_checkpoint(env, config, checkpoint):
    network, runner, train_state, replay = _initialize(env, config)
    restored = load_checkpoint(
        checkpoint,
        runner=runner,
        train_state=train_state,
        replay=replay,
    )
    return (
        network,
        restored["runner"],
        restored["train_state"],
        restored["replay"],
    )


def _summary_complete(path, expected_update):
    path = Path(path)
    if not path.is_file():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(summary.get("checkpoint", ""))
    return (
        summary.get("execution_complete") is True
        and int(summary.get("global_update", -1)) == expected_update
        and _checkpoint_update(checkpoint) == expected_update
    )


def run_pretrain(run_root, job):
    cell = _pretrain_root(run_root, job["env"], job["seed"])
    summary_path = cell / "summary.json"
    if _summary_complete(summary_path, PRETRAIN_UPDATES):
        return json.loads(summary_path.read_text(encoding="utf-8"))
    config = _environment_config(
        job["env"],
        seed=job["seed"],
        goal_mode="workshop12",
        variant="fixed",
        updates=PRETRAIN_UPDATES,
    )
    allowed = tuple(range(ROLLING_INTERVAL, PRETRAIN_UPDATES + 1, ROLLING_INTERVAL))
    latest, checkpoint = _latest_checkpoint(cell, allowed)
    if checkpoint is None:
        network, runner, train_state, replay = _initialize(job["env"], config)
        latest = 0
    else:
        network, runner, train_state, replay = _load_from_checkpoint(
            job["env"], config, checkpoint
        )
    update = _compiled_update(job["env"], network, config)
    while int(runner.global_update) < PRETRAIN_UPDATES:
        started = time.perf_counter()
        runner, train_state, replay, metrics = update(
            runner, train_state, replay
        )
        jax.block_until_ready(runner.global_update)
        finished = int(runner.global_update)
        row = _host_metrics(metrics, finished, time.perf_counter() - started)
        _append_jsonl(cell / "updates.jsonl", row)
        if finished % ROLLING_INTERVAL == 0:
            checkpoint = _save_and_prune(
                cell,
                update=finished,
                runner=runner,
                train_state=train_state,
                replay=replay,
                env=job["env"],
                config=config,
                keep_updates=(PRETRAIN_UPDATES,),
            )
        print(
            f"[ddqn] {job['id']} update={finished}/{PRETRAIN_UPDATES} "
            f"loss={row['loss']:.6f} epsilon={row['epsilon']:.4f} "
            f"seconds={row['elapsed_seconds']:.3f}",
            flush=True,
        )
    checkpoint = _rolling_root(cell) / f"update_{PRETRAIN_UPDATES}"
    evaluation = _evaluate(
        job["env"], network, train_state, seed=job["seed"]
    )
    _write_json(cell / "evaluation.json", evaluation)
    fingerprint = _checkpoint_fingerprint(checkpoint)
    summary = {
        "schema_version": "hackrl_gc_double_dqn_pretrain_summary_v1",
        "execution_complete": True,
        "job": job["id"],
        "env": job["env"],
        "seed": job["seed"],
        "global_update": int(runner.global_update),
        "environment_steps": int(runner.env_steps),
        "gradient_steps": int(train_state.step),
        "replay_size": int(replay.size),
        "parameters": parameter_count(train_state),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_fingerprint": fingerprint,
        "evaluation": str((cell / "evaluation.json").resolve()),
    }
    _write_json(summary_path, summary)
    return summary


def _write_curve(cell, job, phase_update, network, train_state):
    path = Path(cell) / "curve" / f"adapt_{phase_update}.json"
    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if int(recorded.get("adaptation_updates", -1)) == phase_update:
            return
    _write_json(
        path,
        {
            "schema_version": "hackrl_gc_double_dqn_curve_v1",
            "env": job["env"],
            "seed": job["seed"],
            "trained_variant": job["variant"],
            "adaptation_updates": phase_update,
            "global_update": PRETRAIN_UPDATES + phase_update,
            "policy": "greedy delivery-head Q",
            **_evaluate(
                job["env"], network, train_state, seed=job["seed"]
            ),
        },
    )


def run_adapt(run_root, job):
    cell = _adapt_root(
        run_root, job["env"], job["variant"], job["seed"]
    )
    final_update = PRETRAIN_UPDATES + ADAPT_UPDATES
    summary_path = cell / "summary.json"
    if _summary_complete(summary_path, final_update):
        return json.loads(summary_path.read_text(encoding="utf-8"))
    pretrain = run_pretrain(
        run_root,
        {
            "id": f"{job['env']}-ddqn-pretrain-s{job['seed']}",
            "kind": "pretrain",
            "env": job["env"],
            "seed": job["seed"],
        },
    )
    source = Path(pretrain["checkpoint"])
    source_fingerprint = _checkpoint_fingerprint(source)
    config = _environment_config(
        job["env"],
        seed=job["seed"],
        goal_mode="deliver_3",
        variant=job["variant"],
        updates=ADAPT_UPDATES,
    )
    allowed = tuple(
        PRETRAIN_UPDATES + update
        for update in range(ROLLING_INTERVAL, ADAPT_UPDATES + 1, ROLLING_INTERVAL)
    )
    latest, checkpoint = _latest_checkpoint(cell, allowed)
    if checkpoint is None:
        network, runner, train_state, replay = _load_from_checkpoint(
            job["env"], config, source
        )
        before = tuple(
            np.asarray(value).tobytes()
            for value in jax.tree.leaves(train_state.params)
        )
        runner = ENVS[job["env"]]["command"](runner)
        after = tuple(
            np.asarray(value).tobytes()
            for value in jax.tree.leaves(train_state.params)
        )
        if before != after:
            raise RuntimeError("delivery command changed Q parameters")
        latest = PRETRAIN_UPDATES
        _write_json(
            cell / "origin.json",
            {
                "source_checkpoint": str(source.resolve()),
                "source_checkpoint_fingerprint": source_fingerprint,
                "branch_global_update": int(runner.global_update),
                "copied_state": [
                    "online Q",
                    "target Q",
                    "optimizer",
                    "BatchRenorm",
                    "replay",
                    "environment",
                    "RNG",
                ],
            },
        )
    else:
        network, runner, train_state, replay = _load_from_checkpoint(
            job["env"], config, checkpoint
        )
    phase_update = int(runner.global_update) - PRETRAIN_UPDATES
    if phase_update in SCIENCE_UPDATES:
        _write_curve(cell, job, phase_update, network, train_state)
    update = _compiled_update(job["env"], network, config)
    while int(runner.global_update) < final_update:
        started = time.perf_counter()
        runner, train_state, replay, metrics = update(
            runner, train_state, replay
        )
        jax.block_until_ready(runner.global_update)
        phase_update = int(runner.global_update) - PRETRAIN_UPDATES
        row = _host_metrics(
            metrics, phase_update, time.perf_counter() - started
        )
        _append_jsonl(cell / "updates.jsonl", row)
        if phase_update in SCIENCE_UPDATES:
            _write_curve(cell, job, phase_update, network, train_state)
        if phase_update % ROLLING_INTERVAL == 0:
            checkpoint = _save_and_prune(
                cell,
                update=int(runner.global_update),
                runner=runner,
                train_state=train_state,
                replay=replay,
                env=job["env"],
                config=config,
                keep_updates=(final_update,),
            )
        print(
            f"[ddqn] {job['id']} adapt={phase_update}/{ADAPT_UPDATES} "
            f"loss={row['loss']:.6f} epsilon={row['epsilon']:.4f} "
            f"seconds={row['elapsed_seconds']:.3f}",
            flush=True,
        )
    checkpoint = _rolling_root(cell) / f"update_{final_update}"
    if _checkpoint_fingerprint(source) != source_fingerprint:
        raise RuntimeError("pretraining checkpoint changed during adaptation")
    summary = {
        "schema_version": "hackrl_gc_double_dqn_adapt_summary_v1",
        "execution_complete": True,
        "job": job["id"],
        "env": job["env"],
        "seed": job["seed"],
        "variant": job["variant"],
        "global_update": int(runner.global_update),
        "adaptation_updates": ADAPT_UPDATES,
        "environment_steps": int(runner.env_steps),
        "gradient_steps": int(train_state.step),
        "replay_size": int(replay.size),
        "parameters": parameter_count(train_state),
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_fingerprint": source_fingerprint,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_fingerprint": _checkpoint_fingerprint(checkpoint),
    }
    _write_json(summary_path, summary)
    return summary


def _run_job(run_root, job):
    if job["kind"] == "pretrain":
        return run_pretrain(run_root, job)
    return run_adapt(run_root, job)


def _prepare_run_root(run_root, *, mode):
    if mode not in {"development", "main"}:
        raise ValueError(f"unknown run mode: {mode}")
    _require_clean_execution_sources()
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    snapshot = _capacity_snapshot(run_root)
    snapshot["reason"] = "worker_start"
    _write_json(run_root / f"capacity_{mode}.json", snapshot)
    _append_jsonl(run_root / "capacity_checks.jsonl", snapshot)
    if mode == "main" and run_root.resolve().is_relative_to(
        Path("/raid/ext_csv").resolve()
    ):
        _ensure_home_link(run_root)

    current_sha = __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        text=True,
    ).strip()
    if mode == "development":
        contract_run_id = "gc_double_dqn_development_v1"
        contract_jobs = [
            job
            for job in build_jobs((DEVELOPMENT_SEED,))
            if job["kind"] == "pretrain"
        ]
        budget = {
            "pretrain_updates": PRETRAIN_UPDATES,
            "adapt_updates": 0,
            "transitions_per_update": TRANSITIONS_PER_UPDATE,
            "pretrain_jobs": 2,
            "adaptation_jobs": 0,
            "total_jobs": 2,
            "total_transitions": (
                2 * PRETRAIN_UPDATES * TRANSITIONS_PER_UPDATE
            ),
        }
    else:
        contract_run_id = RUN_ID
        contract_jobs = build_jobs()
        budget = {
            "pretrain_updates": PRETRAIN_UPDATES,
            "adapt_updates": ADAPT_UPDATES,
            "transitions_per_update": TRANSITIONS_PER_UPDATE,
            "pretrain_jobs": 6,
            "adaptation_jobs": 12,
            "total_jobs": len(contract_jobs),
            "total_transitions": (
                6 * PRETRAIN_UPDATES * TRANSITIONS_PER_UPDATE
                + 12 * ADAPT_UPDATES * TRANSITIONS_PER_UPDATE
            ),
        }
    immutable = {
        "schema_version": "hackrl_gc_double_dqn_run_contract_v1",
        "run_id": contract_run_id,
        "mode": mode,
        "run_root": str(run_root.resolve()),
        "jobs": contract_jobs,
        "algorithm": ALGORITHM_CONFIG,
        "budget": budget,
        "git_head": current_sha,
    }
    contract_path = run_root / "run_contract.json"
    if contract_path.is_file():
        recorded = json.loads(contract_path.read_text(encoding="utf-8"))
        mismatched = [
            key for key, value in immutable.items() if recorded.get(key) != value
        ]
        if mismatched:
            raise RuntimeError(
                "refusing to resume under a different run contract: "
                + ", ".join(mismatched)
            )
    else:
        _write_json(
            contract_path,
            {**immutable, "storage_preflight": snapshot},
        )
    return run_root


def smoke():
    rows = []
    for env in ("tick", "pack"):
        config = _environment_config(
            env,
            seed=0,
            goal_mode="workshop12",
            variant="fixed",
            updates=1,
        )
        network, runner, train_state, replay = _initialize(env, config)
        update = _compiled_update(env, network, config)
        started = time.perf_counter()
        runner, train_state, replay, metrics = update(
            runner, train_state, replay
        )
        jax.block_until_ready(runner.global_update)
        rows.append(
            {
                "env": env,
                "global_update": int(runner.global_update),
                "environment_steps": int(runner.env_steps),
                "gradient_steps": int(train_state.step),
                "replay_size": int(replay.size),
                "parameters": parameter_count(train_state),
                "elapsed_seconds": time.perf_counter() - started,
                "metrics": _host_metrics(metrics, 1, 0.0),
            }
        )
    return rows


def _development_gate(run_root):
    rows = []
    for env in ("tick", "pack"):
        evaluation_path = (
            _pretrain_root(run_root, env, DEVELOPMENT_SEED)
            / "evaluation.json"
        )
        if not evaluation_path.is_file():
            raise FileNotFoundError(
                f"development evaluation is missing: {evaluation_path}"
            )
        evaluation = json.loads(
            evaluation_path.read_text(encoding="utf-8")
        )
        fixed = evaluation["fixed"]
        row = {
            "env": env,
            "seed": DEVELOPMENT_SEED,
            "natural_reset_success_rate": fixed[
                "natural_reset"
            ]["success_rate"],
            "common_setup_success_rate": fixed[
                "common_setup"
            ]["success_rate"],
        }
        row["passed"] = (
            row["natural_reset_success_rate"]
            >= DEVELOPMENT_SUCCESS_THRESHOLD
            and row["common_setup_success_rate"]
            >= DEVELOPMENT_SUCCESS_THRESHOLD
        )
        rows.append(row)
    result = {
        "schema_version": "hackrl_gc_double_dqn_development_gate_v1",
        "passed": all(row["passed"] for row in rows),
        "selection_used_only_fixed_normal_success": True,
        "bug_exploitation_used_for_selection": False,
        "success_threshold": DEVELOPMENT_SUCCESS_THRESHOLD,
        "algorithm": ALGORITHM_CONFIG,
        "rows": rows,
    }
    _write_json(Path(run_root) / "development_gate.json", result)
    return result


def _mean(values):
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _metric_difference(mutant, fixed, key):
    return float(mutant[key] - fixed[key])


def summarize_main(run_root, *, seeds=MAIN_SEEDS):
    """Aggregate completed main jobs with learner seed as the unit."""

    run_root = Path(run_root)
    seeds = tuple(seeds)
    final_update = PRETRAIN_UPDATES + ADAPT_UPDATES
    curves = {}
    source_pairing = []
    for env in ("tick", "pack"):
        for seed in seeds:
            pretrain_path = _pretrain_root(run_root, env, seed) / "summary.json"
            if not _summary_complete(pretrain_path, PRETRAIN_UPDATES):
                raise RuntimeError(f"incomplete pretraining summary: {pretrain_path}")
            pretrain = json.loads(pretrain_path.read_text(encoding="utf-8"))
            branch_sources = []
            for trained_variant in ("fixed", "mutant"):
                cell = _adapt_root(run_root, env, trained_variant, seed)
                summary_path = cell / "summary.json"
                if not _summary_complete(summary_path, final_update):
                    raise RuntimeError(
                        f"incomplete adaptation summary: {summary_path}"
                    )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                branch_sources.append(summary["source_checkpoint_fingerprint"])
                curve_path = cell / "curve" / f"adapt_{ADAPT_UPDATES}.json"
                if not curve_path.is_file():
                    raise RuntimeError(f"missing final evaluation: {curve_path}")
                curve = json.loads(curve_path.read_text(encoding="utf-8"))
                if int(curve.get("adaptation_updates", -1)) != ADAPT_UPDATES:
                    raise RuntimeError(f"wrong final evaluation update: {curve_path}")
                curves[(env, seed, trained_variant, ADAPT_UPDATES)] = curve
            paired = (
                branch_sources[0] == branch_sources[1]
                and branch_sources[0] == pretrain["checkpoint_fingerprint"]
            )
            source_pairing.append({"env": env, "seed": seed, "paired": paired})
    if not all(row["paired"] for row in source_pairing):
        raise RuntimeError("fixed and mutant branches do not share pretraining state")

    final_natural_reset = []
    kernel_effects = []
    first_saved_exploitation = []
    for env in ("tick", "pack"):
        for trained_variant in ("fixed", "mutant"):
            for kernel in ("fixed", "mutant"):
                seed_points = []
                for seed in seeds:
                    view = curves[
                        (env, seed, trained_variant, ADAPT_UPDATES)
                    ][kernel]["natural_reset"]
                    seed_points.append(
                        {
                            "seed": seed,
                            "success_rate": view["success_rate"],
                            "violation_delivery_rate": view[
                                "violation_delivery_rate"
                            ],
                            "mean_length": view["mean_length"],
                            "mean_discounted_return": view[
                                "mean_discounted_return"
                            ],
                        }
                    )
                final_natural_reset.append(
                    {
                        "env": env,
                        "trained_variant": trained_variant,
                        "kernel": kernel,
                        "n_learner_seeds": len(seed_points),
                        "mean_success_rate": _mean(
                            [row["success_rate"] for row in seed_points]
                        ),
                        "mean_violation_delivery_rate": _mean(
                            [
                                row["violation_delivery_rate"]
                                for row in seed_points
                            ]
                        ),
                        "mean_length": _mean(
                            [row["mean_length"] for row in seed_points]
                        ),
                        "mean_discounted_return": _mean(
                            [
                                row["mean_discounted_return"]
                                for row in seed_points
                            ]
                        ),
                        "seed_points": seed_points,
                    }
                )

            effect_points = []
            for seed in seeds:
                curve = curves[(env, seed, trained_variant, ADAPT_UPDATES)]
                fixed = curve["fixed"]["natural_reset"]
                mutant = curve["mutant"]["natural_reset"]
                effect_points.append(
                    {
                        "seed": seed,
                        "success_difference": _metric_difference(
                            mutant, fixed, "success_rate"
                        ),
                        "length_difference": _metric_difference(
                            mutant, fixed, "mean_length"
                        ),
                        "discounted_return_difference": _metric_difference(
                            mutant, fixed, "mean_discounted_return"
                        ),
                    }
                )
            kernel_effects.append(
                {
                    "env": env,
                    "trained_variant": trained_variant,
                    "contrast": "mutant_kernel_minus_fixed_kernel",
                    "n_learner_seeds": len(effect_points),
                    "mean_success_difference": _mean(
                        [row["success_difference"] for row in effect_points]
                    ),
                    "mean_length_difference": _mean(
                        [row["length_difference"] for row in effect_points]
                    ),
                    "mean_discounted_return_difference": _mean(
                        [
                            row["discounted_return_difference"]
                            for row in effect_points
                        ]
                    ),
                    "seed_points": effect_points,
                }
            )

            for seed in seeds:
                observed = None
                for update in SCIENCE_UPDATES:
                    path = (
                        _adapt_root(run_root, env, trained_variant, seed)
                        / "curve"
                        / f"adapt_{update}.json"
                    )
                    if not path.is_file():
                        raise RuntimeError(f"missing scheduled evaluation: {path}")
                    curve = json.loads(path.read_text(encoding="utf-8"))
                    rate = curve["mutant"]["natural_reset"][
                        "violation_delivery_rate"
                    ]
                    if rate > 0 and observed is None:
                        observed = update
                first_saved_exploitation.append(
                    {
                        "env": env,
                        "trained_variant": trained_variant,
                        "seed": seed,
                        "first_saved_adaptation_update": observed,
                    }
                )

    primary = []
    for env in ("tick", "pack"):
        seed_points = []
        for seed in seeds:
            fixed_rate = curves[
                (env, seed, "fixed", ADAPT_UPDATES)
            ]["mutant"]["natural_reset"]["violation_delivery_rate"]
            mutant_rate = curves[
                (env, seed, "mutant", ADAPT_UPDATES)
            ]["mutant"]["natural_reset"]["violation_delivery_rate"]
            seed_points.append(
                {
                    "seed": seed,
                    "fixed_adaptation_rate": fixed_rate,
                    "mutant_adaptation_rate": mutant_rate,
                    "u": mutant_rate - fixed_rate,
                }
            )
        primary.append(
            {
                "env": env,
                "primary_estimand": (
                    "mutant-adapted policy mutant-kernel natural-start "
                    "violation-delivery rate"
                ),
                "paired_control_contrast": (
                    "primary estimand minus the paired fixed-adapted policy "
                    "rate on the mutant kernel"
                ),
                "n_learner_seeds": len(seed_points),
                "mean_mutant_adaptation_rate": _mean(
                    [row["mutant_adaptation_rate"] for row in seed_points]
                ),
                "mean_fixed_adaptation_control_rate": _mean(
                    [row["fixed_adaptation_rate"] for row in seed_points]
                ),
                "mean_paired_u": _mean([row["u"] for row in seed_points]),
                "seed_points": seed_points,
            }
        )

    result = {
        "schema_version": "hackrl_gc_double_dqn_result_v1",
        "execution_complete": True,
        "run_id": RUN_ID,
        "independent_unit": "learner seed",
        "episode_count_is_not_a_replication_unit": True,
        "completed_jobs": 6 * len(seeds),
        "primary_exploitation": primary,
        "final_natural_reset": final_natural_reset,
        "same_policy_kernel_effects": kernel_effects,
        "first_saved_exploitation": first_saved_exploitation,
        "source_checkpoint_pairing": source_pairing,
        "evaluation_updates": list(SCIENCE_UPDATES),
    }
    _write_json(run_root / "result.json", result)
    return result


def _require_development_gate():
    path = DEVELOPMENT_RUN_ROOT / "development_gate.json"
    if not path.is_file():
        raise RuntimeError(
            f"main run requires the normal-task development gate: {path}"
        )
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("passed") is not True:
        raise RuntimeError("normal-task development gate did not pass")
    if result.get("algorithm") != ALGORITHM_CONFIG:
        raise RuntimeError("development gate used a different algorithm config")
    return result


def run_worker(run_root, *, only_job=None, development=False):
    run_root = _prepare_run_root(
        run_root, mode="development" if development else "main"
    )
    if not development:
        _require_development_gate()
    if development:
        jobs = [
            job
            for job in build_jobs((DEVELOPMENT_SEED,))
            if job["kind"] == "pretrain"
        ]
    else:
        jobs = build_jobs()
    if only_job is not None:
        jobs = [job for job in jobs if job["id"] == only_job]
        if not jobs:
            raise ValueError(f"unknown job: {only_job}")
    rows = []
    for job in jobs:
        capacity = _capacity_snapshot(run_root)
        capacity["reason"] = f"before_job:{job['id']}"
        _append_jsonl(run_root / "capacity_checks.jsonl", capacity)
        rows.append(_run_job(run_root, job))
        capacity = _capacity_snapshot(run_root)
        capacity["reason"] = f"after_job:{job['id']}"
        _append_jsonl(run_root / "capacity_checks.jsonl", capacity)
    if development and only_job is None:
        _development_gate(run_root)
    if not development and only_job is None:
        summarize_main(run_root)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir")
    parser.add_argument("--worker", default="manual")
    parser.add_argument("--job")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--development", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        result = smoke()
        print(json.dumps({"smoke": result}, indent=2, sort_keys=True))
        return
    run_root = arguments.log_dir
    if run_root is None:
        run_root = str(
            DEVELOPMENT_RUN_ROOT if arguments.development else DEFAULT_RUN_ROOT
        )
    rows = run_worker(
        run_root,
        only_job=arguments.job,
        development=arguments.development,
    )
    print(
        json.dumps(
            {
                "worker": arguments.worker,
                "jobs_completed": len(rows),
                "summaries": [
                    {
                        "job": row["job"],
                        "checkpoint": row["checkpoint"],
                    }
                    for row in rows
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
