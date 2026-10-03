#!/usr/bin/env python3
"""GC-PPO versus Dual LEO on TICK-CLAIM and PACK-RESTORE.

Workshop12 pretraining on the fixed kernel, then the same start branched into
fixed continued learning and mutant adaptation. The teacher is extra state:
its parameters, Adam steps, and wall clock are recorded apart from PPO.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

if os.environ.get("HACKRL_DEVICE", "cpu") != "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

from hackrl.dual_leo import (
    LEO_EPOCHS,
    LEO_MINIBATCH_SIZE,
    init_dual_leo_teacher,
    load_dual_checkpoint,
    make_dual_leo_update,
    save_dual_checkpoint,
    teacher_parameter_count,
)
from hackrl.pack_restore_gc import (
    NUM_ACTIONS as PACK_ACTIONS,
    NUM_GOALS as PACK_GOALS,
    PackRestoreGCActorCritic,
    PackRestoreGCConfig,
    checkpoint_files_present as pack_checkpoint_present,
    command_deliver_3 as pack_command,
    config_from_pack_restore_gc_payload,
    evaluate_pack_restore_gc_frozen,
    initialize_pack_restore_gc,
    load_pack_restore_gc_checkpoint,
    load_pack_restore_gc_history_branch,
    make_pack_restore_gc_update,
    pack_restore_gc_config_payload,
    pack_restore_gc_parameter_count,
    save_pack_restore_gc_checkpoint,
    step_pack_restore_gc_workers,
    _BRANCH_LOCKED_FIELDS as PACK_LOCKED,
    _batch_inputs as pack_inputs,
)
from hackrl.tick_claim_gc import (
    NUM_ACTIONS as TICK_ACTIONS,
    NUM_GOALS as TICK_GOALS,
    TickClaimGCActorCritic,
    TickClaimGCConfig,
    checkpoint_files_present as tick_checkpoint_present,
    command_deliver_3 as tick_command,
    config_from_tick_claim_gc_payload,
    evaluate_tick_claim_gc_frozen,
    initialize_tick_claim_gc,
    load_tick_claim_gc_checkpoint,
    load_tick_claim_gc_history_branch,
    make_tick_claim_gc_update,
    save_tick_claim_gc_checkpoint,
    step_tick_claim_gc_workers,
    tick_claim_gc_config_payload,
    tick_claim_gc_parameter_count,
    _BRANCH_LOCKED_FIELDS as TICK_LOCKED,
    _batch_inputs as tick_inputs,
)

ADAPT_UPDATES = 4096
PRETRAIN_UPDATES = 512
SEEDS = (20, 21, 22, 23, 24)
SCIENCE_UPDATES = (0, 32, 128, 256, 512, 1024, 2048, 4096)
SAVE_UPDATES = tuple(sorted(set(SCIENCE_UPDATES) | set(range(256, ADAPT_UPDATES + 1, 256))))
FAMILIES = ("natural_reset", "common_setup")
DISCOUNT = 0.995


def _tick_outcome(runner, actions, config):
    runner, event = step_tick_claim_gc_workers(runner, actions, config)
    return (
        runner,
        event.done_for_gae,
        event.valid_transition,
        event.reward,
        event.terminal_goals,
        event.world_done,
        event.goal_done,
        event.observed_goals,
    )


def _pack_outcome(runner, actions, config):
    runner, event = step_pack_restore_gc_workers(runner, actions, config)
    return (
        runner,
        event.done,
        event.valid,
        event.reward,
        event.terminal_goals,
        event.world_done,
        event.goal_done,
        event.observed_goals,
    )


ENVS = {
    "tick": {
        "config": TickClaimGCConfig,
        "actor": TickClaimGCActorCritic,
        "initialize": initialize_tick_claim_gc,
        "gc_update": make_tick_claim_gc_update,
        "save": save_tick_claim_gc_checkpoint,
        "load": load_tick_claim_gc_checkpoint,
        "branch": load_tick_claim_gc_history_branch,
        "command": tick_command,
        "evaluate": evaluate_tick_claim_gc_frozen,
        "inputs": tick_inputs,
        "goals": TICK_GOALS,
        "actions": TICK_ACTIONS,
        "payload": tick_claim_gc_config_payload,
        "from_payload": config_from_tick_claim_gc_payload,
        "present": tick_checkpoint_present,
        "parameters": tick_claim_gc_parameter_count,
        "locked": TICK_LOCKED,
        "outcome": _tick_outcome,
    },
    "pack": {
        "config": PackRestoreGCConfig,
        "actor": PackRestoreGCActorCritic,
        "initialize": initialize_pack_restore_gc,
        "gc_update": make_pack_restore_gc_update,
        "save": save_pack_restore_gc_checkpoint,
        "load": load_pack_restore_gc_checkpoint,
        "branch": load_pack_restore_gc_history_branch,
        "command": pack_command,
        "evaluate": evaluate_pack_restore_gc_frozen,
        "inputs": pack_inputs,
        "goals": PACK_GOALS,
        "actions": PACK_ACTIONS,
        "payload": pack_restore_gc_config_payload,
        "from_payload": config_from_pack_restore_gc_payload,
        "present": pack_checkpoint_present,
        "parameters": pack_restore_gc_parameter_count,
        "locked": PACK_LOCKED,
        "outcome": _pack_outcome,
    },
}


def build_jobs():
    jobs = []
    for env in ("tick", "pack"):
        for method in ("gc", "dual"):
            for seed in SEEDS:
                pretrain = f"{env}-{method}-pretrain-s{seed}"
                jobs.append(
                    {
                        "id": pretrain,
                        "kind": "pretrain",
                        "env": env,
                        "method": method,
                        "seed": seed,
                    }
                )
                for variant in ("fixed", "mutant"):
                    jobs.append(
                        {
                            "id": f"{env}-{method}-s{seed}-{variant}",
                            "kind": "adapt",
                            "env": env,
                            "method": method,
                            "seed": seed,
                            "variant": variant,
                            "depends_on": [pretrain],
                        }
                    )
    return jobs


def _config(env, *, seed, goal_mode, variant, updates):
    return ENVS[env]["config"](
        variant=variant,
        seed=int(seed),
        num_envs=512,
        num_steps=64,
        num_updates=int(updates),
        minibatch_size=1024,
        hidden_size=512,
        entropy_coefficient=0.005,
        goal_mode=goal_mode,
        checkpoint_updates=(),
    )


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _host_metrics(metrics, adaptation_update):
    host = {key: np.asarray(jax.device_get(value)).tolist() for key, value in metrics.items()}
    host["adaptation_update"] = int(adaptation_update)
    return host


def _append(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _meta_update(path):
    path = Path(path)
    if not (path / "metadata.json").is_file() or not (path / "state.msgpack").is_file():
        return None
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return int(meta.get("global_update", -1))


def _pretrain_dir(log_dir, job):
    return Path(log_dir) / job["env"] / job["method"] / "pretrain" / f"seed{job['seed']}"


def _cell_dir(log_dir, job):
    return Path(log_dir) / job["env"] / job["method"] / job["variant"] / f"seed{job['seed']}"


def _adapt_ckpt(cell, update):
    return Path(cell) / "checkpoints" / f"adapt_{update}"


def _curve(cell, update):
    return Path(cell) / "curve" / f"adapt_{update}.json"


def _family_return(records, family):
    rows = [row for row in records if row["family"] == family]
    values = []
    for row in rows:
        length = int(row["length"])
        success = 1.0 if row["success"] else 0.0
        values.append(success * (DISCOUNT ** (length - 1)) if length >= 1 else 0.0)
    return float(np.mean(values)) if values else None


def _evaluate(spec, network, params, *, variant, seed):
    views = {}
    for stochastic, repeats, name in ((False, 1, "mode"), (True, 4, "sample")):
        result = spec["evaluate"](
            network,
            params,
            variant=variant,
            stochastic=stochastic,
            repeats_per_state=repeats,
            seed_base=20000,
            learner_seed=int(seed),
            record_episodes=True,
        )
        views[name] = {}
        for family in FAMILIES:
            block = dict(result[family])
            block["mean_discounted_return"] = _family_return(result["episode_records"], family)
            views[name][family] = block
        views[name]["episodes"] = result["episode_records"]
    return views


def _curve_complete(path, update):
    path = Path(path)
    if not path.is_file():
        return False
    document = json.loads(path.read_text(encoding="utf-8"))
    if int(document.get("adaptation_updates", -1)) != int(update):
        return False
    for dynamics in ("fixed", "mutant"):
        block = document.get(dynamics)
        if not isinstance(block, dict):
            return False
        for name in ("mode", "sample"):
            view = block.get(name, {})
            for family in FAMILIES:
                if "success_rate" not in view.get(family, {}):
                    return False
                if "mean_discounted_return" not in view.get(family, {}):
                    return False
    return True


def _write_curve(cell, job, runner, update, network):
    destination = _curve(cell, update)
    if _curve_complete(destination, update):
        return
    _write_json(
        destination,
        {
            "schema_version": "hackrl_dual_leo_compare_curve_v1",
            "env": job["env"],
            "method": job["method"],
            "seed": int(job["seed"]),
            "trained_variant": job["variant"],
            "adaptation_updates": int(update),
            "global_update": int(runner.global_update),
            "discounted_return": "success * 0.995 ** (length - 1) for length >= 1",
            "fixed": _evaluate(ENVS[job["env"]], network, runner.train_state.params, variant="fixed", seed=job["seed"]),
            "mutant": _evaluate(ENVS[job["env"]], network, runner.train_state.params, variant="mutant", seed=job["seed"]),
        },
    )
    print(f"[curve] {destination}", flush=True)


def _start_teacher(spec, config, runner):
    inputs = spec["inputs"](runner.env_state, runner.current_goal)
    return init_dual_leo_teacher(config, inputs[0], inputs[1], spec["goals"], spec["actions"])


def _dual_update(spec, network, teacher, config, minibatch):
    return jax.jit(
        make_dual_leo_update(
            network, teacher, config, spec["outcome"], spec["inputs"], minibatch
        )
    )


def _check_branch(spec, source, branch):
    recorded = json.loads((Path(source) / "config.json").read_text(encoding="utf-8"))
    origin = spec["from_payload"](recorded)
    if origin.goal_mode != "workshop12" or branch.goal_mode != "deliver_3":
        raise RuntimeError(f"adaptation source is not workshop12: {source}")
    for name in spec["locked"]:
        if getattr(origin, name) != getattr(branch, name):
            raise RuntimeError(f"branch changed locked field {name}")


def _load_adapt_start(log_dir, job, template, branch):
    spec = ENVS[job["env"]]
    source = _pretrain_dir(log_dir, job) / "checkpoints" / f"update_{PRETRAIN_UPDATES}"
    _check_branch(spec, source, branch)
    if job["method"] == "gc":
        runner = spec["branch"](source, template, branch)
        leo = None
    else:
        _, leo_template, _ = _start_teacher(spec, branch, template)
        runner, leo = load_dual_checkpoint(source, template, leo_template)
    if int(runner.global_update) != PRETRAIN_UPDATES:
        raise RuntimeError(f"pretrain loaded at update {int(runner.global_update)}")
    before = tuple(np.asarray(jax.device_get(leaf)).tobytes() for leaf in jax.tree.leaves(runner.train_state.params))
    runner = spec["command"](runner)
    after = tuple(np.asarray(jax.device_get(leaf)).tobytes() for leaf in jax.tree.leaves(runner.train_state.params))
    if before != after:
        raise RuntimeError("command switch changed PPO parameters")
    return runner, leo


def _cell_finished(cell):
    final = _adapt_ckpt(cell, ADAPT_UPDATES)
    if _meta_update(final) != PRETRAIN_UPDATES + ADAPT_UPDATES:
        return False
    return all(_curve_complete(_curve(cell, update), update) for update in SCIENCE_UPDATES)


def _latest_adapt(cell):
    best = None
    for update in SAVE_UPDATES:
        found = _meta_update(_adapt_ckpt(cell, update))
        if found == PRETRAIN_UPDATES + int(update):
            best = int(update)
    return best


def _save_state(spec, method, directory, runner, leo, config):
    if method == "gc":
        spec["save"](directory, runner, config)
    else:
        save_dual_checkpoint(directory, runner, leo, spec["payload"](config))


def _load_state(spec, method, directory, template, leo_template, config):
    if method == "gc":
        return spec["load"](directory, template, config), None
    return load_dual_checkpoint(directory, template, leo_template)


def _train_loop(update, runner, leo, method, target, on_step):
    while int(runner.global_update) < target:
        started = time.perf_counter()
        if method == "gc":
            runner, metrics = update(runner)
        else:
            runner, leo, metrics = update(runner, leo)
        jax.block_until_ready(runner.global_update)
        on_step(runner, leo, metrics, time.perf_counter() - started)
    return runner, leo


def run_pretrain(log_dir, job):
    spec = ENVS[job["env"]]
    config = _config(job["env"], seed=job["seed"], goal_mode="workshop12", variant="fixed", updates=PRETRAIN_UPDATES)
    destination = _pretrain_dir(log_dir, job)
    final = destination / "checkpoints" / f"update_{PRETRAIN_UPDATES}"
    if _meta_update(final) == PRETRAIN_UPDATES:
        print(f"[skip] {destination}", flush=True)
        return
    network, runner = spec["initialize"](config)
    leo = None
    teacher_count = 0
    if job["method"] == "gc":
        update = jax.jit(spec["gc_update"](network, config))
    else:
        teacher, leo, minibatch = _start_teacher(spec, config, runner)
        teacher_count = teacher_parameter_count(leo)
        update = _dual_update(spec, network, teacher, config, minibatch)
    latest = None
    for update_index in (0, 256, PRETRAIN_UPDATES):
        path = destination / "checkpoints" / f"update_{update_index}"
        if _meta_update(path) == update_index:
            latest = update_index
    if latest is None:
        _save_state(spec, job["method"], destination / "checkpoints" / "update_0", runner, leo, config)
        latest = 0
    elif latest:
        template_network, template = spec["initialize"](config)
        del template_network
        leo_template = None
        if job["method"] == "dual":
            _, leo_template, _ = _start_teacher(spec, config, template)
        runner, leo = _load_state(
            spec, job["method"], destination / "checkpoints" / f"update_{latest}", template, leo_template, config
        )
    ppo_count = spec["parameters"](runner.train_state.params)
    started = time.perf_counter()
    step_seconds = []

    def on_step(current, current_leo, metrics, seconds):
        finished = int(current.global_update)
        step_seconds.append(seconds)
        if finished in (256, PRETRAIN_UPDATES):
            _save_state(spec, job["method"], destination / "checkpoints" / f"update_{finished}", current, current_leo, config)
        if finished % 32 == 0:
            host = _host_metrics(metrics, finished)
            print(
                f"[pretrain] {job['id']} step {finished}/{PRETRAIN_UPDATES} "
                f"seconds={seconds:.3f} valid_transitions={host.get('valid_transitions')}",
                flush=True,
            )

    runner, leo = _train_loop(update, runner, leo, job["method"], PRETRAIN_UPDATES, on_step)
    summary = {
        "env": job["env"],
        "method": job["method"],
        "seed": int(job["seed"]),
        "updates": PRETRAIN_UPDATES,
        "ppo_parameters": int(ppo_count),
        "teacher_parameters": int(teacher_count),
        "ppo_applied_grad_steps": int(runner.train_state.step),
        "ppo_scheduled_grad_steps": int(runner.global_update)
        * config.num_minibatches
        * config.update_epochs,
        "seconds": time.perf_counter() - started,
        "seconds_per_update": float(np.mean(step_seconds)) if step_seconds else None,
    }
    if leo is not None:
        teacher_per_update = (config.batch_size // LEO_MINIBATCH_SIZE) * LEO_EPOCHS
        summary["teacher_applied_grad_steps"] = int(leo.step)
        summary["teacher_scheduled_grad_steps"] = int(runner.global_update) * teacher_per_update
    _write_json(destination / "summary.json", summary)


def run_adapt(log_dir, job):
    spec = ENVS[job["env"]]
    cell = _cell_dir(log_dir, job)
    if _cell_finished(cell):
        print(f"[skip] {cell}", flush=True)
        return
    branch = _config(job["env"], seed=job["seed"], goal_mode="deliver_3", variant=job["variant"], updates=ADAPT_UPDATES)
    network, template = spec["initialize"](branch)
    update = None
    leo_template = None
    teacher_count = 0
    if job["method"] == "dual":
        teacher, leo_template, minibatch = _start_teacher(spec, branch, template)
        teacher_count = teacher_parameter_count(leo_template)
        update = _dual_update(spec, network, teacher, branch, minibatch)
    else:
        update = jax.jit(spec["gc_update"](network, branch))
    saved = _latest_adapt(cell)
    if saved is None:
        runner, leo = _load_adapt_start(log_dir, job, template, branch)
        _save_state(spec, job["method"], _adapt_ckpt(cell, 0), runner, leo, branch)
        saved = 0
    else:
        runner, leo = _load_state(spec, job["method"], _adapt_ckpt(cell, saved), template, leo_template, branch)
    if saved in SCIENCE_UPDATES:
        _write_curve(cell, job, runner, saved, network)
    ppo_count = spec["parameters"](runner.train_state.params)
    started = time.perf_counter()
    step_seconds = []

    def on_step(current, current_leo, metrics, seconds):
        finished = int(current.global_update) - PRETRAIN_UPDATES
        step_seconds.append(seconds)
        _append(cell / "updates.jsonl", _host_metrics(metrics, finished) | {"seconds": seconds})
        if finished in SAVE_UPDATES:
            _save_state(spec, job["method"], _adapt_ckpt(cell, finished), current, current_leo, branch)
        if finished in SCIENCE_UPDATES:
            _write_curve(cell, job, current, finished, network)
        if finished in SCIENCE_UPDATES or finished % 32 == 0:
            host = _host_metrics(metrics, finished)
            extra = ""
            if job["method"] == "dual":
                extra = (
                    f" bc={host.get('bc_policy_coef')} teacher_loss={host.get('teacher_td_loss')} "
                    f"teacher_steps={host.get('teacher_grad_steps')}"
                )
            print(
                f"[adapt] {job['id']} step {finished}/{ADAPT_UPDATES} seconds={seconds:.3f} "
                f"valid_transitions={host.get('valid_transitions')} "
                f"goal_successes={host.get('goal_successes')}{extra}",
                flush=True,
            )

    runner, leo = _train_loop(update, runner, leo, job["method"], PRETRAIN_UPDATES + ADAPT_UPDATES, on_step)
    summary = {
        "env": job["env"],
        "method": job["method"],
        "variant": job["variant"],
        "seed": int(job["seed"]),
        "ppo_parameters": int(ppo_count),
        "teacher_parameters": int(teacher_count),
        "ppo_applied_grad_steps": int(runner.train_state.step),
        "ppo_scheduled_grad_steps": int(runner.global_update)
        * branch.num_minibatches
        * branch.update_epochs,
        "global_update": int(runner.global_update),
        "adaptation_updates": ADAPT_UPDATES,
        "seconds_this_process": time.perf_counter() - started,
        "seconds_per_update_this_process": float(np.mean(step_seconds)) if step_seconds else None,
    }
    if leo is not None:
        teacher_per_update = (branch.batch_size // LEO_MINIBATCH_SIZE) * LEO_EPOCHS
        summary["teacher_applied_grad_steps"] = int(leo.step)
        summary["teacher_scheduled_grad_steps"] = int(runner.global_update) * teacher_per_update
    _write_json(cell / "summary.json", summary)
    print(f"[done] {cell}", flush=True)


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reclaim(log_dir):
    claims = Path(log_dir) / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    for path in claims.iterdir():
        try:
            pid = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if not _pid_alive(pid):
            path.unlink(missing_ok=True)


def _pretrain_complete(log_dir, job):
    return _meta_update(_pretrain_dir(log_dir, job) / "checkpoints" / f"update_{PRETRAIN_UPDATES}") == PRETRAIN_UPDATES


def _job_complete(log_dir, job):
    if job["kind"] == "pretrain":
        return _pretrain_complete(log_dir, job)
    return _cell_finished(_cell_dir(log_dir, job))


def _ready(log_dir, job):
    lookup = {item["id"]: item for item in build_jobs()}
    return all(_job_complete(log_dir, lookup[name]) for name in job.get("depends_on", []))


def _claim(log_dir, job_id):
    path = Path(log_dir) / "claims" / job_id
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def _release(log_dir, job_id):
    (Path(log_dir) / "claims" / job_id).unlink(missing_ok=True)


def measure(env, updates):
    """Time the real 512 by 64 update used by the queue."""

    spec = ENVS[env]
    config = _config(env, seed=20, goal_mode="workshop12", variant="fixed", updates=updates)
    network, runner = spec["initialize"](config)
    teacher, leo, minibatch = _start_teacher(spec, config, runner)
    update = _dual_update(spec, network, teacher, config, minibatch)
    samples = []
    for _ in range(int(updates)):
        started = time.perf_counter()
        runner, leo, metrics = update(runner, leo)
        jax.block_until_ready((runner.global_update, leo.step, metrics["teacher_td_loss"]))
        samples.append(time.perf_counter() - started)
        print(
            f"[measure] {env} dual update {int(runner.global_update)} seconds={samples[-1]:.3f} "
            f"teacher_steps={int(leo.step)} bc={float(metrics['bc_policy_coef']):.6f}",
            flush=True,
        )
    steady = samples[1:] if len(samples) > 1 else samples
    print(
        json.dumps(
            {
                "env": env,
                "method": "dual",
                "updates": int(updates),
                "compile_and_first_seconds": samples[0],
                "steady_seconds_per_update": float(np.mean(steady)),
                "ppo_parameters": spec["parameters"](runner.train_state.params),
                "teacher_parameters": teacher_parameter_count(leo),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir")
    parser.add_argument("--worker")
    parser.add_argument("--measure-env", choices=("tick", "pack"))
    parser.add_argument("--measure-updates", type=int, default=0)
    args = parser.parse_args()
    if args.measure_updates:
        measure(args.measure_env or "tick", args.measure_updates)
        return
    if not args.log_dir or not args.worker:
        raise SystemExit("queue mode needs --log-dir and --worker")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    print(f"[worker] {args.worker} jobs={len(jobs)}", flush=True)
    while True:
        _reclaim(log_dir)
        chosen = None
        for job in jobs:
            if _job_complete(log_dir, job) or not _ready(log_dir, job):
                continue
            if _claim(log_dir, job["id"]):
                chosen = job
                break
        if chosen is None:
            if all(_job_complete(log_dir, job) for job in jobs):
                print(f"[worker] {args.worker} sweep complete", flush=True)
                return
            time.sleep(15)
            continue
        try:
            print(f"[job] {chosen['id']}", flush=True)
            if chosen["kind"] == "pretrain":
                run_pretrain(log_dir, chosen)
            else:
                run_adapt(log_dir, chosen)
        finally:
            _release(log_dir, chosen["id"])


if __name__ == "__main__":
    main()
