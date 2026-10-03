"""CRAFT-REMAIN GC-PPO vs Dual LEO. Same contract as the frozen 9155a0d comparison."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import jax
import numpy as np

from hackrl.craft_remain import GROWTH_PERIOD
from hackrl.craft_remain_gc import (
    ADAPT_UPDATES,
    COMPARE_SEEDS,
    PRETRAIN_UPDATES,
    SCIENCE_UPDATES,
    _BRANCH_LOCKED_FIELDS,
    CraftRemainGCActorCritic,
    CraftRemainGCConfig,
    command_deliver_3,
    config_from_craft_remain_gc_payload,
    craft_remain_gc_config_payload,
    craft_remain_gc_parameter_count,
    craft_remain_outcome,
    evaluate_craft_remain_gc_frozen,
    initialize_craft_remain_gc,
    load_craft_remain_gc_checkpoint,
    load_craft_remain_gc_history_branch,
    make_craft_remain_gc_update,
    save_craft_remain_gc_checkpoint,
)
from hackrl.craft_remain_gc import NUM_ACTIONS, NUM_GOALS, _batch_inputs
from hackrl.dual_leo import (
    LEO_EPOCHS,
    LEO_MINIBATCH_SIZE,
    init_dual_leo_teacher,
    load_dual_checkpoint,
    make_dual_leo_update,
    save_dual_checkpoint,
    teacher_parameter_count,
)

DISCOUNT = 0.995
SAVE_UPDATES = tuple(sorted(set(SCIENCE_UPDATES) | set(range(256, ADAPT_UPDATES + 1, 256))))
FAMILIES = ("natural_reset", "path_check")
SPEC = {
    "config": CraftRemainGCConfig,
    "actor": CraftRemainGCActorCritic,
    "initialize": initialize_craft_remain_gc,
    "gc_update": make_craft_remain_gc_update,
    "save": save_craft_remain_gc_checkpoint,
    "load": load_craft_remain_gc_checkpoint,
    "branch": load_craft_remain_gc_history_branch,
    "command": command_deliver_3,
    "evaluate": evaluate_craft_remain_gc_frozen,
    "inputs": _batch_inputs,
    "goals": NUM_GOALS,
    "actions": NUM_ACTIONS,
    "payload": craft_remain_gc_config_payload,
    "from_payload": config_from_craft_remain_gc_payload,
    "parameters": craft_remain_gc_parameter_count,
    "locked": _BRANCH_LOCKED_FIELDS,
    "outcome": craft_remain_outcome,
}


def build_jobs():
    jobs = []
    for method in ("gc", "dual"):
        for seed in COMPARE_SEEDS:
            pretrain = f"craft-{method}-pretrain-s{seed}"
            jobs.append({"id": pretrain, "kind": "pretrain", "method": method, "seed": seed})
            for variant in ("fixed", "mutant"):
                jobs.append(
                    {
                        "id": f"craft-{method}-s{seed}-{variant}",
                        "kind": "adapt",
                        "method": method,
                        "seed": seed,
                        "variant": variant,
                        "depends_on": [pretrain],
                    }
                )
    return jobs


def _config(*, seed, goal_mode, variant, updates):
    return CraftRemainGCConfig(
        variant=variant,
        seed=int(seed),
        num_envs=512,
        num_steps=64,
        num_updates=int(updates),
        minibatch_size=1024,
        hidden_size=512,
        entropy_coefficient=0.005,
        goal_mode=goal_mode,
        growth_period=GROWTH_PERIOD,
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
    return Path(log_dir) / job["method"] / "pretrain" / f"seed{job['seed']}"


def _cell_dir(log_dir, job):
    return Path(log_dir) / job["method"] / job["variant"] / f"seed{job['seed']}"


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


def _evaluate(network, params, *, variant, seed):
    views = {}
    for stochastic, repeats, name in ((False, 1, "mode"), (True, 4, "sample")):
        result = SPEC["evaluate"](
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
                family_block = view.get(family, {})
                for key in ("success_rate", "mean_length", "mean_discounted_return", "trigger_rate", "exploit_rate"):
                    if key not in family_block:
                        return False
    return True


def _write_curve(cell, job, runner, update, network):
    destination = _curve(cell, update)
    if _curve_complete(destination, update):
        return
    _write_json(
        destination,
        {
            "schema_version": "hackrl_craft_remain_compare_curve_v1",
            "method": job["method"],
            "seed": int(job["seed"]),
            "trained_variant": job["variant"],
            "adaptation_updates": int(update),
            "global_update": int(runner.global_update),
            "growth_period": GROWTH_PERIOD,
            "discounted_return": "success * 0.995 ** (length - 1) for length >= 1",
            "reading": (
                "adaptation_updates 0 is immediate transfer of the pretrained policy. "
                "A later increase is the adaptation effect. The final exploit rate alone is not discovery. "
                "Path-check lengths 37 and 24 are the ripe-setup script bound, not this policy score."
            ),
            "fixed": _evaluate(network, runner.train_state.params, variant="fixed", seed=job["seed"]),
            "mutant": _evaluate(network, runner.train_state.params, variant="mutant", seed=job["seed"]),
        },
    )
    print(f"[curve] {destination}", flush=True)


def _start_teacher(config, runner):
    inputs = SPEC["inputs"](runner.env_state, runner.current_goal)
    return init_dual_leo_teacher(config, inputs[0], inputs[1], SPEC["goals"], SPEC["actions"])


def _dual_update(network, teacher, config, minibatch):
    return jax.jit(
        make_dual_leo_update(network, teacher, config, SPEC["outcome"], SPEC["inputs"], minibatch)
    )


def _check_branch(source, branch):
    recorded = json.loads((Path(source) / "config.json").read_text(encoding="utf-8"))
    origin = SPEC["from_payload"](recorded)
    if origin.goal_mode != "workshop12" or branch.goal_mode != "deliver_3":
        raise RuntimeError(f"adaptation source is not workshop12: {source}")
    for name in SPEC["locked"]:
        if getattr(origin, name) != getattr(branch, name):
            raise RuntimeError(f"branch changed locked field {name}")


def _load_adapt_start(log_dir, job, template, branch):
    source = _pretrain_dir(log_dir, job) / "checkpoints" / f"update_{PRETRAIN_UPDATES}"
    _check_branch(source, branch)
    if job["method"] == "gc":
        runner = SPEC["branch"](source, template, branch)
        leo = None
    else:
        _, leo_template, _ = _start_teacher(branch, template)
        runner, leo = load_dual_checkpoint(source, template, leo_template)
    if int(runner.global_update) != PRETRAIN_UPDATES:
        raise RuntimeError(f"pretrain loaded at update {int(runner.global_update)}")
    before = tuple(np.asarray(jax.device_get(leaf)).tobytes() for leaf in jax.tree.leaves(runner.train_state.params))
    runner = SPEC["command"](runner)
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


def _save_state(method, directory, runner, leo, config):
    if method == "gc":
        SPEC["save"](directory, runner, config)
    else:
        save_dual_checkpoint(directory, runner, leo, SPEC["payload"](config))


def _load_state(method, directory, template, leo_template, config):
    if method == "gc":
        return SPEC["load"](directory, template, config), None
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
    config = _config(seed=job["seed"], goal_mode="workshop12", variant="fixed", updates=PRETRAIN_UPDATES)
    destination = _pretrain_dir(log_dir, job)
    final = destination / "checkpoints" / f"update_{PRETRAIN_UPDATES}"
    if _meta_update(final) == PRETRAIN_UPDATES:
        print(f"[skip] {destination}", flush=True)
        return
    network, runner = SPEC["initialize"](config)
    leo = None
    teacher_count = 0
    if job["method"] == "gc":
        update = jax.jit(SPEC["gc_update"](network, config))
    else:
        teacher, leo, minibatch = _start_teacher(config, runner)
        teacher_count = teacher_parameter_count(leo)
        update = _dual_update(network, teacher, config, minibatch)
    latest = None
    for update_index in (0, 256, PRETRAIN_UPDATES):
        path = destination / "checkpoints" / f"update_{update_index}"
        if _meta_update(path) == update_index:
            latest = update_index
    if latest is None:
        _save_state(job["method"], destination / "checkpoints" / "update_0", runner, leo, config)
        latest = 0
    elif latest:
        _template_network, template = SPEC["initialize"](config)
        del _template_network
        leo_template = None
        if job["method"] == "dual":
            _, leo_template, _ = _start_teacher(config, template)
        runner, leo = _load_state(
            job["method"], destination / "checkpoints" / f"update_{latest}", template, leo_template, config
        )
    ppo_count = SPEC["parameters"](runner.train_state.params)
    started = time.perf_counter()
    step_seconds = []

    def on_step(current, current_leo, metrics, seconds):
        finished = int(current.global_update)
        step_seconds.append(seconds)
        if finished in (256, PRETRAIN_UPDATES):
            _save_state(job["method"], destination / "checkpoints" / f"update_{finished}", current, current_leo, config)
        if finished % 32 == 0:
            host = _host_metrics(metrics, finished)
            print(
                f"[pretrain] {job['id']} step {finished}/{PRETRAIN_UPDATES} "
                f"seconds={seconds:.3f} valid_transitions={host.get('valid_transitions')}",
                flush=True,
            )

    runner, leo = _train_loop(update, runner, leo, job["method"], PRETRAIN_UPDATES, on_step)
    summary = {
        "method": job["method"],
        "seed": int(job["seed"]),
        "updates": PRETRAIN_UPDATES,
        "growth_period": GROWTH_PERIOD,
        "ppo_parameters": int(ppo_count),
        "teacher_parameters": int(teacher_count),
        "ppo_applied_grad_steps": int(runner.train_state.step),
        "ppo_scheduled_grad_steps": int(runner.global_update) * config.num_minibatches * config.update_epochs,
        "seconds": time.perf_counter() - started,
        "seconds_per_update": float(np.mean(step_seconds)) if step_seconds else None,
    }
    if leo is not None:
        teacher_per_update = (config.batch_size // LEO_MINIBATCH_SIZE) * LEO_EPOCHS
        summary["teacher_applied_grad_steps"] = int(leo.step)
        summary["teacher_scheduled_grad_steps"] = int(runner.global_update) * teacher_per_update
    _write_json(destination / "summary.json", summary)


def run_adapt(log_dir, job):
    cell = _cell_dir(log_dir, job)
    if _cell_finished(cell):
        print(f"[skip] {cell}", flush=True)
        return
    branch = _config(seed=job["seed"], goal_mode="deliver_3", variant=job["variant"], updates=ADAPT_UPDATES)
    network, template = SPEC["initialize"](branch)
    leo_template = None
    teacher_count = 0
    if job["method"] == "dual":
        teacher, leo_template, minibatch = _start_teacher(branch, template)
        teacher_count = teacher_parameter_count(leo_template)
        update = _dual_update(network, teacher, branch, minibatch)
    else:
        update = jax.jit(SPEC["gc_update"](network, branch))
    saved = _latest_adapt(cell)
    if saved is None:
        runner, leo = _load_adapt_start(log_dir, job, template, branch)
        _save_state(job["method"], _adapt_ckpt(cell, 0), runner, leo, branch)
        saved = 0
    else:
        runner, leo = _load_state(job["method"], _adapt_ckpt(cell, saved), template, leo_template, branch)
    if saved in SCIENCE_UPDATES:
        _write_curve(cell, job, runner, saved, network)
    ppo_count = SPEC["parameters"](runner.train_state.params)
    started = time.perf_counter()
    step_seconds = []

    def on_step(current, current_leo, metrics, seconds):
        finished = int(current.global_update) - PRETRAIN_UPDATES
        step_seconds.append(seconds)
        _append(cell / "updates.jsonl", _host_metrics(metrics, finished) | {"seconds": seconds})
        if finished in SAVE_UPDATES:
            _save_state(job["method"], _adapt_ckpt(cell, finished), current, current_leo, branch)
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
        "method": job["method"],
        "variant": job["variant"],
        "seed": int(job["seed"]),
        "growth_period": GROWTH_PERIOD,
        "ppo_parameters": int(ppo_count),
        "teacher_parameters": int(teacher_count),
        "ppo_applied_grad_steps": int(runner.train_state.step),
        "ppo_scheduled_grad_steps": int(runner.global_update) * branch.num_minibatches * branch.update_epochs,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir")
    parser.add_argument("--worker")
    args = parser.parse_args()
    if not args.log_dir or not args.worker:
        raise SystemExit("queue mode needs --log-dir and --worker")
    jobs = build_jobs()
    if len(jobs) != 30:
        raise SystemExit(f"expected 30 jobs, found {len(jobs)}")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
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
