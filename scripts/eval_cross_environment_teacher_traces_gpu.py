#!/usr/bin/env python3
"""GPU-batched full-trace evaluator for common teacher matrix v1.

The scientific contract matches eval_cross_environment_teacher_traces.py, but
all episodes in one task are rolled out together with vmap + lax.scan. Complete
step data are transferred to the host once, then serialized and independently
replayed using the frozen trace verifier.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import os
from pathlib import Path
import time
from typing import Any

if os.environ.get("HACKRL_DEVICE", "cuda") == "cuda":
    os.environ.setdefault("JAX_PLATFORMS", "cuda")

import distrax
import jax
import jax.numpy as jnp
import numpy as np

import eval_cross_environment_teacher_traces as base
from hackrl.cross_environment_teacher_matrix import DEFAULT_MATRIX, load_matrix, sha256_file
from hackrl.cross_environment_teacher_trace import (
    ACTION_ENUMS,
    DISCOUNT,
    HORIZON,
    TRACE_SCHEMA_VERSION,
    build_panel_manifest,
    start_states,
    verify_trace_file,
)
from hackrl.craft_remain_env import (
    CraftRemainAction,
    CraftRemainVariant,
    craft_remain_goal_vector,
    craft_remain_step,
    craft_remain_world_done,
    observe_craft_remain,
    physical_total as craft_physical_total,
)
from hackrl.pack_restore import (
    PackRestoreAction,
    PackRestoreVariant,
    pack_restore_goal_vector,
    pack_restore_step,
    pack_restore_world_done,
    physical_grain_total,
    same_action_effect_gap,
)
from hackrl.pack_restore_gc import _trigger_ready as pack_legacy_opportunity
from hackrl.tick_claim import (
    TickClaimAction,
    TickClaimVariant,
    observe_tick_claim,
    tick_claim_goal_vector,
    tick_claim_step_with_snapshot,
    tick_claim_world_done,
)
from hackrl.tick_claim_gc import _opportunity_exposure as tick_legacy_opportunity
from hackrl.tick_claim_oracle import (
    audit_tick_claim_transition,
    compare_tick_claim_effect,
    initial_tick_claim_oracle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "docs/manifests/cross_environment_teacher_trace_panel_v1.json"
DEFAULT_OUTPUT = Path("/raid/ext_csv/HackRL/evaluations/cross_environment_teacher_matrix_v1_gpu")
BACKEND_VERSION = "hackrl_cross_environment_teacher_gpu_batched_scan_v1"
TRACE_GZIP_LEVEL = 1
MAX_BUFFERED_TRACE_CHARS = 512 * 1024 * 1024
_VALIDATED_PANELS: dict[str, dict] = {}
_ROLLOUT_CACHE: dict[tuple, Any] = {}


class _BufferedGzipText:
    """Collect one task in memory, then compress it in one write."""

    def __init__(self, path: Path, compresslevel: int):
        self.path = path
        self.compresslevel = compresslevel
        self.parts: list[str] = []
        self.buffered_chars = 0

    def __enter__(self):
        return self

    def write(self, value: str) -> None:
        self.buffered_chars += len(value)
        if self.buffered_chars > MAX_BUFFERED_TRACE_CHARS:
            raise RuntimeError(
                f"trace buffer exceeds {MAX_BUFFERED_TRACE_CHARS} characters"
            )
        self.parts.append(value)

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None:
            with gzip.open(
                self.path, "wt", encoding="utf-8", compresslevel=self.compresslevel
            ) as handle:
                handle.write("".join(self.parts))
        self.parts.clear()
        return False


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tree_where(mask, when_true, when_false):
    def choose(a, b):
        shaped = mask.reshape(mask.shape + (1,) * (a.ndim - mask.ndim))
        return jnp.where(shaped, a, b)
    return jax.tree.map(choose, when_true, when_false)


def _broadcast_tree(tree, count: int):
    return jax.tree.map(
        lambda value: jnp.broadcast_to(value, (count,) + value.shape), tree
    )


def _stack_initial(environment: str, family: str, repeats: int):
    starts = start_states(environment, family)
    states = jax.tree.map(
        lambda *values: jnp.stack(values, axis=0),
        *[row["state"] for row in starts],
    )
    states = jax.tree.map(
        lambda value: jnp.repeat(value, repeats, axis=0), states
    )
    metadata = []
    for row in starts:
        for repeat in range(repeats):
            metadata.append({
                "id": row["id"],
                "state_index": int(row["state_index"]),
                "layout": row["layout"],
                "phase": int(row["phase"]),
                "repeat": repeat,
            })
    return states, metadata


def _physical_total(environment: str, state):
    if environment == "tick_claim":
        return state.grain + state.delivered_total
    if environment == "pack_restore":
        return physical_grain_total(state)
    return craft_physical_total(state)


def _observe(environment: str, state):
    if environment == "tick_claim":
        return observe_tick_claim(state)
    if environment == "pack_restore":
        # PACK's actor observes the state through its registered input adapter;
        # the public trace contract stores the public state observation helper.
        from hackrl.pack_restore import observe_pack_restore
        return observe_pack_restore(state)
    return observe_craft_remain(state)


def _goal_vector(environment: str, state):
    if environment == "tick_claim":
        return tick_claim_goal_vector(observe_tick_claim(state))
    if environment == "pack_restore":
        return pack_restore_goal_vector(state)
    return craft_remain_goal_vector(state)


def _world_done(environment: str, state):
    if environment == "tick_claim":
        return tick_claim_world_done(state)
    if environment == "pack_restore":
        return pack_restore_world_done(state)
    return craft_remain_world_done(state)


def _step(environment: str, state, action, variant: str):
    if environment == "tick_claim":
        return tick_claim_step_with_snapshot(state, action, TickClaimVariant(variant))
    if environment == "pack_restore":
        return pack_restore_step(state, action, PackRestoreVariant(variant)), None
    return craft_remain_step(state, action, CraftRemainVariant(variant)), None


def _legal_effect(environment: str, state, action):
    if environment == "tick_claim":
        fixed, _ = tick_claim_step_with_snapshot(state, action, TickClaimVariant.FIXED)
        mutant, _ = tick_claim_step_with_snapshot(state, action, TickClaimVariant.MUTANT)
        return compare_tick_claim_effect(state, action, fixed, mutant).effect
    if environment == "pack_restore":
        return same_action_effect_gap(state, action) > 0
    fixed = craft_remain_step(state, action, CraftRemainVariant.FIXED)
    mutant = craft_remain_step(state, action, CraftRemainVariant.MUTANT)
    return craft_physical_total(mutant) > craft_physical_total(fixed)


def _legacy_opportunity(environment: str, state):
    if environment == "tick_claim":
        return tick_legacy_opportunity(state)
    if environment == "pack_restore":
        return pack_legacy_opportunity(state)
    return jnp.asarray(False)


def _noop(environment: str) -> int:
    return {
        "tick_claim": int(TickClaimAction.NOOP),
        "pack_restore": int(PackRestoreAction.NOOP),
        "craft_remain": int(CraftRemainAction.NOOP),
    }[environment]


def _parameter_signature(params) -> tuple:
    leaves, treedef = jax.tree_util.tree_flatten(params)
    return str(treedef), tuple((tuple(x.shape), str(x.dtype)) for x in leaves)


def _build_rollout(environment: str, spec, network, variant: str, view: str, episode_count: int):
    stochastic = view == "sample"
    action_values = jnp.asarray([int(a) for a in ACTION_ENUMS[environment]], dtype=jnp.int32)
    is_mutant = variant == "mutant"
    noop = _noop(environment)

    batch_observe = jax.vmap(lambda state: _observe(environment, state))
    batch_goal = jax.vmap(lambda state: _goal_vector(environment, state))
    batch_done = jax.vmap(lambda state: _world_done(environment, state))
    batch_total = jax.vmap(lambda state: _physical_total(environment, state))
    batch_legacy = jax.vmap(lambda state: _legacy_opportunity(environment, state))
    batch_step = jax.vmap(lambda state, action: _step(environment, state, action, variant))
    batch_legal = jax.vmap(
        lambda state: jax.vmap(lambda action: _legal_effect(environment, state, action))(
            action_values
        )
    )

    def rollout(params, initial_state, initial_keys, goal_index):
        zeros_bool = jnp.zeros((episode_count,), dtype=jnp.bool_)
        zeros_int = jnp.zeros((episode_count,), dtype=jnp.int32)
        oracle = _broadcast_tree(initial_tick_claim_oracle(), episode_count)

        def one_step(carry, step_index):
            (
                state,
                action_keys,
                done,
                success,
                length,
                oracle_state,
                defect_balance,
                defect_delivered,
                triggered,
                retained_recovered,
                excess_delivery,
            ) = carry
            active = jnp.logical_not(done)
            observations = batch_observe(state)
            goals = jnp.full((episode_count,), goal_index, dtype=jnp.int32)
            policy, _ = network.apply(params, *spec["inputs"](state, goals))
            split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(action_keys)
            next_keys = split_keys[:, 0]
            draw_keys = split_keys[:, 1]
            if stochastic:
                actions = jax.vmap(
                    lambda logits, key: distrax.Categorical(logits=logits).sample(seed=key)
                )(policy.logits, draw_keys)
            else:
                actions = jnp.argmax(policy.logits, axis=-1)
            actions = jnp.where(active, actions, noop)
            legal_mask = batch_legal(state)
            legacy = batch_legacy(state)
            stepped, auxiliary = batch_step(state, actions)
            totals_before = batch_total(state)
            totals_after = batch_total(stepped)
            delivered_delta = jnp.where(
                active, stepped.delivered_total - state.delivered_total, 0
            )

            if environment == "tick_claim":
                audits = jax.vmap(audit_tick_claim_transition)(
                    oracle_state, state, actions, stepped, auxiliary
                )
                same_cycle = jnp.logical_and(
                    oracle_state.initialized,
                    jnp.logical_and(
                        oracle_state.object_id == state.crop_object_id,
                        jnp.logical_and(
                            oracle_state.generation == state.crop_generation,
                            oracle_state.cycle_id == state.crop_cycle_id,
                        ),
                    ),
                )
                prior_payout = jnp.where(same_cycle, oracle_state.payout_total, 0)
                created = jnp.maximum(
                    audits.cycle_payout_total - jnp.maximum(prior_payout, 1), 0
                )
                next_oracle = _tree_where(active, audits.oracle_state, oracle_state)
                retained = zeros_bool
                excess = zeros_bool
            elif environment == "pack_restore":
                created = jnp.maximum(totals_after - totals_before, 0)
                next_oracle = oracle_state
                retained = zeros_bool
                excess = zeros_bool
            else:
                growth = (stepped.source_grain > state.source_grain).astype(jnp.int32)
                created = jnp.maximum(totals_after - totals_before - growth, 0)
                next_oracle = oracle_state
                retained = jnp.logical_and(
                    triggered,
                    jnp.logical_and(
                        jnp.logical_or(
                            actions == int(CraftRemainAction.TAKE_A),
                            actions == int(CraftRemainAction.TAKE_B),
                        ),
                        jnp.logical_and(
                            stepped.carried_grain > state.carried_grain,
                            jnp.logical_or(
                                stepped.slot_a < state.slot_a,
                                stepped.slot_b < state.slot_b,
                            ),
                        ),
                    ),
                )
                excess = jnp.logical_and(
                    delivered_delta > 0,
                    stepped.delivered_total > stepped.harvested_count,
                )

            created = jnp.where(active, created, 0)
            trigger_now = jnp.logical_and(active, jnp.logical_and(created > 0, is_mutant))
            retained = jnp.logical_and(active, retained)
            excess = jnp.logical_and(active, excess)
            balance_before_delivery = defect_balance + created
            attributed_delivery = jnp.minimum(
                jnp.maximum(delivered_delta, 0), balance_before_delivery
            )
            next_balance = jnp.where(
                active, balance_before_delivery - attributed_delivery, defect_balance
            )
            next_defect_delivered = defect_delivered + jnp.where(
                active, attributed_delivery, 0
            )
            next_triggered = jnp.logical_or(triggered, trigger_now)
            next_retained = jnp.logical_or(retained_recovered, retained)
            next_excess = jnp.logical_or(excess_delivery, excess)

            achieved = batch_goal(stepped)[:, goal_index]
            ended = batch_done(stepped)
            just_success = jnp.logical_and(active, achieved)
            just_done = jnp.logical_and(active, jnp.logical_or(achieved, ended))
            next_done = jnp.logical_or(done, just_done)
            next_success = jnp.logical_or(success, just_success)
            next_length = length + active.astype(jnp.int32)
            next_state = _tree_where(active, stepped, state)

            trace = {
                "active": active,
                "state_before": state,
                "observation": observations,
                "action": actions,
                "logits": policy.logits,
                "legal_mask": legal_mask,
                "legacy": legacy,
                "state_after": stepped,
                "success_after": achieved,
                "world_done_after": ended,
                "physical_resource_before": totals_before,
                "physical_resource_after": totals_after,
                "defect_created_amount": created,
                "delivered_delta": delivered_delta,
                "defect_attributed_delivery": attributed_delivery,
                "actual_defect_trigger": trigger_now,
                "retained_resource_recovered": retained,
                "excess_delivery_event": excess,
            }
            next_carry = (
                next_state,
                next_keys,
                next_done,
                next_success,
                next_length,
                next_oracle,
                next_balance,
                next_defect_delivered,
                next_triggered,
                next_retained,
                next_excess,
            )
            return next_carry, trace

        initial_carry = (
            initial_state,
            initial_keys,
            zeros_bool,
            zeros_bool,
            zeros_int,
            oracle,
            zeros_int,
            zeros_int,
            zeros_bool,
            zeros_bool,
            zeros_bool,
        )
        final, trace = jax.lax.scan(
            one_step, initial_carry, jnp.arange(HORIZON, dtype=jnp.int32)
        )
        return final, trace

    return jax.jit(rollout)


def _compiled_rollout(environment: str, spec, network, params, variant: str, view: str, episode_count: int):
    key = (
        environment,
        variant,
        view,
        episode_count,
        repr(network),
        _parameter_signature(params),
    )
    compiled = _ROLLOUT_CACHE.get(key)
    if compiled is None:
        compiled = _build_rollout(environment, spec, network, variant, view, episode_count)
        _ROLLOUT_CACHE[key] = compiled
    return compiled


def _validated_panel(path: Path) -> dict:
    digest = sha256_file(path)
    if digest not in _VALIDATED_PANELS:
        frozen = json.loads(path.read_text())
        current = build_panel_manifest()
        if frozen != current:
            raise RuntimeError("frozen evaluation panel differs from current generators")
        _VALIDATED_PANELS[digest] = frozen
    return _VALIDATED_PANELS[digest]


def _public_at(value: Any, *index: int) -> Any:
    """Convert one already-host-resident tree element without rebuilding a pytree."""
    if dataclasses.is_dataclass(value):
        return {
            field.name: _public_at(getattr(value, field.name), *index)
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _public_at(item, *index) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_public_at(item, *index) for item in value]
    item = np.asarray(value)[index]
    return item.item() if item.ndim == 0 else item.tolist()


def evaluate(
    *,
    environment: str,
    checkpoint: Path,
    checkpoint_origin: str,
    learner_seed: int,
    variant: str,
    view: str,
    family: str,
    matrix_path: Path = DEFAULT_MATRIX,
    panel_path: Path = DEFAULT_PANEL,
    output_root: Path = DEFAULT_OUTPUT,
    scientific: bool = True,
    actor_bundle=None,
    checkpoint_fingerprint=None,
    policy_id: str | None = None,
    max_starts: int | None = None,
    horizon: int = HORIZON,
) -> dict:
    if int(horizon) != HORIZON:
        raise ValueError("GPU scientific evaluator requires the frozen 128-step horizon")
    if scientific and not policy_id:
        raise ValueError("scientific evaluation requires a frozen policy_id")
    if policy_id is not None and Path(policy_id).name != policy_id:
        raise ValueError(f"invalid policy_id path component: {policy_id!r}")
    matrix = load_matrix(matrix_path)
    panel = _validated_panel(panel_path)
    mapping = matrix["shared_learning_contract"]["task_goal_mapping"][environment]
    goal_index = int(mapping["goal_index"])
    repeats = 1 if view == "mode" else 4
    initial, metadata = _stack_initial(environment, family, repeats)
    if max_starts is not None:
        keep = int(max_starts) * repeats
        initial = jax.tree.map(lambda value: value[:keep], initial)
        metadata = metadata[:keep]
    episode_count = len(metadata)
    seed_values = [
        20000 + 1000 * int(learner_seed) + 4 * row["state_index"] + row["repeat"]
        for row in metadata
    ]
    if len(seed_values) != len(set(seed_values)):
        raise RuntimeError("evaluation RNG seed collision")
    if metadata and max(
        4 * row["state_index"] + row["repeat"] for row in metadata
    ) >= 1000:
        raise RuntimeError("evaluation RNG seed band exceeds learner stride")
    seeds = jnp.asarray(seed_values, dtype=jnp.int32)
    keys = jax.vmap(jax.random.PRNGKey)(seeds)
    spec, network, params = (
        base._load_actor(environment, checkpoint) if actor_bundle is None else actor_bundle
    )
    fingerprint = (
        base._checkpoint_fingerprint(checkpoint)
        if checkpoint_fingerprint is None else checkpoint_fingerprint
    )
    policy_component = policy_id or "_non_scientific_unscoped"
    stem = (
        output_root / "policies" / policy_component / checkpoint_origin
        / environment / fingerprint["sha256"] / variant / view / family
    )
    trace_path = stem / "episodes.jsonl.gz"
    summary_path = stem / "summary.json"
    replay_path = stem / "independent_replay.json"
    if scientific and any(path.exists() for path in (trace_path, summary_path, replay_path)):
        raise RuntimeError(
            "scientific GPU evaluation artifact already exists; verify its receipt or quarantine it"
        )
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    compiled = _compiled_rollout(
        environment, spec, network, params, variant, view, episode_count
    )
    rollout_started = time.perf_counter()
    final, trace = compiled(params, initial, keys, jnp.asarray(goal_index, dtype=jnp.int32))
    final, trace = jax.device_get((final, trace))
    rollout_seconds = time.perf_counter() - rollout_started
    (
        final_state,
        _final_keys,
        done,
        success,
        length,
        _oracle,
        _defect_balance,
        defect_delivered,
        triggered,
        retained_recovered,
        excess_delivery,
    ) = final
    action_values = np.asarray([int(a) for a in ACTION_ENUMS[environment]], dtype=np.int32)
    episode_rows = []
    total_steps = 0
    serialization_started = time.perf_counter()
    # Full-state traces are audit artifacts, so preserve every record while
    # favouring evaluation throughput over maximum compression. Buffering one
    # task avoids thousands of tiny gzip writes; independent replay verifies
    # the decompressed content before the task receives a passing receipt.
    with _BufferedGzipText(trace_path, TRACE_GZIP_LEVEL) as handle:
        header = {
            "record_type": "header",
            "schema_version": TRACE_SCHEMA_VERSION,
            "scientific_performance_result": bool(scientific),
            "checkpoint_origin": checkpoint_origin,
            "policy_id": policy_id,
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
            "evaluation_backend": BACKEND_VERSION,
        }
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for episode_index, meta in enumerate(metadata):
            episode_id = f"{meta['id']}:repeat{meta['repeat']}"
            episode_length = int(length[episode_index])
            last_termination = "trace_horizon"
            for timestep in range(episode_length):
                if not bool(trace["active"][timestep, episode_index]):
                    raise RuntimeError("inactive step lies inside recorded episode length")
                legal_mask = np.asarray(trace["legal_mask"][timestep, episode_index], dtype=bool)
                legal = action_values[legal_mask].astype(int).tolist()
                action = int(trace["action"][timestep, episode_index])
                step_success = bool(trace["success_after"][timestep, episode_index])
                step_world_done = bool(trace["world_done_after"][timestep, episode_index])
                termination = "goal_complete" if step_success else (
                    "world_horizon" if step_world_done else "active"
                )
                last_termination = termination
                legacy = None if environment == "craft_remain" else bool(
                    trace["legacy"][timestep, episode_index]
                )
                before_total = int(trace["physical_resource_before"][timestep, episode_index])
                after_total = int(trace["physical_resource_after"][timestep, episode_index])
                row = {
                    "record_type": "step",
                    "episode_id": episode_id,
                    "start_id": meta["id"],
                    "state_index": meta["state_index"],
                    "layout": meta["layout"],
                    "phase": meta["phase"],
                    "repeat": meta["repeat"],
                    "rng_seed": int(seeds[episode_index]),
                    "timestep": timestep,
                    "state_before": _public_at(trace["state_before"], timestep, episode_index),
                    "public_observation": _public_at(trace["observation"], timestep, episode_index),
                    "action": action,
                    "action_name": ACTION_ENUMS[environment](action).name,
                    "policy_logits": np.asarray(trace["logits"][timestep, episode_index]).tolist(),
                    "legal_trigger_action_ids_exhaustive": legal,
                    "legal_opportunity_any": bool(legal),
                    "selected_legal_trigger_action": action in legal,
                    "legacy_opportunity_reached_narrow": legacy,
                    "state_after": _public_at(trace["state_after"], timestep, episode_index),
                    "termination_after_step": termination,
                    "success_after_step": step_success,
                    "physical_resource_before": before_total,
                    "physical_resource_after": after_total,
                    "physical_resource_delta": after_total - before_total,
                    "defect_created_amount": int(trace["defect_created_amount"][timestep, episode_index]),
                    "delivered_delta": int(trace["delivered_delta"][timestep, episode_index]),
                    "defect_attributed_delivery": int(trace["defect_attributed_delivery"][timestep, episode_index]),
                    "actual_defect_trigger": bool(trace["actual_defect_trigger"][timestep, episode_index]),
                    "retained_resource_recovered": bool(trace["retained_resource_recovered"][timestep, episode_index]),
                    "excess_delivery_event": bool(trace["excess_delivery_event"][timestep, episode_index]),
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                total_steps += 1
            exploited = bool(triggered[episode_index] and (
                retained_recovered[episode_index] and excess_delivery[episode_index]
                if environment == "craft_remain"
                else defect_delivered[episode_index] > 0
            ))
            terminal = {
                "record_type": "episode_terminal",
                "episode_id": episode_id,
                "success": bool(success[episode_index]),
                "length": episode_length,
                "discounted_return": float(
                    DISCOUNT ** (episode_length - 1)
                    if bool(success[episode_index]) and episode_length else 0.0
                ),
                "termination_reason": (
                    last_termination if last_termination != "active" else "trace_horizon"
                ),
                "actual_defect_triggered": bool(triggered[episode_index]),
                "actual_exploitation": exploited,
                "defect_resource_delivered": int(defect_delivered[episode_index]),
                "final_state": _public_at(final_state, episode_index),
            }
            handle.write(json.dumps(terminal, sort_keys=True) + "\n")
            episode_rows.append(terminal)
    serialization_seconds = time.perf_counter() - serialization_started

    replay_started = time.perf_counter()
    replay = verify_trace_file(trace_path)
    replay_seconds = time.perf_counter() - replay_started
    if not replay["passed"]:
        raise RuntimeError(f"independent trace replay failed: {replay['errors']}")
    _write_json(replay_path, replay)
    trace_bytes = trace_path.stat().st_size
    summary = {
        "schema_version": "hackrl_cross_environment_teacher_trace_summary_v1",
        "scientific_performance_result": bool(scientific),
        "checkpoint_origin": checkpoint_origin,
        "policy_id": policy_id,
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
        "independent_replay": replay,
        "evaluation_backend": BACKEND_VERSION,
        "trace_gzip_level": TRACE_GZIP_LEVEL,
        "timing_seconds": {
            "rollout_and_device_transfer": rollout_seconds,
            "serialization": serialization_seconds,
            "independent_replay": replay_seconds,
        },
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--environment", choices=tuple(base.ENV_TO_ENGINE))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-origin", choices=("reused_result", "new_matrix_result"))
    parser.add_argument("--learner-seed", type=int)
    parser.add_argument("--variant", choices=("fixed", "mutant"))
    parser.add_argument("--view", choices=("mode", "sample"))
    parser.add_argument("--family")
    parser.add_argument("--policy-id")
    parser.add_argument("--max-starts", type=int)
    parser.add_argument("--non-scientific-smoke", action="store_true")
    args = parser.parse_args()
    required = (
        args.environment, args.checkpoint, args.checkpoint_origin,
        args.learner_seed, args.variant, args.view, args.family,
    )
    if any(value is None for value in required):
        raise SystemExit("all evaluation arguments are required")
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
        scientific=not args.non_scientific_smoke,
        policy_id=args.policy_id,
        max_starts=args.max_starts,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
