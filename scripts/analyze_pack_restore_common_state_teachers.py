#!/usr/bin/env python3
"""Cross-evaluate PACK teachers and policies on an exact common observation pool.

This is a read-only checkpoint analysis.  It regenerates deterministic validation
rollouts, selects stage-action opportunities using environment transitions rather
than the action chosen by a policy, deduplicates exact observation/goal tensors,
and applies every same-seed teacher and policy to every pooled observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from hackrl.dual_leo import init_dual_leo_teacher, load_dual_checkpoint
from hackrl.pack_restore import (
    PackRestoreAction,
    PackRestoreSplit,
    PackRestoreVariant,
    pack_restore_goal_vector,
    pack_restore_step,
    pack_restore_world_done,
)
from hackrl.pack_restore_gc import (
    DELIVER_3_GOAL_INDEX,
    NUM_ACTIONS,
    NUM_GOALS,
    _BRANCH_LOCKED_FIELDS,
    _batch_inputs,
    _evaluation_states,
    config_from_pack_restore_gc_payload,
    initialize_pack_restore_gc,
)

from run_pack_restore_pretrained_teacher_freeze import (
    EXPECTED_SOURCE_FINGERPRINTS,
    SEEDS,
    SOURCE_RUN_ROOT,
    source_checkpoint,
)
from run_dual_leo_compare import _checkpoint_fingerprint
from trace_pack_restore_teacher_recommendations import (
    ACTION_NAMES,
    TARGET_ACTIONS,
    _successful_stage,
    checkpoint_for as final_checkpoint_for,
)


REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path("docs/manifests/pack_restore_common_state_teacher_analysis_v1.json")
FREEZE_RUN_ROOT = Path(
    "/raid/ext_csv/HackRL/runs/pack_restore_pretrained_teacher_freeze_v1"
)
OUTPUT_PATH = FREEZE_RUN_ROOT / "common_state_teacher_analysis.json"
STATES_PATH = FREEZE_RUN_ROOT / "common_state_teacher_analysis_states.npz"
EVALUATORS = ("pretrain", "D-off", "D-frozen", "D-on")
SOURCE_POLICIES = EVALUATORS
ACTION_NAMES_BY_INDEX = tuple(ACTION_NAMES[index] for index in range(NUM_ACTIONS))
TARGET_ACTION_INDICES = tuple(int(action) for action in TARGET_ACTIONS)
NON_TARGET_ACTION_INDICES = tuple(
    index for index in range(NUM_ACTIONS) if index not in TARGET_ACTION_INDICES
)
ROLLOUT_HORIZON = 128
BOOTSTRAP_SEED = 20261006
BOOTSTRAP_RESAMPLES = 20_000
OUTPUT_RESERVE_BYTES = 8 * 1024**3
EXECUTION_SOURCES = (
    MANIFEST_PATH,
    Path("scripts/analyze_pack_restore_common_state_teachers.py"),
    Path("scripts/run_pack_restore_pretrained_teacher_freeze.py"),
    Path("scripts/trace_pack_restore_teacher_recommendations.py"),
    Path("scripts/run_dual_leo_compare.py"),
    Path("src/hackrl/pack_restore.py"),
    Path("src/hackrl/pack_restore_gc.py"),
    Path("src/hackrl/dual_leo.py"),
    Path("src/hackrl/batch_renorm.py"),
)
CONTRASTS = (
    ("D-frozen", "pretrain", "D-frozen_minus_pretrain"),
    ("D-off", "pretrain", "D-off_minus_pretrain"),
    ("D-on", "pretrain", "D-on_minus_pretrain"),
    ("D-on", "D-frozen", "D-on_minus_D-frozen"),
    ("D-on", "D-off", "D-on_minus_D-off"),
)
METRICS = (
    "teacher_recommendation_rate",
    "teacher_q",
    "teacher_q_margin_vs_best_non_target",
    "policy_mode_selection_rate",
    "policy_probability",
    "policy_probability_margin_vs_best_non_target",
)
VISITATION_METRICS = (
    "eligible_state_rate",
    "episode_opportunity_rate",
    "policy_selection_given_opportunity",
)



def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def _checkpoint(label, seed, freeze_run_root=FREEZE_RUN_ROOT):
    if label == "pretrain":
        return source_checkpoint(seed)
    if label not in EVALUATORS:
        raise ValueError(f"unknown evaluator: {label}")
    return final_checkpoint_for(Path(freeze_run_root), label, seed)


def _observation_sha256(map_channels, numeric_features, goal_index):
    digest = hashlib.sha256()
    for name, value in (
        ("map_channels", map_channels),
        ("numeric_features", numeric_features),
    ):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    digest.update(np.asarray([goal_index], dtype=np.int32).tobytes())
    return digest.hexdigest()


def _eligibility_for_state(state):
    values = []
    for action in TARGET_ACTIONS:
        action_value = jnp.asarray(int(action), dtype=jnp.int32)
        stepped = pack_restore_step(state, action_value, PackRestoreVariant.MUTANT)
        values.append(_successful_stage(action_value, state, stepped))
    return jnp.stack(values)


def _make_pool_rollout(network, initial):
    episode_count = initial.tick.shape[0]

    def rollout(policy_params):
        def step(carry, step_index):
            state, done = carry
            goals = jnp.full(
                (episode_count,), DELIVER_3_GOAL_INDEX, dtype=jnp.int32
            )
            inputs = _batch_inputs(state, goals)
            policy, _ = network.apply(policy_params, *inputs)
            actions = jnp.argmax(policy.logits, axis=-1)
            active = jnp.logical_not(done)
            eligibility = jax.vmap(_eligibility_for_state)(state)
            safe_actions = jnp.where(active, actions, int(PackRestoreAction.NOOP))
            stepped = jax.vmap(
                lambda item, action: pack_restore_step(
                    item, action, PackRestoreVariant.MUTANT
                )
            )(state, safe_actions)
            achieved = jax.vmap(pack_restore_goal_vector)(stepped)[
                :, DELIVER_3_GOAL_INDEX
            ]
            world_done = jax.vmap(pack_restore_world_done)(stepped)
            done_next = jnp.logical_or(done, jnp.logical_or(achieved, world_done))
            observation = {
                "step": jnp.full((episode_count,), step_index, dtype=jnp.int32),
                "active": active,
                "eligibility": eligibility,
                "map_channels": inputs[0],
                "numeric_features": inputs[1],
                "policy_action": safe_actions,
            }
            return (stepped, done_next), observation

        (_, _), observations = jax.lax.scan(
            step,
            (initial, jnp.zeros((episode_count,), dtype=jnp.bool_)),
            jnp.arange(ROLLOUT_HORIZON, dtype=jnp.int32),
        )
        return observations

    return jax.jit(rollout)


def _make_cross_evaluator(network, teacher_network):
    @jax.jit
    def evaluate(policy_params, teacher_params, teacher_batch_stats, maps, numeric):
        goals = jnp.full((maps.shape[0],), DELIVER_3_GOAL_INDEX, dtype=jnp.int32)
        goal_one_hot = jax.nn.one_hot(goals, NUM_GOALS, dtype=jnp.float32)
        policy, _ = network.apply(policy_params, maps, numeric, goal_one_hot)
        probabilities = jax.nn.softmax(policy.logits, axis=-1)
        q_values = teacher_network.apply(
            {"params": teacher_params, "batch_stats": teacher_batch_stats},
            maps,
            numeric,
            train=False,
        )[:, DELIVER_3_GOAL_INDEX, :]
        return probabilities, q_values

    return evaluate


def _bootstrap(values, rng):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a finite nonempty vector")
    draws = rng.choice(array, size=(BOOTSTRAP_RESAMPLES, array.size), replace=True)
    means = np.mean(draws, axis=1)
    return {
        "n_seeds": int(array.size),
        "mean": float(np.mean(array)),
        "sample_standard_deviation": (
            float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        ),
        "bootstrap_95_percentile_interval": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
    }


def _capacity_snapshot():
    result = {}
    for name, path in (("home", Path("/home/ext_csv")), ("raid", Path("/raid/ext_csv"))):
        usage = shutil.disk_usage(path)
        result[name] = {
            "path": str(path),
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
        }
    return result


def _validate_manifest_contract(manifest):
    checks = {
        "manifest_id": (
            manifest["manifest_id"],
            "pack_restore_common_state_teacher_analysis_v1",
        ),
        "seeds": (manifest["inputs"]["seeds"], list(SEEDS)),
        "source run root": (
            manifest["inputs"]["source_scale_run_root"],
            str(SOURCE_RUN_ROOT),
        ),
        "freeze run root": (
            manifest["inputs"]["teacher_freeze_run_root"],
            str(FREEZE_RUN_ROOT),
        ),
        "source policies": (
            manifest["state_pool"]["source_policies"],
            list(SOURCE_POLICIES),
        ),
        "evaluators": (
            manifest["cross_evaluation"]["evaluators"],
            list(EVALUATORS),
        ),
        "target actions": (
            manifest["state_pool"]["target_actions"],
            [action.name for action in TARGET_ACTIONS],
        ),
        "horizon": (manifest["state_pool"]["horizon"], ROLLOUT_HORIZON),
        "episodes": (
            manifest["state_pool"]["episodes_per_source_policy_and_seed"],
            64,
        ),
        "bootstrap seed": (
            manifest["statistics"]["bootstrap_seed"],
            BOOTSTRAP_SEED,
        ),
        "bootstrap resamples": (
            manifest["statistics"]["bootstrap_resamples"],
            BOOTSTRAP_RESAMPLES,
        ),
        "reserve": (
            manifest["storage"]["minimum_free_space_before_write_bytes"],
            OUTPUT_RESERVE_BYTES,
        ),
        "result output": (manifest["outputs"]["result"], str(OUTPUT_PATH)),
        "state output": (
            manifest["outputs"]["exact_state_tensors"],
            str(STATES_PATH),
        ),
        "execution sources": (
            manifest["implementation"]["execution_sources"],
            [path.as_posix() for path in EXECUTION_SOURCES],
        ),
        "no learning": (
            manifest["no_learning_contract"]["training_or_optimizer_updates"],
            0,
        ),
    }
    mismatches = {
        name: {"manifest": actual, "implementation": expected}
        for name, (actual, expected) in checks.items()
        if actual != expected
    }
    if mismatches:
        raise RuntimeError(f"analysis manifest contract mismatch: {mismatches}")


def _verify_prior_artifacts(run_root, manifest):
    paths = {
        "result": Path(run_root) / "result.json",
        "teacher_trace": Path(run_root) / "teacher_recommendations.json",
        "run_manifest": Path(run_root) / "run_manifest.json",
    }
    expected = {
        "result": manifest["inputs"]["teacher_freeze_result"]["sha256"],
        "teacher_trace": manifest["inputs"]["prior_teacher_trace"]["sha256"],
        "run_manifest": manifest["inputs"]["teacher_freeze_run_manifest"]["sha256"],
    }
    measured = {name: _sha256(path) for name, path in paths.items()}
    if measured != expected:
        raise RuntimeError(f"prior artifact digest mismatch: {measured}")
    result = _read(paths["result"])
    if not result.get("execution_complete") or result.get("errors") or result.get(
        "missing_curves"
    ):
        raise RuntimeError("prior teacher-freeze result is not complete")
    if len(result.get("teacher_freeze_checks", [])) != 20 or not all(
        row.get("passed") for row in result["teacher_freeze_checks"]
    ):
        raise RuntimeError("prior teacher freeze checks are incomplete")
    trace = _read(paths["teacher_trace"])
    final_checkpoints = {}
    for row in trace.get("checkpoints", []):
        key = (row["condition"], int(row["seed"]))
        if key in final_checkpoints:
            raise RuntimeError(f"duplicate checkpoint in authenticated trace: {key}")
        final_checkpoints[key] = {
            "checkpoint": str(Path(row["checkpoint"]).resolve()),
            "state_sha256": row["state_sha256"],
        }
    expected_keys = {
        (label, seed)
        for label in ("D-off", "D-frozen", "D-on")
        for seed in SEEDS
    }
    if set(final_checkpoints) != expected_keys:
        raise RuntimeError("authenticated trace checkpoint set is incomplete")
    artifacts = {
        name: {"path": str(paths[name].resolve()), "sha256": measured[name]}
        for name in paths
    }
    return artifacts, final_checkpoints


def _verify_execution(expected_execution_sha, expected_manifest_sha256):
    actual_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    if actual_sha != expected_execution_sha:
        raise RuntimeError(
            f"execution SHA mismatch: expected {expected_execution_sha}, got {actual_sha}"
        )
    manifest_path = REPOSITORY / MANIFEST_PATH
    manifest_digest = _sha256(manifest_path)
    if manifest_digest != expected_manifest_sha256:
        raise RuntimeError(
            "analysis manifest SHA mismatch: "
            f"expected {expected_manifest_sha256}, got {manifest_digest}"
        )
    manifest = _read(manifest_path)
    _validate_manifest_contract(manifest)
    source_hashes = {}
    for relative in EXECUTION_SOURCES:
        working = (REPOSITORY / relative).read_bytes()
        committed = subprocess.check_output(
            ["git", "show", f"{expected_execution_sha}:{relative.as_posix()}"],
            cwd=REPOSITORY,
        )
        if working != committed:
            raise RuntimeError(f"execution source differs from commit: {relative}")
        source_hashes[relative.as_posix()] = _bytes_sha256(working)
    provenance = {
        "execution_code_sha": actual_sha,
        "analysis_manifest_sha256": manifest_digest,
        "execution_source_sha256": source_hashes,
        "runtime": {
            "python": sys.version,
            "jax": jax.__version__,
            "numpy": np.__version__,
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_devices": [str(device) for device in jax.devices()],
        },
    }
    return provenance, manifest


def _validate_config(reference, recorded, checkpoint):
    fields = tuple(field for field in _BRANCH_LOCKED_FIELDS if field != "seed")
    mismatches = {
        field: (getattr(reference, field), getattr(recorded, field))
        for field in fields
        if getattr(reference, field) != getattr(recorded, field)
    }
    if mismatches:
        raise RuntimeError(f"checkpoint config mismatch {checkpoint}: {mismatches}")


def _validate_checkpoint_contract(
    label,
    seed,
    checkpoint,
    reference_config,
    authenticated_final_checkpoints,
):
    checkpoint = Path(checkpoint).resolve()
    expected_path = _checkpoint(label, seed, FREEZE_RUN_ROOT).resolve()
    if checkpoint != expected_path:
        raise RuntimeError(
            f"checkpoint path mismatch for {label} seed {seed}: {checkpoint}"
        )
    for filename in ("arm.json", "config.json", "metadata.json", "state.msgpack"):
        if not (checkpoint / filename).is_file():
            raise FileNotFoundError(f"checkpoint file missing: {checkpoint / filename}")
    config_payload = _read(checkpoint / "config.json")
    recorded = config_from_pack_restore_gc_payload(config_payload)
    _validate_config(reference_config, recorded, checkpoint)
    metadata = _read(checkpoint / "metadata.json")
    arm = _read(checkpoint / "arm.json")
    expected_arm = {
        "pretrain": {"learn_teacher": True, "imitate_teacher": True},
        "D-off": {"learn_teacher": True, "imitate_teacher": False},
        "D-frozen": {"learn_teacher": False, "imitate_teacher": True},
        "D-on": {"learn_teacher": True, "imitate_teacher": True},
    }[label]
    expected_contract = (
        {
            "seed": seed,
            "variant": "fixed",
            "goal_mode": "workshop12",
            "num_updates": 512,
            "global_update": 512,
            "directory": "update_512",
        }
        if label == "pretrain"
        else {
            "seed": seed,
            "variant": "mutant",
            "goal_mode": "deliver_3",
            "num_updates": 4096,
            "global_update": 4608,
            "directory": "adapt_4096",
        }
    )
    measured_contract = {
        "seed": int(recorded.seed),
        "variant": str(recorded.variant),
        "goal_mode": recorded.goal_mode,
        "num_updates": int(recorded.num_updates),
        "global_update": int(metadata["global_update"]),
        "directory": checkpoint.name,
    }
    if arm != expected_arm or measured_contract != expected_contract:
        raise RuntimeError(
            f"checkpoint treatment mismatch for {label} seed {seed}: "
            f"arm={arm}, contract={measured_contract}"
        )
    state_sha256 = _sha256(checkpoint / "state.msgpack")
    if label == "pretrain":
        fingerprint = _checkpoint_fingerprint(checkpoint)
        if fingerprint["sha256"] != EXPECTED_SOURCE_FINGERPRINTS[seed]:
            raise RuntimeError(
                f"pretraining fingerprint mismatch for seed {seed}: {fingerprint}"
            )
    else:
        authenticated = authenticated_final_checkpoints[(label, seed)]
        if (
            authenticated["checkpoint"] != str(checkpoint)
            or authenticated["state_sha256"] != state_sha256
        ):
            raise RuntimeError(
                f"authenticated final checkpoint mismatch for {label} seed {seed}"
            )
        fingerprint = None
    return {
        "seed": int(seed),
        "label": label,
        "checkpoint": str(checkpoint),
        "state_sha256": state_sha256,
        "checkpoint_fingerprint": fingerprint,
        "arm": arm,
        "config_contract": measured_contract,
        "metadata_schema_version": metadata.get("schema_version"),
    }


def _pool_seed(
    seed,
    checkpoints,
    network,
    template,
    leo_template,
    rollout,
    labels,
    state_indices,
):
    pool = {}
    visitation = []
    for source_label in SOURCE_POLICIES:
        checkpoint = checkpoints[source_label]
        runner, _ = load_dual_checkpoint(checkpoint, template, leo_template)
        observations = rollout(runner.train_state.params)
        jax.block_until_ready(observations["eligibility"])
        values = jax.device_get(observations)
        active = np.asarray(values["active"], dtype=bool)
        eligibility = np.asarray(values["eligibility"], dtype=bool)
        opportunity = active & np.any(eligibility, axis=-1)
        contributed = set()
        selected = np.asarray(values["policy_action"])
        for step, episode in np.argwhere(opportunity):
            map_channels = np.asarray(values["map_channels"][step, episode])
            numeric_features = np.asarray(values["numeric_features"][step, episode])
            digest = _observation_sha256(
                map_channels, numeric_features, DELIVER_3_GOAL_INDEX
            )
            signature = tuple(bool(value) for value in eligibility[step, episode])
            if digest not in pool:
                pool[digest] = {
                    "map_channels": map_channels,
                    "numeric_features": numeric_features,
                    "eligibility_signatures": set(),
                    "provenance": [],
                }
            else:
                if not np.array_equal(pool[digest]["map_channels"], map_channels):
                    raise RuntimeError(f"observation hash collision for {digest}")
                if not np.array_equal(
                    pool[digest]["numeric_features"], numeric_features
                ):
                    raise RuntimeError(f"observation hash collision for {digest}")
            pool[digest]["eligibility_signatures"].add(signature)
            pool[digest]["provenance"].append(
                {
                    "source_policy": source_label,
                    "family": (
                        "natural_reset"
                        if int(labels[episode]) == 0
                        else "common_setup"
                    ),
                    "validation_state_index": int(state_indices[episode]),
                    "rollout_step": int(step),
                    "policy_action": ACTION_NAMES[int(selected[step, episode])],
                }
            )
            contributed.add(digest)
        active_count = int(np.sum(active))
        episode_count = int(active.shape[1])
        opportunity_count = int(np.sum(opportunity))
        episodes_with_opportunity = int(np.sum(np.any(opportunity, axis=0)))
        selected_target = opportunity & np.isin(selected, TARGET_ACTION_INDICES)
        selected_target_count = int(np.sum(selected_target))
        per_action = {}
        for action_offset, action in enumerate(TARGET_ACTIONS):
            mask = active & eligibility[..., action_offset]
            selected_mask = mask & (selected == int(action))
            eligible_count = int(np.sum(mask))
            episode_opportunities = int(np.sum(np.any(mask, axis=0)))
            selected_count = int(np.sum(selected_mask))
            per_action[action.name] = {
                "eligible_state_occurrences": eligible_count,
                "episodes_with_opportunity": episode_opportunities,
                "policy_selected_at_opportunity": selected_count,
                "eligible_state_rate": eligible_count / max(active_count, 1),
                "episode_opportunity_rate": (
                    episode_opportunities / max(episode_count, 1)
                ),
                "policy_selection_given_opportunity": (
                    selected_count / max(eligible_count, 1)
                ),
            }
        visitation.append(
            {
                "seed": int(seed),
                "source_policy": source_label,
                "episodes": episode_count,
                "active_state_occurrences": active_count,
                "eligible_state_occurrences": opportunity_count,
                "episodes_with_opportunity": episodes_with_opportunity,
                "policy_selected_target_at_opportunity": selected_target_count,
                "eligible_state_rate": opportunity_count / max(active_count, 1),
                "episode_opportunity_rate": (
                    episodes_with_opportunity / max(episode_count, 1)
                ),
                "policy_selection_given_opportunity": (
                    selected_target_count / max(opportunity_count, 1)
                ),
                "unique_observations_contributed": int(len(contributed)),
                "by_action": per_action,
            }
        )
    return pool, visitation


def _evaluate_seed(
    seed,
    pool,
    checkpoints,
    network,
    template,
    leo_template,
    cross_evaluator,
):
    ordered = sorted(pool)
    unambiguous = [
        digest
        for digest in ordered
        if len(pool[digest]["eligibility_signatures"]) == 1
    ]
    if not unambiguous:
        raise RuntimeError(f"seed {seed} has no unambiguous common observations")
    unambiguous_set = set(unambiguous)
    ambiguous = [key for key in ordered if key not in unambiguous_set]
    maps = jnp.asarray(np.stack([pool[key]["map_channels"] for key in ordered]))
    numeric = jnp.asarray(
        np.stack([pool[key]["numeric_features"] for key in ordered])
    )
    evaluations = {}
    for label in EVALUATORS:
        checkpoint = checkpoints[label]
        runner, leo = load_dual_checkpoint(checkpoint, template, leo_template)
        probabilities, q_values = cross_evaluator(
            runner.train_state.params,
            leo.params,
            leo.batch_stats,
            maps,
            numeric,
        )
        probabilities, q_values = jax.device_get((probabilities, q_values))
        evaluations[label] = {
            "policy_probabilities": np.asarray(probabilities),
            "teacher_q": np.asarray(q_values),
        }
    if not np.array_equal(
        evaluations["pretrain"]["teacher_q"],
        evaluations["D-frozen"]["teacher_q"],
    ):
        raise RuntimeError(
            f"seed {seed} frozen teacher output differs from pretraining teacher "
            "on the complete pooled observation set"
        )

    records = []
    for state_offset, digest in enumerate(ordered):
        if digest not in unambiguous_set:
            continue
        item = pool[digest]
        signature = next(iter(item["eligibility_signatures"]))
        eligible_actions = [
            TARGET_ACTIONS[offset].name
            for offset, eligible in enumerate(signature)
            if eligible
        ]
        record = {
            "seed": int(seed),
            "observation_sha256": digest,
            "goal": "delivery/count_ge_3",
            "eligible_target_actions": eligible_actions,
            "provenance": item["provenance"],
            "evaluations": {},
        }
        for label in EVALUATORS:
            probabilities = evaluations[label]["policy_probabilities"][state_offset]
            q_values = evaluations[label]["teacher_q"][state_offset]
            teacher_action = int(np.argmax(q_values))
            policy_action = int(np.argmax(probabilities))
            stage_metrics = {}
            for action_name in eligible_actions:
                action_index = int(PackRestoreAction[action_name])
                best_q_index = max(
                    NON_TARGET_ACTION_INDICES, key=lambda index: q_values[index]
                )
                best_probability_index = max(
                    NON_TARGET_ACTION_INDICES,
                    key=lambda index: probabilities[index],
                )
                stage_metrics[action_name] = {
                    "teacher_q": float(q_values[action_index]),
                    "best_non_target_action": ACTION_NAMES[best_q_index],
                    "best_non_target_q": float(q_values[best_q_index]),
                    "teacher_q_margin_vs_best_non_target": float(
                        q_values[action_index] - q_values[best_q_index]
                    ),
                    "policy_probability": float(probabilities[action_index]),
                    "best_non_target_policy_action": ACTION_NAMES[
                        best_probability_index
                    ],
                    "best_non_target_policy_probability": float(
                        probabilities[best_probability_index]
                    ),
                    "policy_probability_margin_vs_best_non_target": float(
                        probabilities[action_index]
                        - probabilities[best_probability_index]
                    ),
                }
            record["evaluations"][label] = {
                "teacher_recommendation": ACTION_NAMES[teacher_action],
                "teacher_argmax_tie_count": int(
                    np.sum(q_values == q_values[teacher_action])
                ),
                "teacher_q_by_action": {
                    ACTION_NAMES_BY_INDEX[index]: float(q_values[index])
                    for index in range(NUM_ACTIONS)
                },
                "policy_mode_action": ACTION_NAMES[policy_action],
                "policy_argmax_tie_count": int(
                    np.sum(probabilities == probabilities[policy_action])
                ),
                "policy_probability_by_action": {
                    ACTION_NAMES_BY_INDEX[index]: float(probabilities[index])
                    for index in range(NUM_ACTIONS)
                },
                "eligible_stage_metrics": stage_metrics,
            }
        records.append(record)
    return records, ambiguous


def _seed_metric_rows(records):
    grouped = defaultdict(list)
    for record in records:
        for action_name in record["eligible_target_actions"]:
            for evaluator in EVALUATORS:
                evaluation = record["evaluations"][evaluator]
                stage = evaluation["eligible_stage_metrics"][action_name]
                grouped[(record["seed"], evaluator, action_name)].append(
                    {
                        "teacher_recommendation_rate": float(
                            evaluation["teacher_recommendation"] == action_name
                        ),
                        "teacher_q": stage["teacher_q"],
                        "teacher_q_margin_vs_best_non_target": stage[
                            "teacher_q_margin_vs_best_non_target"
                        ],
                        "policy_mode_selection_rate": float(
                            evaluation["policy_mode_action"] == action_name
                        ),
                        "policy_probability": stage["policy_probability"],
                        "policy_probability_margin_vs_best_non_target": stage[
                            "policy_probability_margin_vs_best_non_target"
                        ],
                    }
                )
                grouped[(record["seed"], evaluator, "ALL_TARGET_STAGE_PAIRS")].append(
                    grouped[(record["seed"], evaluator, action_name)][-1]
                )
    rows = []
    for (seed, evaluator, action), values in sorted(grouped.items(), key=repr):
        row = {
            "seed": int(seed),
            "evaluator": evaluator,
            "eligible_action": action,
            "state_action_pairs": len(values),
        }
        for metric in METRICS:
            row[metric] = float(np.mean([value[metric] for value in values]))
        rows.append(row)
    return rows


def _aggregate_metrics(seed_rows):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    grouped = defaultdict(list)
    for row in seed_rows:
        for metric in METRICS:
            grouped[(row["evaluator"], row["eligible_action"], metric)].append(
                row[metric]
            )
    summaries = []
    for key, values in sorted(grouped.items(), key=repr):
        summaries.append(
            {
                "evaluator": key[0],
                "eligible_action": key[1],
                "metric": key[2],
                **_bootstrap(values, rng),
            }
        )

    lookup = {
        (row["seed"], row["evaluator"], row["eligible_action"]): row
        for row in seed_rows
    }
    contrasts = []
    action_names = sorted({row["eligible_action"] for row in seed_rows})
    for left, right, name in CONTRASTS:
        for action_name in action_names:
            for metric in METRICS:
                values = []
                seed_points = []
                for seed in SEEDS:
                    left_row = lookup.get((seed, left, action_name))
                    right_row = lookup.get((seed, right, action_name))
                    if left_row is None or right_row is None:
                        continue
                    value = float(left_row[metric] - right_row[metric])
                    values.append(value)
                    seed_points.append({"seed": int(seed), "value": value})
                if values:
                    contrasts.append(
                        {
                            "contrast": name,
                            "eligible_action": action_name,
                            "metric": metric,
                            "seed_points": seed_points,
                            **_bootstrap(values, rng),
                        }
                    )
    return summaries, contrasts


def _aggregate_visitation(visitation):
    seed_rows = []
    for row in visitation:
        scopes = {"ANY_TARGET_ACTION": row}
        scopes.update(row["by_action"])
        for action_name, values in scopes.items():
            seed_rows.append(
                {
                    "seed": int(row["seed"]),
                    "source_policy": row["source_policy"],
                    "eligible_action": action_name,
                    **{
                        metric: float(values[metric])
                        for metric in VISITATION_METRICS
                    },
                }
            )
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    grouped = defaultdict(list)
    for row in seed_rows:
        for metric in VISITATION_METRICS:
            grouped[
                (row["source_policy"], row["eligible_action"], metric)
            ].append(row[metric])
    summaries = [
        {
            "source_policy": key[0],
            "eligible_action": key[1],
            "metric": key[2],
            **_bootstrap(values, rng),
        }
        for key, values in sorted(grouped.items(), key=repr)
    ]
    lookup = {
        (row["seed"], row["source_policy"], row["eligible_action"]): row
        for row in seed_rows
    }
    contrasts = []
    action_names = sorted({row["eligible_action"] for row in seed_rows})
    for left, right, name in CONTRASTS:
        for action_name in action_names:
            for metric in VISITATION_METRICS:
                points = []
                values = []
                for seed in SEEDS:
                    left_row = lookup.get((seed, left, action_name))
                    right_row = lookup.get((seed, right, action_name))
                    if left_row is None or right_row is None:
                        continue
                    value = float(left_row[metric] - right_row[metric])
                    points.append({"seed": int(seed), "value": value})
                    values.append(value)
                if values:
                    contrasts.append(
                        {
                            "contrast": name,
                            "eligible_action": action_name,
                            "metric": metric,
                            "seed_points": points,
                            **_bootstrap(values, rng),
                        }
                    )
    return seed_rows, summaries, contrasts



def analyze(
    freeze_run_root,
    *,
    expected_execution_sha,
    expected_manifest_sha256,
):
    freeze_run_root = Path(freeze_run_root).resolve()
    if freeze_run_root != FREEZE_RUN_ROOT.resolve():
        raise RuntimeError(
            f"run root differs from manifest contract: {freeze_run_root}"
        )
    provenance, manifest = _verify_execution(
        expected_execution_sha, expected_manifest_sha256
    )
    input_artifacts, authenticated_final_checkpoints = _verify_prior_artifacts(
        freeze_run_root, manifest
    )
    provenance["inputs"] = input_artifacts
    capacity_before = _capacity_snapshot()
    if capacity_before["raid"]["free_bytes"] < OUTPUT_RESERVE_BYTES:
        raise RuntimeError("RAID free space is below the 8 GiB analysis reserve")

    reference_checkpoint = _checkpoint("D-frozen", SEEDS[0], freeze_run_root)
    config = config_from_pack_restore_gc_payload(_read(reference_checkpoint / "config.json"))
    network, template = initialize_pack_restore_gc(config)
    example_inputs = _batch_inputs(template.env_state, template.current_goal)
    teacher_network, leo_template, _ = init_dual_leo_teacher(
        config, example_inputs[0], example_inputs[1], NUM_GOALS, NUM_ACTIONS
    )
    initial, labels, state_indices, _ = _evaluation_states(
        PackRestoreSplit.VALIDATION,
        1,
        config.source_growth_period,
    )
    rollout = _make_pool_rollout(network, initial)
    cross_evaluator = _make_cross_evaluator(network, teacher_network)
    labels_host = np.asarray(labels)
    state_indices_host = np.asarray(state_indices)
    if int(initial.tick.shape[0]) != 64:
        raise RuntimeError("evaluation state generator did not produce 64 episodes")
    if (
        labels_host.shape != (64,)
        or int(np.sum(labels_host == 0)) != 32
        or int(np.sum(labels_host == 1)) != 32
        or set(np.unique(labels_host).tolist()) != {0, 1}
    ):
        raise RuntimeError("evaluation start-family labels do not match the manifest")
    if state_indices_host.shape != (64,) or set(
        np.unique(state_indices_host).tolist()
    ) != set(range(32)):
        raise RuntimeError("evaluation validation-state indices are incomplete")
    if (
        np.asarray(example_inputs[0]).dtype != np.float32
        or np.asarray(example_inputs[1]).dtype != np.float32
    ):
        raise RuntimeError("model observation tensors are not exact float32 inputs")

    all_records = []
    checkpoint_rows = []
    visitation = []
    ambiguity = []
    state_maps = []
    state_numeric = []
    state_seeds = []
    state_hashes = []
    checkpoint_hashes_before = {}
    for seed in SEEDS:
        checkpoints = {
            label: _checkpoint(label, seed, freeze_run_root)
            for label in EVALUATORS
        }
        for label, checkpoint in checkpoints.items():
            checkpoint_row = _validate_checkpoint_contract(
                label,
                seed,
                checkpoint,
                config,
                authenticated_final_checkpoints,
            )
            checkpoint_rows.append(checkpoint_row)
            checkpoint_hashes_before[(seed, label)] = checkpoint_row[
                "state_sha256"
            ]
        pool, seed_visitation = _pool_seed(
            seed,
            checkpoints,
            network,
            template,
            leo_template,
            rollout,
            labels_host,
            state_indices_host,
        )
        records, ambiguous = _evaluate_seed(
            seed,
            pool,
            checkpoints,
            network,
            template,
            leo_template,
            cross_evaluator,
        )
        for record in records:
            item = pool[record["observation_sha256"]]
            state_seeds.append(seed)
            state_hashes.append(record["observation_sha256"])
            state_maps.append(item["map_channels"])
            state_numeric.append(item["numeric_features"])
        all_records.extend(records)
        visitation.extend(seed_visitation)
        ambiguity.extend(
            {"seed": int(seed), "observation_sha256": digest}
            for digest in ambiguous
        )

    for row in checkpoint_rows:
        key = (row["seed"], row["label"])
        measured = _sha256(Path(row["checkpoint"]) / "state.msgpack")
        if measured != checkpoint_hashes_before[key] or measured != row["state_sha256"]:
            raise RuntimeError(f"checkpoint changed during analysis: {row['checkpoint']}")
        row["state_immutable"] = True

    seed_rows = _seed_metric_rows(all_records)
    summaries, contrasts = _aggregate_metrics(seed_rows)
    visitation_seed_rows, visitation_summaries, visitation_contrasts = (
        _aggregate_visitation(visitation)
    )
    tie_counts = {
        label: {
            "teacher_argmax_ties": sum(
                record["evaluations"][label]["teacher_argmax_tie_count"] > 1
                for record in all_records
            ),
            "policy_argmax_ties": sum(
                record["evaluations"][label]["policy_argmax_tie_count"] > 1
                for record in all_records
            ),
        }
        for label in EVALUATORS
    }
    result = {
        "schema_version": "hackrl_pack_restore_common_state_teacher_analysis_v1",
        "analysis_id": "pack_restore_common_state_teacher_analysis_v1",
        "execution_complete": True,
        "training_or_optimizer_updates": 0,
        "provenance": provenance,
        "design": {
            "seeds": list(SEEDS),
            "source_policies": list(SOURCE_POLICIES),
            "cross_evaluators": list(EVALUATORS),
            "trained_variant": "mutant",
            "evaluation_kernel": "mutant",
            "evaluation_split": "validation",
            "policy": "mode",
            "start_families": ["natural_reset", "common_setup"],
            "goal": "delivery/count_ge_3",
            "target_actions": [action.name for action in TARGET_ACTIONS],
            "same_state_definition": (
                "Exact equality of map_channels, numeric_features, and goal tensors "
                "seen by the models; it does not assert latent environment-state identity."
            ),
            "argmax_tie_rule": (
                "JAX/NumPy first-index argmax is used and every state with more than "
                "one exactly maximal value is counted separately."
            ),
            "decision_rule": (
                "Report seed points, paired mean differences, and bootstrap intervals. "
                "No equivalence or minimum-effect threshold is declared; exact frozen "
                "identity is an implementation gate, not a statistical decision."
            ),
            "state_selection": (
                "Include every active rollout state where applying at least one target "
                "action produces the declared stage transition, independent of the "
                "source policy's selected action. Pool the four same-seed source-policy "
                "rollouts and weight each exact observation/goal tensor once."
            ),
            "normal_alternative": (
                "For each eligible target action, the highest-Q or highest-probability "
                "action outside the four target-stage actions. This is an operational "
                "non-target comparator, not a claim that it is globally optimal."
            ),
            "independent_unit": "learner_seed",
            "state_or_episode_is_not_a_replication_unit": True,
        },
        "mandela_audit": {
            "selection_tautology_fix": (
                "States are selected by environment transition eligibility, not by a "
                "policy choosing or a teacher recommending the target action."
            ),
            "shared_pool_bias_fix": (
                "The exact observation pool is the same-seed union of pretrain, D-off, "
                "D-frozen, and D-on paths; visitation frequency is reported separately."
            ),
            "wrong_null_fix": (
                "D-on versus pretrain/D-frozen is labeled as the effect of the complete "
                "teacher learning state, including Q parameters and BatchRenorm state, "
                "not Q-weight learning alone."
            ),
            "remaining_limit": (
                "The environment transition predicates and candidate actions are "
                "experimenter-defined. Cross-evaluation separates teacher output from "
                "state selection but is descriptive and is not an intervention on advice."
            ),
        },
        "capacity_before": capacity_before,
        "checkpoints": checkpoint_rows,
        "state_pool": {
            "unique_unambiguous_observations": len(all_records),
            "ambiguous_observations_excluded": len(ambiguity),
            "ambiguity_records": ambiguity,
            "visitation": visitation,
        },
        "statistics": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "seed_metric_rows": seed_rows,
            "summaries": summaries,
            "paired_contrasts": contrasts,
            "visitation_seed_rows": visitation_seed_rows,
            "visitation_summaries": visitation_summaries,
            "visitation_paired_contrasts": visitation_contrasts,
            "argmax_tie_counts": tie_counts,
        },
        "records": all_records,
        "claim_limit": (
            "This no-learning same-observation analysis can show that saved teachers "
            "or policies produce different outputs on identical inputs. It cannot by "
            "itself establish that teacher advice caused a later policy action or "
            "separate teacher Q parameters from BatchRenorm-state updating."
        ),
    }
    map_array = np.stack(state_maps)
    numeric_array = np.stack(state_numeric)
    if map_array.dtype != np.float32 or numeric_array.dtype != np.float32:
        raise RuntimeError(
            "pooled observation dtype changed before exact tensor serialization"
        )
    state_arrays = {
        "seed": np.asarray(state_seeds, dtype=np.int32),
        "observation_sha256": np.asarray(state_hashes, dtype="S64"),
        "map_channels": map_array,
        "numeric_features": numeric_array,
        "goal_index": np.full(
            (len(state_seeds),), DELIVER_3_GOAL_INDEX, dtype=np.int32
        ),
    }
    return result, state_arrays


def _write_outputs(result, state_arrays, output, states_output):
    output = Path(output).resolve()
    states_output = Path(states_output).resolve()
    if output != OUTPUT_PATH.resolve() or states_output != STATES_PATH.resolve():
        raise RuntimeError(
            "analysis output paths differ from the committed manifest contract"
        )
    raid_root = Path("/raid/ext_csv").resolve()
    if raid_root not in output.parents or raid_root not in states_output.parents:
        raise RuntimeError("analysis outputs must be stored under /raid/ext_csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    states_output.parent.mkdir(parents=True, exist_ok=True)
    states_temporary = states_output.with_name(states_output.name + ".tmp")
    with states_temporary.open("wb") as handle:
        np.savez_compressed(handle, **state_arrays)
    states_temporary.replace(states_output)
    with np.load(states_output, allow_pickle=False) as restored:
        if set(restored.files) != set(state_arrays):
            raise RuntimeError("serialized state tensor keys changed on round trip")
        for name, expected in state_arrays.items():
            actual = restored[name]
            if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                raise RuntimeError(
                    f"serialized state tensor changed on round trip: {name}"
                )
        for index, expected_hash in enumerate(
            state_arrays["observation_sha256"].astype(str)
        ):
            measured_hash = _observation_sha256(
                restored["map_channels"][index],
                restored["numeric_features"][index],
                int(restored["goal_index"][index]),
            )
            if measured_hash != expected_hash:
                raise RuntimeError(
                    f"serialized observation hash mismatch at row {index}"
                )
    result["states_artifact"] = {
        "path": str(states_output),
        "sha256": _sha256(states_output),
        "exact_tensor_round_trip_verified": True,
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in state_arrays.items()
        },
    }
    result["capacity_after_state_write"] = _capacity_snapshot()
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return {
        "output": str(output),
        "output_sha256": _sha256(output),
        "states_output": str(states_output),
        "states_sha256": _sha256(states_output),
        "unique_observations": result["state_pool"][
            "unique_unambiguous_observations"
        ],
        "ambiguous_observations": result["state_pool"][
            "ambiguous_observations_excluded"
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(FREEZE_RUN_ROOT))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--states-output", default=str(STATES_PATH))
    parser.add_argument("--expected-execution-sha", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    arguments = parser.parse_args()
    result, state_arrays = analyze(
        arguments.run_root,
        expected_execution_sha=arguments.expected_execution_sha,
        expected_manifest_sha256=arguments.expected_manifest_sha256,
    )
    written = _write_outputs(
        result, state_arrays, arguments.output, arguments.states_output
    )
    print(json.dumps(written, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
