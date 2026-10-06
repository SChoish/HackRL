import copy
import importlib.util
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from hackrl.pack_restore import (
    PackRestoreAction,
    PackRestorePhase,
    PackRestoreStart,
    PackRestoreVariant,
    make_pack_restore_state,
    pack_restore_step,
)


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "analyze_pack_restore_common_state_teachers",
        REPOSITORY / "scripts/analyze_pack_restore_common_state_teachers.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ANALYSIS = _load()


def test_manifest_locks_a_no_learning_same_seed_cross_evaluation():
    manifest = json.loads(
        (REPOSITORY / ANALYSIS.MANIFEST_PATH).read_text(encoding="utf-8")
    )
    assert manifest["no_learning_contract"]["training_or_optimizer_updates"] == 0
    assert manifest["no_learning_contract"]["checkpoint_access"] == "read_only"
    assert manifest["inputs"]["seeds"] == list(range(30, 40))
    assert tuple(manifest["state_pool"]["source_policies"]) == ANALYSIS.SOURCE_POLICIES
    assert tuple(manifest["cross_evaluation"]["evaluators"]) == ANALYSIS.EVALUATORS
    assert tuple(manifest["state_pool"]["target_actions"]) == tuple(
        action.name for action in ANALYSIS.TARGET_ACTIONS
    )
    assert manifest["statistics"]["independent_unit"] == "learner seed"
    assert manifest["statistics"]["state_or_episode_is_not_a_replication_unit"]
    assert manifest["statistics"]["bootstrap_resamples"] == ANALYSIS.BOOTSTRAP_RESAMPLES
    assert manifest["implementation"]["execution_sources"] == [
        path.as_posix() for path in ANALYSIS.EXECUTION_SOURCES
    ]
    assert manifest["outputs"]["result"] == str(ANALYSIS.OUTPUT_PATH)
    assert manifest["outputs"]["exact_state_tensors"] == str(ANALYSIS.STATES_PATH)
    ANALYSIS._validate_manifest_contract(manifest)


def test_manifest_contract_rejects_a_runtime_design_change():
    manifest = json.loads(
        (REPOSITORY / ANALYSIS.MANIFEST_PATH).read_text(encoding="utf-8")
    )
    changed = copy.deepcopy(manifest)
    changed["state_pool"]["horizon"] += 1
    with pytest.raises(RuntimeError, match="manifest contract mismatch"):
        ANALYSIS._validate_manifest_contract(changed)


def test_manifest_authenticates_completed_prior_artifacts_and_checkpoints():
    manifest = json.loads(
        (REPOSITORY / ANALYSIS.MANIFEST_PATH).read_text(encoding="utf-8")
    )
    artifacts, checkpoints = ANALYSIS._verify_prior_artifacts(
        ANALYSIS.FREEZE_RUN_ROOT, manifest
    )
    assert set(artifacts) == {"result", "teacher_trace", "run_manifest"}
    assert len(checkpoints) == 30
    assert set(checkpoints) == {
        (label, seed)
        for label in ("D-off", "D-frozen", "D-on")
        for seed in ANALYSIS.SEEDS
    }


def test_target_stage_opportunities_are_environment_defined_not_policy_selected():
    state = make_pack_restore_state(
        0,
        int(PackRestorePhase.LOADED),
        start=PackRestoreStart.PATH_CHECK,
    )
    initial = np.asarray(ANALYSIS._eligibility_for_state(state), dtype=bool)
    assert initial.tolist() == [True, True, False, False]

    recorded = pack_restore_step(
        state, int(PackRestoreAction.MAKE_RECORD), PackRestoreVariant.MUTANT
    )
    after_record = np.asarray(
        ANALYSIS._eligibility_for_state(recorded), dtype=bool
    )
    assert after_record.tolist() == [False, True, False, True]

    packed = pack_restore_step(
        recorded, int(PackRestoreAction.PACK_STORAGE), PackRestoreVariant.MUTANT
    )
    after_pack = np.asarray(ANALYSIS._eligibility_for_state(packed), dtype=bool)
    # An adjacent empty unpack site can also be packed at this point.
    assert after_pack.tolist() == [False, True, True, False]

    rebuilt = pack_restore_step(
        packed, int(PackRestoreAction.REBUILD_EMPTY), PackRestoreVariant.MUTANT
    )
    after_rebuild = np.asarray(
        ANALYSIS._eligibility_for_state(rebuilt), dtype=bool
    )
    assert after_rebuild.tolist() == [False, True, False, True]


def test_observation_hash_uses_exact_model_inputs_and_goal():
    maps = np.zeros((7, 9, 4), dtype=np.float32)
    numeric = np.zeros((18,), dtype=np.float32)
    one = ANALYSIS._observation_sha256(maps, numeric, 11)
    assert one == ANALYSIS._observation_sha256(maps.copy(), numeric.copy(), 11)
    changed = numeric.copy()
    changed[3] = 1.0
    assert one != ANALYSIS._observation_sha256(maps, changed, 11)
    assert one != ANALYSIS._observation_sha256(maps, numeric, 10)


def test_non_target_comparator_excludes_all_four_stage_actions():
    assert set(ANALYSIS.TARGET_ACTION_INDICES).isdisjoint(
        ANALYSIS.NON_TARGET_ACTION_INDICES
    )
    assert set(ANALYSIS.TARGET_ACTION_INDICES) | set(
        ANALYSIS.NON_TARGET_ACTION_INDICES
    ) == set(range(ANALYSIS.NUM_ACTIONS))


def test_bootstrap_uses_seed_values_as_inputs():
    result = ANALYSIS._bootstrap(
        [0.0, 0.5, 1.0], np.random.default_rng(ANALYSIS.BOOTSTRAP_SEED)
    )
    assert result["n_seeds"] == 3
    assert result["mean"] == 0.5
    assert len(result["bootstrap_95_percentile_interval"]) == 2


def test_seed_metric_rows_weight_each_eligible_state_action_pair_once():
    evaluations = {}
    for label in ANALYSIS.EVALUATORS:
        evaluations[label] = {
            "teacher_recommendation": "REBUILD_EMPTY",
            "policy_mode_action": "REBUILD_EMPTY",
            "eligible_stage_metrics": {
                "REBUILD_EMPTY": {
                    "teacher_q": 0.8,
                    "teacher_q_margin_vs_best_non_target": 0.2,
                    "policy_probability": 0.7,
                    "policy_probability_margin_vs_best_non_target": 0.3,
                }
            },
        }
    records = [
        {
            "seed": 30,
            "eligible_target_actions": ["REBUILD_EMPTY"],
            "evaluations": evaluations,
        }
    ]
    rows = ANALYSIS._seed_metric_rows(records)
    rebuild = next(
        row
        for row in rows
        if row["evaluator"] == "D-on"
        and row["eligible_action"] == "REBUILD_EMPTY"
    )
    overall = next(
        row
        for row in rows
        if row["evaluator"] == "D-on"
        and row["eligible_action"] == "ALL_TARGET_STAGE_PAIRS"
    )
    assert rebuild["state_action_pairs"] == overall["state_action_pairs"] == 1
    assert rebuild["teacher_recommendation_rate"] == 1.0
    assert rebuild["policy_mode_selection_rate"] == 1.0
    assert jnp.isclose(rebuild["teacher_q"], 0.8)


def test_visitation_aggregation_is_seed_paired_and_rate_based():
    visitation = []
    for seed in ANALYSIS.SEEDS:
        for label, value in (("pretrain", 0.1), ("D-on", 0.4)):
            values = {
                "eligible_state_rate": value,
                "episode_opportunity_rate": value + 0.1,
                "policy_selection_given_opportunity": value + 0.2,
            }
            visitation.append(
                {
                    "seed": seed,
                    "source_policy": label,
                    **values,
                    "by_action": {"REBUILD_EMPTY": dict(values)},
                }
            )
    seed_rows, summaries, contrasts = ANALYSIS._aggregate_visitation(visitation)
    assert len(seed_rows) == 40
    summary = next(
        row
        for row in summaries
        if row["source_policy"] == "D-on"
        and row["eligible_action"] == "REBUILD_EMPTY"
        and row["metric"] == "eligible_state_rate"
    )
    assert summary["n_seeds"] == 10
    assert np.isclose(summary["mean"], 0.4)
    contrast = next(
        row
        for row in contrasts
        if row["contrast"] == "D-on_minus_pretrain"
        and row["eligible_action"] == "REBUILD_EMPTY"
        and row["metric"] == "eligible_state_rate"
    )
    assert contrast["n_seeds"] == 10
    assert np.isclose(contrast["mean"], 0.3)
    assert len(contrast["seed_points"]) == 10
