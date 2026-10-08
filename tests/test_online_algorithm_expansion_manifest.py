import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    REPOSITORY / "docs/manifests/online_algorithm_expansion_v1.json"
)


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_phase_1_has_no_execution_authority_and_preserves_source_boundary():
    manifest = _manifest()
    assert manifest["schema_version"] == "hackrl_online_algorithm_expansion_v1"
    assert manifest["manifest_id"] == "online_algorithm_expansion_v1"
    assert manifest["status"] == "phase_1_design_frozen_no_training_authority"
    authority = manifest["phase_1_authority"]
    assert "does not authorize training" in manifest["understood_as"]
    for key in (
        "training_authorized",
        "queue_launch_authorized",
        "checkpoint_writes_authorized",
        "dependency_installation_authorized",
        "cpu_smoke_authorized_by_this_manifest",
        "gpu_smoke_authorized_by_this_manifest",
    ):
        assert authority[key] is False

    completion = manifest["phase_1_completion"]
    assert completion["phase_1_design_artifacts_completed"] is True
    assert completion["experiment_execution_started"] is False
    sources = {
        item["repository"]: item for item in manifest["provenance"]["repositories"]
    }
    excluded = sources["coldsummerday/SD-SAC"]
    assert excluded["sha"] == "006809db08513fbeea2bfaecea1c03b59c3bb0d5"
    assert excluded["boundary"] == "provenance/reference boundary only"
    assert excluded["implementation_text_use"].startswith("forbidden")
    assert excluded["external_implementation_text_inspected_in_this_phase"] is False
    assert sources["MichaelTMatthews/purejaxgcrl"]["license"] == "MIT"
    assert sources["mttga/purejaxql"]["license"] == "Apache-2.0"
    assert sources["mttga/purejaxql"][
        "external_implementation_text_inspected_in_this_phase"
    ] is True
    assert all(
        not item["vendored_in_this_phase"]
        for item in manifest["provenance"]["repositories"]
    )

    ledger = manifest["autobahn_scope_ledger"]
    assert ledger["status"].startswith("closed")
    assert ledger["items"] == [
        {
            "item": (
                "coldsummerday/SD-SAC implementation text at "
                "006809db08513fbeea2bfaecea1c03b59c3bb0d5"
            ),
            "risk_class": "bright-line licensing/IP boundary",
            "verdict": "descoped",
            "reason": (
                "the pinned GitHub repository exposed no explicit license in "
                "repository metadata or its file tree at audit time"
            ),
            "safe_alternative": (
                "independent specification and implementation from "
                "arXiv:1910.07207 and arXiv:2209.10081, without consulting "
                "that repository's implementation text"
            ),
        }
    ]


def test_matrix_seeds_and_cell_local_gate_are_frozen():
    manifest = _manifest()
    matrix = manifest["matrix"]
    assert matrix["methods"] == [
        "GC-PQN",
        "LEO",
        "Dual LEO(PQN)",
        "GC-SD-SAC",
    ]
    assert matrix["environments"] == ["TICK-CLAIM", "PACK-RESTORE"]
    assert matrix["method_environment_cells"] == (
        len(matrix["methods"]) * len(matrix["environments"])
    )
    assert manifest["development"]["learner_seeds"] == [110, 111]
    assert manifest["final_exploratory"]["learner_seeds"] == [120, 121, 122]
    assert "do not imply paired initialization" in manifest["final_exploratory"][
        "cross_algorithm_seed_note"
    ]

    gate = manifest["gate"]
    assert gate["scope"] == "cell-local method x environment"
    assert gate["threshold_inclusive"] == 0.9
    assert gate["required_development_seeds"] == [110, 111]
    assert "each" in gate["rule"]
    assert gate["failure_blocks_other_cells"] is False
    assert gate["common_setup_role"].startswith("diagnostic only")
    assert gate["automatic_main_launch"] is False


def test_configuration_selection_is_normal_only_and_capped():
    manifest = _manifest()
    development = manifest["development"]
    contract = development["configuration_contract"]
    maximum = contract["maximum_predeclared_configurations_per_method"]
    assert maximum == 4
    assert set(contract["registries"]) == set(manifest["matrix"]["methods"])
    assert all(configs == [] for configs in contract["registries"].values())
    assert all(len(configs) <= maximum for configs in contract["registries"].values())
    assert "shared normal-only candidate registry per method" in contract["cap_scope"]
    assert "not multiplied by environment" in contract["cap_scope"]

    selection = development["selection"]
    assert selection["training_kernel"] == "fixed normal only"
    assert selection["evaluation_kernel"] == "fixed normal only"
    assert selection["evaluation_start"] == "natural-start only"
    assert selection["primary_metric"] == "delivery success"
    assert selection["tie_breaker"] == "fixed-normal return"
    assert selection["bug_or_exploitation_metric_may_select"] is False
    assert selection["mutant_result_may_select"] is False
    assert selection["final_evaluation_layout_may_select"] is False
    assert selection["common_setup_may_select_or_substitute"] is False


def test_full_new_method_budget_arithmetic():
    manifest = _manifest()
    budget = manifest["budget"]
    assert budget["transition_unit"].startswith("physical environment transitions")
    assert budget["transition_total_scope"].startswith(
        "declared totals cover training transitions only"
    )
    symbols = budget["symbols"]
    p = symbols["P_pretraining_transitions_per_seed"]
    a = symbols["A_adaptation_transitions_per_branch_per_seed"]
    assert p == 16_777_216
    assert a == 134_217_728

    cell = budget["per_passing_method_environment_cell"]
    seeds = cell["learner_seeds"]
    branches = len(cell["adaptation_branches"])
    assert cell["jobs_per_seed"] == 1 + branches == 3
    assert cell["jobs"] == seeds * cell["jobs_per_seed"] == 9
    assert cell["transitions_per_seed"] == p + branches * a == 285_212_672
    assert cell["transitions"] == seeds * (p + branches * a) == 855_638_016

    full = budget["all_eight_new_method_cells"]
    cells = manifest["matrix"]["method_environment_cells"]
    assert full["pretraining_jobs"] == cells * seeds == 24
    assert full["adaptation_jobs"] == cells * seeds * branches == 48
    assert full["jobs"] == cells * cell["jobs"] == 72
    assert full["pretraining_transitions"] == full["pretraining_jobs"] * p
    assert full["adaptation_transitions"] == full["adaptation_jobs"] * a
    assert full["transitions"] == cells * cell["transitions"] == 6_845_104_128
    assert full["transitions"] == (
        full["pretraining_transitions"] + full["adaptation_transitions"]
    )

    development = budget["maximum_normal_only_development"]
    assert development["jobs"] == 4 * 4 * 2 * 2 == 64
    assert development["transitions"] == development["jobs"] * p
    combined = budget["maximum_development_plus_full_main"]
    assert combined["jobs"] == development["jobs"] + full["jobs"] == 136
    assert combined["transitions"] == (
        development["transitions"] + full["transitions"]
    ) == 7_918_845_952


def test_branch_state_compute_and_evaluation_contracts_are_explicit():
    manifest = _manifest()
    cloned = set(manifest["branch_fork"]["clone_exactly"])
    for required in (
        "parameters",
        "optimizer state",
        "normalization state",
        "RNG state",
        "replay state when applicable",
    ):
        assert required in cloned
    assert manifest["branch_fork"]["no_cross_branch_state_sharing"] is True
    assert manifest["branch_fork"]["branch_kernel_mapping"] == {
        "fixed": "continue adaptation on the fixed kernel",
        "mutant": "adapt on the mutant kernel",
    }
    assert "exact RNG state" in manifest["branch_fork"]["rng_semantics"]
    assert "do not claim episode-wise common random numbers" in manifest[
        "branch_fork"
    ]["rng_nonclaim"]
    assert "clone identical" in manifest["algorithm_contracts"]["GC-SD-SAC"][
        "replay_retention"
    ]

    logging = set(manifest["compute_logging"]["required_per_seed_branch_fields"])
    assert {
        "physical_transitions",
        "valid_transitions",
        "td_targets",
        "scheduled_gradient_steps",
        "applied_gradient_steps",
        "update_to_data_ratio",
        "wall_time_seconds",
        "parameter_counts_by_component",
    } <= logging

    assert manifest["evaluation"]["adaptation_transition_checkpoints"] == [
        0,
        1_048_576,
        4_194_304,
        8_388_608,
        16_777_216,
        33_554_432,
        67_108_864,
        134_217_728,
    ]
    primary = manifest["evaluation"]["primary_final_view"]
    assert primary["start"] == "natural"
    assert primary["action_selection"] == "deterministic/greedy"

    dual = manifest["algorithm_contracts"]["Dual LEO(PQN)"]
    assert "combines LEO and UVFA Q-values" in dual["behavior"]
    assert dual["default_acting_equation"] == "Q_act = 0.7*Q_UVFA + 0.3*Q_LEO"
    assert "own TD objectives" in dual["family"]
    assert dual["ppo_style_behavior_cloning"].startswith("not part")
    assert all("imitation" not in item for item in dual["learning_state"])
    assert dual["pqn_target_network"].startswith("none")

    pqn = manifest["algorithm_contracts"]["GC-PQN"]
    assert "neither a replay buffer nor a target network" in pqn["family"]
    assert all("target parameters" not in item for item in pqn["learning_state"])
    assert "Q(lambda)" in pqn["targets"]

    sd_sac = manifest["algorithm_contracts"]["GC-SD-SAC"]
    assert "entropy-penalty" in sd_sac["family"]
    assert "double-average" in sd_sac["family"]
    assert "Q-clip" in sd_sac["family"]
    assert "average of the two target-critic" in sd_sac["critic_target"]
    assert "minimum" not in sd_sac["critic_target"]
    assert "H_old" in sd_sac["stored_entropy"]


def test_leakage_baseline_storage_and_preflight_guards():
    manifest = _manifest()
    audit = manifest["mandela_leakage_audit"]
    assert {hit["pattern"] for hit in audit["likely_hits"]} == {
        "Verifier = designer",
        "Tautology",
    }
    fixes = set(audit["independence_fixes"])
    assert "normal-only development selection" in fixes
    assert "no bug metric in tuning" in fixes
    assert "kernel/oracle validation separate from learned-policy outcome" in fixes

    baseline = manifest["baseline_policy"]
    assert baseline["PPO_rerun_authorized"] is False
    assert baseline["Dual_PPO_rerun_authorized"] is False
    assert "Git SHA" in baseline["historical_baseline_rule"]
    assert "evaluation-contract" in baseline["historical_baseline_rule"]
    assert "working-tree changes" in baseline["existing_Dual_result_boundary"]
    assert "same-code simultaneous baseline" in baseline[
        "existing_Dual_result_boundary"
    ]

    compatibility = manifest["dependency_api_compatibility"]
    assert compatibility["current_offrl_environment"] == {
        "jax": "0.10.2",
        "jaxlib": "0.10.2",
        "flax": "0.12.7",
        "optax": "0.2.8",
        "chex": "0.1.92",
        "craftax": "1.6.1",
    }
    constraints = {
        item["source"]: item["constraint"]
        for item in compatibility["upstream_declared_constraints"]
    }
    assert constraints[
        "MichaelTMatthews/purejaxgcrl@eafccd995ac2736df5fbc05ac4bb6b5d58cfcce7"
    ] == "jax[cuda12]==0.6.0 and orbax-checkpoint==0.5.0"
    assert constraints[
        "mttga/purejaxql@47af6d7b35c89ddfe633aaf7341bdb8964cb7cce"
    ] == "jax and jaxlib <=0.4.38"
    assert compatibility["dependency_installation_authorized"] is False
    assert compatibility["environment_mutation_authorized"] is False

    storage = manifest["storage_contract"]
    assert storage["default_run_path"] == "/raid/ext_csv/HackRL/runs/<run-id>"
    assert storage["live_tree_move_forbidden"] is True
    assert storage["safety_reserve_formula"] == (
        "max(8 GiB, 20% of projected remaining writes before adding the reserve)"
    )
    assert "above 90%" in storage["home_rejection_rule"]
    assert storage["migration_sequence"][0] == (
        "stop at a completed resumable checkpoint"
    )
    assert storage["migration_sequence"][-1] == (
        "only then remove the source copy"
    )

    assert [item["step"] for item in manifest["preflight_order"]] == [
        "upstream/adaptation audit",
        "implementation and unit tests",
        "CPU minimal smoke",
        "512x64 GPU one-update smoke per method x environment",
        "capacity forecast",
        "normal-only development",
    ]
    assert [item["stage"] for item in manifest["authorization_stages"]] == [
        "Phase-1 design",
        "implementation and unit tests",
        "CPU minimal smoke",
        "per-cell 512x64 GPU one-update smoke and capacity forecast",
        "normal-only development",
        "per-cell main",
    ]
    assert manifest["authorization_stages"][0]["authority_in_this_manifest"] is True
    assert all(
        item["authority_in_this_manifest"] is False
        for item in manifest["authorization_stages"][1:]
    )
    implementation_stage = manifest["authorization_stages"][1]
    assert "not required before it" in implementation_stage["entry"]
    upstream_exit = manifest["preflight_order"][0]["exit_requirement"]
    implementation_exit = manifest["preflight_order"][1]["exit_requirement"]
    assert "resolved implementation hash" not in upstream_exit
    assert "resolved hashes" in implementation_exit
    assert "not a circular prerequisite" in manifest[
        "preflight_authorization_rule"
    ]
    assert manifest["main_opening_rule"].startswith("Main stays closed")

    environments = manifest["environment_contracts"]
    assert environments["TICK-CLAIM"] == "docs/manifests/tick_claim_v1.json"
    assert environments["PACK-RESTORE"] == "docs/manifests/pack_restore_v1.json"
    assert "not executable" in environments["phase_1_executability"]


def test_measurement_categories_and_failed_seeds_cannot_be_collapsed():
    manifest = _manifest()
    taxonomy = manifest["measurement_taxonomy"]
    assert set(taxonomy["separate_categories"]) == {
        "normal_ability",
        "trigger",
        "exploit",
        "adaptation_effect",
        "actual_value",
        "discovery_interval",
        "maintenance",
        "failure",
    }
    assert taxonomy["no_category_substitution_or_pooling"] is True
    assert taxonomy["independent_unit"] == "learner seed"
    assert taxonomy["failed_seeds_stay_included"] is True
