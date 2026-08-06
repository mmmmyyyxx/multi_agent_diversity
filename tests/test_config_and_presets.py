from dataclasses import asdict

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from multi_dataset_diverse_rl.protocol import (
    EXPECTED_ADJACENT_MODULE,
    LEGACY_CONTROL_SETTINGS,
    MAIN_ABLATION_SETTINGS,
    AUXILIARY_SEARCH_CONTROL_SETTINGS,
    candidate_budget_contract,
    changed_ablation_modules,
    experiment_protocol,
)
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.versions import (
    CHECKPOINT_VERSION,
    EXPERIMENT_MATRIX_VERSION,
    METHOD_VERSION,
    PROTOCOL_RESOLUTION_VERSION,
)
from scripts.experiment_config import DEFAULT_EXPERIMENT_SETTING_NAMES, select_settings


def identity(setting="shared_full_dual_target_rcru"):
    return RunIdentity(
        method_version=METHOD_VERSION,
        experiment_setting=setting,
        git_commit="test",
        git_dirty=False,
        config_fingerprint="config",
        manifest_sha256="manifest",
        train_file_sha256="train",
        val_file_sha256="val",
        test_file_sha256="test",
        train_question_set_hash="train-q",
        val_question_set_hash="val-q",
        test_question_set_hash="test-q",
    )


def budget(name):
    return candidate_budget_contract(
        name,
        candidates_per_target_branch=2,
        stage_b_budget_per_branch=2,
        stage_a_channel_top_k=2,
        representative_size=12,
        coverage_size=6,
        conversion_size=6,
        preservation_size=4,
    )


def protocols():
    return {
        name: experiment_protocol(
            name,
            initialization_mode="shared_identical",
            tie_policy="abstain",
            candidate_budget_contract=budget(name),
        )
        for name in MAIN_ABLATION_SETTINGS
    }


def test_config_is_sectioned_and_canonical_defaults_are_explicit():
    cfg = Config()
    assert cfg.training.method_version == "member_aware_peer_state_v13"
    assert cfg.training.experiment_setting == "shared_full_dual_target_rcru"
    assert cfg.training.allow_legacy_setting is False
    assert cfg.training.allow_auxiliary_setting is False
    assert cfg.responsibility.responsibility_mode == "single_service_member_aware_v13"
    assert cfg.tcs.proposal_memory_mode == "off"
    assert cfg.training.initialization_mode == "shared_identical"
    assert cfg.peer_state.vote_tie_break == "abstain"
    assert cfg.models.solver_max_tokens == 1800
    assert cfg.tcs.num_candidates_per_parent == 2
    assert cfg.evaluation.stage_b_candidate_budget == 2
    with pytest.raises(AttributeError):
        _ = cfg.method_version


def test_main_ablation_settings_are_exact_and_legacy_names_fail_closed():
    assert tuple(DEFAULT_EXPERIMENT_SETTING_NAMES) == MAIN_ABLATION_SETTINGS
    assert MAIN_ABLATION_SETTINGS == (
        "shared_static_reference",
        "shared_generic_evolution",
        "shared_member_aware_dual_target",
        "shared_responsibility_conditioned_dual_target",
        "shared_full_dual_target_rcru",
    )
    assert len(DEFAULT_EXPERIMENT_SETTING_NAMES) == 5
    assert not set(AUXILIARY_SEARCH_CONTROL_SETTINGS) & set(
        DEFAULT_EXPERIMENT_SETTING_NAMES
    )
    assert not any(
        "vote_first" in name or "member_first" in name
        for name in DEFAULT_EXPERIMENT_SETTING_NAMES
    )
    for old in (
        "shared_independent_accuracy",
        "shared_peer_state_vote_first",
        "shared_peer_state_member_first_safe",
        "shared_member_aware_full",
        "shared_baseline",
        "shared_vote_state_diagnosis",
        "shared_repairability_adjusted_responsibility",
        "shared_dual_target_competition",
        "shared_member_aware_responsibility",
        "shared_responsibility_conditioned_evolution",
        "shared_full_rcru",
    ):
        with pytest.raises(ValueError, match="allow_legacy_setting=1"):
            select_settings(old)
    with pytest.raises(ValueError, match="Unknown experiment setting"):
        select_settings("shared_v9_sequential_accuracy")


def test_legacy_controls_require_explicit_opt_in_and_keep_legacy_identity():
    selected = select_settings(
        "shared_peer_state_member_pareto",
        allow_legacy_setting=True,
    )
    assert selected[0].name == "legacy_shared_peer_state_member_first_safe_v11"
    assert selected[0].name in LEGACY_CONTROL_SETTINGS
    protocol = experiment_protocol(
        "shared_peer_state_member_pareto",
        initialization_mode="shared_identical",
        tie_policy="abstain",
        candidate_budget_contract=budget(
            "legacy_shared_peer_state_member_first_safe_v11"
        ),
        allow_legacy_setting=True,
    )
    assert protocol.legacy_protocol
    assert protocol.requested_name == "shared_peer_state_member_pareto"
    assert protocol.name == "legacy_shared_peer_state_member_first_safe_v11"
    assert protocol.candidate_ranking_policy == "member_first_safe"
    old_seven = select_settings(
        "shared_dual_target_competition",
        allow_legacy_setting=True,
    )
    assert old_seven[0].name == (
        "legacy_v13_shared_dual_target_competition"
    )
    assert old_seven[0].name in LEGACY_CONTROL_SETTINGS
    old_budget = budget(old_seven[0].name)
    assert (
        old_budget.target_branch_count,
        old_budget.candidates_per_target_branch,
    ) == (2, 2)


def test_auxiliary_budget_controls_are_explicit_and_outside_main_matrix():
    for name, expected in zip(
        AUXILIARY_SEARCH_CONTROL_SETTINGS,
        ((2, 1, 2), (1, 4, 4)),
        strict=True,
    ):
        with pytest.raises(ValueError, match="allow_auxiliary_setting=1"):
            select_settings(name)
        selected = select_settings(name, allow_auxiliary_setting=True)
        assert selected[0].name == name
        protocol = experiment_protocol(
            name,
            initialization_mode="shared_identical",
            tie_policy="abstain",
            candidate_budget_contract=budget(name),
            allow_auxiliary_setting=True,
        )
        actual = protocol.candidate_budget_contract
        assert (
            actual.target_branch_count,
            actual.candidates_per_target_branch,
            actual.total_generated_candidates_per_update,
        ) == expected
        assert actual.stage_b_budget_per_branch == (
            actual.candidates_per_target_branch
        )
        assert actual.total_stage_b_candidate_budget == (
            actual.total_generated_candidates_per_update
        )
        assert protocol.auxiliary_protocol
        assert name not in MAIN_ABLATION_SETTINGS


def test_main_candidate_budget_is_fixed_and_runtime_rejects_config_drift(
    tmp_path,
):
    actual = candidate_budget_contract(
        "shared_member_aware_dual_target",
        candidates_per_target_branch=9,
        stage_b_budget_per_branch=11,
        stage_a_channel_top_k=2,
        representative_size=12,
        coverage_size=6,
        conversion_size=6,
        preservation_size=4,
    )
    assert (
        actual.target_branch_count,
        actual.candidates_per_target_branch,
        actual.stage_b_budget_per_branch,
        actual.total_generated_candidates_per_update,
        actual.total_stage_b_candidate_budget,
    ) == (2, 2, 2, 4, 4)
    with pytest.raises(ValueError, match="exactly two generated"):
        PromptEnsembleOptimizationSystem(Config.from_flat(
            out_dir=str(tmp_path),
            num_candidates_per_parent=1,
            stage_b_candidate_budget=1,
        ))


def test_static_and_reduced_main_candidate_budgets_are_exact():
    rows = protocols()
    static = rows["shared_static_reference"]
    assert static.optimization_enabled is False
    assert static.module_vector is None
    assert asdict(static.candidate_budget_contract) == {
        "target_branch_count": 0,
        "candidates_per_target_branch": 0,
        "total_generated_candidates_per_update": 0,
        "stage_b_budget_per_branch": 0,
        "total_stage_b_candidate_budget": 0,
        "stage_a_channel_top_k": 2,
        "representative_size": 12,
        "coverage_size": 6,
        "conversion_size": 6,
        "preservation_size": 4,
    }
    assert [
        (
            row.target_branch_count,
            row.candidates_per_target_branch,
            row.candidate_budget_contract.total_generated_candidates_per_update,
            row.candidate_budget_contract.total_stage_b_candidate_budget,
        )
        for row in rows.values()
    ] == [
        (0, 0, 0, 0),
        (1, 2, 2, 2),
        (2, 2, 4, 4),
        (2, 2, 4, 4),
        (2, 2, 4, 4),
    ]


def test_module_vectors_and_adjacent_diffs_are_exact():
    rows = protocols()
    assert rows["shared_static_reference"].module_vector is None
    assert [
        tuple(int(value) for value in rows[name].modules.as_tuple())
        for name in MAIN_ABLATION_SETTINGS[1:]
    ] == [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
    ]
    for left, right, expected in EXPECTED_ADJACENT_MODULE:
        assert changed_ablation_modules(rows[left], rows[right]) == (expected,)


def test_s0_through_s2_share_safety_policy_and_no_hidden_stage_a():
    rows = protocols()
    shared = [rows[name] for name in MAIN_ABLATION_SETTINGS[1:4]]
    assert {
        (
            row.candidate_acceptance_policy,
            row.candidate_ranking_policy,
            row.stage_a_policy,
        )
        for row in shared
    } == {
        (
            "fixed_peer_monotone_target_or_vote",
            "common_monotone_safe",
            "matched_all_generated",
        )
    }
    assert rows["shared_full_dual_target_rcru"].stage_a_policy == (
        "matched_all_generated"
    )
    assert [row.target_branch_count for row in rows.values()] == [
        0, 1, 2, 2, 2
    ]


def test_adjacent_low_level_protocol_isolation():
    rows = protocols()
    static = asdict(rows["shared_static_reference"])
    s0 = asdict(rows["shared_generic_evolution"])
    s1 = asdict(rows["shared_member_aware_dual_target"])
    s2 = asdict(rows["shared_responsibility_conditioned_dual_target"])
    s3 = asdict(rows["shared_full_dual_target_rcru"])
    ignored = {"name", "requested_name", "display_name", "modules"}

    def diff(left, right):
        return {
            key for key in left
            if key not in ignored and left[key] != right[key]
        }

    assert diff(static, s0) == {
        "optimization_enabled",
        "target_selection_policy",
        "sample_pool_policy",
        "tcs_context_policy",
        "candidate_acceptance_policy",
        "candidate_ranking_policy",
        "stage_a_policy",
        "candidate_budget_contract",
    }
    assert diff(s0, s1) == {
        "target_selection_policy",
        "sample_pool_policy",
        "tcs_context_policy",
        "responsibility_refresh_policy",
        "service_routing_enabled",
        "candidate_budget_contract",
    }
    assert diff(s1, s2) == {"tcs_context_policy"}
    assert diff(s2, s3) == {
        "candidate_acceptance_policy",
        "candidate_ranking_policy",
    }


def test_run_metadata_records_module_fingerprint(tmp_path):
    system = PromptEnsembleOptimizationSystem(Config.from_flat(out_dir=str(tmp_path)))
    system.set_run_identity(identity())
    metadata = system.run_meta()
    assert metadata["method_version"] == METHOD_VERSION
    assert metadata["checkpoint_version"] == CHECKPOINT_VERSION == 22
    assert metadata["experiment_matrix_version"] == EXPERIMENT_MATRIX_VERSION
    assert metadata["protocol_resolution_version"] == PROTOCOL_RESOLUTION_VERSION
    assert metadata["setting_index"] == 4
    assert metadata["setting_display_name"] == (
        "S3 Robust Contribution Update (Full)"
    )
    assert metadata["module_vector"] == {
        "member_aware_dual_target_search": True,
        "responsibility_conditioned_evolution": True,
        "robust_contribution_update": True,
    }
    assert metadata["added_module_vs_previous"] == "robust_contribution_update"
    assert metadata["candidate_acceptance_policy"] == (
        "responsibility_robust_contribution"
    )
    assert metadata["candidate_ranking_policy"] == (
        "responsibility_contribution_pareto"
    )
    assert metadata["stage_a_policy"] == "matched_all_generated"
    assert metadata["legacy_compatibility_enabled"] is False


def test_invalid_module_combinations_fail_without_auto_completion():
    from multi_dataset_diverse_rl.protocol import (
        AblationModules,
        validate_ablation_modules,
    )

    invalid = (
        AblationModules(False, True, False),
        AblationModules(True, False, True),
    )
    for modules in invalid:
        with pytest.raises(ValueError):
            validate_ablation_modules(modules)


def test_initialization_modes_are_explicit_and_five_prompt_bounded(tmp_path):
    shared = PromptEnsembleOptimizationSystem(
        Config.from_flat(out_dir=str(tmp_path / "shared"))
    )
    assert len({agent.initial_prompt for agent in shared.agents}) == 1
    supplied = PromptEnsembleOptimizationSystem(Config.from_flat(
        out_dir=str(tmp_path / "supplied"),
        initialization_mode="provided_prompt_set",
        provided_prompts_json='["p0", "p1", "p2", "p3", "p4"]',
    ))
    assert [agent.initial_prompt for agent in supplied.agents] == [
        "p0", "p1", "p2", "p3", "p4"
    ]
    with pytest.raises(ValueError, match="exactly five"):
        PromptEnsembleOptimizationSystem(Config.from_flat(
            out_dir=str(tmp_path / "bad"),
            initialization_mode="provided_prompt_set",
            provided_prompts_json='["p0"]',
        ))


def test_formal_system_rejects_non_abstain_tie_policy(tmp_path):
    with pytest.raises(ValueError, match="tie-as-abstain"):
        PromptEnsembleOptimizationSystem(Config.from_flat(
            out_dir=str(tmp_path), vote_tie_break="random",
        ))
