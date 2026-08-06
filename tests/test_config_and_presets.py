from dataclasses import asdict
import json

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from multi_dataset_diverse_rl.protocol import (
    CandidateBudgetContract,
    canonical_candidate_selection_policy,
    experiment_protocol,
)
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.versions import CHECKPOINT_VERSION, METHOD_VERSION, TARGET_SELECTION_VERSION
from scripts.experiment_config import DEFAULT_EXPERIMENT_SETTING_NAMES, select_settings


def identity(setting="shared_member_aware_full"):
    return RunIdentity(
        method_version="member_aware_peer_state_v11",
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


def protocols():
    budget = CandidateBudgetContract(2, 2, 6, 12, 6, 6, 4)
    return {
        name: experiment_protocol(
            name,
            initialization_mode="shared_identical",
            tie_policy="abstain",
            candidate_budget_contract=budget,
        )
        for name in DEFAULT_EXPERIMENT_SETTING_NAMES
    }


def test_config_is_sectioned_and_canonical_defaults_are_explicit():
    cfg = Config()
    assert cfg.training.method_version == "member_aware_peer_state_v11"
    assert cfg.responsibility.responsibility_mode == "single_service_member_aware_v10"
    removed_wait_field = "responsibility_" + "max_wait_updates"
    removed_compensation_field = "member_" + "catchup_mode"
    assert not hasattr(cfg.responsibility, removed_wait_field)
    assert not hasattr(cfg.responsibility, removed_compensation_field)
    with pytest.raises(TypeError, match="Unknown Config fields"):
        Config.from_flat(**{removed_wait_field: 8})
    with pytest.raises(TypeError, match="Unknown Config fields"):
        Config.from_flat(**{removed_compensation_field: "off"})
    assert cfg.tcs.proposal_memory_mode == "off"
    assert cfg.training.initialization_mode == "shared_identical"
    assert cfg.peer_state.vote_tie_break == "abstain"
    assert cfg.models.solver_api_key_env == "DASHSCOPE_API_KEY"
    assert cfg.models.optimizer_api_key_env == "DASHSCOPE_API_KEY"
    assert cfg.models.evaluator_api_key_env == "DASHSCOPE_API_KEY"
    assert cfg.models.solver_base_url_env == "DASHSCOPE_BASE_URL"
    assert cfg.models.optimizer_base_url_env == "DASHSCOPE_BASE_URL"
    assert cfg.models.evaluator_base_url_env == "DASHSCOPE_BASE_URL"
    assert cfg.tcs.critic_json_max_retries == 1
    assert cfg.tcs.teacher_json_max_retries == 1
    assert cfg.tcs.teacher_critic_max_rounds == 2
    assert cfg.tcs.student_invalid_max_retries == 3
    assert cfg.tcs.student_upstream_regeneration_max_count == 1
    assert "validation_unique_state_cache_enabled" not in cfg.to_flat_dict()
    assert "test_evaluation_after_selection_only" not in cfg.to_flat_dict()
    assert cfg.tcs.tcs_max_pattern_summaries == 3
    assert cfg.tcs.tcs_max_evidence_cases == 3
    with pytest.raises(AttributeError):
        _ = cfg.method_version


def test_only_six_settings_exist_and_old_setting_fails():
    assert DEFAULT_EXPERIMENT_SETTING_NAMES == [
        "shared_baseline",
        "shared_independent_accuracy",
        "shared_peer_state_vote_first",
        "shared_peer_state_member_first_safe",
        "shared_member_aware_responsibility",
        "shared_member_aware_full",
    ]
    with pytest.raises(ValueError, match="Unknown experiment setting"):
        select_settings("shared_v9_sequential_accuracy")


def test_legacy_s3_alias_resolves_to_canonical_protocol_and_setting(tmp_path):
    budget = CandidateBudgetContract(2, 2, 6, 12, 6, 6, 4)
    legacy = experiment_protocol(
        "shared_peer_state_member_pareto",
        initialization_mode="shared_identical",
        tie_policy="abstain",
        candidate_budget_contract=budget,
    )
    canonical = experiment_protocol(
        "shared_peer_state_member_first_safe",
        initialization_mode="shared_identical",
        tie_policy="abstain",
        candidate_budget_contract=budget,
    )
    assert legacy.name == canonical.name == "shared_peer_state_member_first_safe"
    assert legacy.requested_name == "shared_peer_state_member_pareto"
    assert canonical.requested_name == canonical.name
    assert legacy.candidate_selection_policy == "member_first_safe"
    assert canonical_candidate_selection_policy("individual_accuracy") == (
        "individual_first_safe"
    )
    assert canonical_candidate_selection_policy("vote_first") == "vote_first_safe"
    assert canonical_candidate_selection_policy("member_aware_pareto") == (
        "member_first_safe"
    )
    assert select_settings("shared_peer_state_member_pareto")[0].name == canonical.name
    system = PromptEnsembleOptimizationSystem(Config.from_flat(
        out_dir=str(tmp_path),
        experiment_setting="shared_peer_state_member_pareto",
    ))
    system.set_run_identity(identity("shared_peer_state_member_first_safe"))
    metadata = system.run_meta()
    assert metadata["requested_experiment_setting"] == (
        "shared_peer_state_member_pareto"
    )
    assert metadata["canonical_experiment_setting"] == canonical.name
    assert metadata["config"]["experiment_setting"] == canonical.name


def test_ablation_protocols_are_field_isolated_and_budget_matched():
    rows = protocols()
    b2 = asdict(rows["shared_peer_state_vote_first"])
    member_first = asdict(rows["shared_peer_state_member_first_safe"])
    responsibility = asdict(rows["shared_member_aware_responsibility"])
    full = asdict(rows["shared_member_aware_full"])
    b2_member_first_differences = {
        key for key in b2 if b2[key] != member_first[key]
    }
    assert b2_member_first_differences == {
        "name", "requested_name", "candidate_selection_policy"
    }
    member_first_responsibility_differences = {
        key for key in member_first if member_first[key] != responsibility[key]
    }
    assert member_first_responsibility_differences == {
        "name",
        "requested_name",
        "target_selection_policy",
        "sample_pool_policy",
        "responsibility_refresh_policy",
        "repairability_freeze_enabled",
        "service_routing_enabled",
    }
    responsibility_full_differences = {
        key for key in responsibility if responsibility[key] != full[key]
    }
    assert responsibility_full_differences == {
        "name", "requested_name", "tcs_context_policy"
    }
    assert len({repr(row.candidate_budget_contract) for row in rows.values()}) == 1
    assert len({row.tie_policy for row in rows.values()}) == 1
    assert len({row.initialization_mode for row in rows.values()}) == 1
    assert {
        name: row.repairability_freeze_enabled for name, row in rows.items()
    } == {
        "shared_baseline": False,
        "shared_independent_accuracy": False,
        "shared_peer_state_vote_first": False,
        "shared_peer_state_member_first_safe": False,
        "shared_member_aware_responsibility": True,
        "shared_member_aware_full": True,
    }


def test_run_metadata_records_initialization_protocol_and_no_legacy_search(tmp_path):
    system = PromptEnsembleOptimizationSystem(Config.from_flat(out_dir=str(tmp_path)))
    system.set_run_identity(identity())
    metadata = system.run_meta()
    assert metadata["method_version"] == METHOD_VERSION
    assert metadata["initialization_mode"] == "shared_identical"
    assert metadata["initial_prompts_identical"] is True
    assert metadata["tie_policy"] == "abstain"
    assert metadata["generic_diversity_reward_used"] is False
    assert metadata["legacy_compatibility_enabled"] is False
    assert metadata["tcs_protocol_version"] == "assigned_residual_only_context_v1"
    assert metadata["candidate_acceptance_version"] == (
        "fixed_peer_monotone_target_or_vote_v2"
    )
    assert metadata["candidate_selection_version"] == (
        "member_first_safe_selection_v1"
    )
    assert metadata["candidate_selection_policy"] == "member_first_safe"
    assert metadata["acceptance_policy"] == "fixed_peer_monotone_target_or_vote"
    assert metadata["proposal_memory_mode"] == "off"
    assert metadata["preservation_policy_version"] == (
        "diagnostic_only_sample_preservation_v1"
    )
    assert metadata["evaluation_protocol_version"] == (
        "final_active_state_no_validation_v1"
    )
    assert metadata["checkpoint_selection_version"] == "none_final_state_v1"
    assert metadata["student_invalid_recovery_version"] == (
        "feedback_retry_then_upstream_regenerate_v1"
    )
    assert metadata["mutable_prompt_contract_version"] == (
        "reasoning_only_no_response_format_v2"
    )
    assert metadata["student_prompt_contract_version"] == (
        "mutable_reasoning_only_v2"
    )
    assert metadata["candidate_protocol_filter_version"] == (
        "output_contract_contamination_v2"
    )
    assert metadata["teacher_revision_protocol_version"] == (
        "critic_grounded_full_plan_revision_v1"
    )
    assert metadata["critic_approval_basis"] == "failed_checks_empty"
    assert metadata["diagnosis_aggregation_version"] == (
        "single_lane_pattern_aggregation_v1"
    )
    assert metadata["target_selection_version"] == TARGET_SELECTION_VERSION
    assert metadata["checkpoint_version"] == CHECKPOINT_VERSION
    assert metadata["task_general_scope"] == "unseen_examples_within_current_task"
    assert metadata["student_sample_memorization_filter"] == "exact_supplied_example_text_v1"
    assert metadata["solver_request_template_version"] == (
        "decision_procedure_then_mandatory_output_contract_v2"
    )
    assert "prompt_memory_search_enabled" not in metadata


def test_v5_behavior_versions_are_consistent(tmp_path):
    system = PromptEnsembleOptimizationSystem(Config.from_flat(out_dir=str(tmp_path)))
    system.set_run_identity(identity())
    metadata = system.run_meta()
    assert metadata["method_version"] == METHOD_VERSION
    assert metadata["target_selection_version"] == TARGET_SELECTION_VERSION
    assert metadata["checkpoint_version"] == CHECKPOINT_VERSION
    assert "five_axis_overdue_member_pareto_v2" not in json.dumps(metadata)


def test_initialization_modes_are_explicit_and_five_prompt_bounded(tmp_path):
    shared = PromptEnsembleOptimizationSystem(Config.from_flat(out_dir=str(tmp_path / "shared")))
    assert len({agent.initial_prompt for agent in shared.agents}) == 1
    supplied = PromptEnsembleOptimizationSystem(Config.from_flat(
        out_dir=str(tmp_path / "supplied"),
        initialization_mode="provided_prompt_set",
        provided_prompts_json='["p0", "p1", "p2", "p3", "p4"]',
    ))
    assert [agent.initial_prompt for agent in supplied.agents] == ["p0", "p1", "p2", "p3", "p4"]
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
