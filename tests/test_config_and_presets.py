from dataclasses import asdict

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from multi_dataset_diverse_rl.protocol import CandidateBudgetContract, experiment_protocol
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from scripts.experiment_config import DEFAULT_EXPERIMENT_SETTING_NAMES, select_settings


def identity(setting="shared_peer_state_full"):
    return RunIdentity(
        method_version="peer_state_counterfactual_v2",
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
    assert cfg.training.method_version == "peer_state_counterfactual_v2"
    assert cfg.training.initialization_mode == "shared_identical"
    assert cfg.peer_state.vote_tie_break == "abstain"
    assert cfg.models.optimizer_api_key_env == ""
    assert cfg.tcs.critic_json_max_retries == 2
    with pytest.raises(AttributeError):
        _ = cfg.method_version


def test_only_five_settings_exist_and_old_setting_fails():
    assert DEFAULT_EXPERIMENT_SETTING_NAMES == [
        "shared_baseline",
        "shared_independent_accuracy_tcs",
        "shared_peer_state_credit_round_robin",
        "shared_peer_state_responsibility",
        "shared_peer_state_full",
    ]
    with pytest.raises(ValueError, match="Unknown experiment setting"):
        select_settings("shared_v9_sequential_accuracy")


def test_ablation_protocols_are_field_isolated_and_budget_matched():
    rows = protocols()
    b2 = asdict(rows["shared_peer_state_credit_round_robin"])
    b3 = asdict(rows["shared_peer_state_responsibility"])
    b4 = asdict(rows["shared_peer_state_full"])
    b2_b3_differences = {key for key in b2 if b2[key] != b3[key]}
    assert b2_b3_differences == {
        "name",
        "target_selection_policy",
        "sample_pool_policy",
        "responsibility_refresh_policy",
    }
    b3_b4_differences = {key for key in b3 if b3[key] != b4[key]}
    assert b3_b4_differences == {"name", "tcs_context_policy"}
    assert len({repr(row.candidate_budget_contract) for row in rows.values()}) == 1
    assert len({row.tie_policy for row in rows.values()}) == 1
    assert len({row.initialization_mode for row in rows.values()}) == 1


def test_run_metadata_records_initialization_protocol_and_no_legacy_search(tmp_path):
    system = PromptEnsembleOptimizationSystem(Config.from_flat(out_dir=str(tmp_path)))
    system.set_run_identity(identity())
    metadata = system.run_meta()
    assert metadata["method_version"] == "peer_state_counterfactual_v2"
    assert metadata["initialization_mode"] == "shared_identical"
    assert metadata["initial_prompts_identical"] is True
    assert metadata["tie_policy"] == "abstain"
    assert metadata["generic_diversity_reward_used"] is False
    assert metadata["legacy_compatibility_enabled"] is False
    assert metadata["tcs_protocol_version"] == "hard_blocker_gate_v2"
    assert metadata["critic_approval_basis"] == "all_hard_checks_passed"
    assert metadata["critic_score_controls_approval"] is False
    assert metadata["critic_case_fact_restatement_required"] is True
    assert metadata["task_general_scope"] == "unseen_examples_within_current_task"
    assert metadata["student_sample_memorization_filter"] == "exact_supplied_example_text_v1"
    assert "prompt_memory_search_enabled" not in metadata


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
