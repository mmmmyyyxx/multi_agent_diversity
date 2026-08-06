import asyncio

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import (
    CandidateFunnel,
    PromptEnsembleOptimizationSystem,
)


def test_infrastructure_and_parser_failures_do_not_count_as_complete_failures():
    assert not PromptEnsembleOptimizationSystem.is_complete_repairability_failure(
        CandidateFunnel(
            infrastructure_failed_updates=1,
            stage_a_evaluated=2,
            terminal_failure_class="transport_failure",
        )
    )
    assert not PromptEnsembleOptimizationSystem.is_complete_repairability_failure(
        CandidateFunnel(
            student_calls=8,
            terminal_failure_class="student_invalid_exhausted",
        )
    )
    assert not PromptEnsembleOptimizationSystem.is_complete_repairability_failure(
        CandidateFunnel(
            student_calls=8,
            output_contract_contamination_count=16,
            terminal_failure_class="proposal_protocol_failure",
        )
    )
    assert PromptEnsembleOptimizationSystem.is_complete_repairability_failure(
        CandidateFunnel(stage_a_evaluated=2)
    )
    assert PromptEnsembleOptimizationSystem.is_complete_repairability_failure(
        CandidateFunnel(
            terminal_failure_class="critic_semantic_rejection_exhausted"
        )
    )


def test_all_actionable_members_frozen_sets_method_early_stop(tmp_path):
    system = PromptEnsembleOptimizationSystem(
        Config.from_flat(
            out_dir=str(tmp_path),
            experiment_setting="shared_member_aware_responsibility",
            allow_legacy_setting=True,
        )
    )
    system.ensure_responsibility_current = lambda: ({}, {})
    system.target_priority_audit.append({
        "no_actionable_reason": "no_actionable_repairability",
        "update_lane": "no_actionable_repairability",
    })
    system.select_target = lambda _assigned, _index: (None, [])

    assert asyncio.run(system.update_once(0)) is False
    assert system.early_stop_reason == "all_actionable_members_frozen"
    assert system.candidate_decisions[-1]["stop_reason"] == (
        "no_actionable_repairability"
    )
    system.completed_update_count = 1
    system.mark_training_complete(32)
    assert system.training_completed
