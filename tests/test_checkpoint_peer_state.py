import asyncio
import json

import pytest

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.persistence.checkpoint import build_checkpoint, restore_checkpoint
from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from multi_dataset_diverse_rl.proposal_memory import (
    ProposalMemoryEntry,
    ProposalMemoryKey,
    SanitizedCandidateSummary,
)
from multi_dataset_diverse_rl.responsibility import RepairLane
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.versions import CHECKPOINT_VERSION


async def solver(_question, agent_id, _prompt):
    answer = "A" if agent_id == 0 else "B"
    return PromptAnswer(answer, f"FINAL_ANSWER: {answer}", True)


def identity():
    return RunIdentity(
        method_version="member_aware_peer_state_v11", experiment_setting="shared_member_aware_full",
        git_commit="commit", git_dirty=False, config_fingerprint="config", manifest_sha256="manifest",
        train_file_sha256="train", val_file_sha256="val", test_file_sha256="test",
        train_question_set_hash="train-q", val_question_set_hash="val-q", test_question_set_hash="test-q",
    )


def build_system(tmp_path, run_identity=None):
    system = PromptEnsembleOptimizationSystem(
        Config.from_flat(
            out_dir=str(tmp_path),
            answer_format="option_letter",
            initialization_mode="provided_prompt_set",
            provided_prompts_json='["p0", "p1", "p2", "p3", "p4"]',
        ),
        solver=solver,
    )
    system.set_run_identity(run_identity or identity())
    asyncio.run(system.initialize_fixed_probe([{"question": "q", "answer": "A"}]))
    return system


def test_v19_checkpoint_restores_freeze_state_and_responsibility_portfolios(tmp_path):
    source = build_system(tmp_path / "source")
    source.planned_update_count = 24
    source.completed_update_count = 3
    source.training_dynamics = [{"update_index": -1}, {"update_index": 0}]
    source.team_differentiation_trajectory = [{"update_index": -1}]
    source.update_transition_decomposition = [{"update_index": 0}]
    source.final_state_selection = {"selected_checkpoint_source": "final_active_state"}
    source.responsibility_state.specialization_anchor_by_agent[2] = (
        RepairLane.COVERAGE
    )
    source.ensure_responsibility_current()
    question_hash = source.fixed_probe.examples[0].question_hash
    assert 2 in source.cached_responsibility_eligibility[question_hash]
    source.responsibility_state.updates_since_selected_by_agent[2] = 6
    source.responsibility_state.consecutive_failed_updates_by_agent[2] = 2
    source.responsibility_state.last_failed_portfolio_signature_by_agent[2] = "sig"
    source.responsibility_state.frozen_by_agent[2] = True
    source.responsibility_state.frozen_portfolio_signature_by_agent[2] = "sig"
    source.responsibility_state.frozen_residual_hashes_by_agent[2] = (question_hash,)
    source.responsibility_state.frozen_direct_fix_count_by_agent[2] = 1
    source.responsibility_state.frozen_margin_gain_sum_by_agent[2] = 2
    source.responsibility_state.other_accepted_updates_since_freeze_by_agent[2] = 1
    source.responsibility_state.freeze_count_by_agent[2] = 3
    source.repairability_freeze_events = [{"agent_id": 2, "update_index": 2}]
    source.repairability_unfreeze_events = [{"agent_id": 1, "update_index": 1}]
    source_target, _ = source.select_target(
        source.cached_responsibility_assignments, update_index=3
    )
    memory_key = source._proposal_memory_key(
        target_agent_id=2,
        parent_prompt=source.agents[2].current_prompt,
        assigned_hashes={question_hash},
    )
    source.proposal_memory_entries[memory_key.key_hash()] = ProposalMemoryEntry(
        key=memory_key,
        assigned_question_hashes=(question_hash,),
        attempt_count=3,
        previous_evidence_bundle_hashes=("bundle-1", "bundle-2"),
        previous_repair_plan_hashes=("plan-1",),
        last_failure_stage="regressive_progress",
        last_rejection_reason_histogram={"team_vote_regression": 2},
        candidate_summaries=(SanitizedCandidateSummary(
            prompt_hash="candidate", target_gain=0, vote_gain_count=1,
            vote_loss_count=0, vote_net_gain=1,
            assigned_residual_repair_count=1,
            assigned_residual_utility_delta=0.5,
            coverage_gain_count=1, coverage_loss_count=0,
            unique_correct_gain_count=0, unique_correct_loss_count=0,
            pivotal_correct_gain_count=0, pivotal_correct_loss_count=0,
            rejection_reasons=("member_objective_regression",),
        ),),
        max_target_gain=0,
        max_vote_net_gain=1,
        max_assigned_residual_repair_count=1,
        rotation_cursor=3,
        immediate_tabu_bundle_hash="bundle-2",
        rotation_exhausted=True,
    )
    source.proposal_memory_events = [{"target_agent_id": 2, "memory_hit": True}]
    source.proposal_rotation_trajectory = [{"target_agent_id": 2, "rotation_level": "preservation"}]
    payload = build_checkpoint(source, epoch_index=1, update_index=0, training_state={"planned_update_count": 24})
    assert payload["checkpoint_version"] == CHECKPOINT_VERSION == 20
    assert payload["mutable_prompt_contract_version"] == (
        "reasoning_only_no_response_format_v2"
    )
    assert payload["student_prompt_contract_version"] == (
        "mutable_reasoning_only_v2"
    )
    assert payload["candidate_protocol_filter_version"] == (
        "output_contract_contamination_v2"
    )
    assert "responsibility_first_seen_update" not in json.dumps(payload)
    assert "cached_responsibility_assignments" not in payload
    assert "cached_member_opportunities" not in payload
    assert "validation_state_cache" not in payload
    assert "validation_probe" not in payload
    assert payload["service_routing_version"] == "single_service_anchor_routing_v1"
    assert payload["cached_service_assignments"]
    assert payload["cached_service_portfolios"]
    assert payload["cached_active_lane_by_agent"]
    target = build_system(tmp_path / "source")
    epoch, update, state = restore_checkpoint(target, payload)
    assert (epoch, update, state) == (1, 0, {"planned_update_count": 24})
    assert target.planned_update_count == 24
    assert target.completed_update_count == 3
    assert target.responsibility_state.updates_since_selected_by_agent[2] == 6
    assert target.responsibility_state.frozen_by_agent[2]
    assert target.responsibility_state.frozen_residual_hashes_by_agent[2] == (
        question_hash,
    )
    assert target.responsibility_state.other_accepted_updates_since_freeze_by_agent[2] == 1
    assert target.responsibility_state.freeze_count_by_agent[2] == 3
    assert target.repairability_freeze_events == source.repairability_freeze_events
    assert target.repairability_unfreeze_events == source.repairability_unfreeze_events
    assert (
        target.responsibility_state.specialization_anchor_by_agent
        == source.responsibility_state.specialization_anchor_by_agent
    )
    assert target.cached_service_assignments == source.cached_service_assignments
    assert target.cached_active_lane_by_agent == source.cached_active_lane_by_agent
    restored_target, _ = target.select_target(
        target.cached_responsibility_assignments, update_index=3
    )
    assert restored_target == source_target
    assert target.training_dynamics == source.training_dynamics
    assert target.team_differentiation_trajectory == source.team_differentiation_trajectory
    assert target.update_transition_decomposition == source.update_transition_decomposition
    assert target.proposal_memory_entries == source.proposal_memory_entries
    assert target.proposal_memory_events == source.proposal_memory_events
    assert target.proposal_rotation_trajectory == source.proposal_rotation_trajectory
    assert (
        target.cached_responsibility_eligibility
        == source.cached_responsibility_eligibility
    )
    assert target.cached_responsibility_assignments
    restored = target._proposal_memory_entry(memory_key, {question_hash})
    assert restored is not None and restored.rotation_cursor == 3
    target.cached_responsibility_eligibility = {question_hash: (3,)}
    other_agent_key = target._proposal_memory_key(
        target_agent_id=3,
        parent_prompt=target.agents[3].current_prompt,
        assigned_hashes={question_hash},
    )
    assert target._proposal_memory_entry(
        other_agent_key,
        {question_hash},
    ) is None
    target.team_state_version += 1
    successor_key = target._proposal_memory_key(
        target_agent_id=3,
        parent_prompt=target.agents[3].current_prompt,
        assigned_hashes={question_hash},
    )
    assert target._proposal_memory_entry(
        successor_key,
        {question_hash},
    ) is None


def test_checkpoint_restore_rejects_contaminated_active_prompt_before_mutation(
    tmp_path,
):
    source = build_system(tmp_path / "source")
    payload = build_checkpoint(
        source,
        epoch_index=0,
        update_index=0,
        training_state={},
    )
    payload["prompts"][2] = "Reason carefully.\nFINAL_ANSWER: A"
    target = build_system(tmp_path / "target")
    original_prompts = [agent.current_prompt for agent in target.agents]

    with pytest.raises(ValueError, match="output_contract_contamination"):
        restore_checkpoint(target, payload)

    assert [agent.current_prompt for agent in target.agents] == original_prompts


def test_v19_checkpoint_is_explicitly_incompatible(tmp_path):
    system = build_system(tmp_path)
    payload = build_checkpoint(system, epoch_index=0, update_index=0, training_state={})
    payload["checkpoint_version"] = 19
    with pytest.raises(ValueError, match="incompatible"):
        restore_checkpoint(system, payload)
