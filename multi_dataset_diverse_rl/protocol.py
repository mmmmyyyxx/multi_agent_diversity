from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InitializationMode(str, Enum):
    SHARED_IDENTICAL = "shared_identical"
    PROVIDED_PROMPT_SET = "provided_prompt_set"


@dataclass(frozen=True)
class CandidateBudgetContract:
    generated_per_update: int
    stage_a_channel_top_k: int
    stage_b_candidate_budget: int
    representative_size: int
    coverage_size: int
    conversion_size: int
    preservation_size: int


@dataclass(frozen=True)
class ExperimentProtocol:
    name: str
    optimization_enabled: bool
    target_selection_policy: str
    sample_pool_policy: str
    tcs_context_policy: str
    candidate_selection_policy: str
    responsibility_refresh_policy: str
    initialization_mode: InitializationMode
    tie_policy: str
    candidate_budget_contract: CandidateBudgetContract


def experiment_protocol(
    name: str,
    *,
    initialization_mode: str,
    tie_policy: str,
    candidate_budget_contract: CandidateBudgetContract,
) -> ExperimentProtocol:
    common = {
        "initialization_mode": InitializationMode(initialization_mode),
        "tie_policy": str(tie_policy),
        "candidate_budget_contract": candidate_budget_contract,
    }
    definitions = {
        "shared_baseline": dict(
            optimization_enabled=False,
            target_selection_policy="none",
            sample_pool_policy="none",
            tcs_context_policy="none",
            candidate_selection_policy="none",
            responsibility_refresh_policy="off",
        ),
        "shared_independent_accuracy": dict(
            optimization_enabled=True,
            target_selection_policy="round_robin",
            sample_pool_policy="individual_errors",
            tcs_context_policy="generic_accuracy",
            candidate_selection_policy="individual_accuracy",
            responsibility_refresh_policy="off",
        ),
        "shared_peer_state_vote_first": dict(
            optimization_enabled=True,
            target_selection_policy="round_robin",
            sample_pool_policy="global_peer_state",
            tcs_context_policy="generic_peer_state",
            candidate_selection_policy="competence_constrained_vote_first",
            responsibility_refresh_policy="off",
        ),
        "shared_peer_state_member_pareto": dict(
            optimization_enabled=True,
            target_selection_policy="round_robin",
            sample_pool_policy="global_peer_state",
            tcs_context_policy="generic_peer_state",
            candidate_selection_policy="member_aware_pareto",
            responsibility_refresh_policy="off",
        ),
        "shared_member_aware_responsibility": dict(
            optimization_enabled=True,
            target_selection_policy="member_aware_responsibility",
            sample_pool_policy="member_aware_residuals",
            tcs_context_policy="generic_peer_state",
            candidate_selection_policy="member_aware_pareto",
            responsibility_refresh_policy="online",
        ),
        "shared_member_aware_full": dict(
            optimization_enabled=True,
            target_selection_policy="member_aware_responsibility",
            sample_pool_policy="member_aware_residuals",
            tcs_context_policy="member_aware_responsibility_conditioned",
            candidate_selection_policy="member_aware_pareto",
            responsibility_refresh_policy="online",
        ),
    }
    if name not in definitions:
        raise ValueError(f"Unknown experiment protocol: {name}")
    return ExperimentProtocol(name=name, **definitions[name], **common)
