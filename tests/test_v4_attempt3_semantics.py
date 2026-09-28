"""Offline equivalence of ordinary V4 decisions with attempt3 observation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.governance import v4_attempt3_contract as contract
from multi_dataset_diverse_rl.local_optimizers.schemas import (
    LocalOptimizationResult, LocalPromptCandidate, OpaqueOptimizerState,
)
from multi_dataset_diverse_rl.production_transfer_diagnostic import _diagnostic_stop_reason
from multi_dataset_diverse_rl.team_search.candidate_evaluator import EvaluationCost
from multi_dataset_diverse_rl.team_search.candidate_selector import CommonSafeTeamCandidateSelector
from multi_dataset_diverse_rl.team_search.controller import TeamSearchController
from multi_dataset_diverse_rl.team_search.schemas import TeamMiniBatchMetrics, TeamSearchRequest
from multi_dataset_diverse_rl.team_search.task_builder import LocalTaskBuilder
from test_online_transfer_diagnostic import _Assignment
from scripts.prepare_online_transfer_diagnostic_v4 import (
    ATTEMPT2_ID, frozen_payload as attempt2_payload,
)
from scripts.prepare_online_transfer_diagnostic_v4_attempt3 import (
    ATTEMPT3_ID, frozen_payload as attempt3_payload,
)


def _evaluation(vote: int, member: int, candidate_id: str):
    return SimpleNamespace(
        team_outcome=SimpleNamespace(vote_correct_count=vote, mean_soft_vote_utility=0.5),
        competence=SimpleNamespace(correct_count=member, invalid_count=0),
        prompt_hash=candidate_id,
        marginal=SimpleNamespace(
            vote_gain_count=max(0, vote - 50), vote_loss_count=max(0, 50 - vote),
            coverage_gain_count=1, coverage_loss_count=0,
        ),
        protection=SimpleNamespace(
            pivotal_correct_gain_count=0, pivotal_correct_loss_count=0,
        ),
    )


class _Local:
    def __init__(self, returned: int, accepted: int):
        self.returned = returned
        self.accepted = accepted

    async def optimize(self, task):
        candidates = tuple(
            LocalPromptCandidate(
                f"candidate-{index}", f"changed reasoning {index}", 0.7 + index / 100,
                {}, (), 1,
                backend_metadata={"local_acceptance_delta": 1.0,
                                  "local_parent_score": 0.5},
            )
            for index in range(self.returned)
        )
        telemetry = {
            "accepted_mutations": self.accepted,
            "proposal_attempts": self.accepted,
            "accepted_event_indices": list(range(1, self.accepted + 1)),
            "accepted_event_iterations": list(range(1, self.accepted + 1)),
            "frontier_candidate_indices": [0, *range(1, self.accepted + 1)],
            "changed_frontier_indices": list(range(1, self.accepted + 1)),
            "valid_unique_frontier_indices": list(range(1, self.returned + 1)),
            "returned_candidate_indices": list(range(1, self.returned + 1)),
            "returned_candidate_count": self.returned,
        }
        return LocalOptimizationResult(
            candidates, "gepa", "frozen", OpaqueOptimizerState(
                "gepa", "frozen", {"telemetry": telemetry},
            ), 20, self.accepted, 20, 2, 22, "complete",
        )


class _Evaluator:
    def __init__(self, signals: tuple[int, ...], shadow_pass: bool):
        self.signals = signals
        self.shadow_pass = shadow_pass
        self.minibatch = []
        self.full = []
        self.diagnostic_full = []
        self.shadow = []

    def active_evaluation(self, assignment):
        return _evaluation(50, 60, "parent")

    def parent_vote_responsiveness(self, assignment):
        return {"single_member_vote_changeable_cases": 1}

    def evaluate_minibatch(self, assignment, candidate, minibatch):
        index = int(candidate.candidate_id.split("-")[1])
        self.minibatch.append(candidate.candidate_id)
        return TeamMiniBatchMetrics(target_delta=self.signals[index]), EvaluationCost(12, 12, 0)

    def evaluate_full(self, assignment, candidate):
        self.full.append(candidate.candidate_id)
        index = int(candidate.candidate_id.split("-")[1])
        return _evaluation(51 + index, 61 + index, candidate.candidate_id), EvaluationCost(100, 100, 0)

    def evaluate_diagnostic_full(self, assignment, candidate):
        self.diagnostic_full.append(candidate.candidate_id)
        index = int(candidate.candidate_id.split("-")[1])
        return _evaluation(51 + index, 61 + index, candidate.candidate_id), EvaluationCost(100, 100, 0)

    def evaluate_shadow(self, assignment, candidate):
        self.shadow.append(candidate.candidate_id)
        return SimpleNamespace(passed=self.shadow_pass), EvaluationCost(50, 50, 0)


class _Committer:
    def __init__(self):
        self.committed = []

    def commit(self, **kwargs):
        self.committed.append(kwargs["candidate"].candidate_id)


def _run(returned, accepted, signals, shadow_pass, diagnostic):
    evaluator = _Evaluator(signals, shadow_pass)
    committer = _Committer()
    controller = TeamSearchController(
        responsibility=_Assignment(), task_builder=LocalTaskBuilder(),
        local_optimizer=_Local(returned, accepted), evaluator=evaluator,
        selector=CommonSafeTeamCandidateSelector(), committer=committer,
        diagnostic_full_for_local_accepts=diagnostic,
        diagnostic_allow_multi_accepted=diagnostic,
    )
    outcome = asyncio.run(controller.run_opportunity(TeamSearchRequest(
        81, 0, "parent-team", 36, "COMMON_SOLVER_CONTRACT_V1", "task_output_contract_v1",
    )))
    ordinary = {
        "local_ids": [row.local_candidate.candidate_id for row in outcome.candidates],
        "minibatch": [(row.local_candidate.candidate_id, row.minibatch_metrics.target_delta)
                      for row in outcome.candidates],
        "promotion": [row.local_candidate.candidate_id for row in outcome.candidates if row.promoted],
        "full": evaluator.full,
        "common_safe": [row.local_candidate.candidate_id for row in outcome.candidates
                        if row.constraint is not None and row.constraint.passed],
        "winner_shadow": evaluator.shadow,
        "committed": committer.committed,
        "committed_candidate_id": outcome.committed_candidate_id,
        "termination_reason": outcome.termination_reason,
    }
    return outcome, evaluator, ordinary


@pytest.mark.parametrize("returned,accepted,signals,shadow_pass", [
    (0, 0, (), True),
    (1, 1, (0,), True),
    (1, 3, (1,), True),  # Internal GEPA accepts can compress at the boundary.
    (2, 2, (1, 0), True),
    (2, 2, (1, 2), False),
    (2, 2, (1, 2), True),
    (3, 3, (1, 2, 0), True),
])
def test_attempt3_observation_preserves_ordinary_v4(
    monkeypatch, returned, accepted, signals, shadow_pass,
):
    import multi_dataset_diverse_rl.team_search.controller as controller_module
    import multi_dataset_diverse_rl.team_search.candidate_selector as selector_module

    safe = lambda candidate, active: SimpleNamespace(passed=True)
    monkeypatch.setattr(controller_module, "evaluate_constraints", safe)
    monkeypatch.setattr(selector_module, "evaluate_constraints", safe)
    baseline, _, ordinary = _run(returned, accepted, signals, shadow_pass, False)
    observed, evaluator, observed_ordinary = _run(returned, accepted, signals, shadow_pass, True)
    assert observed_ordinary == ordinary
    assert observed.audit_metadata["local_optimizer_telemetry"]["accepted_mutations"] == accepted
    assert observed.funnel["diagnostic_full_evaluated_candidates"] == returned
    assert evaluator.minibatch == ordinary["local_ids"]
    assert set(evaluator.diagnostic_full) == set(ordinary["local_ids"]) - set(ordinary["promotion"])
    assert all(row.full_evaluation is None and row.constraint is None
               for row in observed.candidates if not row.promoted)
    assert baseline.funnel["committed_candidates"] == observed.funnel["committed_candidates"]


def test_attempt3_stop_and_resource_bounds():
    assert _diagnostic_stop_reason(4, 4, attempt3=True) is None
    assert _diagnostic_stop_reason(6, 6, attempt3=True) == "RETURNED_CANDIDATE_TARGET_REACHED"
    assert _diagnostic_stop_reason(4, 9, attempt3=True) == "REFLECTION_PROPOSAL_PREOPPORTUNITY_GUARD"
    assert _diagnostic_stop_reason(0, 20, attempt3=True) == "REFLECTION_PROPOSAL_CEILING_REACHED"
    assert contract.MAX_LOGICAL_PROVIDER_CALLS == 6260
    assert contract.MAX_LOGICAL_PROVIDER_CALLS * contract.MAX_ATTEMPTS_PER_LOGICAL_CALL == 143980
    assert contract.SUCCESSFUL_PROVIDER_CEILING > contract.MAX_LOGICAL_PROVIDER_CALLS
    assert contract.TRANSPORT_ATTEMPT_CEILING > 143980


def test_attempt3_freeze_preserves_v4_science_and_denies_api(monkeypatch):
    from multi_dataset_diverse_rl.provider_credentials import LWJ_DASHSCOPE_BASE_URL_ENV

    monkeypatch.setenv(LWJ_DASHSCOPE_BASE_URL_ENV, "https://example.invalid/compatible-mode/v1")
    prior, prior_protocol = attempt2_payload(execution_source_sha="a" * 40,
                                             attempt_id=ATTEMPT2_ID)
    current, protocol = attempt3_payload(execution_source_sha="b" * 40)
    for key in ("scientific", "method_identity", "spec_identity", "semantic_contract",
                "runtime", "models", "dependency", "access", "api_authorization"):
        assert prior[key] == current[key]
    for key in ("semantic_contract", "target_policy", "packet_capacity", "seed",
                "local_metric_call_budget_per_opportunity", "team_minibatch_rows",
                "ordinary_shadow_policy"):
        assert prior_protocol[key] == protocol[key]
    assert current["attempt_id"] == ATTEMPT3_ID
    assert current["execution"]["scientific_method_anchor_sha"] == (
        "85812a7d891e6a2c3bfdca00a1cb4d14074735a4"
    )
    assert current["api_authorization"]["authorized"] is False
    assert current["access"] == {"validation50_calls": 0, "test50_calls": 0}
    assert current["diagnostic_contract"] == contract.diagnostic_contract()
    assert protocol["resource_upper_bounds"] == contract.resource_upper_bounds()
    assert "multi_dataset_diverse_rl/team_search/v4_opportunity.py" in current[
        "execution"
    ]["source_paths"]
