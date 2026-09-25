"""Layer-2 allocation and evidence must not inherit historical service routing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.peer_state import build_peer_vote_context, build_team_vote_state
from multi_dataset_diverse_rl.responsibility import (
    MemberAwareRepairOpportunity, compute_member_aware_repair_opportunity,
    ResponsibilityState,
)
from multi_dataset_diverse_rl.team_search.primary_responsibility_scheduler import (
    PrimaryResponsibilityPersistentRealizabilityScheduler,
)
from multi_dataset_diverse_rl.team_search.schemas import TeamSearchRequest
from multi_dataset_diverse_rl.team_search.system_runtime import (
    SystemResponsibilityAssignmentFactory,
    freeze_current_responsibility,
)
from multi_dataset_diverse_rl.team_search.task_builder import Layer2EvidenceRequestBuilder


def _opportunity(question_hash: str, member: int, *, direct: bool) -> MemberAwareRepairOpportunity:
    return MemberAwareRepairOpportunity(
        agent_id=member,
        question_hash=question_hash,
        vote_flip_gain=int(direct),
        margin_gain=int(direct),
        member_error=direct,
        coverage_opportunity=False,
        conversion_opportunity=direct,
        dominant_wrong_member=False,
        unique_correct=False,
        pivotal_correct=False,
        oracle_soft_utility_gain=0.0,
    )


class _System:
    def __init__(self, routed_member: int) -> None:
        self.routed_member = routed_member
        self.routing_calls = 0
        self.responsibility_state = ResponsibilityState(
            updates_since_selected_by_agent={member: 0 for member in range(5)},
        )
        self.protocol = SimpleNamespace(service_routing_enabled=True)
        self.agents = [SimpleNamespace(current_prompt=f"parent-{member}") for member in range(5)]
        self.fixed_probe = SimpleNamespace(examples=tuple(
            SimpleNamespace(question_hash=f"q{index}", question=f"case-{index}", gold_answer="A")
            for index in range(12)
        ))
        self.active_profiles = [tuple(
            SimpleNamespace(valid=True, answer="B") for _ in range(12)
        ) for _ in range(5)]

    def current_states_and_opportunities(self):
        states = tuple(
            SimpleNamespace(
                question_hash=f"q{index}", vote_correct=index >= 4,
                team_correctness=(False,) * 5,
                gold_vote_count=1, plurality_margin=-1,
            )
            for index in range(12)
        )
        opportunities = {
            f"q{index}": tuple(
                _opportunity(f"q{index}", member, direct=index < 4 and member in (0, 1))
                for member in range(5)
            )
            for index in range(12)
        }
        return states, {}, opportunities

    def assign_responsibilities(self, *, update_index: int):
        self.routing_calls += 1
        _, _, opportunities = self.current_states_and_opportunities()
        routed = {
            member: [opportunities[f"q{index}"][member] for index in range(4)]
            if member == self.routed_member else []
            for member in range(5)
        }
        return {}, routed


class _AllEvidenceMinibatch:
    def select_team_minibatch(self, evidence, *, primary_responsibility_lane):
        return evidence


def _layer2_result(routed_member: int):
    system = _System(routed_member)
    snapshot = freeze_current_responsibility(system, update_index=0)
    scheduler = PrimaryResponsibilityPersistentRealizabilityScheduler()
    decision = scheduler.select(
        assigned=snapshot.assigned,
        current_margin_by_question=snapshot.current_margin_by_question,
        seed=80,
        update_index=0,
        target_count=1,
    )
    assignment = SystemResponsibilityAssignmentFactory(
        system=system,
        snapshot_reader=lambda: snapshot,
        task_builder=_AllEvidenceMinibatch(),
    ).build_from_member(
        request=TeamSearchRequest(80, 0, "parent", 36, "solver", "output"),
        member_id=1,
        primary_lane="direct_flip",
        responsibility_identity="raw-legal",
    )
    assert system.routing_calls == 0
    assert system.responsibility_state.eligible_agents_by_question == {}
    return snapshot, decision, assignment


def test_historical_service_routing_poison_cannot_change_layer2_scores_targets_or_evidence():
    left, left_decision, left_assignment = _layer2_result(routed_member=0)
    right, right_decision, right_assignment = _layer2_result(routed_member=1)

    # Each tied legal residual belongs to both member-local opportunity sets;
    # a historical unique-route policy would have removed one of these sets.
    expected = {f"q{index}" for index in range(4)}
    for snapshot in (left, right):
        assert {row.question_hash for row in snapshot.assigned[0]} == expected
        assert {row.question_hash for row in snapshot.assigned[1]} == expected
    assert left.assigned == right.assigned
    assert left_decision.summaries == right_decision.summaries
    assert left_decision.selected_member_ids == right_decision.selected_member_ids
    assert [
        (row.direct_count, row.near_margin_count, row.coverage_count)
        for row in left_decision.summaries[:2]
    ] == [(4, 0, 0), (4, 0, 0)]
    assert {
        row.example_id for row in left_assignment.evidence
        if row.evidence_group == "responsibility"
    } == expected
    assert [
        row.example_id for row in left_assignment.evidence
        if row.evidence_group == "responsibility"
    ] == [
        row.example_id for row in right_assignment.evidence
        if row.evidence_group == "responsibility"
    ]


def test_shared_identical_parent_exposes_unmodified_coalition_quota_blocker():
    system = _System(routed_member=0)
    states = tuple(build_team_vote_state(
        question_hash=f"q{index}", gold_answer="A",
        answers=["B" if index < 4 else "A"] * 5,
        valid_vector=[True] * 5,
    ) for index in range(12))
    opportunities = {
        state.question_hash: tuple(compute_member_aware_repair_opportunity(
            team_state=state,
            peer_context=build_peer_vote_context(state, member),
        ) for member in range(5))
        for state in states
    }
    system.current_states_and_opportunities = lambda: (states, {}, opportunities)
    system.active_profiles = [tuple(
        SimpleNamespace(valid=True, answer="B" if index < 4 else "A")
        for index in range(12)
    ) for _ in range(5)]
    snapshot = freeze_current_responsibility(system, update_index=0)
    assert all(len(snapshot.assigned[member]) == 4 for member in range(5))
    factory = SystemResponsibilityAssignmentFactory(
        system=system,
        snapshot_reader=lambda: snapshot,
        task_builder=Layer2EvidenceRequestBuilder(),
    )
    with pytest.raises(ValueError, match="4 unique coalition examples"):
        factory.build_from_member(
            request=TeamSearchRequest(80, 0, "parent", 36, "solver", "output"),
            member_id=1,
            primary_lane="coverage",
            responsibility_identity="raw-legal",
        )
