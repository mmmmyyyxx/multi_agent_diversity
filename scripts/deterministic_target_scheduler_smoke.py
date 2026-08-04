from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.responsibility import (
    MemberAwareRepairOpportunity,
    RepairLane,
    ResponsibilityState,
    build_service_routing,
    build_target_selection_decision,
    initialize_repairability_state,
    record_target_update_acceptance,
    record_target_update_failure,
    record_specialization_anchor_acceptance,
    refresh_frozen_member_states,
    responsibility_portfolios,
    target_priorities,
)
from multi_dataset_diverse_rl.peer_state import build_team_vote_state


def opportunity(
    question_hash: str, *, direct: int, margin: int, agent_id: int = 0
):
    return MemberAwareRepairOpportunity(
        agent_id=agent_id,
        question_hash=question_hash,
        vote_flip_gain=direct,
        margin_gain=margin,
        member_error=True,
        coverage_opportunity=False,
        conversion_opportunity=True,
        dominant_wrong_member=False,
        unique_correct=False,
        pivotal_correct=False,
        oracle_soft_utility_gain=0.0,
    )


def team(question_hash: str, answers: list[str]):
    return build_team_vote_state(
        question_hash=question_hash,
        gold_answer="A",
        answers=answers,
        normalize_answer=str.upper,
        match_answer=lambda left, right: left == right,
        tie_break="abstain",
    )


def decision(state: ResponsibilityState, assignments):
    priorities = target_priorities(
        assignments=assignments,
        state=state,
        seed=42,
        current_member_correct_counts=(0, 10, 10, 10, 10),
        initial_member_correct_counts=(0, 10, 10, 10, 10),
        member_uplift_tolerance=5,
    )
    return build_target_selection_decision(priorities)


def main() -> int:
    state = ResponsibilityState(
        updates_since_selected_by_agent={agent: 0 for agent in range(5)},
        accepted_updates_by_agent={agent: 0 for agent in range(5)},
        target_attempt_count_by_agent={agent: 0 for agent in range(5)},
    )
    initialize_repairability_state(state, range(5))
    coverage_state = team("coverage", ["B", "B", "C", "C", "D"])
    direct_state = team("direct", ["A", "A", "B", "B", "B"])
    route_rows = {
        "coverage": (
            opportunity("coverage", direct=0, margin=2, agent_id=0),
            opportunity("coverage", direct=0, margin=2, agent_id=2),
        ),
        "direct": (
            opportunity("direct", direct=1, margin=2, agent_id=1),
            opportunity("direct", direct=1, margin=2, agent_id=2),
        ),
    }
    initial_route = build_service_routing(
        team_states={"coverage": coverage_state, "direct": direct_state},
        opportunities=route_rows,
        eligible_agents_by_question={"coverage": (0, 2), "direct": (1, 2)},
        state=state,
        seed=42,
    )
    routed_hashes = [
        row.question_hash
        for rows in initial_route.service_portfolios.values()
        for row in rows
    ]
    assert sorted(routed_hashes) == ["coverage", "direct"]
    assert len(routed_hashes) == len(set(routed_hashes))

    record_specialization_anchor_acceptance(
        state=state,
        accepted_agent_id=0,
        active_lane=RepairLane.COVERAGE,
        update_index=0,
    )
    anchored_route = build_service_routing(
        team_states={"coverage": coverage_state, "direct": direct_state},
        opportunities=route_rows,
        eligible_agents_by_question={"coverage": (0, 2), "direct": (1, 2)},
        state=state,
        seed=42,
    )
    assert anchored_route.assignments_by_question["coverage"].service_agent_id == 0
    assert anchored_route.active_lane_by_agent[0] is RepairLane.COVERAGE
    original = {0: [opportunity("q0", direct=1, margin=2)]}
    portfolio = responsibility_portfolios(
        assignments=original, state=state
    )[0]

    assert decision(state, original).selected_agent_id == 0
    assert record_target_update_failure(
        state=state, portfolio=portfolio, update_index=0
    ) is None
    assert record_target_update_failure(
        state=state, portfolio=portfolio, update_index=1
    ) is not None
    state.updates_since_selected_by_agent[0] = 10_000
    assert decision(state, original).no_actionable_reason == (
        "no_actionable_repairability"
    )

    record_target_update_acceptance(state=state, accepted_agent_id=1)
    assert not refresh_frozen_member_states(
        state=state, assignments=original, update_index=2
    )
    record_target_update_acceptance(state=state, accepted_agent_id=2)
    assert not refresh_frozen_member_states(
        state=state, assignments=original, update_index=3
    )
    record_target_update_acceptance(state=state, accepted_agent_id=3)
    changed = {0: [opportunity("q1", direct=0, margin=3)]}
    assert refresh_frozen_member_states(
        state=state, assignments=changed, update_index=4
    )
    assert state.specialization_anchor_by_agent[0] is None
    assert decision(state, changed).selected_agent_id == 0

    print("deterministic target scheduler smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
