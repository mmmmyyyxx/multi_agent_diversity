from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.responsibility import (
    MemberAwareRepairOpportunity,
    ResponsibilityState,
    build_target_selection_decision,
    initialize_repairability_state,
    record_target_update_acceptance,
    record_target_update_failure,
    refresh_frozen_member_states,
    responsibility_portfolios,
    target_priorities,
)


def opportunity(question_hash: str, *, direct: int, margin: int):
    return MemberAwareRepairOpportunity(
        agent_id=0,
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
    assert decision(state, changed).selected_agent_id == 0

    print("deterministic target scheduler smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
