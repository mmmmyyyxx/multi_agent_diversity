from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.responsibility import (
    MemberAwareRepairOpportunity,
    RepairLane,
    ResponsibilityState,
    initialize_repairability_state,
    repairability_adjusted_target_scores,
    select_repairability_targets,
)


def _row(
    agent_id: int,
    question_hash: str,
    *,
    direct: bool,
    margin: int,
) -> MemberAwareRepairOpportunity:
    return MemberAwareRepairOpportunity(
        agent_id=agent_id,
        question_hash=question_hash,
        vote_flip_gain=int(direct),
        margin_gain=margin,
        member_error=True,
        coverage_opportunity=False,
        conversion_opportunity=True,
        dominant_wrong_member=False,
        unique_correct=False,
        pivotal_correct=False,
        oracle_soft_utility_gain=0.0,
    )


def main() -> None:
    state = ResponsibilityState()
    initialize_repairability_state(state, range(5))
    state.branch_failure_count_by_agent.update({0: 2, 1: 0, 2: 0})
    state.updates_since_selected_by_agent.update({0: 0, 1: 1, 2: 9})
    active = {
        0: (
            _row(0, "a-direct-1", direct=True, margin=4),
            _row(0, "a-direct-2", direct=True, margin=4),
            _row(0, "a-support", direct=False, margin=2),
        ),
        1: (_row(1, "b-direct", direct=True, margin=5),),
        2: (_row(2, "c-support", direct=False, margin=1),),
    }
    lanes = {
        0: RepairLane.DIRECT_FLIP,
        1: RepairLane.DIRECT_FLIP,
        2: RepairLane.MARGIN_SUPPORT,
    }
    scores = repairability_adjusted_target_scores(
        active_assignments=active,
        state=state,
        seed=46,
        current_member_correct_counts=(50, 49, 47, 50, 50),
        initial_member_correct_counts=(45, 45, 45, 45, 45),
        member_uplift_tolerance=5,
        legal_assignments=active,
        service_portfolios=active,
        active_lane_by_agent=lanes,
    )
    by_agent = {row.agent_id: row for row in scores}
    assert by_agent[0].direct_fix_count == 2
    assert by_agent[0].support_margin_sum == 2
    assert by_agent[0].repairability_discount == 1 / 3
    assert by_agent[2].normalized_wait == 1.0
    assert tuple(row.agent_id for row in scores) == tuple(
        row.agent_id for row in sorted(scores, key=lambda row: row.ranking_key())
    )
    selected = select_repairability_targets(scores, target_branch_count=2)
    assert len(selected) == 2
    assert len({row.agent_id for row in selected}) == 2
    assert not any(state.frozen_by_agent.values())
    print(json.dumps({
        "ok": True,
        "ranking": [row.agent_id for row in scores],
        "selected": [row.agent_id for row in selected],
        "expected_update_values": {
            str(row.agent_id): row.expected_update_value for row in scores
        },
        "pareto_used": False,
        "freeze_used": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
