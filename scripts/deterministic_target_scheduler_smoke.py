from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.responsibility import (  # noqa: E402
    AgentTargetPriority,
    select_target_agent,
)


def row(agent_id: int, *, headroom: int, gain: int, cooling: bool, overdue: bool):
    return AgentTargetPriority(
        agent_id=agent_id,
        individual_error_count=1,
        assigned_load=0,
        direct_vote_fix_count=0,
        oracle_soft_utility_gain_sum=0.0,
        coverage_opportunity_count=0,
        dominant_wrong_count=0,
        gain_count=gain,
        current_correct_count=10 - headroom,
        best_current_correct_count=10,
        headroom_to_best=headroom,
        unimproved=gain <= 0,
        improvement_need=0,
        unique_correct_count=0,
        pivotal_correct_count=0,
        updates_since_selected=0,
        overdue=overdue,
        pareto_front=1,
        seeded_rank=str(agent_id),
        best_observed_target_gain=0,
        no_positive_candidate_streak=1 if cooling else 0,
        next_regular_eligible_update=2 if cooling else 0,
        cooling_down=cooling,
        target_attempt_count=1,
    )


def main() -> int:
    assert select_target_agent((
        row(0, headroom=20, gain=0, cooling=False, overdue=False),
        row(1, headroom=30, gain=5, cooling=False, overdue=False),
    )) == 0
    assert select_target_agent((
        row(0, headroom=20, gain=0, cooling=True, overdue=False),
        row(1, headroom=5, gain=0, cooling=False, overdue=False),
    )) == 1
    assert select_target_agent((
        row(0, headroom=1, gain=0, cooling=True, overdue=True),
        row(1, headroom=30, gain=0, cooling=False, overdue=False),
    )) == 0
    assert select_target_agent((
        row(0, headroom=1, gain=0, cooling=True, overdue=False),
        row(1, headroom=2, gain=0, cooling=True, overdue=False),
    )) in {0, 1}
    print("deterministic target scheduler smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
