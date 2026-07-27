from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.responsibility import (  # noqa: E402
    AgentTargetPriority,
    CandidateSearchOutcome,
    select_target_agent,
)


def row(agent_id: int, *, rank: int, gain: int, cooling: bool, overdue: bool):
    return AgentTargetPriority(
        agent_id=agent_id,
        individual_error_count=1,
        assigned_load=0,
        direct_vote_fix_count=0,
        oracle_soft_utility_gain_sum=0.0,
        coverage_opportunity_count=0,
        dominant_wrong_count=0,
        gain_count=gain,
        current_correct_count=10 - rank,
        best_current_correct_count=10,
        current_correct_gap_to_best=rank,
        best_team_gain_count=max(0, gain + rank),
        gain_gap_to_best=rank,
        relative_gain_tolerance_count=5,
        within_relative_gain_band=rank == 0,
        has_relative_improvement_potential=rank > 0,
        relative_improvement_potential_rank=rank,
        improvement_need=0,
        unique_correct_count=0,
        pivotal_correct_count=0,
        updates_since_selected=0,
        overdue=overdue,
        pareto_front=1,
        seeded_rank=str(agent_id),
        candidate_search_outcome=CandidateSearchOutcome(
            no_positive_candidate_streak=1 if cooling else 0,
            cooldown_until_update=2 if cooling else 0,
            cooling_down=cooling,
        ),
        target_attempt_count=1,
    )


def main() -> int:
    assert select_target_agent((
        row(0, rank=1, gain=0, cooling=False, overdue=False),
        row(1, rank=0, gain=5, cooling=False, overdue=False),
    )) == 0
    assert select_target_agent((
        row(0, rank=1, gain=0, cooling=True, overdue=False),
        row(1, rank=0, gain=0, cooling=False, overdue=False),
    )) == 1
    assert select_target_agent((
        row(0, rank=1, gain=0, cooling=True, overdue=True),
        row(1, rank=3, gain=0, cooling=False, overdue=False),
    )) == 0
    assert select_target_agent((
        row(0, rank=1, gain=0, cooling=True, overdue=False),
        row(1, rank=2, gain=0, cooling=True, overdue=False),
    )) in {0, 1}
    print("deterministic target scheduler smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
