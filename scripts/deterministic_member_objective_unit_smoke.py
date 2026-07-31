from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.member_objectives import (
    member_gain_metrics,
    pareto_dominates,
    team_objective_vector,
)
from multi_dataset_diverse_rl.responsibility import (
    MemberAwareRepairOpportunity,
    ResponsibilityState,
    select_target_agent,
    target_priorities,
)


def opportunity(
    agent: int,
    direct: bool = False,
) -> MemberAwareRepairOpportunity:
    return MemberAwareRepairOpportunity(
        agent_id=agent,
        question_hash=f"q{agent}",
        vote_flip_gain=int(direct),
        margin_gain=2 if direct else 1,
        member_error=True,
        coverage_opportunity=True,
        conversion_opportunity=False,
        dominant_wrong_member=False,
        unique_correct=False,
        pivotal_correct=False,
        oracle_soft_utility_gain=1.0 if direct else 0.5,
    )


def main() -> None:
    state = ResponsibilityState(
        updates_since_selected_by_agent={i: 0 for i in range(5)},
        accepted_updates_by_agent={i: 0 for i in range(5)},
        target_attempt_count_by_agent={i: 0 for i in range(5)},
    )
    assignments = {i: [opportunity(i, i == 0)] for i in range(5)}
    priorities = target_priorities(
        assignments=assignments,
        state=state,
        seed=42,
        max_wait_updates=8,
        current_member_correct_counts=(10, 10, 10, 10, 10),
        initial_member_correct_counts=(10, 10, 10, 10, 10),
        member_uplift_tolerance=5,
    )
    first = select_target_agent(priorities)
    empty = target_priorities(
        assignments={i: [] for i in range(5)},
        state=state,
        seed=42,
        max_wait_updates=8,
        current_member_correct_counts=(10, 10, 10, 10, 10),
        initial_member_correct_counts=(10, 10, 10, 10, 10),
        member_uplift_tolerance=5,
    )
    initial = (10, 10, 10, 10, 10)
    incumbent = team_objective_vector(
        12,
        member_gain_metrics(initial, initial, initial, 0),
    )
    vote_positive_regressing = team_objective_vector(
        13,
        member_gain_metrics(
            initial,
            initial,
            (9, 10, 10, 10, 12),
            0,
        ),
    )
    vote_neutral_worst_positive = team_objective_vector(
        12,
        member_gain_metrics(
            initial,
            initial,
            (11, 11, 11, 11, 11),
            0,
        ),
    )
    report = {
        "first_target_has_responsibility_portfolio": first in range(5),
        "empty_portfolio_set_returns_no_target": (
            select_target_agent(empty) is None
        ),
        "target_vector_is_direct_margin_uplift": all(
            len(row.target_values()) == 3 for row in priorities
        ),
        "vote_positive_member_regressing_rejected": not pareto_dominates(
            vote_positive_regressing,
            incumbent,
        ),
        "vote_neutral_worst_member_positive_accepted": pareto_dominates(
            vote_neutral_worst_positive,
            incumbent,
        ),
    }
    if not all(report.values()):
        raise SystemExit(
            f"deterministic member-objective smoke failed: {report}"
        )
    print(json.dumps({"unit_smoke": report}, indent=2))


if __name__ == "__main__":
    main()
