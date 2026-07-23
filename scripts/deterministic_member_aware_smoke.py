from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def opportunity(agent_id: int, error_count: int, need: int) -> MemberAwareRepairOpportunity:
    return MemberAwareRepairOpportunity(
        agent_id=agent_id,
        question_hash=f"q{error_count}-{agent_id}",
        current_correct=False,
        current_invalid=False,
        direct_vote_fix=agent_id % 2 == 0,
        oracle_soft_utility_gain=0.1,
        coverage_opportunity=True,
        dominant_wrong_member=False,
        unique_correct=False,
        pivotal_correct=False,
        member_correct_count=8 - error_count,
        team_correct_count_sum=40,
        improvement_need=need,
        member_error=True,
        protection_need_count=0,
    )


def main() -> None:
    state = ResponsibilityState(
        assigned_load_by_agent={agent_id: 0 for agent_id in range(5)},
        updates_since_selected_by_agent={agent_id: 0 for agent_id in range(5)},
        accepted_updates_by_agent={agent_id: 0 for agent_id in range(5)},
    )
    opportunities = {
        f"q{agent_id}": (opportunity(agent_id, agent_id + 1, 5 - agent_id),)
        for agent_id in range(5)
    }
    assignments = {
        agent_id: list(opportunities[f"q{agent_id}"]) for agent_id in range(5)
    }
    target_sequence: list[int] = []
    for _ in range(8):
        priorities = target_priorities(
            opportunities=opportunities,
            assignments=assignments,
            state=state,
            seed=42,
            max_wait_updates=4,
        )
        target = select_target_agent(priorities)
        target_sequence.append(target)
        for agent_id in state.updates_since_selected_by_agent:
            state.updates_since_selected_by_agent[agent_id] += 1
        state.updates_since_selected_by_agent[target] = 0

    initial_counts = (10, 10, 10, 10, 10)
    incumbent_gain = member_gain_metrics(initial_counts, initial_counts)
    incumbent = team_objective_vector(12, incumbent_gain)

    vote_positive_regressing = team_objective_vector(
        13,
        member_gain_metrics(initial_counts, (9, 10, 10, 10, 12)),
    )
    vote_neutral_worst_positive = team_objective_vector(
        12,
        member_gain_metrics(initial_counts, (11, 11, 11, 11, 11)),
    )
    final = team_objective_vector(
        14,
        member_gain_metrics(initial_counts, (11, 12, 11, 13, 11)),
    )
    report = {
        "target_sequence": target_sequence,
        "all_erroneous_selected_within_8": set(target_sequence) == set(range(5)),
        "vote_positive_member_regressing_rejected": not pareto_dominates(
            vote_positive_regressing, incumbent
        ),
        "vote_neutral_worst_member_positive_accepted": pareto_dominates(
            vote_neutral_worst_positive, incumbent
        ),
        "final_strictly_dominates_initial": pareto_dominates(final, incumbent),
        "initial_objective": incumbent.as_tuple(),
        "final_objective": final.as_tuple(),
    }
    if not all(
        report[key]
        for key in (
            "all_erroneous_selected_within_8",
            "vote_positive_member_regressing_rejected",
            "vote_neutral_worst_member_positive_accepted",
            "final_strictly_dominates_initial",
        )
    ):
        raise SystemExit(f"deterministic member-aware smoke failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
