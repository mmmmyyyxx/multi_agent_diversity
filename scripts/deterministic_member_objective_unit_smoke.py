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


def opportunities_for_counts(
    initial_counts: tuple[int, ...],
    current_counts: tuple[int, ...],
) -> dict[str, tuple[MemberAwareRepairOpportunity, ...]]:
    gains = tuple(
        current - initial
        for current, initial in zip(current_counts, initial_counts, strict=True)
    )
    total_gain = sum(gains)
    member_count = len(gains)
    rows: dict[str, tuple[MemberAwareRepairOpportunity, ...]] = {}
    for agent_id in range(member_count):
        need = max(0, total_gain - member_count * gains[agent_id])
        row = MemberAwareRepairOpportunity(
            agent_id=agent_id,
            question_hash=f"q{agent_id}",
            current_correct=False,
            current_invalid=False,
            direct_vote_fix=agent_id == 0,
            oracle_soft_utility_gain=1.0 if agent_id == 0 else 0.5,
            coverage_opportunity=True,
            dominant_wrong_member=False,
            unique_correct=False,
            pivotal_correct=False,
            initial_correct_count=initial_counts[agent_id],
            current_correct_count=current_counts[agent_id],
            gain_count=gains[agent_id],
            improvement_need=need,
            unique_correct_count=0,
            pivotal_correct_count=0,
            member_error=True,
            protection_need_count=0,
        )
        rows[row.question_hash] = (row,)
    return rows


def priorities_for_counts(
    *,
    initial_counts: tuple[int, ...],
    current_counts: tuple[int, ...],
    state: ResponsibilityState,
):
    opportunities = opportunities_for_counts(initial_counts, current_counts)
    assignments = {
        agent_id: list(opportunities[f"q{agent_id}"])
        for agent_id in range(5)
    }
    priorities = target_priorities(
        opportunities=opportunities,
        assignments=assignments,
        state=state,
        seed=42,
        max_wait_updates=4,
    )
    return priorities


def main() -> None:
    initial_counts = (10, 10, 10, 10, 10)
    current_counts = list(initial_counts)
    state = ResponsibilityState(
        assigned_load_by_agent={agent_id: 0 for agent_id in range(5)},
        updates_since_selected_by_agent={agent_id: 0 for agent_id in range(5)},
        accepted_updates_by_agent={agent_id: 0 for agent_id in range(5)},
    )
    target_sequence: list[int] = []
    selection_counts = [0] * 5
    accepted_counts = [0] * 5
    vote_count = 12
    need_after_first_uplift: tuple[int, ...] | None = None

    for update_index in range(8):
        priorities = priorities_for_counts(
            initial_counts=initial_counts,
            current_counts=tuple(current_counts),
            state=state,
        )
        target = select_target_agent(priorities)
        target_sequence.append(target)
        selection_counts[target] += 1

        # The deterministic trajectory accepts one substantive improvement for
        # agents 0-3 and deliberately leaves agent 4 unimproved.
        if target != 4 and accepted_counts[target] == 0:
            current_counts[target] += 1
            accepted_counts[target] += 1
            state.accepted_updates_by_agent[target] += 1
            vote_count = min(14, vote_count + 1)

        for agent_id in state.updates_since_selected_by_agent:
            state.updates_since_selected_by_agent[agent_id] += 1
        state.updates_since_selected_by_agent[target] = 0

        if update_index == 0:
            refreshed = priorities_for_counts(
                initial_counts=initial_counts,
                current_counts=tuple(current_counts),
                state=state,
            )
            need_after_first_uplift = tuple(
                row.improvement_need for row in refreshed
            )

    incumbent_gain = member_gain_metrics(
        initial_counts,
        initial_counts,
        initial_counts,
        0,
    )
    incumbent = team_objective_vector(12, incumbent_gain)
    vote_positive_regressing = team_objective_vector(
        13,
        member_gain_metrics(
            initial_counts,
            initial_counts,
            (9, 10, 10, 10, 12),
            0,
        ),
    )
    vote_neutral_worst_positive = team_objective_vector(
        12,
        member_gain_metrics(
            initial_counts,
            initial_counts,
            (11, 11, 11, 11, 11),
            0,
        ),
    )
    final_gain = member_gain_metrics(
        initial_counts,
        tuple(current_counts),
        tuple(current_counts),
        0,
    )
    final = team_objective_vector(vote_count, final_gain)

    report = {
        "target_sequence": target_sequence,
        "per_agent_selection_counts": selection_counts,
        "per_agent_accepted_counts": accepted_counts,
        "gain_vector": final_gain.gain_counts,
        "minimum_gain": final_gain.minimum_gain_count,
        "total_gain": final_gain.total_gain_count,
        "vote_count": vote_count,
        "all_members_improved": final_gain.all_members_improved,
        "agent_0_selected_first_for_vote_leverage": target_sequence[0] == 0,
        "other_member_need_increased_after_agent_0_uplift": bool(
            need_after_first_uplift
            and all(value > 0 for value in need_after_first_uplift[1:])
        ),
        "all_erroneous_selected_within_8": set(target_sequence) == set(range(5)),
        "vote_positive_member_regressing_rejected": not pareto_dominates(
            vote_positive_regressing,
            incumbent,
        ),
        "vote_neutral_worst_member_positive_accepted": pareto_dominates(
            vote_neutral_worst_positive,
            incumbent,
        ),
        "final_strictly_dominates_initial": pareto_dominates(final, incumbent),
    }
    required_checks = (
        "agent_0_selected_first_for_vote_leverage",
        "other_member_need_increased_after_agent_0_uplift",
        "all_erroneous_selected_within_8",
        "vote_positive_member_regressing_rejected",
        "vote_neutral_worst_member_positive_accepted",
        "final_strictly_dominates_initial",
    )
    if not all(report[key] for key in required_checks):
        raise SystemExit(f"deterministic member-aware smoke failed: {report}")
    print(json.dumps({"unit_smoke": report}, indent=2))


if __name__ == "__main__":
    main()
