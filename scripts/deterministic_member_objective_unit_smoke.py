from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.member_objectives import member_gain_metrics, pareto_dominates, team_objective_vector
from multi_dataset_diverse_rl.responsibility import MemberAwareRepairOpportunity, ResponsibilityState, select_target_agent, target_priorities


def opportunity(agent: int, direct: bool = False) -> MemberAwareRepairOpportunity:
    return MemberAwareRepairOpportunity(agent, f"q{agent}", False, False, direct, 1.0 if direct else 0.5, True, False, False, False, True)


def main() -> None:
    state = ResponsibilityState(
        assigned_load_by_agent={i: 0 for i in range(5)},
        updates_since_selected_by_agent={i: 0 for i in range(5)},
        accepted_updates_by_agent={i: 0 for i in range(5)},
    )
    assignments = {i: [opportunity(i, i == 0)] for i in range(5)}
    priorities = target_priorities(assignments=assignments, state=state, seed=42, max_wait_updates=8)
    first = select_target_agent(priorities)
    empty = target_priorities(assignments={i: [] for i in range(5)}, state=state, seed=42, max_wait_updates=8)
    initial = (10, 10, 10, 10, 10)
    incumbent = team_objective_vector(12, member_gain_metrics(initial, initial, initial, 0))
    vote_positive_regressing = team_objective_vector(13, member_gain_metrics(initial, initial, (9, 10, 10, 10, 12), 0))
    vote_neutral_worst_positive = team_objective_vector(12, member_gain_metrics(initial, initial, (11, 11, 11, 11, 11), 0))
    report = {
        "first_target_is_assigned_owner": first in range(5),
        "empty_owner_set_returns_no_target": select_target_agent(empty) is None,
        "target_has_no_competence_fields": not any(name in {"gain_count", "improvement_need", "relative_rank"} for name in priorities[0].__dict__),
        "vote_positive_member_regressing_rejected": not pareto_dominates(vote_positive_regressing, incumbent),
        "vote_neutral_worst_member_positive_accepted": pareto_dominates(vote_neutral_worst_positive, incumbent),
    }
    if not all(report.values()):
        raise SystemExit(f"deterministic member-objective smoke failed: {report}")
    print(json.dumps({"unit_smoke": report}, indent=2))


if __name__ == "__main__":
    main()
