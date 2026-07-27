from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.responsibility import ResponsibilityTargetPriority, select_target_agent


def row(agent_id: int, *, direct: int, age: int, wait: int, overdue: bool):
    return ResponsibilityTargetPriority(
        agent_id=agent_id, assigned_load=1,
        owned_direct_vote_fix_count=direct, owned_oracle_soft_utility_gain_sum=0.0,
        owned_coverage_opportunity_count=0, owned_dominant_wrong_count=0,
        oldest_owned_responsibility_age=age, updates_since_selected=wait,
        overdue=overdue, pareto_front=1, seeded_rank=str(agent_id),
    )


def main() -> int:
    assert select_target_agent((row(0, direct=1, age=0, wait=0, overdue=False), row(1, direct=0, age=9, wait=0, overdue=False))) == 1
    assert select_target_agent((row(0, direct=1, age=0, wait=0, overdue=False), row(1, direct=0, age=0, wait=8, overdue=True))) == 1
    assert select_target_agent(()) is None
    print("deterministic target scheduler smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
