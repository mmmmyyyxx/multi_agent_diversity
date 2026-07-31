from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.responsibility import (
    ResponsibilityTargetPriority,
    select_target_agent,
)


def row(
    agent_id: int,
    *,
    direct: int,
    margin: int,
    wait: int,
    overdue: bool,
):
    return ResponsibilityTargetPriority(
        agent_id=agent_id,
        direct_fix_count=direct,
        margin_gain_sum=margin,
        member_gain=0,
        maximum_member_gain=0,
        uplift_deficit=0,
        updates_since_selected=wait,
        overdue=overdue,
        target_pareto_front=1,
        seeded_rank=str(agent_id),
    )


def main() -> int:
    assert select_target_agent(
        (
            row(0, direct=1, margin=2, wait=0, overdue=False),
            row(1, direct=0, margin=9, wait=1, overdue=False),
        )
    ) == 1
    assert select_target_agent(
        (
            row(0, direct=1, margin=2, wait=8, overdue=True),
            row(1, direct=0, margin=9, wait=9, overdue=True),
        )
    ) == 1
    assert select_target_agent(()) is None
    print("deterministic target scheduler smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
