"""Read-only, sanitized observed-state replay for responsibility policy history."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.responsibility import (
    MemberAwareRepairOpportunity,
    ResponsibilityState,
    build_target_selection_decision,
    target_priorities,
)


SOURCE = ROOT / "reports" / "v11_full_seed43_32updates_20260727"
OWNER = ROOT / "reports" / "v11_full_seed43_32updates_20260727_owner_alignment_audit"
OUT = ROOT / "reports" / "responsibility_mechanism_history_audit"


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def opportunity(row: dict) -> MemberAwareRepairOpportunity:
    return MemberAwareRepairOpportunity(**{
        key: row[key] for key in MemberAwareRepairOpportunity.__dataclass_fields__
    })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assignments = rows(SOURCE / "responsibility_assignments_sanitized.jsonl")
    replay = rows(OWNER / "owner_policy_replay.jsonl")
    by_state_policy: dict[tuple[int, str], list[dict]] = {}
    for row in replay:
        by_state_policy.setdefault((int(row["team_state_version"]), row["policy"]), []).append(row)
    policy_rows = []
    for state_row in assignments:
        version = int(state_row["team_state_version"])
        ages = {str(k): int(v) for k, v in state_row["owner_age"].items()}
        for label, owner_policy in (("B_repair_only_historical_owner", "B_repair_only"), ("R_repair_only_portfolio", "B_repair_only")):
            assigned = {agent: [] for agent in range(5)}
            for owner_row in by_state_policy[(version, owner_policy)]:
                owner = int(owner_row["chosen_owner"])
                # The v11 owner replay stores exactly the repair vector needed here.
                source_row = next(
                    item for items in state_row["assigned_opportunities"].values() for item in items
                    if item["question_hash"] == owner_row["question_hash"]
                )
                vector = owner_row["candidate_vectors"][str(owner)]
                coverage_index = 3 if len(vector) >= 5 else 2
                dominant_index = 4 if len(vector) >= 5 else 3
                assigned[owner].append(opportunity({
                    **source_row,
                    "agent_id": owner,
                    "direct_vote_fix": bool(vector[0]),
                    "oracle_soft_utility_gain": float(vector[1]),
                    "coverage_opportunity": bool(vector[coverage_index]),
                    "dominant_wrong_member": bool(vector[dominant_index]),
                }))
            state = ResponsibilityState(
                owner_age_by_question=ages,
                updates_since_selected_by_agent={agent: 0 for agent in range(5)},
                assigned_load_by_agent={agent: len(items) for agent, items in assigned.items()},
            )
            priorities = target_priorities(assignments=assigned, state=state, seed=43, max_wait_updates=8)
            decision = build_target_selection_decision(priorities)
            policy_rows.append({
                "artifact_schema_version": "observed_state_target_replay_v1",
                "team_state_version": version,
                "policy": label,
                "selected_agent_id": decision.selected_agent_id,
                "eligible_agent_ids": list(decision.eligible_agent_ids),
                "selection_pool_stage": decision.selection_pool_stage,
                "priority_rows": [asdict(row) for row in priorities],
            })
    (OUT / "observed_state_policy_replay.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in policy_rows), encoding="utf-8"
    )
    availability = {
        "source": "sanitized v11 Full seed43 artifacts only",
        "available": {
            "A_legacy_observed_targets": 32,
            "B_repair_only_owner_states": 14,
            "R_repair_only_portfolio_states": 14,
        },
        "unavailable": {
            "H0_pre_member_aware_observed_state_replay": "historical sanitized per-question state is unavailable",
            "H1_or_29f2_target_replay": "no compatible sanitized target/state join exists",
            "counterfactual_training_trajectory": "not identifiable from fixed observed states",
            "counterfactual_test_or_efficacy": "not evaluated",
        },
        "rule": "unavailable is not encoded as zero",
    }
    (OUT / "historical_evidence_availability.json").write_text(json.dumps(availability, indent=2) + "\n", encoding="utf-8")
    matrix = [
        {"label": "H0", "period": "pre-member-aware peer-state", "owner_semantics": "repair evidence with distinct all-agent target selection", "replay": "unavailable", "evidence": "source inspection only"},
        {"label": "H1", "period": "member-aware v1", "owner_semantics": "owner vector included improvement_need", "replay": "unavailable", "evidence": "source inspection only"},
        {"label": "A", "period": "v11 archived behavior", "owner_semantics": "five-axis owner and global-error target pool", "replay": "available: 32 archived target decisions", "evidence": "candidate_decisions_sanitized"},
        {"label": "B", "period": "repair-only historical owner replay", "owner_semantics": "four repair axes only", "replay": "available: 14 observed states", "evidence": "owner_policy_replay"},
        {"label": "P", "period": "fec8f8d", "owner_semantics": "relative-gain potential precedes team repair", "replay": "source/audit only", "evidence": "scheduler_rank_direction_summary"},
        {"label": "R", "period": "v5 proposed", "owner_semantics": "repair-only owner plus assigned-portfolio target", "replay": "available: 14 observed states", "evidence": "this report"},
    ]
    (OUT / "responsibility_history_matrix.json").write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    markdown = "# Responsibility mechanism history audit\n\n"
    markdown += "This is a read-only replay over sanitized v11 observed states. It does not estimate counterfactual training, final test, or efficacy.\n\n"
    markdown += "| Label | Owner/target semantics | Replay availability |\n|---|---|---|\n"
    markdown += "\n".join(f"| {r['label']} | {r['owner_semantics']} | {r['replay']} |" for r in matrix)
    markdown += "\n\nPolicy A remains the archived behavior. B and R are deterministic replays on 14 fixed responsibility states; their target rows are not a counterfactual 32-update trajectory because historical decision-to-state joins are unavailable. H0/H1 unavailable fields are explicitly not zero.\n"
    (OUT / "README.md").write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
