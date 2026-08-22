from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_v16_residual_diag_probe_registry import read_jsonl
from build_v17_module1_2x2_registry import reconstruct_case, source_run
from v17_hybrid_target_allocation_support import rr_eligible_order


SEEDS = (56, 57, 58)
SIGNALS = (
    "conversion_responsibility_count",
    "singleton_conversion_count",
    "direct_vote_flip_count",
    "dominant_wrong_weakening_count",
)
PROSPECTIVE_PARENTS = {(56, 4), (56, 7), (57, 6), (57, 7), (58, 6)}


def is_conversion_residual(row: dict[str, Any]) -> bool:
    gold = int(row["gold_vote_count"])
    wrong = int(row["largest_wrong_vote_count"])
    return 0 < gold <= wrong


def select_signal_target(
    rr_order: list[int], scores: dict[int, dict[str, int]], signal: str
) -> int:
    if not rr_order or signal not in SIGNALS:
        raise ValueError("invalid signal selector input")
    return max(
        rr_order,
        key=lambda agent: (int(scores[agent][signal]), -rr_order.index(agent)),
    )


def _responsibility_row(seed: int, team_state_version: int) -> dict[str, Any]:
    rows = [
        row
        for row in read_jsonl(source_run(seed) / "responsibility_assignments.jsonl")
        if int(row["team_state_version"]) == int(team_state_version)
    ]
    if len(rows) != 1:
        raise ValueError("responsibility state is not uniquely reconstructable")
    return rows[0]


def reconstruct_state(seed: int, update_index: int) -> dict[str, Any]:
    case = reconstruct_case({
        "case_id": f"historical_seed{seed}_u{update_index}",
        "seed": int(seed),
        "update_index": int(update_index),
    })
    eligible = [int(row["agent_id"]) for row in case["w1_priority_rows"]]
    exploit = eligible[0]
    rr_order = [
        agent
        for agent in rr_eligible_order(seed, update_index, eligible)
        if agent != exploit
    ]
    if not rr_order:
        raise ValueError("Base Hybrid has no exploration target")

    states = {str(row["question_hash"]): row for row in case["active_profiles"]}
    conversion = {
        question_hash: row
        for question_hash, row in states.items()
        if is_conversion_residual(row)
    }
    responsibility = _responsibility_row(seed, int(case["team_state_version"]))
    service: dict[int, dict[str, dict[str, Any]]] = {agent: {} for agent in range(5)}
    for question_hash, assignment in responsibility["service_assignment_by_question"].items():
        service[int(assignment["service_agent_id"])][str(question_hash)] = assignment

    scores: dict[int, dict[str, int]] = {}
    for agent in rr_order:
        hashes = sorted(set(service[agent]) & set(conversion))
        for question_hash in hashes:
            if bool(conversion[question_hash]["team_correctness"][agent]):
                raise ValueError("responsibility member is already correct on residual")
        scores[agent] = {
            "conversion_responsibility_count": len(hashes),
            "singleton_conversion_count": sum(
                int(conversion[question_hash]["gold_vote_count"]) == 1
                for question_hash in hashes
            ),
            "direct_vote_flip_count": sum(
                str(service[agent][question_hash]["repair_lane"]) == "direct_flip"
                for question_hash in hashes
            ),
            "dominant_wrong_weakening_count": sum(
                str(conversion[question_hash]["team_answers"][agent])
                in set(map(str, conversion[question_hash]["dominant_wrong_answers"]))
                for question_hash in hashes
            ),
        }

    base = rr_order[0]
    selections = {
        signal: select_signal_target(rr_order, scores, signal) for signal in SIGNALS
    }
    row: dict[str, Any] = {
        "parent_id": case["case_id"],
        "seed": int(seed),
        "update_index": int(update_index),
        "prospective_five_parent": (int(seed), int(update_index)) in PROSPECTIVE_PARENTS,
        "w1_rank1_target": exploit,
        "base_target2": base,
        "conversion_residual_count": len(conversion),
        "conversion_eligible_alternative_count": sum(
            scores[agent]["conversion_responsibility_count"] > 0 for agent in rr_order
        ),
        "rr_order": rr_order,
        "member_scores": {str(agent): scores[agent] for agent in rr_order},
        "selections": selections,
        "different_from_base": {
            signal: selections[signal] != base for signal in SIGNALS
        },
    }
    return row


def historical_states() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        run = source_run(seed)
        updates = sorted({
            int(row["update_index"])
            for row in read_jsonl(run / "candidate_decisions.jsonl")
        })
        rows.extend(reconstruct_state(seed, update) for update in updates)
    return rows


def _signal_summary(rows: list[dict[str, Any]], signal: str) -> dict[str, Any]:
    differences = [row for row in rows if row["different_from_base"][signal]]
    return {
        "state_count": len(rows),
        "target2_different_count": len(differences),
        "target2_same_count": len(rows) - len(differences),
        "different_parent_ids": [row["parent_id"] for row in differences],
    }


def analyze() -> dict[str, Any]:
    rows = historical_states()
    conversion_active = [row for row in rows if row["conversion_residual_count"] > 0]
    prospective = [row for row in rows if row["prospective_five_parent"]]
    if len(rows) != 24 or len(prospective) != 5:
        raise ValueError("historical inventory mismatch")
    identical_count_signals = all(
        all(
            scores["conversion_responsibility_count"]
            == scores["dominant_wrong_weakening_count"]
            for scores in row["member_scores"].values()
        )
        for row in rows
    )
    return {
        "analysis_version": "v17_conversion_target2_differentiation_v1",
        "zero_api": True,
        "candidate_outcomes_used": False,
        "validation_used": False,
        "test_used": False,
        "method_changed": False,
        "historical_state_count": len(rows),
        "conversion_active_state_count": len(conversion_active),
        "prospective_five_parent_count": len(prospective),
        "signals": {
            signal: {
                "all_historical": _signal_summary(rows, signal),
                "conversion_active": _signal_summary(conversion_active, signal),
                "prospective_five_parent": _signal_summary(prospective, signal),
            }
            for signal in SIGNALS
        },
        "dominant_wrong_signal_identical_to_conversion_count": identical_count_signals,
        "states": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    result = analyze()
    print(json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if args.compact else None,
        indent=None if args.compact else 2,
    ))


if __name__ == "__main__":
    main()
