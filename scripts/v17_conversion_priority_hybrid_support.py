from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from v17_hybrid_target_allocation_support import (
    branch_key,
    branch_object,
    canonical_hash,
    choose_would_commit,
    context_hashes,
    generation_config,
    immutable_state_hash,
    probe_system,
    realized_delta,
    rr_eligible_order,
)


BASE = "HYBRID_BASE"
BREADTH = "HYBRID_BREADTH_PRIORITY"
DIRECT = "HYBRID_DIRECT_FLIP_PRIORITY"
ARMS = (BASE, BREADTH, DIRECT)
AUTHORIZATION_ENV = "V17_CONVERSION_PRIORITY_LOW_API_AUTHORIZED"
CLASSIFIER_VERSION = "v17_conversion_priority_three_arm_classifier_v1"
ALLOWED_DIAGNOSES = (
    "BOTH_LOCAL_SIGNALS",
    "BREADTH_LOCAL_SIGNAL",
    "DIRECT_FLIP_LOCAL_SIGNAL",
    "PRIORITY_FILTER_HARMS_THROUGHPUT",
    "NO_CLEAR_PRIORITY_SIGNAL",
)


def _priority_target(
    rr_order: list[int], scores: dict[int, dict[str, int]], field: str
) -> int:
    if not rr_order:
        raise ValueError("empty exploration RR order")
    return max(
        rr_order,
        key=lambda agent: (int(scores[agent][field]), -rr_order.index(agent)),
    )


def arm_specs(case: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    eligible = [int(row["agent_id"]) for row in case["w1_priority_rows"]]
    exploit = eligible[0]
    rr_order = [
        agent
        for agent in rr_eligible_order(
            case["source_seed"], case["source_update_index"], eligible
        )
        if agent != exploit
    ]
    scores = {
        int(agent): {key: int(value) for key, value in row.items()}
        for agent, row in case["selector_scores_by_agent"].items()
    }
    base_target = rr_order[0]
    breadth_target = _priority_target(
        rr_order, scores, "conversion_responsibility_count"
    )
    direct_target = _priority_target(rr_order, scores, "direct_vote_flip_count")
    result = {
        BASE: [
            {"target_member": exploit, "branch_type": "exploit"},
            {"target_member": base_target, "branch_type": "base_explore"},
        ],
        BREADTH: [
            {"target_member": exploit, "branch_type": "exploit"},
            {"target_member": breadth_target, "branch_type": "breadth_explore"},
        ],
        DIRECT: [
            {"target_member": exploit, "branch_type": "exploit"},
            {"target_member": direct_target, "branch_type": "direct_flip_explore"},
        ],
    }
    legal = set(eligible)
    for rows in result.values():
        targets = [int(row["target_member"]) for row in rows]
        if len(targets) != 2 or len(set(targets)) != 2 or not set(targets).issubset(legal):
            raise ValueError("three-arm target contract failed")
    return result


def _wtl(left: Iterable[int], right: Iterable[int]) -> dict[str, int]:
    deltas = [a - b for a, b in zip(left, right, strict=True)]
    return {
        "wins": sum(value > 0 for value in deltas),
        "ties": sum(value == 0 for value in deltas),
        "losses": sum(value < 0 for value in deltas),
    }


def classify(parent_rows: list[dict[str, Any]], funnel: dict[str, dict[str, int]]) -> dict[str, Any]:
    def values(arm: str, metric: str) -> list[int]:
        return [int(row[arm][metric]) for row in parent_rows]

    base_deeper = values(BASE, "deeper_support_gain_count")
    breadth_deeper = values(BREADTH, "deeper_support_gain_count")
    base_vote = values(BASE, "validation_vote_delta")
    breadth_vote = values(BREADTH, "validation_vote_delta")
    direct_vote = values(DIRECT, "validation_vote_delta")
    base_conversions = values(BASE, "vote_conversion_count")
    direct_conversions = values(DIRECT, "vote_conversion_count")
    base_oracle = values(BASE, "validation_oracle_delta")
    breadth_oracle = values(BREADTH, "validation_oracle_delta")
    direct_oracle = values(DIRECT, "validation_oracle_delta")

    breadth_wtl = _wtl(breadth_deeper, base_deeper)
    direct_vote_wtl = _wtl(direct_vote, base_vote)
    direct_conversion_wtl = _wtl(direct_conversions, base_conversions)
    breadth_vote_wtl = _wtl(breadth_vote, base_vote)
    breadth_deeper_supported = (
        sum(breadth_deeper) > sum(base_deeper)
        and breadth_wtl["wins"] > breadth_wtl["losses"]
    )
    direct_vote_signal = (
        sum(direct_vote) > sum(base_vote)
        and direct_vote_wtl["wins"] > direct_vote_wtl["losses"]
    )
    direct_conversion_signal = (
        sum(direct_conversions) > sum(base_conversions)
        and direct_conversion_wtl["wins"] > direct_conversion_wtl["losses"]
    )
    throughput = {
        arm: (
            funnel[arm]["feasible_branch_count"] >= funnel[BASE]["feasible_branch_count"]
            and funnel[arm]["would_commit_count"] >= funnel[BASE]["would_commit_count"]
        )
        for arm in (BREADTH, DIRECT)
    }
    oracle_not_harmed = {
        BREADTH: sum(breadth_oracle) >= sum(base_oracle),
        DIRECT: sum(direct_oracle) >= sum(base_oracle),
    }
    breadth_supported = (
        breadth_deeper_supported
        and throughput[BREADTH]
        and sum(breadth_vote) >= sum(base_vote)
    )
    direct_supported = (
        (direct_vote_signal or direct_conversion_signal)
        and throughput[DIRECT]
        and sum(direct_vote) >= sum(base_vote)
    )
    both_throughput_harmed = all(not throughput[arm] for arm in (BREADTH, DIRECT))
    if breadth_supported and direct_supported:
        diagnosis = "BOTH_LOCAL_SIGNALS"
    elif breadth_supported:
        diagnosis = "BREADTH_LOCAL_SIGNAL"
    elif direct_supported:
        diagnosis = "DIRECT_FLIP_LOCAL_SIGNAL"
    elif both_throughput_harmed:
        diagnosis = "PRIORITY_FILTER_HARMS_THROUGHPUT"
    else:
        diagnosis = "NO_CLEAR_PRIORITY_SIGNAL"
    if diagnosis not in ALLOWED_DIAGNOSES:
        raise AssertionError("invalid diagnosis")
    return {
        "classifier_version": CLASSIFIER_VERSION,
        "breadth_deeper_support_signal": breadth_deeper_supported,
        "direct_flip_vote_signal": direct_vote_signal,
        "direct_flip_conversion_signal": direct_conversion_signal,
        "throughput_not_harmed": throughput,
        "oracle_not_harmed": oracle_not_harmed,
        "breadth_deeper_support_wtl": breadth_wtl,
        "breadth_vote_wtl": breadth_vote_wtl,
        "direct_vote_wtl": direct_vote_wtl,
        "direct_conversion_wtl": direct_conversion_wtl,
        "breadth_local_mechanism_signal": breadth_supported,
        "direct_flip_local_mechanism_signal": direct_supported,
        "final_diagnosis": diagnosis,
    }
