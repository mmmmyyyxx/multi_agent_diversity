from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from scripts.v17_hybrid_target_allocation_support import rr_eligible_order


W1 = "W1_TOP2"
HYBRID = "HYBRID_BASE"
ARMS = (W1, HYBRID)
SEEDS = (59, 60, 61)
UPDATES = 8
AUTHORIZATION_ENV = "V18_HYBRID_ONLINE_AUTHORIZED"
ALLOWED_DIAGNOSES = (
    "LONGITUDINAL_ACCUMULATION_WITH_VOTE_CONVERSION",
    "LONGITUDINAL_ACCUMULATION_WITHOUT_VOTE_CONVERSION",
    "THROUGHPUT_RECOVERY_WITHOUT_ACCUMULATION",
    "NO_LONGITUDINAL_ACCUMULATION_SIGNAL",
    "HYBRID_ONLINE_HARMFUL",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hybrid_targets(
    *,
    seed: int,
    update_index: int,
    w1_order: Sequence[int],
    responsibility_eligible: Iterable[int],
) -> tuple[int, int]:
    ordered = tuple(map(int, w1_order))
    eligible = tuple(sorted(set(map(int, responsibility_eligible))))
    if len(ordered) < 2 or len(eligible) < 2:
        raise ValueError("hybrid dual-target selection requires two actionable members")
    if not set(ordered).issubset(set(eligible)):
        raise ValueError("W1 order must be contained in the actionable pool")
    target1 = ordered[0]
    rr_order = rr_eligible_order(seed, update_index, eligible)
    target2 = next((agent for agent in rr_order if agent != target1), None)
    if target2 is None:
        raise ValueError("hybrid exploration has no distinct actionable member")
    return target1, int(target2)


def generation_key(
    *,
    experiment_seed: int,
    update_index: int,
    target_member: int,
    source_slot: int,
    candidate_stage: str,
    parent_team_hash: str,
) -> str:
    return sha256_json({
        "experiment_seed": int(experiment_seed),
        "update_index": int(update_index),
        "target_member": int(target_member),
        "source_slot": int(source_slot),
        "candidate_stage": str(candidate_stage),
        "parent_team_hash": str(parent_team_hash),
    })


def _wtl(values: Sequence[float]) -> dict[str, int]:
    return {
        "wins": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "losses": sum(value < 0 for value in values),
    }


def _transition_counts(states: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "0_to_1": 0,
        "1_to_2": 0,
        "1_to_3_plus": 0,
        "2_to_3": 0,
        "2_to_4_plus": 0,
        "3_to_4_plus": 0,
    }
    for before, after in zip(states, states[1:]):
        left = {row["example_id_hash"]: row for row in before["examples"]}
        right = {row["example_id_hash"]: row for row in after["examples"]}
        if left.keys() != right.keys():
            raise ValueError("longitudinal example identity drift")
        for key in left:
            old = int(left[key]["G"])
            new = int(right[key]["G"])
            if old == 0 and new == 1:
                counts["0_to_1"] += 1
            if old == 1 and new == 2:
                counts["1_to_2"] += 1
            if old == 1 and new >= 3:
                counts["1_to_3_plus"] += 1
            if old == 2 and new == 3:
                counts["2_to_3"] += 1
            if old == 2 and new >= 4:
                counts["2_to_4_plus"] += 1
            if old == 3 and new >= 4:
                counts["3_to_4_plus"] += 1
    return counts


def build_residual_lineage(
    states: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not states:
        raise ValueError("at least one validation state is required")
    by_state = [
        {row["example_id_hash"]: row for row in state["examples"]}
        for state in states
    ]
    if any(rows.keys() != by_state[0].keys() for rows in by_state[1:]):
        raise ValueError("validation example set changed across states")
    results: list[dict[str, Any]] = []
    for example_hash in sorted(by_state[0]):
        sequence = [rows[example_hash] for rows in by_state]
        first_recovery = None
        for index in range(1, len(sequence)):
            if int(sequence[index - 1]["G"]) == 0 and int(sequence[index]["G"]) == 1:
                first_recovery = index
                break
        if first_recovery is None:
            continue
        first_members = set(map(int, sequence[first_recovery]["correct_member_ids"]))
        prior_members = set(map(int, sequence[first_recovery - 1]["correct_member_ids"]))
        new_first = sorted(first_members - prior_members)
        first_member = new_first[0] if len(new_first) == 1 else None
        deep_state = deep_member = two_to_three_state = two_to_three_member = None
        vote_state = None
        for index in range(first_recovery + 1, len(sequence)):
            previous = set(map(int, sequence[index - 1]["correct_member_ids"]))
            current = set(map(int, sequence[index]["correct_member_ids"]))
            newly_correct = sorted(current - previous)
            if deep_state is None and int(sequence[index]["G"]) >= 2:
                deep_state = index
                deep_member = newly_correct[0] if len(newly_correct) == 1 else None
            if (
                two_to_three_state is None
                and int(sequence[index - 1]["G"]) == 2
                and int(sequence[index]["G"]) >= 3
            ):
                two_to_three_state = index
                two_to_three_member = newly_correct[0] if len(newly_correct) == 1 else None
            if vote_state is None and int(sequence[index]["M"]) > 0:
                vote_state = index
        after = sequence[first_recovery + 1 :]
        maximum_later_g = max(
            [int(row["G"]) for row in after], default=int(sequence[first_recovery]["G"])
        )
        margins = [int(row["M"]) for row in sequence[first_recovery:]]
        wrong = [int(row["H"]) for row in sequence[first_recovery:]]
        result = {
            "example_id_hash": example_hash,
            "initial_G": int(sequence[0]["G"]),
            "initial_H": int(sequence[0]["H"]),
            "initial_M": int(sequence[0]["M"]),
            "first_0_to_1_state": first_recovery,
            "first_supporting_member": first_member,
            "later_1_to_2_state": deep_state,
            "later_supporting_member": deep_member,
            "later_2_to_3_state": two_to_three_state,
            "later_2_to_3_supporting_member": two_to_three_member,
            "first_vote_conversion_state": vote_state,
            "final_G": int(sequence[-1]["G"]),
            "final_H": int(sequence[-1]["H"]),
            "final_M": int(sequence[-1]["M"]),
            "persistent_singleton": deep_state is None and maximum_later_g == 1,
            "cross_member_accumulation": bool(
                deep_state is not None
                and first_member is not None
                and deep_member is not None
                and first_member != deep_member
            ),
            "margin_improved": max(margins) > margins[0],
            "margin_unchanged": max(margins) == min(margins) == margins[0],
            "margin_worsened": min(margins) < margins[0],
            "min_margin": min(margins),
            "max_margin": max(margins),
            "final_margin": margins[-1],
            "wrong_coalition_decreased": min(wrong) < wrong[0],
            "wrong_coalition_unchanged": min(wrong) == max(wrong) == wrong[0],
            "wrong_coalition_increased": max(wrong) > wrong[0],
        }
        results.append(result)
    return results


def summarize_trajectory(
    *,
    states: Sequence[Mapping[str, Any]],
    accepted_commit_count: int,
    feasible_branch_count: int,
    feasible_candidate_count: int,
    update_opportunities: int,
) -> dict[str, Any]:
    lineage = build_residual_lineage(states)
    deep = sum(row["later_1_to_2_state"] is not None for row in lineage)
    persistent = sum(bool(row["persistent_singleton"]) for row in lineage)
    denominator = deep + persistent
    initial = states[0]["metrics"]
    final = states[-1]["metrics"]
    return {
        "update_opportunities": int(update_opportunities),
        "accepted_commit_count": int(accepted_commit_count),
        "no_commit_update_count": int(update_opportunities - accepted_commit_count),
        "feasible_branch_count": int(feasible_branch_count),
        "feasible_candidate_count": int(feasible_candidate_count),
        "recovered_singleton_count": len(lineage),
        "longitudinal_deepened_coverage_count": deep,
        "persistent_singleton_count": persistent,
        "deepening_rate": (deep / denominator if denominator else None),
        "deepening_rate_support": denominator,
        "cross_member_support_accumulation_count": sum(
            bool(row["cross_member_accumulation"]) for row in lineage
        ),
        "recovered_coverage_to_vote_count": sum(
            row["first_vote_conversion_state"] is not None for row in lineage
        ),
        "support_transitions": _transition_counts(states),
        "initial_validation_vote_correct": int(initial["vote_correct_count"]),
        "final_validation_vote_correct": int(final["vote_correct_count"]),
        "final_validation_vote_acc": float(final["plurality_vote_acc"]),
        "final_validation_vote_delta": int(
            final["vote_correct_count"] - initial["vote_correct_count"]
        ),
        "initial_validation_oracle_correct": int(initial["oracle_correct_count"]),
        "final_validation_oracle_correct": int(final["oracle_correct_count"]),
        "final_validation_oracle_acc": float(final["oracle_acc"]),
        "final_validation_oracle_delta": int(
            final["oracle_correct_count"] - initial["oracle_correct_count"]
        ),
    }


def classify(trajectories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {
        (int(row["seed"]), str(row["arm"])): row for row in trajectories
    }
    if set(by_key) != {(seed, arm) for seed in SEEDS for arm in ARMS}:
        raise ValueError("classifier requires exactly three matched two-arm seeds")

    def diffs(field: str) -> list[float]:
        return [
            float(by_key[(seed, HYBRID)][field])
            - float(by_key[(seed, W1)][field])
            for seed in SEEDS
        ]

    deep = diffs("longitudinal_deepened_coverage_count")
    vote_conversion = diffs("recovered_coverage_to_vote_count")
    accepted = diffs("accepted_commit_count")
    feasible = diffs("feasible_branch_count")
    final_vote = diffs("final_validation_vote_acc")
    final_oracle = diffs("final_validation_oracle_acc")
    deep_wtl = _wtl(deep)
    vote_conversion_wtl = _wtl(vote_conversion)
    accepted_wtl = _wtl(accepted)
    feasible_wtl = _wtl(feasible)
    vote_wtl = _wtl(final_vote)

    accumulation = sum(deep) / len(deep) > 0 and deep_wtl["wins"] > deep_wtl["losses"]
    conversion = (
        sum(vote_conversion) > 0
        and vote_conversion_wtl["wins"] > vote_conversion_wtl["losses"]
    )
    accepted_recovery = (
        sum(accepted) / len(accepted) > 0
        and accepted_wtl["wins"] > accepted_wtl["losses"]
    )
    feasible_recovery = (
        sum(feasible) > 0
        and feasible_wtl["wins"] > feasible_wtl["losses"]
    )
    throughput = accepted_recovery or feasible_recovery

    rate_diffs: list[float] = []
    low_support = False
    for seed in SEEDS:
        left = by_key[(seed, HYBRID)]
        right = by_key[(seed, W1)]
        if min(
            int(left["deepening_rate_support"]),
            int(right["deepening_rate_support"]),
        ) < 2:
            low_support = True
        if left["deepening_rate"] is not None and right["deepening_rate"] is not None:
            rate_diffs.append(float(left["deepening_rate"]) - float(right["deepening_rate"]))
    rate_wtl = _wtl(rate_diffs)
    singleton_reduced = bool(
        rate_diffs
        and sum(rate_diffs) / len(rate_diffs) > 0
        and rate_wtl["wins"] > rate_wtl["losses"]
    )
    final_vote_signal = (
        sum(final_vote) / len(final_vote) > 0
        and vote_wtl["wins"] > vote_wtl["losses"]
    )
    harmful = (
        sum(final_vote) / len(final_vote) < 0
        and vote_wtl["losses"] > vote_wtl["wins"]
    )
    if harmful:
        diagnosis = "HYBRID_ONLINE_HARMFUL"
    elif accumulation and conversion:
        diagnosis = "LONGITUDINAL_ACCUMULATION_WITH_VOTE_CONVERSION"
    elif accumulation:
        diagnosis = "LONGITUDINAL_ACCUMULATION_WITHOUT_VOTE_CONVERSION"
    elif throughput:
        diagnosis = "THROUGHPUT_RECOVERY_WITHOUT_ACCUMULATION"
    else:
        diagnosis = "NO_LONGITUDINAL_ACCUMULATION_SIGNAL"
    if diagnosis not in ALLOWED_DIAGNOSES:
        raise AssertionError("unfrozen final diagnosis")
    return {
        "classifier_version": "v18_hybrid_online_accumulation_classifier_v1",
        "per_seed_deepening_delta": dict(zip(map(str, SEEDS), deep, strict=True)),
        "per_seed_vote_conversion_delta": dict(
            zip(map(str, SEEDS), vote_conversion, strict=True)
        ),
        "per_seed_final_vote_acc_delta": dict(
            zip(map(str, SEEDS), final_vote, strict=True)
        ),
        "per_seed_final_oracle_acc_delta": dict(
            zip(map(str, SEEDS), final_oracle, strict=True)
        ),
        "online_accumulation_supported": accumulation,
        "online_vote_conversion_signal": conversion,
        "hybrid_throughput_recovery_reproduced": throughput,
        "accepted_commit_recovery": accepted_recovery,
        "feasible_branch_recovery": feasible_recovery,
        "persistent_singleton_reduced": singleton_reduced,
        "persistent_singleton_low_support": low_support,
        "final_validation_vote_signal": final_vote_signal,
        "deepening_wtl": deep_wtl,
        "vote_conversion_wtl": vote_conversion_wtl,
        "accepted_commit_wtl": accepted_wtl,
        "feasible_branch_wtl": feasible_wtl,
        "final_vote_wtl": vote_wtl,
        "final_diagnosis": diagnosis,
    }
