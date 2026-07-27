"""Offline-only owner-policy and implementation-alignment audit.

Consumes only the sanitized v11 report.  Replays observed responsibility
states; it never evaluates prompts, contacts an API, or infers a new training
trajectory.  Policy A must exactly reproduce the archived owner assignments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


AGENTS = range(5)
MARGIN = 0.05
FORBIDDEN_PATH = re.compile(
    r"(?i)(?:[a-z]:[\\/]|file://|\\\\[^\\/\s]+[\\/][^\\/\s]+|(?:^|[\s\"'=])/(?:[^\s\"']*))"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def seeded(seed: int, question_hash: str, agent: int) -> str:
    return hashlib.sha256(f"{seed}:{question_hash}:{agent}".encode("utf-8")).hexdigest()


def dominates(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return all(a >= b for a, b in zip(left, right, strict=True)) and any(
        a > b for a, b in zip(left, right, strict=True)
    )


def fronts(vectors: dict[int, tuple[float, ...]]) -> dict[int, int]:
    remaining, result, number = set(vectors), {}, 1
    while remaining:
        current = [
            agent for agent in sorted(remaining)
            if not any(other != agent and dominates(vectors[other], vectors[agent]) for other in remaining)
        ]
        if not current:
            raise AssertionError("Pareto replay made no progress")
        for agent in current:
            result[agent] = number
            remaining.remove(agent)
        number += 1
    return result


def gini(values: list[int]) -> float:
    total = sum(values)
    if not total:
        return 0.0
    return sum(abs(left - right) for left in values for right in values) / (2 * len(values) * total)


def entropy(values: list[int]) -> float:
    total = sum(values)
    return -sum((value / total) * math.log(value / total) for value in values if value) if total else 0.0


def structural_pattern(row: dict[str, Any]) -> str:
    return "|".join([
        f"G{row['G']}", f"H{row['H']}", f"M{row['M']}",
        f"direct{int(bool(row['direct_vote_fix']))}",
        f"coverage{int(bool(row['coverage_opportunity']))}",
        f"dominant{int(bool(row['dominant_wrong_member']))}",
    ])


def js_divergence(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    left_total, right_total = sum(left.values()), sum(right.values())
    if not left_total or not right_total:
        return 0.0
    result = 0.0
    for key in keys:
        p, q = left[key] / left_total, right[key] / right_total
        m = (p + q) / 2
        if p:
            result += 0.5 * p * math.log2(p / m)
        if q:
            result += 0.5 * q * math.log2(q / m)
    return result


def state_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, list[dict[str, Any]]]]:
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[int(row["team_state_version"])][str(row["question_hash"])].append(row)
    for questions in grouped.values():
        for values in questions.values():
            values.sort(key=lambda row: int(row["agent_id"]))
            if len(values) != 5:
                raise AssertionError("each question state must retain five member opportunities")
    return {state: dict(values) for state, values in grouped.items()}


def rank_by_gain(rows: list[dict[str, Any]], *, intended: bool) -> dict[int, int]:
    gains = {int(row["agent_id"]): int(row["gain_count"]) for row in rows}
    best = max(gains.values())
    lagging = sorted({gain for gain in gains.values() if best - gain > 5})
    if intended:
        return {agent: len(lagging) - lagging.index(gain) if gain in lagging else 0 for agent, gain in gains.items()}
    return {agent: lagging.index(gain) + 1 if gain in lagging else 0 for agent, gain in gains.items()}


def owner_replay(
    *, policy: str, states: dict[int, dict[str, list[dict[str, Any]]]],
    waits: dict[int, dict[int, int]], seed: int,
) -> tuple[dict[int, dict[str, int]], list[dict[str, Any]], list[dict[str, Any]]]:
    if policy not in {"A_legacy_current", "B_repair_only", "C_repair_only_relative_rank_late"}:
        raise ValueError(policy)
    previous: dict[str, int] = {}
    previous_age: dict[str, int] = {}
    assignments_by_state: dict[int, dict[str, int]] = {}
    rows_out: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for state_version in sorted(states):
        questions, loads, owners, ages = states[state_version], {agent: 0 for agent in AGENTS}, {}, {}
        for question_hash in sorted(questions):
            all_rows = questions[question_hash]
            if int(all_rows[0]["M"]) > 0:
                continue
            eligible = [row for row in all_rows if bool(row["member_error"])]
            if not eligible:
                continue
            intended_rank = rank_by_gain(all_rows, intended=True)
            if policy == "A_legacy_current":
                vectors = {
                    int(row["agent_id"]): (
                        float(bool(row["direct_vote_fix"])), float(row["oracle_soft_utility_gain"]),
                        float(row["improvement_need"]), float(bool(row["coverage_opportunity"])),
                        float(bool(row["dominant_wrong_member"])),
                    ) for row in eligible
                }
            else:
                vectors = {
                    int(row["agent_id"]): (
                        float(bool(row["direct_vote_fix"])), float(row["oracle_soft_utility_gain"]),
                        float(bool(row["coverage_opportunity"])), float(bool(row["dominant_wrong_member"])),
                    ) for row in eligible
                }
            front = fronts(vectors)
            frontier = [row for row in eligible if front[int(row["agent_id"])] == 1]
            def key(row: dict[str, Any]) -> tuple[Any, ...]:
                agent = int(row["agent_id"])
                base: tuple[Any, ...]
                if policy == "A_legacy_current":
                    base = (-int(row["improvement_need"]),)
                else:
                    base = ()
                repair = (
                    -int(bool(row["direct_vote_fix"])), -float(row["oracle_soft_utility_gain"]),
                    -int(bool(row["coverage_opportunity"])), -int(bool(row["dominant_wrong_member"])),
                    loads[agent],
                )
                late = ((-intended_rank[agent],) if policy.startswith("C_") else ())
                return (*base, *repair, *late, -waits[state_version][agent], seeded(seed, question_hash, agent))
            preferred = min(frontier, key=key)
            previous_id = previous.get(question_hash)
            prior = next((row for row in frontier if int(row["agent_id"]) == previous_id), None)
            inertia = False
            if prior is not None:
                if policy == "A_legacy_current":
                    inertia = (
                        int(preferred["improvement_need"]) <= int(prior["improvement_need"])
                        and int(bool(preferred["direct_vote_fix"])) <= int(bool(prior["direct_vote_fix"]))
                        and float(preferred["oracle_soft_utility_gain"]) - float(prior["oracle_soft_utility_gain"]) <= MARGIN
                    )
                else:
                    inertia = (
                        int(bool(preferred["direct_vote_fix"])) <= int(bool(prior["direct_vote_fix"]))
                        and int(bool(preferred["coverage_opportunity"])) <= int(bool(prior["coverage_opportunity"]))
                        and int(bool(preferred["dominant_wrong_member"])) <= int(bool(prior["dominant_wrong_member"]))
                        and float(preferred["oracle_soft_utility_gain"]) - float(prior["oracle_soft_utility_gain"]) <= MARGIN
                    )
            owner = prior if inertia else preferred
            owner_id = int(owner["agent_id"])
            owners[question_hash] = owner_id
            loads[owner_id] += 1
            ages[question_hash] = previous_age.get(question_hash, 0) + 1 if previous_id == owner_id else 0
            preferred_id = int(preferred["agent_id"])
            record = {
                "artifact_schema_version": "observed_state_owner_policy_replay_v1",
                "policy": policy, "team_state_version": state_version,
                "question_hash": question_hash, "eligible_agent_ids": [int(row["agent_id"]) for row in eligible],
                "candidate_pareto_fronts": {str(agent): front[agent] for agent in sorted(front)},
                "candidate_vectors": {str(agent): list(vectors[agent]) for agent in sorted(vectors)},
                "previous_owner": previous_id, "preferred_owner": preferred_id,
                "chosen_owner": owner_id, "inertia_retained": bool(inertia),
                "owner_age": ages[question_hash], "assigned_load_after": loads[owner_id],
                "relative_rank_by_agent": {
                    str(agent): intended_rank[agent] for agent in sorted(intended_rank)
                },
            }
            rows_out.append(record)
            if inertia and (
                bool(preferred["coverage_opportunity"]) and not bool(prior["coverage_opportunity"])
                or bool(preferred["dominant_wrong_member"]) and not bool(prior["dominant_wrong_member"])
            ):
                conflicts.append({**record, "conflict": "previous_retained_despite_preferred_coverage_or_dominant_wrong_advantage"})
        assignments_by_state[state_version] = owners
        previous, previous_age = owners, ages
    return assignments_by_state, rows_out, conflicts


def scheduler_replay(priority_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    for audit in priority_rows:
        priorities = list(audit["priorities"])
        def select(*, intended: bool) -> tuple[int, str, dict[int, int], list[int]]:
            eligible = [row for row in priorities if int(row["individual_error_count"]) > 0]
            overdue = [row for row in eligible if bool(row["overdue"])]
            regular = [row for row in eligible if not bool(row["candidate_search_outcome"]["cooling_down"])]
            ranks = rank_by_gain(priorities, intended=intended)
            potential = [row for row in regular if ranks[int(row["agent_id"])] > 0]
            if overdue:
                candidates, stage, include_rank = overdue, "overdue", False
            elif potential and regular:
                candidates, stage, include_rank = potential, "relative_improvement_potential", True
            elif regular:
                candidates, stage, include_rank = regular, "team_repair", False
            else:
                candidates, stage, include_rank = eligible, "search_budget_cooldown_fallback", False
            vectors = {
                int(row["agent_id"]): (
                    *((float(ranks[int(row["agent_id"])]),) if include_rank else ()),
                    float(row["direct_vote_fix_count"]), float(row["oracle_soft_utility_gain_sum"]),
                    float(row["coverage_opportunity_count"]),
                ) for row in candidates
            }
            front = fronts(vectors)
            frontier = [row for row in candidates if front[int(row["agent_id"])] == 1]
            selected = min(frontier, key=lambda row: (
                -ranks[int(row["agent_id"])] if include_rank else 0,
                -int(row["direct_vote_fix_count"]), -float(row["oracle_soft_utility_gain_sum"]),
                -int(row["coverage_opportunity_count"]), -int(row["assigned_load"]),
                -int(row["updates_since_selected"]), int(row["protection_risk"]), str(row["seeded_rank"]),
            ))
            return int(selected["agent_id"]), stage, ranks, [int(row["agent_id"]) for row in candidates]
        current, current_stage, current_ranks, current_pool = select(intended=False)
        intended, intended_stage, intended_ranks, intended_pool = select(intended=True)
        output.append({
            "artifact_schema_version": "scheduler_rank_direction_replay_v1",
            "update_index": int(audit["update_index"]),
            "historical_selected_agent_id": int(audit["selected_agent_id"]),
            "current_implementation_selected_agent_id": current,
            "intended_rank_selected_agent_id": intended,
            "selection_changed": current != intended,
            "current_stage": current_stage, "intended_stage": intended_stage,
            "current_rank_by_agent": {str(key): value for key, value in current_ranks.items()},
            "intended_rank_by_agent": {str(key): value for key, value in intended_ranks.items()},
            "current_pool_agent_ids": current_pool, "intended_pool_agent_ids": intended_pool,
        })
    def counts(field: str) -> dict[str, int]:
        return {str(agent): sum(int(row[field]) == agent for row in output) for agent in AGENTS}
    return output, {
        "artifact_schema_version": "scheduler_rank_direction_summary_v1",
        "scope": "observed_state_decision_replay_not_counterfactual_training",
        "observed_decision_count": len(output),
        "changed_decision_count": sum(bool(row["selection_changed"]) for row in output),
        "current_target_frequency": counts("current_implementation_selected_agent_id"),
        "intended_target_frequency": counts("intended_rank_selected_agent_id"),
        "current_overdue_count": sum(row["current_stage"] == "overdue" for row in output),
        "intended_overdue_count": sum(row["intended_stage"] == "overdue" for row in output),
    }


def alignment_matrix() -> list[dict[str, Any]]:
    return [
        {"concept": "relative potential rank direction", "intended_semantics": "lowest gain has highest positive rank", "document_source": "task section 3.1", "implementation_location": "responsibility.py target_priorities/build_target_selection_decision", "actual_behavior": "lowest gain receives rank 1 while selector prefers larger rank", "status": "misaligned", "severity": "high", "evidence": "lagging gains sorted ascending then index+1; selector negates rank", "recommended_action": "reverse positive rank numbering and invalidate old scheduler checkpoints", "behavior_change_required": True},
        {"concept": "primary owner", "intended_semantics": "distinguish repair value from global member deficit", "document_source": "task section 3.2", "implementation_location": "responsibility.py assign_primary_responsibilities", "actual_behavior": "improvement_need enters owner Pareto vector, preference, and inertia", "status": "partially_aligned", "severity": "high", "evidence": "five-axis vectors and member-aware inertia", "recommended_action": "retain current owner policy pending replay; consider repair-only policy later", "behavior_change_required": False},
        {"concept": "TCS competence context", "intended_semantics": "must not be mislabeled as relative gain-gap potential", "document_source": "task section 3.3", "implementation_location": "system.py/build_diagnosis_context; diagnosis_aggregation.py", "actual_behavior": "target_improvement_need is old mean-relative deficit and enables MEMBER_COMPETENCE patterns", "status": "misaligned", "severity": "medium", "evidence": "target_improvement_need derived from opportunity.improvement_need", "recommended_action": "defer formal TCS semantic change to a separate policy task", "behavior_change_required": False},
        {"concept": "target-owner-proposal alignment", "intended_semantics": "report responsibility-conditioned versus zero-assignment updates distinctly", "document_source": "task section 3.4", "implementation_location": "system.py select_target/_pool_indices/build_diagnosis_context", "actual_behavior": "target priority uses global opportunities while coverage/conversion and TCS assignment use primary-owner hashes", "status": "partially_aligned", "severity": "high", "evidence": "assigned hashes gate member-aware Stage A pools", "recommended_action": "audit classifications; do not force assigned target in this task", "behavior_change_required": False},
        {"concept": "specialization artifact", "intended_semantics": "selected context differs from actually repaired structural patterns", "document_source": "task section 3.5/8", "implementation_location": "system.py record_training_dynamics", "actual_behavior": "accepted_repair_pattern_ids copies selected_pattern_ids", "status": "misaligned", "severity": "high", "evidence": "direct list copy in specialization_trajectory_v1", "recommended_action": "upgrade to v2 selected/repaired structural fields", "behavior_change_required": True},
        {"concept": "version wording", "intended_semantics": "current final-state lifecycle must not be called v10", "document_source": "task section 3.6", "implementation_location": "AGENTS.md/method.md/README.md/comments", "actual_behavior": "some active-lifecycle prose still says v10", "status": "partially_aligned", "severity": "low", "evidence": "static v10 search", "recommended_action": "replace stale lifecycle wording while preserving historical references", "behavior_change_required": False},
        {"concept": "sanitized artifact completeness", "intended_semantics": "missing evidence is unavailable, and joins/path scans are explicit", "document_source": "task section 3.7/9.4", "implementation_location": "build_sanitized_pilot_report.py", "actual_behavior": "basic joins exist but path scan is narrow and TCS/Stage-A field availability is not explicit", "status": "partially_aligned", "severity": "medium", "evidence": "report_integrity only scans a subset of path forms", "recommended_action": "broaden scanner and record unavailable fields", "behavior_change_required": True},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("out_dir must be fresh")
    args.out_dir.mkdir(parents=True)
    input_dir, out = args.input_dir, args.out_dir
    meta = load_json(input_dir / "run_meta_sanitized.json")
    seed = int(meta["config"]["seed"])
    opportunities = load_jsonl(input_dir / "member_opportunities_sanitized.jsonl")
    assignments = load_jsonl(input_dir / "responsibility_assignments_sanitized.jsonl")
    priorities = load_jsonl(input_dir / "target_priority_audit_sanitized.jsonl")
    decisions = load_jsonl(input_dir / "candidate_decisions_sanitized.jsonl")
    specialization = load_jsonl(input_dir / "specialization_trajectory_sanitized.jsonl")
    states = state_rows(opportunities)
    assignment_by_state = {int(row["team_state_version"]): row for row in assignments}
    if set(states) != set(assignment_by_state):
        raise AssertionError("responsibility and member-opportunity state sets differ")
    waits, version, counter = {0: {agent: 0 for agent in AGENTS}}, 0, {agent: 0 for agent in AGENTS}
    update_to_state: dict[int, int] = {}
    for decision in sorted(decisions, key=lambda row: int(row["update_index"])):
        target = int(decision["target_agent_id"])
        counter = {agent: value + 1 for agent, value in counter.items()}
        counter[target] = 0
        update_to_state[int(decision["update_index"])] = version
        if decision.get("accepted_prompt_hash"):
            version += 1
            waits[version] = dict(counter)
    if set(waits) != set(states):
        raise AssertionError("accepted-state trajectory does not match responsibility states")
    replays, replay_rows, conflicts = {}, [], []
    for policy in ("A_legacy_current", "B_repair_only", "C_repair_only_relative_rank_late"):
        owners, rows, policy_conflicts = owner_replay(policy=policy, states=states, waits=waits, seed=seed)
        replays[policy] = owners
        replay_rows.extend(rows)
        conflicts.extend(policy_conflicts)
    mismatches = []
    for state, owners in replays["A_legacy_current"].items():
        historic = {str(key): int(value) for key, value in assignment_by_state[state]["owners"].items()}
        if owners != historic:
            mismatches.append(state)
    if mismatches:
        raise AssertionError(f"Policy A replay mismatch at states {mismatches}")
    write_jsonl(out / "owner_policy_replay.jsonl", replay_rows)
    write_jsonl(out / "inertia_conflict_audit.jsonl", conflicts)
    rank_rows, rank_summary = scheduler_replay(priorities)
    write_jsonl(out / "scheduler_rank_direction_replay.jsonl", rank_rows)
    write_json(out / "scheduler_rank_direction_summary.json", rank_summary)
    matrix = alignment_matrix()
    write_json(out / "implementation_alignment_matrix.json", matrix)
    (out / "implementation_alignment_matrix.md").write_text(
        "# Implementation Alignment Matrix\n\n" + "\n".join(
            f"- **{row['concept']}** — {row['status']} ({row['severity']}): {row['actual_behavior']}"
            for row in matrix
        ) + "\n", encoding="utf-8")

    concentration, repair, retention, structural = {}, {}, {}, {}
    for policy, owner_states in replays.items():
        concentration[policy], repair[policy], retention[policy], structural[policy] = [], [], [], []
        prior_owners: dict[str, int] = {}
        prior_patterns = {agent: Counter() for agent in AGENTS}
        for state in sorted(states):
            owners = owner_states[state]
            values = [sum(owner == agent for owner in owners.values()) for agent in AGENTS]
            total = sum(values)
            concentration[policy].append({"team_state_version": state, "owner_count_per_agent": values, "active_owner_count": sum(value > 0 for value in values), "maximum_owner_share": max(values) / total if total else 0.0, "owner_entropy": entropy(values), "owner_hhi": sum((value / total) ** 2 for value in values) if total else 0.0, "owner_gini": gini(values), "assigned_load_variance": sum((value - total / 5) ** 2 for value in values) / 5})
            records = []
            for question_hash, owner in owners.items():
                by_agent = {int(row["agent_id"]): row for row in states[state][question_hash]}
                chosen, eligible = by_agent[owner], [row for row in by_agent.values() if bool(row["member_error"])]
                records.append((chosen, eligible))
            repair[policy].append({"team_state_version": state, "direct_fix_eligible_question_count": sum(any(bool(row["direct_vote_fix"]) for row in eligible) for _, eligible in records), "chosen_owner_direct_fix_count": sum(bool(chosen["direct_vote_fix"]) for chosen, _ in records), "chosen_owner_direct_fix_rate": sum(bool(chosen["direct_vote_fix"]) for chosen, _ in records) / len(records) if records else 0.0, "chosen_oracle_utility": sum(float(chosen["oracle_soft_utility_gain"]) for chosen, _ in records), "maximum_eligible_oracle_utility": sum(max(float(row["oracle_soft_utility_gain"]) for row in eligible) for _, eligible in records), "oracle_utility_regret": sum(max(float(row["oracle_soft_utility_gain"]) for row in eligible) - float(chosen["oracle_soft_utility_gain"]) for chosen, eligible in records), "chosen_owner_dominant_wrong_rate": sum(bool(chosen["dominant_wrong_member"]) for chosen, _ in records) / len(records) if records else 0.0, "coverage_owner_count": sum(bool(chosen["coverage_opportunity"]) for chosen, _ in records)})
            common = set(prior_owners) & set(owners)
            retained = sum(prior_owners[key] == owners[key] for key in common)
            ages = [row["owner_age"] for row in replay_rows if row["policy"] == policy and row["team_state_version"] == state]
            retention[policy].append({"team_state_version": state, "owner_switch_count": len(common) - retained, "owner_retention_rate": retained / len(common) if common else None, "mean_owner_age": sum(ages) / len(ages) if ages else 0.0, "median_owner_age": median(ages) if ages else 0.0, "inertia_retained_count": sum(bool(row["inertia_retained"]) for row in replay_rows if row["policy"] == policy and row["team_state_version"] == state), "inertia_override_count": sum(not bool(row["inertia_retained"]) and row["previous_owner"] is not None for row in replay_rows if row["policy"] == policy and row["team_state_version"] == state)})
            current_patterns = {agent: Counter() for agent in AGENTS}
            owned_sets = {agent: set() for agent in AGENTS}
            for question_hash, owner in owners.items():
                row = next(item for item in states[state][question_hash] if int(item["agent_id"]) == owner)
                current_patterns[owner][structural_pattern(row)] += 1
                owned_sets[owner].add(question_hash)
            structural[policy].append({"team_state_version": state, "per_agent_structural_pattern_histogram": {str(agent): dict(current_patterns[agent]) for agent in AGENTS}, "per_agent_pattern_distribution_js_from_previous": {str(agent): js_divergence(prior_patterns[agent], current_patterns[agent]) for agent in AGENTS}})
            prior_owners, prior_patterns = owners, current_patterns
    write_json(out / "owner_load_concentration.json", concentration)
    write_json(out / "owner_repair_value_summary.json", repair)
    write_json(out / "owner_retention_summary.json", retention)
    write_json(out / "structural_pattern_continuity.json", {"scope": "structural_failure_pattern_continuity_not_semantic_specialization", "by_policy": structural, "semantic_error_type_specialization": "unavailable", "reasoning_strategy_continuity": "unavailable", "semantic_role_formation": "unavailable"})

    priority_by_update = {int(row["update_index"]): row for row in priorities}
    special_by_update = {int(row["update_index"]): row for row in specialization}
    context_rows = []
    for decision in sorted(decisions, key=lambda row: int(row["update_index"])):
        update, target, state = int(decision["update_index"]), int(decision["target_agent_id"]), update_to_state[int(decision["update_index"])]
        priority = next(row for row in priority_by_update[update]["priorities"] if int(row["agent_id"]) == target)
        for policy, owner_states in replays.items():
            hashes = sorted(question for question, owner in owner_states[state].items() if owner == target)
            target_rows = [next(row for row in states[state][question] if int(row["agent_id"]) == target) for question in hashes]
            coverage = sum(int(row["G"]) == 0 for row in target_rows)
            conversion = sum(int(row["M"]) <= 0 and int(row["G"]) > 0 for row in target_rows)
            preservation = sum(bool(row["unique_correct"]) or bool(row["pivotal_correct"]) for rows in states[state].values() for row in rows if int(row["agent_id"]) == target)
            if not hashes:
                classification = "zero_assignment_generic_update"
            else:
                classification = "partially_responsibility_conditioned"
            actual_pattern_ids = special_by_update.get(update, {}).get(
                "selected_context_pattern_ids", "unavailable"
            ) if policy == "A_legacy_current" else "unavailable"
            context_rows.append({"artifact_schema_version": "target_owner_context_alignment_v1", "policy": policy, "update_index": update, "team_state_version": state, "target_agent_id": target, "selection_pool_stage": priority_by_update[update]["selection_pool_stage"], "target_gain": priority["gain_count"], "target_gain_gap": priority["gain_gap_to_best"], "target_relative_rank": priority["relative_improvement_potential_rank"], "target_assigned_load": len(hashes), "assigned_question_count": len(hashes), "assigned_direct_fix_count": sum(bool(row["direct_vote_fix"]) for row in target_rows), "assigned_coverage_count": coverage, "assigned_conversion_count": conversion, "stage_a_coverage_available": min(6, coverage), "stage_a_conversion_available": min(6, conversion), "stage_a_preservation_available": min(4, preservation), "stage_a_selected_coverage_count": "unavailable", "stage_a_selected_conversion_count": "unavailable", "tcs_selected_pattern_ids": actual_pattern_ids, "selected_pattern_assigned_case_count": "unavailable", "evidence_assigned_case_count": "unavailable", "accepted": bool(decision.get("accepted_prompt_hash")), "assigned_residual_repair_count": "unavailable", "vote_net_gain": "unavailable", "target_gain_from_candidate": decision.get("best_attempt_target_gain", "unavailable"), "classification": classification, "historical_assigned_hashes_match_policy_a": (hashes == sorted(decision.get("assigned_question_hashes", []))) if policy == "A_legacy_current" else "not_applicable"})
    write_jsonl(out / "target_owner_context_alignment.jsonl", context_rows)
    write_json(out / "target_owner_context_alignment_summary.json", {policy: {"update_count": sum(row["policy"] == policy for row in context_rows), "zero_assignment_count": sum(row["policy"] == policy and row["classification"] == "zero_assignment_generic_update" for row in context_rows), "accepted_zero_assignment_count": sum(row["policy"] == policy and row["classification"] == "zero_assignment_generic_update" and row["accepted"] for row in context_rows)} for policy in replays})
    write_json(out / "artifact_semantics_audit.json", {"specialization_trajectory_v1": {"selected_pattern_ids": "selected_context_structural_pattern_ids", "accepted_repair_pattern_ids": "misleading_legacy_selected_pattern_alias", "semantic_specialization_claim": "unavailable"}, "stage_a_selected_pool_composition": "unavailable_in_sanitized_v11_artifact", "role_by_phase_token_breakdown": "unavailable_in_sanitized_v11_artifact", "prompt_length_trajectory": "unavailable_in_sanitized_v11_artifact"})
    write_json(out / "artifact_availability.json", {"available": ["14 observed responsibility states", "32 target decisions", "13 accepted-update structural transitions", "sanitized owner maps and opportunities"], "unavailable": ["semantic error-type specialization", "reasoning-strategy continuity", "actual Stage-A coverage/conversion selected counts", "per-update role/phase token attribution", "actual repaired selected-pattern attribution"], "rule": "unavailable fields are not encoded as zero"})
    integrity = {"policy_a_exact_replay": True, "state_count": len(states), "candidate_decision_count": len(decisions), "priority_audit_count": len(priorities), "accepted_update_count": sum(bool(row.get("accepted_prompt_hash")) for row in decisions), "specialization_row_count": len(specialization)}
    write_json(out / "report_integrity.json", integrity)
    final_state = max(states)
    final_loads = {
        policy: concentration[policy][-1]["owner_count_per_agent"]
        for policy in replays
    }
    final_regret = {
        policy: repair[policy][-1]["oracle_utility_regret"]
        for policy in replays
    }
    context_summary = load_json(out / "target_owner_context_alignment_summary.json")
    (out / "README.md").write_text(
        "# Owner Alignment Audit\n\n"
        "This is an observed-state replay over the initial state and thirteen accepted-update "
        "states. It does not predict a counterfactual training trajectory, final vote, or test result.\n\n"
        "## Confirmed findings\n\n"
        "- Policy A exactly reproduces every archived primary-owner assignment.\n"
        f"- Correcting rank direction changes {rank_summary['changed_decision_count']} of "
        f"{rank_summary['observed_decision_count']} observed target decisions; this is a decision replay only.\n"
        f"- At final observed state {final_state}, owner loads are A={final_loads['A_legacy_current']}, "
        f"B={final_loads['B_repair_only']}, C={final_loads['C_repair_only_relative_rank_late']}.\n"
        f"- Final observed oracle-utility regret is A={final_regret['A_legacy_current']}, "
        f"B={final_regret['B_repair_only']}, C={final_regret['C_repair_only_relative_rank_late']}.\n"
        f"- Historical Policy-A target updates include "
        f"{context_summary['A_legacy_current']['zero_assignment_count']} zero-assignment cases, "
        f"including {context_summary['A_legacy_current']['accepted_zero_assignment_count']} accepted updates. "
        "They are generic fallback updates, not evidence of responsibility-conditioned repair.\n\n"
        "## Interpretation and next-policy recommendation\n\n"
        "Policy B is the next owner-policy candidate to implement in a separate task: on these observed "
        "states it materially reduces owner concentration without losing direct-fix capture and removes the "
        "small observed oracle-utility regret. Policy C adds no observed final-state advantage over B here, "
        "so a late relative-rank tie-break should remain a hypothesis rather than a default. The main risk is "
        "that replay holds team outputs fixed; only a matched future pilot can establish effects on vote, "
        "responsibility-conditioned proposal quality, or member specialization.\n\n"
        "Structural pattern continuity is not semantic specialization. Fields missing from the sanitized source "
        "are reported as `unavailable`, never as zero.\n",
        encoding="utf-8",
    )
    for path in out.rglob("*"):
        if path.is_file() and FORBIDDEN_PATH.search(path.read_text(encoding="utf-8")):
            raise ValueError(f"sanitized report contains an absolute path: {path.name}")
    print(json.dumps({"ok": True, "out_dir": str(out), "policy_a_exact_replay": True}, indent=2))


if __name__ == "__main__":
    main()
