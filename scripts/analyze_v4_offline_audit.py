"""Offline-only, sanitized audit for an already completed v4 pilot.

It deliberately performs observed-state scheduler replay, not a counterfactual
training rollout.  Missing historical instrumentation is represented as
``unavailable`` rather than zero.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def relative_rows(priorities: list[dict[str, Any]], *, tolerance: int = 5) -> list[dict[str, Any]]:
    best_gain = max(int(row["gain_count"]) for row in priorities)
    lagging = sorted({int(row["gain_count"]) for row in priorities if best_gain - int(row["gain_count"]) > tolerance})
    ranks = {gain: index + 1 for index, gain in enumerate(lagging)}
    rows = []
    for source in priorities:
        gain = int(source["gain_count"])
        gap = best_gain - gain
        row = {
            "agent_id": int(source["agent_id"]),
            "gain_count": gain,
            "best_team_gain_count": best_gain,
            "gain_gap_to_best": gap,
            "relative_gain_tolerance_count": tolerance,
            "within_relative_gain_band": gap <= tolerance,
            "has_relative_improvement_potential": gap > tolerance,
            "relative_improvement_potential_rank": ranks.get(gain, 0),
            "assigned_load": int(source["assigned_load"]),
            "direct_vote_fix_count": int(source["direct_vote_fix_count"]),
            "oracle_soft_utility_gain_sum": float(source["oracle_soft_utility_gain_sum"]),
            "coverage_opportunity_count": int(source["coverage_opportunity_count"]),
            "updates_since_selected": int(source["updates_since_selected"]),
            "overdue": bool(source["overdue"]),
            "cooling_down": bool(source["cooling_down"]),
            "candidate_search_outcome": {
                "best_observed_target_gain": int(source["best_observed_target_gain"]),
                "no_positive_candidate_streak": int(source["no_positive_candidate_streak"]),
                "cooldown_until_update": int(source["next_regular_eligible_update"]),
            },
            "seeded_rank": str(source["seeded_rank"]),
            "individual_error_count": int(source["individual_error_count"]),
        }
        rows.append(row)
    return rows


def choose(rows: list[dict[str, Any]], *, max_wait: int) -> tuple[int, str, list[int]]:
    eligible = [row for row in rows if row["individual_error_count"] > 0]
    overdue = [row for row in eligible if row["updates_since_selected"] >= max_wait]
    regular = [row for row in eligible if not row["cooling_down"]]
    potential = [row for row in regular if row["has_relative_improvement_potential"]]
    if overdue:
        pool, stage, use_rank = overdue, "overdue", False
    elif potential:
        pool, stage, use_rank = potential, "relative_improvement_potential", True
    elif regular:
        pool, stage, use_rank = regular, "team_repair", False
    else:
        pool, stage, use_rank = eligible, "search_budget_cooldown_fallback", False
    selected = min(pool, key=lambda row: (
        -row["relative_improvement_potential_rank"] if use_rank else 0,
        -row["direct_vote_fix_count"],
        -row["oracle_soft_utility_gain_sum"],
        -row["coverage_opportunity_count"],
        -row["assigned_load"],
        -row["updates_since_selected"],
        row["seeded_rank"],
    ))
    return int(selected["agent_id"]), stage, [int(row["agent_id"]) for row in pool]


def candidate_protection_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    # Historical candidate records have full post-G vectors but not pre-G vectors
    # at this level, so exact local G-transition preference is unavailable.
    return (
        int(candidate.get("unique_correct_loss_count", 0)),
        -int(candidate.get("unique_correct_gain_count", 0)),
        int(candidate.get("pivotal_correct_loss_count", 0)),
        -int(candidate.get("pivotal_correct_gain_count", 0)),
        -int(candidate.get("vote_correct_candidate", -10**9)),
        str(candidate.get("prompt_hash", "")),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    decisions = load_jsonl(run / "candidate_decisions.jsonl")
    priority_audit = load_jsonl(run / "target_priority_audit.jsonl")
    assignments = load_jsonl(run / "responsibility_assignments.jsonl")
    states = load_jsonl(run / "peer_state_history.jsonl")
    calls = load_jsonl(run / "llm_calls.jsonl")
    contexts = load_jsonl(run / "tcs_context_history.jsonl")
    assert len(decisions) == len(priority_audit)

    replay_rows: list[dict[str, Any]] = []
    for decision, audit in zip(decisions, priority_audit, strict=True):
        rows = relative_rows(audit["priorities"])
        old = int(decision["target_agent_id"])
        for name, wait in (("historical_max_wait_4", 4), ("relative_gain_max_wait_4", 4), ("relative_gain_max_wait_8", 8)):
            if name == "historical_max_wait_4":
                selected, stage, pool = old, "historical_observed", []
            else:
                selected, stage, pool = choose(rows, max_wait=wait)
            replay_rows.append({
                "artifact_schema_version": "observed_state_scheduler_replay_v1",
                "update_index": int(decision["update_index"]),
                "scheme": name,
                "historical_selected_agent_id": old,
                "replayed_selected_agent_id": selected,
                "selection_changed": selected != old,
                "selection_pool_stage": stage,
                "selection_pool_agent_ids": pool,
                "agents": rows,
            })
    dump_jsonl(out / "observed_state_scheduler_replay.jsonl", replay_rows)
    summary: dict[str, Any] = {"artifact_schema_version": "scheduler_replay_summary_v1", "scope": "observed_state_scheduler_decision_replay_not_counterfactual_training"}
    for scheme in {row["scheme"] for row in replay_rows}:
        rows = [row for row in replay_rows if row["scheme"] == scheme]
        selected = [row["replayed_selected_agent_id"] for row in rows]
        summary[scheme] = {
            "target_frequency": {str(agent): selected.count(agent) for agent in range(5)},
            "overdue_selected_count": sum(row["selection_pool_stage"] == "overdue" for row in rows),
            "relative_potential_selected_count": sum(row["selection_pool_stage"] == "relative_improvement_potential" for row in rows),
            "team_repair_selected_count": sum(row["selection_pool_stage"] == "team_repair" for row in rows),
            "changed_decision_count": sum(row["selection_changed"] for row in rows),
        }
    dump(out / "scheduler_replay_summary.json", summary)

    accepted = [row for row in decisions if row.get("accepted_prompt_hash")]
    state_blocks = [states[index:index + 75] for index in range(0, len(states), 75)]
    g_rows: list[dict[str, Any]] = []
    accepted_state_index = 0
    for decision in decisions:
        if not decision.get("accepted_prompt_hash"):
            continue
        before, after = state_blocks[accepted_state_index], state_blocks[accepted_state_index + 1]
        accepted_state_index += 1
        target = int(decision["target_agent_id"])
        for old, new in zip(before, after, strict=True):
            old_correct, new_correct = old["team_correctness"][target], new["team_correctness"][target]
            g_rows.append({
                "artifact_schema_version": "reconstructed_g_transition_audit_v1",
                "update_index": int(decision["update_index"]), "target_agent_id": target,
                "question_hash": old["question_hash"],
                "G_before": old["gold_vote_count"], "G_after": new["gold_vote_count"],
                "H_before": old["largest_wrong_vote_count"], "H_after": new["largest_wrong_vote_count"],
                "M_before": old["plurality_margin"], "M_after": new["plurality_margin"],
                "vote_correct_before": old["vote_correct"], "vote_correct_after": new["vote_correct"],
                "target_correct_before": old_correct, "target_correct_after": new_correct,
                "target_valid_before": old["team_validity"][target], "target_valid_after": new["team_validity"][target],
                "transition_class": f"G={old['gold_vote_count']}->G={new['gold_vote_count']}",
            })
    dump_jsonl(out / "g_transition_audit_sanitized.jsonl", g_rows)
    transitions = Counter(row["transition_class"] for row in g_rows)
    dump(out / "protection_transition_summary.json", {
        "artifact_schema_version": "protection_transition_summary_v1",
        "accepted_update_count": len(accepted), "row_count": len(g_rows),
        "G_transition_counts": dict(sorted(transitions.items())),
        "available": ["G/H/M", "vote flips", "target correctness"],
        "unavailable": ["per-example unique/pivotal retention", "new unique coverage lifetime"],
        "reason_unavailable": "the historical peer-state artifact lacks per-target leave-one-out protection flags",
    })

    specialization = []
    for decision in accepted:
        context = next((row for row in contexts if row["update_index"] == decision["update_index"]), {})
        rows = [row for row in g_rows if row["update_index"] == decision["update_index"]]
        specialization.append({
            "artifact_schema_version": "historical_specialization_stability_v1",
            "update_index": decision["update_index"], "agent_id": decision["target_agent_id"],
            "selected_pattern_ids": list(context.get("selected_pattern_ids", [])),
            "correct_set_gain_count": sum(not row["target_correct_before"] and row["target_correct_after"] for row in rows),
            "correct_set_loss_count": sum(row["target_correct_before"] and not row["target_correct_after"] for row in rows),
            "responsibility_overlap": "unavailable",
            "owner_retention": "unavailable",
        })
    dump_jsonl(out / "specialization_stability_sanitized.jsonl", specialization)

    roles = {}
    total_tokens = sum(int(row["total_tokens"]) for row in calls)
    for role in sorted({row["role"] for row in calls}):
        group = [row for row in calls if row["role"] == role]
        role_tokens = sum(int(row["total_tokens"]) for row in group)
        roles[role] = {"call_count": len(group), "prompt_tokens": sum(int(row["prompt_tokens"]) for row in group), "completion_tokens": sum(int(row["completion_tokens"]) for row in group), "total_tokens": role_tokens, "token_share": role_tokens / total_tokens if total_tokens else 0.0}
    dump(out / "token_cost_breakdown_sanitized.json", {
        "artifact_schema_version": "token_cost_breakdown_v1", "total_tokens": total_tokens,
        "by_role": roles,
        "by_phase": "unavailable",
        "reason_phase_unavailable": "historical llm_calls.jsonl has no phase identifier",
        "stage_b_budget_replay": "see stage_b_budget_replay.json",
    })

    stage_b = []
    for decision in decisions:
        candidates = [row for row in decision.get("candidates", []) if row.get("hard_feasible")]
        historical = str(decision.get("accepted_prompt_hash", ""))
        preferred = min(candidates, key=candidate_protection_key)["prompt_hash"] if candidates else None
        stage_b.append({"update_index": decision["update_index"], "historical_accepted_prompt_hash": historical, "offline_protection_preferred_prompt_hash": preferred, "same_as_historical": preferred == historical if historical else None, "completed_stage_b_candidate_count": len(candidates)})
    dump_jsonl(out / "stage_b_preference_replay.jsonl", stage_b)

    budget_rows = []
    for decision in decisions:
        ranked = sorted(
            [row for row in decision.get("candidates", []) if row.get("stage_a_decision")],
            key=lambda row: (
                int((row["stage_a_decision"] or {}).get("aggregate_rank", 10**9)),
                int(row.get("generation", 10**9)), str(row.get("prompt_hash", "")),
            ),
        )
        top = ranked[0] if ranked else None
        second = ranked[1] if len(ranked) > 1 else None
        top_passes = bool(top and top.get("passed"))
        fallback_passes = bool(second and second.get("passed"))
        historical = str(decision.get("accepted_prompt_hash", ""))
        plan_a = str(top.get("prompt_hash", "")) if top_passes else ""
        plan_b = plan_a or (str(second.get("prompt_hash", "")) if fallback_passes else "")
        budget_rows.append({
            "update_index": decision["update_index"],
            "completed_stage_b_candidate_count": len(ranked),
            "stage_a_top_prompt_hash": top.get("prompt_hash") if top else None,
            "stage_a_top_acceptable": top_passes if top else None,
            "fallback_second_evaluated": bool(top and not top_passes and second),
            "historical_accepted_prompt_hash": historical,
            "plan_a_selected_prompt_hash": plan_a,
            "plan_b_selected_prompt_hash": plan_b,
            "plan_a_matches_historical": plan_a == historical if historical else None,
            "plan_b_matches_historical": plan_b == historical if historical else None,
        })
    accepted_rows = [row for row in budget_rows if row["historical_accepted_prompt_hash"]]
    dump(out / "stage_b_budget_replay.json", {
        "artifact_schema_version": "stage_b_budget_replay_v1",
        "scope": "single-update local replay; no downstream trajectory claim",
        "rows": budget_rows,
        "plan_a_top_rank_historical_acceptance_rate": sum(bool(row["plan_a_matches_historical"]) for row in accepted_rows) / max(1, len(accepted_rows)),
        "plan_b_historical_acceptance_retention_rate": sum(bool(row["plan_b_matches_historical"]) for row in accepted_rows) / max(1, len(accepted_rows)),
        "solver_call_saving": "unavailable",
        "reason_solver_call_saving_unavailable": "historical llm_calls has no candidate/phase join key",
    })

    dump(out / "artifact_availability.json", {
        "artifact_schema_version": "historical_audit_availability_v1",
        "available": ["observed scheduler states", "G/H/M accepted transitions", "role token totals", "selected anonymized pattern IDs"],
        "unavailable": ["historical per-example unique/pivotal flags", "role-phase token attribution", "candidate prompt length trajectory", "exact Stage-B incremental cost", "full owner/pattern Jaccard continuity"],
    })


if __name__ == "__main__":
    main()
