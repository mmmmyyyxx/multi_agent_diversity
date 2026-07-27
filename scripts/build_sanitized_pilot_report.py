"""Build a version-controlled, secret-free report from one completed raw pilot.

The raw run remains local and ignored.  This tool deliberately copies only
hashes, method labels, counts, booleans, and numeric matrices into a fresh
report directory; it never copies prompts, questions, answers, role text,
responses, paths, caches, or checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


FORBIDDEN_KEYS = {
    "prompt", "question", "gold", "answer", "answers", "raw", "response",
    "content", "text", "endpoint", "api_key", "cache", "checkpoint", "path",
}
SAFE_HASH_KEYS = {"question_hash", "prompt_hash", "repair_plan_hash", "probe_hash"}
FORBIDDEN_TEXT = ("http://", "https://", "final_answer:", "openai_api_key", "d:\\\\")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def allowed_key(name: str) -> bool:
    lowered = name.lower()
    if name in SAFE_HASH_KEYS or lowered.endswith("_hash") or lowered.endswith("_hashes"):
        return True
    return not any(token in lowered for token in FORBIDDEN_KEYS)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items() if allowed_key(str(key))}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def pick(row: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: sanitize(row[key]) for key in keys if key in row}


def numeric_map(value: Any) -> Any:
    """Keep a nested diagnostic only when it contains no free-form text."""
    if isinstance(value, dict):
        return {str(key): numeric_map(item) for key, item in value.items()}
    if isinstance(value, list):
        return [numeric_map(item) for item in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return "unavailable"


def candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_keys = (
        "prompt_hash", "generation", "repair_plan_hash", "stage_a_decision",
        "passed", "hard_feasible", "target_correct_incumbent",
        "target_correct_candidate", "target_gain", "vote_correct_incumbent",
        "vote_correct_candidate", "vote_gain_count", "vote_loss_count",
        "vote_net_gain", "unique_correct_gain_count", "unique_correct_loss_count",
        "pivotal_correct_gain_count", "pivotal_correct_loss_count",
        "incumbent_objective", "candidate_objective", "pareto_dominates_incumbent",
        "target_improvement_passed", "team_vote_nonregression_passed",
        "member_objective_nonregression_passed",
        "terminal_invalid_nonregression_passed", "rejection_reasons",
    )
    decision_keys = (
        "update_index", "target_agent_id", "assigned_question_hashes",
        "max_wait_fairness_trigger_count", "best_attempt_target_gain",
        "positive_target_gain_candidate_found", "candidate_search_outcome_updated",
        "cooldown_length_assigned", "accepted_prompt_hash",
    )
    output: list[dict[str, Any]] = []
    for row in rows:
        result = pick(row, decision_keys)
        result["artifact_schema_version"] = "sanitized_candidate_decision_v1"
        result["funnel"] = numeric_map(row.get("funnel", {}))
        result["incumbent"] = pick(
            row.get("incumbent", {}),
            ("prompt_hash", "competence", "team_outcome", "marginal", "protection", "member_gain"),
        )
        result["candidates"] = [pick(candidate, candidate_keys) for candidate in row.get("candidates", [])]
        output.append(result)
    return output


def responsibility_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "team_state_version", "member_gain_counts", "minimum_member_gain_count",
        "total_member_gain_count", "improvement_need_by_agent",
        "protection_counts_by_agent", "owner_distribution", "owners",
        "owner_switch_count", "owner_age", "assigned_load_by_agent",
        "direct_fix_responsibility_count", "coverage_responsibility_count",
        "dominant_wrong_responsibility_count", "owner_candidate_pareto_fronts",
        "owner_chosen_reasons", "owner_assignment_audit", "assigned_opportunities",
    )
    return [
        {"artifact_schema_version": "sanitized_responsibility_assignment_v1", **pick(row, keys)}
        for row in rows
    ]


def priority_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "update_index", "overdue_first", "selection_pool_stage", "eligible_agent_ids",
        "overdue_agent_ids", "non_cooling_agent_ids", "relative_potential_agent_ids",
        "actual_candidate_agent_ids", "actual_candidate_pareto_fronts",
        "actual_frontier_agent_ids", "search_budget_cooldown_fallback",
        "relative_potential_pool_used", "selected_agent_id",
    )
    return [
        {"artifact_schema_version": "sanitized_target_priority_audit_v1", **pick(row, keys),
         "priorities": sanitize(row.get("priorities", []))}
        for row in rows
    ]


def meta_row(meta: dict[str, Any]) -> dict[str, Any]:
    identity = meta.get("run_identity", {})
    identity_safe = {
        key: value for key, value in identity.items()
        if "sha256" in key.lower() or key.lower().endswith("_count") or key in {"task_id", "seed"}
    }
    config = meta.get("config", {})
    config_safe = {
        key: value for key, value in config.items()
        if allowed_key(key) and key not in {"shared_prompt", "provided_prompts_json"}
    }
    keys = (
        "method_version", "experiment_protocol", "initialization_mode", "tie_policy",
        "update_mode", "candidate_selector", "candidate_generator",
        "member_objective_version", "responsibility_version",
        "responsibility_lifecycle_version", "target_selection_version",
        "pareto_preference_version", "stage_a_version", "stage_b_version",
        "candidate_acceptance_version", "preservation_policy_version",
        "evaluation_protocol_version", "checkpoint_selection_version",
        "test_isolation_version", "tcs_context_version", "diagnosis_aggregation_version",
        "checkpoint_version", "solver_output_contract_version",
        "solver_request_template_version", "validation_used",
        "validation_unique_state_count", "validation_evaluation_count",
        "validation_reuse_count", "planned_update_count", "completed_update_count",
        "training_completed", "test_evaluation_count", "test_used_for_selection",
        "test_used_for_training", "test_called_before_training_complete",
        "initial_prompt_hashes", "initial_prompts_identical", "probe_hash",
    )
    return {
        "artifact_schema_version": "sanitized_run_meta_v1",
        **pick(meta, keys),
        "run_identity": identity_safe,
        "final_state_selection": sanitize(meta.get("final_state_selection", {})),
        "config": config_safe,
    }


def verify_report(out: Path, *, candidates: list[dict[str, Any]], priorities: list[dict[str, Any]],
                  opportunities: list[dict[str, Any]], transitions: list[dict[str, Any]],
                  specialization: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_updates = {
        int(row["update_index"]) for row in candidates if row.get("accepted_prompt_hash")
    }
    transition_updates = {int(row["update_index"]) for row in transitions}
    specialization_updates = {int(row["update_index"]) for row in specialization}
    opportunity_by_state = Counter(int(row["team_state_version"]) for row in opportunities)
    transition_by_update = Counter(int(row["update_index"]) for row in transitions)
    question_hashes = {str(row["question_hash"]) for row in opportunities}
    checks = {
        "candidate_priority_update_join": {
            "passed": {int(row["update_index"]) for row in candidates} == {int(row["update_index"]) for row in priorities},
            "candidate_count": len(candidates), "priority_count": len(priorities),
        },
        "accepted_transition_join": {
            "passed": accepted_updates == transition_updates == specialization_updates,
            "accepted_update_count": len(accepted_updates),
            "transition_update_count": len(transition_updates),
            "specialization_update_count": len(specialization_updates),
        },
        "expected_rows_per_state": {
            "passed": all(count == len(question_hashes) * 5 for count in opportunity_by_state.values())
            and all(count == len(question_hashes) for count in transition_by_update.values()),
            "question_hash_unique_count": len(question_hashes),
            "member_opportunity_rows_by_state": dict(sorted(opportunity_by_state.items())),
            "g_transition_rows_by_update": dict(sorted(transition_by_update.items())),
        },
    }
    for path in out.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8").lower()
            if any(token in text for token in FORBIDDEN_TEXT):
                raise ValueError(f"sanitized scan failed: {path}")
    if not all(value["passed"] for value in checks.values()):
        raise ValueError("sanitized report fact assertions failed")
    return {"artifact_schema_version": "sanitized_report_integrity_v1", "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run_dir, args.out_dir
    if out.exists():
        raise FileExistsError(f"out_dir must be fresh: {out}")
    out.mkdir(parents=True)

    candidates = load_jsonl(run / "candidate_decisions.jsonl")
    priorities = load_jsonl(run / "target_priority_audit.jsonl")
    assignments = load_jsonl(run / "responsibility_assignments.jsonl")
    opportunities = load_jsonl(run / "member_opportunities.jsonl")
    transitions = load_jsonl(run / "g_transition_audit.jsonl")
    specialization = load_jsonl(run / "specialization_trajectory.jsonl")
    dynamics = load_jsonl(run / "training_dynamics.jsonl")
    trajectory = load_jsonl(run / "team_differentiation_trajectory.jsonl")
    meta = meta_row(load_json(run / "run_meta.json"))
    cost = load_json(run / "cost_summary.json")
    recovery = load_json(run / "solver_recovery_summary.json")
    final_behavior = load_json(run / "final_test_differentiation.json")

    sanitized_candidates = candidate_rows(candidates)
    sanitized_priorities = priority_rows(priorities)
    sanitized_assignments = responsibility_rows(assignments)
    sanitized_opportunities = [
        {"artifact_schema_version": "sanitized_member_opportunity_v1", **sanitize(row)}
        for row in opportunities
    ]
    sanitized_transitions = [sanitize(row) for row in transitions]
    sanitized_specialization = [sanitize(row) for row in specialization]
    sanitized_dynamics = [sanitize(row) for row in dynamics]
    sanitized_trajectory = [sanitize(row) for row in trajectory]

    dump_json(out / "run_meta_sanitized.json", meta)
    dump_jsonl(out / "candidate_decisions_sanitized.jsonl", sanitized_candidates)
    dump_jsonl(out / "responsibility_assignments_sanitized.jsonl", sanitized_assignments)
    dump_jsonl(out / "member_opportunities_sanitized.jsonl", sanitized_opportunities)
    dump_jsonl(out / "target_priority_audit_sanitized.jsonl", sanitized_priorities)
    dump_jsonl(out / "g_transition_audit_sanitized.jsonl", sanitized_transitions)
    dump_jsonl(out / "specialization_trajectory_sanitized.jsonl", sanitized_specialization)
    dump_jsonl(out / "training_dynamics_sanitized.jsonl", sanitized_dynamics)
    dump_jsonl(out / "team_differentiation_trajectory_sanitized.jsonl", sanitized_trajectory)
    dump_json(out / "final_behavior_matrices_sanitized.json", sanitize(final_behavior))
    dump_json(out / "token_cost_breakdown_sanitized.json", {
        "artifact_schema_version": "sanitized_token_cost_breakdown_v1",
        "cost_summary": sanitize(cost), "solver_recovery_summary": sanitize(recovery),
    })
    dump_json(out / "candidate_funnel_sanitized.json", {
        "artifact_schema_version": "sanitized_candidate_funnel_v1",
        "update_count": len(sanitized_candidates),
        "accepted_update_count": sum(bool(row.get("accepted_prompt_hash")) for row in sanitized_candidates),
        "per_update_funnel": [
            {"update_index": row["update_index"], "funnel": row["funnel"]}
            for row in sanitized_candidates
        ],
    })
    integrity = verify_report(
        out, candidates=sanitized_candidates, priorities=sanitized_priorities,
        opportunities=sanitized_opportunities, transitions=sanitized_transitions,
        specialization=sanitized_specialization,
    )
    dump_json(out / "report_integrity.json", integrity)

    accepted_count = integrity["checks"]["accepted_transition_join"]["accepted_update_count"]
    final_vote = final_behavior.get("team_vote_correct_count", "unavailable")
    total_tokens = cost.get("total_tokens", "unavailable")
    (out / "README.md").write_text(
        "# v11 Full Seed-43 32-Update Pilot\n\n"
        "This directory contains sanitized, analysis-ready artifacts for one completed "
        "`shared_member_aware_full` pilot. It is a single-seed development result, not a "
        "formal efficacy or generalization claim.\n\n"
        "## Protocol facts\n\n"
        f"- Method: `{meta.get('method_version')}`; checkpoint: `{meta.get('checkpoint_version')}`.\n"
        f"- Planned/completed updates: `{meta.get('planned_update_count')}`/`{meta.get('completed_update_count')}`.\n"
        f"- Validation used: `{meta.get('validation_used')}`; selected state: `final_active_state`.\n"
        f"- Final test evaluations: `{meta.get('test_evaluation_count')}`; test before training complete: `{meta.get('test_called_before_training_complete')}`.\n"
        f"- Accepted updates: `{accepted_count}`; final test vote-correct count: `{final_vote}`.\n"
        f"- Recorded total LLM tokens: `{total_tokens}`.\n\n"
        "All per-example artifacts use hashes only. Prompts, questions, gold labels, literal "
        "answers, raw role/API output, credentials, cache locations, checkpoints, and absolute "
        "paths are excluded. `report_integrity.json` records coverage and join assertions.\n",
        encoding="utf-8",
    )
    manifest = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.iterdir()) if path.is_file()
    }
    dump_json(out / "sha256_manifest.json", manifest)
    verify_report(
        out, candidates=sanitized_candidates, priorities=sanitized_priorities,
        opportunities=sanitized_opportunities, transitions=sanitized_transitions,
        specialization=sanitized_specialization,
    )
    print(json.dumps({"ok": True, "out_dir": str(out), "accepted_updates": accepted_count}, indent=2))


if __name__ == "__main__":
    main()
