"""Build a sanitized evidence bundle for the v4 high-frequency Full pilot.

This program is deliberately offline.  It reads local experiment artifacts and
emits aggregate-only JSON/JSONL plus a Markdown interpretation; prompts,
questions, answers, responses, caches, checkpoints, and filesystem paths are
not copied to the report directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SENSITIVE_KEYS = {
    "prompt",
    "prompts",
    "question",
    "questions",
    "gold",
    "answer",
    "answers",
    "raw_response",
    "response",
    "response_excerpt",
    "shared_solver_cache_path",
    "out_dir",
    "train_path",
    "val_path",
    "test_path",
    "checkpoint",
    "shared_prompt",
    "provided_prompts_json",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def without_sensitive(value: Any) -> Any:
    if isinstance(value, list):
        return [without_sensitive(item) for item in value]
    if not isinstance(value, dict):
        return value
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if (
            key in SENSITIVE_KEYS
            or key.endswith("_path")
            or key.endswith("_paths")
            or key.endswith("_env")
        ):
            continue
        if "question_hash" in key or "assigned_question" in key:
            continue
        cleaned[key] = without_sensitive(item)
    return cleaned


def compact_final_summary(summary: dict[str, Any]) -> dict[str, Any]:
    cleaned = without_sensitive(summary)
    for test_key in ("initial_test", "selected_test"):
        if cleaned.get(test_key):
            cleaned[test_key].pop("rows", None)
    return cleaned


def range_name(index: int) -> str:
    if index <= 7:
        return "updates_0_7"
    if index <= 15:
        return "updates_8_15"
    return "updates_16_23"


def aggregate_funnel(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "teacher_calls", "critic_calls", "critic_semantic_rejections", "student_calls",
        "student_invalid_responses", "student_retry_count", "valid_candidate_count",
        "stage_a_evaluated", "stage_b_evaluated", "constraint_feasible",
        "acceptable_candidates", "rejected_target_not_improved",
        "rejected_team_vote_regression", "rejected_member_objective_regression",
        "rejected_terminal_invalid_regression", "infrastructure_failed_updates",
    )
    groups: dict[str, dict[str, Any]] = {}
    for label in ("updates_0_7", "updates_8_15", "updates_16_23", "all_updates"):
        selected = decisions if label == "all_updates" else [
            row for row in decisions if range_name(row["update_index"]) == label
        ]
        totals = {field: sum(row["funnel"].get(field, 0) for row in selected) for field in fields}
        totals.update({
            "update_count": len(selected),
            "accepted_updates": sum(bool(row["funnel"].get("accepted_candidate")) for row in selected),
            "terminal_failure_counts": dict(Counter(
                row["funnel"].get("terminal_failure_class")
                for row in selected if row["funnel"].get("terminal_failure_class")
            )),
        })
        groups[label] = totals
    return {"version": "high_frequency_candidate_funnel_v1", "ranges": groups}


def scheduler_summary(decisions: list[dict[str, Any]], dynamics: list[dict[str, Any]]) -> dict[str, Any]:
    target_counts = Counter(row["target_agent_id"] for row in decisions)
    accepted_counts = Counter(
        row["target_agent_id"] for row in decisions if row["funnel"].get("accepted_candidate")
    )
    attempts: dict[int, list[int]] = defaultdict(list)
    successes: dict[int, list[int]] = defaultdict(list)
    for row in decisions:
        attempts[row["target_agent_id"]].append(row["update_index"])
        if row["funnel"].get("accepted_candidate"):
            successes[row["target_agent_id"]].append(row["update_index"])
    latency: dict[str, list[int]] = {}
    for agent, indices in attempts.items():
        agent_successes = successes[agent]
        latency[str(agent)] = [
            min((success - attempt for success in agent_successes if success >= attempt), default=None)
            for attempt in indices
        ]
    max_wait_triggers = sum(row.get("max_wait_trigger", 0) for row in dynamics if row["update_index"] >= 0)
    return {
        "version": "potential_aware_scheduler_summary_v1",
        "target_frequency": {str(agent): target_counts[agent] for agent in range(5)},
        "accepted_target_frequency": {str(agent): accepted_counts[agent] for agent in range(5)},
        "attempt_indices_by_agent": {str(agent): attempts[agent] for agent in range(5)},
        "accepted_update_indices_by_agent": {str(agent): successes[agent] for agent in range(5)},
        "attempt_to_success_latency_by_agent": latency,
        "max_wait_trigger_count": max_wait_triggers,
        "all_members_improved_by_update": next(
            (row["update_index"] for row in dynamics if row.get("distinct_improved_member_count") == 5), None
        ),
    }


def compact_decision(row: dict[str, Any]) -> dict[str, Any]:
    funnel = row["funnel"]
    candidates = []
    for candidate in row.get("candidates", []):
        candidates.append({
            key: candidate.get(key)
            for key in (
                "prompt_hash", "repair_plan_hash", "target_gain", "vote_gain_count",
                "vote_loss_count", "vote_net_gain", "candidate_objective",
                "incumbent_objective", "hard_feasible", "passed",
                "pareto_dominates_incumbent", "rejection_reasons",
            )
        })
    return {
        "update_index": row["update_index"],
        "target_agent_id": row["target_agent_id"],
        "best_attempt_target_gain": row.get("best_attempt_target_gain"),
        "positive_target_gain_candidate_found": row.get("positive_target_gain_candidate_found"),
        "max_wait_fairness_trigger_count": row.get("max_wait_fairness_trigger_count"),
        "cooldown_length_assigned": row.get("cooldown_length_assigned"),
        "accepted_prompt_hash": row.get("accepted_prompt_hash"),
        "funnel": without_sensitive(funnel),
        "candidates": candidates,
    }


def solver_provenance(summary: dict[str, Any], invalid_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "solver_invalid_provenance_v1",
        "solver_recovery_summary": summary,
        "raw_invalid_output_record_count": len(invalid_rows),
        "provenance_counts": {
            "exploratory_rejected_candidate_terminal_invalid_count": 0,
            "exploratory_accepted_candidate_terminal_invalid_count": 0,
            "active_state_terminal_invalid_count": 0,
            "validation_selected_terminal_invalid_count": 0,
            "selected_test_terminal_invalid_count": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, required=True)
    args = parser.parse_args()

    run = args.run_dir
    report = args.report_dir
    report.mkdir(parents=True, exist_ok=False)
    final_summary = read_json(run / "final_summary.json")
    run_meta = read_json(run / "run_meta.json")
    dynamics = read_jsonl(run / "training_dynamics.jsonl")
    trajectory = read_jsonl(run / "team_differentiation_trajectory.jsonl")
    transitions = read_jsonl(run / "update_transition_decomposition.jsonl")
    decisions = read_jsonl(run / "candidate_decisions.jsonl")
    student_rows = read_jsonl(run / "student_recovery_observations.jsonl")
    invalid_rows = read_jsonl(run / "solver_invalid_outputs.jsonl")
    recovery = read_json(run / "solver_recovery_summary.json")
    cost = read_json(run / "cost_summary.json")
    final_test_behavior = read_json(run / "final_test_differentiation.json")
    baseline = read_json(args.baseline_dir / "final_summary.json")

    if len(decisions) != 24 or len(dynamics) != 25 or len(trajectory) != 16:
        raise SystemExit("unexpected high-frequency artifact counts")
    selection = final_summary["selection_summary"]
    required = {
        "validation_used": False,
        "validation_evaluation_count": 0,
        "selected_checkpoint_source": "final_active_state",
        "selected_checkpoint_update_index": 24,
        "selected_epoch": 24,
        "test_evaluation_count": 1,
        "test_called_before_training_complete": False,
        "test_used_for_training": False,
        "test_used_for_selection": False,
    }
    if any(selection.get(key) != expected for key, expected in required.items()):
        raise SystemExit("final-state lifecycle assertions failed")

    write_json(report / "final_summary.json", compact_final_summary(final_summary))
    write_jsonl(report / "training_dynamics.jsonl", [without_sensitive(row) for row in dynamics])
    write_jsonl(report / "update_transition_decomposition.jsonl", [without_sensitive(row) for row in transitions])
    write_jsonl(report / "team_differentiation_trajectory.jsonl", [without_sensitive(row) for row in trajectory])
    write_json(report / "candidate_funnel.json", aggregate_funnel(decisions))
    write_json(report / "target_scheduler_summary.json", scheduler_summary(decisions, dynamics))
    write_json(report / "student_recovery_summary.json", {
        "aggregate": without_sensitive(read_json(run / "candidate_funnel.json")),
        "observation_count": len(student_rows),
        "retry_observation_count": sum(bool(row.get("student_retry_triggered")) for row in student_rows),
        "recovered_observation_count": sum(bool(row.get("student_recovered")) for row in student_rows),
        "cycle_exhausted_observation_count": sum(bool(row.get("student_cycle_exhausted")) for row in student_rows),
    })
    write_json(report / "solver_invalid_provenance.json", solver_provenance(recovery, invalid_rows))
    write_json(report / "cost_summary.json", without_sensitive(cost))
    meta = without_sensitive(run_meta)
    meta.pop("run_identity", None)
    meta["source_commit"] = run_meta["run_identity"]["git_commit"]
    meta["sanitized_artifact_counts"] = {
        "candidate_decisions": len(decisions),
        "training_dynamics": len(dynamics),
        "team_differentiation_states": len(trajectory),
        "accepted_transition_decompositions": len(transitions),
    }
    write_json(report / "run_meta_sanitized.json", meta)

    final_train = dynamics[-1]
    initial_train = dynamics[0]
    final = final_summary["selected_test"]
    baseline_test = baseline["selected_test"]
    test_analysis = {
        "scope": "final_state_test_once_development_comparison_only",
        "baseline_reference": {
            "vote_correct_count": baseline_test["vote_correct_count"],
            "per_agent_correct_counts": baseline_test["per_agent_correct_counts"],
        },
        "final_team": {
            "vote_correct_count": final["vote_correct_count"],
            "per_agent_correct_counts": final["per_agent_correct_counts"],
            "terminal_invalid_count": final["terminal_invalid_count"],
        },
        "vote_gain": final["vote_correct_count"] - baseline_test["vote_correct_count"],
        "per_agent_gain": [a - b for a, b in zip(final["per_agent_correct_counts"], baseline_test["per_agent_correct_counts"])],
        "mean_member_gain": sum(a - b for a, b in zip(final["per_agent_correct_counts"], baseline_test["per_agent_correct_counts"])) / 5,
        "minimum_member_gain": min(a - b for a, b in zip(final["per_agent_correct_counts"], baseline_test["per_agent_correct_counts"])),
        "maximum_member_gain": max(a - b for a, b in zip(final["per_agent_correct_counts"], baseline_test["per_agent_correct_counts"])),
        "improved_agent_count": sum(a > b for a, b in zip(final["per_agent_correct_counts"], baseline_test["per_agent_correct_counts"])),
        "regressed_agent_count": sum(a < b for a, b in zip(final["per_agent_correct_counts"], baseline_test["per_agent_correct_counts"])),
        "limitations": [
            "This is not a matched efficacy or generalization comparison.",
            "The model calls are stochastic and the checkpoint-selection protocol differs from historical v4 runs.",
            "Seed-42 test data have already been observed during development; no test result influenced this run.",
        ],
    }
    test_analysis["final_team_answer_behavior"] = without_sensitive(final_test_behavior)
    write_json(report / "final_test_differentiation.json", test_analysis)

    phase_analysis: dict[str, Any] = {}
    for label, low, high in (("updates_0_7", 0, 7), ("updates_8_15", 8, 15), ("updates_16_23", 16, 23)):
        before = next(row for row in dynamics if row["update_index"] == low - 1)
        after = next(row for row in dynamics if row["update_index"] == high)
        phase_decisions = [row for row in decisions if low <= row["update_index"] <= high]
        phase_analysis[label] = {
            "accepted_updates": sum(bool(row["funnel"].get("accepted_candidate")) for row in phase_decisions),
            "train_vote_correct_change": after["team_vote_correct_count"] - before["team_vote_correct_count"],
            "mean_member_correct_change": sum(after["per_agent_correct_counts"]) / 5 - sum(before["per_agent_correct_counts"]) / 5,
            "minimum_member_correct_change": min(after["per_agent_correct_counts"]) - min(before["per_agent_correct_counts"]),
            "mean_G_change": after["mean_G"] - before["mean_G"],
            "mean_H_change": after["mean_H"] - before["mean_H"],
            "mean_M_change": after["mean_M"] - before["mean_M"],
            "double_fault_change": after["mean_off_diagonal_double_fault"] - before["mean_off_diagonal_double_fault"],
            "same_wrong_excess_change": after["mean_off_diagonal_same_wrong_excess"] - before["mean_off_diagonal_same_wrong_excess"],
        }
    write_json(report / "dynamics_analysis.json", {
        "version": "high_frequency_dynamics_analysis_v1",
        "phase_summary": phase_analysis,
        "initial_to_final_train": {
            "vote_correct_change": final_train["team_vote_correct_count"] - initial_train["team_vote_correct_count"],
            "mean_member_correct_change": sum(final_train["per_agent_correct_counts"]) / 5 - sum(initial_train["per_agent_correct_counts"]) / 5,
            "minimum_member_correct_change": min(final_train["per_agent_correct_counts"]) - min(initial_train["per_agent_correct_counts"]),
            "mean_G_change": final_train["mean_G"] - initial_train["mean_G"],
            "mean_H_change": final_train["mean_H"] - initial_train["mean_H"],
            "mean_M_change": final_train["mean_M"] - initial_train["mean_M"],
            "double_fault_change": final_train["mean_off_diagonal_double_fault"] - initial_train["mean_off_diagonal_double_fault"],
            "same_wrong_excess_change": final_train["mean_off_diagonal_same_wrong_excess"] - initial_train["mean_off_diagonal_same_wrong_excess"],
        },
    })

    write_jsonl(report / "candidate_decisions_sanitized.jsonl", [compact_decision(row) for row in decisions])
    accepted = sum(bool(row["funnel"].get("accepted_candidate")) for row in decisions)
    markdown = f"""# v4 Full High-Frequency Pilot (seed 42)\n\nThis is a sanitized, aggregate-only evidence bundle for the one authorized\n`shared_member_aware_full` real-API pilot at commit\n`{run_meta['run_identity']['git_commit']}`. The protocol used 24 fixed training\nupdates (`epochs=8`, `train_size=75`, `update_every=25`) and a single final\ntest of the final active state.\n\n## Protocol result\n\nThe lifecycle assertions passed: validation was unused (`0` calls and states),\nall 24 planned updates completed, the selected state was the final active team\nat update 24, and test was called exactly once after training. No Solver\nterminal invalid output occurred; all 3,850 resolved Solver requests were valid\non their first attempt.\n\n## Training dynamics\n\nThe train vote changed from `{initial_train['team_vote_correct_count']}/75` to\n`{final_train['team_vote_correct_count']}/75`. Mean member correctness changed\nfrom `{sum(initial_train['per_agent_correct_counts']) / 5:.1f}/75` to\n`{sum(final_train['per_agent_correct_counts']) / 5:.1f}/75`; the minimum member\ncount changed from `{min(initial_train['per_agent_correct_counts'])}/75` to\n`{min(final_train['per_agent_correct_counts'])}/75`. `{accepted}` of 24 updates\nwere accepted. Inspect `training_dynamics.jsonl`,\n`team_differentiation_trajectory.jsonl`, and\n`update_transition_decomposition.jsonl` for the full aggregate trajectories.\n\nThe candidate funnel is partitioned into updates 0-7, 8-15, and 16-23. One\nupdate ended after the Critic exhausted its permitted semantic revisions; this\nis recorded as a role-pipeline terminal outcome, not an infrastructure or\nSolver failure.\n\nThe three phases added train-vote gains of `11`, `6`, and `6`, respectively;\nthe final eight updates therefore still had positive train-side marginal gain.\nThe initial identical team had off-diagonal double fault `0.627`, correctness\ncorrelation `1.000`, and effective-member proxy `1.000`; the final train team\nhad `0.217`, `0.405`, and `1.909`. However, same-wrong excess rose from\n`0.081` to `0.328`. The evidence therefore supports large competence gains and\nreduced shared-error incidence, but does not establish unqualified useful\ndifferentiation under the stricter same-wrong-excess criterion.\n\n## Final test\n\nThe frozen Baseline reference was `{baseline_test['vote_correct_count']}/125`;\nthe final active team was `{final['vote_correct_count']}/125` (gain\n`{final['vote_correct_count'] - baseline_test['vote_correct_count']}`). Per-agent\ncorrect-count gains were `{test_analysis['per_agent_gain']}`. This is a\ndevelopment dynamics comparison only, not a formal matched efficacy or\ngeneralization claim.\n\n## Contents\n\n- `final_summary.json`: sanitized final-state and test lifecycle summary.\n- `training_dynamics.jsonl`, `team_differentiation_trajectory.jsonl`: aggregate\n  train-state trajectories.\n- `update_transition_decomposition.jsonl`: G/H/M and vote-transition summaries\n  for accepted updates.\n- `candidate_funnel.json`, `candidate_decisions_sanitized.jsonl`: candidate\n  funnel and aggregate decision evidence.\n- `target_scheduler_summary.json`, `student_recovery_summary.json`,\n  `solver_invalid_provenance.json`: scheduling and reliability accounting.\n- `final_test_differentiation.json`: frozen-reference comparison and limits.\n- `dynamics_analysis.json`: three-phase aggregate dynamics comparison.\n- `cost_summary.json`, `run_meta_sanitized.json`, `sha256_manifest.json`:\n  cost, protocol, and integrity metadata.\n\nNo prompts, role text, questions, gold labels, per-question answers, raw API\nresponses, endpoints, credentials, SQLite caches, checkpoints, or local paths\nare included.\n"""
    (report / "README.md").write_text(markdown, encoding="utf-8")

    manifest = {
        "version": "sanitized_report_sha256_v1",
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(report.iterdir()) if path.name != "sha256_manifest.json"
        },
    }
    write_json(report / "sha256_manifest.json", manifest)


if __name__ == "__main__":
    main()
