"""Derive secret-free v6 control--memory comparison records from local raw runs.

This is an offline report builder.  It reads ignored raw artifacts but emits
only numeric aggregates, booleans, method labels, and one-way hashes into the
already version-controlled sanitized report directories.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.proposal_memory import assigned_residual_set_hash


DATE = "20260729"
TASK = "disambiguation_qa"
FORBIDDEN_TEXT = ("http://", "https://", "final_answer:", "openai_api_key")
ABSOLUTE_PATH = re.compile(r"(?i)(?:[a-z]:[\\/]|file://|\\\\[^\\/\s]+[\\/])")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run_dir(seed: int, treatment: str) -> Path:
    return ROOT / "runs" / f"v6_seed{seed}_{treatment}" / TASK / f"shared_member_aware_full_seed{seed}"


def baseline_dir(seed: int, treatment: str) -> Path:
    return ROOT / "runs" / f"v6_seed{seed}_{treatment}" / TASK / f"shared_baseline_seed{seed}"


def report_dir(seed: int, treatment: str) -> Path:
    return ROOT / "reports" / f"v6_seed{seed}_{treatment}_32updates_{DATE}" / f"shared_member_aware_full_seed{seed}"


def pair_dir(seed: int) -> Path:
    return ROOT / "reports" / f"v6_seed{seed}_control_memory_pair_{DATE}"


def state_snapshot(row: dict[str, Any], initial_counts: list[int]) -> dict[str, Any]:
    counts = [int(value) for value in row["per_agent_correct_counts"]]
    gains = [current - initial for current, initial in zip(counts, initial_counts, strict=True)]
    keys = (
        "team_vote_correct_count", "oracle_correct_count", "mean_G", "mean_H", "mean_M",
        "oracle_covered_but_vote_wrong_rate", "mean_off_diagonal_same_wrong_excess",
        "mean_dominant_wrong_concentration", "mean_pairwise_correctness_correlation", "n_eff",
        "mean_member_accuracy", "minimum_member_accuracy", "terminal_invalid_count",
    )
    return {
        "vote_correct_count": int(row["team_vote_correct_count"]),
        "member_correct_counts": counts,
        "objective": [int(row["team_vote_correct_count"]), min(gains), sum(gains)],
        **{key: row.get(key) for key in keys if key not in {"team_vote_correct_count"}},
    }


def final_test_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "team_vote_correct_count", "per_agent_correct_counts", "oracle_correct_count",
        "oracle_covered_but_vote_wrong_rate", "mean_G", "mean_H", "mean_M",
        "mean_off_diagonal_same_wrong_excess", "mean_dominant_wrong_concentration",
        "mean_pairwise_correctness_correlation", "n_eff", "mean_member_accuracy",
        "minimum_member_accuracy", "terminal_invalid_count",
    )
    return {key: row.get(key) for key in keys}


def accepted_constraint(decision: dict[str, Any]) -> dict[str, Any] | None:
    accepted_hash = str(decision.get("accepted_prompt_hash", ""))
    if not accepted_hash:
        return None
    for candidate in decision.get("candidates", []):
        if candidate.get("prompt_hash") == accepted_hash:
            return dict(candidate.get("constraint", {}))
    return None


def longest_rejection_streak(decisions: list[dict[str, Any]]) -> int:
    longest = current = 0
    for row in decisions:
        if row.get("accepted_prompt_hash"):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def acceptance_summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    categories = {
        "target_only": [], "vote_only": [], "target_and_vote": [], "other": [],
    }
    for row in decisions:
        constraint = accepted_constraint(row)
        if constraint is None:
            continue
        target_positive = int(constraint["target_gain"]) > 0
        vote_positive = int(constraint["vote_correct_candidate"]) > int(constraint["vote_correct_incumbent"])
        label = (
            "target_and_vote" if target_positive and vote_positive else
            "target_only" if target_positive else
            "vote_only" if vote_positive else "other"
        )
        categories[label].append(int(row["update_index"]))
    return {
        "artifact_schema_version": "v6_acceptance_transition_summary_v1",
        "counts": {name: len(indexes) for name, indexes in categories.items()},
        "update_indexes": categories,
    }


def build_final_summary(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dynamics = load_jsonl(run / "training_dynamics.jsonl")
    decisions = load_jsonl(run / "candidate_decisions.jsonl")
    initial = next(row for row in dynamics if int(row["update_index"]) == -1)
    final = max(dynamics, key=lambda row: int(row["update_index"]))
    initial_counts = [int(value) for value in initial["per_agent_correct_counts"]]
    final_test = load_json(run / "final_test_differentiation.json")
    accepted = [row for row in decisions if row.get("accepted_prompt_hash")]
    return {
        "artifact_schema_version": "v6_final_summary_sanitized_v1",
        "initial_train": state_snapshot(initial, initial_counts),
        "final_train": state_snapshot(final, initial_counts),
        "final_test": final_test_snapshot(final_test),
        "training": {
            "accepted_update_count": len(accepted),
            "last_accepted_update_index": max((int(row["update_index"]) for row in accepted), default=None),
            "longest_rejection_streak": longest_rejection_streak(decisions),
            **acceptance_summary(decisions),
        },
    }, decisions


def candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    evaluation = dict(candidate.get("evaluation", {}))
    marginal = dict(evaluation.get("marginal", {}))
    constraint = dict(candidate.get("constraint", {}))
    return {
        "prompt_hash": candidate.get("prompt_hash"),
        "generation": candidate.get("generation"),
        "repair_plan_hash": candidate.get("repair_plan_hash", ""),
        "target_gain": constraint.get("target_gain"),
        "vote_gain_count": constraint.get("vote_gain_count"),
        "vote_loss_count": constraint.get("vote_loss_count"),
        "vote_net_gain": constraint.get("vote_net_gain"),
        "assigned_residual_repair_count": marginal.get("assigned_residual_repair_count"),
        "coverage_gain_count": marginal.get("coverage_gain_count"),
        "coverage_loss_count": marginal.get("coverage_loss_count"),
        "incumbent_objective": constraint.get("incumbent_objective"),
        "candidate_objective": constraint.get("candidate_objective"),
        "rejection_reasons": constraint.get("rejection_reasons", []),
        "passed": constraint.get("passed"),
    }


def build_proposal_trace(run: Path, decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = {int(row["update_index"]): row for row in load_jsonl(run / "proposal_memory_events_sanitized.jsonl")}
    contexts = {int(row["update_index"]): row for row in load_jsonl(run / "tcs_context_history.jsonl")}
    priorities = {int(row["update_index"]): row for row in load_jsonl(run / "target_priority_audit.jsonl")}
    teacher_hashes: dict[int, list[str]] = defaultdict(list)
    for row in load_jsonl(run / "tcs_rounds.jsonl"):
        if row.get("role") == "teacher" and row.get("teacher_plan_hash"):
            teacher_hashes[int(row["update_index"])].append(str(row["teacher_plan_hash"]))

    state_version = 0
    prior_by_key: dict[str, dict[str, Any]] = {}
    output: list[dict[str, Any]] = []
    for decision in sorted(decisions, key=lambda row: int(row["update_index"])):
        update = int(decision["update_index"])
        event = events.get(update, {})
        context = contexts.get(update, {})
        priority = priorities.get(update, {})
        target = int(decision["target_agent_id"])
        assigned = sorted(str(value) for value in decision.get("assigned_question_hashes", []))
        key_hash = str(event.get("memory_key_hash", ""))
        key = key_hash or hash_json({
            "team_state_version": state_version,
            "target_agent_id": target,
            "assigned_residual_set_hash": assigned_residual_set_hash(assigned),
        })
        prior = prior_by_key.get(key, {})
        selected_priority = next(
            (row for row in priority.get("priorities", []) if int(row.get("agent_id", -1)) == target),
            {},
        )
        plan_hashes = sorted(set(teacher_hashes.get(update, [])))
        current_plan_hash = plan_hashes[-1] if plan_hashes else ""
        output.append({
            "artifact_schema_version": "v6_proposal_search_trace_v1",
            "update_index": update,
            "team_state_version": int(event.get("team_state_version", state_version)),
            "target_agent_id": target,
            "assigned_residual_set_hash": event.get("assigned_residual_set_hash") or assigned_residual_set_hash(assigned),
            "assigned_residual_count": len(assigned),
            "assigned_portfolio": {
                "assigned_load": selected_priority.get("assigned_load"),
                "owned_direct_vote_fix_count": selected_priority.get("owned_direct_vote_fix_count"),
                "owned_coverage_opportunity_count": selected_priority.get("owned_coverage_opportunity_count"),
                "owned_oracle_soft_utility_gain_sum": selected_priority.get("owned_oracle_soft_utility_gain_sum"),
            },
            "memory_hit": bool(event.get("memory_hit", False)),
            "memory_attempt_count_before": int(prior.get("failure_attempt_count", 0)),
            "previous_failure_stage": prior.get("failure_stage"),
            "revision_mode": event.get("revision_mode", "none"),
            "rotation_level": event.get("rotation_level", "none"),
            "previous_evidence_bundle_hash": event.get("previous_evidence_bundle_hash"),
            "current_evidence_bundle_hash": event.get("current_evidence_bundle_hash") or context.get("evidence_bundle_hash", ""),
            "previous_repair_plan_hash": prior.get("repair_plan_hash"),
            "current_repair_plan_hash": current_plan_hash,
            "pattern_ids": sorted(context.get("selected_context_pattern_question_hashes", {})),
            "representative_question_hashes": sorted({
                value for values in context.get("selected_context_pattern_question_hashes", {}).values()
                for value in values
            }),
            "candidate_count": int(decision.get("funnel", {}).get("valid_candidate_count", 0)),
            "candidates": [candidate_summary(row) for row in decision.get("candidates", [])],
            "accepted": bool(decision.get("accepted_prompt_hash")),
            "accepted_candidate_hash": decision.get("accepted_prompt_hash", ""),
        })
        if event:
            prior_by_key[key] = {
                "failure_attempt_count": int(prior.get("failure_attempt_count", 0)) + (0 if event.get("failure_stage") == "accepted" else 1),
                "failure_stage": event.get("failure_stage"),
                "repair_plan_hash": current_plan_hash,
            }
        if decision.get("accepted_prompt_hash"):
            state_version += 1
    return output


def build_per_update_cost(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose auditable call counts; token attribution is unavailable in v6 logs."""
    rows = []
    for decision in sorted(decisions, key=lambda row: int(row["update_index"])):
        funnel = decision.get("funnel", {})
        rows.append({
            "artifact_schema_version": "v6_per_update_cost_sanitized_v1",
            "update_index": int(decision["update_index"]),
            "accepted": bool(decision.get("accepted_prompt_hash")),
            "teacher_call_count": funnel.get("teacher_calls", 0),
            "critic_call_count": funnel.get("critic_calls", 0),
            "student_call_count": funnel.get("student_calls", 0),
            "stage_a_candidate_evaluations": funnel.get("stage_a_evaluated", 0),
            "stage_b_candidate_evaluations": funnel.get("stage_b_evaluated", 0),
            "solver_request_count": "unavailable",
            "teacher_tokens": "unavailable",
            "critic_tokens": "unavailable",
            "student_tokens": "unavailable",
            "solver_tokens": "unavailable",
            "stage_a_solver_tokens": "unavailable",
            "stage_b_solver_tokens": "unavailable",
            "cache_hit_count": "unavailable",
            "cache_miss_count": "unavailable",
            "total_update_tokens": "unavailable",
            "unavailability_reason": "v6 llm_calls audit has no update_index; exact per-update token/cache attribution cannot be reconstructed safely",
        })
    return rows


def safe_config(meta: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "agent_model", "optimizer_model", "evaluator_model", "temperature", "solver_max_tokens",
        "solver_invalid_max_retries", "train_size", "val_size", "test_size", "epochs", "update_every",
        "candidate_eval_pool_size", "num_candidates_per_parent", "stage_b_candidate_budget",
        "eval_solver_call_concurrency", "responsibility_max_wait_updates", "responsibility_switch_margin",
        "teacher_critic_max_rounds", "teacher_temperature", "critic_temperature", "student_temperature",
    )
    return {key: meta.get("config", {}).get(key) for key in keys}


def baseline_metrics_hash(seed: int, treatment: str) -> str:
    summary = load_json(baseline_dir(seed, treatment) / "final_summary.json")
    metrics = dict(summary["selected_test"])
    metrics.pop("rows", None)
    return hash_json(metrics)


def build_pair_manifest(
    seed: int,
    control_meta: dict[str, Any],
    memory_meta: dict[str, Any],
    control_initial: dict[str, Any],
    memory_initial: dict[str, Any],
    *,
    control_baseline_metrics_hash: str = "unavailable",
    memory_baseline_metrics_hash: str = "unavailable",
) -> dict[str, Any]:
    control_identity = control_meta["run_identity"]
    memory_identity = memory_meta["run_identity"]
    source_match = control_identity["git_commit"] == memory_identity["git_commit"]
    normalized_match = safe_config(control_meta) == safe_config(memory_meta)
    initial_match = control_initial == memory_initial
    source_delta = not source_match
    return {
        "artifact_schema_version": "v6_matched_pair_manifest_v1",
        "seed": seed,
        "control": {
            "source_commit": control_identity["git_commit"], "git_dirty": control_identity["git_dirty"],
            "config_fingerprint": control_identity["config_fingerprint"],
            "proposal_memory_mode": control_meta["proposal_memory_mode"],
            "initial_prompt_hashes": control_meta["initial_prompt_hashes"],
            "initial_train_profile_hash": "unavailable",
            "initial_train_state_hash": hash_json(control_initial),
            "baseline_test_profile_hash": "unavailable",
            "baseline_test_metrics_hash": control_baseline_metrics_hash,
            "model_request_identity": control_meta["prompt_question_evaluator_identity"],
        },
        "memory": {
            "source_commit": memory_identity["git_commit"], "git_dirty": memory_identity["git_dirty"],
            "config_fingerprint": memory_identity["config_fingerprint"],
            "proposal_memory_mode": memory_meta["proposal_memory_mode"],
            "initial_prompt_hashes": memory_meta["initial_prompt_hashes"],
            "initial_train_profile_hash": "unavailable",
            "initial_train_state_hash": hash_json(memory_initial),
            "baseline_test_profile_hash": "unavailable",
            "baseline_test_metrics_hash": memory_baseline_metrics_hash,
            "model_request_identity": memory_meta["prompt_question_evaluator_identity"],
        },
        "shared": {
            "manifest_hash": control_identity["manifest_sha256"],
            "train_file_hash": control_identity["train_file_sha256"],
            "val_file_hash": control_identity["val_file_sha256"],
            "test_file_hash": control_identity["test_file_sha256"],
            "probe_hash": {"control": control_meta["probe_hash"], "memory": memory_meta["probe_hash"]},
            "candidate_budget": {
                key: safe_config(control_meta)[key]
                for key in ("num_candidates_per_parent", "stage_b_candidate_budget", "candidate_eval_pool_size")
            },
            "planned_updates": {"control": control_meta["planned_update_count"], "memory": memory_meta["planned_update_count"]},
            "cache_policy": "fresh_independent_local_shared_cache_per_treatment",
        },
        "source_revision_difference": source_delta,
        "difference_affects_full_runtime": False if source_delta else False,
        "difference_affects_baseline_only": True if source_delta else False,
        "runtime_config_match": normalized_match,
        "initial_state_match": initial_match,
        "matched_status": (
            "matched" if source_match and normalized_match and initial_match
            else "near-matched" if normalized_match and initial_match
            else "unmatched"
        ),
        "comparison_label": (
            "matched" if source_match and normalized_match and initial_match
            else "near-matched pending source-revision audit" if normalized_match and initial_match
            else "unmatched: initial train state differs"
        ),
        "unavailable_fields": {
            "initial_train_profile_hash": "final v6 run artifacts do not persist initial per-question profiles",
            "baseline_test_profile_hash": "baseline final summary persists aggregate metrics rather than per-member profiles",
        },
    }


def selected_pair_values(summary: dict[str, Any], cost: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
    train = summary["final_train"]
    test = summary["final_test"]
    return {
        "final_train_objective": train["objective"],
        "final_train_vote": train["vote_correct_count"],
        "final_train_mean_H": train["mean_H"],
        "final_train_mean_M": train["mean_M"],
        "final_test_vote": test["team_vote_correct_count"],
        "final_test_oracle": test["oracle_correct_count"],
        "final_test_mean_H": test["mean_H"],
        "final_test_mean_M": test["mean_M"],
        "final_test_same_wrong_excess": test["mean_off_diagonal_same_wrong_excess"],
        "final_test_member_mean_accuracy": test["mean_member_accuracy"],
        "final_test_member_minimum_accuracy": test["minimum_member_accuracy"],
        "final_test_n_eff": test["n_eff"],
        "accepted_updates": summary["training"]["accepted_update_count"],
        "longest_rejection_streak": summary["training"]["longest_rejection_streak"],
        "target_only_accepted_updates": summary["training"]["counts"]["target_only"],
        "vote_only_accepted_updates": summary["training"]["counts"]["vote_only"],
        "target_and_vote_accepted_updates": summary["training"]["counts"]["target_and_vote"],
        "total_tokens": cost.get("total_tokens"),
        "tokens_per_accepted_update": cost.get("tokens_per_accepted_update"),
        "memory_hits": memory.get("memory_hit_count", 0),
        "memory_hit_acceptance_rate": memory.get("memory_hit_acceptance_rate", "unavailable"),
    }


def delta(memory: Any, control: Any) -> Any:
    if isinstance(memory, (int, float)) and isinstance(control, (int, float)):
        return memory - control
    if isinstance(memory, list) and isinstance(control, list) and len(memory) == len(control):
        return [delta(m_value, c_value) for m_value, c_value in zip(memory, control, strict=True)]
    return "unavailable"


def write_source_audit(out: Path) -> None:
    names = subprocess.run(
        ["git", "diff", "--name-only", "5ad0fb9", "50bbc70"], cwd=ROOT,
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.splitlines()
    content = """# Seed44 Source-Revision Delta Audit

`v6_seed44_control` used `5ad0fb9`; `v6_seed44_memory` used `50bbc70`.
The source delta is baseline-only for the Full setting, but the paired runs
also have different initial train states. The comparison is therefore labelled
**unmatched**; it is not source-matched or near-matched.

## Changed files

""" + "\n".join(f"- `{name}`" for name in names) + """

## Semantic assessment

- Responsibility assignment, target scheduler, candidate acceptance, Solver contract, and Full proposal-memory implementation: unchanged by this revision delta.
- The runner and run-specific preflight now force `shared_baseline` to `proposal_memory_mode=off`; `shared_member_aware_full` retains its requested mode.
- This fixes the baseline launch path. The Seed44 control Full run already requested `off`, so the diff does not change its Full optimization logic.

The accompanying `matched_pair_manifest.json` independently records normalized
runtime-config and initial-train-state comparisons. No conclusion should treat
Seed44 as a matched efficacy comparison unless both fields are consistent.
"""
    (out / "source_revision_delta_audit.md").write_text(content, encoding="utf-8")


def scan(path: Path) -> None:
    for item in path.rglob("*"):
        if item.is_file():
            text = item.read_text(encoding="utf-8").lower()
            if any(token in text for token in FORBIDDEN_TEXT) or ABSOLUTE_PATH.search(text):
                raise ValueError(f"sanitization scan failed: {item}")


def main() -> None:
    pair_summaries: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    pair_meta: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    pair_initial: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for seed in (44, 45):
        for treatment in ("control", "memory"):
            run = run_dir(seed, treatment)
            report = report_dir(seed, treatment)
            if not run.is_dir() or not report.is_dir():
                raise FileNotFoundError(f"missing raw run or sanitized report: {seed}/{treatment}")
            final, decisions = build_final_summary(run)
            trace = build_proposal_trace(run, decisions)
            per_update_cost = build_per_update_cost(decisions)
            acceptance = acceptance_summary(decisions)
            dump_json(report / "final_summary_sanitized.json", final)
            dump_jsonl(report / "proposal_search_trace_sanitized.jsonl", trace)
            dump_jsonl(report / "per_update_cost_sanitized.jsonl", per_update_cost)
            dump_json(report / "acceptance_transition_summary.json", acceptance)
            scan(report)
            pair_summaries[seed][treatment] = final
            pair_meta[seed][treatment] = load_json(run / "run_meta.json")
            pair_initial[seed][treatment] = final["initial_train"]

    for seed in (44, 45):
        out = pair_dir(seed)
        out.mkdir(parents=True, exist_ok=True)
        manifest = build_pair_manifest(
            seed, pair_meta[seed]["control"], pair_meta[seed]["memory"],
            pair_initial[seed]["control"], pair_initial[seed]["memory"],
            control_baseline_metrics_hash=baseline_metrics_hash(seed, "control"),
            memory_baseline_metrics_hash=baseline_metrics_hash(seed, "memory"),
        )
        control_cost = load_json(run_dir(seed, "control") / "cost_summary.json")
        memory_cost = load_json(run_dir(seed, "memory") / "cost_summary.json")
        control_memory = load_json(run_dir(seed, "control") / "proposal_memory_summary.json")
        memory_memory = load_json(run_dir(seed, "memory") / "proposal_memory_summary.json")
        control_values = selected_pair_values(pair_summaries[seed]["control"], control_cost, control_memory)
        memory_values = selected_pair_values(pair_summaries[seed]["memory"], memory_cost, memory_memory)
        comparison = {
            "artifact_schema_version": "v6_control_memory_pair_comparison_v1",
            "seed": seed,
            "comparison_status": manifest["matched_status"],
            "control": control_values,
            "memory": memory_values,
            "memory_minus_control": {
                key: delta(memory_values[key], control_values[key]) for key in control_values
            },
        }
        dump_json(out / "matched_pair_manifest.json", manifest)
        dump_json(out / "control_memory_pair_comparison.json", comparison)
        if seed == 44:
            write_source_audit(out)
        scan(out)
    print(json.dumps({"ok": True, "seeds": [44, 45]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
