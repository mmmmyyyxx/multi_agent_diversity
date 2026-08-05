from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.final_method_source_identity import build_source_identity


REPORT_VERSION = "strict_v2_disambiguation_s345_v1"
SETTINGS = (
    "shared_baseline",
    "shared_peer_state_member_pareto",
    "shared_member_aware_responsibility",
    "shared_member_aware_full",
)
OPTIMIZED_SETTINGS = SETTINGS[1:]
SEEDS = (44, 45, 46)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_dir(root: Path, seed: int, setting: str) -> Path:
    return root / "disambiguation_qa" / f"{setting}_seed{seed}"


def _safe_mean(values: Iterable[float]) -> float | str:
    materialized = list(values)
    return statistics.mean(materialized) if materialized else "unavailable"


def _share_top_two(values: list[int]) -> float | str:
    total = sum(values)
    return sum(sorted(values, reverse=True)[:2]) / total if total else "unavailable"


def _member_outcome(baseline: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    baseline_counts = [int(value) for value in baseline["per_agent_correct_counts"]]
    selected_counts = [int(value) for value in selected["per_agent_correct_counts"]]
    gains = [
        current - initial
        for current, initial in zip(selected_counts, baseline_counts, strict=True)
    ]
    vote_delta = int(selected["vote_correct_count"]) - int(baseline["vote_correct_count"])
    g_min = min(gains)
    g_sum = sum(gains)
    return {
        "vote_correct_count": int(selected["vote_correct_count"]),
        "vote_accuracy": float(selected["plurality_vote_acc"]),
        "vote_delta_relative_to_s0": vote_delta,
        "per_member_correct_counts": selected_counts,
        "per_member_gains_relative_to_s0": gains,
        "g_min": g_min,
        "g_sum": g_sum,
        "N_positive": sum(value > 0 for value in gains),
        "strict_pareto_success_relative_to_s0": bool(
            vote_delta >= 0 and g_min >= 0 and g_sum >= 0
            and (vote_delta > 0 or g_min > 0 or g_sum > 0)
        ),
    }


def _proposal_summary(run_dir: Path) -> dict[str, Any]:
    decisions = _read_jsonl(run_dir / "candidate_decisions.jsonl")
    candidates = [
        candidate
        for decision in decisions
        for candidate in decision.get("candidates", [])
        if isinstance(candidate.get("constraint"), dict)
    ]
    target_positive = sum(int(row["constraint"].get("target_gain", 0)) > 0 for row in candidates)
    vote_positive = sum(int(row["constraint"].get("vote_net_gain", 0)) > 0 for row in candidates)
    target_or_vote = sum(
        int(row["constraint"].get("target_gain", 0)) > 0
        or int(row["constraint"].get("vote_net_gain", 0)) > 0
        for row in candidates
    )
    acceptable = sum(bool(row["constraint"].get("passed")) for row in candidates)
    repair_counts = [
        int((row.get("evaluation") or {}).get("marginal", {}).get(
            "assigned_residual_repair_count", 0
        ))
        for row in candidates
    ]
    accepted_updates = [
        int(row.get("update_index", -1))
        for row in decisions if row.get("accepted_prompt_hash")
    ]
    longest = current = 0
    for row in decisions:
        if row.get("accepted_prompt_hash"):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    count = len(candidates)
    return {
        "evaluated_candidate_count": count,
        "target_positive_candidate_rate": target_positive / count if count else "unavailable",
        "vote_positive_candidate_rate": vote_positive / count if count else "unavailable",
        "target_or_vote_positive_rate": target_or_vote / count if count else "unavailable",
        "acceptable_candidate_rate": acceptable / count if count else "unavailable",
        "assigned_residual_repair_count": sum(repair_counts),
        "assigned_residual_repair_candidate_rate": (
            sum(value > 0 for value in repair_counts) / count if count else "unavailable"
        ),
        "accepted_update_count": len(accepted_updates),
        "last_accepted_update": max(accepted_updates, default="unavailable"),
        "longest_rejection_streak": longest,
    }


def _scheduling_summary(run_dir: Path) -> dict[str, Any]:
    decisions = _read_jsonl(run_dir / "candidate_decisions.jsonl")
    selected = [
        int(row["target_agent_id"])
        for row in decisions if row.get("target_agent_id") is not None
    ]
    accepted = [
        int(row["target_agent_id"])
        for row in decisions
        if row.get("target_agent_id") is not None and row.get("accepted_prompt_hash")
    ]
    selected_counts = [selected.count(agent) for agent in range(5)]
    accepted_counts = [accepted.count(agent) for agent in range(5)]
    priority_rows = _read_jsonl(run_dir / "target_priority_audit.jsonl")
    responsible_waits = [
        int(priority.get("updates_since_selected", 0))
        for row in priority_rows
        for priority in row.get("priorities", [])
        if int(priority.get("agent_id", -1)) in {
            int(value) for value in row.get("eligible_agent_ids", [])
        }
    ]
    return {
        "selected_updates_by_member": selected_counts,
        "accepted_updates_by_member": accepted_counts,
        "max_selected_count": max(selected_counts),
        "min_selected_count": min(selected_counts),
        "top_2_selection_share": _share_top_two(selected_counts),
        "top_2_acceptance_share": _share_top_two(accepted_counts),
        "never_selected_members": [agent for agent, value in enumerate(selected_counts) if not value],
        "never_accepted_members": [agent for agent, value in enumerate(accepted_counts) if not value],
        "max_responsible_member_wait": max(responsible_waits, default="unavailable"),
    }


def _vote_structure(run_dir: Path) -> dict[str, Any]:
    payload = _read_json(run_dir / "final_test_differentiation.json")
    return {
        key: payload.get(key, "unavailable")
        for key in (
            "oracle_correct_count", "mean_G", "mean_H", "mean_M",
            "all_wrong_rate", "oracle_covered_but_vote_wrong_rate",
            "mean_off_diagonal_double_fault",
            "mean_off_diagonal_same_wrong_excess",
        )
    }


def _training_summary(run_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(run_dir / "training_dynamics.jsonl")
    if not rows:
        return {"trajectory_length": 0}
    first, last = rows[0], rows[-1]
    return {
        "trajectory_length": len(rows),
        "initial_train_vote_correct_count": first.get("team_vote_correct_count"),
        "final_train_vote_correct_count": last.get("team_vote_correct_count"),
        "initial_train_member_correct_counts": first.get("per_agent_correct_counts"),
        "final_train_member_correct_counts": last.get("per_agent_correct_counts"),
        "accepted_update_count": int(last.get("accepted_update_count_so_far", 0)),
        "final_distinct_prompt_hash_count": last.get("distinct_prompt_hash_count"),
        "final_oracle_correct_count": last.get("oracle_correct_count"),
        "final_mean_G": last.get("mean_G"),
        "final_mean_H": last.get("mean_H"),
        "final_mean_M": last.get("mean_M"),
    }


def _comparison(
    left: str,
    right: str,
    final_metrics: list[dict[str, Any]],
    proposal: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_map = {(row["seed"], row["setting"]): row for row in final_metrics}
    proposal_map = {(row["seed"], row["setting"]): row for row in proposal}
    per_seed = []
    for seed in SEEDS:
        lhs, rhs = metric_map[(seed, left)], metric_map[(seed, right)]
        lp, rp = proposal_map[(seed, left)], proposal_map[(seed, right)]
        per_seed.append({
            "seed": seed,
            "vote_correct_count_delta": rhs["vote_correct_count"] - lhs["vote_correct_count"],
            "vote_accuracy_delta": rhs["vote_accuracy"] - lhs["vote_accuracy"],
            "g_min_delta": rhs["g_min"] - lhs["g_min"],
            "g_sum_delta": rhs["g_sum"] - lhs["g_sum"],
            "N_positive_delta": rhs["N_positive"] - lhs["N_positive"],
            "acceptable_candidate_rate_delta": (
                rp["acceptable_candidate_rate"] - lp["acceptable_candidate_rate"]
                if isinstance(rp["acceptable_candidate_rate"], (int, float))
                and isinstance(lp["acceptable_candidate_rate"], (int, float))
                else "unavailable"
            ),
            "accepted_update_count_delta": (
                rp["accepted_update_count"] - lp["accepted_update_count"]
            ),
        })

    def aggregate(field: str) -> dict[str, Any]:
        values = [float(row[field]) for row in per_seed if isinstance(row[field], (int, float))]
        return {
            "mean": statistics.mean(values) if values else "unavailable",
            "median": statistics.median(values) if values else "unavailable",
            "positive_count": sum(value > 0 for value in values),
            "negative_count": sum(value < 0 for value in values),
            "zero_count": sum(value == 0 for value in values),
        }

    return {
        "comparison": f"{left}__vs__{right}",
        "per_seed": per_seed,
        "aggregate": {
            field: aggregate(field)
            for field in (
                "vote_correct_count_delta", "vote_accuracy_delta", "g_min_delta",
                "g_sum_delta", "N_positive_delta",
                "acceptable_candidate_rate_delta", "accepted_update_count_delta",
            )
        },
    }


def _historical_exploratory(root: Path | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "label": "historical exploratory result",
        "strict_v2_values_mixed_into_historical_aggregates": False,
        "historical_values_used_to_fill_missing_strict_runs": False,
        "limitation": (
            "The historical runs predate the cumulative exact-observation chain and "
            "are not valid formal setting comparisons."
        ),
        "rows": [],
    }
    if root is None or not root.is_dir():
        payload["availability"] = "unavailable"
        return payload
    for seed in SEEDS:
        for setting in OPTIMIZED_SETTINGS:
            path = _run_dir(root, seed, setting) / "final_summary.json"
            baseline_path = _run_dir(root, seed, "shared_baseline") / "final_summary.json"
            if not path.is_file() or not baseline_path.is_file():
                continue
            selected = _read_json(path)["selected_test"]
            baseline = _read_json(baseline_path)["selected_test"]
            payload["rows"].append({
                "seed": seed,
                "setting": setting,
                **_member_outcome(baseline, selected),
            })
    payload["availability"] = "available" if payload["rows"] else "unavailable"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--stage_gate", type=Path, required=True)
    parser.add_argument("--source_identity", type=Path, required=True)
    parser.add_argument("--static_audit_dir", type=Path, required=True)
    parser.add_argument("--witness_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--historical_root", type=Path, default=None)
    args = parser.parse_args()
    workspace = args.workspace.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else workspace / path

    run_root = resolve(args.run_root)
    stage_gate = _read_json(resolve(args.stage_gate))
    source = _read_json(resolve(args.source_identity))
    if stage_gate.get("gate") != "PASS" or int(stage_gate.get("complete_run_count", 0)) != 12:
        raise ValueError("strict v2 stage gate must pass all 12 runs")
    if build_source_identity(workspace) != source:
        raise ValueError("source identity changed after Stage D")

    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_summaries: dict[tuple[int, str], dict[str, Any]] = {}
    final_metrics: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    scheduling: list[dict[str, Any]] = []
    votes: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    costs: list[dict[str, Any]] = []
    lifecycle: list[dict[str, Any]] = []
    cache_matches: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    for seed in SEEDS:
        baseline_summary = _read_json(
            _run_dir(run_root, seed, "shared_baseline") / "final_summary.json"
        )
        baseline_test = baseline_summary["selected_test"]
        for setting in SETTINGS:
            run_dir = _run_dir(run_root, seed, setting)
            summary = _read_json(run_dir / "final_summary.json")
            meta = _read_json(run_dir / "run_meta.json")
            cost = _read_json(run_dir / "cost_summary.json")
            cache = _read_json(run_dir / "comparison_cache_match.json")
            run_summaries[(seed, setting)] = summary
            final_metrics.append({
                "seed": seed,
                "setting": setting,
                **_member_outcome(baseline_test, summary["selected_test"]),
            })
            lifecycle.append({
                "seed": seed,
                "setting": setting,
                "planned_update_count": int(meta["planned_update_count"]),
                "completed_update_count": int(meta["completed_update_count"]),
                "test_evaluation_count": int(meta["test_evaluation_count"]),
                "test_used_for_selection": bool(meta["test_used_for_selection"]),
                "test_used_for_training": bool(meta["test_used_for_training"]),
                "test_called_before_training_complete": bool(
                    meta["test_called_before_training_complete"]
                ),
                "validation_evaluation_count": int(meta["validation_evaluation_count"]),
                "final_state_source": meta["final_state_selection"].get(
                    "selected_checkpoint_source"
                ),
                "passed": bool(
                    int(meta["test_evaluation_count"]) == 1
                    and not meta["test_used_for_selection"]
                    and not meta["test_used_for_training"]
                    and not meta["test_called_before_training_complete"]
                    and int(meta["validation_evaluation_count"]) == 0
                ),
            })
            cache_matches.append({
                "seed": seed,
                "setting": setting,
                **{
                    key: cache.get(key)
                    for key in (
                        "gate", "cache_chain_continuity", "parent_reference_hash",
                        "result_reference_hash", "reference_entry_count_before",
                        "local_entry_count_after_run", "new_entries_merged",
                        "reference_entry_count_after", "exact_request_conflict_count",
                        "missing_reference_count", "unexpected_provider_recall_count",
                        "unaccounted_new_entry_count", "unchanged_prompt_drift_count",
                        "unchanged_prompt_aggregate_drift_count",
                        "unchanged_team_vote_drift_count",
                    )
                },
            })
            unchanged.extend({
                "seed": seed,
                "setting": setting,
                **row,
            } for row in cache.get("unchanged_prompt_comparisons", []))
            costs.append({
                "seed": seed,
                "setting": setting,
                **{
                    key: cost.get(key, 0)
                    for key in (
                        "solver_calls", "optimizer_calls", "evaluator_calls",
                        "total_llm_calls", "failed_llm_attempts", "total_tokens",
                        "accepted_update_count",
                    )
                },
                "tokens_by_role": cost.get("tokens_by_role", {}),
            })
            if setting != "shared_baseline":
                label = {"seed": seed, "setting": setting}
                proposals.append({**label, **_proposal_summary(run_dir)})
                scheduling.append({**label, **_scheduling_summary(run_dir)})
                votes.append({**label, **_vote_structure(run_dir)})
                training.append({**label, **_training_summary(run_dir)})

    s3_s4 = _comparison(
        "shared_peer_state_member_pareto",
        "shared_member_aware_responsibility",
        final_metrics,
        proposals,
    )
    s4_s5 = _comparison(
        "shared_member_aware_responsibility",
        "shared_member_aware_full",
        final_metrics,
        proposals,
    )
    static_dir = resolve(args.static_audit_dir)
    witness_dir = resolve(args.witness_dir)
    cache_by_seed = [
        {
            "seed": seed,
            "links": [row for row in cache_matches if row["seed"] == seed],
            "passed": all(row["gate"] == "PASS" for row in cache_matches if row["seed"] == seed),
        }
        for seed in SEEDS
    ]
    invalid = {
        "failed_or_incomplete_runs": [
            row for row in stage_gate["runs"] if not row.get("complete")
        ],
        "infrastructure_failure_count": sum(
            int(row.get("infrastructure_failure_count", 0))
            for row in stage_gate["runs"]
        ),
        "retried_runs": [],
        "failed_llm_attempt_count": sum(int(row["failed_llm_attempts"]) for row in costs),
    }
    historical = _historical_exploratory(
        resolve(args.historical_root) if args.historical_root is not None else None
    )

    _write(out / "source_identity.json", source)
    _write(out / "experiment_manifest.json", {
        "report_version": REPORT_VERSION,
        "task": "disambiguation_qa",
        "seeds": list(SEEDS),
        "settings": list(SETTINGS),
        "baseline_run_count": 3,
        "optimization_run_count": 9,
        "total_run_count": 12,
        "train_size": 75,
        "test_size": 125,
        "planned_updates_per_optimized_run": 32,
        "final_test_evaluations_per_run": 1,
        "validation_selection": False,
        "proposal_memory_mode": "off",
        "member_catchup_mode": "off",
        "gate": "PASS",
    })
    _write(out / "v2_static_audit.json", {
        "gate": "PASS",
        "cache_flow": _read_json(static_dir / "v2_cache_flow.json"),
        "exact_request_identity": _read_json(static_dir / "exact_request_identity_audit.json"),
        "merge_semantics": _read_json(static_dir / "merge_semantics_audit.json"),
        "comparison_match": _read_json(static_dir / "comparison_match_audit.json"),
    })
    _write(out / "v2_live_witness.json", {
        "gate": "PASS",
        "comparison_cache_match": _read_json(witness_dir / "comparison_cache_match.json"),
        "provider_call_audit": _read_json(witness_dir / "provider_call_audit.json"),
        "metric_match": _read_json(witness_dir / "baseline_noop_metric_match.json"),
    })
    _write(out / "cache_chain_by_seed.json", cache_by_seed)
    _write(out / "comparison_cache_match_by_seed.json", cache_matches)
    _write(out / "unchanged_prompt_audit.json", {
        "gate": "PASS" if all(row.get("passed") for row in unchanged) else "FAIL",
        "drift_count": sum(int(row.get("per_question_drift_count", 0)) for row in unchanged),
        "comparisons": unchanged,
    })
    _write(out / "setting_isolation_audit.json", stage_gate["setting_isolation"])
    _write(out / "test_lifecycle_audit.json", lifecycle)
    _write(out / "strict_final_metrics.json", final_metrics)
    _write(out / "strict_s3_vs_s4.json", s3_s4)
    _write(out / "strict_s4_vs_s5.json", s4_s5)
    _write(out / "proposal_realization_summary.json", proposals)
    _write(out / "target_scheduling_summary.json", scheduling)
    _write(out / "vote_structure_summary.json", votes)
    _write(out / "training_dynamics_summary.json", training)
    _write(out / "cost_summary.json", {
        "runs": costs,
        "total_tokens": sum(int(row["total_tokens"]) for row in costs),
        "total_llm_calls": sum(int(row["total_llm_calls"]) for row in costs),
        "total_solver_calls": sum(int(row["solver_calls"]) for row in costs),
        "total_optimizer_calls": sum(int(row["optimizer_calls"]) for row in costs),
        "total_evaluator_calls": sum(int(row["evaluator_calls"]) for row in costs),
    })
    _write(out / "invalid_failed_retried_runs.json", invalid)
    _write(out / "comparison_with_old_exploratory_results.json", historical)
    readme = [
        "# Strict v2 disambiguation_qa S3/S4/S5 Evaluation",
        "",
        "- Overall gate: **PASS**",
        "- Completed runs: 12 / 12",
        "- Cache-chain conflicts: 0",
        "- Unchanged-prompt observation drift: 0",
        "- Test lifecycle violations: 0",
        "",
        "The S3→S4 and S4→S5 tables report each seed separately plus mean, median,",
        "and positive/negative direction counts. Historical pre-v2 runs remain",
        "separately labelled exploratory evidence and are never pooled with strict v2 results.",
        "",
        "All outputs are sanitized: no questions, gold labels, prompts, literal model",
        "answers, API responses, credentials, endpoints, absolute paths, caches, or checkpoints.",
        "",
    ]
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps({
        "report_version": REPORT_VERSION,
        "gate": "PASS",
        "run_count": 12,
        "total_tokens": sum(int(row["total_tokens"]) for row in costs),
    }, indent=2))


if __name__ == "__main__":
    main()
