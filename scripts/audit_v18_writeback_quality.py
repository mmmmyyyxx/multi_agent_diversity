from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARMS = ("W1_TOP2", "HYBRID_BASE")
SEEDS = (59, 60, 61)
SIGNALS = (
    "train_vote_loss_positive",
    "train_pivotal_loss_positive",
    "train_unique_loss_positive",
    "train_coverage_loss_positive",
    "train_vote_gain_and_loss_cooccur",
    "train_soft_utility_negative",
    "train_target_only_progress",
    "generic_revision_stage",
    "assigned_residual_repair_positive",
)
ALLOWED_PRIVATE_INPUTS = (
    "candidate_decisions.jsonl",
    "dual_target_commit_decisions.jsonl",
    "online_compatibility_repair_events.jsonl",
    "experimental_module2_context_diagnostics.jsonl",
    "online_run_summary.json",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "NA" if value is None else value for key, value in row.items()})


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _common_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    evaluation = candidate["evaluation"]
    return (
        int(evaluation["team_outcome"]["vote_correct_count"]),
        int(evaluation["competence"]["correct_count"]),
        float(evaluation["team_outcome"]["mean_soft_vote_utility"]),
        -int(evaluation["marginal"]["vote_loss_count"]),
        -int(evaluation["competence"]["invalid_count"]),
        -int(candidate["generation"]),
        str(candidate["prompt_hash"]),
    )


def _signals(row: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "train_vote_loss_positive": int(row["train_vote_loss_count"]) > 0,
        "train_pivotal_loss_positive": int(row["train_pivotal_loss_count"]) > 0,
        "train_unique_loss_positive": int(row["train_unique_loss_count"]) > 0,
        "train_coverage_loss_positive": int(row["train_coverage_loss_count"]) > 0,
        "train_vote_gain_and_loss_cooccur": (
            int(row["train_vote_gain_count"]) > 0 and int(row["train_vote_loss_count"]) > 0
        ),
        "train_soft_utility_negative": float(row["train_soft_utility_delta"]) < 0,
        "train_target_only_progress": (
            int(row["train_target_gain"]) > 0 and int(row["train_vote_net_gain"]) == 0
        ),
        "generic_revision_stage": row["candidate_stage"] == "loss_blind_generic_revision",
        "assigned_residual_repair_positive": int(row["assigned_residual_repair_count"]) > 0,
    }


def extract_committed_candidate(
    *, seed: int, arm: str, decision: dict[str, Any], commit: dict[str, Any],
    validation: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    committed_hash = str(decision["accepted_prompt_hash"])
    candidates = [row for row in decision["candidates"] if row.get("evaluation") and row.get("constraint")]
    feasible = [row for row in candidates if bool(row["constraint"]["passed"])]
    selected = next((row for row in feasible if row["prompt_hash"] == committed_hash), None)
    if selected is None:
        raise ValueError("committed candidate missing from feasible pool")
    branch_groups: dict[int, list[dict[str, Any]]] = {}
    for candidate in feasible:
        branch_groups.setdefault(int(candidate["target_agent_id"]), []).append(candidate)
    for branch in decision["branches"]:
        winner_hash = str(branch.get("branch_winner_hash", ""))
        group = branch_groups.get(int(branch["target_agent_id"]), [])
        expected = max(group, key=_common_key, default=None)
        if winner_hash:
            if expected is None or expected["prompt_hash"] != winner_hash:
                raise ValueError("persisted branch ranking mismatch")
        elif expected is not None:
            raise ValueError("missing persisted branch winner")
    if str(commit["committed_prompt_hash"]) != committed_hash:
        raise ValueError("commit decision hash mismatch")
    constraint = selected["constraint"]
    evaluation = selected["evaluation"]
    min_loss = min(int(row["constraint"]["vote_loss_count"]) for row in feasible)
    lower_loss = sum(
        int(row["constraint"]["vote_loss_count"]) < int(constraint["vote_loss_count"])
        for row in feasible
    )
    row = {
        "seed": seed,
        "arm": arm,
        "update_index": int(decision["update_index"]),
        "committed_candidate_hash": committed_hash,
        "committed_target": int(commit["committed_target_id"]),
        "candidate_stage": str(selected["candidate_stage"]),
        "common_safe_passed": bool(constraint["passed"]),
        "target_nonregression_passed": bool(constraint["target_nonregression_passed"]),
        "team_vote_nonregression_passed": bool(constraint["team_vote_nonregression_passed"]),
        "target_or_vote_progress_passed": bool(constraint["target_or_vote_progress_passed"]),
        "terminal_invalid_nonregression_passed": bool(constraint["terminal_invalid_nonregression_passed"]),
        "train_target_gain": int(constraint["target_gain"]),
        "train_vote_gain_count": int(constraint["vote_gain_count"]),
        "train_vote_loss_count": int(constraint["vote_loss_count"]),
        "train_vote_net_gain": int(constraint["vote_net_gain"]),
        "train_pivotal_loss_count": int(constraint["pivotal_correct_loss_count"]),
        "train_unique_loss_count": int(constraint["unique_correct_loss_count"]),
        "train_coverage_loss_count": int(evaluation["marginal"]["coverage_loss_count"]),
        "train_soft_utility_delta": float(evaluation["marginal"]["soft_utility_delta"]),
        "assigned_residual_repair_count": int(evaluation["marginal"]["assigned_residual_repair_count"]),
        "validation_gain_count": int(validation["validation_gain_count"]),
        "validation_loss_count": int(validation["validation_loss_count"]),
        "validation_net_delta": int(validation["validation_net_delta"]),
        "validation_outcome_class": (
            "negative" if int(validation["validation_net_delta"]) < 0
            else "positive" if int(validation["validation_net_delta"]) > 0
            else "neutral"
        ),
        "validation_gain_bearing": int(validation["validation_gain_count"]) > 0,
        "validation_collateral_loss_bearing": int(validation["validation_loss_count"]) > 0,
        "feasible_candidate_count": len(feasible),
        "branch_winner_count": len(commit["branch_winner_hashes"]),
        "zero_train_vote_loss_feasible_count": sum(
            int(candidate["constraint"]["vote_loss_count"]) == 0 for candidate in feasible
        ),
        "minimum_train_vote_loss_in_pool": min_loss,
        "selected_train_vote_loss_is_pool_minimum": int(constraint["vote_loss_count"]) == min_loss,
        "strictly_lower_vote_loss_feasible_count": lower_loss,
        "current_ranking_verified": True,
    }
    row.update(_signals(row))
    safe_candidates = []
    for candidate in feasible:
        c = candidate["constraint"]
        e = candidate["evaluation"]
        safe_candidates.append({
            "seed": seed,
            "arm": arm,
            "update_index": int(decision["update_index"]),
            "candidate_hash": str(candidate["prompt_hash"]),
            "target_member": int(candidate["target_agent_id"]),
            "candidate_stage": str(candidate["candidate_stage"]),
            "committed": candidate["prompt_hash"] == committed_hash,
            "train_target_gain": int(c["target_gain"]),
            "train_vote_gain_count": int(c["vote_gain_count"]),
            "train_vote_loss_count": int(c["vote_loss_count"]),
            "train_vote_net_gain": int(c["vote_net_gain"]),
            "train_pivotal_loss_count": int(c["pivotal_correct_loss_count"]),
            "train_unique_loss_count": int(c["unique_correct_loss_count"]),
            "train_coverage_loss_count": int(e["marginal"]["coverage_loss_count"]),
            "assigned_residual_repair_count": int(e["marginal"]["assigned_residual_repair_count"]),
            "validation_counterfactual_evaluated": False,
        })
    return row, safe_candidates


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def signal_table(rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    arm_rows = [row for row in rows if row["arm"] == arm]
    harmful = [row for row in arm_rows if row["validation_net_delta"] < 0]
    output = []
    for signal in SIGNALS:
        flagged = [row for row in arm_rows if bool(row[signal])]
        true_positive = sum(int(row["validation_net_delta"]) < 0 for row in flagged)
        validation_loss_bearing = sum(int(row["validation_loss_count"]) > 0 for row in flagged)
        output.append({
            "arm": arm,
            "signal": signal,
            "accepted_commit_count": len(arm_rows),
            "flagged_commit_count": len(flagged),
            "negative_net_commit_count": len(harmful),
            "flagged_negative_net_count": true_positive,
            "flagged_validation_loss_bearing_count": validation_loss_bearing,
            "negative_net_precision": _ratio(true_positive, len(flagged)),
            "negative_net_sensitivity": _ratio(true_positive, len(harmful)),
            "false_positive_count": len(flagged) - true_positive,
        })
    return output


def audit(*, root: Path, trajectory_report: Path, registry: dict[str, Any], out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("fresh output directory required")
    if registry["protocol"]["candidate_acceptance_policy"] != "fixed_peer_monotone_target_or_vote":
        raise ValueError("unexpected Common-Safe policy")
    if registry["protocol"]["candidate_ranking_policy"] != "common_monotone_safe":
        raise ValueError("unexpected ranking policy")
    validation_rows = read_csv(trajectory_report / "accepted_commit_quality.csv")
    validation_by_key = {
        (int(row["seed"]), row["arm"], int(row["update_index"])): row for row in validation_rows
    }
    accepted_rows: list[dict[str, Any]] = []
    feasible_rows: list[dict[str, Any]] = []
    source_hashes: list[dict[str, Any]] = []
    compatibility_event_count = 0
    module2_context_diagnostic_count = 0
    candidate_responsibility_contribution_nonnull_count = 0
    for seed in SEEDS:
        for arm in ARMS:
            run = root / f"seed{seed}" / arm
            for filename in ALLOWED_PRIVATE_INPUTS:
                path = run / filename
                source_hashes.append({
                    "seed": seed, "arm": arm, "artifact_role": filename,
                    "sha256": sha256_file(path),
                })
            decisions = read_jsonl(run / "candidate_decisions.jsonl")
            commits = {int(row["update_index"]): row for row in read_jsonl(run / "dual_target_commit_decisions.jsonl")}
            summary = read_json(run / "online_run_summary.json")
            if int(summary["new_test_calls"]) != 0 or int(summary["infrastructure_failure_count"]) != 0:
                raise ValueError("forbidden test call or infrastructure failure")
            compatibility_event_count += len(read_jsonl(run / "online_compatibility_repair_events.jsonl"))
            module2_context_diagnostic_count += len(read_jsonl(run / "experimental_module2_context_diagnostics.jsonl"))
            for decision in decisions:
                for candidate in decision.get("candidates", []):
                    evaluation = candidate.get("evaluation")
                    if evaluation and evaluation.get("responsibility_contribution") is not None:
                        candidate_responsibility_contribution_nonnull_count += 1
                if not decision.get("accepted_prompt_hash"):
                    continue
                key = (seed, arm, int(decision["update_index"]))
                if key not in validation_by_key:
                    raise ValueError("accepted commit missing validation decomposition")
                row, candidates = extract_committed_candidate(
                    seed=seed, arm=arm, decision=decision,
                    commit=commits[int(decision["update_index"])], validation=validation_by_key[key],
                )
                accepted_rows.append(row)
                feasible_rows.extend(candidates)
    if len(accepted_rows) != 18 or len(validation_by_key) != 18:
        raise ValueError("accepted commit inventory mismatch")
    hybrid = [row for row in accepted_rows if row["arm"] == "HYBRID_BASE"]
    harmful = [row for row in hybrid if row["validation_net_delta"] < 0]
    gain_bearing = [row for row in hybrid if row["validation_gain_count"] > 0]
    positive = [row for row in hybrid if row["validation_net_delta"] > 0]
    if sum(int(row["validation_loss_count"]) for row in hybrid) != 7:
        raise ValueError("Hybrid collateral event count mismatch")
    if sum(int(row["validation_gain_count"]) for row in hybrid) != 5:
        raise ValueError("Hybrid beneficial event count mismatch")
    risk_admission = bool(harmful) and all(
        row["common_safe_passed"]
        and row["target_nonregression_passed"]
        and row["team_vote_nonregression_passed"]
        and row["target_or_vote_progress_passed"]
        and row["terminal_invalid_nonregression_passed"]
        and row["train_vote_loss_positive"]
        for row in harmful
    )
    ranking_misselection = any(
        not row["selected_train_vote_loss_is_pool_minimum"]
        and int(row["strictly_lower_vote_loss_feasible_count"]) > 0
        for row in harmful
    )
    feasible_set_gap = any(int(row["zero_train_vote_loss_feasible_count"]) == 0 for row in harmful)
    compatibility_available = (
        bool(registry["protocol"]["compatibility_repair_enabled"])
        and compatibility_event_count > 0
        and candidate_responsibility_contribution_nonnull_count > 0
    )
    signals = signal_table(accepted_rows, "W1_TOP2") + signal_table(accepted_rows, "HYBRID_BASE")
    hybrid_vote_loss_signal = next(
        row for row in signals if row["arm"] == "HYBRID_BASE" and row["signal"] == "train_vote_loss_positive"
    )
    classifier = {
        "classifier_version": "v18_writeback_quality_classifier_v1",
        "rules_frozen_before_result_readout": True,
        "common_safe_risk_admission_supported": risk_admission,
        "train_side_risk_signal_available": (
            hybrid_vote_loss_signal["negative_net_precision"] == 1.0
            and hybrid_vote_loss_signal["negative_net_sensitivity"] == 1.0
        ),
        "identified_train_side_risk_signal": "train_vote_loss_positive",
        "ranking_misselection_supported": ranking_misselection,
        "feasible_set_quality_gap_supported": feasible_set_gap,
        "m2f_compatibility_signal_available_in_v18": compatibility_available,
        "final_diagnosis": (
            "COMMON_SAFE_FEASIBLE_SET_QUALITY_GAP_WITH_EXISTING_TRAIN_VOTE_LOSS_RISK_SIGNAL"
            if risk_admission and feasible_set_gap and not ranking_misselection
            else "MIXED_OR_INCONCLUSIVE_WRITEBACK_QUALITY_SOURCE"
        ),
    }
    summary = {
        "audit_version": "v18_writeback_quality_diagnostic_v1",
        "scope": {
            "zero_api": True, "new_api_calls": 0, "new_model_calls": 0,
            "new_test_calls": 0, "new_validation_evaluations": 0,
            "method_modified": False, "selector_modified": False,
            "gate_modified": False, "ranking_modified": False,
            "uncommitted_candidate_validation_evaluations": 0,
        },
        "units": {
            "accepted_commit_count": len(accepted_rows),
            "hybrid_accepted_commit_count": len(hybrid),
            "hybrid_validation_collateral_loss_event_count": 7,
            "hybrid_validation_collateral_loss_commit_count": sum(row["validation_loss_count"] > 0 for row in hybrid),
            "hybrid_validation_gain_event_count": 5,
            "hybrid_validation_gain_bearing_commit_count": len(gain_bearing),
            "hybrid_positive_net_commit_count": len(positive),
            "gain_and_loss_bearing_commit_overlap_count": sum(
                row["validation_gain_count"] > 0 and row["validation_loss_count"] > 0 for row in hybrid
            ),
        },
        "harmful_hybrid_commits": [
            {key: row[key] for key in (
                "seed", "update_index", "candidate_stage", "train_target_gain",
                "train_vote_gain_count", "train_vote_loss_count", "train_vote_net_gain",
                "train_pivotal_loss_count", "validation_gain_count", "validation_loss_count",
                "validation_net_delta", "feasible_candidate_count",
                "zero_train_vote_loss_feasible_count", "minimum_train_vote_loss_in_pool",
                "selected_train_vote_loss_is_pool_minimum", "branch_winner_count",
            )}
            for row in harmful
        ],
        "compatibility_evidence": {
            "registry_compatibility_repair_enabled": bool(registry["protocol"]["compatibility_repair_enabled"]),
            "compatibility_event_count": compatibility_event_count,
            "module2_context_diagnostic_count": module2_context_diagnostic_count,
            "candidate_responsibility_contribution_nonnull_count": candidate_responsibility_contribution_nonnull_count,
        },
        "classifier": classifier,
    }
    out.mkdir(parents=True)
    write_csv(out / "accepted_commit_train_evidence.csv", accepted_rows)
    write_csv(out / "feasible_candidate_pool.csv", feasible_rows)
    write_csv(out / "risk_signal_diagnostics.csv", signals)
    write_csv(out / "source_artifact_hashes.csv", source_hashes)
    write_json(out / "classifier.json", classifier)
    write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trajectory_report", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        root=args.root.resolve(), trajectory_report=args.trajectory_report.resolve(),
        registry=read_json(args.registry.resolve()), out=args.out.resolve(),
    )
    print(json.dumps({
        "audit_version": result["audit_version"],
        "units": result["units"],
        "final_diagnosis": result["classifier"]["final_diagnosis"],
        "new_api_calls": 0, "new_test_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
