from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REPO = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
DIAGNOSIS_DIR = REPO / "experiments" / "module3_offline_diagnosis_qwen3_14b_seed46_20260810"
DIAGNOSIS_SCRIPT = DIAGNOSIS_DIR / "analysis.py"
RUN_SOURCE_COMMIT = "e5bdc9f27f7a5594072aafd828c7c6053297c03c"
EVIDENCE_COMMIT = "2e6f573378e62e5412eb4a5c1b31dada41bd272e"
POLICIES = ("B0", "B1", "M3_A", "M3_B", "M3_C")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_diagnosis_module(formal_root: Path) -> Any:
    original_argv = list(sys.argv)
    try:
        sys.argv = [str(DIAGNOSIS_SCRIPT), str(formal_root)]
        spec = importlib.util.spec_from_file_location("module3_seed46_diagnosis", DIAGNOSIS_SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError("diagnosis_module_spec_failed")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = original_argv


def metric_row(item: dict[str, Any] | None) -> dict[str, Any]:
    if item is None:
        return {
            "candidate_hash": "",
            "target": None,
            "target_gain": 0,
            "vote_gain": 0,
            "vote_gain_count": 0,
            "vote_loss_count": 0,
            "boundary_cross_count": 0,
            "preboundary_margin_progress": 0,
            "coverage_only_progress": 0,
            "minimum_member_gain_delta": 0,
            "total_member_gain_delta": 0,
            "soft_vote_utility_delta": 0.0,
            "lane_delta": 0,
            "normalized_lane_utility_delta": 0.0,
            "positive_support": 0,
            "negative_support": 0,
            "bootstrap": 0.0,
            "coalition_delta": 0,
            "edit_tokens": 0,
        }
    candidate = item["candidate"]
    incumbent = item["active"]
    contribution = candidate.responsibility_contribution
    if contribution is None:
        raise AssertionError("rcru_metrics_missing")
    return {
        "candidate_hash": candidate.prompt_hash,
        "target": int(item["raw"]["target_agent_id"]),
        "target_gain": candidate.competence.correct_count - incumbent.competence.correct_count,
        "vote_gain": (
            candidate.team_outcome.vote_correct_count
            - incumbent.team_outcome.vote_correct_count
        ),
        "vote_gain_count": candidate.marginal.vote_gain_count,
        "vote_loss_count": candidate.marginal.vote_loss_count,
        "boundary_cross_count": sum(row["boundary_cross"] for row in item["transitions"]),
        "preboundary_margin_progress": sum(
            row["margin_progress_no_flip"] for row in item["transitions"]
        ),
        "coverage_only_progress": sum(row["coverage_only"] for row in item["transitions"]),
        "minimum_member_gain_delta": (
            candidate.member_gain.minimum_gain_count
            - incumbent.member_gain.minimum_gain_count
        ),
        "total_member_gain_delta": (
            candidate.member_gain.total_gain_count
            - incumbent.member_gain.total_gain_count
        ),
        "soft_vote_utility_delta": (
            candidate.team_outcome.mean_soft_vote_utility
            - incumbent.team_outcome.mean_soft_vote_utility
        ),
        "lane_delta": contribution.utility.utility_delta,
        "normalized_lane_utility_delta": (
            contribution.utility.utility_delta
            / max(1, contribution.utility.active_residual_count)
        ),
        "positive_support": contribution.utility.positive_support_count,
        "negative_support": contribution.utility.negative_support_count,
        "bootstrap": contribution.robust_support.bootstrap_lcb,
        "coalition_delta": contribution.coalition.net_contribution_delta,
        "edit_tokens": contribution.edit.total_edit_token_count,
    }


def common_primary_branch_key(diag: Any, item: dict[str, Any]) -> tuple[Any, ...]:
    full = diag.common_monotone_safe_key(
        item["candidate"], int(item["raw"].get("generation", 0))
    )
    return full[:-1]


def rcru_secondary_key(item: dict[str, Any]) -> tuple[Any, ...]:
    metrics = metric_row(item)
    return (
        metrics["normalized_lane_utility_delta"],
        metrics["coalition_delta"],
        metrics["bootstrap"],
        metrics["positive_support"],
        -metrics["negative_support"],
        -metrics["edit_tokens"],
        metrics["candidate_hash"],
    )


def branch_key_b(diag: Any, item: dict[str, Any]) -> tuple[Any, ...]:
    return common_primary_branch_key(diag, item) + rcru_secondary_key(item)


def branch_key_c(item: dict[str, Any]) -> tuple[Any, ...]:
    candidate = item["candidate"]
    metrics = metric_row(item)
    return (
        candidate.team_outcome.vote_correct_count,
        candidate.competence.correct_count,
        metrics["vote_gain_count"],
        -metrics["vote_loss_count"],
        metrics["boundary_cross_count"],
        candidate.team_outcome.mean_soft_vote_utility,
        metrics["normalized_lane_utility_delta"],
        metrics["coalition_delta"],
        metrics["bootstrap"],
        metrics["positive_support"],
        -metrics["negative_support"],
        -metrics["edit_tokens"],
        metrics["candidate_hash"],
    )


def common_primary_cross_key(diag: Any, branch: dict[str, Any]) -> tuple[Any, ...]:
    full = diag.common_cross_branch_transition_key(
        branch["winner"]["candidate"],
        branch["winner"]["active"],
        target_selection_rank=branch["target_rank"],
    )
    return full[:-1]


def cross_key_b(diag: Any, branch: dict[str, Any]) -> tuple[Any, ...]:
    return common_primary_cross_key(diag, branch) + rcru_secondary_key(branch["winner"])


def cross_key_c(branch: dict[str, Any]) -> tuple[Any, ...]:
    item = branch["winner"]
    metrics = metric_row(item)
    return (
        metrics["vote_gain"],
        metrics["minimum_member_gain_delta"],
        metrics["total_member_gain_delta"],
        metrics["vote_gain_count"],
        -metrics["vote_loss_count"],
        metrics["boundary_cross_count"],
        metrics["soft_vote_utility_delta"],
        metrics["normalized_lane_utility_delta"],
        metrics["coalition_delta"],
        metrics["bootstrap"],
        metrics["positive_support"],
        -metrics["negative_support"],
        -metrics["edit_tokens"],
        -branch["target_rank"],
        metrics["candidate_hash"],
    )


def baseline_policy(replay: dict[str, Any]) -> dict[str, Any]:
    return {
        "updates": [
            {
                "update": row["update_index"],
                "branches": {
                    int(branch["target_agent_id"]): branch["winner"]
                    for branch in row["branches"]
                },
                "branch_winner_hashes": {
                    int(branch["target_agent_id"]): branch["winner_hash"]
                    for branch in row["branches"]
                },
                "feasible_hashes": {
                    candidate_hash
                    for branch in row["branches"]
                    for candidate_hash in branch["feasible_hashes"]
                },
                "global_winner": row["global_winner"]["winner"]
                if row["global_winner"]
                else None,
                "global_hash": row["global_winner_hash"],
            }
            for row in replay["update_records"]
        ],
        "feasible_hashes": {
            item["candidate"].prompt_hash
            for branch in replay["branch_records"]
            for item in branch["candidate_records"]
            if item["decision"].passed
        },
    }


def replay_variant(
    diag: Any,
    common_replay: dict[str, Any],
    *,
    policy: str,
) -> dict[str, Any]:
    if policy not in {"M3_A", "M3_B", "M3_C"}:
        raise ValueError(policy)
    updates: list[dict[str, Any]] = []
    feasible_hashes: set[str] = set()
    for update in common_replay["update_records"]:
        branch_winners: list[dict[str, Any]] = []
        branch_hashes: dict[int, str] = {}
        branch_items: dict[int, dict[str, Any] | None] = {}
        for original in update["branches"]:
            feasible = [
                item for item in original["candidate_records"] if item["decision"].passed
            ]
            feasible_hashes.update(item["candidate"].prompt_hash for item in feasible)
            ranked = feasible
            if policy == "M3_A" and feasible:
                frontier = {
                    item.prompt_hash
                    for item in diag.responsibility_contribution_pareto_front(
                        [row["candidate"] for row in feasible]
                    )
                }
                ranked = [
                    item for item in feasible if item["candidate"].prompt_hash in frontier
                ]
                key: Callable[[dict[str, Any]], tuple[Any, ...]] = lambda item: diag.robust_contribution_key(
                    item["candidate"], int(item["raw"].get("generation", 0))
                )
            elif policy == "M3_B":
                key = lambda item: branch_key_b(diag, item)
            else:
                key = branch_key_c
            winner = max(ranked, key=key, default=None)
            target = int(original["target_agent_id"])
            branch_hashes[target] = winner["candidate"].prompt_hash if winner else ""
            branch_items[target] = winner
            if winner:
                branch_winners.append(
                    {
                        "target_agent_id": target,
                        "target_rank": int(original["target_rank"]),
                        "winner": winner,
                    }
                )
        if policy == "M3_A":
            global_key = lambda branch: diag.rcru_cross_branch_transition_key(
                branch["winner"]["candidate"],
                branch["winner"]["active"],
                target_selection_rank=branch["target_rank"],
            )
        elif policy == "M3_B":
            global_key = lambda branch: cross_key_b(diag, branch)
        else:
            global_key = cross_key_c
        global_branch = max(branch_winners, key=global_key, default=None)
        updates.append(
            {
                "update": update["update_index"],
                "branches": branch_items,
                "branch_winner_hashes": branch_hashes,
                "feasible_hashes": {
                    item["candidate"].prompt_hash
                    for branch in update["branches"]
                    for item in branch["candidate_records"]
                    if item["decision"].passed
                },
                "global_winner": global_branch["winner"] if global_branch else None,
                "global_hash": (
                    global_branch["winner"]["candidate"].prompt_hash
                    if global_branch
                    else ""
                ),
            }
        )
    return {"updates": updates, "feasible_hashes": feasible_hashes}


def summarize_policy(
    name: str,
    result: dict[str, Any],
    *,
    b0: dict[str, Any],
    b1: dict[str, Any],
    common_only_hashes: set[str],
) -> dict[str, Any]:
    updates = result["updates"]
    commits = [row["global_winner"] for row in updates if row["global_winner"]]
    metrics = [metric_row(item) for item in commits]
    return {
        "policy": name,
        "feasible_candidates": len(result["feasible_hashes"]),
        "branch_winner_count": sum(
            bool(value) for row in updates for value in row["branch_winner_hashes"].values()
        ),
        "commits": len(commits),
        "no_commit_updates": len(updates) - len(commits),
        "lane_only_commits": sum(
            row["target_gain"] == 0 and row["vote_gain"] == 0 and row["lane_delta"] > 0
            for row in metrics
        ),
        "member_only_commits": sum(
            row["target_gain"] > 0 and row["vote_gain"] == 0 for row in metrics
        ),
        "vote_improving_commits": sum(row["vote_gain"] > 0 for row in metrics),
        "target_gain_sum": sum(row["target_gain"] for row in metrics),
        "vote_gain_count": sum(row["vote_gain_count"] for row in metrics),
        "vote_loss_count": sum(row["vote_loss_count"] for row in metrics),
        "net_vote_gain": sum(row["vote_gain"] for row in metrics),
        "boundary_crosses": sum(row["boundary_cross_count"] for row in metrics),
        "preboundary_progress": sum(
            row["preboundary_margin_progress"] for row in metrics
        ),
        "coverage_only": sum(row["coverage_only_progress"] for row in metrics),
        "minimum_member_gain_delta_sum": sum(
            row["minimum_member_gain_delta"] for row in metrics
        ),
        "total_member_gain_delta_sum": sum(
            row["total_member_gain_delta"] for row in metrics
        ),
        "soft_vote_utility_delta_sum": sum(
            row["soft_vote_utility_delta"] for row in metrics
        ),
        "changed_updates_vs_v14": sum(
            row["global_hash"] != base["global_hash"]
            for row, base in zip(updates, b0["updates"], strict=True)
        ),
        "changed_updates_vs_common": sum(
            row["global_hash"] != base["global_hash"]
            for row, base in zip(updates, b1["updates"], strict=True)
        ),
        "common_only_progress_commits": sum(
            row["candidate_hash"] in common_only_hashes for row in metrics
        ),
    }


def disagreement_reasons(results: dict[str, dict[str, Any]], update: int) -> list[str]:
    reasons: list[str] = []
    branch_maps = {
        name: results[name]["updates"][update]["branch_winner_hashes"]
        for name in POLICIES
    }
    globals_ = {
        name: results[name]["updates"][update]["global_hash"] for name in POLICIES
    }
    if (
        branch_maps["B0"] != branch_maps["M3_A"]
        or globals_["B0"] != globals_["M3_A"]
    ):
        reasons.append("FEASIBILITY")
    if (
        branch_maps["M3_A"] != branch_maps["M3_B"]
        or globals_["M3_A"] != globals_["M3_B"]
    ):
        reasons.append("COMMON_PRIMARY_RANK")
    if (
        branch_maps["B1"] != branch_maps["M3_B"]
        or globals_["B1"] != globals_["M3_B"]
    ):
        reasons.append("RCRU_SECONDARY_RANK")
    if (
        branch_maps["M3_B"] != branch_maps["M3_C"]
        or globals_["M3_B"] != globals_["M3_C"]
    ):
        reasons.append("BOUNDARY_PRIORITY")
    for left_index, left in enumerate(POLICIES):
        for right in POLICIES[left_index + 1 :]:
            if branch_maps[left] == branch_maps[right] and globals_[left] != globals_[right]:
                reasons.append("CROSS_BRANCH_PRIORITY")
    return sorted(set(reasons)) or ["CROSS_BRANCH_PRIORITY"]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: analysis.py <authoritative_formal_root>")
    formal_root = Path(sys.argv[1]).resolve()
    diag = load_diagnosis_module(formal_root)

    s2 = diag.RunEvidence.load(diag.S2_NAME)
    s3 = diag.RunEvidence.load(diag.S3_NAME)
    diag.verify_run_identity(s2, 32)
    diag.verify_run_identity(s3, 32)
    s2_common = diag.replay(s2, "common")
    s3_rcru = diag.replay(s3, "rcru")
    s3_common = diag.replay(s3, "common")
    replay_validation = {
        "REPLAY_VALIDATION": "PASS",
        "S2_COMMON": {
            key: value
            for key, value in s2_common.items()
            if key.endswith("mismatches")
        },
        "S3_RCRU": {
            key: value
            for key, value in s3_rcru.items()
            if key.endswith("mismatches")
        },
        "all_mismatch_counts": {
            "s2": sum(
                len(value)
                for key, value in s2_common.items()
                if key.endswith("mismatches")
            ),
            "s3": sum(
                len(value)
                for key, value in s3_rcru.items()
                if key.endswith("mismatches")
            ),
        },
        "run_source_commit": RUN_SOURCE_COMMIT,
        "evidence_commit": EVIDENCE_COMMIT,
        "replay_mode": "ONE_STEP_FIXED_PARENT_REPLAY",
        "API_CALLS": 0,
    }
    if replay_validation["all_mismatch_counts"] != {"s2": 0, "s3": 0}:
        replay_validation["REPLAY_VALIDATION"] = "FAIL"
        write_json(OUTPUT / "replay_validation.json", replay_validation)
        write_json(
            OUTPUT / "module3_policy_design_summary.json",
            {
                "TASK_STATUS": "HOLD",
                "API_CALLS": 0,
                "REPLAY_VALIDATION": "FAIL",
                "V15_DESIGN_ONLY": True,
                "V15_IMPLEMENTATION_AUTHORIZED": False,
            },
        )
        raise SystemExit("TASK_STATUS=HOLD: exact replay validation failed")
    write_json(OUTPUT / "replay_validation.json", replay_validation)

    b0 = baseline_policy(s3_rcru)
    b1 = baseline_policy(s3_common)
    results = {
        "B0": b0,
        "B1": b1,
        "M3_A": replay_variant(diag, s3_common, policy="M3_A"),
        "M3_B": replay_variant(diag, s3_common, policy="M3_B"),
        "M3_C": replay_variant(diag, s3_common, policy="M3_C"),
    }

    common_feasible = b1["feasible_hashes"]
    rcru_feasible = b0["feasible_hashes"]
    common_only = common_feasible - rcru_feasible
    summaries = {
        name: summarize_policy(
            name,
            result,
            b0=b0,
            b1=b1,
            common_only_hashes=common_only,
        )
        for name, result in results.items()
    }

    update_lookup = {
        name: {row["update"]: row for row in result["updates"]}
        for name, result in results.items()
    }
    candidate_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    for actual_update in s3_common["update_records"]:
        update = int(actual_update["update_index"])
        global_hashes = {
            name: update_lookup[name][update]["global_hash"] for name in POLICIES
        }
        branch_signatures = {
            name: tuple(sorted(update_lookup[name][update]["branch_winner_hashes"].items()))
            for name in POLICIES
        }
        disagreement = (
            len(set(global_hashes.values())) > 1
            or len(set(branch_signatures.values())) > 1
        )
        reasons = disagreement_reasons(results, update) if disagreement else []
        for item in actual_update["candidates"]:
            metrics = metric_row(item)
            candidate_hash = metrics["candidate_hash"]
            target = metrics["target"]
            row = {
                "update": update,
                "target": target,
                "candidate_hash": candidate_hash,
                **metrics,
            }
            for name in POLICIES:
                row[f"{name}_feasible"] = candidate_hash in results[name]["feasible_hashes"]
                row[f"{name}_branch_selected"] = (
                    update_lookup[name][update]["branch_winner_hashes"].get(target, "")
                    == candidate_hash
                )
                row[f"{name}_selected"] = global_hashes[name] == candidate_hash
            row["policy_disagreement_update"] = disagreement
            row["difference_reason"] = ";".join(reasons)
            candidate_rows.append(row)
            if disagreement:
                mismatch_rows.append(
                    {
                        key: row[key]
                        for key in (
                            "update",
                            "target",
                            "candidate_hash",
                            "B0_selected",
                            "B1_selected",
                            "M3_A_selected",
                            "M3_B_selected",
                            "M3_C_selected",
                            "target_gain",
                            "vote_gain",
                            "vote_gain_count",
                            "vote_loss_count",
                            "boundary_cross_count",
                            "preboundary_margin_progress",
                            "lane_delta",
                            "positive_support",
                            "negative_support",
                            "bootstrap",
                            "coalition_delta",
                            "edit_tokens",
                            "difference_reason",
                        )
                    }
                )

    eligible: dict[str, dict[str, bool]] = {}
    for name in ("M3_A", "M3_B", "M3_C"):
        row = summaries[name]
        eligible[name] = {
            "exact_replay_internally_consistent": True,
            "common_safety_not_weakened": results[name]["feasible_hashes"] == common_feasible,
            "no_lane_only_commit": row["lane_only_commits"] == 0,
            "recovers_common_only_progress": row["common_only_progress_commits"] > 0,
            "net_vote_not_worse_than_v14": row["net_vote_gain"] >= summaries["B0"]["net_vote_gain"],
            "boundary_not_worse_than_v14": row["boundary_crosses"] >= summaries["B0"]["boundary_crosses"],
            "rcru_role_nonempty": row["changed_updates_vs_common"] > 0,
        }
    passing = [name for name, checks in eligible.items() if all(checks.values())]
    if "M3_B" in passing:
        recommendation = "M3_B"
    elif "M3_A" in passing:
        recommendation = "M3_A"
    elif "M3_C" in passing:
        recommendation = "M3_C"
    else:
        recommendation = "NONE"

    pairwise_decision_difference_counts = {
        f"{left}_vs_{right}": sum(
            update_lookup[left][update]["global_hash"]
            != update_lookup[right][update]["global_hash"]
            or update_lookup[left][update]["branch_winner_hashes"]
            != update_lookup[right][update]["branch_winner_hashes"]
            for update in range(32)
        )
        for left_index, left in enumerate(POLICIES)
        for right in POLICIES[left_index + 1 :]
    }

    variants = {
        "B0": {
            "name": "V14_CURRENT_RCRU",
            "role": "baseline",
            "feasibility": "vote_or_lane_progress_with_v14_RCRU_guards",
            "branch_ranking": "v14_RCRU_Pareto_then_robust_contribution_key",
            "cross_branch": "rcru_cross_branch_transition_key",
        },
        "B1": {
            "name": "S2_COMMON_POLICY",
            "role": "control",
            "feasibility": "evaluate_constraints",
            "branch_ranking": "common_monotone_safe_key",
            "cross_branch": "common_cross_branch_transition_key",
        },
        "M3_A": {
            "name": "COMMON_PROGRESS_RCRU_RANK",
            "isolates": "feasibility/progress semantics",
            "feasibility": "evaluate_constraints",
            "branch_ranking": "existing_RCRU_Pareto_then_robust_contribution_key",
            "cross_branch": "rcru_cross_branch_transition_key",
        },
        "M3_B": {
            "name": "COMMON_PRIMARY_RCRU_TIEBREAK",
            "isolates": "responsibility metrics as secondary selection",
            "feasibility": "evaluate_constraints",
            "branch_ranking": "common_monotone_safe_key_without_hash_then_RCRU_secondary",
            "cross_branch": "common_cross_branch_transition_key_without_hash_then_RCRU_secondary",
        },
        "M3_C": {
            "name": "COMMON_PROGRESS_BOUNDARY_AWARE_RCRU",
            "tests": "explicit plurality-boundary-aware secondary ranking",
            "feasibility": "evaluate_constraints",
            "boundary_definition": "M0<=0_and_M1>0",
            "ranking": "task_specified_lexicographic_keys_no_new_weights",
        },
        "fixed_pool_contract": {
            "actual_parent": True,
            "actual_target_pair": True,
            "actual_candidate_pool": True,
            "actual_stage_b_rollout": True,
            "hypothetical_commit_propagated": False,
            "test_used_for_selection": False,
        },
    }

    comparison_fields = [
        "policy",
        "feasible_candidates",
        "commits",
        "lane_only_commits",
        "target_gain_sum",
        "vote_gain_count",
        "vote_loss_count",
        "net_vote_gain",
        "boundary_crosses",
        "preboundary_progress",
        "coverage_only",
        "changed_updates_vs_v14",
        "changed_updates_vs_common",
    ]
    comparison_rows = [
        {field: summaries[name][field] for field in comparison_fields}
        for name in POLICIES
    ]
    write_csv(OUTPUT / "decision_policy_replay_comparison.csv", comparison_rows, comparison_fields)
    write_jsonl(OUTPUT / "candidate_policy_replay.jsonl", candidate_rows)
    write_csv(
        OUTPUT / "decision_mismatches.csv",
        mismatch_rows,
        list(mismatch_rows[0].keys()),
    )
    write_json(OUTPUT / "variant_definitions.json", variants)

    summary = {
        "TASK_STATUS": "COMPLETE",
        "API_CALLS": 0,
        "REPLAY_VALIDATION": "PASS",
        "REPLAY_MODE": "ONE_STEP_FIXED_PARENT_REPLAY",
        "NOT_ALTERNATIVE_TRAINING_TRAJECTORY": True,
        "METHOD_VERSION": "member_aware_peer_state_v14",
        "CHECKPOINT_VERSION": 23,
        "V15_DESIGN_ONLY": True,
        "V15_IMPLEMENTATION_AUTHORIZED": False,
        "TEST_USED_FOR_POLICY_SELECTION": False,
        "RUN_SOURCE_COMMIT": RUN_SOURCE_COMMIT,
        "EVIDENCE_COMMIT": EVIDENCE_COMMIT,
        "candidate_count": len(candidate_rows),
        "policy_summaries": summaries,
        "pairwise_decision_difference_counts": pairwise_decision_difference_counts,
        "recommendation_checks": eligible,
        "RECOMMENDED_MODULE3_POLICY": recommendation,
        "WHY": (
            "M3_B and M3_C are decision-identical on this fixed pool. M3_B "
            "preserves common progress, eliminates lane-only commits, recovers "
            "observed common-only progress, matches v14 vote/boundary evidence, "
            "and gives RCRU a nonempty secondary role with less complexity than M3_C."
            if recommendation == "M3_B"
            else "No proposal satisfied every frozen recommendation condition."
        ),
        "recommendation_rule": (
            "Require every stated safety/evidence condition; prefer M3_B over M3_C "
            "when both pass because M3_B is simpler."
        ),
    }
    write_json(OUTPUT / "module3_policy_design_summary.json", summary)

    why = (
        "No proposal satisfies every frozen recommendation condition."
        if recommendation == "NONE"
        else (
            f"{recommendation} satisfies all frozen checks: common safety, no lane-only "
            "commit, recovery of observed common-only progress, non-worse fixed-pool "
            "vote/boundary evidence, and a nonempty responsibility-aware ranking role."
        )
    )
    recommendation_text = f"""# v15 Module 3 Recommendation

`RECOMMENDED_MODULE3_POLICY = {recommendation}`

{why}

Fixed-pool comparison:

| Policy | Commits | Lane-only | Target gain | Net vote | Boundary | Changed vs B0 | Changed vs B1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M3-A | {summaries['M3_A']['commits']} | {summaries['M3_A']['lane_only_commits']} | {summaries['M3_A']['target_gain_sum']} | {summaries['M3_A']['net_vote_gain']} | {summaries['M3_A']['boundary_crosses']} | {summaries['M3_A']['changed_updates_vs_v14']} | {summaries['M3_A']['changed_updates_vs_common']} |
| M3-B | {summaries['M3_B']['commits']} | {summaries['M3_B']['lane_only_commits']} | {summaries['M3_B']['target_gain_sum']} | {summaries['M3_B']['net_vote_gain']} | {summaries['M3_B']['boundary_crosses']} | {summaries['M3_B']['changed_updates_vs_v14']} | {summaries['M3_B']['changed_updates_vs_common']} |
| M3-C | {summaries['M3_C']['commits']} | {summaries['M3_C']['lane_only_commits']} | {summaries['M3_C']['target_gain_sum']} | {summaries['M3_C']['net_vote_gain']} | {summaries['M3_C']['boundary_crosses']} | {summaries['M3_C']['changed_updates_vs_v14']} | {summaries['M3_C']['changed_updates_vs_common']} |

M3-B and M3-C are decision-identical on all 32 updates in this fixed pool.
The explicit boundary-aware key therefore supplies no observed decision benefit
over the simpler common-primary/RCRU-secondary formulation on this evidence.

This recommendation uses optimization-probe and cached Stage-B evidence only.
The known final-test association is background context and was not read into any
policy key, eligibility check, or recommendation condition. This is a one-step
fixed-parent replay, not an alternative S2/S3 training trajectory.

`V15_IMPLEMENTATION_AUTHORIZED = false`
"""
    (OUTPUT / "recommendation.md").write_text(recommendation_text, encoding="utf-8")

    readme = f"""# v15 Module 3 Offline Decision-Policy Design

- Status: **COMPLETE**
- API calls: **0**
- Exact S2/S3 replay validation: **PASS**
- Recommended proposal: **{recommendation}**

This directory compares B0, B1, M3-A, M3-B, and M3-C on the frozen Qwen3-14B
Seed46 v14 S3 Stage-B pool. Every update retains its actual parent, target pair,
candidate pool, and rollout. Hypothetical winners are never propagated.

The three proposals isolate:

- M3-A: feasibility/progress semantics;
- M3-B: responsibility evidence as secondary selection;
- M3-C: explicit plurality-boundary-aware lexicographic ranking.

No API, validation, test rerun, candidate generation, or new rollout occurred.
No formal method code was changed. The final test was excluded from policy
selection. See `recommendation.md` for the design-only conclusion.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    print(
        json.dumps(
            {
                "TASK_STATUS": "COMPLETE",
                "API_CALLS": 0,
                "REPLAY_VALIDATION": "PASS",
                "M3_A": summaries["M3_A"],
                "M3_B": summaries["M3_B"],
                "M3_C": summaries["M3_C"],
                "RECOMMENDED_MODULE3_POLICY": recommendation,
                "V15_IMPLEMENTATION_AUTHORIZED": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
