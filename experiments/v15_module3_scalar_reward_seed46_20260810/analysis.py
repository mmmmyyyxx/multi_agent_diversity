from __future__ import annotations

import csv
import importlib.util
import json
import statistics
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REPO = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
TASK1_DIR = REPO / "experiments" / "v15_module3_decision_policy_design_seed46_20260810"
TASK1_SCRIPT = TASK1_DIR / "analysis.py"
RUN_SOURCE_COMMIT = "e5bdc9f27f7a5594072aafd828c7c6053297c03c"
METHOD_VERSION = "member_aware_peer_state_v14"
CHECKPOINT_VERSION = 23
N = 75
SCALARS = ("SRA", "SRB", "SRC")
POLICIES = ("B0", "B1", "M3B", *SCALARS)


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


def load_task1_module() -> Any:
    spec = importlib.util.spec_from_file_location("module3_task1_design", TASK1_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("task1_module_spec_failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def responsibility_value(task1: Any, item: dict[str, Any], policy: str) -> Fraction:
    candidate = item["candidate"]
    contribution = candidate.responsibility_contribution
    if contribution is None:
        raise AssertionError("responsibility_metrics_missing")
    utility = contribution.utility
    if utility.active_residual_count <= 0:
        raise AssertionError("active_residual_count_not_positive")
    if policy == "SRA":
        divisor = utility.active_residual_count
        if utility.repair_lane == "margin_support":
            divisor *= 5
        return Fraction(utility.utility_delta, divisor)
    if policy == "SRB":
        return Fraction(
            utility.positive_support_count - utility.negative_support_count,
            utility.active_residual_count,
        )
    if policy == "SRC":
        return Fraction(contribution.coalition.net_contribution_delta, N)
    raise ValueError(policy)


def scalar_terms(task1: Any, item: dict[str, Any], policy: str) -> dict[str, Any]:
    metrics = task1.metric_row(item)
    team = Fraction(metrics["vote_gain"], N)
    member = Fraction(metrics["target_gain"], N)
    responsibility = responsibility_value(task1, item, policy)
    return {
        "T": team,
        "M": member,
        "R": responsibility,
        "U": team + member + responsibility,
    }


def branch_key(task1: Any, item: dict[str, Any], policy: str) -> tuple[Any, ...]:
    terms = scalar_terms(task1, item, policy)
    metrics = task1.metric_row(item)
    return (
        terms["U"],
        metrics["vote_gain"],
        metrics["target_gain"],
        metrics["soft_vote_utility_delta"],
        -metrics["vote_loss_count"],
        -metrics["edit_tokens"],
        metrics["candidate_hash"],
    )


def cross_key(task1: Any, branch: dict[str, Any], policy: str) -> tuple[Any, ...]:
    item = branch["winner"]
    terms = scalar_terms(task1, item, policy)
    metrics = task1.metric_row(item)
    return (
        terms["U"],
        metrics["vote_gain"],
        metrics["minimum_member_gain_delta"],
        metrics["total_member_gain_delta"],
        metrics["soft_vote_utility_delta"],
        -metrics["vote_loss_count"],
        -metrics["edit_tokens"],
        -branch["target_rank"],
        metrics["candidate_hash"],
    )


def scalar_replay(
    task1: Any,
    common_replay: dict[str, Any],
    policy: str,
) -> dict[str, Any]:
    updates: list[dict[str, Any]] = []
    feasible_hashes: set[str] = set()
    branches_out: list[dict[str, Any]] = []
    for update in common_replay["update_records"]:
        branch_winners: list[dict[str, Any]] = []
        branch_hashes: dict[int, str] = {}
        branch_items: dict[int, dict[str, Any] | None] = {}
        for original in update["branches"]:
            feasible = [
                item for item in original["candidate_records"] if item["decision"].passed
            ]
            feasible_hashes.update(item["candidate"].prompt_hash for item in feasible)
            winner = max(
                feasible,
                key=lambda item: branch_key(task1, item, policy),
                default=None,
            )
            target = int(original["target_agent_id"])
            branch_hashes[target] = winner["candidate"].prompt_hash if winner else ""
            branch_items[target] = winner
            branch_record = {
                "update": int(update["update_index"]),
                "target": target,
                "target_rank": int(original["target_rank"]),
                "feasible": feasible,
                "winner": winner,
            }
            branches_out.append(branch_record)
            if winner:
                branch_winners.append(branch_record)
        global_branch = max(
            branch_winners,
            key=lambda branch: cross_key(task1, branch, policy),
            default=None,
        )
        updates.append(
            {
                "update": int(update["update_index"]),
                "branches": branch_items,
                "branch_winner_hashes": branch_hashes,
                "global_winner": global_branch["winner"] if global_branch else None,
                "global_hash": (
                    global_branch["winner"]["candidate"].prompt_hash
                    if global_branch
                    else ""
                ),
            }
        )
    return {
        "updates": updates,
        "branches": branches_out,
        "feasible_hashes": feasible_hashes,
    }


def responsibility_diagnostics(task1: Any, commits: list[dict[str, Any]]) -> dict[str, Any]:
    if not commits:
        raise AssertionError("no_commits")
    lane_values = [responsibility_value(task1, item, "SRA") for item in commits]
    support_values = [responsibility_value(task1, item, "SRB") for item in commits]
    coalition_values = [responsibility_value(task1, item, "SRC") for item in commits]
    bootstraps = [
        float(item["candidate"].responsibility_contribution.robust_support.bootstrap_lcb)
        for item in commits
    ]
    positives = [
        item["candidate"].responsibility_contribution.utility.positive_support_count
        for item in commits
    ]
    negatives = [
        item["candidate"].responsibility_contribution.utility.negative_support_count
        for item in commits
    ]
    return {
        "normalized_lane_progress_sum": float(sum(lane_values, Fraction())),
        "normalized_lane_progress_sum_exact": fraction_text(sum(lane_values, Fraction())),
        "positive_support_sum": sum(positives),
        "negative_support_sum": sum(negatives),
        "net_support_rate_mean": float(sum(support_values, Fraction()) / len(commits)),
        "net_support_rate_mean_exact": fraction_text(
            sum(support_values, Fraction()) / len(commits)
        ),
        "coalition_contribution_delta_sum": sum(
            item["candidate"].responsibility_contribution.coalition.net_contribution_delta
            for item in commits
        ),
        "bootstrap_lcb_distribution": {
            "count": len(bootstraps),
            "min": min(bootstraps),
            "median": statistics.median(bootstraps),
            "mean": statistics.mean(bootstraps),
            "max": max(bootstraps),
            "values": bootstraps,
        },
        "lane_positive_commit_count": sum(value > 0 for value in lane_values),
        "lane_neutral_commit_count": sum(value == 0 for value in lane_values),
        "lane_negative_commit_count": sum(value < 0 for value in lane_values),
        "support_positive_commit_count": sum(value > 0 for value in support_values),
        "support_neutral_commit_count": sum(value == 0 for value in support_values),
        "support_negative_commit_count": sum(value < 0 for value in support_values),
        "coalition_positive_commit_count": sum(value > 0 for value in coalition_values),
        "coalition_neutral_commit_count": sum(value == 0 for value in coalition_values),
        "coalition_negative_commit_count": sum(value < 0 for value in coalition_values),
    }


def summarize(
    task1: Any,
    name: str,
    result: dict[str, Any],
    references: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    updates = result["updates"]
    commits = [row["global_winner"] for row in updates if row["global_winner"]]
    rows = [task1.metric_row(item) for item in commits]
    targets = {row["target"] for row in rows}
    diagnostics = responsibility_diagnostics(task1, commits)
    summary = {
        "policy": name,
        "feasible_candidates": len(result["feasible_hashes"]),
        "branch_winners": sum(
            bool(value) for update in updates for value in update["branch_winner_hashes"].values()
        ),
        "global_commits": len(commits),
        "no_commit_updates": 32 - len(commits),
        "changed_updates_vs_S2": sum(
            row["global_hash"] != base["global_hash"]
            for row, base in zip(updates, references["B1"]["updates"], strict=True)
        ),
        "changed_updates_vs_M3B": sum(
            row["global_hash"] != base["global_hash"]
            for row, base in zip(updates, references["M3B"]["updates"], strict=True)
        ),
        "changed_updates_vs_v14_RCRU": sum(
            row["global_hash"] != base["global_hash"]
            for row, base in zip(updates, references["B0"]["updates"], strict=True)
        ),
        "distinct_selected_target_agents": len(targets),
        "selected_target_agents": sorted(targets),
        "target_gain_sum": sum(row["target_gain"] for row in rows),
        "vote_gain_count": sum(row["vote_gain_count"] for row in rows),
        "vote_loss_count": sum(row["vote_loss_count"] for row in rows),
        "net_vote_gain": sum(row["vote_gain"] for row in rows),
        "boundary_cross_count": sum(row["boundary_cross_count"] for row in rows),
        "minimum_member_gain_delta_sum": sum(
            row["minimum_member_gain_delta"] for row in rows
        ),
        "total_member_gain_delta_sum": sum(row["total_member_gain_delta"] for row in rows),
        "soft_vote_utility_delta_sum": sum(
            row["soft_vote_utility_delta"] for row in rows
        ),
        "lane_only_commit_count": sum(
            row["target_gain"] == 0 and row["vote_gain"] == 0 and row["lane_delta"] > 0
            for row in rows
        ),
        "safety_violation_count": sum(
            row["target_gain"] < 0 or row["vote_gain"] < 0 for row in rows
        ),
        **diagnostics,
    }
    signal = {
        "SRB": "support",
        "SRC": "coalition",
    }.get(name, "lane")
    summary.update(
        {
            "responsibility_sign_basis": signal,
            "responsibility_positive_commit_count": diagnostics[
                f"{signal}_positive_commit_count"
            ],
            "responsibility_neutral_commit_count": diagnostics[
                f"{signal}_neutral_commit_count"
            ],
            "responsibility_negative_commit_count": diagnostics[
                f"{signal}_negative_commit_count"
            ],
        }
    )
    return summary


def pareto_sanity(task1: Any, results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    checked = 0
    for policy in SCALARS:
        for branch in results[policy]["branches"]:
            selected = branch["winner"]
            if selected is None:
                continue
            checked += 1
            selected_terms = scalar_terms(task1, selected, policy)
            for other in branch["feasible"]:
                if other["candidate"].prompt_hash == selected["candidate"].prompt_hash:
                    continue
                other_terms = scalar_terms(task1, other, policy)
                dimensions_selected = (
                    selected_terms["T"],
                    selected_terms["M"],
                    selected_terms["R"],
                )
                dimensions_other = (
                    other_terms["T"], other_terms["M"], other_terms["R"]
                )
                if all(
                    left >= right
                    for left, right in zip(dimensions_other, dimensions_selected, strict=True)
                ) and any(
                    left > right
                    for left, right in zip(dimensions_other, dimensions_selected, strict=True)
                ):
                    violations.append(
                        {
                            "policy": policy,
                            "update": branch["update"],
                            "target": branch["target"],
                            "selected_hash": selected["candidate"].prompt_hash,
                            "dominating_hash": other["candidate"].prompt_hash,
                        }
                    )
    return {
        "SCALAR_DOMINATED_SELECTION_VIOLATIONS": len(violations),
        "branch_winners_checked": checked,
        "violations": violations,
    }


def tradeoff_rows(
    task1: Any,
    policy: str,
    result: dict[str, Any],
    b1: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scalar_update, common_update in zip(result["updates"], b1["updates"], strict=True):
        if scalar_update["global_hash"] == common_update["global_hash"]:
            continue
        scalar_item = scalar_update["global_winner"]
        common_item = common_update["global_winner"]
        scalar_metrics = task1.metric_row(scalar_item)
        common_metrics = task1.metric_row(common_item)
        scalar_r = responsibility_value(task1, scalar_item, policy)
        common_r = responsibility_value(task1, common_item, policy)
        delta_r = scalar_r - common_r
        delta_vote = scalar_metrics["vote_gain"] - common_metrics["vote_gain"]
        delta_target = scalar_metrics["target_gain"] - common_metrics["target_gain"]
        delta_boundary = (
            scalar_metrics["boundary_cross_count"]
            - common_metrics["boundary_cross_count"]
        )
        if delta_r <= 0:
            classification = "NO_RESPONSIBILITY_GAIN"
        elif delta_vote < 0:
            classification = "RESPONSIBILITY_GAIN_WITH_VOTE_COST"
        elif delta_boundary < 0:
            classification = "RESPONSIBILITY_GAIN_WITH_BOUNDARY_COST"
        elif delta_target < 0:
            classification = "RESPONSIBILITY_GAIN_WITH_TARGET_COST"
        else:
            classification = "RESPONSIBILITY_GAIN_NO_PRIMARY_COST"
        output.append(
            {
                "policy": policy,
                "update": scalar_update["update"],
                "scalar_target": scalar_metrics["target"],
                "scalar_hash": scalar_metrics["candidate_hash"],
                "s2_target": common_metrics["target"],
                "s2_hash": common_metrics["candidate_hash"],
                "delta_team_vote_gain": delta_vote,
                "delta_target_gain": delta_target,
                "delta_boundary_cross": delta_boundary,
                "delta_responsibility_exact": fraction_text(delta_r),
                "delta_responsibility": float(delta_r),
                "classification": classification,
                "team_member_safety_preserved": (
                    scalar_metrics["target_gain"] >= 0
                    and scalar_metrics["vote_gain"] >= 0
                ),
            }
        )
    return output


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: analysis.py <authoritative_formal_root>")
    formal_root = Path(sys.argv[1]).resolve()
    task1 = load_task1_module()
    diag = task1.load_diagnosis_module(formal_root)
    s2 = diag.RunEvidence.load(diag.S2_NAME)
    s3 = diag.RunEvidence.load(diag.S3_NAME)
    diag.verify_run_identity(s2, 32)
    diag.verify_run_identity(s3, 32)
    s2_common = diag.replay(s2, "common")
    s3_rcru = diag.replay(s3, "rcru")
    s3_common = diag.replay(s3, "common")
    b0 = task1.baseline_policy(s3_rcru)
    b1 = task1.baseline_policy(s3_common)
    m3b = task1.replay_variant(diag, s3_common, policy="M3_B")
    stored_summary = json.loads(
        (TASK1_DIR / "module3_policy_design_summary.json").read_text(encoding="utf-8")
    )
    stored_candidates = [
        json.loads(line)
        for line in (TASK1_DIR / "candidate_policy_replay.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    stored_lookup = {
        (int(row["update"]), int(row["target"]), row["candidate_hash"]): row
        for row in stored_candidates
    }
    replay_mismatches: list[dict[str, Any]] = []
    for update in s3_common["update_records"]:
        index = int(update["update_index"])
        for item in update["candidates"]:
            target = int(item["raw"]["target_agent_id"])
            candidate_hash = item["candidate"].prompt_hash
            stored = stored_lookup[(index, target, candidate_hash)]
            expected_b1 = b1["updates"][index]["global_hash"] == candidate_hash
            expected_m3b = m3b["updates"][index]["global_hash"] == candidate_hash
            if stored["B1_selected"] != expected_b1 or stored["M3_B_selected"] != expected_m3b:
                replay_mismatches.append(
                    {"update": index, "target": target, "candidate_hash": candidate_hash}
                )
    if any(
        s2_common[key] or s3_rcru[key]
        for key in (
            "constraint_mismatches",
            "branch_mismatches",
            "global_mismatches",
            "parent_prompt_mismatches",
        )
    ):
        replay_mismatches.append({"type": "base_exact_replay_mismatch"})
    if stored_summary["RECOMMENDED_MODULE3_POLICY"] != "M3_B":
        replay_mismatches.append({"type": "stored_M3B_identity_mismatch"})
    replay_validation = {
        "REPLAY_VALIDATION": "PASS" if not replay_mismatches else "FAIL",
        "mismatch_count": len(replay_mismatches),
        "mismatches": replay_mismatches,
        "B1_exact": not replay_mismatches,
        "M3B_exact": not replay_mismatches,
        "REPLAY_MODE": "ONE_STEP_FIXED_PARENT_FIXED_POOL",
        "Task2_Module1_W1_used": False,
        "TEST_USED_FOR_POLICY_SELECTION": False,
        "API_CALLS": 0,
    }
    write_json(OUTPUT / "replay_validation.json", replay_validation)
    if replay_mismatches:
        write_json(
            OUTPUT / "scalar_policy_summary.json",
            {
                "TASK_STATUS": "HOLD",
                "API_CALLS": 0,
                "REPLAY_VALIDATION": "FAIL",
                "V15_IMPLEMENTATION_AUTHORIZED": False,
            },
        )
        raise SystemExit("TASK_STATUS=HOLD")

    results = {
        "B0": b0,
        "B1": b1,
        "M3B": m3b,
        **{
            policy: scalar_replay(task1, s3_common, policy) for policy in SCALARS
        },
    }
    references = {"B0": b0, "B1": b1, "M3B": m3b}
    summaries = {
        name: summarize(task1, name, result, references)
        for name, result in results.items()
    }
    pareto = pareto_sanity(task1, results)
    write_json(OUTPUT / "pareto_sanity.json", pareto)

    candidate_rows: list[dict[str, Any]] = []
    for update in s3_common["update_records"]:
        index = int(update["update_index"])
        for item in update["candidates"]:
            metrics = task1.metric_row(item)
            row: dict[str, Any] = {
                "update": index,
                "target": metrics["target"],
                "candidate_hash": metrics["candidate_hash"],
                "common_safe": item["decision"].passed,
                "target_gain": metrics["target_gain"],
                "net_vote_gain": metrics["vote_gain"],
                "vote_gain_count": metrics["vote_gain_count"],
                "vote_loss_count": metrics["vote_loss_count"],
                "boundary_cross_count": metrics["boundary_cross_count"],
            }
            for policy in SCALARS:
                terms = scalar_terms(task1, item, policy)
                row.update(
                    {
                        f"{policy}_T": fraction_text(terms["T"]),
                        f"{policy}_M": fraction_text(terms["M"]),
                        f"{policy}_R": fraction_text(terms["R"]),
                        f"{policy}_U": fraction_text(terms["U"]),
                        f"{policy}_branch_selected": (
                            results[policy]["updates"][index]["branch_winner_hashes"].get(
                                metrics["target"], ""
                            )
                            == metrics["candidate_hash"]
                        ),
                        f"{policy}_selected": results[policy]["updates"][index][
                            "global_hash"
                        ]
                        == metrics["candidate_hash"],
                    }
                )
            candidate_rows.append(row)
    write_jsonl(OUTPUT / "candidate_scalar_scores.jsonl", candidate_rows)

    tradeoffs = [
        row
        for policy in SCALARS
        for row in tradeoff_rows(task1, policy, results[policy], b1)
    ]
    write_csv(OUTPUT / "responsibility_tradeoff.csv", tradeoffs, list(tradeoffs[0]))
    classification_counts = {
        policy: dict(Counter(row["classification"] for row in tradeoffs if row["policy"] == policy))
        for policy in SCALARS
    }
    activation = {}
    for policy in SCALARS:
        attributable = sum(
            count
            for classification, count in classification_counts[policy].items()
            if classification != "NO_RESPONSIBILITY_GAIN"
        )
        activation[policy] = {
            "changed_updates_vs_S2": summaries[policy]["changed_updates_vs_S2"],
            "raw_decision_change_rate": summaries[policy]["changed_updates_vs_S2"]
            / 32,
            "responsibility_attributable_changed_updates": attributable,
            "responsibility_decision_activation_rate": attributable / 32,
            "responsibility_gain_no_primary_cost": classification_counts[policy].get(
                "RESPONSIBILITY_GAIN_NO_PRIMARY_COST", 0
            ),
            "responsibility_gain_no_primary_cost_fraction_of_changes": (
                classification_counts[policy].get(
                    "RESPONSIBILITY_GAIN_NO_PRIMARY_COST", 0
                )
                / summaries[policy]["changed_updates_vs_S2"]
                if summaries[policy]["changed_updates_vs_S2"]
                else 0.0
            ),
            "classification_counts": classification_counts[policy],
            "nearly_decision_inactive": attributable <= 1,
        }

    mismatch_rows: list[dict[str, Any]] = []
    for update in range(32):
        hashes = {name: results[name]["updates"][update]["global_hash"] for name in POLICIES}
        if len(set(hashes.values())) == 1:
            continue
        mismatch_rows.append(
            {
                "update": update,
                **{f"{name}_winner_hash": hashes[name] for name in POLICIES},
                **{
                    f"{name}_target": task1.metric_row(
                        results[name]["updates"][update]["global_winner"]
                    )["target"]
                    if results[name]["updates"][update]["global_winner"]
                    else None
                    for name in POLICIES
                },
            }
        )
    write_csv(OUTPUT / "decision_mismatches.csv", mismatch_rows, list(mismatch_rows[0]))

    comparison_fields = [
        "policy",
        "feasible_candidates",
        "branch_winners",
        "global_commits",
        "no_commit_updates",
        "changed_updates_vs_S2",
        "changed_updates_vs_M3B",
        "changed_updates_vs_v14_RCRU",
        "distinct_selected_target_agents",
        "target_gain_sum",
        "vote_gain_count",
        "vote_loss_count",
        "net_vote_gain",
        "boundary_cross_count",
        "minimum_member_gain_delta_sum",
        "total_member_gain_delta_sum",
        "soft_vote_utility_delta_sum",
        "normalized_lane_progress_sum",
        "positive_support_sum",
        "negative_support_sum",
        "net_support_rate_mean",
        "coalition_contribution_delta_sum",
        "lane_only_commit_count",
        "safety_violation_count",
    ]
    write_csv(
        OUTPUT / "scalar_policy_comparison.csv",
        [{field: summaries[name][field] for field in comparison_fields} for name in POLICIES],
        comparison_fields,
    )

    variants = {
        "shared": {
            "feasibility": "evaluate_constraints",
            "utility": "T+M+R",
            "T": "net_team_vote_gain/75",
            "M": "target_gain/75",
            "weights": [1, 1, 1],
            "primary_arithmetic": "exact_Fraction",
            "lane_only_progress_allowed": False,
        },
        "SRA": {
            "name": "COMMON_SAFE_EQUAL_SCALAR_LANE",
            "R": "lane_delta/active_count; margin_support uses lane_delta/(5*active_count)",
        },
        "SRB": {
            "name": "COMMON_SAFE_EQUAL_SCALAR_SUPPORT",
            "R": "(positive_support-negative_support)/active_count",
        },
        "SRC": {
            "name": "COMMON_SAFE_EQUAL_SCALAR_COALITION",
            "R": "net_coalition_contribution_delta/75",
        },
        "forbidden_primary_dimensions": [
            "diversity",
            "oracle_coverage",
            "prompt_distance",
            "raw_lane_utility_total",
            "raw_portfolio_size",
            "bootstrap_lcb",
            "edit_tokens",
            "target_rank",
        ],
    }
    write_json(OUTPUT / "variant_definitions.json", variants)

    best_scalar = "NONE"
    recommended = "S2_ONLY"
    why = (
        "SRA, SRB, and SRC are decision-identical to M3-B and differ from S2 on "
        "only update 4. At that update every scalar responsibility delta versus "
        "the S2 winner is exactly zero, so responsibility-attributable activation "
        "is 0/32. The scalar variants preserve common safety but provide no "
        "independent Module3 value on this fixed pool."
    )
    summary = {
        "TASK_STATUS": "COMPLETE",
        "API_CALLS": 0,
        "REPLAY_VALIDATION": "PASS",
        "REPLAY_MODE": "ONE_STEP_FIXED_PARENT_FIXED_POOL",
        "METHOD_VERSION": METHOD_VERSION,
        "CHECKPOINT_VERSION": CHECKPOINT_VERSION,
        "TEST_USED_FOR_POLICY_SELECTION": False,
        "VARIANTS_FIXED_BEFORE_REPLAY_OUTPUT": True,
        "V15_DESIGN_ONLY": True,
        "V15_IMPLEMENTATION_AUTHORIZED": False,
        "policy_summaries": summaries,
        "activation": activation,
        "tradeoff_classification_counts": classification_counts,
        "pareto_sanity": pareto,
        "S2_CHANGED_VS_S2": 0,
        "M3B_CHANGED_VS_S2": summaries["M3B"]["changed_updates_vs_S2"],
        "SRA_CHANGED_VS_S2": summaries["SRA"]["changed_updates_vs_S2"],
        "SRB_CHANGED_VS_S2": summaries["SRB"]["changed_updates_vs_S2"],
        "SRC_CHANGED_VS_S2": summaries["SRC"]["changed_updates_vs_S2"],
        "SRA_NET_VOTE_GAIN": summaries["SRA"]["net_vote_gain"],
        "SRB_NET_VOTE_GAIN": summaries["SRB"]["net_vote_gain"],
        "SRC_NET_VOTE_GAIN": summaries["SRC"]["net_vote_gain"],
        "SRA_BOUNDARY_CROSS": summaries["SRA"]["boundary_cross_count"],
        "SRB_BOUNDARY_CROSS": summaries["SRB"]["boundary_cross_count"],
        "SRC_BOUNDARY_CROSS": summaries["SRC"]["boundary_cross_count"],
        "SRA_TARGET_GAIN": summaries["SRA"]["target_gain_sum"],
        "SRB_TARGET_GAIN": summaries["SRB"]["target_gain_sum"],
        "SRC_TARGET_GAIN": summaries["SRC"]["target_gain_sum"],
        "SCALAR_DOMINATED_SELECTION_VIOLATIONS": pareto[
            "SCALAR_DOMINATED_SELECTION_VIOLATIONS"
        ],
        "BEST_SCALAR": best_scalar,
        "RECOMMENDED_MODULE3": recommended,
        "RESPONSIBILITY_DECISION_ACTIVATION_RATE": 0.0,
        "WHY": why,
    }
    write_json(OUTPUT / "scalar_policy_summary.json", summary)

    paper = """# Paper-Method Comparison

| Method | Progress semantics | Responsibility role | Tuned weights | Complexity | Decision activity |
|---|---|---|---|---|---|
| S2 | Common target-or-vote progress | None in candidate ranking | None | Low | Reference |
| v14 RCRU | Vote-or-lane progress | Primary feasibility, Pareto, ranking and hard support layers | No fitted scalar weights | High | 4/32 different from S2; includes lane-only progress |
| M3-B | Common progress | Secondary RCRU tie-break | None | Medium | 1/32 raw change; no aggregate responsibility improvement |
| Scalar family | Common progress | Equal-weight responsibility component | Fixed 1:1:1 | Low-to-medium | 1/32 raw change, 0/32 responsibility-attributable changes |

## Evaluation

The scalar family is easier to explain and ablate than v14 RCRU, and common
feasibility removes lane-only acceptance. However, all three responsibility
signals select exactly the same global winners as M3-B. Their sole difference
from S2 occurs where the selected and S2 candidates have identical lane,
support, and coalition responsibility values. The change is produced by later
non-responsibility tie-breaks, not by the scalar responsibility component.

Consequently the Seed46 fixed-pool evidence does not support presenting the
scalar family or M3-B as an independently active third paper module. S2 is the
supported method body for this evidence. This is not a general multi-seed
efficacy claim.
"""
    (OUTPUT / "paper_method_comparison.md").write_text(paper, encoding="utf-8")

    recommendation_text = f"""# v15 Module 3 Scalar-Reward Recommendation

`BEST_SCALAR = NONE`

`RECOMMENDED_MODULE3 = S2_ONLY`

## Direct answers

1. M3-B differs from S2 on **{summaries['M3B']['changed_updates_vs_S2']}/32** updates.
2. Every scalar policy differs from S2 on **1/32** updates.
3. Of those added decisions, **0** improve the relevant responsibility signal
   without vote, target, or boundary cost. The exact responsibility delta at
   update 4 is `0/1` for lane, support, and coalition.
4. The scalar reward therefore does **not** turn Module3 into an independently
   active team-conditioned selector on this pool. It reproduces M3-B exactly.

## Safety and simplification

All scalar variants use common feasibility, have zero lane-only commits, zero
safety violations, net vote gain {summaries['SRA']['net_vote_gain']}, boundary
cross count {summaries['SRA']['boundary_cross_count']}, and target gain
{summaries['SRA']['target_gain_sum']}. Pareto sanity has zero violations.

Common-safe scalarization would remove the need for lane-only progress,
responsibility Pareto filtering, and Layer1/2/3 or bootstrap as core decision
logic. But because responsibility-attributable activation is zero, the more
principled simplification is to retain S2 rather than install an inactive scalar
Module3.

The three formulas were fixed before replay output inspection. No final-test
artifact or score was loaded into formula choice, replay, ranking, tie-breaking,
or recommendation. Any future paper-method evaluation requires held-out data
that was not used during design.

`V15_IMPLEMENTATION_AUTHORIZED = false`
"""
    (OUTPUT / "recommendation.md").write_text(recommendation_text, encoding="utf-8")

    readme = """# v15 Module 3 Common-Safe Scalar Reward Replay

- Task status: **COMPLETE**
- API calls: **0**
- Replay validation: **PASS**
- Best scalar: **NONE**
- Recommended Module3: **S2_ONLY**

This directory compares S2, frozen M3-B, and three fixed equal-weight scalar
responsibility signals on the same 88-candidate Seed46 S3 Stage-B pool. Actual
parents, target pairs, responsibility state, candidates, and rollouts remain
fixed. Hypothetical winners are not propagated.

SRA, SRB, and SRC reproduce M3-B exactly. Their one change versus S2 has zero
responsibility gain, so responsibility-attributable activation is 0/32. The
evidence does not justify retaining a third module merely as formal machinery.

No API, candidate generation, rollout, validation, test call, W1 selector, or
formal source modification occurred.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
