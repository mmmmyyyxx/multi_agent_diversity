from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts.diversity_matrix_d0_d5_support import (
    AGENTS,
    ARM_ORDER,
    CONTRASTS,
    classifier,
    entropy,
    read_json,
    read_jsonl,
    recursive_sanitize,
    sha256_file,
    write_csv,
    write_json,
)


ANALYSIS_VERSION = "diversity_matrix_d0_d5_analysis_v1"


def _cell(root: Path, seed: int, arm: str) -> Path:
    return root / f"seed{seed}" / arm


def _mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / len(rows) if rows else 0.0


def _correlation(left: Sequence[bool], right: Sequence[bool]) -> float:
    x = [float(value) for value in left]
    y = [float(value) for value in right]
    mx, my = _mean(x), _mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    )
    if denominator:
        return numerator / denominator
    return 1.0 if x == y else 0.0


def _validation_diversity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    member_vectors = [
        [bool(row["member_correctness"][member]) for row in rows]
        for member in range(AGENTS)
    ]
    correlation = [[0.0] * AGENTS for _ in range(AGENTS)]
    disagreement = [[0.0] * AGENTS for _ in range(AGENTS)]
    off_corr: list[float] = []
    off_disagreement: list[float] = []
    for left in range(AGENTS):
        for right in range(AGENTS):
            correlation[left][right] = _correlation(
                member_vectors[left], member_vectors[right]
            )
            disagreement[left][right] = _mean(
                a != b for a, b in zip(member_vectors[left], member_vectors[right])
            )
            if left < right:
                off_corr.append(correlation[left][right])
                off_disagreement.append(disagreement[left][right])
    g_counts = Counter(int(row["G"]) for row in rows)
    member_accuracy = [_mean(vector) for vector in member_vectors]
    oracle_accuracy = _mean(int(row["G"]) > 0 for row in rows)
    return {
        "mean_pairwise_correctness_correlation": _mean(off_corr),
        "pairwise_correctness_correlation_matrix": correlation,
        "mean_pairwise_correctness_disagreement": _mean(off_disagreement),
        "pairwise_correctness_disagreement_matrix": disagreement,
        "coverage_depth": [g_counts[index] for index in range(AGENTS + 1)],
        "unique_coverage_count": g_counts[1],
        "unique_coverage_rate": g_counts[1] / len(rows),
        "singleton_coverage_count": g_counts[1],
        "oracle_minus_best_member": oracle_accuracy - max(member_accuracy),
        "member_accuracy": member_accuracy,
        "mean_member_accuracy": _mean(member_accuracy),
        "best_member_accuracy": max(member_accuracy),
        "worst_member_accuracy": min(member_accuracy),
    }


def _support_accumulation(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: (int(row["update_index"]), row["question_hash"]))
    recovered: dict[str, int] = {}
    last_g: dict[str, int] = {}
    deepened: set[str] = set()
    cross_member: set[str] = set()
    conversions: set[tuple[str, int]] = set()
    successful_repairs: Counter[int] = Counter()
    recoveries = 0
    for row in ordered:
        key = str(row["question_hash"])
        target = int(row["target_agent_id"])
        before, after = int(row["G_before"]), int(row["G_after"])
        last_g[key] = after
        if not bool(row["target_correct_before"]) and bool(row["target_correct_after"]):
            successful_repairs[target] += 1
        if before == 0 and after >= 1:
            recoveries += 1
            recovered.setdefault(key, target)
        if key in recovered and before == 1 and after >= 2:
            deepened.add(key)
            if recovered[key] != target:
                cross_member.add(key)
        if (
            key in recovered
            and not bool(row["vote_correct_before"])
            and bool(row["vote_correct_after"])
        ):
            conversions.add((key, int(row["update_index"])))
    return {
        "zero_to_one_recoveries": recoveries,
        "zero_to_one_to_two_plus_deepenings": len(deepened),
        "cross_member_accumulation": len(cross_member),
        "coverage_to_vote_conversions": len(conversions),
        "persistent_singletons": sum(
            key not in deepened and last_g.get(key) == 1 for key in recovered
        ),
        "successful_repairs_by_member": [successful_repairs[index] for index in range(AGENTS)],
    }


def _search(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    branches = [branch for row in decisions for branch in row.get("branches", ())]
    funnels = [branch.get("funnel", {}) for branch in branches]
    if not funnels:
        funnels = [row.get("funnel", {}) for row in decisions]
    candidates = [candidate for row in decisions for candidate in row.get("candidates", ())]
    targets = Counter(
        int(target) for row in decisions for target in row.get("selected_target_ids", ())
    )
    commits = Counter(
        int(row["target_agent_id"])
        for row in decisions
        if row.get("accepted_prompt_hash") and row.get("target_agent_id") is not None
    )
    target_counts = [targets[index] for index in range(AGENTS)]
    commit_counts = [commits[index] for index in range(AGENTS)]
    target_total = sum(target_counts)
    return {
        "student_reach": sum(int(row.get("student_calls", 0)) for row in funnels),
        "valid_candidates": sum(
            candidate.get("candidate_stage") != "loss_blind_generic_revision"
            and bool(candidate.get("evaluation"))
            for candidate in candidates
        ),
        "evaluable_candidates": sum(bool(candidate.get("evaluation")) for candidate in candidates),
        "feasible_candidates": sum(
            bool((candidate.get("constraint") or {}).get("passed"))
            for candidate in candidates
        ),
        "accepted_commits": sum(commit_counts),
        "target_opportunities_by_member": target_counts,
        "accepted_commits_by_member": commit_counts,
        "unique_targeted_members": sum(value > 0 for value in target_counts),
        "target_entropy": entropy(target_counts),
        "target_concentration": max(target_counts, default=0) / target_total if target_total else 0.0,
        "min_target_opportunities_per_member": min(target_counts, default=0),
        "max_target_opportunities_per_member": max(target_counts, default=0),
    }


def _train_transition(decisions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    accepted = [row for row in decisions if row.get("accepted_prompt_hash")]
    winners = []
    for decision in accepted:
        prompt_hash = decision["accepted_prompt_hash"]
        winner = next(
            (candidate for candidate in decision.get("candidates", ())
             if candidate.get("prompt_hash") == prompt_hash),
            None,
        )
        if winner and winner.get("evaluation"):
            winners.append(winner["evaluation"])
    return {
        "train_target_gain": sum(
            int(row.get("member_gain", {}).get("target_gain_vs_incumbent", 0))
            for row in winners
        ),
        "train_vote_gains": sum(
            int(row.get("marginal", {}).get("vote_gain_count", 0)) for row in winners
        ),
        "train_vote_losses": sum(
            int(row.get("marginal", {}).get("vote_loss_count", 0)) for row in winners
        ),
    }


def _specialization(
    run: Path,
    decisions: Sequence[Mapping[str, Any]],
    support: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    assignments = read_jsonl(run / "responsibility_assignments.jsonl")
    latest = assignments[-1].get("assigned_opportunities", {}) if assignments else {}
    targeted: dict[int, set[str]] = defaultdict(set)
    for decision in decisions:
        for branch in decision.get("branches", ()):
            member = int(branch["target_agent_id"])
            targeted[member].update(map(str, branch.get("assigned_question_hashes", ())))
    lane_rows: list[dict[str, Any]] = []
    lane_vectors: list[list[int]] = []
    for member in range(AGENTS):
        counts = Counter()
        for item in latest.get(str(member), latest.get(member, ())):
            if int(item.get("vote_flip_gain", 0)) > 0:
                counts["direct_flip"] += 1
            elif bool(item.get("coverage_opportunity")):
                counts["coverage"] += 1
            else:
                counts["margin_support"] += 1
        vector = [counts["coverage"], counts["direct_flip"], counts["margin_support"]]
        lane_vectors.append(vector)
        lane_rows.append({
            "member": member,
            "coverage_responsibility": vector[0],
            "direct_flip_responsibility": vector[1],
            "margin_support_responsibility": vector[2],
            "targeted_residuals": len(targeted[member]),
            "unique_repaired_residuals": int(support["successful_repairs_by_member"][member]),
            "successful_repairs": int(support["successful_repairs_by_member"][member]),
        })
    distances = []
    normalized = []
    for vector in lane_vectors:
        total = sum(vector)
        normalized.append([value / total if total else 0.0 for value in vector])
    for left in range(AGENTS):
        for right in range(left + 1, AGENTS):
            distances.append(sum(abs(a - b) for a, b in zip(normalized[left], normalized[right])) / 2)
    return lane_rows, {
        "specialization_entropy": _mean(entropy(vector) for vector in lane_vectors),
        "between_member_lane_differentiation": _mean(distances),
    }


def _paired_change(
    baseline: Sequence[Mapping[str, Any]], final: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    base = {row["example_id_hash"]: row for row in baseline}
    after = {row["example_id_hash"]: row for row in final}
    if set(base) != set(after):
        raise RuntimeError("validation row identity mismatch")
    vote_gain = vote_loss = oracle_gain = oracle_loss = 0
    for key in base:
        before, current = base[key], after[key]
        vote_gain += not before["vote_correct"] and current["vote_correct"]
        vote_loss += before["vote_correct"] and not current["vote_correct"]
        oracle_gain += int(before["G"]) == 0 and int(current["G"]) > 0
        oracle_loss += int(before["G"]) > 0 and int(current["G"]) == 0
    return {
        "validation_vote_gains": vote_gain,
        "validation_vote_losses": vote_loss,
        "validation_oracle_gains": oracle_gain,
        "validation_oracle_losses": oracle_loss,
        "collateral_regression_count": vote_loss,
    }


def analyze(prep_root: Path, run_root: Path, audit_root: Path, out: Path) -> dict[str, Any]:
    audit = read_json(audit_root / "execution_audit.json")
    if audit.get("scientific_analysis_gate") != "PASS":
        raise RuntimeError("scientific analysis requires execution audit PASS")
    if out.exists():
        raise FileExistsError("fresh report root required")
    out.mkdir(parents=True)
    registry = read_json(prep_root / "registry.json")
    trajectory_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    diversity_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    specialization_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    validation_by_cell: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for seed in registry["seeds"]:
        for arm in ARM_ORDER:
            run = _cell(run_root, seed, arm)
            validation_dir = run_root / "validation" / f"seed{seed}" / arm
            validation = read_json(validation_dir / "evaluation_summary_private.json")
            validation_rows = read_jsonl(validation_dir / "validation_rows_sanitized.jsonl")
            validation_by_cell[(seed, arm)] = validation_rows
            metrics = _validation_diversity(validation_rows)
            decisions = read_jsonl(run / "candidate_decisions.jsonl")
            search = _search(decisions)
            support = _support_accumulation(read_jsonl(run / "g_transition_audit.jsonl"))
            transitions = _train_transition(decisions)
            lanes, specialization = _specialization(run, decisions, support)
            baseline_rows = validation_by_cell.get((seed, "D0"), validation_rows)
            change = _paired_change(baseline_rows, validation_rows)
            row = {
                "seed": seed,
                "arm": arm,
                "validation_vote_accuracy": float(validation["vote_accuracy"]),
                "validation_oracle_accuracy": float(validation["oracle_accuracy"]),
                "initial_to_final_vote_delta": int(validation["vote_correct_count"]) - sum(bool(item["vote_correct"]) for item in baseline_rows),
                "initial_to_final_oracle_delta": int(validation["oracle_correct_count"]) - sum(int(item["G"]) > 0 for item in baseline_rows),
                **{key: value for key, value in metrics.items() if not isinstance(value, list)},
                **search,
                **{key: value for key, value in support.items() if key != "successful_repairs_by_member"},
                **transitions,
                **change,
                **specialization,
            }
            trajectory_rows.append(row)
            diversity_rows.append({
                "seed": seed, "arm": arm,
                "mean_pairwise_correctness_correlation": metrics["mean_pairwise_correctness_correlation"],
                "mean_pairwise_correctness_disagreement": metrics["mean_pairwise_correctness_disagreement"],
                "oracle_minus_best_member": metrics["oracle_minus_best_member"],
                "correlation_matrix_json": json.dumps(metrics["pairwise_correctness_correlation_matrix"], separators=(",", ":")),
                "disagreement_matrix_json": json.dumps(metrics["pairwise_correctness_disagreement_matrix"], separators=(",", ":")),
            })
            for depth, count in enumerate(metrics["coverage_depth"]):
                coverage_rows.append({"seed": seed, "arm": arm, "G": depth, "count": count})
            for member in range(AGENTS):
                member_rows.append({
                    "seed": seed, "arm": arm, "member": member,
                    "validation_accuracy": metrics["member_accuracy"][member],
                    "target_opportunities": search["target_opportunities_by_member"][member],
                    "accepted_commits": search["accepted_commits_by_member"][member],
                    "successful_repairs": support["successful_repairs_by_member"][member],
                })
                specialization_rows.append({"seed": seed, "arm": arm, **lanes[member]})
            for decision in decisions:
                update_rows.append({
                    "seed": seed,
                    "arm": arm,
                    "update_index": int(decision["update_index"]),
                    "target_count": len(decision.get("selected_target_ids", ())),
                    "accepted": bool(decision.get("accepted_prompt_hash")),
                    "selected_target_ids_json": json.dumps(decision.get("selected_target_ids", ())),
                    "student_calls": int(decision.get("funnel", {}).get("student_calls", 0)),
                    "valid_candidates": int(decision.get("funnel", {}).get("valid_candidate_count", 0)),
                    "feasible_candidates": int(decision.get("funnel", {}).get("constraint_feasible", 0)),
                })

    lookup = {(row["seed"], row["arm"]): row for row in trajectory_rows}
    contrast_rows: list[dict[str, Any]] = []
    classifiers: dict[str, Any] = {}
    metrics = ("validation_vote_accuracy", "validation_oracle_accuracy")
    for name, (left, right, matched) in CONTRASTS.items():
        for metric in metrics:
            values = [lookup[(seed, left)][metric] - lookup[(seed, right)][metric] for seed in registry["seeds"]]
            result = classifier(values)
            classifiers[f"{name}:{metric}"] = result
            contrast_rows.append({
                "contrast": name, "metric": metric,
                "left": left, "right": right,
                "compute_matched": matched,
                "seed_values_json": json.dumps(values),
                "mean_delta": result["mean"],
                "wins": result["wins"], "ties": result["ties"], "losses": result["losses"],
                "classifier": result["label"],
            })
    interaction_values = [
        (lookup[(seed, "D5")]["validation_vote_accuracy"] - lookup[(seed, "D3")]["validation_vote_accuracy"])
        - (lookup[(seed, "D4")]["validation_vote_accuracy"] - lookup[(seed, "D2")]["validation_vote_accuracy"])
        for seed in registry["seeds"]
    ]
    interaction = classifier(interaction_values)
    classifiers["C4_interaction:validation_vote_accuracy"] = interaction
    contrast_rows.append({
        "contrast": "C4_factorial_interaction", "metric": "validation_vote_accuracy",
        "left": "(D5-D3)", "right": "(D4-D2)", "compute_matched": True,
        "seed_values_json": json.dumps(interaction_values), "mean_delta": interaction["mean"],
        "wins": interaction["wins"], "ties": interaction["ties"], "losses": interaction["losses"],
        "classifier": interaction["label"],
    })

    arm_summary = []
    for arm in ARM_ORDER:
        rows = [row for row in trajectory_rows if row["arm"] == arm]
        arm_summary.append({
            "arm": arm,
            **{key: _mean(row[key] for row in rows) for key in (
                "validation_vote_accuracy", "validation_oracle_accuracy", "mean_member_accuracy",
                "mean_pairwise_correctness_correlation", "mean_pairwise_correctness_disagreement",
                "unique_coverage_rate", "zero_to_one_recoveries",
                "zero_to_one_to_two_plus_deepenings", "cross_member_accumulation",
                "coverage_to_vote_conversions", "persistent_singletons", "accepted_commits",
            )},
        })
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "execution_gate": audit["execution_gate"],
        "scientific_analysis_gate": "PASS",
        "seeds": registry["seeds"],
        "trajectory_count": len(trajectory_rows),
        "new_test_calls": 0,
        "primary_metric": "final validation plurality-vote accuracy",
        "arm_summary": arm_summary,
        "classifiers": classifiers,
        "interpretation_guard": (
            "Useful complementarity requires joint evidence from coverage, correctness structure, "
            "member competence, and vote utility; lower correlation alone is not improvement."
        ),
    }
    write_csv(out / "trajectory_level.csv", trajectory_rows)
    write_csv(out / "update_level.csv", update_rows)
    write_csv(out / "member_level.csv", member_rows)
    write_csv(out / "diversity_metrics.csv", diversity_rows)
    write_csv(out / "coverage_depth.csv", coverage_rows)
    write_csv(out / "specialization.csv", specialization_rows)
    write_csv(out / "contrast_summary.csv", contrast_rows)
    write_json(out / "classifier.json", classifiers)
    write_json(out / "execution_summary.json", summary)
    for source, name in (
        (prep_root / "source_freeze.json", "source_freeze.json"),
        (prep_root / "arm_configs.json", "arm_configs.json"),
        (prep_root / "compute_parity.json", "compute_parity.json"),
    ):
        write_json(out / name, read_json(source))
    header = "| Arm | Vote | Oracle | Mean Member | Pairwise Corr | Unique Coverage | Deepenings | Vote Conversions | Commits |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    table = [header]
    for row in arm_summary:
        table.append(
            f"| {row['arm']} | {row['validation_vote_accuracy']:.3f} | {row['validation_oracle_accuracy']:.3f} | "
            f"{row['mean_member_accuracy']:.3f} | {row['mean_pairwise_correctness_correlation']:.3f} | "
            f"{row['unique_coverage_rate']:.3f} | {row['zero_to_one_to_two_plus_deepenings']:.2f} | "
            f"{row['coverage_to_vote_conversions']:.2f} | {row['accepted_commits']:.2f} |"
        )
    readme = "\n".join([
        "# Diversity Matrix D0-D5",
        "",
        "Three fresh seeds compare a Static reference, canonical Generic-S0, and a compute-matched RR/W1 by Generic/Responsibility-conditioned 2x2 factorial.",
        "D1 is a strong practical reference and is not compute matched to D2-D5.",
        "The primary metric is final frozen-state Validation Vote accuracy. Validation was never used for trajectory decisions. Test data was neither loaded nor evaluated.",
        "",
        *table,
        "",
        "`D4-D2` is the primary diversity contrast. The full metric vector must be interpreted jointly; disagreement or lower correlation alone is not useful diversity.",
        "",
        "NEW_TEST_CALLS=0",
    ]) + "\n"
    (out / "README.md").write_text(readme, encoding="utf-8")
    write_json(out / "preregistration.json", {
        "runtime_version": registry["runtime_version"],
        "seeds": registry["seeds"],
        "arms": list(ARM_ORDER),
        "updates": registry["updates"],
        "solver_model": registry["solver_model"],
        "role_model": registry["role_model"],
        "test_access": registry["test_access"],
    })
    preliminary = []
    for path in out.iterdir():
        if path.suffix == ".json":
            preliminary.extend(recursive_sanitize(json.loads(path.read_text(encoding="utf-8"))))
        else:
            preliminary.extend(recursive_sanitize(path.read_text(encoding="utf-8")))
    (out / "sanitization_report.txt").write_text(
        "PASS\nforbidden_findings=0\n" if not preliminary else "FAIL\n" + "\n".join(preliminary) + "\n",
        encoding="utf-8",
    )
    (out / "test_report.txt").write_text(
        "execution_audit=PASS\nscientific_analysis=PASS\nnew_test_calls=0\ndeterministic_replay=PENDING_FINAL_VERIFICATION\n",
        encoding="utf-8",
    )
    if preliminary:
        raise RuntimeError(f"sanitization failed: {preliminary[:3]}")
    manifest_lines = []
    for path in sorted(out.iterdir()):
        if path.name == "sha256_manifest.txt":
            continue
        manifest_lines.append(f"{sha256_file(path)}  {path.name}")
    (out / "sha256_manifest.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(
        args.prep_root.resolve(), args.run_root.resolve(),
        args.audit_root.resolve(), args.out.resolve(),
    )
    print(json.dumps({
        "scientific_analysis_gate": summary["scientific_analysis_gate"],
        "trajectory_count": summary["trajectory_count"],
        "new_test_calls": summary["new_test_calls"],
    }, indent=2))


if __name__ == "__main__":
    main()
