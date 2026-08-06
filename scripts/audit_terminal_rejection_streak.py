"""Create a sanitized audit for a terminal consecutive-rejection streak.

The tool is offline-only.  It preserves numerical ownership, candidate, and
pattern identifiers while excluding prompts, examples, answers, responses,
paths, caches, and checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


FORBIDDEN_TOKENS = ("http://", "https://", "final_answer:", "openai_api_key")
ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?i)(?:[a-z]:[\\/]|file://|\\\\[^\\/\s]+[\\/][^\\/\s]+|(?:^|[\s\"'=])/(?:[^\s\"']*))"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def strictly_dominates(left: list[int], right: list[int]) -> bool:
    return len(left) == len(right) and all(a >= b for a, b in zip(left, right)) and any(
        a > b for a, b in zip(left, right)
    )


def state_by_update(decisions: list[dict[str, Any]]) -> dict[int, int]:
    state = 0
    result: dict[int, int] = {}
    for decision in decisions:
        update_index = int(decision["update_index"])
        result[update_index] = state
        if decision.get("accepted_prompt_hash"):
            state += 1
    return result


def candidate_projection(candidate: dict[str, Any], index: int) -> dict[str, Any]:
    marginal = candidate["evaluation"]["marginal"]
    return {
        "candidate_index": index,
        "prompt_hash": candidate["prompt_hash"],
        "target_gain": candidate["target_gain"],
        "vote_gain_count": candidate["vote_gain_count"],
        "vote_loss_count": candidate["vote_loss_count"],
        "vote_net_gain": candidate["vote_net_gain"],
        "assigned_residual_repair_count": marginal["assigned_residual_repair_count"],
        "coverage_gain_count": marginal["coverage_gain_count"],
        "coverage_loss_count": marginal["coverage_loss_count"],
        "candidate_objective": candidate["candidate_objective"],
        "incumbent_objective": candidate["incumbent_objective"],
        "derived_team_pareto_passed": candidate.get(
            "derived_team_pareto_passed"
        ),
        "objective_invariant_checked": candidate.get(
            "objective_invariant_checked"
        ),
        "minimum_gain_delta": candidate.get("minimum_gain_delta"),
        "total_gain_delta": candidate.get("total_gain_delta"),
        "target_is_unique_weakest": candidate.get("target_is_unique_weakest"),
        "target_is_tied_weakest": candidate.get("target_is_tied_weakest"),
        "target_nonregression_passed": candidate.get("target_nonregression_passed"),
        "target_strict_improvement": candidate.get("target_strict_improvement"),
        "team_vote_nonregression_passed": candidate["team_vote_nonregression_passed"],
        "vote_strict_improvement": candidate.get("vote_strict_improvement"),
        "target_or_vote_progress_passed": candidate.get("target_or_vote_progress_passed"),
        "member_objective_dominance_passed": candidate.get("member_objective_dominance_passed"),
        "terminal_invalid_nonregression_passed": candidate["terminal_invalid_nonregression_passed"],
        "pareto_dominates_incumbent": candidate["pareto_dominates_incumbent"],
        "hard_feasible": candidate["hard_feasible"],
        "acceptable": candidate["passed"],
        "rejection_reasons": candidate["rejection_reasons"],
    }


def scan_report(report_dir: Path) -> None:
    for path in report_dir.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").lower()
        if any(token in text for token in FORBIDDEN_TOKENS) or ABSOLUTE_PATH_PATTERN.search(text):
            raise ValueError(f"sanitized scan failed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--streak_length", type=int, default=7)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"out_dir must be fresh: {args.out_dir}")

    decisions = load_jsonl(args.run_dir / "candidate_decisions.jsonl")
    assignments = {
        int(row["team_state_version"]): row
        for row in load_jsonl(args.run_dir / "responsibility_assignments.jsonl")
    }
    contexts = {
        int(row["update_index"]): row
        for row in load_jsonl(args.run_dir / "tcs_context_history.jsonl")
    }
    state_for_update = state_by_update(decisions)
    streak = decisions[-args.streak_length:]
    indices = [int(row["update_index"]) for row in streak]
    if len(streak) != args.streak_length or indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError("terminal updates are not a consecutive streak")
    if any(row.get("accepted_prompt_hash") for row in streak):
        raise ValueError("terminal streak contains an accepted update")

    rows: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    target_counts: Counter[int] = Counter()
    pattern_updates: dict[tuple[str, ...], list[int]] = defaultdict(list)
    subgroup_updates: Counter[str] = Counter()
    for decision in streak:
        update_index = int(decision["update_index"])
        target = int(decision["target_agent_id"])
        state = state_for_update[update_index]
        assignment = assignments[state]
        owned = assignment["assigned_opportunities"][str(target)]
        owner_ages = [
            int(age) for question_hash, age in assignment["owner_age"].items()
            if int(assignment["owners"][question_hash]) == target
        ]
        if len(owned) != int(decision["target_assigned_residual_count"]) or not owner_ages:
            raise ValueError(f"ownership audit mismatch at update {update_index}")
        context = contexts[update_index]
        patterns = tuple(context["selected_pattern_ids"])
        candidates = [candidate_projection(candidate, index) for index, candidate in enumerate(decision["candidates"], 1)]
        for candidate in candidates:
            rejection_counts.update(candidate["rejection_reasons"])
        second_dominates_first = (
            len(candidates) == 2
            and strictly_dominates(candidates[1]["candidate_objective"], candidates[0]["candidate_objective"])
        )
        direct_fix_count = sum(bool(item["direct_vote_fix"]) for item in owned)
        coverage_count = sum(bool(item["coverage_opportunity"]) for item in owned)
        subgroup = "direct_fix_positive" if direct_fix_count else "coverage_or_oracle_only"
        subgroup_updates[subgroup] += 1
        target_counts[target] += 1
        pattern_updates[patterns].append(update_index)
        rows.append({
            "update_index": update_index,
            "team_state_version": state,
            "target_agent_id": target,
            "target_assigned_residual_count": int(decision["target_assigned_residual_count"]),
            "owned_direct_fix_count": direct_fix_count,
            "owned_coverage_count": coverage_count,
            "owned_dominant_wrong_count": sum(bool(item["dominant_wrong_member"]) for item in owned),
            "owned_oracle_soft_utility_sum": sum(float(item["oracle_soft_utility_gain"]) for item in owned),
            "responsibility_age": {
                "min": min(owner_ages), "mean": mean(owner_ages), "max": max(owner_ages),
            },
            "owner_subgroup": subgroup,
            "selected_structural_pattern_ids": list(patterns),
            "candidate_count": len(candidates),
            "second_candidate_objective_dominates_first": second_dominates_first,
            "candidates": candidates,
        })

    candidate_rows = [candidate for row in rows for candidate in row["candidates"]]
    assertions = {
        "terminal_streak_is_consecutive_and_rejected": True,
        "all_decisions_use_one_reused_responsibility_state": len({row["team_state_version"] for row in rows}) == 1,
        "owned_residual_count_matches_target_audit": all(
            row["target_assigned_residual_count"] > 0 for row in rows
        ),
        "all_candidates_are_unacceptable": all(not row["acceptable"] for row in candidate_rows),
    }
    if not all(assertions.values()):
        raise ValueError("rejection-streak fact assertion failed")
    result = {
        "artifact_schema_version": "terminal_rejection_streak_audit_v1",
        "method_version": "member_aware_peer_state_v6",
        "streak_update_indices": indices,
        "per_update": rows,
        "summary": {
            "decision_count": len(rows),
            "candidate_count": len(candidate_rows),
            "candidate_rejection_reason_counts": dict(sorted(rejection_counts.items())),
            "target_frequency": {str(agent): target_counts[agent] for agent in range(5)},
            "agent_4_target_share": target_counts[4] / len(rows),
            "owner_subgroup_frequency": dict(sorted(subgroup_updates.items())),
            "assigned_residual_repairs_by_candidates": sum(
                int(row["assigned_residual_repair_count"]) for row in candidate_rows
            ),
            "coverage_gains_by_candidates": sum(int(row["coverage_gain_count"]) for row in candidate_rows),
            "second_candidate_objective_dominates_first_updates": [
                row["update_index"] for row in rows if row["second_candidate_objective_dominates_first"]
            ],
            "repeated_structural_pattern_bundles": [
                {"pattern_ids": list(patterns), "update_indices": updates}
                for patterns, updates in sorted(pattern_updates.items()) if len(updates) > 1
            ],
        },
        "assertions": assertions,
    }
    args.out_dir.mkdir(parents=True)
    audit_path = args.out_dir / "last7_rejection_audit.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = result["summary"]
    (args.out_dir / "README.md").write_text(
        "# v5 Seed-44 Terminal Rejection-Streak Audit\n\n"
        "Offline audit of the final seven rejected Full updates (25 through 31). "
        "This is diagnostic evidence from one seed, not a method-effect claim.\n\n"
        "## Findings\n\n"
        f"- All {summary['decision_count']} updates reused team-state version {rows[0]['team_state_version']} and all "
        f"{summary['candidate_count']} candidates were unacceptable.\n"
        f"- Candidate assigned-residual repairs total `{summary['assigned_residual_repairs_by_candidates']}`; "
        f"coverage gains total `{summary['coverage_gains_by_candidates']}`.\n"
        f"- Rejection reasons: `{summary['candidate_rejection_reason_counts']}`.\n"
        f"- Target frequencies: `{summary['target_frequency']}`; Agent 4 share is "
        f"`{summary['agent_4_target_share']:.3f}`, so the streak is not exclusively Agent 4.\n"
        f"- The second candidate objective strictly dominates the first at updates "
        f"`{summary['second_candidate_objective_dominates_first_updates']}` but remains unacceptable under the hard guards.\n"
        f"- Repeated structural-pattern bundles: `{summary['repeated_structural_pattern_bundles']}`.\n\n"
        "`last7_rejection_audit.json` contains the target-owned direct-fix, coverage, "
        "oracle and responsibility-age statistics, candidate repair counts, guard outcomes, "
        "and hash-only structural-pattern identifiers. Prompts, questions, answers, raw role "
        "outputs, cache references, checkpoints, and local paths are excluded.\n",
        encoding="utf-8",
    )
    manifest = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.out_dir.iterdir()) if path.is_file()
    }
    (args.out_dir / "sha256_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    scan_report(args.out_dir)
    print(json.dumps({"ok": True, "out_dir": str(args.out_dir), "updates": indices}, indent=2))


if __name__ == "__main__":
    main()
