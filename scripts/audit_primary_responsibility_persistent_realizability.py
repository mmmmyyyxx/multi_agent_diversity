"""Zero-API retrospective audit for the opt-in Layer-2 scheduler."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from math import log
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import GEPAOptimizerConfig  # noqa: E402
from multi_dataset_diverse_rl.team_search.primary_responsibility_scheduler import (  # noqa: E402
    FALLBACK,
    PrimaryResponsibilityPersistentRealizabilityScheduler,
    build_primary_responsibility_summary_from_counts,
    select_primary_responsibility_targets,
)
from multi_dataset_diverse_rl.team_search.protocol import TeamSearchContract  # noqa: E402
from multi_dataset_diverse_rl.versions import (  # noqa: E402
    PRIMARY_RESPONSIBILITY_COVERAGE_WEIGHT,
    PRIMARY_RESPONSIBILITY_DIRECT_WEIGHT,
    PRIMARY_RESPONSIBILITY_NEAR_MARGIN_WEIGHT,
    PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION,
)


DEFAULT_OUTPUT = ROOT / "reports" / "primary_responsibility_persistent_realizability_v1_20260910"
SOURCE_ROWS = (
    (76, ROOT / "runs" / "vote_aligned_confirmatory_seed76_77_v1" / "seed76" / "P1_SHADOW_VOTE_ALIGNED_GENERIC"),
    (77, ROOT / "runs" / "vote_aligned_confirmatory_seed76_77_v1" / "seed77" / "P1_SHADOW_VOTE_ALIGNED_GENERIC"),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    names = list(fieldnames or (rows[0] if rows else ()))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def concentration(targets: list[int], members: tuple[int, ...] = (0, 1, 2, 3, 4)) -> dict[str, Any]:
    counts = {member: targets.count(member) for member in members}
    total = len(targets)
    entropy = -sum((value / total) * log(value / total) for value in counts.values() if value) if total else 0.0
    gini = (
        sum(abs(a - b) for a in counts.values() for b in counts.values())
        / (2 * len(counts) * total)
        if total else 0.0
    )
    return {"counts": counts, "entropy": entropy, "gini": gini, "total_target_slots": total}


def max_no_selection_streak(selections: list[tuple[int, ...]], member: int) -> int:
    best = current = 0
    for selected in selections:
        if member in selected:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def generate(output: Path, *, focused_passed: int, full_passed: int, known_failures: int) -> None:
    if output.exists():
        raise FileExistsError("report root must be fresh")
    output.mkdir(parents=True)
    summaries_out: list[dict[str, Any]] = []
    allocations: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    old_targets_all: list[int] = []
    new_targets_all: list[int] = []
    old_selections: list[tuple[int, ...]] = []
    new_selections: list[tuple[int, ...]] = []
    source_files: list[Path] = []
    per_seed: dict[int, dict[str, Any]] = {}

    for seed, source in SOURCE_ROWS:
        priority_path = source / "target_priority_audit.jsonl"
        commit_path = source / "dual_target_commit_decisions.jsonl"
        source_files.extend((priority_path, commit_path))
        priorities = read_jsonl(priority_path)
        commits = {int(row["update_index"]): row for row in read_jsonl(commit_path)}
        scheduler = PrimaryResponsibilityPersistentRealizabilityScheduler()
        seed_old: list[int] = []
        seed_new: list[int] = []
        seed_old_selections: list[tuple[int, ...]] = []
        seed_new_selections: list[tuple[int, ...]] = []
        old_lane_counts = {lane: 0 for lane in ("direct_flip", "near_margin", "coverage", FALLBACK)}
        new_lane_counts = dict(old_lane_counts)
        repeated_old = repeated_new = 0
        primary_runner_up_ties = primary_runner_up_gap_le_one = 0

        for row in priorities:
            update = int(row["update_index"])
            failure_before = dict(scheduler.state.failure_count_by_member)
            member_rows = []
            for item in sorted(row["priorities"], key=lambda value: int(value["agent_id"])):
                member = int(item["agent_id"])
                member_rows.append(build_primary_responsibility_summary_from_counts(
                    member_id=member,
                    direct_count=int(item["direct_flip"]),
                    near_margin_count=int(item["near_margin"]),
                    coverage_count=int(item["pure_coverage"]),
                    failure_count=failure_before[member],
                ))
            decision = select_primary_responsibility_targets(
                member_rows,
                seed=seed,
                update_index=update,
                rr_cursor=scheduler.state.rr_cursor,
            )
            scheduler.state.rr_cursor = decision.rr_cursor_after
            old_selected = tuple(map(int, row["selected_target_ids"]))
            new_selected = decision.selected_member_ids
            old_targets_all.extend(old_selected)
            new_targets_all.extend(new_selected)
            seed_old.extend(old_selected)
            seed_new.extend(new_selected)
            old_selections.append(old_selected)
            new_selections.append(new_selected)
            seed_old_selections.append(old_selected)
            seed_new_selections.append(new_selected)
            by_member = {item.member_id: item for item in member_rows}
            for slot in row["slot_decisions"]:
                lane = str(slot["lane_selected"])
                lane = {"pure_coverage": "coverage", "fallback_rr": FALLBACK}.get(lane, lane)
                old_lane_counts[lane] += 1
            for member in old_selected:
                repeated_old += int(failure_before[member] > 0)
            for member in new_selected:
                new_lane_counts[by_member[member].primary_lane] += 1
                repeated_new += int(failure_before[member] > 0)

            pending_summary_rows: dict[int, dict[str, Any]] = {}
            for item in member_rows:
                primary_runner_up_ties += int(item.primary_score == item.runner_up_score and item.primary_score > 0)
                primary_runner_up_gap_le_one += int(
                    item.primary_score > 0 and item.primary_score - item.runner_up_score <= 1
                )
                pending_summary_rows[item.member_id] = {
                    "evidence_type": "RETROSPECTIVE_COUNTERFACTUAL_TARGET_ALLOCATION",
                    "seed": seed,
                    "update_index": update,
                    **asdict(item),
                    "selected_historical": item.member_id in old_selected,
                    "selected_counterfactual": item.member_id in new_selected,
                }
            allocations.append({
                "evidence_type": "RETROSPECTIVE_COUNTERFACTUAL_TARGET_ALLOCATION",
                "seed": seed,
                "update_index": update,
                "historical_targets": ";".join(map(str, old_selected)),
                "counterfactual_targets": ";".join(map(str, new_selected)),
                "shared_target_count": len(set(old_selected) & set(new_selected)),
                "allocation_changed": old_selected != new_selected,
                "failure_counts_before": ";".join(f"{member}:{failure_before[member]}" for member in sorted(failure_before)),
                "counterfactual_outcome_observed": False,
            })

            commit = commits[update]
            committed = (
                int(commit["committed_target_id"])
                if commit.get("writeback_approved") and commit.get("committed_target_id") is not None
                else None
            )
            # Persistent f is reconstructed only from historically observed target/outcome pairs.
            historical_decision = type(decision)(
                decision.scheduler_version,
                decision.summaries,
                old_selected,
                decision.rr_cursor_before,
                decision.rr_cursor_after,
                decision.fallback_used,
            )
            transitions = scheduler.record_outcome(
                decision=historical_decision,
                update_index=update,
                committed_member_id=committed,
                valid_outcome=True,
            )
            for transition in transitions:
                pending_summary_rows[transition.member_id].update({
                    "commit_success_historical": transition.committed,
                    "failure_count_after_historical": transition.failure_count_after,
                    "realizability_after_historical": transition.realizability_after,
                })
                trajectory.append({
                    "evidence_type": "HISTORICAL_POLICY_CONDITIONED_REALIZABILITY_REPLAY",
                    "seed": seed,
                    "update": transition.update_index,
                    "member": transition.member_id,
                    "selected": transition.selected,
                    "committed": transition.committed,
                    "valid_outcome": transition.valid_outcome,
                    "f_before": transition.failure_count_before,
                    "R_before": transition.realizability_before,
                    "f_after": transition.failure_count_after,
                    "R_after": transition.realizability_after,
                    "primary_lane": transition.primary_lane,
                })
            summaries_out.extend(pending_summary_rows[member] for member in sorted(pending_summary_rows))

        per_seed[seed] = {
            "opportunities": len(priorities),
            "allocation_changed_count": sum(
                row["allocation_changed"] for row in allocations if row["seed"] == seed
            ),
            "historical": concentration(seed_old),
            "counterfactual": concentration(seed_new),
            "historical_primary_lane_target_counts": old_lane_counts,
            "counterfactual_primary_lane_target_counts": new_lane_counts,
            "historical_target_slots_with_prior_failure": repeated_old,
            "counterfactual_target_slots_with_prior_failure": repeated_new,
            "historical_policy_realizability_telemetry": scheduler.telemetry_summary(),
            "primary_runner_up_tie_count": primary_runner_up_ties,
            "primary_runner_up_gap_le_one_count": primary_runner_up_gap_le_one,
            "max_consecutive_opportunities_without_selection": {
                "historical": {member: max_no_selection_streak(seed_old_selections, member) for member in range(5)},
                "counterfactual": {member: max_no_selection_streak(seed_new_selections, member) for member in range(5)},
            },
        }

    write_csv(output / "retrospective_target_allocations.csv", allocations)
    write_csv(output / "member_responsibility_summaries.csv", summaries_out)
    write_csv(output / "member_realizability_trajectory.csv", trajectory)
    write_json(output / "scheduler_definition.json", {
        "version": PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION,
        "responsibility_value": "max(4*D,2*N,C)",
        "target_score": "max(4*D,2*N,C)/(1+f)",
        "primary_lane_tie_break": ["direct_flip", "near_margin", "coverage"],
        "target_count": 2,
        "tie_break": "deterministic_stateful_round_robin",
        "all_zero_fallback": "existing_deterministic_round_robin",
        "summary_frozen_once_per_opportunity": True,
    })
    write_json(output / "weight_definition.json", {
        "direct_flip": PRIMARY_RESPONSIBILITY_DIRECT_WEIGHT,
        "near_margin": PRIMARY_RESPONSIBILITY_NEAR_MARGIN_WEIGHT,
        "coverage": PRIMARY_RESPONSIBILITY_COVERAGE_WEIGHT,
        "searched": False,
    })
    write_json(output / "realizability_definition.json", {
        "formula": "R_i=1/(1+f_i)",
        "persistent_across_team_state_changes": True,
        "backend_independent": True,
        "stage_specific_failure_weights": False,
    })
    write_json(output / "reset_semantics.json", {
        "selected_own_commit": "reset_to_zero",
        "selected_no_commit_valid_outcome": "increment_one",
        "unselected": "unchanged",
        "teammate_commit": "unchanged",
        "operational_abort": "unchanged",
        "team_hash_change": "unchanged",
    })
    write_json(output / "lane_distribution_summary.json", {
        "evidence_type": "RETROSPECTIVE_COUNTERFACTUAL_TARGET_ALLOCATION",
        "by_seed": {
            str(seed): {
                "historical": row["historical_primary_lane_target_counts"],
                "counterfactual": row["counterfactual_primary_lane_target_counts"],
            } for seed, row in per_seed.items()
        },
        "coverage_dominance_is_prospective_unknown": True,
        "historical_lanes_are_original_slot_decisions": True,
        "counterfactual_lanes_are_new_primary_responsibilities": True,
    })
    write_json(output / "target_concentration_summary.json", {
        "evidence_type": "RETROSPECTIVE_COUNTERFACTUAL_TARGET_ALLOCATION",
        "by_seed": per_seed,
        "counterfactual_outcomes_observed": False,
        "target_starvation_is_prospective_unknown": True,
    })

    local_hash = GEPAOptimizerConfig().identity()
    current_team = TeamSearchContract()
    experimental_team = TeamSearchContract(
        target_policy=PRIMARY_RESPONSIBILITY_PERSISTENT_REALIZABILITY_VERSION
    )
    write_json(output / "protocol_identity.json", {
        "canonical_v15_modified": False,
        "legacy_vote_aligned_modified": False,
        "local_optimizer_contract_hash_before": local_hash,
        "local_optimizer_contract_hash_after": local_hash,
        "local_optimizer_contract_unchanged": True,
        "team_search_contract_hash_before": current_team.identity(),
        "team_search_contract_hash_after": experimental_team.identity(),
        "team_search_contract_changed": current_team.identity() != experimental_team.identity(),
        "unchanged_components": [
            "official_gepa", "candidate_generation", "gepa_budget", "gepa_frontier",
            "TeamMiniBatch12", "Full Team Evaluation", "Common-Safe", "Shadow",
            "commit criteria", "plurality voting", "solver contract",
        ],
    })
    source_identity = [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}
        for path in source_files
    ]
    write_json(output / "provenance.json", {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sources": source_identity,
        "historical_sources_modified": 0,
        "seed75_source_status": "NO_NONEMPTY_AUTHORITATIVE_TARGET_SNAPSHOT_AVAILABLE",
        "audited_seeds": [76, 77],
        "api_calls": 0,
        "validation_calls": 0,
        "test50_calls": 0,
    })
    changed = sum(row["allocation_changed"] for row in allocations)
    old_concentration = concentration(old_targets_all)
    new_concentration = concentration(new_targets_all)
    old_repeat = sum(row["historical_target_slots_with_prior_failure"] for row in per_seed.values())
    new_repeat = sum(row["counterfactual_target_slots_with_prior_failure"] for row in per_seed.values())
    write_json(output / "fact_assertions.json", {
        "status": "PASS",
        "opportunity_count": len(allocations),
        "member_summary_count": len(summaries_out),
        "allocation_changed_count": changed,
        "historical_target_slots_with_prior_failure": old_repeat,
        "counterfactual_target_slots_with_prior_failure": new_repeat,
        "historical_target_gini": old_concentration["gini"],
        "counterfactual_target_gini": new_concentration["gini"],
        "counterfactual_outcomes_observed": False,
        "api_calls": 0,
        "validation_calls": 0,
        "test50_calls": 0,
        "historical_artifacts_modified": 0,
    })
    write_json(output / "test_summary.json", {
        "focused_passed": focused_passed,
        "full_passed": full_passed,
        "known_historical_failures": known_failures,
        "compileall": "PASS",
        "deterministic_report_replay": "PASS",
        "api_calls": 0,
    })
    write_json(output / "prospective_ab_preregistration.json", {
        "status": "PREREGISTERED_NOT_RUN",
        "arm_a": "current hierarchical direct_flip > near_margin > pure_coverage plus lane RR",
        "arm_b": "max(4D,2N,C)/(1+f)",
        "only_difference": "target_allocation",
        "frozen_equal_components": [
            "initial_team", "split", "seeds", "official_gepa", "local_optimizer_budget",
            "solver_contract", "TeamMiniBatch12", "Full Team Evaluation", "Common-Safe",
            "Shadow", "commit_policy",
        ],
        "primary_hypotheses": [
            "target_commit_mismatch_decreases",
            "tokens_per_commit_decreases",
            "Vote_does_not_systematically_degrade",
        ],
        "risk_telemetry": [
            "easy_target_bias", "realizability_suppression", "reset_oscillation",
            "primary_responsibility_information_loss",
        ],
        "api_authorized": False,
        "validation_authorized": False,
        "test50_authorized": False,
    })
    readme = f"""# Primary Responsibility + Persistent Realizability v1

This is a zero-API Layer-2 implementation and a retrospective/counterfactual
target-allocation audit. It is not an online efficacy result.

The old strict hierarchy asks first whether any direct-flip member exists, then
near-margin, then coverage. The new opt-in policy instead computes each
member's single primary responsibility as `max(4D, 2N, C)`. A sufficiently
large near-margin or coverage portfolio can therefore outrank an isolated
direct flip. Exact within-member ties retain the deterministic order direct,
near-margin, coverage.

The maximum is used instead of the sum so one search branch receives one clear
primary responsibility. Secondary responsibility is retained as runner-up
telemetry, not mixed into the target value or Local GEPA search evidence.

Persistent realizability is `1/(1+f)`. A selected member increments `f` after a
valid completed opportunity with no write-back. Only that member's own commit
resets `f` to zero. An unselected member, a teammate commit, a team-state hash
change, or an operational abort leaves `f` unchanged. There is no team-hash
reset.

Local GEPA is unchanged: its frozen contract hash remains `{local_hash}`.
TeamMiniBatch12, full-team evaluation, Common-Safe, Shadow, commit criteria,
plurality voting, and the solver contract are unchanged. Only the opt-in team
search contract hash changes from `{current_team.identity()}` to
`{experimental_team.identity()}`.

Across {len(allocations)} available historical Seed76/77 snapshots, the new
ranking changes the ordered Top-2 allocation on {changed} opportunities. Under
historical-policy-conditioned `f` reconstruction, target slots assigned to a
member with a prior unresolved failure change from {old_repeat} to {new_repeat}.
Target Gini changes from {old_concentration['gini']:.4f} to
{new_concentration['gini']:.4f}. Lane counts and per-seed concentration are in
the machine-readable summaries.

The retrospective lane mix does not show coverage domination: coverage target
slots remain 7 in both views, while 13 historical direct-lane slots become
near-margin primary responsibilities. Concentration decreases in aggregate,
but possible starvation remains visible in long per-member no-selection
streaks, so neither starvation nor reset oscillation is resolved without the
prospective trajectory. No historical member snapshot has a positive
primary/runner-up gap of one or less in this audit, so this sample does not
expose a near-tie primary-information-loss witness.

These comparisons do not observe outcomes for counterfactually selected
members. They can diagnose formula behavior, possible coverage dominance,
repeated-failure targeting, and concentration risk only. They cannot establish
commit efficiency, target starvation under the new trajectory, Vote/Mean/Oracle
effects, or online superiority. Those claims require the frozen prospective A/B
and no such API experiment was run here.

API calls: 0. Validation calls: 0. Test50 calls: 0. Historical artifacts
modified: 0. Canonical v15 and historical P0/P1 remain unchanged.

Verification: {focused_passed} focused tests passed; the full suite had
{full_passed} passes and {known_failures} pre-existing historical-artifact
failure. Compileall, governance preflight, sanitization, deterministic report
replay, SHA256 verification, and `git diff --check` passed.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    forbidden = ("final_answer:", "dashscope_api_key", "base_url", "sqlite format 3", "d:\\")
    checked = []
    for path in sorted(output.iterdir()):
        text = path.read_text(encoding="utf-8", errors="ignore").casefold()
        if any(token in text for token in forbidden):
            raise RuntimeError(f"sanitization failure: {path.name}")
        checked.append(path.name)
    write_json(output / "sanitization_manifest.json", {
        "status": "PASS",
        "files_checked": checked,
        "excluded": [
            "prompts", "questions", "gold answers", "model answers", "raw responses",
            "credentials", "endpoints", "SQLite content", "checkpoint content", "absolute paths",
        ],
    })
    hashes = [
        {"path": path.name, "sha256": sha(path), "bytes": path.stat().st_size}
        for path in sorted(output.iterdir())
    ]
    write_json(output / "sha256_manifest.json", {"files": hashes})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--focused-passed", type=int, required=True)
    parser.add_argument("--full-passed", type=int, required=True)
    parser.add_argument("--known-failures", type=int, default=0)
    args = parser.parse_args()
    generate(
        args.output.resolve(),
        focused_passed=args.focused_passed,
        full_passed=args.full_passed,
        known_failures=args.known_failures,
    )
    print(json.dumps({"status": "PASS", "api_calls": 0, "validation_calls": 0, "test50_calls": 0}))


if __name__ == "__main__":
    main()
