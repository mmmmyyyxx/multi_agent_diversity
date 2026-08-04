from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.versions import (
    CANDIDATE_ACCEPTANCE_VERSION,
    CHECKPOINT_VERSION,
    METHOD_VERSION,
    RESPONSIBILITY_VERSION,
    TARGET_SELECTION_VERSION,
    TCS_CONTEXT_VERSION,
)
from scripts.experiment_config import SETTING_NAMES
from scripts.final_method_source_identity import build_source_identity


AUDIT_VERSION = "final_method_stage_gate_v2"
MEMBER_AWARE_SETTINGS = {
    "shared_member_aware_responsibility",
    "shared_member_aware_full",
}
PROTOCOL_FIELDS = (
    "optimization_enabled",
    "target_selection_policy",
    "sample_pool_policy",
    "tcs_context_policy",
    "candidate_selection_policy",
    "responsibility_refresh_policy",
    "repairability_freeze_enabled",
)
EXPECTED_PROTOCOLS = {
    "shared_baseline": (False, "none", "none", "none", "none", "off", False),
    "shared_independent_accuracy": (
        True, "round_robin", "individual_errors", "generic_accuracy",
        "individual_accuracy", "off", False,
    ),
    "shared_peer_state_vote_first": (
        True, "round_robin", "global_peer_state", "generic_peer_state",
        "vote_first", "off", False,
    ),
    "shared_peer_state_member_pareto": (
        True, "round_robin", "global_peer_state", "generic_peer_state",
        "member_aware_pareto", "off", False,
    ),
    "shared_member_aware_responsibility": (
        True, "member_aware_responsibility", "member_aware_residuals",
        "generic_peer_state", "member_aware_pareto", "online", True,
    ),
    "shared_member_aware_full": (
        True, "member_aware_responsibility", "member_aware_residuals",
        "member_aware_responsibility_conditioned", "member_aware_pareto", "online", True,
    ),
}
INFRASTRUCTURE_FAILURES = {
    "transport_failure",
    "teacher_provider_truncation",
    "critic_provider_truncation",
    "student_provider_truncation",
}


@dataclass(frozen=True)
class Finding:
    severity: str
    requirement: str
    current_implementation: str
    evidence: str
    required_action: str
    blocks_real_api: bool


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _expected_matrix(stage: str) -> tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...], int, bool]:
    if stage == "pilot":
        return ("disambiguation_qa",), (46,), tuple(SETTING_NAMES), 8, False
    if stage == "disambiguation":
        return ("disambiguation_qa",), (44, 45, 46), tuple(SETTING_NAMES), 32, True
    if stage == "cross_task":
        return (
            ("geometric_shapes", "ruin_names"),
            (44, 45, 46),
            ("shared_baseline", "shared_member_aware_full"),
            32,
            True,
        )
    if stage == "strict_v2_witness":
        return (
            ("disambiguation_qa",),
            (46,),
            ("shared_baseline", "shared_peer_state_member_pareto"),
            0,
            True,
        )
    if stage == "strict_v2_disambiguation":
        return (
            ("disambiguation_qa",),
            (44, 45, 46),
            (
                "shared_baseline",
                "shared_peer_state_member_pareto",
                "shared_member_aware_responsibility",
                "shared_member_aware_full",
            ),
            32,
            True,
        )
    raise ValueError(stage)


def _finding(
    findings: list[Finding],
    severity: str,
    requirement: str,
    evidence: str,
    action: str,
    *,
    blocks: bool = True,
) -> None:
    findings.append(Finding(
        severity=severity,
        requirement=requirement,
        current_implementation="stage artifact did not satisfy the requirement",
        evidence=evidence,
        required_action=action,
        blocks_real_api=blocks,
    ))


def _priority_key(priority: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(priority["updates_since_selected"]),
        str(priority["seeded_rank"]),
    )


def _audit_member_responsibility(
    run_label: str,
    run_dir: Path,
    findings: list[Finding],
) -> dict[str, int]:
    outside_eligibility = 0
    for row in _read_jsonl(run_dir / "responsibility_assignments.jsonl"):
        eligible = {
            question_hash: {int(agent) for agent in agents}
            for question_hash, agents in row.get("eligible_agents_by_question", {}).items()
        }
        for raw_agent, opportunities in row.get("assigned_opportunities", {}).items():
            agent = int(raw_agent)
            for opportunity in opportunities:
                if agent not in eligible.get(str(opportunity.get("question_hash", "")), set()):
                    outside_eligibility += 1

    non_responsible = target_front = frozen_pool = 0
    for row in _read_jsonl(run_dir / "target_priority_audit.jsonl"):
        selected = row.get("selected_agent_id")
        priorities = list(row.get("priorities", []))
        by_agent = {int(priority["agent_id"]): priority for priority in priorities}
        frozen_ids = {int(agent) for agent in row.get("frozen_agent_ids", [])}
        active_ids = {
            int(agent) for agent in row.get("active_candidate_agent_ids", [])
        }
        if selected is None:
            reason = row.get("no_actionable_reason")
            if priorities and reason != "no_actionable_repairability":
                non_responsible += 1
            if priorities and frozen_ids != set(by_agent):
                frozen_pool += 1
            continue
        selected = int(selected)
        if selected not in by_agent:
            non_responsible += 1
            continue
        if selected in frozen_ids or selected not in active_ids:
            frozen_pool += 1
        frontier = [
            priority for priority in priorities
            if not priority.get("frozen")
            and int(priority.get("target_pareto_front", 0)) == 1
        ]
        expected = min(frontier, key=_priority_key) if frontier else None
        if expected is None or selected != int(expected["agent_id"]):
            target_front += 1
        recorded = {int(agent) for agent in row.get("target_frontier_agent_ids", [])}
        if recorded != {int(priority["agent_id"]) for priority in frontier}:
            target_front += 1

    repairability_event = sum(
        int(event.get("failure_streak", 0)) != 2
        for event in _read_jsonl(run_dir / "repairability_freeze_events.jsonl")
    )
    repairability_event += sum(
        int(event.get("other_accepted_updates", 0)) < 2
        or not (
            float(event.get("portfolio_jaccard", 1.0)) < 0.8
            or int(event.get("D_before", 0)) != int(event.get("D_after", 0))
        )
        for event in _read_jsonl(run_dir / "repairability_unfreeze_events.jsonl")
    )

    for count, requirement, name in (
        (outside_eligibility, "Every assigned residual must include its target in E_x", "assignment outside eligibility"),
        (non_responsible, "Member-aware targets must have non-empty portfolios", "non-responsible target"),
        (target_front, "Normal scheduling must select from the single (D,S,d) frontier", "target-front"),
        (frozen_pool, "Frozen members must remain outside the active target pool", "frozen-pool"),
        (repairability_event, "Freeze and unfreeze events must satisfy fixed v10 thresholds", "repairability-event"),
    ):
        if count:
            _finding(
                findings,
                "BLOCKER",
                requirement,
                f"{run_label}: {name} violations={count}",
                "stop the stage and repair the scheduler or audit production",
            )
    return {
        "assignment_outside_eligibility": outside_eligibility,
        "non_responsible_target_selection": non_responsible,
        "target_front_violation": target_front,
        "frozen_pool_violation": frozen_pool,
        "repairability_event_violation": repairability_event,
    }


def _audit_run(
    *,
    stage: str,
    task: str,
    seed: int,
    setting: str,
    run_dir: Path,
    expected_updates: int,
    final_test_enabled: bool,
    source_identity: dict[str, Any],
    findings: list[Finding],
) -> dict[str, Any]:
    label = f"{task}/{setting}_seed{seed}"
    required = (
        "run_meta.json",
        "final_summary.json",
        "candidate_funnel.json",
        "candidate_decisions.jsonl",
        "responsibility_assignments.jsonl",
        "target_priority_audit.jsonl",
        "proposal_memory_summary.json",
        "cost_summary.json",
        "frozen_initialization_match.json",
        "comparison_cache_match.json",
        "best_prompts.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        _finding(
            findings, "BLOCKER", "Every matrix run must be complete",
            f"{label}: missing={missing}", "rerun once from a clean run directory",
        )
        return {"run": label, "complete": False, "missing": missing}

    meta = _read_json(run_dir / "run_meta.json")
    summary = _read_json(run_dir / "final_summary.json")
    funnel = _read_json(run_dir / "candidate_funnel.json")
    memory = _read_json(run_dir / "proposal_memory_summary.json")
    cost = _read_json(run_dir / "cost_summary.json")
    frozen = _read_json(run_dir / "frozen_initialization_match.json")
    comparison_cache = _read_json(run_dir / "comparison_cache_match.json")
    prompts = _read_json(run_dir / "best_prompts.json")
    config = meta.get("config", {})
    selection = summary.get("selection_summary", {})
    identity = meta.get("run_identity", {})
    expected_completed = 0 if setting == "shared_baseline" else expected_updates

    exact_checks = {
        "method_version": (meta.get("method_version"), METHOD_VERSION),
        "responsibility_version": (meta.get("responsibility_version"), RESPONSIBILITY_VERSION),
        "target_selection_version": (meta.get("target_selection_version"), TARGET_SELECTION_VERSION),
        "tcs_context_version": (meta.get("tcs_context_version"), TCS_CONTEXT_VERSION),
        "candidate_acceptance_version": (
            meta.get("candidate_acceptance_version"), CANDIDATE_ACCEPTANCE_VERSION,
        ),
        "checkpoint_version": (meta.get("checkpoint_version"), CHECKPOINT_VERSION),
        "agents": (config.get("agents"), 5),
        "train_size": (config.get("train_size"), 75),
        "test_size": (config.get("test_size"), 125),
        "num_candidates_per_parent": (config.get("num_candidates_per_parent"), 2),
        "stage_b_candidate_budget": (config.get("stage_b_candidate_budget"), 2),
        "member_uplift_tolerance": (config.get("member_uplift_tolerance"), 5),
        "proposal_memory_mode": (config.get("proposal_memory_mode"), "off"),
        "planned_update_count": (meta.get("planned_update_count"), expected_completed),
        "test_evaluation_count": (
            meta.get("test_evaluation_count"), 1 if final_test_enabled else 0,
        ),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in exact_checks.items()
        if actual != expected
    }
    if mismatches:
        _finding(
            findings, "BLOCKER", "Run identity, budget, defaults, and lifecycle must match",
            f"{label}: {mismatches}", "stop and rerun with the frozen formal configuration",
        )
    completed = int(meta.get("completed_update_count", -1))
    early_stop = str(meta.get("early_stop_reason", ""))
    if not (
        completed == expected_completed
        or (
            early_stop == "all_actionable_members_frozen"
            and 0 < completed <= expected_completed
        )
    ):
        _finding(
            findings,
            "BLOCKER",
            "Runs must finish the budget or stop because all actionable members are frozen",
            f"{label}: completed={completed} expected={expected_completed} early_stop={early_stop!r}",
            "stop the stage and repair lifecycle accounting",
        )

    expected_protocol = dict(zip(PROTOCOL_FIELDS, EXPECTED_PROTOCOLS[setting], strict=True))
    actual_protocol = meta.get("experiment_protocol", {})
    protocol_mismatches = {
        key: {"actual": actual_protocol.get(key), "expected": expected}
        for key, expected in expected_protocol.items()
        if actual_protocol.get(key) != expected
    }
    if protocol_mismatches:
        _finding(
            findings, "BLOCKER", "Experiment setting must isolate its registered module",
            f"{label}: {protocol_mismatches}", "repair protocol dispatch before continuing",
        )

    if frozen.get("matched") is not True:
        _finding(
            findings, "BLOCKER", "Every run must exact-match the task-seed frozen initialization",
            f"{label}: frozen match={frozen.get('matched')}", "stop; do not reuse this run",
        )
    expected_cache_role = "cumulative_task_seed_observation_reference"
    cache_gate_failures = {
        "manifest_version": comparison_cache.get("manifest_version")
        != "matched_task_seed_observation_cache_v2",
        "gate": comparison_cache.get("gate") != "PASS",
        "matched": comparison_cache.get("matched") is not True,
        "source_role": comparison_cache.get("source_role") != expected_cache_role,
        "cache_chain_continuity": comparison_cache.get("cache_chain_continuity") is not True,
        "exact_request_conflict_count": int(
            comparison_cache.get("exact_request_conflict_count", -1)
        ) != 0,
        "missing_reference_count": int(
            comparison_cache.get("missing_reference_count", -1)
        ) != 0,
        "unexpected_provider_recall_count": int(
            comparison_cache.get("unexpected_provider_recall_count", -1)
        ) != 0,
        "unaccounted_new_entry_count": int(
            comparison_cache.get("unaccounted_new_entry_count", -1)
        ) != 0,
        "unchanged_prompt_drift_count": int(
            comparison_cache.get("unchanged_prompt_drift_count", -1)
        ) != 0,
        "unchanged_prompt_aggregate_drift_count": int(
            comparison_cache.get("unchanged_prompt_aggregate_drift_count", -1)
        ) != 0,
        "unchanged_team_vote_drift_count": int(
            comparison_cache.get("unchanged_team_vote_drift_count", -1)
        ) != 0,
        "test_observation_missing_count": int(
            comparison_cache.get("test_observation_missing_count", -1)
        ) != 0,
    }
    if any(cache_gate_failures.values()):
        _finding(
            findings,
            "BLOCKER",
            "Matched settings must pass the cumulative exact-observation cache gate",
            f"{label}: failures={sorted(key for key, failed in cache_gate_failures.items() if failed)}",
            "stop and rerun from a valid cumulative task-seed observation reference",
        )
    if identity.get("git_commit") != source_identity.get("git_commit"):
        _finding(
            findings, "BLOCKER", "All stages must use the frozen source commit",
            f"{label}: run={identity.get('git_commit')} source={source_identity.get('git_commit')}",
            "mark source mismatch and restart from stage A",
        )
    if bool(identity.get("git_dirty")) != bool(source_identity.get("git_dirty")):
        _finding(
            findings, "BLOCKER", "All stages must preserve the frozen dirty-state identity",
            f"{label}: dirty-state mismatch", "mark source mismatch and restart from stage A",
        )
    if meta.get("validation_used") or int(meta.get("validation_evaluation_count", 0)):
        _finding(
            findings, "BLOCKER", "Validation selection must remain disabled",
            f"{label}: validation was used", "stop the pipeline",
        )
    if selection.get("selected_checkpoint_source") != "final_active_state":
        _finding(
            findings, "BLOCKER", "The final active state must be selected",
            f"{label}: selected source={selection.get('selected_checkpoint_source')}",
            "stop the pipeline",
        )
    if memory.get("memory_mode") != "off" or int(memory.get("memory_hit_count", 0)):
        _finding(
            findings, "BLOCKER", "Proposal Memory must be off with zero hits",
            f"{label}: mode={memory.get('memory_mode')} hits={memory.get('memory_hit_count')}",
            "stop the pipeline",
        )

    decisions = _read_jsonl(run_dir / "candidate_decisions.jsonl")
    terminal_counts = funnel.get("terminal_failure_counts", {})
    infrastructure_count = sum(
        int(terminal_counts.get(name, 0)) for name in INFRASTRUCTURE_FAILURES
    )
    if infrastructure_count:
        _finding(
            findings, "BLOCKER", "The run must have zero infrastructure failures",
            f"{label}: infrastructure terminal failures={infrastructure_count}",
            "retry the run once from scratch; stop if it fails again",
        )

    responsibility = (
        _audit_member_responsibility(label, run_dir, findings)
        if setting in MEMBER_AWARE_SETTINGS
        else {
            "assignment_outside_eligibility": 0,
            "non_responsible_target_selection": 0,
            "target_front_violation": 0,
            "frozen_pool_violation": 0,
            "repairability_event_violation": 0,
        }
    )
    selected_test = summary.get("selected_test")
    if final_test_enabled and not isinstance(selected_test, dict):
        _finding(
            findings, "BLOCKER", "A full run must contain exactly one final test",
            f"{label}: selected_test missing", "rerun once from scratch",
        )
    if not final_test_enabled and selected_test is not None:
        _finding(
            findings, "BLOCKER", "A pilot must not evaluate test",
            f"{label}: selected_test is present", "stop the pipeline",
        )

    accepted = int(cost.get("accepted_update_count", 0))
    return {
        "run": label,
        "complete": True,
        "task": task,
        "seed": seed,
        "setting": setting,
        "initial_train_state_hash": frozen.get("initialization_snapshot", {}).get(
            "initial_train_state_hash", ""
        ),
        "solver_request_identity": frozen.get("initialization_snapshot", {}).get(
            "solver_request_identity", ""
        ),
        "mutable_cache_identity": hashlib.sha256(
            str(meta.get("shared_solver_cache_path", "")).lower().encode("utf-8")
        ).hexdigest(),
        "planned_update_count": meta.get("planned_update_count"),
        "completed_update_count": meta.get("completed_update_count"),
        "test_evaluation_count": meta.get("test_evaluation_count"),
        "repairability_freeze_count": len(
            _read_jsonl(run_dir / "repairability_freeze_events.jsonl")
        ),
        "proposal_memory_hit_count": int(memory.get("memory_hit_count", 0)),
        "infrastructure_failure_count": infrastructure_count,
        "accepted_update_count": accepted,
        "total_tokens": int(cost.get("total_tokens", 0)),
        "tokens_per_accepted_update": cost.get("tokens_per_accepted_update"),
        "selected_test": selected_test,
        "final_prompt_hashes": [
            hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()
            for prompt in prompts
        ],
        "comparison_cache_match": comparison_cache,
        "responsibility_gate": responsibility,
        "protocol": {key: actual_protocol.get(key) for key in PROTOCOL_FIELDS},
        "candidate_budget_contract": actual_protocol.get("candidate_budget_contract", {}),
        "substantive_config": {
            key: value for key, value in config.items()
            if key not in {
                "experiment_setting", "out_dir", "shared_solver_cache_path",
                "frozen_initialization_manifest_path",
            }
        },
    }


def _matched_observation_consistency(
    summaries: list[dict[str, Any]],
    findings: list[Finding],
) -> list[dict[str, Any]]:
    by_key = {
        (row["task"], row["seed"], row["setting"]): row
        for row in summaries if row.get("complete")
    }
    rows: list[dict[str, Any]] = []
    for row in summaries:
        if not row.get("complete") or row["setting"] == "shared_baseline":
            continue
        baseline = by_key.get((row["task"], row["seed"], "shared_baseline"))
        if baseline is None:
            continue
        baseline_counts = list((baseline.get("selected_test") or {}).get(
            "per_agent_correct_counts", []
        ))
        selected_counts = list((row.get("selected_test") or {}).get(
            "per_agent_correct_counts", []
        ))
        unchanged = [
            agent
            for agent, (initial_hash, final_hash) in enumerate(zip(
                baseline.get("final_prompt_hashes", []),
                row.get("final_prompt_hashes", []),
                strict=True,
            ))
            if initial_hash == final_hash
        ]
        mismatched = [
            agent for agent in unchanged
            if agent >= len(baseline_counts)
            or agent >= len(selected_counts)
            or baseline_counts[agent] != selected_counts[agent]
        ]
        exact_unchanged_team = len(unchanged) == 5
        exact_team_mismatch = (
            exact_unchanged_team
            and baseline.get("selected_test") != row.get("selected_test")
        )
        passed = not mismatched and not exact_team_mismatch
        rows.append({
            "task": row["task"],
            "seed": row["seed"],
            "setting": row["setting"],
            "unchanged_member_ids": unchanged,
            "mismatched_unchanged_member_ids": mismatched,
            "exact_unchanged_team": exact_unchanged_team,
            "passed": passed,
        })
        if not passed:
            _finding(
                findings,
                "BLOCKER",
                "An unchanged prompt request must have the same observation across matched settings",
                f"{row['task']}/seed{row['seed']}/{row['setting']}: "
                f"mismatched unchanged members={mismatched}, "
                f"unchanged team mismatch={exact_team_mismatch}",
                "invalidate the comparison and rerun with the baseline-derived observation cache",
            )
    return rows


def _comparison_cache_chain(
    summaries: list[dict[str, Any]],
    findings: list[Finding],
) -> list[dict[str, Any]]:
    setting_order = {name: index for index, name in enumerate(SETTING_NAMES)}
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in summaries:
        if row.get("complete"):
            groups.setdefault((row["task"], row["seed"]), []).append(row)
    audit_rows: list[dict[str, Any]] = []
    for (task, seed), group in sorted(groups.items()):
        previous_post_hash: str | None = None
        for row in sorted(group, key=lambda value: setting_order[value["setting"]]):
            cache = row["comparison_cache_match"]
            starting_hash = str(cache.get("starting_cache_sha256", ""))
            reference_hash = str(
                cache.get("parent_reference_hash", cache.get("reference_cache_sha256", ""))
            )
            post_hash = str(
                cache.get("result_reference_hash", cache.get("post_run_reference_cache_sha256", ""))
            )
            passed = (
                cache.get("gate") == "PASS"
                and cache.get("matched") is True
                and cache.get("cache_chain_continuity") is True
                and bool(starting_hash)
                and starting_hash == reference_hash
                and bool(post_hash)
                and int(cache.get("exact_request_conflict_count", -1)) == 0
                and int(cache.get("missing_reference_count", -1)) == 0
                and int(cache.get("unexpected_provider_recall_count", -1)) == 0
                and (
                    previous_post_hash is None
                    or reference_hash == previous_post_hash
                )
            )
            audit_rows.append({
                "task": task,
                "seed": seed,
                "setting": row["setting"],
                "chain_continuity": passed,
            })
            if not passed:
                _finding(
                    findings,
                    "BLOCKER",
                    "Per-setting caches must form one cumulative task-seed observation chain",
                    f"{task}/seed{seed}/{row['setting']}: cache chain discontinuity",
                    "stop and rerun from the cumulative task-seed reference cache",
                )
            previous_post_hash = post_hash
    return audit_rows


def _setting_isolation(
    summaries: list[dict[str, Any]],
    findings: list[Finding],
) -> list[dict[str, Any]]:
    by_key = {
        (row["task"], row["seed"], row["setting"]): row
        for row in summaries if row.get("complete")
    }
    comparisons = (
        ("shared_peer_state_vote_first", "shared_peer_state_member_pareto", {"candidate_selection_policy"}),
        ("shared_peer_state_member_pareto", "shared_member_aware_responsibility", {
            "target_selection_policy", "sample_pool_policy", "responsibility_refresh_policy",
        }),
        ("shared_member_aware_responsibility", "shared_member_aware_full", {"tcs_context_policy"}),
    )
    rows = []
    task_seeds = sorted({(row["task"], row["seed"]) for row in summaries if row.get("complete")})
    for task, seed in task_seeds:
        for left, right, expected_differences in comparisons:
            if (task, seed, left) not in by_key or (task, seed, right) not in by_key:
                continue
            lhs, rhs = by_key[(task, seed, left)], by_key[(task, seed, right)]
            actual_differences = {
                key for key in PROTOCOL_FIELDS
                if lhs["protocol"].get(key) != rhs["protocol"].get(key)
            }
            substantive_match = (
                lhs["substantive_config"] == rhs["substantive_config"]
                and lhs["candidate_budget_contract"] == rhs["candidate_budget_contract"]
                and lhs["initial_train_state_hash"] == rhs["initial_train_state_hash"]
                and lhs["solver_request_identity"] == rhs["solver_request_identity"]
            )
            passed = actual_differences == expected_differences and substantive_match
            rows.append({
                "task": task,
                "seed": seed,
                "comparison": f"{left}__vs__{right}",
                "expected_protocol_differences": sorted(expected_differences),
                "actual_protocol_differences": sorted(actual_differences),
                "substantive_match": substantive_match,
                "passed": passed,
            })
            if not passed:
                _finding(
                    findings, "BLOCKER", "Registered ablations must differ only by their target module",
                    f"{task}/seed{seed}/{left} vs {right}: protocol={actual_differences}, substantive_match={substantive_match}",
                    "stop and repair setting isolation",
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument(
        "--stage",
        choices=(
            "pilot", "disambiguation", "cross_task",
            "strict_v2_witness", "strict_v2_disambiguation",
        ),
        required=True,
    )
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, required=True)
    parser.add_argument("--source_identity", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    run_root = args.run_root if args.run_root.is_absolute() else workspace / args.run_root
    report_dir = args.report_dir if args.report_dir.is_absolute() else workspace / args.report_dir
    source_path = (
        args.source_identity
        if args.source_identity.is_absolute()
        else workspace / args.source_identity
    )
    source_identity = _read_json(source_path)
    findings: list[Finding] = []
    current_source = build_source_identity(workspace)
    if current_source != source_identity:
        _finding(
            findings, "BLOCKER", "The source snapshot must remain frozen across real runs",
            "current source identity differs from the stage-C frozen identity",
            "mark completed runs source_mismatch and restart at stage A",
        )

    tasks, seeds, settings, expected_updates, final_test_enabled = _expected_matrix(args.stage)
    summaries = [
        _audit_run(
            stage=args.stage,
            task=task,
            seed=seed,
            setting=setting,
            run_dir=run_root / task / f"{setting}_seed{seed}",
            expected_updates=expected_updates,
            final_test_enabled=final_test_enabled,
            source_identity=source_identity,
            findings=findings,
        )
        for task in tasks
        for seed in seeds
        for setting in settings
    ]
    complete = [row for row in summaries if row.get("complete")]
    init_groups: dict[tuple[str, int], set[str]] = {}
    cache_ids: set[str] = set()
    for row in complete:
        init_groups.setdefault((row["task"], row["seed"]), set()).add(
            row["initial_train_state_hash"]
        )
        cache_id = row["mutable_cache_identity"]
        if cache_id in cache_ids:
            _finding(
                findings, "BLOCKER", "Every setting must have an independent mutable cache",
                f"duplicate mutable cache/config identity={cache_id}",
                "stop and rerun with per-run cloned caches",
            )
        cache_ids.add(cache_id)
    for (task, seed), hashes in init_groups.items():
        if len(hashes) != 1:
            _finding(
                findings, "BLOCKER", "All settings in a task-seed must exact-match update zero",
                f"{task}/seed{seed}: initialization hashes={len(hashes)}",
                "stop and rerun from one frozen initialization",
            )

    comparison_cache_chain = _comparison_cache_chain(complete, findings)
    matched_observation_consistency = _matched_observation_consistency(
        complete, findings
    )
    isolation = _setting_isolation(complete, findings)
    blocker_count = sum(row.severity == "BLOCKER" for row in findings)
    major_count = sum(row.severity == "MAJOR" for row in findings)
    gate = "PASS" if blocker_count == 0 and major_count == 0 else "FAIL"
    total_cost = {
        "total_tokens": sum(int(row.get("total_tokens", 0)) for row in complete),
        "accepted_update_count": sum(int(row.get("accepted_update_count", 0)) for row in complete),
        "run_count": len(complete),
    }
    accepted_rates = [
        row["accepted_update_count"] / row["completed_update_count"]
        for row in complete if int(row.get("completed_update_count") or 0) > 0
    ]
    total_cost["mean_accepted_update_rate"] = (
        statistics.mean(accepted_rates) if accepted_rates else None
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "audit_version": AUDIT_VERSION,
        "stage": args.stage,
        "gate": gate,
        "blocker_count": blocker_count,
        "major_count": major_count,
        "expected_run_count": len(tasks) * len(seeds) * len(settings),
        "complete_run_count": len(complete),
        "source_identity": source_identity,
        "runs": summaries,
        "comparison_cache_chain": comparison_cache_chain,
        "matched_observation_consistency": matched_observation_consistency,
        "setting_isolation": isolation,
        "cost": total_cost,
    }
    (report_dir / "stage_gate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "unresolved_findings.json").write_text(
        json.dumps([asdict(row) for row in findings], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "README.md").write_text(
        "\n".join((
            f"# Final Method {args.stage.title()} Stage Gate",
            "",
            f"- Gate: **{gate}**",
            f"- Complete runs: {len(complete)} / {len(tasks) * len(seeds) * len(settings)}",
            f"- BLOCKER: {blocker_count}",
            f"- MAJOR: {major_count}",
            f"- Total tokens: {total_cost['total_tokens']}",
            "",
            "This report contains hashes, counts, aggregate metrics, and method identifiers only.",
            "",
        )),
        encoding="utf-8",
    )
    print(json.dumps({
        "stage": args.stage,
        "gate": gate,
        "complete_run_count": len(complete),
        "blocker_count": blocker_count,
        "major_count": major_count,
        "total_tokens": total_cost["total_tokens"],
    }, ensure_ascii=False, indent=2))
    if gate != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
