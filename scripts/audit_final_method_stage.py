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
    CANDIDATE_SELECTION_VERSION,
    CHECKPOINT_VERSION,
    COMMON_UPDATE_POLICY_VERSION,
    DUAL_TARGET_SEARCH_VERSION,
    EXPERIMENT_MATRIX_VERSION,
    METHOD_VERSION,
    PROTOCOL_RESOLUTION_VERSION,
    REPAIRABILITY_VERSION,
    RCRU_VERSION,
    RESPONSIBILITY_VERSION,
    TARGET_SELECTION_VERSION,
    TCS_CONTEXT_VERSION,
)
from scripts.experiment_config import SETTING_NAMES
from scripts.final_method_source_identity import build_source_identity
from multi_dataset_diverse_rl.protocol import MAIN_ABLATION_MODULES


AUDIT_VERSION = "final_method_stage_gate_v13_reduced_v1"
AUDIT_MODES = (
    "frozen_source_execution",
    "offline_existing_artifact_revalidation",
)
LEGACY_NO_TEST_NORMALIZATION = "legacy_no_test_manifest_v1"
MEMBER_AWARE_SETTINGS = {
    "shared_member_aware_dual_target",
    "shared_responsibility_conditioned_dual_target",
    "shared_full_dual_target_rcru",
}
PROTOCOL_FIELDS = (
    "optimization_enabled",
    "target_selection_policy",
    "sample_pool_policy",
    "tcs_context_policy",
    "candidate_selection_policy",
    "candidate_acceptance_policy",
    "candidate_ranking_policy",
    "stage_a_policy",
    "responsibility_refresh_policy",
    "repairability_freeze_enabled",
    "service_routing_enabled",
)
EXPECTED_PROTOCOLS = {
    "shared_static_reference": (
        False, "none", "none", "none", "none", "none", "none", "none",
        "off", False, False,
    ),
    "shared_generic_evolution": (
        True, "round_robin", "individual_errors", "generic_accuracy",
        "common_monotone_safe", "fixed_peer_monotone_target_or_vote",
        "common_monotone_safe", "matched_all_generated", "off", False, False,
    ),
    "shared_member_aware_dual_target": (
        True, "repairability_adjusted_responsibility", "member_aware_residuals",
        "generic_peer_state", "common_monotone_safe",
        "fixed_peer_monotone_target_or_vote", "common_monotone_safe",
        "matched_all_generated", "online", False, True,
    ),
    "shared_responsibility_conditioned_dual_target": (
        True, "repairability_adjusted_responsibility", "member_aware_residuals",
        "member_aware_responsibility_conditioned", "common_monotone_safe",
        "fixed_peer_monotone_target_or_vote", "common_monotone_safe",
        "matched_all_generated", "online", False, True,
    ),
    "shared_full_dual_target_rcru": (
        True, "repairability_adjusted_responsibility", "member_aware_residuals",
        "member_aware_responsibility_conditioned",
        "responsibility_contribution_pareto",
        "responsibility_robust_contribution",
        "responsibility_contribution_pareto", "matched_all_generated",
        "online", False, True,
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


def _portable_config_value(key: str, value: Any) -> Any:
    if key not in {"train_path", "val_path", "test_path"}:
        return value
    path = Path(str(value))
    if not path.is_absolute():
        return path.as_posix()
    return Path(*path.parts[-3:]).as_posix()


def _test_artifact_contract(
    meta: dict[str, Any],
    summary: dict[str, Any],
    comparison_cache: dict[str, Any],
    *,
    final_test_enabled: bool,
) -> dict[str, Any]:
    """Validate test evidence without equating absence with zero accuracy."""
    config = meta.get("config", {})
    selection = summary.get("selection_summary", {})
    failures: list[str] = []
    meta_count = meta.get("test_evaluation_count")
    selection_count = selection.get("test_evaluation_count")
    lifecycle_flags = {
        "test_called_before_training_complete": meta.get(
            "test_called_before_training_complete"
        ),
        "test_used_for_selection": meta.get("test_used_for_selection"),
        "test_used_for_training": meta.get("test_used_for_training"),
    }

    if final_test_enabled:
        if config.get("final_test_enabled") is not True:
            failures.append("config_final_test_enabled")
        if meta_count != 1 or selection_count != 1:
            failures.append("final_test_evaluation_count")
        if any(value is not False for value in lifecycle_flags.values()):
            failures.append("test_isolation")
        selected_test = summary.get("selected_test")
        selected_counts = (
            selected_test.get("per_agent_correct_counts")
            if isinstance(selected_test, dict)
            else None
        )
        if not (
            isinstance(selected_counts, list)
            and len(selected_counts) == 5
            and all(isinstance(value, int) for value in selected_counts)
        ):
            failures.append("selected_test_member_counts")
        cache_counts = comparison_cache.get("test_per_agent_correct_counts")
        if not (
            isinstance(cache_counts, list)
            and len(cache_counts) == 5
            and all(isinstance(value, int) for value in cache_counts)
        ):
            failures.append("cache_test_member_counts")
        if not comparison_cache.get("test_question_set_hash"):
            failures.append("test_question_set_hash")
        if not comparison_cache.get("test_team_vote_vector_hash"):
            failures.append("test_team_vote_vector_hash")
        if int(comparison_cache.get("test_observation_missing_count", -1)) != 0:
            failures.append("test_observation_missing_count")
        explicit_status = comparison_cache.get("test_observation_status")
        if explicit_status is not None and explicit_status != "evaluated":
            failures.append("test_observation_status")
        return {
            "final_test_enabled": True,
            "final_test_evaluated": meta_count == 1,
            "final_test_evaluation_count": meta_count,
            "test_observation_status": "evaluated",
            "test_member_count_status": "available",
            "test_drift_status": "checked",
            "artifact_normalization": "none",
            "original_artifacts_modified": False,
            "failures": sorted(set(failures)),
        }

    if config.get("final_test_enabled") is not False:
        failures.append("config_final_test_disabled")
    if selection.get("final_test_enabled") is not False:
        failures.append("selection_final_test_disabled")
    if meta_count != 0 or selection_count != 0:
        failures.append("final_test_evaluation_count")
    if any(value is not False for value in lifecycle_flags.values()):
        failures.append("test_isolation")
    if any(
        summary.get(key) is not None
        for key in ("initial_test", "selected_test", "member_gain")
    ):
        failures.append("test_selection_or_metrics_present")

    nonempty_observation_fields = {
        "test_question_set_hash": comparison_cache.get("test_question_set_hash"),
        "test_team_vote_vector_hash": comparison_cache.get(
            "test_team_vote_vector_hash"
        ),
        "test_per_agent_correct_counts": comparison_cache.get(
            "test_per_agent_correct_counts"
        ),
    }
    if any(value not in (None, "", []) for value in nonempty_observation_fields.values()):
        failures.append("test_observation_present")
    missing_count = comparison_cache.get("test_observation_missing_count")
    if missing_count not in (None, 0):
        failures.append("test_observation_attempt_present")

    status_fields = (
        "final_test_enabled",
        "final_test_evaluated",
        "final_test_evaluation_count",
        "test_observation_status",
        "test_member_count_status",
        "test_drift_status",
    )
    has_explicit_status = any(key in comparison_cache for key in status_fields)
    if has_explicit_status:
        expected_status = {
            "final_test_enabled": False,
            "final_test_evaluated": False,
            "final_test_evaluation_count": 0,
            "test_observation_status": "not_applicable",
            "test_member_count_status": "not_applicable",
            "test_drift_status": "not_applicable",
        }
        if any(
            comparison_cache.get(key) != value
            for key, value in expected_status.items()
        ):
            failures.append("no_test_status_contract")
        normalization = "none"
    else:
        normalization = LEGACY_NO_TEST_NORMALIZATION

    return {
        "final_test_enabled": False,
        "final_test_evaluated": False,
        "final_test_evaluation_count": 0,
        "test_observation_status": "not_applicable",
        "test_member_count_status": "not_applicable",
        "test_drift_status": "not_applicable",
        "artifact_normalization": normalization,
        "original_artifacts_modified": False,
        "failures": sorted(set(failures)),
    }


def _audit_source_contract(
    run_source_identity: dict[str, Any],
    auditor_identity: dict[str, Any],
    *,
    audit_mode: str,
    stage: str,
) -> list[str]:
    failures: list[str] = []
    if audit_mode == "frozen_source_execution":
        if auditor_identity != run_source_identity:
            failures.append("current_source_differs_from_frozen_run_source")
        return failures
    if audit_mode != "offline_existing_artifact_revalidation":
        return ["unknown_audit_mode"]
    if stage != "pilot":
        failures.append("offline_revalidation_is_pilot_only")
    if run_source_identity.get("git_dirty") is not False:
        failures.append("run_source_not_clean")
    if auditor_identity.get("git_dirty") is not False:
        failures.append("auditor_source_not_clean")
    if (
        auditor_identity.get("method_identifiers")
        != run_source_identity.get("method_identifiers")
    ):
        failures.append("method_runtime_semantics_changed")
    return failures


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
            ("shared_static_reference", "shared_full_dual_target_rcru"),
            32,
            True,
        )
    if stage == "strict_v2_witness":
        return (
            ("disambiguation_qa",),
            (46,),
            ("shared_static_reference", "shared_member_aware_dual_target"),
            0,
            True,
        )
    if stage == "strict_v2_disambiguation":
        return (
            ("disambiguation_qa",),
            (44, 45, 46),
            (
                "shared_static_reference",
                "shared_generic_evolution",
                "shared_member_aware_dual_target",
                "shared_responsibility_conditioned_dual_target",
                "shared_full_dual_target_rcru",
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
        -float(priority["expected_update_value"]),
        -float(priority["opportunity_value"]),
        -float(priority["normalized_direct_fix"]),
        -float(priority["normalized_support_margin"]),
        -float(priority["normalized_uplift_deficit"]),
        -float(priority["normalized_wait"]),
        str(priority["seeded_rank"]),
        int(priority["agent_id"]),
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

    non_responsible = scalar_order = scalar_formula = legacy_scheduler = 0
    for row in _read_jsonl(run_dir / "target_priority_audit.jsonl"):
        selected_ids = [int(agent) for agent in row.get("selected_target_ids", [])]
        priorities = list(row.get("priorities", []))
        by_agent = {int(priority["agent_id"]): priority for priority in priorities}
        if any(
            key in row or any(key in priority for priority in priorities)
            for key in (
                "target_pareto_front",
                "target_frontier_agent_ids",
                "frozen_agent_ids",
                "frozen",
            )
        ):
            legacy_scheduler += 1
        if not selected_ids:
            reason = row.get("no_actionable_reason")
            if priorities or reason != "no_actionable_responsibility":
                non_responsible += 1
            continue
        if any(selected not in by_agent for selected in selected_ids):
            non_responsible += 1
            continue
        maxima = {
            source: max(
                (int(priority[source]) for priority in priorities),
                default=0,
            )
            for source in (
                "direct_fix_count",
                "support_margin_sum",
                "uplift_deficit",
                "updates_since_selected",
            )
        }
        normalized_fields = {
            "direct_fix_count": "normalized_direct_fix",
            "support_margin_sum": "normalized_support_margin",
            "uplift_deficit": "normalized_uplift_deficit",
            "updates_since_selected": "normalized_wait",
        }
        for priority in priorities:
            for source, normalized_field in normalized_fields.items():
                expected_normalized = (
                    int(priority[source]) / maxima[source]
                    if maxima[source] > 0 else 0.0
                )
                if abs(
                    expected_normalized
                    - float(priority[normalized_field])
                ) > 1e-12:
                    scalar_formula += 1
            opportunity = (
                0.5 * float(priority["normalized_direct_fix"])
                + 0.3 * float(priority["normalized_support_margin"])
                + 0.2 * float(priority["normalized_uplift_deficit"])
            )
            discount = 1.0 / (
                1.0 + int(priority["branch_failure_count"])
            )
            expected_value = (
                opportunity * discount
                + 0.05 * float(priority["normalized_wait"])
            )
            if (
                abs(opportunity - float(priority["opportunity_value"]))
                > 1e-12
                or abs(
                    discount - float(priority["repairability_discount"])
                ) > 1e-12
                or abs(
                    expected_value
                    - float(priority["expected_update_value"])
                ) > 1e-12
            ):
                scalar_formula += 1
        expected = [
            int(priority["agent_id"])
            for priority in sorted(priorities, key=_priority_key)[
                : len(selected_ids)
            ]
        ]
        if selected_ids != expected:
            scalar_order += 1

    freeze_artifact_rows = sum(
        len(_read_jsonl(run_dir / name))
        for name in (
            "repairability_freeze_events.jsonl",
            "repairability_unfreeze_events.jsonl",
        )
    )
    branch_parent_violation = 0
    branch_counter_violation = 0
    prior_counters: dict[tuple[str, int], tuple[int, int, int]] = {}
    for event in _read_jsonl(
        run_dir / "repairability_failure_events.jsonl"
    ):
        key = (
            str(event.get("team_prompt_state_hash", "")),
            int(event["agent_id"]),
        )
        previous = prior_counters.get(key, (0, 0, 0))
        current = (
            int(event["branch_attempt_count"]),
            int(event["branch_feasible_count"]),
            int(event["branch_failure_count"]),
        )
        if event.get("normal_completion"):
            expected = (
                previous[0] + 1,
                previous[1] + int(bool(event.get("passed_candidate_found"))),
                previous[2] + int(
                    not bool(event.get("passed_candidate_found"))
                ),
            )
        else:
            expected = previous
        if current != expected:
            branch_counter_violation += 1
        prior_counters[key] = current
    reset_violation = sum(
        not event.get("old_team_hash")
        or not event.get("new_team_hash")
        or event.get("old_team_hash") == event.get("new_team_hash")
        for event in _read_jsonl(
            run_dir / "repairability_reset_events.jsonl"
        )
    )
    by_update: dict[int, list[dict[str, Any]]] = {}
    for row in _read_jsonl(run_dir / "dual_target_branch_decisions.jsonl"):
        by_update.setdefault(int(row["update_index"]), []).append(row)
    commits = {
        int(row["update_index"]): row
        for row in _read_jsonl(
            run_dir / "dual_target_commit_decisions.jsonl"
        )
    }
    for update_index, rows in by_update.items():
        parent_hashes = {str(row.get("parent_team_hash", "")) for row in rows}
        target_ids = [int(row["target_agent_id"]) for row in rows]
        commit = commits.get(update_index, {})
        if (
            len(parent_hashes) != 1
            or len(target_ids) != len(set(target_ids))
            or (
                commit.get("committed_target_id") is not None
                and int(commit["committed_target_id"]) not in target_ids
            )
        ):
            branch_parent_violation += 1

    for count, requirement, name in (
        (outside_eligibility, "Every assigned residual must include its target in E_x", "assignment outside eligibility"),
        (non_responsible, "Member-aware targets must have non-empty portfolios", "non-responsible target"),
        (scalar_order, "Target selection must follow the scalar v13 total order", "scalar-order"),
        (scalar_formula, "Target scores must follow the frozen v13 formula", "scalar-formula"),
        (legacy_scheduler, "v13 target selection must contain no Pareto or freeze fields", "legacy-scheduler"),
        (freeze_artifact_rows, "v13 must emit no active freeze or unfreeze events", "freeze-event"),
        (branch_counter_violation, "State-local branch counters must follow normal/operational semantics", "branch-counter"),
        (reset_violation, "Repairability counters reset only across a changed team hash", "repairability-reset"),
        (branch_parent_violation, "Dual branches must share one parent and commit at most one selected target", "dual-branch"),
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
        "scalar_order_violation": scalar_order,
        "scalar_formula_violation": scalar_formula,
        "legacy_scheduler_field_violation": legacy_scheduler,
        "freeze_event_violation": freeze_artifact_rows,
        "branch_counter_violation": branch_counter_violation,
        "repairability_reset_violation": reset_violation,
        "dual_branch_parent_or_commit_violation": branch_parent_violation,
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
        "repairability_adjusted_target_scores.jsonl",
        "dual_target_branch_decisions.jsonl",
        "dual_target_commit_decisions.jsonl",
        "repairability_failure_events.jsonl",
        "repairability_reset_events.jsonl",
    ) + (
        ("rcru_candidate_decisions_sanitized.jsonl",)
        if setting == "shared_full_dual_target_rcru"
        else ()
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
    expected_completed = (
        0 if setting == "shared_static_reference" else expected_updates
    )
    test_contract = _test_artifact_contract(
        meta,
        summary,
        comparison_cache,
        final_test_enabled=final_test_enabled,
    )

    exact_checks = {
        "method_version": (meta.get("method_version"), METHOD_VERSION),
        "experiment_matrix_version": (
            meta.get("experiment_matrix_version"), EXPERIMENT_MATRIX_VERSION,
        ),
        "protocol_resolution_version": (
            meta.get("protocol_resolution_version"), PROTOCOL_RESOLUTION_VERSION,
        ),
        "common_update_policy_version": (
            meta.get("common_update_policy_version"), COMMON_UPDATE_POLICY_VERSION,
        ),
        "responsibility_version": (meta.get("responsibility_version"), RESPONSIBILITY_VERSION),
        "target_selection_version": (meta.get("target_selection_version"), TARGET_SELECTION_VERSION),
        "repairability_version": (
            meta.get("repairability_version"), REPAIRABILITY_VERSION,
        ),
        "dual_target_search_version": (
            meta.get("dual_target_search_version"), DUAL_TARGET_SEARCH_VERSION,
        ),
        "tcs_context_version": (meta.get("tcs_context_version"), TCS_CONTEXT_VERSION),
        "candidate_acceptance_version": (
            meta.get("candidate_acceptance_version"), CANDIDATE_ACCEPTANCE_VERSION,
        ),
        "candidate_selection_version": (
            meta.get("candidate_selection_version"), CANDIDATE_SELECTION_VERSION,
        ),
        "checkpoint_version": (meta.get("checkpoint_version"), CHECKPOINT_VERSION),
        "agents": (config.get("agents"), 5),
        "train_size": (config.get("train_size"), 75),
        "test_size": (config.get("test_size"), 125),
        "num_candidates_per_parent": (config.get("num_candidates_per_parent"), 2),
        "stage_b_candidate_budget": (config.get("stage_b_candidate_budget"), 2),
        "target_branch_count": (
            meta.get("target_branch_count"),
            (
                0
                if setting == "shared_static_reference"
                else (
                    2
                    if MAIN_ABLATION_MODULES[
                        setting
                    ].member_aware_dual_target_search
                    else 1
                )
            ),
        ),
        "total_generated_candidates_per_update": (
            meta.get("total_generated_candidates_per_update"),
            (
                0
                if setting == "shared_static_reference"
                else (
                    4
                    if MAIN_ABLATION_MODULES[
                        setting
                    ].member_aware_dual_target_search
                    else 2
                )
            ),
        ),
        "member_uplift_tolerance": (config.get("member_uplift_tolerance"), 5),
        "proposal_memory_mode": (config.get("proposal_memory_mode"), "off"),
        "planned_update_count": (meta.get("planned_update_count"), expected_completed),
        "test_evaluation_count": (
            meta.get("test_evaluation_count"), 1 if final_test_enabled else 0,
        ),
        "module_vector": (
            meta.get("module_vector"),
            (
                None
                if setting == "shared_static_reference"
                else asdict(MAIN_ABLATION_MODULES[setting])
            ),
        ),
        "setting_index": (
            meta.get("setting_index"),
            SETTING_NAMES.index(setting),
        ),
    }
    if setting == "shared_full_dual_target_rcru":
        exact_checks.update({
            "resolved_candidate_acceptance_version": (
                meta.get("resolved_candidate_acceptance_version"),
                RCRU_VERSION,
            ),
            "candidate_acceptance_policy": (
                meta.get("candidate_acceptance_policy"),
                "responsibility_robust_contribution",
            ),
            "candidate_ranking_policy": (
                meta.get("candidate_ranking_policy"),
                "responsibility_contribution_pareto",
            ),
        })
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
            early_stop == "no_actionable_responsibility"
            and 0 < completed <= expected_completed
        )
    ):
        _finding(
            findings,
            "BLOCKER",
            "Runs must finish the budget or stop because no responsibility is actionable",
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
    }
    if final_test_enabled:
        cache_gate_failures.update({
            "unchanged_prompt_drift_count": int(
                comparison_cache.get("unchanged_prompt_drift_count", -1)
            ) != 0,
            "unchanged_prompt_aggregate_drift_count": int(
                comparison_cache.get(
                    "unchanged_prompt_aggregate_drift_count", -1
                )
            ) != 0,
            "unchanged_team_vote_drift_count": int(
                comparison_cache.get("unchanged_team_vote_drift_count", -1)
            ) != 0,
            "test_observation_missing_count": int(
                comparison_cache.get("test_observation_missing_count", -1)
            ) != 0,
        })
    if any(cache_gate_failures.values()):
        _finding(
            findings,
            "BLOCKER",
            "Matched settings must pass the cumulative exact-observation cache gate",
            f"{label}: failures={sorted(key for key, failed in cache_gate_failures.items() if failed)}",
            "stop and rerun from a valid cumulative task-seed observation reference",
        )
    if test_contract["failures"]:
        _finding(
            findings,
            "BLOCKER",
            "Test artifacts must satisfy the stage-specific lifecycle contract",
            f"{label}: failures={test_contract['failures']}",
            "repair the test lifecycle or reject the artifact",
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
            "scalar_order_violation": 0,
            "scalar_formula_violation": 0,
            "legacy_scheduler_field_violation": 0,
            "freeze_event_violation": 0,
            "branch_counter_violation": 0,
            "repairability_reset_violation": 0,
            "dual_branch_parent_or_commit_violation": 0,
        }
    )
    selected_test = summary.get("selected_test")

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
        "repairability_reset_count": len(
            _read_jsonl(run_dir / "repairability_reset_events.jsonl")
        ),
        "proposal_memory_hit_count": int(memory.get("memory_hit_count", 0)),
        "infrastructure_failure_count": infrastructure_count,
        "accepted_update_count": accepted,
        "total_tokens": int(cost.get("total_tokens", 0)),
        "tokens_per_accepted_update": cost.get("tokens_per_accepted_update"),
        "selected_test": selected_test,
        "test_artifact_contract": test_contract,
        "artifact_normalization": test_contract["artifact_normalization"],
        "final_prompt_hashes": [
            hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()
            for prompt in prompts
        ],
        "comparison_cache_match": comparison_cache,
        "responsibility_gate": responsibility,
        "protocol": {key: actual_protocol.get(key) for key in PROTOCOL_FIELDS},
        "candidate_budget_contract": actual_protocol.get("candidate_budget_contract", {}),
        "substantive_config": {
            key: _portable_config_value(key, value)
            for key, value in config.items()
            if key not in {
                "experiment_setting", "out_dir", "shared_solver_cache_path",
                "frozen_initialization_manifest_path",
            }
        },
    }


def _matched_observation_consistency(
    summaries: list[dict[str, Any]],
    findings: list[Finding],
    *,
    final_test_enabled: bool = True,
) -> list[dict[str, Any]]:
    by_key = {
        (row["task"], row["seed"], row["setting"]): row
        for row in summaries if row.get("complete")
    }
    rows: list[dict[str, Any]] = []
    for row in summaries:
        if (
            not row.get("complete")
            or row["setting"] == "shared_static_reference"
        ):
            continue
        if not final_test_enabled:
            rows.append({
                "task": row["task"],
                "seed": row["seed"],
                "setting": row["setting"],
                "test_observation_status": "not_applicable",
                "test_member_count_status": "not_applicable",
                "test_drift_status": "not_applicable",
                "artifact_normalization": row.get(
                    "artifact_normalization", "none"
                ),
                "passed": True,
            })
            continue
        baseline = by_key.get(
            (row["task"], row["seed"], "shared_static_reference")
        )
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
        (
            "shared_static_reference",
            "shared_generic_evolution",
            {
                "optimization_enabled",
                "target_selection_policy",
                "sample_pool_policy",
                "tcs_context_policy",
                "candidate_selection_policy",
                "candidate_acceptance_policy",
                "candidate_ranking_policy",
                "stage_a_policy",
            },
        ),
        (
            "shared_generic_evolution",
            "shared_member_aware_dual_target",
            {
                "target_selection_policy",
                "sample_pool_policy",
                "tcs_context_policy",
                "responsibility_refresh_policy",
                "service_routing_enabled",
            },
        ),
        (
            "shared_member_aware_dual_target",
            "shared_responsibility_conditioned_dual_target",
            {"tcs_context_policy"},
        ),
        (
            "shared_responsibility_conditioned_dual_target",
            "shared_full_dual_target_rcru",
            {
                "candidate_selection_policy",
                "candidate_acceptance_policy",
                "candidate_ranking_policy",
            },
        ),
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
            budget_match = (
                lhs["candidate_budget_contract"]
                == rhs["candidate_budget_contract"]
            )
            if (
                left == "shared_static_reference"
                and right == "shared_generic_evolution"
            ):
                left_budget = lhs["candidate_budget_contract"]
                right_budget = rhs["candidate_budget_contract"]
                budget_match = (
                    int(left_budget.get("target_branch_count", -1)) == 0
                    and int(right_budget.get("target_branch_count", -1)) == 1
                    and int(right_budget.get(
                        "candidates_per_target_branch", -1
                    )) == 2
                )
            elif (
                left == "shared_generic_evolution"
                and right == "shared_member_aware_dual_target"
            ):
                left_budget = lhs["candidate_budget_contract"]
                right_budget = rhs["candidate_budget_contract"]
                budget_match = (
                    int(left_budget.get("target_branch_count", -1)) == 1
                    and int(right_budget.get("target_branch_count", -1)) == 2
                    and int(left_budget.get(
                        "candidates_per_target_branch", -1
                    )) == int(right_budget.get(
                        "candidates_per_target_branch", -2
                    )) == 2
                )
            substantive_match = (
                lhs["substantive_config"] == rhs["substantive_config"]
                and budget_match
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
    parser.add_argument(
        "--audit_mode",
        choices=AUDIT_MODES,
        default="frozen_source_execution",
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    run_root = args.run_root if args.run_root.is_absolute() else workspace / args.run_root
    report_dir = args.report_dir if args.report_dir.is_absolute() else workspace / args.report_dir
    source_path = (
        args.source_identity
        if args.source_identity.is_absolute()
        else workspace / args.source_identity
    )
    run_source_identity = _read_json(source_path)
    findings: list[Finding] = []
    auditor_identity = build_source_identity(workspace)
    source_failures = _audit_source_contract(
        run_source_identity,
        auditor_identity,
        audit_mode=args.audit_mode,
        stage=args.stage,
    )
    if source_failures:
        _finding(
            findings,
            "BLOCKER",
            "Run-source and auditor identities must satisfy the selected audit mode",
            f"failures={source_failures}",
            "use a clean auditor and the exact recorded run source identity",
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
            source_identity=run_source_identity,
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
        complete,
        findings,
        final_test_enabled=final_test_enabled,
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
    normalized_runs = [
        row["run"]
        for row in complete
        if row.get("artifact_normalization") == LEGACY_NO_TEST_NORMALIZATION
    ]
    normalization = {
        "artifact_normalization": (
            LEGACY_NO_TEST_NORMALIZATION if normalized_runs else "none"
        ),
        "normalized_run_count": len(normalized_runs),
        "normalized_runs": normalized_runs,
        "original_artifacts_modified": False,
    }
    payload = {
        "audit_version": AUDIT_VERSION,
        "audit_mode": args.audit_mode,
        "stage": args.stage,
        "gate": gate,
        "blocker_count": blocker_count,
        "major_count": major_count,
        "expected_run_count": len(tasks) * len(seeds) * len(settings),
        "complete_run_count": len(complete),
        "source_identity": run_source_identity,
        "run_source_identity": run_source_identity,
        "auditor_identity": auditor_identity,
        "run_source_commit": run_source_identity.get("git_commit"),
        "auditor_commit": auditor_identity.get("git_commit"),
        "method_runtime_semantics_changed": (
            run_source_identity.get("method_identifiers")
            != auditor_identity.get("method_identifiers")
        ),
        "original_run_artifacts_modified": False,
        "artifact_normalization": normalization,
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
            f"- Audit mode: `{args.audit_mode}`",
            f"- Run source commit: `{run_source_identity.get('git_commit')}`",
            f"- Auditor commit: `{auditor_identity.get('git_commit')}`",
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
