from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_VARIANTS = ("g0_fixed_target_generic", "m20_current_v15")


def canonical_registry_hash(registry: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in registry.items()
        if key != "registry_content_hash"
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def audit(
    registry: dict[str, Any],
    summary: dict[str, Any],
    source_freeze: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    source_freeze_provided = source_freeze is not None
    if registry.get("registry_content_hash") != canonical_registry_hash(registry):
        blockers.append("registry_content_hash")
    expected = {
        (str(case["case_id"]), variant)
        for case in registry.get("cases", [])
        for variant in EXPECTED_VARIANTS
    }
    cells = summary.get("cells", [])
    observed = {
        (str(row.get("case_id")), str(row.get("variant"))) for row in cells
    }
    if expected != observed:
        blockers.append("cell_inventory_mismatch")
    if len(cells) != 16 or len(observed) != 16:
        blockers.append("cell_count")
    if summary.get("registry_hash") != registry.get("registry_content_hash"):
        blockers.append("registry_hash_mismatch")
    if summary.get("execution_commit") != registry.get("execution_commit"):
        blockers.append("execution_commit_mismatch")
    if int(summary.get("requested_candidate_count", -1)) != 32:
        blockers.append("requested_candidate_budget")
    source_freeze = source_freeze or {}
    marker = summary.get("first_success_source_freeze")
    hard_freeze_ok = bool(
        summary.get("tracked_source_freeze_hard") is True
        and isinstance(marker, dict)
        and marker.get("status") == "HARD"
        and marker.get("execution_commit") == registry.get("execution_commit")
        and marker.get("registry_content_hash")
        == registry.get("registry_content_hash")
        and (
            not source_freeze_provided
            or (
                marker.get("working_tree_source_hash")
                == source_freeze.get("working_tree_source_hash")
                and marker.get("registry_file_sha256")
                == source_freeze.get("registry_file_sha256")
                and marker.get("frozen_definition_sha256")
                == source_freeze.get("frozen_definition_sha256")
            )
        )
    )
    if not hard_freeze_ok:
        blockers.append("first_success_source_freeze")
    cases = {str(row["case_id"]): row for row in registry.get("cases", [])}
    parent_state_by_case: dict[str, set[str]] = {}
    target_by_case: dict[str, set[int]] = {}
    evaluation_policy_hashes: set[str] = set()
    g0_leakage = budget_violations = parent_mutations = 0
    commits = validation_calls = test_calls = infrastructure_failures = 0
    identity_mismatches = source_mismatches = optimizer_updates = 0
    if not hard_freeze_ok:
        source_mismatches += 1
    for row in cells:
        case_id = str(row.get("case_id"))
        variant = str(row.get("variant"))
        case = cases.get(case_id)
        if case is None:
            continue
        identity_checks = (
            int(row.get("seed", -1)) == int(case["source_seed"]),
            int(row.get("update_index", -1))
            == int(case["source_update_index"]),
            row.get("parent_team_hash") == case["parent_team_hash"],
            int(row.get("target_agent_id", -1))
            == int(case["target_agent_id"]),
            row.get("responsibility_evidence_hash")
            == case["frozen_responsibility_evidence_hash"],
        )
        if not all(identity_checks):
            identity_mismatches += 1
            blockers.append(f"case_identity:{case_id}:{variant}")
        if row.get("execution_commit") != registry.get("execution_commit"):
            source_mismatches += 1
            blockers.append(f"source_identity:{case_id}:{variant}")
        if int(row.get("requested_candidate_count", -1)) != 2:
            budget_violations += 1
            blockers.append(f"candidate_budget:{case_id}:{variant}")
        context = row.get("generation_context", {})
        if variant == "g0_fixed_target_generic":
            leak = (
                int(
                    context.get(
                        "responsibility_question_hashes_exposed_to_generator",
                        -1,
                    )
                )
                != 0
                or bool(
                    context.get(
                        "member_specific_responsibility_summary_exposed"
                    )
                )
                or bool(
                    context.get(
                        "coverage_conversion_responsibility_labels_exposed"
                    )
                )
                or bool(
                    context.get(
                        "repair_distance_responsibility_metadata_exposed"
                    )
                )
                or context.get("context_type") != "AccuracyDiagnosisContext"
                or int(
                    context.get(
                        "generator_assigned_responsibility_hash_count", -1
                    )
                ) != 0
                or bool(context.get("forbidden_field_violations"))
            )
            g0_leakage += int(leak)
            if leak:
                blockers.append(f"g0_responsibility_leakage:{case_id}")
        elif context.get("context_type") != "SingleLaneDiagnosisContext":
            blockers.append(f"m20_context_semantics:{case_id}")
        if int(context.get("frozen_responsibility_hash_overlap_count", -1)) < 0:
            blockers.append(f"generator_leakage_audit_missing:{case_id}:{variant}")
        if context.get("forbidden_field_violations"):
            blockers.append(f"generator_forbidden_fields:{case_id}:{variant}")
        policy = row.get("evaluation_policy", {})
        required_policy = {
            "experiment_setting": "experimental_v16_m20_current_v15",
            "sample_pool_policy": "member_aware_residuals",
            "stage_a_policy": "matched_all_generated",
            "candidate_acceptance_policy": (
                "fixed_peer_monotone_target_or_vote"
            ),
            "candidate_ranking_policy": "common_monotone_safe",
        }
        if policy != required_policy:
            blockers.append(f"evaluation_policy:{case_id}:{variant}")
        evaluation_policy_hashes.add(json.dumps(policy, sort_keys=True))
        mutation = (
            row.get("parent_state_hash_before")
            != row.get("parent_state_hash_after")
            or row.get("generation_parent_state_hash_before")
            != row.get("generation_parent_state_hash_after")
            or int(row.get("parent_state_mutation_count", -1)) != 0
        )
        parent_mutations += int(mutation)
        if mutation:
            blockers.append(f"parent_mutation:{case_id}:{variant}")
        commit = bool(row.get("commit_performed")) or int(
            row.get("team_prompt_commit_count", -1)
        ) != 0
        commits += int(commit)
        if commit:
            blockers.append(f"team_commit:{case_id}:{variant}")
        optimizer_update = int(row.get("optimizer_state_update_count", -1)) != 0
        optimizer_updates += int(optimizer_update)
        if optimizer_update:
            blockers.append(f"optimizer_state_update:{case_id}:{variant}")
        validation_calls += int(row.get("validation_calls", 0))
        test_calls += int(row.get("test_calls", 0))
        terminal = str(row.get("funnel", {}).get("terminal_failure_class", ""))
        infrastructure_terminal_classes = {
            "transport_failure",
            "persistence_failure",
            "provider_completion_truncation",
            "teacher_provider_truncation",
            "critic_provider_truncation",
            "teacher_schema_exhausted",
            "critic_schema_exhausted",
            "upstream_teacher_schema_exhausted",
            "upstream_critic_schema_exhausted",
            "student_invalid_exhausted",
            "student_invalid_exhausted_after_upstream_regeneration",
            "proposal_protocol_failure",
            "solver_infrastructure_failure",
        }
        if terminal in infrastructure_terminal_classes:
            infrastructure_failures += 1
            blockers.append(f"infrastructure_failure:{case_id}:{variant}")
        parent_state_by_case.setdefault(case_id, set()).add(
            str(row.get("parent_state_hash_before"))
        )
        target_by_case.setdefault(case_id, set()).add(
            int(row.get("target_agent_id", -1))
        )
    for case_id in cases:
        if len(parent_state_by_case.get(case_id, set())) != 1:
            blockers.append(f"cross_arm_parent_state:{case_id}")
        if target_by_case.get(case_id) != {
            int(cases[case_id]["target_agent_id"])
        }:
            blockers.append(f"cross_arm_target:{case_id}")
    if len(evaluation_policy_hashes) != 1:
        blockers.append("cross_arm_evaluation_policy_mismatch")
    if validation_calls:
        blockers.append("validation_calls")
    if test_calls:
        blockers.append("test_calls")
    if int(summary.get("commit_count", -1)) != 0:
        blockers.append("summary_commit_count")
    if int(summary.get("parent_state_mutation_count", -1)) != 0:
        blockers.append("summary_parent_mutation_count")
    if int(summary.get("optimizer_state_update_count", -1)) != 0:
        blockers.append("summary_optimizer_state_update_count")
    blockers = sorted(set(blockers))
    return {
        "audit_version": "v16_generic_m20_protocol_gate_v1",
        "gate": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "expected_cell_count": 16,
        "observed_cell_count": len(observed),
        "fixed_parent_cases_reproduced": sum(
            len(parent_state_by_case.get(case_id, set())) == 1
            for case_id in cases
        ),
        "responsibility_case_identity_mismatch": identity_mismatches,
        "g0_responsibility_leakage": g0_leakage,
        "candidate_budget_violations": budget_violations,
        "team_prompt_commits": commits,
        "parent_mutations": parent_mutations,
        "optimizer_state_updates": optimizer_updates,
        "validation_calls": validation_calls,
        "test_calls": test_calls,
        "source_identity_mismatch": source_mismatches,
        "first_success_source_freeze_hard": hard_freeze_ok,
        "module1_semantic_mismatch": 0,
        "common_safe_semantic_mismatch": int(
            len(evaluation_policy_hashes) != 1
        ),
        "terminal_infrastructure_failures": infrastructure_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source_freeze", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    summary = json.loads(
        (args.run_root / "probe_summary.json").read_text(encoding="utf-8")
    )
    source_freeze = json.loads(args.source_freeze.read_text(encoding="utf-8"))
    report = audit(registry, summary, source_freeze)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["gate"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
