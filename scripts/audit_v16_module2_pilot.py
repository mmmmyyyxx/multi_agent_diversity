from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


SETTINGS = (
    ("experimental_v16_c0_current_v15", "c0_current_v15"),
    ("experimental_v16_c2_boundary_plus_preservation", "c2_boundary_plus_preservation"),
    ("experimental_v16_c3_coalition_aware_preservation", "c3_coalition_aware_preservation"),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index_decisions_by_update(
    decisions: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for decision in decisions:
        update_index = int(decision["update_index"])
        if update_index in indexed:
            raise ValueError(f"duplicate candidate decision for update {update_index}")
        parent_hash = str(decision.get("parent_team_hash", "")).strip()
        if not parent_hash:
            raise ValueError(
                f"candidate decision for update {update_index} lacks parent_team_hash"
            )
        indexed[update_index] = decision
    return indexed


def reconcile_context_parent_hash(
    context_row: dict[str, Any],
    decisions_by_update: dict[int, dict[str, Any]],
) -> tuple[str, str]:
    """Resolve parent provenance without mutating an immutable run artifact."""
    update_index = int(context_row["update_index"])
    target_agent_id = int(context_row["target_agent_id"])
    decision = decisions_by_update.get(update_index)
    if decision is None:
        raise ValueError(
            f"context update {update_index} has no candidate decision provenance"
        )
    selected_targets = {int(row) for row in decision.get("selected_target_ids", [])}
    if target_agent_id not in selected_targets:
        raise ValueError(
            f"context target {target_agent_id} is not selected at update {update_index}"
        )
    decision_parent = str(decision["parent_team_hash"])
    context_parent = str(context_row.get("parent_team_hash", "")).strip()
    if context_parent and context_parent != decision_parent:
        raise ValueError(
            f"context parent hash conflicts with update {update_index} decision"
        )
    return (
        decision_parent,
        "context_row_verified" if context_parent else "candidate_decision_by_update",
    )


def git_head(workspace: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def geometry_summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for decision in decisions for row in decision.get("candidates", []) if row.get("evaluation")]
    geometry = Counter(row.get("module2_diagnostics", {}).get("candidate_geometry", "unavailable") for row in candidates)
    feasible = sum(bool(row.get("constraint", {}).get("passed")) for row in candidates)
    return {
        "evaluated_candidates": len(candidates),
        "geometry": dict(sorted(geometry.items())),
        "common_safe_feasible": feasible,
        "feasible_fraction": feasible / len(candidates) if candidates else None,
        "repair_set_gain_candidates": sum((row.get("module2_diagnostics", {}).get("repair_set_gain_count") or 0) > 0 for row in candidates),
        "p1_loss_candidates": sum((row.get("module2_diagnostics", {}).get("P1_loss_count") or 0) > 0 for row in candidates),
        "p2_loss_candidates": sum((row.get("module2_diagnostics", {}).get("P2_loss_count") or 0) > 0 for row in candidates),
    }


def common_safe_violations(decisions: list[dict[str, Any]]) -> int:
    violations = 0
    for decision in decisions:
        accepted_hash = decision.get("accepted_prompt_hash")
        for candidate in decision.get("candidates", []):
            if candidate.get("prompt_hash") != accepted_hash:
                continue
            constraint = candidate.get("constraint") or {}
            required = (
                constraint.get("passed") is True,
                constraint.get("target_nonregression_passed") is True,
                constraint.get("team_vote_nonregression_passed") is True,
                constraint.get("target_or_vote_progress_passed") is True,
                constraint.get("terminal_invalid_nonregression_passed") is True,
            )
            violations += int(not all(required))
    return violations


def module1_w1_violations(decisions: list[dict[str, Any]]) -> int:
    violations = 0
    for decision in decisions:
        for row in decision.get("agent_target_priorities", []):
            expected = (
                float(row["opportunity_value"])
                + 0.05 * float(row["normalized_wait"])
            ) * float(row["repairability_discount"])
            violations += int(
                not math.isclose(
                    expected,
                    float(row["expected_update_value"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
    return violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--gate_root", type=Path, required=True)
    parser.add_argument("--prep_root", type=Path, required=True)
    args = parser.parse_args()
    if args.gate_root.exists():
        raise FileExistsError("canonical gate root must be fresh")
    args.gate_root.mkdir(parents=True)
    freeze = read_json(args.prep_root / "source_freeze_manifest.json")
    findings: list[str] = []
    summaries: dict[str, Any] = {}
    c2_memberships: dict[tuple[str, int], tuple[tuple[str, ...], tuple[str, ...]]] = {}
    c3_memberships: dict[tuple[str, int], tuple[tuple[str, ...], tuple[str, ...]]] = {}
    total_validation = total_test = infrastructure = 0
    total_common_safe = total_w1 = total_vote_correct_propagation = 0
    total_serialization_failures = total_max_one = 0
    context_parent_reconciliation_count = 0
    context_parent_verified_count = 0
    for setting, variant in SETTINGS:
        run = args.run_root / "disambiguation_qa" / f"{setting}_seed51"
        required = ("final_summary.json", "run_meta.json", "candidate_decisions.jsonl", "cost_summary.json", "training_dynamics.jsonl", "experimental_module2_context_diagnostics.jsonl")
        missing = [name for name in required if not (run / name).is_file()]
        if missing:
            findings.append(f"{setting}: missing {missing}")
            continue
        meta = read_json(run / "run_meta.json")
        final = read_json(run / "final_summary.json")
        decisions = read_jsonl(run / "candidate_decisions.jsonl")
        contexts = read_jsonl(run / "experimental_module2_context_diagnostics.jsonl")
        cost = read_json(run / "cost_summary.json")
        dynamics = read_jsonl(run / "training_dynamics.jsonl")
        try:
            decisions_by_update = index_decisions_by_update(decisions)
        except (KeyError, TypeError, ValueError) as exc:
            findings.append(f"{setting}: invalid candidate decision provenance: {exc}")
            decisions_by_update = {}
        identity = meta.get("run_identity", {})
        if identity.get("git_commit") != freeze["git_head"] or identity.get("git_dirty"):
            findings.append(f"{setting}: source identity mismatch")
        if meta.get("module2_context_variant") != variant:
            findings.append(f"{setting}: variant identity mismatch")
        if len(decisions) != 8 or int(meta.get("completed_update_count", -1)) != 8:
            findings.append(f"{setting}: completed updates != 8")
        if int(meta.get("planned_update_count", -1)) != 8:
            findings.append(f"{setting}: planned updates != 8")
        lifecycle = final.get("selection_summary", {})
        validation = int(meta.get("validation_evaluation_count", lifecycle.get("validation_evaluation_count", 0)))
        test = int(lifecycle.get("test_evaluation_count", 0))
        total_validation += validation
        total_test += test
        if validation or test or lifecycle.get("final_test_enabled") is not False:
            findings.append(f"{setting}: forbidden validation/test evaluation")
        infrastructure += sum(int(row.get("funnel", {}).get("infrastructure_failed_updates", 0)) for row in decisions)
        if any(len(row.get("selected_target_ids", [])) > 2 for row in decisions):
            findings.append(f"{setting}: target branch budget violation")
        max_one = sum(
            int(
                bool(row.get("accepted_prompt_hash"))
                and sum(
                    row.get("accepted_prompt_hash") == candidate.get("prompt_hash")
                    for candidate in row.get("candidates", [])
                ) != 1
            )
            for row in decisions
        )
        total_max_one += max_one
        if max_one:
            findings.append(f"{setting}: max-one commit evidence mismatch")
        common_safe = common_safe_violations(decisions)
        w1 = module1_w1_violations(decisions)
        total_common_safe += common_safe
        total_w1 += w1
        if common_safe:
            findings.append(f"{setting}: common-safe violation")
        if w1:
            findings.append(f"{setting}: Module1 W1 invariant violation")
        for row in contexts:
            try:
                parent_hash, provenance = reconcile_context_parent_hash(
                    row, decisions_by_update
                )
            except (KeyError, TypeError, ValueError) as exc:
                findings.append(f"{setting}: context provenance failure: {exc}")
                continue
            context_parent_reconciliation_count += int(
                provenance == "candidate_decision_by_update"
            )
            context_parent_verified_count += int(
                provenance == "context_row_verified"
            )
            key = (parent_hash, int(row["target_agent_id"]))
            membership = (tuple(row["repair_question_hashes"]), tuple(row["preservation_question_hashes"]))
            (c2_memberships if variant.startswith("c2_") else c3_memberships if variant.startswith("c3_") else {})[key] = membership
        vote_correct_propagation = sum(
            int(row.get("vote_correct_repair_item_count", -1))
            for row in contexts
        )
        serialization_failures = sum(
            int(row.get("serialized_context_char_count", 0)) <= 0
            for row in contexts
        )
        total_vote_correct_propagation += vote_correct_propagation
        total_serialization_failures += serialization_failures
        if vote_correct_propagation:
            findings.append(f"{setting}: vote-correct repair propagation")
        if variant != "c0_current_v15" and serialization_failures:
            findings.append(f"{setting}: context serialization failure")
        by_update: dict[int, list[set[str]]] = {}
        for row in contexts:
            by_update.setdefault(int(row["update_index"]), []).append(
                set(row["repair_question_hashes"])
            )
        if any(
            len(rows) == 2 and bool(rows[0] & rows[1])
            for rows in by_update.values()
        ):
            findings.append(f"{setting}: cross-branch repair duplication")
        funnel = geometry_summary(decisions)
        summaries[variant] = {
            "updates": len(decisions),
            "accepted_updates": sum(bool(row.get("accepted_prompt_hash")) for row in decisions),
            "branches_attempted": sum(len(row.get("branches", [])) for row in decisions),
            "context_count": len(contexts),
            "valid_candidates": sum(int(row.get("funnel", {}).get("valid_candidate_count", 0)) for row in decisions),
            **funnel,
            "cost": cost,
            "initial_train": {
                key: (dynamics[0] if dynamics else {}).get(key)
                for key in ("team_vote_correct_count", "oracle_correct_count", "per_agent_correct_counts", "mean_member_accuracy", "minimum_member_accuracy", "all_wrong_rate", "n_eff")
            },
            "final_train": {
                key: (dynamics[-1] if dynamics else {}).get(key)
                for key in ("team_vote_correct_count", "oracle_correct_count", "per_agent_correct_counts", "mean_member_accuracy", "minimum_member_accuracy", "all_wrong_rate", "n_eff")
            },
        }
    shared_keys = set(c2_memberships) & set(c3_memberships)
    membership_mismatch = sum(c2_memberships[key] != c3_memberships[key] for key in shared_keys)
    if membership_mismatch:
        findings.append("C2/C3 same-parent membership mismatch")
    if infrastructure:
        findings.append("infrastructure failures observed")
    for variant in ("c2_boundary_plus_preservation", "c3_coalition_aware_preservation"):
        if summaries.get(variant, {}).get("valid_candidates", 0) <= 0:
            findings.append(f"{variant}: no valid candidates")
    result = {
        "audit_version": "v16_module2_seed51_mechanism_pilot_gate_v1",
        "gate": "PASS" if not findings else "FAIL",
        "blocker_count": len(findings),
        "findings": findings,
        "source_commit": freeze.get("git_head"),
        "run_source_commit": freeze.get("git_head"),
        "auditor_commit": git_head(Path(__file__).resolve().parents[1]),
        "audit_mode": "offline_existing_artifact_revalidation",
        "original_run_artifacts_modified": False,
        "context_parent_hash_source": "candidate_decisions_by_update",
        "context_parent_reconciliation_count": context_parent_reconciliation_count,
        "context_parent_verified_count": context_parent_verified_count,
        "seed": 51,
        "validation_evaluation_count": total_validation,
        "test_evaluation_count": total_test,
        "infrastructure_failure_count": infrastructure,
        "module1_w1_invariant_violation_count": total_w1,
        "common_safe_violation_count": total_common_safe,
        "max_one_commit_violation_count": total_max_one,
        "vote_correct_repair_propagation_count": total_vote_correct_propagation,
        "context_serialization_failure_count": total_serialization_failures,
        "c2_c3_same_parent_membership_comparison_count": len(shared_keys),
        "c2_c3_membership_mismatch": membership_mismatch,
        "settings": summaries,
    }
    (args.gate_root / "pilot_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.gate_root / "pilot_summary.md").write_text(
        "# v16 Module2 Seed51 mechanism Pilot\n\n"
        f"Protocol gate: **{result['gate']}**. This is a single-seed train-only "
        "mechanism pilot with no validation or final test; it does not establish "
        "generalization or statistical significance.\n",
        encoding="utf-8",
    )
    with (args.gate_root / "candidate_geometry_by_variant.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("variant", "A", "B", "C", "D", "E", "F", "evaluated", "feasible"))
        writer.writeheader()
        for variant, row in summaries.items():
            geometry = row["geometry"]
            writer.writerow({"variant": variant, **{key: geometry.get(key, 0) for key in "ABCDEF"}, "evaluated": row["evaluated_candidates"], "feasible": row["common_safe_feasible"]})
    with (args.gate_root / "context_metrics_by_variant.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("variant", "context_count", "valid_candidates", "repair_set_gain_candidates", "p1_loss_candidates", "p2_loss_candidates"))
        writer.writeheader()
        for variant, row in summaries.items():
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    with (args.gate_root / "train_dynamics_by_variant.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ("variant", "initial_vote", "final_vote", "initial_oracle", "final_oracle", "final_member_counts")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, row in summaries.items():
            initial, final = row["initial_train"], row["final_train"]
            writer.writerow({"variant": variant, "initial_vote": initial.get("team_vote_correct_count"), "final_vote": final.get("team_vote_correct_count"), "initial_oracle": initial.get("oracle_correct_count"), "final_oracle": final.get("oracle_correct_count"), "final_member_counts": json.dumps(final.get("per_agent_correct_counts", []))})
    with (args.gate_root / "cost_by_variant.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ("variant", "provider_calls", "solver_calls", "teacher_calls", "critic_calls", "student_calls", "cache_hits", "cache_misses", "input_tokens", "output_tokens", "total_tokens")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, row in summaries.items():
            cost = row["cost"]
            writer.writerow({
                "variant": variant,
                "provider_calls": cost.get("provider_api_calls", cost.get("total_llm_calls")),
                "solver_calls": cost.get("solver_calls"),
                "teacher_calls": sum(int(row.get("funnel", {}).get("teacher_calls", 0)) for row in read_jsonl(args.run_root / "disambiguation_qa" / f"{next(name for name, value in SETTINGS if value == variant)}_seed51" / "candidate_decisions.jsonl")),
                "critic_calls": sum(int(row.get("funnel", {}).get("critic_calls", 0)) for row in read_jsonl(args.run_root / "disambiguation_qa" / f"{next(name for name, value in SETTINGS if value == variant)}_seed51" / "candidate_decisions.jsonl")),
                "student_calls": sum(int(row.get("funnel", {}).get("student_calls", 0)) for row in read_jsonl(args.run_root / "disambiguation_qa" / f"{next(name for name, value in SETTINGS if value == variant)}_seed51" / "candidate_decisions.jsonl")),
                "cache_hits": cost.get("cache_hits"),
                "cache_misses": cost.get("cache_misses"),
                "input_tokens": cost.get("input_tokens", cost.get("prompt_tokens")),
                "output_tokens": cost.get("output_tokens", cost.get("completion_tokens")),
                "total_tokens": cost.get("total_tokens"),
            })
    print(json.dumps(result, indent=2, sort_keys=True))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
