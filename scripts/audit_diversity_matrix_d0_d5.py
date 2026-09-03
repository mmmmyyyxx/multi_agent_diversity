from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts.diversity_matrix_d0_d5_support import (
    AGENTS,
    ARMS,
    ARM_ORDER,
    ROLE_MODEL,
    SOLVER_MODEL,
    UPDATES,
    count_by_member,
    git,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)


def _cell(root: Path, seed: int, arm: str) -> Path:
    return root / f"seed{seed}" / arm


def _actual_counts(decisions: list[dict[str, Any]], run: Path) -> dict[str, int]:
    branch_funnels = [
        branch.get("funnel", {})
        for decision in decisions
        for branch in decision.get("branches", ())
    ]
    candidates = [
        candidate
        for decision in decisions
        for candidate in decision.get("candidates", ())
    ]
    revisions = read_jsonl(run / "loss_blind_generic_revision_events.jsonl")
    return {
        "scheduled_target_opportunities": sum(
            len(row.get("selected_target_ids", ())) for row in decisions
        ),
        "scheduled_source_candidate_slots": sum(
            int(row.get("requested_candidate_count", 0)) for row in branch_funnels
        ),
        "teacher_attempts": sum(int(row.get("teacher_calls", 0)) for row in branch_funnels),
        "critic_attempts": sum(int(row.get("critic_calls", 0)) for row in branch_funnels),
        "student_attempts": sum(int(row.get("student_calls", 0)) for row in branch_funnels),
        "raw_generated_candidates": sum(int(row.get("raw_candidate_count", 0)) for row in branch_funnels),
        "strict_valid_source_candidates": sum(
            candidate.get("candidate_stage") != "loss_blind_generic_revision"
            and bool(candidate.get("evaluation"))
            for candidate in candidates
        ),
        "revision_attempts": len(revisions),
        "strict_valid_revision_candidates": sum(
            candidate.get("candidate_stage") == "loss_blind_generic_revision"
            and bool(candidate.get("evaluation"))
            for candidate in candidates
        ),
        "evaluable_candidates": sum(bool(candidate.get("evaluation")) for candidate in candidates),
        "feasible_candidates": sum(
            bool((candidate.get("constraint") or {}).get("passed"))
            for candidate in candidates
        ),
        "accepted_commits": sum(bool(row.get("accepted_prompt_hash")) for row in decisions),
        "infrastructure_failures": sum(
            int(row.get("infrastructure_failed_updates", 0)) for row in branch_funnels
        ),
    }


def audit(prep_root: Path, run_root: Path) -> dict[str, Any]:
    registry = read_json(prep_root / "registry.json")
    freeze = read_json(prep_root / "source_freeze.json")
    blockers: list[str] = []
    if freeze.get("phase_a_gate") != "PASS":
        blockers.append("phase_a_gate")
    if git("rev-parse", "HEAD") != freeze.get("execution_commit"):
        blockers.append("execution_commit")
    for row in freeze.get("files", ()):  # reports are deliberately outside freeze
        path = ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            blockers.append(f"source_freeze:{row['path']}")

    training = read_json(run_root / "training_execution_summary.json")
    validation = read_json(run_root / "validation_execution_summary.json")
    if training.get("execution_gate") != "PASS":
        blockers.append("training_execution_gate")
    if validation.get("validation_gate") != "PASS":
        blockers.append("validation_execution_gate")
    if training.get("new_test_calls") != 0 or validation.get("new_test_calls") != 0:
        blockers.append("new_test_calls")

    rows: list[dict[str, Any]] = []
    for seed in registry["seeds"]:
        initialization_hashes = set()
        probe_hashes = set()
        initial_prompt_hashes = set()
        for arm in ARM_ORDER:
            run = _cell(run_root, seed, arm)
            try:
                meta = read_json(run / "run_meta.json")
                final = read_json(run / "final_summary.json")
                status = read_json(run / "trajectory_status.json")
                runtime = read_json(run / "matrix_arm_runtime.json")
                validation_result = read_json(
                    run_root / "validation" / f"seed{seed}" / arm
                    / "evaluation_summary_private.json"
                )
                decisions = read_jsonl(run / "candidate_decisions.jsonl")
                frozen_match = read_json(run / "frozen_initialization_match.json")
                cost = read_json(run / "cost_summary.json")
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                blockers.append(f"artifact:{seed}:{arm}:{type(exc).__name__}")
                continue
            planned = 0 if arm == "D0" else UPDATES
            completed = int(meta.get("completed_update_count", -1))
            if status.get("status") != "COMPLETED":
                blockers.append(f"status:{seed}:{arm}")
            if int(meta.get("planned_update_count", -1)) != planned or completed != planned:
                blockers.append(f"update_budget:{seed}:{arm}")
            selection = final.get("selection_summary", {})
            if (
                selection.get("selected_checkpoint_source") != "final_active_state"
                or selection.get("validation_used") is not False
                or int(selection.get("test_evaluation_count", -1)) != 0
                or selection.get("final_test_enabled") is not False
            ):
                blockers.append(f"lifecycle:{seed}:{arm}")
            config = meta.get("config", {})
            if (
                config.get("agent_model") != SOLVER_MODEL
                or config.get("optimizer_model") != ROLE_MODEL
                or config.get("evaluator_model") != ROLE_MODEL
                or float(config.get("temperature", -1)) != 0
                or int(config.get("agents", -1)) != AGENTS
            ):
                blockers.append(f"model_or_team:{seed}:{arm}")
            if meta.get("canonical_experiment_setting") != ARMS[arm]["setting"]:
                blockers.append(f"setting:{seed}:{arm}")
            if meta.get("candidate_acceptance_policy") not in {
                "none", "fixed_peer_monotone_target_or_vote",
            }:
                blockers.append(f"common_safe:{seed}:{arm}")
            if bool(meta.get("compatibility_repair_enabled")):
                blockers.append(f"m2f_enabled:{seed}:{arm}")
            if runtime.get("m2f_enabled") is not False:
                blockers.append(f"runtime_m2f:{seed}:{arm}")
            if runtime.get("test_rows_loaded") != 0 or runtime.get("test_calls") != 0:
                blockers.append(f"runtime_test_access:{seed}:{arm}")
            if validation_result.get("test_rows_loaded") != 0 or validation_result.get("test_calls") != 0:
                blockers.append(f"validation_test_access:{seed}:{arm}")
            if validation_result.get("logical_validation_evaluations") != 1:
                blockers.append(f"validation_count:{seed}:{arm}")
            if validation_result.get("state_mutation") or validation_result.get("checkpoint_mutation"):
                blockers.append(f"validation_mutation:{seed}:{arm}")
            if validation_result.get("final_state_hash") != selection.get("selected_team_prompt_state_hash"):
                blockers.append(f"validation_state:{seed}:{arm}")
            if not frozen_match.get("matched"):
                blockers.append(f"initialization:{seed}:{arm}")
            snapshot = frozen_match.get("initialization_snapshot", {})
            initialization_hashes.add(snapshot.get("initial_train_state_hash"))
            probe_hashes.add(snapshot.get("probe_hash"))
            initial_prompt_hashes.add(json.dumps(snapshot.get("initial_prompt_hashes"), sort_keys=True))
            actual = _actual_counts(decisions, run)
            if actual["infrastructure_failures"]:
                blockers.append(f"infrastructure:{seed}:{arm}")
            if arm == "D0" and decisions:
                blockers.append(f"static_decisions:{seed}")
            if arm != "D0" and len(decisions) != UPDATES:
                blockers.append(f"decision_count:{seed}:{arm}")
            target_counts = count_by_member(decisions, "targets")
            commit_counts = count_by_member(decisions, "commits")
            if arm in {"D2", "D3", "D4", "D5"}:
                protocol = registry["arms"][arm]
                if protocol["target_branch_count"] != 2 or protocol["candidates_per_target_branch"] != 2:
                    blockers.append(f"factorial_budget:{seed}:{arm}")
                if not protocol["generic_revision_enabled"]:
                    blockers.append(f"revision_policy:{seed}:{arm}")
                if actual["revision_attempts"] != actual["strict_valid_source_candidates"]:
                    blockers.append(f"revision_attempt_parity:{seed}:{arm}")
            if arm in {"D2", "D4"}:
                for decision in decisions:
                    selected = set(map(int, decision.get("selected_target_ids", ())))
                    pool = {
                        int(item["agent_id"])
                        for item in decision.get("agent_target_priorities", ())
                    }
                    if not selected.issubset(pool):
                        blockers.append(f"rr_outside_actionable:{seed}:{arm}")
            if arm in {"D4", "D5"}:
                if not runtime.get("teacher_clean_enabled") or not runtime.get("deterministic_hard_gate_enabled"):
                    blockers.append(f"pipeline:{seed}:{arm}")
                if int(cost.get("tokens_by_role", {}).get("critic", -1)) != 0:
                    blockers.append(f"semantic_critic_tokens:{seed}:{arm}")
            rows.append({
                "seed": seed,
                "arm": arm,
                "planned_updates": planned,
                "completed_updates": completed,
                "target_opportunities_by_member": target_counts,
                "accepted_commits_by_member": commit_counts,
                **actual,
            })
        if len(initialization_hashes) != 1 or len(probe_hashes) != 1 or len(initial_prompt_hashes) != 1:
            blockers.append(f"same_start:{seed}")

    expected = len(registry["seeds"]) * len(ARM_ORDER)
    if len(rows) != expected:
        blockers.append("trajectory_inventory")
    # D2-D5 parity is equality of frozen scheduled opportunity budgets and
    # revision eligibility semantics, not equality of realized actionable,
    # valid, or evaluable rows. An arm may scientifically run out of two
    # actionable members without changing its frozen budget.
    for seed in registry["seeds"]:
        factorial = [row for row in rows if row["seed"] == seed and row["arm"] in {"D2", "D3", "D4", "D5"}]
        if len(factorial) == 4:
            if any(
                row["planned_updates"] != UPDATES
                or sum(row["target_opportunities_by_member"]) > UPDATES * 2
                for row in factorial
            ):
                blockers.append(f"scheduled_compute_parity:{seed}")

    return {
        "audit_version": "diversity_matrix_execution_audit_v1",
        "execution_gate": "PASS" if not blockers else "HOLD",
        "scientific_analysis_gate": "PASS" if not blockers else "NOT_RUN",
        "blockers": sorted(set(blockers)),
        "trajectory_count": len(rows),
        "expected_trajectory_count": expected,
        "new_test_calls": 0,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh audit root required")
    result = audit(args.prep_root.resolve(), args.run_root.resolve())
    args.out.mkdir(parents=True)
    write_json(args.out / "execution_audit.json", result)
    print(json.dumps({
        key: result[key]
        for key in ("execution_gate", "scientific_analysis_gate", "trajectory_count", "new_test_calls", "blockers")
    }, indent=2))
    raise SystemExit(0 if result["execution_gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
