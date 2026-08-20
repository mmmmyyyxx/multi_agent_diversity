"""Zero-API reconstruction of the frozen V17 failure-decomposition evidence."""
from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from v17_failure_decomposition_support import (
    ARMS,
    RUN_ID,
    SEEDS,
    SOURCE_COMMIT,
    SOURCE_COMMIT_PREFIX,
    SPLITS,
    ZERO_API_COUNTERS,
    EvidenceError,
    arm_run_dir,
    cache_path,
    classify_hypotheses,
    load_cached_profiles,
    load_examples,
    load_train_profiles,
    mean,
    pct,
    read_json,
    reconstruct_rows,
    sha256_file,
    sha256_text,
    summarise_rows,
    target_metrics,
    training_log_summary,
    transition_summary,
    make_transition_rows,
    write_json,
    write_jsonl,
)


CONTRASTS = (("S0", "S1"), ("S1", "S2"), ("S2", "S3"), ("S3", "S4"), ("S1", "S4"))


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True, encoding="utf-8").strip()


def verify_frozen_source(repo: Path, allow_dirty: bool) -> dict[str, Any]:
    current = git_value(repo, "rev-parse", "HEAD")
    origin = git_value(repo, "rev-parse", "origin/main")
    status = git_value(repo, "status", "--porcelain")
    if not current.startswith(SOURCE_COMMIT_PREFIX) or current != SOURCE_COMMIT:
        raise EvidenceError("V17 failure decomposition must begin at the frozen ef9124e source commit")
    if origin != current:
        raise EvidenceError("origin/main differs from the frozen V17 source commit")
    if status and not allow_dirty:
        raise EvidenceError("tracked worktree is dirty; rerun with --allow_dirty 1 only for audit source edits")
    return {
        "V17_FAILURE_DECOMP_SOURCE_COMMIT": SOURCE_COMMIT,
        "origin_main": origin,
        "repo_dirty_at_audit": bool(status),
        "allow_dirty": bool(allow_dirty),
    }


def _private_row(*, seed: int, arm: str, split: str, row: dict[str, Any]) -> dict[str, Any]:
    return {"seed": seed, "arm": arm, "split": split, **row}


def _member_rows(seed: int, arm: str, split: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for agent_id in range(5):
        correct = sum(int(row["team_correctness"][agent_id]) for row in rows)
        unique = sum(agent_id in row["unique_agents"] for row in rows)
        pivotal = sum(agent_id in row["pivotal_agents"] for row in rows)
        output.append({
            "seed": seed,
            "arm": arm,
            "split": split,
            "agent_id": agent_id,
            "question_count": len(rows),
            "correct_count": correct,
            "accuracy": pct(correct, len(rows)),
            "unique_correct_count": unique,
            "pivotal_correct_count": pivotal,
        })
    return output


def _summary_matches(summary: dict[str, Any], expected: dict[str, Any]) -> None:
    fields = ("row_count", "vote_correct_count", "oracle_correct_count", "per_agent_correct_counts")
    for field in fields:
        if summary.get(field) != expected.get(field):
            raise EvidenceError(f"held-out summary mismatch for {field}")


def _efficiency_rows(seed: int, arm: str, log: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dynamics = list(log["dynamics"])
    accepts = [row for row in dynamics if bool(row.get("accepted"))]
    initial = next((row for row in dynamics if int(row.get("update_index", -2)) == -1), None)
    final = log["final_dynamics"]
    vote_delta = (int(final.get("team_vote_correct_count", 0)) - int(initial.get("team_vote_correct_count", 0))) if initial else 0
    oracle_delta = (int(final.get("oracle_correct_count", 0)) - int(initial.get("oracle_correct_count", 0))) if initial else 0
    row = {
        "seed": seed,
        "arm": arm,
        "planned_updates": log["planned_update_count"],
        "accepted_updates": log["accepted_update_count"],
        "generated_candidates": log["generated_candidate_count"],
        "valid_candidates": log["valid_candidate_count"],
        "passed_candidates": log["passed_candidate_count"],
        "pass_rate": pct(log["passed_candidate_count"], log["generated_candidate_count"]),
        "accepted_rate": pct(log["accepted_update_count"], log["planned_update_count"]),
        "train_vote_delta": vote_delta,
        "train_oracle_delta": oracle_delta,
        "vote_gain_per_commit": pct(vote_delta, log["accepted_update_count"]),
        "oracle_gain_per_commit": pct(oracle_delta, log["accepted_update_count"]),
        "repair_eligible": sum(bool(event.get("repair_eligible")) for event in log["repair_events"]),
        "repair_attempted": sum(bool(event.get("repair_attempted")) for event in log["repair_events"]),
        "repair_feasible": sum(bool(event.get("repair_feasible")) for event in log["repair_events"]),
        "repair_committed": sum(bool(event.get("repair_committed")) for event in log["repair_events"]),
    }
    commits: list[dict[str, Any]] = []
    previous = initial or {}
    for update in sorted((item for item in dynamics if int(item.get("update_index", -1)) >= 0), key=lambda item: int(item["update_index"])):
        if bool(update.get("accepted")):
            commits.append({
                "seed": seed,
                "arm": arm,
                "update_index": int(update["update_index"]),
                "target_agent_id": update.get("target_agent_id"),
                "vote_delta": int(update.get("team_vote_correct_count", 0)) - int(previous.get("team_vote_correct_count", 0)),
                "oracle_delta": int(update.get("oracle_correct_count", 0)) - int(previous.get("oracle_correct_count", 0)),
                "min_member_delta": min(update.get("per_agent_correct_counts", [0])) - min(previous.get("per_agent_correct_counts", [0])),
                "sum_member_delta": sum(update.get("per_agent_correct_counts", [0])) - sum(previous.get("per_agent_correct_counts", [0])),
                "target_member_gain": update.get("target_gain"),
                "vote_gain_count": update.get("vote_gain_count"),
                "vote_loss_count": update.get("vote_loss_count"),
            })
        previous = update
    return row, commits


def run_audit(repo: Path, out_dir: Path, *, allow_dirty: bool) -> dict[str, Any]:
    frozen = verify_frozen_source(repo, allow_dirty)
    historical_protocol = read_json(repo / "reports" / "v17_formal_5arm_3seed_20260813" / "protocol_gate.json")
    if any(historical_protocol.get(name) != "PASS" for name in ("train", "validation", "test", "pre_test_seal")):
        raise EvidenceError("historical V17 protocol gate is not PASS")
    if (
        int(historical_protocol.get("formal_cells", -1)) != 15
        or int(historical_protocol.get("validation_logical_evaluations", -1)) != 15
        or int(historical_protocol.get("test_logical_evaluations", -1)) != 15
        or any(int(historical_protocol.get(name, -1)) != 0 for name in ("state_mutations", "selection_changes", "checkpoint_mutations"))
    ):
        raise EvidenceError("historical V17 lifecycle/isolation gate mismatch")
    preregistration = read_json(repo / "experiments" / "v17_formal_5arm_3seed_20260813" / "preregistration.json")
    if bool(preregistration.get("test_used_for_selection", True)):
        raise EvidenceError("historical test-used-for-selection invariant failed")
    source_freeze = read_json(repo / "reports" / "v17_formal_5arm_3seed_20260813" / "source_freeze_sanitized.json")
    if source_freeze.get("source_freeze_status") != "PASS":
        raise EvidenceError("historical source freeze is not PASS")
    if out_dir.exists() and any(out_dir.iterdir()):
        expected = {
            "evidence_inventory.json",
            "reconstructed_row_metrics.jsonl",
            "private_question_transitions.jsonl",
            "private_member_transitions.jsonl",
            "analysis_metrics_private.json",
        }
        existing = {path.name for path in out_dir.iterdir() if path.is_file()}
        if not existing <= expected:
            raise EvidenceError("private audit output contains non-audit files")
    out_dir.mkdir(parents=True, exist_ok=True)
    examples_by_split: dict[str, list[Any]] = {}
    raw_by_split: dict[str, list[dict[str, str]]] = {}
    for split in SPLITS:
        examples_by_split[split], raw_by_split[split] = load_examples(repo, split)

    all_rows: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    all_metrics: dict[tuple[int, str, str], dict[str, Any]] = {}
    logs: dict[tuple[int, str], dict[str, Any]] = {}
    private_rows: list[dict[str, Any]] = []
    private_members: list[dict[str, Any]] = []
    inventory_cells: list[dict[str, Any]] = []
    split_hashes = {split: sha256_file(repo / "strict_splits_bbh_seed42" / "disambiguation_qa" / {"train": "opt.csv", "validation": "val.csv", "test": "test.csv"}[split]) for split in SPLITS}

    for seed in SEEDS:
        for arm in ARMS:
            train_profiles, prompt_hashes = load_train_profiles(repo, seed, arm, len(examples_by_split["train"]))
            logs[(seed, arm)] = training_log_summary(repo, seed, arm)
            for split in SPLITS:
                profiles = train_profiles if split == "train" else load_cached_profiles(repo, split, seed, arm, prompt_hashes, examples_by_split[split])
                rows, behavior = reconstruct_rows(examples=examples_by_split[split], profiles=profiles, seed=seed)
                summary = summarise_rows(rows, behavior)
                heldout_state: dict[str, Any] = {}
                if split != "train":
                    heldout = read_json(cache_path(repo, split, seed, arm).parent / "evaluation_summary_private.json")
                    _summary_matches(heldout, {
                        "row_count": len(rows),
                        "vote_correct_count": summary["vote_correct_count"],
                        "oracle_correct_count": summary["oracle_correct_count"],
                        "per_agent_correct_counts": summary["per_agent_correct_count"],
                    })
                    if int(heldout.get("logical_evaluation_count", -1)) != 1:
                        raise EvidenceError(f"held-out logical evaluation count mismatch: {split}/seed{seed}/{arm}")
                    if any(bool(heldout.get(field)) for field in ("state_mutation", "selection_change", "checkpoint_mutation")):
                        raise EvidenceError(f"held-out state isolation failure: {split}/seed{seed}/{arm}")
                    heldout_state = {
                        "final_state_hash": str(heldout.get("final_state_hash", "")),
                        "checkpoint_sha256": str(heldout.get("checkpoint_sha256", "")),
                        "logical_evaluation_count": int(heldout["logical_evaluation_count"]),
                    }
                all_rows[(seed, arm, split)] = rows
                all_metrics[(seed, arm, split)] = summary
                private_rows.extend(_private_row(seed=seed, arm=arm, split=split, row=row) for row in rows)
                private_members.extend(_member_rows(seed, arm, split, rows))
                inventory_cells.append({
                    "seed": seed,
                    "arm": arm,
                    "split": split,
                    "question_count": len(rows),
                    "source_kind": "checkpoint_active_profiles" if split == "train" else "read_only_solver_cache",
                    "final_prompt_hash_set_sha256": sha256_text("|".join(sorted(prompt_hashes))),
                    "split_file_sha256": split_hashes[split],
                    "vote_correct_count": summary["vote_correct_count"],
                    "oracle_correct_count": summary["oracle_correct_count"],
                    "per_agent_correct_count": summary["per_agent_correct_count"],
                    "raw_payload_exported": False,
                    **heldout_state,
                })

    for seed in SEEDS:
        for arm in ARMS:
            val = next(row for row in inventory_cells if row["seed"] == seed and row["arm"] == arm and row["split"] == "validation")
            test = next(row for row in inventory_cells if row["seed"] == seed and row["arm"] == arm and row["split"] == "test")
            if not val["final_state_hash"] or val["final_state_hash"] != test["final_state_hash"]:
                raise EvidenceError(f"validation/test final-state mismatch: seed{seed}/{arm}")
            if val["checkpoint_sha256"] != test["checkpoint_sha256"]:
                raise EvidenceError(f"validation/test checkpoint mismatch: seed{seed}/{arm}")

    transitions: list[dict[str, Any]] = []
    transition_summaries: list[dict[str, Any]] = []
    for seed in SEEDS:
        for left, right in CONTRASTS:
            contrast = f"{left}_to_{right}"
            for split in ("validation", "test"):
                rows = make_transition_rows(all_rows[(seed, left, split)], all_rows[(seed, right, split)], seed=seed, split=split, contrast=contrast)
                transitions.extend(rows)
                transition_summaries.append({"seed": seed, "split": split, "contrast": contrast, **transition_summary(rows)})

    split_metrics: list[dict[str, Any]] = []
    for (seed, arm, split), summary in sorted(all_metrics.items()):
        split_metrics.append({"seed": seed, "arm": arm, "split": split, **summary})
    efficiency: list[dict[str, Any]] = []
    commit_efficiency: list[dict[str, Any]] = []
    concentration: list[dict[str, Any]] = []
    w1_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            e_row, e_commits = _efficiency_rows(seed, arm, logs[(seed, arm)])
            efficiency.append(e_row)
            commit_efficiency.extend(e_commits)
            if arm in {"S1", "S2"}:
                metrics = target_metrics(logs[(seed, arm)])
                concentration.append({"seed": seed, "arm": arm, **metrics})
            if arm == "S2":
                for audit in logs[(seed, arm)]["target_audit"]:
                    for priority in audit.get("priorities", []):
                        w1_rows.append({
                            "seed": seed,
                            "update_index": int(audit.get("update_index", -1)),
                            "agent_id": int(priority["agent_id"]),
                            "selected": int(bool(priority.get("selected"))),
                            "selection_rank": priority.get("selection_rank"),
                            "expected_update_value": priority.get("expected_update_value"),
                            "opportunity_value": priority.get("opportunity_value"),
                            "repairability_discount": priority.get("repairability_discount"),
                            "active_lane_size": priority.get("active_lane_size"),
                            "branch_failure_count": priority.get("branch_failure_count", priority.get("failure_count")),
                            "B": (
                                0.5 * float(priority.get("normalized_direct_fix", 0.0) or 0.0)
                                + 0.3 * float(priority.get("normalized_support_margin", 0.0) or 0.0)
                                + 0.2 * float(priority.get("normalized_uplift_deficit", 0.0) or 0.0)
                            ),
                            "normalized_direct_fix": priority.get("normalized_direct_fix"),
                            "normalized_support_margin": priority.get("normalized_support_margin"),
                            "normalized_uplift_deficit": priority.get("normalized_uplift_deficit"),
                            "normalized_wait": priority.get("normalized_wait"),
                            "service_portfolio_size": priority.get("service_portfolio_size"),
                        })

    seed_evidence: list[dict[str, Any]] = []
    for seed in SEEDS:
        s1_train, s2_train = all_metrics[(seed, "S1", "train")], all_metrics[(seed, "S2", "train")]
        s1_test, s2_test = all_metrics[(seed, "S1", "test")], all_metrics[(seed, "S2", "test")]
        s1_val, s2_val = all_metrics[(seed, "S1", "validation")], all_metrics[(seed, "S2", "validation")]
        s1_conc = next(row for row in concentration if row["seed"] == seed and row["arm"] == "S1")
        s2_conc = next(row for row in concentration if row["seed"] == seed and row["arm"] == "S2")
        selected = logs[(seed, "S2")]["selected_counts"]
        per_agent_delta = [s2_test["per_agent_accuracy"][agent] - s1_test["per_agent_accuracy"][agent] for agent in range(5)]
        mean_target_count = sum(selected) / len(selected)
        high_target_ids = [agent for agent, count in enumerate(selected) if count > mean_target_count]
        low_target_ids = [agent for agent, count in enumerate(selected) if count < mean_target_count]
        high_target_delta = mean(per_agent_delta[agent] for agent in high_target_ids) or 0.0
        low_target_delta = mean(per_agent_delta[agent] for agent in low_target_ids) or 0.0
        s1_eff = next(row for row in efficiency if row["seed"] == seed and row["arm"] == "S1")
        s2_eff = next(row for row in efficiency if row["seed"] == seed and row["arm"] == "S2")
        # Five independent specialization measures, direction fixed before status classification.
        train_measures = [
            sum(s2_train["unique_correct_count"]) - sum(s1_train["unique_correct_count"]),
            sum(s2_train["pivotal_correct_count"]) - sum(s1_train["pivotal_correct_count"]),
            (s2_train["n_eff"] or 0.0) - (s1_train["n_eff"] or 0.0),
            (s1_train["mean_pairwise_correctness_correlation"] or 0.0) - (s2_train["mean_pairwise_correctness_correlation"] or 0.0),
            s2_train["oracle_accuracy"] - s1_train["oracle_accuracy"],
        ]
        test_measures = [
            sum(s2_test["unique_correct_count"]) - sum(s1_test["unique_correct_count"]),
            sum(s2_test["pivotal_correct_count"]) - sum(s1_test["pivotal_correct_count"]),
            (s2_test["n_eff"] or 0.0) - (s1_test["n_eff"] or 0.0),
            (s1_test["mean_pairwise_correctness_correlation"] or 0.0) - (s2_test["mean_pairwise_correctness_correlation"] or 0.0),
            s2_test["oracle_accuracy"] - s1_test["oracle_accuracy"],
        ]
        seed_evidence.append({
            "seed": seed,
            "s2_s1_train_delta": s2_train["vote_accuracy"] - s1_train["vote_accuracy"],
            "s2_s1_test_delta": s2_test["vote_accuracy"] - s1_test["vote_accuracy"],
            "s2_s1_gap": (s2_train["vote_accuracy"] - s1_train["vote_accuracy"]) - (s2_test["vote_accuracy"] - s1_test["vote_accuracy"]),
            "s2_oracle_minus_s1": s2_test["oracle_accuracy"] - s1_test["oracle_accuracy"],
            "s2_vote_minus_s1": s2_test["vote_accuracy"] - s1_test["vote_accuracy"],
            "s2_oracle_vote_gap_minus_s1": (s2_test["oracle_accuracy"] - s2_test["vote_accuracy"]) - (s1_test["oracle_accuracy"] - s1_test["vote_accuracy"]),
            "s2_coverage_delta": s2_test["oracle_covered_but_vote_wrong_rate"] - s1_test["oracle_covered_but_vote_wrong_rate"],
            "s2_entropy_minus_s1": s2_conc["target_entropy"] - s1_conc["target_entropy"],
            "high_target_agent_test_delta": high_target_delta,
            "low_target_agent_test_delta": low_target_delta,
            "s1_accepted_updates": s1_eff["accepted_updates"],
            "s2_accepted_updates": s2_eff["accepted_updates"],
            "s1_test_gain_per_commit": pct(s1_test["vote_correct_count"] - all_metrics[(seed, "S0", "test")]["vote_correct_count"], s1_eff["accepted_updates"]),
            "s2_test_gain_per_commit": pct(s2_test["vote_correct_count"] - all_metrics[(seed, "S0", "test")]["vote_correct_count"], s2_eff["accepted_updates"]),
            "s2_specialization_train_minus_s1": sum(1 for item in train_measures if item > 0),
            "s2_specialization_test_minus_s1": sum(1 for item in test_measures if item > 0),
            "specialization_measure_count": sum(1 for train, test in zip(train_measures, test_measures, strict=True) if train > 0 and test <= 0),
            "s2_validation_minus_s1": s2_val["vote_accuracy"] - s1_val["vote_accuracy"],
        })
    hypotheses = classify_hypotheses(seed_evidence)
    schedule_overlap: list[dict[str, Any]] = []
    for seed in SEEDS:
        s1 = logs[(seed, "S1")]["targets_by_update"]
        s2 = logs[(seed, "S2")]["targets_by_update"]
        for update_index in range(8):
            left = set(s1.get(str(update_index), []))
            right = set(s2.get(str(update_index), []))
            schedule_overlap.append({
                "seed": seed,
                "update_index": update_index,
                "S1_target_count": len(left),
                "S2_target_count": len(right),
                "target_overlap_count": len(left & right),
                "target_replacement_count": len(left ^ right),
                "S2_only_target_count": len(right - left),
            })

    inventory = {
        "audit_id": RUN_ID,
        **frozen,
        "source_commit_verified": True,
        "evidence_complete": len(inventory_cells) == 45,
        "cells": inventory_cells,
        "split_sizes": {split: len(examples_by_split[split]) for split in SPLITS},
        "api_model_solver_optimizer_evaluator_call_counts": ZERO_API_COUNTERS,
        "raw_questions_answers_prompts_traces_exported": False,
        "low_api_followup_required": False,
        "historical_protocol_gate_verified": True,
        "historical_execution_commit": source_freeze["git_head"],
        "historical_source_tree_hash": source_freeze["working_tree_source_hash"],
        "test_used_for_selection": False,
        "validation_test_final_state_hashes_match": True,
    }
    analysis = {
        "audit_id": RUN_ID,
        "source_commit": SOURCE_COMMIT,
        "split_metrics": split_metrics,
        "transition_summaries": transition_summaries,
        "efficiency": efficiency,
        "commit_efficiency": commit_efficiency,
        "concentration": concentration,
        "w1_rows": w1_rows,
        "schedule_overlap": schedule_overlap,
        "seed_evidence": seed_evidence,
        "hypotheses": hypotheses,
        "api_model_solver_optimizer_evaluator_call_counts": ZERO_API_COUNTERS,
    }
    write_json(out_dir / "evidence_inventory.json", inventory)
    write_jsonl(out_dir / "reconstructed_row_metrics.jsonl", private_rows)
    write_jsonl(out_dir / "private_question_transitions.jsonl", transitions)
    write_jsonl(out_dir / "private_member_transitions.jsonl", private_members)
    write_json(out_dir / "analysis_metrics_private.json", analysis)
    return {"inventory": inventory, "analysis": analysis, "out_dir": str(out_dir)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--out_dir", default=f"runs/{RUN_ID}")
    parser.add_argument("--allow_dirty", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    repo = Path(args.workspace).resolve()
    out_dir = (repo / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    if repo not in out_dir.parents:
        raise EvidenceError("private audit output must remain beneath the repository")
    result = run_audit(repo, out_dir, allow_dirty=bool(args.allow_dirty))
    print(f"PASS zero-api V17 failure decomposition: {result['out_dir']}")


if __name__ == "__main__":
    main()
