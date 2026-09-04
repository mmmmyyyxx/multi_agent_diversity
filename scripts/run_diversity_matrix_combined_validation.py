from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.cli import _load
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.peer_state import build_team_vote_state
from multi_dataset_diverse_rl.persistence.checkpoint import restore_checkpoint
from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from scripts.diversity_matrix_d0_d5_support import (
    AGENTS,
    ARM_ORDER,
    ROLE_MODEL,
    SOLVER_MODEL,
    git,
    read_json,
    read_jsonl,
    sha256_file,
    source_inventory,
    write_json,
)


TRAIN_ROOT = ROOT / "runs" / "diversity_matrix_d0_d5_20260903"
RECOVERY_AUDIT = TRAIN_ROOT / "audit_recovery_v1" / "execution_audit.json"
PREP_ROOT = ROOT / "runs" / "diversity_matrix_d0_d5_combined_validation_prep_20260904"
FORMER_TEST_ROOT = ROOT / "runs" / "diversity_matrix_d0_d5_former_test125_20260904"
REPORT_ROOT = ROOT / "reports" / "diversity_matrix_d0_d5_combined_validation175_20260904"
AUTH_ENV = "DIVERSITY_MATRIX_COMBINED_VALIDATION_AUTHORIZED"
SEEDS = (72, 73, 74)
ORIGINAL_TRAINING_COMMIT = "ca2c4b2e7e78d5594b702298c7a392ed3ca5ee28"
VALIDATOR_FIX_COMMIT = "16abda4f8ec65caed5ad8cd2ae2005dc16a416ec"
AMENDMENT_VERSION = "combined_development_validation175_posthoc_v1"
PROTECTED_NAMES = (
    "training_checkpoint.json",
    "run_meta.json",
    "final_summary.json",
    "trajectory_status.json",
    "history.json",
    "candidate_decisions.jsonl",
    "training_dynamics.jsonl",
)


def _cell(root: Path, seed: int, arm: str) -> Path:
    return root / f"seed{seed}" / arm


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _protected_manifest() -> list[dict[str, Any]]:
    paths = [
        TRAIN_ROOT / "training_execution_summary.json",
        TRAIN_ROOT / "validation_execution_summary.json",
        RECOVERY_AUDIT,
    ]
    for seed in SEEDS:
        for arm in ARM_ORDER:
            paths.extend(_cell(TRAIN_ROOT, seed, arm) / name for name in PROTECTED_NAMES)
            validation = _cell(TRAIN_ROOT / "validation", seed, arm)
            paths.extend((
                validation / "evaluation_summary_private.json",
                validation / "validation_rows_sanitized.jsonl",
            ))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"protected artifacts missing: {len(missing)}")
    return [
        {"path": _relative(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(paths)
    ]


def _verify_manifest(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    failures = []
    for row in rows:
        path = ROOT / str(row["path"])
        if not path.is_file():
            failures.append(f"missing:{row['path']}")
        elif path.stat().st_size != int(row["size"]) or sha256_file(path) != row["sha256"]:
            failures.append(f"changed:{row['path']}")
    return failures


def _verify_source(freeze: Mapping[str, Any]) -> None:
    if git("rev-parse", "HEAD") != freeze["execution_commit"]:
        raise RuntimeError("combined-validation source commit mismatch")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("combined-validation requires a clean tracked worktree")
    for row in freeze["files"]:
        path = ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"combined-validation source mismatch: {row['path']}")


def prepare(prep_root: Path, former_test_root: Path, report_root: Path) -> dict[str, Any]:
    if any(path.exists() for path in (prep_root, former_test_root, report_root)):
        raise FileExistsError("prep, FormerTest125, and report roots must be fresh")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("prepare requires a clean tracked worktree")
    validation = read_json(TRAIN_ROOT / "validation_execution_summary.json")
    recovery_audit = read_json(RECOVERY_AUDIT)
    if validation.get("validation_gate") != "PASS" or validation.get("logical_validation_evaluation_count") != 18:
        raise RuntimeError("Validation50 must PASS 18/18 before amendment")
    if recovery_audit.get("execution_gate") != "PASS" or recovery_audit.get("blockers"):
        raise RuntimeError("recovery audit must PASS before amendment")
    protected = _protected_manifest()
    files, tree_hash = source_inventory()
    prep_root.mkdir(parents=True)
    amendment = {
        "amendment_version": AMENDMENT_VERSION,
        "status": "AUTHORIZED_BEFORE_FORMER_TEST_RESULTS",
        "original_design_test_policy": "prohibited_zero_rows_loaded_zero_calls",
        "new_analysis_role": "former_test_converted_to_development_validation",
        "validation50_size": 50,
        "former_test125_size": 125,
        "combined_development_validation_size": 175,
        "combination_rule": "sum_correct_counts_then_divide_by_175",
        "equal_accuracy_averaging_prohibited": True,
        "test_status": "converted_to_development_validation",
        "untouched_heldout_test_remaining": False,
        "checkpoint_selection": "none_final_active_state",
        "training_rerun": False,
        "method_changed": False,
        "user_authorized": True,
        "solver_model": SOLVER_MODEL,
        "evaluator_model": ROLE_MODEL,
        "seeds": list(SEEDS),
        "arm_order": list(ARM_ORDER),
        "logical_former_test_evaluations": 18,
        "rows_per_evaluation": 125,
        "original_training_commit": ORIGINAL_TRAINING_COMMIT,
        "validation_recovery_commit": VALIDATOR_FIX_COMMIT,
        "former_test_execution_commit": git("rev-parse", "HEAD"),
    }
    write_json(prep_root / "amendment.json", amendment)
    write_json(prep_root / "pre_former_test_seal.json", {
        "seal_version": "diversity_matrix_pre_former_test_seal_v1",
        "gate": "PASS",
        "protected_artifact_count": len(protected),
        "protected_artifacts": protected,
        "validation50_evaluation_count": 18,
        "former_test_evaluations_so_far": 0,
    })
    write_json(prep_root / "source_freeze.json", {
        "freeze_version": "diversity_matrix_combined_validation_source_v1",
        "execution_commit": git("rev-parse", "HEAD"),
        "source_tree_hash": tree_hash,
        "files": files,
        "gate": "PASS",
    })
    return {"gate": "PASS", "protected_artifact_count": len(protected), **amendment}


def _config(meta: Mapping[str, Any], out_dir: Path, cache: Path) -> Config:
    values = dict(meta["config"])
    values.update({
        "out_dir": str(out_dir.resolve()),
        "shared_solver_cache_path": str(cache.resolve()),
        "resume_from_checkpoint": False,
        "final_test_enabled": False,
        "preserve_final_checkpoint": False,
    })
    return Config.from_flat(**values)


async def _evaluate_former_test_cell(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    meta = read_json(run_dir / "run_meta.json")
    checkpoint_path = run_dir / "training_checkpoint.json"
    checkpoint = read_json(checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    cache = out_dir.parent / "_solver_cache.sqlite"
    cfg = _config(meta, out_dir, cache)
    system = PromptEnsembleOptimizationSystem(cfg)
    train = _load(cfg.data.train_path, cfg.data.train_size, cfg.data.dataset_format)
    former_test = _load(cfg.data.test_path, cfg.data.test_size, cfg.data.dataset_format)
    system.set_run_identity(RunIdentity(**checkpoint["run_identity"]))
    system.proposal_memory_run_id = str(checkpoint["proposal_memory_run_id"])
    system.fixed_probe = system.build_probe(train[: cfg.evaluation.candidate_eval_pool_size])
    restore_checkpoint(system, checkpoint)
    system.llm.calls = []
    state_before = system.team_prompt_state_hash()
    expected = str(meta["final_state_selection"]["selected_team_prompt_state_hash"])
    if state_before != expected:
        raise RuntimeError("frozen final-state mismatch before FormerTest125")
    metrics = await system.evaluate_dataset(former_test)
    if system.team_prompt_state_hash() != state_before:
        raise RuntimeError("FormerTest125 mutated final state")
    if sha256_file(checkpoint_path) != checkpoint_hash:
        raise RuntimeError("FormerTest125 mutated checkpoint")
    rows = []
    for index, example in enumerate(system._last_evaluated_examples):
        state = build_team_vote_state(
            question_hash=example.question_hash,
            gold_answer=example.gold_answer,
            answers=[profile[index].answer for profile in system._last_evaluated_profiles],
            valid_vector=[profile[index].valid for profile in system._last_evaluated_profiles],
            normalize_answer=system.normalize_answer,
            match_answer=system.match_answer,
            tie_break=system.protocol.tie_policy,
            seed=cfg.training.seed,
        )
        rows.append({
            "example_id_hash": state.question_hash,
            "G": int(state.gold_vote_count),
            "H": int(state.largest_wrong_vote_count),
            "M": int(state.plurality_margin),
            "vote_correct": bool(state.vote_correct),
            "member_correctness": list(map(bool, state.team_correctness)),
            "member_validity": list(map(bool, state.team_validity)),
        })
    cost = system.cost_summary()
    result = {
        "evaluation_version": "diversity_matrix_former_test125_v1",
        "seed": int(meta["config"]["seed"]),
        "arm": out_dir.name,
        "setting": meta["canonical_experiment_setting"],
        "logical_evaluation_count": 1,
        "row_count": len(rows),
        "vote_correct_count": int(metrics.vote_correct_count),
        "vote_accuracy": float(metrics.plurality_vote_acc),
        "oracle_correct_count": sum(row["G"] > 0 for row in rows),
        "oracle_accuracy": sum(row["G"] > 0 for row in rows) / len(rows),
        "per_agent_correct_counts": list(metrics.per_agent_correct_counts),
        "per_agent_invalid_counts": [
            sum(not row["member_validity"][member] for row in rows)
            for member in range(AGENTS)
        ],
        "final_state_hash": state_before,
        "checkpoint_sha256": checkpoint_hash,
        "provider_calls": int(cost["successful_llm_calls"]),
        "total_tokens": int(cost["total_tokens"]),
        "state_mutation": False,
        "checkpoint_mutation": False,
        "selection_change": False,
        "analysis_role": "former_test_converted_to_development_validation",
    }
    out_dir.mkdir(parents=True)
    write_json(out_dir / "evaluation_summary_private.json", result)
    with (out_dir / "former_test_rows_sanitized.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return result


def run(prep_root: Path, former_test_root: Path) -> dict[str, Any]:
    if os.environ.get(AUTH_ENV) != "1":
        raise RuntimeError(f"set {AUTH_ENV}=1 for the explicitly authorized run")
    amendment = read_json(prep_root / "amendment.json")
    freeze = read_json(prep_root / "source_freeze.json")
    seal = read_json(prep_root / "pre_former_test_seal.json")
    if amendment.get("status") != "AUTHORIZED_BEFORE_FORMER_TEST_RESULTS":
        raise RuntimeError("amendment authorization missing")
    _verify_source(freeze)
    failures = _verify_manifest(seal["protected_artifacts"])
    if failures:
        raise RuntimeError(f"pre-FormerTest125 seal changed: {failures[:3]}")
    if former_test_root.exists():
        existing = list(former_test_root.rglob("evaluation_summary_private.json"))
        if existing:
            raise RuntimeError("completed FormerTest125 cells cannot be rerun")
        raise FileExistsError("FormerTest125 root must be fresh")
    results = []
    former_test_root.mkdir(parents=True)
    for seed in SEEDS:
        for arm in ARM_ORDER:
            out = _cell(former_test_root, seed, arm)
            results.append(asyncio.run(_evaluate_former_test_cell(
                _cell(TRAIN_ROOT, seed, arm), out,
            )))
    payload = {
        "execution_version": "diversity_matrix_former_test125_execution_v1",
        "gate": "PASS" if len(results) == 18 else "HOLD",
        "logical_evaluation_count": len(results),
        "row_count": sum(row["row_count"] for row in results),
        "provider_calls": sum(row["provider_calls"] for row in results),
        "total_tokens": sum(row["total_tokens"] for row in results),
        "test_status": "converted_to_development_validation",
    }
    write_json(former_test_root / "execution_summary.json", payload)
    return payload


def audit(prep_root: Path, former_test_root: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("fresh audit root required")
    amendment = read_json(prep_root / "amendment.json")
    seal = read_json(prep_root / "pre_former_test_seal.json")
    blockers = _verify_manifest(seal["protected_artifacts"])
    rows = []
    for seed in SEEDS:
        for arm in ARM_ORDER:
            run = _cell(TRAIN_ROOT, seed, arm)
            cell = _cell(former_test_root, seed, arm)
            try:
                result = read_json(cell / "evaluation_summary_private.json")
                evidence = read_jsonl(cell / "former_test_rows_sanitized.jsonl")
                meta = read_json(run / "run_meta.json")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                blockers.append(f"artifact:{seed}:{arm}:{type(exc).__name__}")
                continue
            expected_state = meta["final_state_selection"]["selected_team_prompt_state_hash"]
            checks = {
                "identity": result.get("seed") == seed and result.get("arm") == arm,
                "logical_count": result.get("logical_evaluation_count") == 1,
                "row_count": result.get("row_count") == 125 and len(evidence) == 125,
                "state": result.get("final_state_hash") == expected_state,
                "checkpoint": result.get("checkpoint_sha256") == sha256_file(run / "training_checkpoint.json"),
                "no_mutation": not result.get("state_mutation") and not result.get("checkpoint_mutation"),
                "no_selection": not result.get("selection_change"),
                "analysis_role": result.get("analysis_role") == "former_test_converted_to_development_validation",
                "vote_count": sum(bool(row["vote_correct"]) for row in evidence) == result.get("vote_correct_count"),
                "oracle_count": sum(int(row["G"]) > 0 for row in evidence) == result.get("oracle_correct_count"),
            }
            blockers.extend(f"{seed}:{arm}:{name}" for name, passed in checks.items() if not passed)
            rows.append({"seed": seed, "arm": arm, **checks})
    execution = read_json(former_test_root / "execution_summary.json")
    if execution.get("gate") != "PASS" or execution.get("logical_evaluation_count") != 18:
        blockers.append("execution_summary")
    if amendment.get("test_status") != "converted_to_development_validation":
        blockers.append("amendment_semantics")
    payload = {
        "audit_version": "diversity_matrix_former_test125_audit_v1",
        "gate": "PASS" if not blockers and len(rows) == 18 else "HOLD",
        "blockers": sorted(set(blockers)),
        "logical_evaluation_count": len(rows),
        "row_count": sum(125 for _ in rows),
        "protected_artifact_count": len(seal["protected_artifacts"]),
        "protected_artifacts_unchanged": not _verify_manifest(seal["protected_artifacts"]),
        "test_status": "converted_to_development_validation",
        "rows": rows,
    }
    out.mkdir(parents=True)
    write_json(out / "audit.json", payload)
    return payload


def combine_counts(validation: Mapping[str, Any], former_test: Mapping[str, Any]) -> dict[str, Any]:
    total = int(validation["validation_row_count"]) + int(former_test["row_count"])
    if total != 175:
        raise ValueError("Combined Development Validation must contain 175 rows")
    vote = int(validation["vote_correct_count"]) + int(former_test["vote_correct_count"])
    oracle = int(validation["oracle_correct_count"]) + int(former_test["oracle_correct_count"])
    member = [
        int(left) + int(right)
        for left, right in zip(validation["per_agent_correct_counts"], former_test["per_agent_correct_counts"])
    ]
    return {
        "row_count": total,
        "vote_correct_count": vote,
        "vote_accuracy": vote / total,
        "oracle_correct_count": oracle,
        "oracle_accuracy": oracle / total,
        "per_agent_correct_counts": member,
        "per_agent_accuracies": [value / total for value in member],
    }


def _train_metrics(run: Path) -> dict[str, Any]:
    history = read_json(run / "history.json")[-1]["active_probe"]
    if (run / "training_dynamics.jsonl").stat().st_size:
        oracle = read_jsonl(run / "training_dynamics.jsonl")[-1]["oracle_correct_count"]
    else:
        oracle = read_json(run / "frozen_initialization_match.json")["initialization_snapshot"]["initial_team_outcome"]["oracle_correct_count"]
    decisions = read_jsonl(run / "candidate_decisions.jsonl")
    return {
        "train_vote_accuracy": float(history["vote_acc"]),
        "train_oracle_accuracy": int(oracle) / 75,
        "accepted_commits": sum(bool(row.get("accepted_prompt_hash")) for row in decisions),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(prep_root: Path, former_test_root: Path, audit_root: Path, report_root: Path) -> dict[str, Any]:
    if report_root.exists():
        raise FileExistsError("fresh report root required")
    gate = read_json(audit_root / "audit.json")
    if gate.get("gate") != "PASS":
        raise RuntimeError("FormerTest125 audit must PASS before analysis")
    rows = []
    for seed in SEEDS:
        for arm in ARM_ORDER:
            run = _cell(TRAIN_ROOT, seed, arm)
            val_dir = _cell(TRAIN_ROOT / "validation", seed, arm)
            test_dir = _cell(former_test_root, seed, arm)
            validation = read_json(val_dir / "evaluation_summary_private.json")
            former_test = read_json(test_dir / "evaluation_summary_private.json")
            val_rows = read_jsonl(val_dir / "validation_rows_sanitized.jsonl")
            test_rows = read_jsonl(test_dir / "former_test_rows_sanitized.jsonl")
            if len({row["example_id_hash"] for row in val_rows + test_rows}) != 175:
                raise RuntimeError(f"Validation50/FormerTest125 overlap or duplicate: {seed}:{arm}")
            combined = combine_counts(validation, former_test)
            train = _train_metrics(run)
            rows.append({
                "seed": seed,
                "arm": arm,
                **train,
                "validation50_vote_accuracy": validation["vote_accuracy"],
                "validation50_oracle_accuracy": validation["oracle_accuracy"],
                "former_test125_vote_accuracy": former_test["vote_accuracy"],
                "former_test125_oracle_accuracy": former_test["oracle_accuracy"],
                "combined175_vote_accuracy": combined["vote_accuracy"],
                "combined175_oracle_accuracy": combined["oracle_accuracy"],
                "combined175_oracle_vote_gap": combined["oracle_accuracy"] - combined["vote_accuracy"],
                "combined175_member_accuracies": "|".join(f"{value:.6f}" for value in combined["per_agent_accuracies"]),
            })
    aggregate = []
    for arm in ARM_ORDER:
        arm_rows = [row for row in rows if row["arm"] == arm]
        item: dict[str, Any] = {"arm": arm, "seed_count": len(arm_rows)}
        for key in (
            "train_vote_accuracy", "train_oracle_accuracy", "accepted_commits",
            "validation50_vote_accuracy", "validation50_oracle_accuracy",
            "former_test125_vote_accuracy", "former_test125_oracle_accuracy",
            "combined175_vote_accuracy", "combined175_oracle_accuracy",
            "combined175_oracle_vote_gap",
        ):
            values = [float(row[key]) for row in arm_rows]
            item[f"mean_{key}"] = statistics.mean(values)
            item[f"min_{key}"] = min(values)
            item[f"max_{key}"] = max(values)
        aggregate.append(item)
    contrasts = []
    for name, left, right in (
        ("D1_minus_D0_generic", "D1", "D0"),
        ("D2_minus_D1_rr_dual_generic", "D2", "D1"),
        ("D3_minus_D2_w1_vs_rr_generic", "D3", "D2"),
        ("D4_minus_D2_rce_under_rr", "D4", "D2"),
        ("D5_minus_D3_rce_under_w1", "D5", "D3"),
        ("D5_minus_D4_w1_vs_rr_rce", "D5", "D4"),
    ):
        deltas = []
        for seed in SEEDS:
            lrow = next(row for row in rows if row["seed"] == seed and row["arm"] == left)
            rrow = next(row for row in rows if row["seed"] == seed and row["arm"] == right)
            deltas.append(lrow["combined175_vote_accuracy"] - rrow["combined175_vote_accuracy"])
        contrasts.append({
            "contrast": name,
            "seed72_delta": deltas[0],
            "seed73_delta": deltas[1],
            "seed74_delta": deltas[2],
            "mean_delta": statistics.mean(deltas),
        })
    report_root.mkdir(parents=True)
    _write_csv(report_root / "per_seed_arm_results.csv", rows)
    _write_csv(report_root / "aggregate_arm_results.csv", aggregate)
    _write_csv(report_root / "contrast_results.csv", contrasts)
    amendment = read_json(prep_root / "amendment.json")
    summary = {
        "report_version": "diversity_matrix_combined_validation175_report_v1",
        "status": "PASS",
        "training_trajectories": 18,
        "validation50_evaluations": 18,
        "former_test125_evaluations": 18,
        "combined175_cells": 18,
        "test_status": amendment["test_status"],
        "untouched_heldout_test_remaining": False,
        "original_training_commit": ORIGINAL_TRAINING_COMMIT,
        "validation_recovery_commit": VALIDATOR_FIX_COMMIT,
        "former_test_execution_commit": amendment["former_test_execution_commit"],
        "aggregate": aggregate,
        "contrasts": contrasts,
    }
    write_json(report_root / "summary.json", summary)
    write_json(report_root / "amendment.json", amendment)
    lines = [
        "# D0-D5 Combined Development Validation175",
        "",
        "This is a post-hoc development evaluation of already-frozen final states.",
        "The original Validation50 and FormerTest125 retain separate provenance, then",
        "correct counts are summed and divided by 175. The two accuracies are not",
        "equally averaged. FormerTest125 is no longer an untouched held-out test.",
        "",
        "- Training trajectories: 18/18",
        "- Validation50 evaluations: 18/18",
        "- FormerTest125 evaluations: 18/18",
        "- Training rerun: false",
        "- Checkpoint selection: none; final active state only",
        "- Untouched held-out test remaining: false",
        "",
        "## Aggregate Combined175 results",
        "",
        "| Arm | VoteAcc | OracleAcc | Oracle-Vote gap |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['arm']} | {row['mean_combined175_vote_accuracy']:.4f} | "
            f"{row['mean_combined175_oracle_accuracy']:.4f} | "
            f"{row['mean_combined175_oracle_vote_gap']:.4f} |"
        )
    (report_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--audit", action="store_true")
    modes.add_argument("--analyze", action="store_true")
    parser.add_argument("--prep-root", type=Path, default=PREP_ROOT)
    parser.add_argument("--former-test-root", type=Path, default=FORMER_TEST_ROOT)
    parser.add_argument("--audit-root", type=Path, default=FORMER_TEST_ROOT.parent / "diversity_matrix_d0_d5_former_test125_gate_20260904")
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.prep_root.resolve(), args.former_test_root.resolve(), args.report_root.resolve())
    elif args.run:
        result = run(args.prep_root.resolve(), args.former_test_root.resolve())
    elif args.audit:
        result = audit(args.prep_root.resolve(), args.former_test_root.resolve(), args.audit_root.resolve())
    else:
        result = analyze(args.prep_root.resolve(), args.former_test_root.resolve(), args.audit_root.resolve(), args.report_root.resolve())
    print(json.dumps(result, indent=2))
    if result.get("gate") == "HOLD":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
