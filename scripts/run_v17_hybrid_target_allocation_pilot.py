from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from multi_dataset_diverse_rl.candidate_selection import common_monotone_safe_key
from multi_dataset_diverse_rl.system import CandidateFunnel
from v17_hybrid_target_allocation_support import (
    ARMS,
    AUTHORIZATION_ENV,
    HYBRID,
    arm_specs,
    branch_key,
    branch_object,
    canonical_hash,
    choose_would_commit,
    context_hashes,
    immutable_state_hash,
    probe_system,
    realized_delta,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{"question": row["question"], "answer": row["answer"]} for row in csv.DictReader(handle)]


def profile_bundle(system: Any, examples: Any, profiles: list[Any]) -> dict[str, Any]:
    metrics = system._dataset_metrics_from_profiles(examples, profiles)
    oracle_vector = []
    for index, example in enumerate(examples):
        oracle_vector.append(any(
            profile[index].valid and system.match_answer(profile[index].answer, example.gold_answer)
            for profile in profiles
        ))
    return {
        "vote": int(metrics.vote_correct_count),
        "oracle": sum(oracle_vector),
        "member_counts": list(map(int, metrics.per_agent_correct_counts)),
        "vote_vector": [bool(row.vote_correct) for row in metrics.rows],
        "oracle_vector": list(map(bool, oracle_vector)),
        "largest_wrong": [int(row.largest_wrong_vote_count) for row in metrics.rows],
    }


def delta_metrics(parent: dict[str, Any], candidate: dict[str, Any], target: int) -> dict[str, int]:
    vote_gain = sum(not left and right for left, right in zip(parent["vote_vector"], candidate["vote_vector"], strict=True))
    vote_loss = sum(left and not right for left, right in zip(parent["vote_vector"], candidate["vote_vector"], strict=True))
    oracle_gain = sum(not left and right for left, right in zip(parent["oracle_vector"], candidate["oracle_vector"], strict=True))
    oracle_loss = sum(left and not right for left, right in zip(parent["oracle_vector"], candidate["oracle_vector"], strict=True))
    wrong_reduced = sum(left > right for left, right in zip(parent["largest_wrong"], candidate["largest_wrong"], strict=True))
    wrong_increased = sum(left < right for left, right in zip(parent["largest_wrong"], candidate["largest_wrong"], strict=True))
    return {
        "target_delta": candidate["member_counts"][target] - parent["member_counts"][target],
        "vote_delta": candidate["vote"] - parent["vote"],
        "oracle_delta": candidate["oracle"] - parent["oracle"],
        "vote_gain_count": vote_gain,
        "vote_loss_count": vote_loss,
        "oracle_gain_count": oracle_gain,
        "oracle_loss_count": oracle_loss,
        "wrong_coalition_reduced_count": wrong_reduced,
        "wrong_coalition_increased_count": wrong_increased,
    }


async def run_unique_branch(case: dict[str, Any], target: int, out_dir: Path, cache_path: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=False)
    system = probe_system(case, target=target, out_dir=out_dir, cache_path=cache_path)
    before = immutable_state_hash(system)
    hashes = context_hashes(case, target)
    funnel = CandidateFunnel()
    source = await system.propose_candidates(target, hashes, funnel, int(case["update_index"]))
    if len(source) > 2:
        raise RuntimeError("source candidate budget exceeded")
    revision_before = len(system.generic_revision_events)
    winner = incumbent = None
    evaluated = []
    if source:
        winner, incumbent, evaluated = await system.evaluate_candidates(
            target, source, hashes, funnel, int(case["update_index"])
        )
        revised = await system._loss_blind_generic_revision_candidates(
            target=target,
            assigned_hashes=hashes,
            source_candidates=evaluated,
            incumbent=incumbent,
            update_index=int(case["update_index"]),
        )
        evaluated.extend(revised)
        feasible = [row for row in evaluated if row.constraint is not None and row.constraint.passed]
        winner = max(
            feasible,
            key=lambda row: common_monotone_safe_key(row.final_evaluation, row.generation),
            default=None,
        )
        funnel.stage_b_evaluated += len(revised)
        funnel.constraint_feasible += sum(bool(row.constraint and row.constraint.passed) for row in revised)
        funnel.acceptable_candidates += sum(bool(row.constraint and row.constraint.passed) for row in revised)
        funnel.accepted_candidate = winner is not None
    revision_count = len(system.generic_revision_events) - revision_before
    source_rows = [
        row for row in evaluated
        if str((row.module2_diagnostics or {}).get("candidate_stage", "")) != "loss_blind_generic_revision"
    ]
    if revision_count != len(source_rows):
        raise RuntimeError("loss-blind revision count does not match valid sources")
    if immutable_state_hash(system) != before:
        raise RuntimeError("fixed parent state mutated")
    parent_train = profile_bundle(system, system.fixed_probe.examples, list(system.active_profiles))
    feasible_order = sorted(
        [row for row in evaluated if row.constraint is not None and row.constraint.passed],
        key=lambda row: common_monotone_safe_key(row.final_evaluation, row.generation),
        reverse=True,
    )
    rank_by_hash = {row.prompt_hash: index + 1 for index, row in enumerate(feasible_order)}
    candidates = []
    for row in evaluated:
        profiles = list(system.active_profiles)
        profiles[target] = row.profile
        train = profile_bundle(system, system.fixed_probe.examples, profiles)
        constraint = row.constraint
        stage = (
            "revision"
            if str((row.module2_diagnostics or {}).get("candidate_stage", "")) == "loss_blind_generic_revision"
            else "source"
        )
        candidates.append({
            "candidate_id": row.prompt_hash,
            "candidate_stage": stage,
            "valid": True,
            "feasible": bool(constraint and constraint.passed),
            "common_safe_outcome": "passed" if constraint and constraint.passed else "rejected",
            "sanitized_rejection_reasons": sorted(map(str, constraint.rejection_reasons)) if constraint else ["constraint_unavailable"],
            "train": delta_metrics(parent_train, train, target),
            "branch_rank": rank_by_hash.get(row.prompt_hash),
            "selected_as_branch_winner": bool(winner and winner.prompt_hash == row.prompt_hash),
        })
    payload = {
        "branch_result_version": "v17_hybrid_unique_branch_v1",
        "case_id": case["case_id"],
        "target_member": int(target),
        "canonical_branch_key": branch_key(case, target),
        "parent_team_hash": case["parent_team_hash"],
        "state_hash_before": before,
        "state_hash_after": immutable_state_hash(system),
        "assigned_residual_hash_count": len(hashes),
        "source_candidate_count": len(source),
        "valid_source_count": len(source_rows),
        "valid_revision_count": revision_count,
        "evaluated_candidate_count": len(evaluated),
        "feasible_candidate_count": len(feasible_order),
        "branch_winner_id": winner.prompt_hash if winner else "",
        "funnel": asdict(funnel),
        "role_api_call_count": len(system.llm.calls),
        "validation_calls": system.validation_evaluation_count,
        "test_calls": system.test_evaluation_count,
        "candidates": candidates,
    }
    write_json(out_dir / "branch_result.json", payload)
    return {
        "payload": payload, "winner": winner, "incumbent": incumbent,
        "evaluated": evaluated, "system": system,
    }


async def evaluate_validation(case: dict[str, Any], runtimes: dict[int, dict[str, Any]], comparator: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]], int]:
    path = Path(comparator.cfg.data.val_path)
    if not path.is_absolute():
        path = ROOT / path
    probe = comparator.build_probe(load_rows(path))
    parent_profiles = []
    for agent, prompt in enumerate(case["parent_prompts"]):
        parent_profiles.append(await probe.evaluate_prompt(agent, prompt, comparator.prompt_hash(prompt), comparator.solve))
    parent = profile_bundle(comparator, probe.examples, parent_profiles)
    candidates: dict[str, dict[str, Any]] = {}
    for target in sorted(runtimes):
        for row in runtimes[target]["evaluated"]:
            profile = await probe.evaluate_prompt(target, row.prompt, row.prompt_hash, comparator.solve)
            profiles = list(parent_profiles)
            profiles[target] = profile
            candidate = profile_bundle(comparator, probe.examples, profiles)
            candidates[row.prompt_hash] = delta_metrics(parent, candidate, target)
    return parent, candidates, len(comparator.llm.calls)


async def run_case(case: dict[str, Any], out_dir: Path, cache_path: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=False)
    arms = arm_specs(case)
    targets = sorted({row["target_member"] for specs in arms.values() for row in specs})
    runtimes: dict[int, dict[str, Any]] = {}
    for target in targets:
        runtimes[target] = await run_unique_branch(
            case, target, out_dir / "branches" / branch_key(case, target), cache_path
        )
    comparator = probe_system(
        case, target=targets[0], out_dir=out_dir / "comparison", cache_path=cache_path
    )
    decisions: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        objects = []
        for rank, spec in enumerate(arms[arm]):
            runtime = runtimes[int(spec["target_member"])]
            objects.append(branch_object(
                int(spec["target_member"]), rank, runtime["winner"], runtime["incumbent"]
            ))
        winner_branch = choose_would_commit(comparator, objects)
        winner = winner_branch.accepted if winner_branch else None
        ranked_winners = sorted(
            [row for row in objects if row.accepted is not None],
            key=comparator._cross_branch_key,
            reverse=True,
        )
        decisions[arm] = {
            "objects": objects,
            "winner_branch": winner_branch,
            "winner": winner,
            "cell_rank_by_target": {
                int(row.target_agent_id): index + 1 for index, row in enumerate(ranked_winners)
            },
        }
    parent_validation, validation_by_candidate, validation_api_calls = await evaluate_validation(
        case, runtimes, comparator
    )
    cell_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        decision = decisions[arm]
        winner = decision["winner"]
        winner_branch = decision["winner_branch"]
        would_commit = winner is not None
        winner_validation = validation_by_candidate.get(winner.prompt_hash) if winner else None
        branches = []
        for rank, spec in enumerate(arms[arm]):
            target = int(spec["target_member"])
            runtime = runtimes[target]
            branch_payload = runtime["payload"]
            branches.append({
                "target_member": target,
                "target_selection_rank": rank,
                "branch_type": spec["branch_type"],
                "canonical_branch_key": branch_payload["canonical_branch_key"],
                "shared_branch_reuse_count": sum(
                    row["target_member"] == target for values in arms.values() for row in values
                ),
                "source_candidate_count": branch_payload["source_candidate_count"],
                "valid_source_count": branch_payload["valid_source_count"],
                "valid_revision_count": branch_payload["valid_revision_count"],
                "feasible_candidate_count": branch_payload["feasible_candidate_count"],
                "branch_winner_id": branch_payload["branch_winner_id"],
                "produced_cell_best": bool(
                    winner_branch and int(winner_branch.target_agent_id) == target
                ),
            })
            for candidate, runtime_candidate in zip(
                branch_payload["candidates"], runtime["evaluated"], strict=True
            ):
                row = dict(candidate)
                row.update({
                    "parent_id": case["case_id"],
                    "arm": arm,
                    "target_member": target,
                    "branch_type": spec["branch_type"],
                    "canonical_branch_key": branch_payload["canonical_branch_key"],
                    "validation": validation_by_candidate[runtime_candidate.prompt_hash],
                    "cell_rank": (
                        decision["cell_rank_by_target"].get(target)
                        if candidate["selected_as_branch_winner"] else None
                    ),
                    "selected_as_cell_winner": bool(
                        winner and winner.prompt_hash == runtime_candidate.prompt_hash
                    ),
                    "would_commit_contribution": bool(
                        winner and winner.prompt_hash == runtime_candidate.prompt_hash
                    ),
                })
                candidate_rows.append(row)
        cell = {
            "cell_result_version": "v17_hybrid_target_allocation_cell_v1",
            "case_id": case["case_id"], "arm": arm,
            "parent_team_hash": case["parent_team_hash"],
            "responsibility_eligible_ids": case["responsibility_eligible_ids"],
            "target_ids": [int(row["target_member"]) for row in arms[arm]],
            "branches": branches,
            "decision_frozen_before_validation": True,
            "would_commit": would_commit,
            "cell_winner_source": (
                next(row["candidate_stage"] for row in candidate_rows[::-1] if row["arm"] == arm and row["selected_as_cell_winner"])
                if would_commit else "none"
            ),
            "cell_winner_target": int(winner_branch.target_agent_id) if winner else None,
            "parent_validation": {
                "vote": parent_validation["vote"], "oracle": parent_validation["oracle"],
                "member_counts": parent_validation["member_counts"],
            },
            "realized_validation_vote_delta": realized_delta(
                would_commit, 0, int(winner_validation["vote_delta"]) if winner_validation else 0
            ),
            "realized_validation_oracle_delta": realized_delta(
                would_commit, 0, int(winner_validation["oracle_delta"]) if winner_validation else 0
            ),
            "realized_validation_target_delta": realized_delta(
                would_commit, 0, int(winner_validation["target_delta"]) if winner_validation else 0
            ),
            "validation_vote_gain_count": int(winner_validation["vote_gain_count"]) if winner_validation else 0,
            "validation_vote_loss_count": int(winner_validation["vote_loss_count"]) if winner_validation else 0,
            "validation_oracle_gain_count": int(winner_validation["oracle_gain_count"]) if winner_validation else 0,
            "validation_oracle_loss_count": int(winner_validation["oracle_loss_count"]) if winner_validation else 0,
            "train_realized_vote_delta": int(winner.constraint.vote_net_gain) if winner else 0,
            "train_realized_target_delta": int(winner.constraint.target_gain) if winner else 0,
            "team_prompt_commit_count": 0,
            "trajectory_mutation_count": 0,
            "test_calls": 0,
        }
        write_json(out_dir / "cells" / arm / "cell_result.json", cell)
        cell_rows.append(cell)
    write_jsonl(out_dir / "candidate_level.jsonl", candidate_rows)
    case_summary = {
        "case_id": case["case_id"],
        "cell_count": 3,
        "conceptual_branch_count": 6,
        "deduplicated_branch_count": len(targets),
        "role_api_call_count": sum(row["payload"]["role_api_call_count"] for row in runtimes.values()),
        "validation_api_call_count": validation_api_calls,
        "actual_api_call_count": sum(row["payload"]["role_api_call_count"] for row in runtimes.values()) + validation_api_calls,
        "cell_decisions_frozen_before_validation": True,
        "team_prompt_commit_count": 0,
        "trajectory_mutation_count": 0,
        "test_calls": 0,
    }
    write_json(out_dir / "case_summary.json", case_summary)
    return {"cells": cell_rows, "candidates": candidate_rows, "summary": case_summary}


async def main_async(args: argparse.Namespace) -> None:
    if os.environ.get(AUTHORIZATION_ENV) != "1":
        raise SystemExit(f"set {AUTHORIZATION_ENV}=1 only for authorized Phase B")
    if args.out_root.exists():
        raise SystemExit("output root must be fresh")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    freeze = json.loads(args.source_freeze.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != registry["execution_commit"]:
        raise SystemExit("execution commit does not match frozen registry")
    payload = {key: value for key, value in registry.items() if key != "registry_content_hash"}
    if canonical_hash(payload) != registry["registry_content_hash"]:
        raise SystemExit("registry content hash mismatch")
    if freeze.get("execution_commit") != head or freeze.get("source_freeze_status") != "PASS":
        raise SystemExit("source freeze identity mismatch")
    if hashlib.sha256(args.registry.read_bytes()).hexdigest() != freeze.get("registry_file_sha256"):
        raise SystemExit("frozen registry file hash mismatch")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("tracked worktree must remain clean")
    for row in freeze.get("files", []):
        path = ROOT / row["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise SystemExit(f"source freeze file mismatch: {row['path']}")
    args.out_root.mkdir(parents=True)
    results = []
    cache_path = args.out_root / "_shared_solver_cache.sqlite"
    for case in registry["cases"]:
        result = await run_case(case, args.out_root / case["case_id"], cache_path)
        results.append(result)
        print(json.dumps({
            "case_complete": case["case_id"],
            "completed_cases": len(results), "total_cases": 6,
            "actual_api_calls_so_far": sum(row["summary"]["actual_api_call_count"] for row in results),
        }), flush=True)
    cells = [cell for result in results for cell in result["cells"]]
    summary = {
        "probe_version": "v17_hybrid_target_allocation_prospective_pilot_v1",
        "execution_commit": head,
        "registry_content_hash": registry["registry_content_hash"],
        "case_count": 6, "cell_count": len(cells),
        "conceptual_branch_count": 36,
        "deduplicated_branch_count": sum(row["summary"]["deduplicated_branch_count"] for row in results),
        "actual_api_call_count": sum(row["summary"]["actual_api_call_count"] for row in results),
        "would_commit_count": sum(row["would_commit"] for row in cells),
        "team_prompt_commit_count": 0, "trajectory_mutation_count": 0,
        "validation_selection_count": 0, "test_calls": 0,
        "results_hash": canonical_hash(cells),
    }
    write_json(args.out_root / "probe_summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source_freeze", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    args = parser.parse_args()
    args.out_root = args.out_root.resolve()
    if ROOT.resolve() not in args.out_root.parents:
        raise SystemExit("output must be project-local")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
