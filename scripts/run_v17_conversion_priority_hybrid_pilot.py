from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_v17_hybrid_target_allocation_pilot import (
    delta_metrics,
    load_rows,
    run_unique_branch,
    write_json,
    write_jsonl,
)
from v17_conversion_priority_hybrid_support import (
    ARMS,
    AUTHORIZATION_ENV,
    arm_specs,
    branch_key,
    branch_object,
    canonical_hash,
    choose_would_commit,
    probe_system,
)


def structured_bundle(system: Any, examples: Any, profiles: list[Any]) -> dict[str, Any]:
    metrics = system._dataset_metrics_from_profiles(examples, profiles)
    oracle_vector = [
        any(
            profile[index].valid
            and system.match_answer(profile[index].answer, example.gold_answer)
            for profile in profiles
        )
        for index, example in enumerate(examples)
    ]
    return {
        "vote": int(metrics.vote_correct_count),
        "oracle": sum(oracle_vector),
        "member_counts": list(map(int, metrics.per_agent_correct_counts)),
        "vote_vector": [bool(row.vote_correct) for row in metrics.rows],
        "oracle_vector": list(map(bool, oracle_vector)),
        "largest_wrong": [int(row.largest_wrong_vote_count) for row in metrics.rows],
        "gold": [int(row.gold_vote_count) for row in metrics.rows],
        "margin": [int(row.plurality_margin) for row in metrics.rows],
    }


def structure_delta(parent: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    histogram = Counter()
    deeper = vote_conversion = vote_regression = 0
    h_decrease = h_increase = margin_improvement = margin_regression = 0
    for gp, hp, mp, gc, hc, mc in zip(
        parent["gold"], parent["largest_wrong"], parent["margin"],
        candidate["gold"], candidate["largest_wrong"], candidate["margin"],
        strict=True,
    ):
        if gc > gp:
            if gp == 0 and gc == 1:
                histogram["0_to_1"] += 1
            elif gp == 1 and gc == 2:
                histogram["1_to_2"] += 1
            elif gp == 1 and gc >= 3:
                histogram["1_to_3_plus"] += 1
            elif gp == 2 and gc == 3:
                histogram["2_to_3"] += 1
            elif gp == 2 and gc >= 4:
                histogram["2_to_4_plus"] += 1
            else:
                histogram["other_positive_g"] += 1
        if 0 < gp <= hp and gc > gp:
            deeper += 1
        if 0 < gp <= hp and gc > hc:
            vote_conversion += 1
        if gp > hp and gc <= hc:
            vote_regression += 1
        h_decrease += hc < hp
        h_increase += hc > hp
        margin_improvement += mc > mp
        margin_regression += mc < mp
    for key in (
        "0_to_1", "1_to_2", "1_to_3_plus", "2_to_3", "2_to_4_plus",
        "other_positive_g",
    ):
        histogram.setdefault(key, 0)
    return {
        "deeper_support_gain_count": deeper,
        "vote_conversion_count": vote_conversion,
        "vote_regression_count": vote_regression,
        "h_decrease_count": h_decrease,
        "h_increase_count": h_increase,
        "margin_improvement_count": margin_improvement,
        "margin_regression_count": margin_regression,
        "transition_histogram": dict(histogram),
    }


async def evaluate_validation(
    case: dict[str, Any], runtimes: dict[int, dict[str, Any]], comparator: Any
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = Path(comparator.cfg.data.val_path)
    if not path.is_absolute():
        path = ROOT / path
    probe = comparator.build_probe(load_rows(path))
    parent_profiles = [
        await probe.evaluate_prompt(
            agent, prompt, comparator.prompt_hash(prompt), comparator.solve
        )
        for agent, prompt in enumerate(case["parent_prompts"])
    ]
    parent = structured_bundle(comparator, probe.examples, parent_profiles)
    candidates: dict[str, dict[str, Any]] = {}
    for target in sorted(runtimes):
        for row in runtimes[target]["evaluated"]:
            profile = await probe.evaluate_prompt(target, row.prompt, row.prompt_hash, comparator.solve)
            profiles = list(parent_profiles)
            profiles[target] = profile
            candidate = structured_bundle(comparator, probe.examples, profiles)
            candidates[row.prompt_hash] = {
                **delta_metrics(parent, candidate, target),
                "structure": structure_delta(parent, candidate),
            }
    return parent, candidates


def _call_summary(calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_recorded_calls": len(calls),
        "successful_calls": sum(bool(row.get("success")) for row in calls),
        "calls_by_role": dict(sorted(Counter(str(row.get("role", "unknown")) for row in calls).items())),
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in calls),
        "completion_tokens": sum(int(row.get("completion_tokens", 0)) for row in calls),
        "total_tokens": sum(int(row.get("total_tokens", 0)) for row in calls),
    }


async def run_case(case: dict[str, Any], out_dir: Path, cache_path: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=False)
    arms = arm_specs(case)
    targets = sorted({int(row["target_member"]) for rows in arms.values() for row in rows})
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
        objects = [
            branch_object(
                int(spec["target_member"]), rank,
                runtimes[int(spec["target_member"])]["winner"],
                runtimes[int(spec["target_member"])]["incumbent"],
            )
            for rank, spec in enumerate(arms[arm])
        ]
        winner_branch = choose_would_commit(comparator, objects)
        ranked = sorted(
            [row for row in objects if row.accepted is not None],
            key=comparator._cross_branch_key,
            reverse=True,
        )
        decisions[arm] = {
            "winner_branch": winner_branch,
            "winner": winner_branch.accepted if winner_branch else None,
            "cell_rank_by_target": {
                int(row.target_agent_id): index + 1 for index, row in enumerate(ranked)
            },
        }
    parent_validation, validation = await evaluate_validation(case, runtimes, comparator)
    cells: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for arm in ARMS:
        decision = decisions[arm]
        winner = decision["winner"]
        winner_branch = decision["winner_branch"]
        would_commit = winner is not None
        winner_validation = validation.get(winner.prompt_hash) if winner else None
        branches = []
        winner_source = "none"
        for rank, spec in enumerate(arms[arm]):
            target = int(spec["target_member"])
            runtime = runtimes[target]
            payload = runtime["payload"]
            branches.append({
                "target_member": target,
                "target_selection_rank": rank,
                "branch_type": spec["branch_type"],
                "canonical_branch_key": payload["canonical_branch_key"],
                "shared_branch_reuse_count": sum(
                    int(row["target_member"]) == target for rows in arms.values() for row in rows
                ),
                "source_candidate_count": payload["source_candidate_count"],
                "valid_source_count": payload["valid_source_count"],
                "valid_revision_count": payload["valid_revision_count"],
                "feasible_candidate_count": payload["feasible_candidate_count"],
                "produced_cell_best": bool(winner_branch and int(winner_branch.target_agent_id) == target),
            })
            for meta, evaluated in zip(payload["candidates"], runtime["evaluated"], strict=True):
                selected = bool(winner and winner.prompt_hash == evaluated.prompt_hash)
                if selected:
                    winner_source = meta["candidate_stage"]
                candidates.append({
                    **meta,
                    "parent_id": case["case_id"],
                    "seed": case["source_seed"],
                    "update_index": case["source_update_index"],
                    "arm": arm,
                    "target_member": target,
                    "branch_type": spec["branch_type"],
                    "is_conversion_eligible": bool(
                        case["selector_scores_by_agent"][str(target)]["conversion_responsibility_count"]
                    ),
                    "conversion_responsibility_count": case["selector_scores_by_agent"][str(target)]["conversion_responsibility_count"],
                    "direct_vote_flip_count": case["selector_scores_by_agent"][str(target)]["direct_vote_flip_count"],
                    "validation": validation[evaluated.prompt_hash],
                    "cell_rank": decision["cell_rank_by_target"].get(target) if meta["selected_as_branch_winner"] else None,
                    "selected_as_cell_winner": selected,
                    "would_commit_contribution": selected,
                })
        structure = winner_validation["structure"] if winner_validation else structure_delta(parent_validation, parent_validation)
        train_oracle = next(
            (row["train"]["oracle_delta"] for row in candidates if row["arm"] == arm and row["selected_as_cell_winner"]),
            0,
        )
        cell = {
            "cell_result_version": "v17_conversion_priority_three_arm_cell_v1",
            "case_id": case["case_id"],
            "arm": arm,
            "parent_team_hash": case["parent_team_hash"],
            "target_ids": [int(row["target_member"]) for row in arms[arm]],
            "branches": branches,
            "decision_frozen_before_validation": True,
            "would_commit": would_commit,
            "cell_winner_source": winner_source,
            "cell_winner_target": int(winner_branch.target_agent_id) if winner else None,
            "parent_validation": {
                "vote": parent_validation["vote"],
                "oracle": parent_validation["oracle"],
                "member_counts": parent_validation["member_counts"],
            },
            "realized_validation_vote_delta": int(winner_validation["vote_delta"]) if winner_validation else 0,
            "realized_validation_oracle_delta": int(winner_validation["oracle_delta"]) if winner_validation else 0,
            "realized_validation_target_delta": int(winner_validation["target_delta"]) if winner_validation else 0,
            "train_realized_vote_delta": int(winner.constraint.vote_net_gain) if winner else 0,
            "train_realized_target_delta": int(winner.constraint.target_gain) if winner else 0,
            "train_realized_oracle_delta": int(train_oracle),
            "target_transfer_gap": (
                int(winner_validation["target_delta"]) - int(winner.constraint.target_gain)
                if winner_validation and winner else 0
            ),
            "validation_structure": structure,
            "team_prompt_commit_count": 0,
            "trajectory_mutation_count": 0,
            "test_calls": 0,
        }
        write_json(out_dir / "cells" / arm / "cell_result.json", cell)
        cells.append(cell)
    write_jsonl(out_dir / "candidate_level.jsonl", candidates)
    all_calls = [call for runtime in runtimes.values() for call in runtime["system"].llm.calls]
    all_calls.extend(comparator.llm.calls)
    summary = {
        "case_id": case["case_id"],
        "cell_count": 3,
        "conceptual_branch_count": 6,
        "deduplicated_branch_count": len(targets),
        "call_summary": _call_summary(all_calls),
        "team_prompt_commit_count": 0,
        "trajectory_mutation_count": 0,
        "test_calls": 0,
    }
    write_json(out_dir / "case_summary.json", summary)
    return {"cells": cells, "candidates": candidates, "summary": summary}


async def main_async(args: argparse.Namespace) -> None:
    if os.environ.get(AUTHORIZATION_ENV) != "1":
        raise SystemExit(f"set {AUTHORIZATION_ENV}=1 only for authorized Phase B")
    if args.out_root.exists():
        raise SystemExit("output root must be fresh")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    freeze = json.loads(args.source_freeze.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != registry["execution_commit"] or freeze.get("execution_commit") != head:
        raise SystemExit("execution source identity mismatch")
    payload = {key: value for key, value in registry.items() if key != "registry_content_hash"}
    if canonical_hash(payload) != registry["registry_content_hash"]:
        raise SystemExit("registry content hash mismatch")
    if freeze.get("source_freeze_status") != "PASS":
        raise SystemExit("source freeze did not pass")
    import hashlib
    if hashlib.sha256(args.registry.read_bytes()).hexdigest() != freeze["registry_file_sha256"]:
        raise SystemExit("registry file hash mismatch")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("tracked worktree must remain clean")
    for row in freeze["files"]:
        path = ROOT / row["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise SystemExit(f"source freeze file mismatch: {row['path']}")
    args.out_root.mkdir(parents=True)
    cache = args.out_root / "_shared_solver_cache.sqlite"
    results = []
    for case in registry["cases"]:
        result = await run_case(case, args.out_root / case["case_id"], cache)
        results.append(result)
        print(json.dumps({
            "case_complete": case["case_id"],
            "completed_cases": len(results),
            "total_cases": 5,
            "total_recorded_calls": sum(row["summary"]["call_summary"]["total_recorded_calls"] for row in results),
        }), flush=True)
    cells = [cell for result in results for cell in result["cells"]]
    call_summaries = [result["summary"]["call_summary"] for result in results]
    summary = {
        "probe_version": "v17_conversion_priority_three_arm_pilot_v1",
        "execution_commit": head,
        "registry_content_hash": registry["registry_content_hash"],
        "case_count": 5,
        "cell_count": len(cells),
        "conceptual_branch_count": 30,
        "deduplicated_branch_count": sum(result["summary"]["deduplicated_branch_count"] for result in results),
        "total_recorded_calls": sum(row["total_recorded_calls"] for row in call_summaries),
        "prompt_tokens": sum(row["prompt_tokens"] for row in call_summaries),
        "completion_tokens": sum(row["completion_tokens"] for row in call_summaries),
        "total_tokens": sum(row["total_tokens"] for row in call_summaries),
        "calls_by_role": dict(sorted(sum((Counter(row["calls_by_role"]) for row in call_summaries), Counter()).items())),
        "would_commit_count": sum(bool(row["would_commit"]) for row in cells),
        "team_prompt_commit_count": 0,
        "trajectory_mutation_count": 0,
        "validation_selection_count": 0,
        "test_calls": 0,
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
