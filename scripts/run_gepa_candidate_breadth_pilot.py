from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from generic_m20_probe_support import profile, state_hash, system_for
from multi_dataset_diverse_rl.system import CandidateFunnel
from gepa_candidate_breadth_support import (
    AUTHORIZATION_ENV, PROBE_VERSION, REQUESTED_SOURCE_COUNT, candidate_row,
    canonical_hash, choose_pool, finalize_pool_comparison, read_json,
    sha256_file, widen_candidate_budget, write_json,
)


SETTING = "experimental_v16_efficacy_g_matched"


def verify_freeze(registry: dict[str, Any], freeze: dict[str, Any]) -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != registry["execution_commit"] or head != freeze["execution_commit"]:
        raise RuntimeError("execution commit mismatch")
    payload = {key: value for key, value in registry.items() if key != "registry_content_hash"}
    if canonical_hash(payload) != registry["registry_content_hash"]:
        raise RuntimeError("registry content hash mismatch")
    for row in freeze["files"]:
        if sha256_file(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"source freeze mismatch: {row['path']}")


def snapshot(system: Any, examples: Sequence[Any], profiles: Sequence[Any]) -> dict[str, Any]:
    metrics = system._dataset_metrics_from_profiles(examples, profiles)
    oracle = sum(
        any(
            member[index].valid
            and system.match_answer(member[index].answer, example.gold_answer)
            for member in profiles
        )
        for index, example in enumerate(examples)
    )
    return {
        "vote": int(metrics.vote_correct_count),
        "oracle": int(oracle),
        "member_counts": list(map(int, metrics.per_agent_correct_counts)),
    }


async def validation_delta(system: Any, case: dict[str, Any], target: int, runtime: Any) -> dict[str, int]:
    data = [
        {"question": row["question"], "answer": row["answer"]}
        for row in case["validation_questions"]
    ]
    probe = system.build_probe(data)
    parent_profiles = [profile(case["validation_parent_profiles"], agent) for agent in range(5)]
    parent = snapshot(system, probe.examples, parent_profiles)
    candidate_profile = await probe.evaluate_prompt(
        target, runtime.prompt, runtime.prompt_hash, system.solve
    )
    child_profiles = list(parent_profiles)
    child_profiles[target] = candidate_profile
    child = snapshot(system, probe.examples, child_profiles)
    return {
        "validation_target_delta": child["member_counts"][target] - parent["member_counts"][target],
        "validation_vote_delta": child["vote"] - parent["vote"],
        "validation_oracle_delta": child["oracle"] - parent["oracle"],
    }


async def run_case(case: dict[str, Any], out: Path, cache: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    system = system_for(
        case, setting=SETTING, out_dir=out / "runtime", cache_path=cache,
        evolution_variant="m20_current_v15",
    )
    if system.team_prompt_state_hash() != case["parent_team_hash"]:
        raise RuntimeError("frozen parent reconstruction mismatch")
    if system.protocol.compatibility_repair_enabled or not system.protocol.generic_revision_enabled:
        raise RuntimeError("proposal mechanism mismatch")
    before = state_hash(system)
    widen_candidate_budget(system)
    target = int(case["target_agent_id"])
    assigned = set(map(str, case["assigned_question_hashes"]))
    funnel = CandidateFunnel()
    source = await system.propose_candidates(
        target, assigned, funnel, int(case["source_update_index"])
    )
    if len(source) > REQUESTED_SOURCE_COUNT:
        raise RuntimeError("source mutation budget exceeded")
    _, incumbent, evaluated = await system.evaluate_candidates(
        target, source, assigned, funnel, int(case["source_update_index"])
    )
    slot_by_hash = {row.prompt_hash: index + 1 for index, row in enumerate(source)}
    revisions = await system._loss_blind_generic_revision_candidates(
        target=target, assigned_hashes=assigned, source_candidates=evaluated,
        incumbent=incumbent, update_index=int(case["source_update_index"]),
    )
    rows = [
        candidate_row(row, stage="source", source_slot=slot_by_hash[row.prompt_hash])
        for row in evaluated
    ]
    rows.extend(
        candidate_row(
            row, stage="revision",
            source_slot=slot_by_hash.get(str(row.module2_diagnostics["source_candidate_hash"])),
        )
        for row in revisions
    )
    n2, n4 = choose_pool(rows, 2), choose_pool(rows, 4)
    finalize_pool_comparison(n2, n4, rows)
    train_decision = {
        "case_id": case["case_id"],
        "parent_team_hash": case["parent_team_hash"],
        "target_agent_id": target,
        "assigned_residual_set_hash": canonical_hash(sorted(assigned)),
        "requested_source_candidate_count": REQUESTED_SOURCE_COUNT,
        "actual_source_candidate_count": len(source),
        "revision_attempt_count": len(evaluated),
        "valid_revision_count": len(revisions),
        "n2": n2,
        "n4": n4,
        "candidate_rows": rows,
        "decision_frozen_before_validation": True,
        "team_prompt_commit_count": 0,
        "trajectory_mutation_count": 0,
        "test_calls": 0,
    }
    write_json(out / "train_decision.json", train_decision)
    runtime_by_hash = {row.prompt_hash: row for row in [*evaluated, *revisions]}
    unique_winners = {n2["winner_hash"], n4["winner_hash"]} - {""}
    validation = {}
    for candidate_hash in sorted(unique_winners):
        validation[candidate_hash] = await validation_delta(
            system, case, target, runtime_by_hash[candidate_hash]
        )
    for pool in (n2, n4):
        pool.update(validation.get(pool["winner_hash"], {
            "validation_target_delta": 0,
            "validation_vote_delta": 0,
            "validation_oracle_delta": 0,
        }))
    if state_hash(system) != before:
        raise RuntimeError("fixed parent state mutated")
    result = {
        **train_decision,
        "n2": n2,
        "n4": n4,
        "validation_winner_evaluation_count": len(unique_winners),
        "decision_frozen_before_validation": True,
        "state_hash_before": before,
        "state_hash_after": state_hash(system),
        "role_call_count": len(system.llm.calls),
        "solver_cache_miss_count": int(system.prompt_question_evaluator.cache_misses),
        "infrastructure_failure_count": int(funnel.infrastructure_failed_updates),
        "test_calls": int(system.test_evaluation_count),
    }
    write_json(out / "case_result.json", result)
    write_json(out / "private_candidate_prompts.json", {
        row.prompt_hash: row.prompt for row in [*evaluated, *revisions]
    })
    return result


async def main_async(args: argparse.Namespace) -> None:
    if os.environ.get(AUTHORIZATION_ENV) != "1":
        raise SystemExit(f"set {AUTHORIZATION_ENV}=1 only for authorized API execution")
    if args.out.exists():
        raise SystemExit("fresh output root required")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("tracked worktree must be clean")
    registry, freeze = read_json(args.registry), read_json(args.freeze)
    verify_freeze(registry, freeze)
    args.out.mkdir(parents=True)
    results = []
    cache = args.out / "_shared_solver_cache.sqlite"
    for case in registry["cases"]:
        result = await run_case(case, args.out / case["case_id"], cache)
        results.append(result)
        print(json.dumps({
            "completed_cases": len(results), "total_cases": len(registry["cases"]),
            "last_case": case["case_id"],
        }), flush=True)
        if result["infrastructure_failure_count"] or result["test_calls"]:
            raise RuntimeError("hard infrastructure or test-isolation failure")
    write_json(args.out / "probe_summary.json", {
        "probe_version": PROBE_VERSION,
        "execution_commit": registry["execution_commit"],
        "case_count": len(results),
        "requested_source_candidate_count": REQUESTED_SOURCE_COUNT,
        "team_prompt_commit_count": 0,
        "trajectory_mutation_count": 0,
        "test_calls": sum(row["test_calls"] for row in results),
        "infrastructure_failure_count": sum(row["infrastructure_failure_count"] for row in results),
        "results_hash": canonical_hash(results),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if ROOT.resolve() not in args.out.resolve().parents:
        raise SystemExit("output must be project-local")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
