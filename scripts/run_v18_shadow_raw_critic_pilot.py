from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.candidate_selection import common_monotone_safe_key
from multi_dataset_diverse_rl.system import CandidateFunnel
from scripts.generic_m20_probe_support import state_hash, system_for
from scripts.gepa_candidate_breadth_support import candidate_row
from scripts.run_gepa_candidate_breadth_pilot import validation_delta
from scripts.run_v18_safety_only_critic_pilot import sanitized_critic_decisions
from scripts.v18_safety_only_critic_pilot_support import canonical_hash, read_json, sha256_file, write_json
from scripts.v18_shadow_raw_critic_support import ARMS, AUTH_ENV, SharedCanonicalReplay, install_mode


SETTING = "experimental_v16_efficacy_g_matched"


def verify_freeze(registry_path: Path, freeze_path: Path) -> dict[str, Any]:
    registry, freeze = read_json(registry_path), read_json(freeze_path)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != registry["execution_commit"] or head != freeze["execution_commit"]:
        raise RuntimeError("execution commit mismatch")
    if sha256_file(registry_path) != freeze["registry_sha256"]:
        raise RuntimeError("registry hash mismatch")
    if canonical_hash({key: value for key, value in registry.items() if key != "registry_content_hash"}) != registry["registry_content_hash"]:
        raise RuntimeError("registry content mismatch")
    for item in freeze["files"]:
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"source freeze mismatch: {item['path']}")
    return registry


async def run_arm(case: dict[str, Any], arm: str, out: Path, cache: Path, replay: SharedCanonicalReplay) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    system = system_for(case, setting=SETTING, out_dir=out / "runtime", cache_path=cache, evolution_variant="m20_current_v15")
    if system.team_prompt_state_hash() != case["parent_team_hash"]:
        raise RuntimeError("parent reconstruction mismatch")
    before = state_hash(system)
    event_start = len(replay.shadow_events)
    replay_start = replay.replay_hits
    install_mode(system, arm, replay)
    target = int(case["target_agent_id"])
    assigned = set(case["assigned_question_hashes"])
    funnel = CandidateFunnel()
    source = await system.propose_candidates(target, assigned, funnel, int(case["source_update_index"]))
    incumbent = None
    evaluated = []
    revisions = []
    if source:
        _, incumbent, evaluated = await system.evaluate_candidates(target, source, assigned, funnel, int(case["source_update_index"]))
        revisions = await system._loss_blind_generic_revision_candidates(
            target=target,
            assigned_hashes=assigned,
            source_candidates=evaluated,
            incumbent=incumbent,
            update_index=int(case["source_update_index"]),
        )
    source_slot = {row.prompt_hash: index + 1 for index, row in enumerate(evaluated)}
    rows = [candidate_row(row, stage="source", source_slot=index + 1) for index, row in enumerate(evaluated)]
    rows.extend(candidate_row(row, stage="revision", source_slot=source_slot.get(str(row.module2_diagnostics.get("source_candidate_hash", "")))) for row in revisions)
    feasible = [row for row in [*evaluated, *revisions] if row.constraint is not None and row.constraint.passed]
    winner = max(feasible, key=lambda row: common_monotone_safe_key(row.final_evaluation, row.generation), default=None)
    shadow_events = replay.shadow_events[event_start:]
    train = {
        "case_id": case["case_id"],
        "arm": arm,
        "parent_hash": case["parent_team_hash"],
        "target_member": target,
        "teacher_plan_hashes": sorted({str(row.get("teacher_plan_hash", "")) for row in system.tcs_rounds if row.get("role") == "teacher" and row.get("teacher_plan_hash")}),
        "teacher_plan_sequence": [str(row["teacher_plan_hash"]) for row in system.tcs_rounds if row.get("role") == "teacher" and row.get("teacher_plan_hash")],
        "critic_decisions": sanitized_critic_decisions(system, "canonical_llm"),
        "critic_api_calls": sum(row.get("role") == "critic" and row.get("success") for row in system.llm.calls),
        "teacher_api_calls": sum(row.get("role") == "teacher" and row.get("success") for row in system.llm.calls),
        "student_api_calls": sum(row.get("role") == "student" and row.get("success") for row in system.llm.calls),
        "shadow_intervention_count": len(shadow_events),
        "shadow_events": shadow_events,
        "role_replay_hits": replay.replay_hits - replay_start,
        "critic_decision_count": int(funnel.critic_calls),
        "critic_approvals": int(funnel.critic_approved),
        "student_reached": bool(funnel.student_calls),
        "student_calls": int(funnel.student_calls),
        "strict_valid_candidates": len(rows),
        "common_safe_feasible_candidates": len(feasible),
        "would_commit": winner is not None,
        "winner_hash": winner.prompt_hash if winner else "",
        "candidate_rows": rows,
        "funnel": vars(funnel),
        "team_prompt_commit_count": 0,
        "trajectory_mutation_count": 0,
        "test_calls": int(system.test_evaluation_count),
        "state_hash_before": before,
    }
    write_json(out / "train_decision.json", train)
    return system, train, {row.prompt_hash: row for row in [*evaluated, *revisions]}


async def run_case(case: dict[str, Any], out: Path, cache: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    replay = SharedCanonicalReplay()
    arm_data = {}
    for arm in ARMS:
        arm_out = out / arm
        arm_out.mkdir()
        system, train, runtime = await run_arm(case, arm, arm_out, cache, replay)
        arm_data[arm] = {"system": system, "train": train, "runtime": runtime, "out": arm_out}
        if arm == "canonical_control":
            replay.valid_rejected_plan_hashes = {
                str(row["teacher_plan_hash"])
                for row in system.tcs_rounds
                if row.get("role") == "critic"
                and row.get("schema_valid")
                and row.get("failure_class") == "semantic_rejection"
                and row.get("failed_checks")
                and row.get("teacher_plan_hash")
            }
    canonical_hashes = arm_data["canonical_control"]["train"]["teacher_plan_sequence"]
    shadow_hashes = arm_data["shadow_raw"]["train"]["teacher_plan_sequence"]
    if not canonical_hashes or not shadow_hashes or canonical_hashes[0] != shadow_hashes[0]:
        raise RuntimeError("first Teacher plan was not shared")
    write_json(out / "train_freeze.json", {arm: data["train"] for arm, data in arm_data.items()})
    for arm, data in arm_data.items():
        train = data["train"]
        winner_hash = train["winner_hash"]
        validation = (
            await validation_delta(data["system"], case, int(case["target_agent_id"]), data["runtime"][winner_hash])
            if winner_hash
            else {"validation_target_delta": 0, "validation_vote_delta": 0, "validation_oracle_delta": 0}
        )
        train.update(validation)
        train["state_hash_after"] = state_hash(data["system"])
        if train["state_hash_after"] != train["state_hash_before"]:
            raise RuntimeError("fixed parent state mutated")
        write_json(data["out"] / "branch_result.json", train)
        write_json(data["out"] / "private_candidate_prompts.json", {key: row.prompt for key, row in data["runtime"].items()})
    result = {
        "case_id": case["case_id"],
        "historical_status": case["historical_status"],
        "parent_hash": case["parent_team_hash"],
        "target_member": case["target_agent_id"],
        "first_teacher_plan_shared": True,
        "arms": {arm: data["train"] for arm, data in arm_data.items()},
        "team_prompt_commit_count": 0,
        "trajectory_mutation_count": 0,
        "test_calls": 0,
    }
    write_json(out / "case_result.json", result)
    return result


async def main_async(args: argparse.Namespace) -> None:
    if os.environ.get(AUTH_ENV) != "1":
        raise SystemExit(f"set {AUTH_ENV}=1")
    if args.out.exists():
        raise SystemExit("fresh output root required")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("tracked worktree must be clean")
    registry = verify_freeze(args.registry, args.freeze)
    args.out.mkdir(parents=True)
    cache = args.out / "_shared_solver_cache.sqlite"
    results = []
    for case in registry["cases"]:
        result = await run_case(case, args.out / case["case_id"], cache)
        results.append(result)
        print(json.dumps({"completed": len(results), "total": len(registry["cases"]), "case": case["case_id"]}), flush=True)
        if result["test_calls"]:
            raise RuntimeError("test isolation failure")
    write_json(args.out / "pilot_summary.json", {
        "case_count": len(results),
        "branch_count": len(results) * 2,
        "shadow_intervention_count": sum(row["arms"]["shadow_raw"]["shadow_intervention_count"] for row in results),
        "shadow_critic_api_calls": sum(row["arms"]["shadow_raw"]["critic_api_calls"] for row in results),
        "team_prompt_commit_count": 0,
        "trajectory_mutation_count": 0,
        "test_calls": 0,
        "results_hash": canonical_hash(results),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if ROOT.resolve() not in args.out.resolve().parents:
        raise SystemExit("project-local output required")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
