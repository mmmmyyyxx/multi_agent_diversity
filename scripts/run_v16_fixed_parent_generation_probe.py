from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.prompt_question import PromptAnswer
from multi_dataset_diverse_rl.responsibility import RepairLane
from multi_dataset_diverse_rl.system import CandidateFunnel, PromptEnsembleOptimizationSystem


SETTING = {
    "c0_current_v15": "experimental_v16_c0_current_v15",
    "c2_boundary_plus_preservation": "experimental_v16_c2_boundary_plus_preservation",
    "c3_coalition_aware_preservation": "experimental_v16_c3_coalition_aware_preservation",
}


def profile(block: list[dict[str, Any]], agent: int) -> tuple[PromptAnswer, ...]:
    return tuple(PromptAnswer(
        answer=str(row["team_answers"][agent]), trace="frozen_seed51_profile",
        valid=bool(row["team_validity"][agent]), validity_status="valid" if row["team_validity"][agent] else "frozen_invalid",
        terminal_invalid=not bool(row["team_validity"][agent]),
    ) for row in block)


def state_hash(system: PromptEnsembleOptimizationSystem) -> str:
    value = {
        "prompts": [agent.current_prompt for agent in system.agents],
        "profiles": [[asdict(answer) for answer in row] for row in system.active_profiles],
        "team_state_version": system.team_state_version,
        "accepted_state_count": system.accepted_state_count,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


async def run_cell(registry: dict[str, Any], case: dict[str, Any], variant: str, out_dir: Path, cache: Path) -> dict[str, Any]:
    flat = dict(registry["base_config"])
    flat.update({
        "experiment_setting": SETTING[variant], "module2_context_variant": variant,
        "initialization_mode": "provided_prompt_set",
        "provided_prompts_json": json.dumps(registry["parent_prompts"]),
        "out_dir": str(out_dir), "shared_solver_cache_path": str(cache),
        "resume_from_checkpoint": False, "final_test_enabled": False,
        "proposal_memory_mode": "off", "seed": 51,
    })
    cfg = Config.from_flat(**{key: flat[key] for key in Config().to_flat_dict()})
    system = PromptEnsembleOptimizationSystem(cfg)
    data = [{"question": row["question"], "answer": row["answer"]} for row in registry["questions"]]
    system.fixed_probe = system.build_probe(data)
    system.initial_profiles = [profile(registry["initial_profiles"], agent) for agent in range(5)]
    system.active_profiles = [profile(registry["active_profiles"], agent) for agent in range(5)]
    system.accepted_state_count = int(registry["accepted_state_count"])
    system.stable_correct_question_hashes_by_agent = {
        int(agent): set(hashes) for agent, hashes in registry["stable_correct_question_hashes_by_agent"].items()
    }
    system.team_state_version = int(case["team_state_version"])
    target = int(case["target_agent_id"])
    system.cached_active_lane_by_agent[target] = RepairLane(str(case["active_lane"]))
    assigned = set(map(str, case["assigned_question_hashes"]))
    before = state_hash(system)
    funnel = CandidateFunnel()
    candidates = await system.propose_candidates(target, assigned, funnel, int(case["source_update_index"]))
    winner = None
    incumbent = None
    evaluated = []
    if candidates:
        winner, incumbent, evaluated = await system.evaluate_candidates(
            target, candidates, assigned, funnel, int(case["source_update_index"])
        )
    after = state_hash(system)
    if before != after:
        raise RuntimeError("fixed-parent probe mutated the parent team")
    result = {
        "result_version": "v16_fixed_parent_generation_probe_cell_v1",
        "case_id": case["case_id"], "variant": variant, "target_agent_id": target,
        "parent_team_hash": case["parent_team_hash"], "active_lane": case["active_lane"],
        "assigned_question_hashes": sorted(assigned), "generated_candidate_count": len(candidates),
        "evaluated_candidate_count": len(evaluated), "funnel": asdict(funnel),
        "incumbent": asdict(incumbent) if incumbent is not None else None,
        "candidates": [{
            "prompt_hash": row.prompt_hash,
            "prompt": row.prompt,
            "evaluation": asdict(row.stage_b_evaluation or row.stage_a_evaluation),
            "constraint": asdict(row.constraint) if row.constraint else None,
        } for row in evaluated],
        "winner_prompt_hash_diagnostic_only": winner.prompt_hash if winner else "",
        "commit_performed": False, "parent_state_hash_before": before, "parent_state_hash_after": after,
        "validation_calls": system.validation_evaluation_count, "test_calls": system.test_evaluation_count,
        "llm_call_count": len(system.llm.calls),
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "cell_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "llm_calls.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in system.llm.calls), encoding="utf-8")
    return result


async def main_async(args: argparse.Namespace) -> None:
    if os.environ.get("V16_FIXED_PARENT_PROBE_AUTHORIZED") != "1":
        raise SystemExit("API execution blocked: set V16_FIXED_PARENT_PROBE_AUTHORIZED=1 only after explicit authorization")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    root = args.out_root.resolve()
    if ROOT.resolve() not in root.parents or root.exists():
        raise SystemExit("out_root must be a fresh project-local directory")
    cache = root / "_shared_solver_cache.sqlite"
    root.mkdir(parents=True)
    results = []
    for case in registry["cases"]:
        for variant in case["cell_order"]:
            results.append(await run_cell(registry, case, variant, root / case["case_id"] / variant, cache))
    (root / "probe_summary.json").write_text(json.dumps({
        "probe_version": "v16_fixed_parent_generation_probe_v1", "registry_hash": registry["registry_content_hash"],
        "cell_count": len(results), "commit_count": 0, "validation_calls": sum(x["validation_calls"] for x in results),
        "test_calls": sum(x["test_calls"] for x in results), "cells": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
