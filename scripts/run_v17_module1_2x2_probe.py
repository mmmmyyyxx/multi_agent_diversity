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
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from multi_dataset_diverse_rl.system import CandidateFunnel
from v17_module1_2x2_support import (
    AUTHORIZATION_ENV,
    CELLS,
    canonical_hash,
    choose_would_commit,
    context_hashes,
    immutable_state_hash,
    probe_system,
    realized_delta,
    targets_for,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{"question": row["question"], "answer": row["answer"]} for row in csv.DictReader(handle)]


async def validation_metrics(system: Any, prompts: list[str]) -> dict[str, int]:
    path = Path(system.cfg.data.val_path)
    if not path.is_absolute():
        path = ROOT / path
    probe = system.build_probe(load_rows(path))
    profiles = []
    for agent, prompt in enumerate(prompts):
        prompt_hash = system.prompt_hash(prompt)
        profiles.append(await probe.evaluate_prompt(agent, prompt, prompt_hash, system.solve))
    metrics = system._dataset_metrics_from_profiles(probe.examples, profiles)
    oracle = sum(
        any(
            answer.valid and system.match_answer(answer.answer, example.gold_answer)
            for answer in (profiles[agent][index] for agent in range(5))
        )
        for index, example in enumerate(probe.examples)
    )
    return {"vote": int(metrics.vote_correct_count), "oracle": int(oracle)}


async def run_branch(
    case: dict[str, Any], cell: str, target: int, rank: int,
    out_dir: Path, cache_path: Path,
) -> tuple[dict[str, Any], Any | None, Any]:
    system = probe_system(
        case, cell, out_dir=out_dir, cache_path=cache_path, target=target
    )
    before = immutable_state_hash(system)
    hashes = context_hashes(case, cell, target)
    funnel = CandidateFunnel()
    source = await system.propose_candidates(
        target, hashes, funnel, int(case["update_index"])
    )
    if len(source) > 2:
        raise RuntimeError("source candidate budget exceeded")
    revision_before = len(system.generic_revision_events)
    winner = incumbent = None
    evaluated = []
    if source:
        winner, incumbent, evaluated = await system.evaluate_candidates(
            target, source, hashes, funnel, int(case["update_index"])
        )
    revision_count = len(system.generic_revision_events) - revision_before
    valid_source_count = sum(
        str((row.module2_diagnostics or {}).get("candidate_stage", ""))
        != "loss_blind_generic_revision"
        for row in evaluated
    )
    if revision_count != valid_source_count:
        raise RuntimeError("loss-blind revision count does not match valid sources")
    if immutable_state_hash(system) != before:
        raise RuntimeError("fixed parent state mutated")
    branch = SimpleNamespace(
        accepted=winner, incumbent=incumbent, target_selection_rank=rank,
        target_agent_id=target,
    )
    payload = {
        "target_agent_id": target,
        "target_selection_rank": rank,
        "parent_team_hash": case["parent_team_hash"],
        "assigned_residual_hash_count": len(hashes),
        "source_candidate_count": len(source),
        "valid_source_candidate_count": valid_source_count,
        "loss_blind_revision_count": revision_count,
        "evaluated_candidate_count": len(evaluated),
        "feasible_candidate_count": sum(
            bool(row.constraint and row.constraint.passed) for row in evaluated
        ),
        "branch_winner_hash": winner.prompt_hash if winner else "",
        "funnel": asdict(funnel),
        "api_calls": len(system.llm.calls),
        "validation_calls": system.validation_evaluation_count,
        "test_calls": system.test_evaluation_count,
    }
    return payload, winner, branch


async def run_cell(
    case: dict[str, Any], cell: str, out_dir: Path, cache_path: Path
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"cell output must be fresh: {out_dir}")
    out_dir.mkdir(parents=True)
    branch_rows, winners, branch_objects = [], [], []
    for rank, target in enumerate(targets_for(case, cell)):
        row, winner, branch = await run_branch(
            case, cell, target, rank, out_dir / f"target{target}", cache_path
        )
        branch_rows.append(row)
        winners.append(winner)
        branch_objects.append(branch)
    comparator = probe_system(
        case, cell, out_dir=out_dir / "comparison", cache_path=cache_path
    )
    winning_branch = choose_would_commit(comparator, branch_objects)
    winner = winning_branch.accepted if winning_branch else None
    would_commit = winner is not None
    parent_prompts = list(case["parent_prompts"])
    hypothetical = list(parent_prompts)
    if winner is not None:
        hypothetical[int(winning_branch.target_agent_id)] = winner.prompt
    parent_metrics = await validation_metrics(comparator, parent_prompts)
    hypothetical_metrics = (
        await validation_metrics(comparator, hypothetical)
        if would_commit else dict(parent_metrics)
    )
    payload = {
        "result_version": "v17_module1_2x2_parent_cell_v1",
        "case_id": case["case_id"], "cell": cell,
        "parent_team_hash": case["parent_team_hash"],
        "target_ids": targets_for(case, cell),
        "branches": branch_rows,
        "would_commit": would_commit,
        "hypothetical_target_agent_id": (
            int(winning_branch.target_agent_id) if winner else None
        ),
        "hypothetical_prompt_hash": winner.prompt_hash if winner else "",
        "parent_validation": parent_metrics,
        "hypothetical_validation": hypothetical_metrics,
        "realized_validation_vote_delta": realized_delta(
            would_commit, parent_metrics["vote"], hypothetical_metrics["vote"]
        ),
        "realized_validation_oracle_delta": realized_delta(
            would_commit, parent_metrics["oracle"], hypothetical_metrics["oracle"]
        ),
        "team_prompt_commit_count": 0,
        "trajectory_mutation_count": 0,
        "test_calls": comparator.test_evaluation_count,
    }
    write_json(out_dir / "cell_result.json", payload)
    return payload


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
    registry_payload = {k: v for k, v in registry.items() if k != "registry_content_hash"}
    if canonical_hash(registry_payload) != registry["registry_content_hash"]:
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
    results = []
    for case in registry["cases"]:
        for cell in CELLS:
            results.append(await run_cell(
                case, cell, args.out_root / case["case_id"] / cell,
                args.out_root / "_shared_solver_cache.sqlite",
            ))
    summary = {
        "probe_version": "v17_module1_2x2_fixed_parent_probe_v1",
        "execution_commit": head,
        "registry_content_hash": registry["registry_content_hash"],
        "case_count": 6, "cell_count": len(results),
        "would_commit_count": sum(row["would_commit"] for row in results),
        "team_prompt_commit_count": 0, "trajectory_mutation_count": 0,
        "test_calls": sum(row["test_calls"] for row in results),
        "results_hash": canonical_hash(results),
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
