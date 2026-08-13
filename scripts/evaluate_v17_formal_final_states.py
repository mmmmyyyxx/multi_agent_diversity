from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.cli import _load
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.persistence.checkpoint import restore_checkpoint
from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from scripts.v17_formal_support import (
    ARMS, EXECUTION_ORDER, MANIFEST, RUN_ROOT, SEEDS, SPLIT_ROOT,
    git, read_json, sha256_file, split_freeze, write_json,
)


def config_from_meta(meta: dict, *, out_dir: Path, cache: Path) -> Config:
    values = dict(meta["config"])
    values.update({
        "out_dir": str(out_dir.resolve()),
        "shared_solver_cache_path": str(cache.resolve()),
        "resume_from_checkpoint": False,
        "final_test_enabled": False,
        "preserve_final_checkpoint": False,
    })
    return Config.from_flat(**values)


async def evaluate_cell(run_dir: Path, phase: str, out_dir: Path) -> dict:
    meta = read_json(run_dir / "run_meta.json")
    checkpoint_path = run_dir / "training_checkpoint.json"
    checkpoint = read_json(checkpoint_path)
    split_path = SPLIT_ROOT / ("val.csv" if phase == "validation" else "test.csv")
    expected_state = str(meta["final_state_selection"]["selected_team_prompt_state_hash"])
    before_checkpoint_hash = sha256_file(checkpoint_path)
    cache = out_dir / "_solver_cache.sqlite"
    cfg = config_from_meta(meta, out_dir=out_dir, cache=cache)
    system = PromptEnsembleOptimizationSystem(cfg)
    train = _load(cfg.data.train_path, cfg.data.train_size, cfg.data.dataset_format)
    val = _load(cfg.data.val_path, cfg.data.val_size, cfg.data.dataset_format)
    test = _load(cfg.data.test_path, cfg.data.test_size, cfg.data.dataset_format)
    system.set_run_identity(RunIdentity(**checkpoint["run_identity"]))
    # The checkpoint binds Proposal Memory to the original training out_dir.
    # Proposal Memory is off and evaluation never reads it, but exact restore
    # still validates the persisted run-scoped identifier.
    system.proposal_memory_run_id = str(checkpoint["proposal_memory_run_id"])
    probe = train[: min(len(train), cfg.evaluation.candidate_eval_pool_size)]
    system.fixed_probe = system.build_probe(probe)
    restore_checkpoint(system, checkpoint)
    # Phase accounting is independent: restored training call records are
    # provenance, not calls made by this validation/test evaluation.
    system.llm.calls = []
    state_before = system.team_prompt_state_hash()
    if state_before != expected_state:
        raise RuntimeError("frozen final state hash mismatch before evaluation")
    rows = val if phase == "validation" else test
    metrics = await system.evaluate_dataset(rows)
    state_after = system.team_prompt_state_hash()
    if state_after != state_before:
        raise RuntimeError("final-state mutation during evaluation")
    if sha256_file(checkpoint_path) != before_checkpoint_hash:
        raise RuntimeError("checkpoint mutation during evaluation")
    cost = system.cost_summary()
    result = {
        "artifact_version": "v17_final_state_evaluation_v1",
        "phase": phase,
        "setting": meta["canonical_experiment_setting"],
        "seed": int(meta["config"]["seed"]),
        "logical_evaluation_count": 1,
        "final_state_hash": state_before,
        "checkpoint_sha256": before_checkpoint_hash,
        "split_file_sha256": sha256_file(split_path),
        "row_count": len(rows),
        "vote_correct_count": metrics.vote_correct_count,
        "vote_accuracy": metrics.plurality_vote_acc,
        "per_agent_correct_counts": list(metrics.per_agent_correct_counts),
        "oracle_correct_count": sum(
            any(profile[index].valid and system.match_answer(profile[index].answer, example.gold_answer)
                for profile in system._last_evaluated_profiles)
            for index, example in enumerate(system._last_evaluated_examples)
        ),
        "provider_calls": int(cost["successful_llm_calls"]),
        "prompt_tokens": int(cost["prompt_tokens"]),
        "completion_tokens": int(cost["completion_tokens"]),
        "total_tokens": int(cost["total_tokens"]),
        "state_mutation": False,
        "selection_change": False,
        "checkpoint_mutation": False,
    }
    write_json(out_dir / "evaluation_summary_private.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("validation", "test"), required=True)
    parser.add_argument("--run_root", type=Path, default=RUN_ROOT)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    authorization = (
        "V17_FORMAL_VALIDATION_AUTHORIZED" if args.phase == "validation"
        else "V17_FORMAL_TEST_AUTHORIZED"
    )
    if os.environ.get(authorization) != "1":
        raise SystemExit(f"{args.phase} API execution is not authorized")
    other = (
        "V17_FORMAL_TEST_AUTHORIZED" if args.phase == "validation"
        else "V17_FORMAL_VALIDATION_AUTHORIZED"
    )
    if os.environ.get(other) == "1" or os.environ.get("V17_FORMAL_TRAIN_AUTHORIZED") == "1":
        raise SystemExit("exactly one V17 phase authorization is allowed")
    freeze = read_json(args.freeze)
    if freeze.get("source_freeze_status") != "PASS" or git("rev-parse", "HEAD") != freeze.get("git_head"):
        raise SystemExit("source freeze gate failed")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("worktree must be fully clean")
    if split_freeze()["gate"] != "PASS":
        raise SystemExit("dataset freeze gate failed")
    if args.phase == "validation":
        train_gate = read_json(args.run_root / "train_protocol_gate.json")
        if train_gate.get("gate") != "PASS":
            raise SystemExit("training gate must PASS before validation")
    else:
        seal = read_json(args.run_root / "pre_test_seal.json")
        if seal.get("gate") != "PASS" or seal.get("test_calls_so_far") != 0:
            raise SystemExit("pre-test seal must PASS")
    phase_root = args.run_root / args.phase
    if phase_root.exists():
        raise FileExistsError(f"fresh {args.phase} output root required")
    results = []
    for seed in SEEDS:
        for arm in EXECUTION_ORDER[seed]:
            setting = ARMS[arm]
            run = args.run_root / f"seed{seed}" / "disambiguation_qa" / f"{setting}_seed{seed}"
            cell = phase_root / f"seed{seed}" / arm
            cell.mkdir(parents=True)
            results.append(asyncio.run(evaluate_cell(run, args.phase, cell)))
    write_json(args.run_root / f"{args.phase}_complete.json", {
        "phase": args.phase, "logical_evaluation_count": len(results),
        "execution_commit": freeze["git_head"],
        "results_hash": hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest(),
    })
    print(json.dumps({"phase": args.phase, "logical_evaluation_count": len(results)}, indent=2))


if __name__ == "__main__":
    main()
