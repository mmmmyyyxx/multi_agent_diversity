from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.cli import _load
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.persistence.checkpoint import restore_checkpoint
from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from scripts.model_headroom_screening_support import (
    ARMS, MODELS, RUN_ROOT, SEEDS, git, read_json, run_dir, sha256_file,
    validation_dir, write_json,
)


def config_from_meta(meta: dict[str, Any], out_dir: Path, cache: Path) -> Config:
    values = dict(meta["config"])
    values.update({
        "out_dir": str(out_dir.resolve()),
        "shared_solver_cache_path": str(cache.resolve()),
        "resume_from_checkpoint": False,
        "final_test_enabled": False,
        "preserve_final_checkpoint": False,
    })
    return Config.from_flat(**values)


def tree_manifest() -> dict[str, Any]:
    rows = []
    combined = hashlib.sha256()
    for model_key in MODELS:
        for seed in SEEDS:
            for arm in ARMS:
                base = run_dir(model_key, seed, arm)
                for path in sorted(row for row in base.rglob("*") if row.is_file()):
                    relative = path.relative_to(RUN_ROOT).as_posix()
                    digest = sha256_file(path)
                    rows.append({"path": relative, "sha256": digest})
                    combined.update(relative.encode() + b"\0" + digest.encode() + b"\n")
    return {
        "file_count": len(rows),
        "content_hash": combined.hexdigest(),
        "files": rows,
    }


async def evaluate_cell(model_key: str, seed: int, arm: str) -> dict[str, Any]:
    source = run_dir(model_key, seed, arm)
    out = validation_dir(model_key, seed, arm)
    out.mkdir(parents=True, exist_ok=False)
    meta = read_json(source / "run_meta.json")
    checkpoint_path = source / "training_checkpoint.json"
    checkpoint = read_json(checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    cfg = config_from_meta(meta, out, out / "_solver_cache.sqlite")
    if cfg.models.agent_model != MODELS[model_key]:
        raise RuntimeError("task model mismatch")
    system = PromptEnsembleOptimizationSystem(cfg)
    train = _load(cfg.data.train_path, cfg.data.train_size, cfg.data.dataset_format)
    validation = _load(cfg.data.val_path, cfg.data.val_size, cfg.data.dataset_format)
    system.set_run_identity(RunIdentity(**checkpoint["run_identity"]))
    system.proposal_memory_run_id = str(checkpoint["proposal_memory_run_id"])
    system.fixed_probe = system.build_probe(
        train[: min(len(train), cfg.evaluation.candidate_eval_pool_size)]
    )
    restore_checkpoint(system, checkpoint)
    system.llm.calls = []
    system.solver_recovery_observations = []
    system.solver_invalid_outputs = []
    expected_state = str(meta["final_state_selection"]["selected_team_prompt_state_hash"])
    before = system.team_prompt_state_hash()
    if before != expected_state:
        raise RuntimeError("final state mismatch before validation")
    metrics = await system.evaluate_dataset(validation)
    after = system.team_prompt_state_hash()
    if after != before or sha256_file(checkpoint_path) != checkpoint_hash:
        raise RuntimeError("validation mutated frozen training state")
    oracle_count = sum(
        any(
            profile[index].valid
            and system.match_answer(profile[index].answer, example.gold_answer)
            for profile in system._last_evaluated_profiles
        )
        for index, example in enumerate(system._last_evaluated_examples)
    )
    invalid_count = sum(
        not output.valid
        for profile in system._last_evaluated_profiles
        for output in profile
    )
    recovery = system.solver_recovery_summary()
    cost = system.cost_summary()
    result = {
        "artifact_version": "model_headroom_validation_v1",
        "model_key": model_key,
        "task_model": MODELS[model_key],
        "seed": seed,
        "arm": arm,
        "setting": ARMS[arm],
        "logical_validation_evaluation_count": 1,
        "validation_row_count": len(validation),
        "vote_correct_count": int(metrics.vote_correct_count),
        "vote_accuracy": float(metrics.plurality_vote_acc),
        "oracle_correct_count": int(oracle_count),
        "oracle_accuracy": oracle_count / len(validation),
        "per_agent_correct_counts": list(map(int, metrics.per_agent_correct_counts)),
        "per_agent_accuracies": [value / len(validation) for value in metrics.per_agent_correct_counts],
        "invalid_output_count": int(invalid_count),
        "invalid_output_rate": invalid_count / (len(validation) * 5),
        "first_attempt_invalid_count": int(recovery["first_attempt_invalid_count"]),
        "terminal_invalid_count": int(recovery["terminal_invalid_count"]),
        "resolved_request_count": int(recovery["unique_resolved_request_count"]),
        "provider_calls": int(cost["successful_llm_calls"]),
        "prompt_tokens": int(cost["prompt_tokens"]),
        "completion_tokens": int(cost["completion_tokens"]),
        "total_tokens": int(cost["total_tokens"]),
        "final_state_hash": before,
        "checkpoint_sha256": checkpoint_hash,
        "state_mutation": False,
        "checkpoint_mutation": False,
        "selection_change": False,
        "test_calls": 0,
    }
    write_json(out / "validation_summary_private.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("MODEL_HEADROOM_VALIDATION_AUTHORIZED") != "1":
        raise SystemExit("validation API execution is not authorized")
    if os.environ.get("MODEL_HEADROOM_SCREENING_AUTHORIZED") == "1":
        raise SystemExit("training and validation authorization cannot coexist")
    freeze = read_json(args.freeze)
    if freeze.get("source_freeze_status") != "PASS" or git("rev-parse", "HEAD") != freeze.get("execution_commit"):
        raise SystemExit("source freeze mismatch")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("worktree must be fully clean")
    if read_json(RUN_ROOT / "train_gate.json").get("gate") != "PASS":
        raise SystemExit("training gate must pass before validation")
    if (RUN_ROOT / "validation").exists():
        raise SystemExit("fresh validation root required")
    before = tree_manifest()
    write_json(RUN_ROOT / "training_pre_validation_manifest.json", before)
    results = []
    for model_key in MODELS:
        for seed in SEEDS:
            for arm in ARMS:
                results.append(asyncio.run(evaluate_cell(model_key, seed, arm)))
    after = tree_manifest()
    write_json(RUN_ROOT / "training_post_validation_manifest.json", after)
    if before != after:
        raise RuntimeError("training directories changed during validation")
    write_json(RUN_ROOT / "validation_complete.json", {
        "logical_validation_evaluation_count": len(results),
        "test_calls": 0,
        "training_tree_unchanged": True,
        "results_hash": hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest(),
    })
    print(json.dumps({"validation_cells": len(results), "test_calls": 0}, indent=2))


if __name__ == "__main__":
    main()

