from __future__ import annotations

import argparse
import asyncio
import hashlib
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
from scripts.solver_headroom_multimodel_support import (
    RETRY_FREEZE_ROOT, RUN_ROOT, candidates_by_key, entrants, git, read_json, run_dir,
    selected_generic, sha256_file, validation_dir, write_json,
)


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


async def evaluate(key: str, model: str, arm: str) -> dict[str, Any]:
    source = run_dir(key, arm)
    out = validation_dir(key, arm)
    out.mkdir(parents=True, exist_ok=False)
    meta = read_json(source / "run_meta.json")
    checkpoint_path = source / "training_checkpoint.json"
    checkpoint = read_json(checkpoint_path)
    before_hash = sha256_file(checkpoint_path)
    before_tree_hash = tree_hash(source)
    values = dict(meta["config"])
    values.update({"out_dir": str(out.resolve()), "shared_solver_cache_path": str((out / "_solver_cache.sqlite").resolve()), "resume_from_checkpoint": False, "final_test_enabled": False, "preserve_final_checkpoint": False})
    cfg = Config.from_flat(**values)
    if cfg.models.agent_model != model:
        raise RuntimeError("model mismatch")
    system = PromptEnsembleOptimizationSystem(cfg)
    train = _load(cfg.data.train_path, cfg.data.train_size, cfg.data.dataset_format)
    validation = _load(cfg.data.val_path, cfg.data.val_size, cfg.data.dataset_format)
    system.set_run_identity(RunIdentity(**checkpoint["run_identity"]))
    system.proposal_memory_run_id = str(checkpoint["proposal_memory_run_id"])
    system.fixed_probe = system.build_probe(train[: min(len(train), cfg.evaluation.candidate_eval_pool_size)])
    restore_checkpoint(system, checkpoint)
    system.llm.calls=[]; system.solver_recovery_observations=[]; system.solver_invalid_outputs=[]
    state = system.team_prompt_state_hash()
    metrics = await system.evaluate_dataset(validation)
    if (system.team_prompt_state_hash() != state or sha256_file(checkpoint_path) != before_hash
            or tree_hash(source) != before_tree_hash):
        raise RuntimeError("validation mutation")
    oracle = sum(any(profile[i].valid and system.match_answer(profile[i].answer, example.gold_answer) for profile in system._last_evaluated_profiles) for i, example in enumerate(system._last_evaluated_examples))
    invalid = sum(not output.valid for profile in system._last_evaluated_profiles for output in profile)
    recovery, cost = system.solver_recovery_summary(), system.cost_summary()
    result = {
        "key": key, "solver_model": model, "seed": 65, "arm": arm,
        "vote_accuracy": float(metrics.plurality_vote_acc),
        "oracle_accuracy": oracle / len(validation),
        "per_agent_accuracies": [v / len(validation) for v in metrics.per_agent_correct_counts],
        "invalid_output_count": invalid,
        "terminal_invalid_count": int(recovery["terminal_invalid_count"]),
        "resolved_request_count": int(recovery["unique_resolved_request_count"]),
        "provider_calls": int(cost["successful_llm_calls"]),
        "prompt_tokens": int(cost["prompt_tokens"]), "completion_tokens": int(cost["completion_tokens"]),
        "state_mutation": False, "checkpoint_mutation": False,
        "source_artifact_tree_mutation": False,
        "test_calls": 0, "validation_rows": len(validation),
    }
    write_json(out / "validation_summary_private.json", result)
    return result


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=("static","generic"),required=True);args=parser.parse_args()
    if os.environ.get("SOLVER_MULTIMODEL_VALIDATION_AUTHORIZED") != "1":
        raise SystemExit("validation not authorized")
    freeze=read_json(RETRY_FREEZE_ROOT/"source_freeze.json")
    if git("rev-parse","HEAD") != freeze["execution_commit"] or git("status","--porcelain","--untracked-files=all"):
        raise SystemExit("source mismatch")
    rows = entrants() if args.phase == "static" else selected_generic()
    if args.phase == "generic" and not any(row["key"] == "Q8" for row in rows):
        anchor = next((row for row in entrants() if row["key"] == "Q8"), None)
        if anchor is not None:
            rows = [*rows, anchor]
    arm = args.phase.upper()
    results=[]
    for row in rows:
        out=validation_dir(row["key"],arm)/"validation_summary_private.json"
        if out.exists():
            if args.phase == "generic" and row["key"] == "Q8":
                results.append(read_json(out));continue
            raise SystemExit("fresh validation output required")
        results.append(asyncio.run(evaluate(row["key"],row["model"],arm)))
    write_json(RUN_ROOT/f"{args.phase}_validation_complete.json",{"phase":args.phase,"count":len(results),"test_calls":0,"results":results})
    print({"phase":args.phase,"count":len(results),"test_calls":0})


if __name__ == "__main__":
    main()
