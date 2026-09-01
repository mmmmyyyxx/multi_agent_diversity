from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.versions import CHECKPOINT_VERSION, METHOD_VERSION
from scripts.model_headroom_screening_support import (
    ARMS, EVALUATOR_MODEL, GENERIC_UPDATES, MANIFEST, MODELS, OPTIMIZER_MODEL,
    ROOT, RUN_ROOT, SCREENING_VERSION, SEEDS, git, sha256_file,
    tracked_source_inventory, write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RUN_ROOT / "freeze")
    args = parser.parse_args()
    if RUN_ROOT.exists():
        raise SystemExit("fresh screening run root required")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("worktree must be fully clean")
    if git("branch", "--show-current") != "main":
        raise SystemExit("screening must freeze main")
    files, source_hash = tracked_source_inventory()
    head = git("rev-parse", "HEAD")
    freeze = {
        "freeze_version": "model_headroom_source_freeze_v1",
        "source_freeze_status": "PASS",
        "execution_commit": head,
        "source_tree_hash": source_hash,
        "source_file_count": len(files),
        "source_files": files,
        "method_version": METHOD_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "screening_version": SCREENING_VERSION,
        "manifest_sha256": sha256_file(MANIFEST),
        "models": MODELS,
        "optimizer_model": OPTIMIZER_MODEL,
        "evaluator_model": EVALUATOR_MODEL,
        "thinking": False,
        "seeds": list(SEEDS),
        "arms": ARMS,
        "generic_planned_updates": GENERIC_UPDATES,
        "full_method_run": False,
        "test_accessed": False,
    }
    preregistration = {
        "preregistration_version": "model_headroom_preregistration_v1",
        "frozen_before_api": True,
        "execution_commit": head,
        "task": "disambiguation_qa",
        "models": MODELS,
        "task_agent_role_only": True,
        "optimizer_model": OPTIMIZER_MODEL,
        "evaluator_model": EVALUATOR_MODEL,
        "thinking": False,
        "seeds": list(SEEDS),
        "arms": ARMS,
        "train_size": 75,
        "validation_size": 50,
        "test_evaluation_enabled": False,
        "agents": 5,
        "aggregation": "equal_weight_plurality_tie_abstain",
        "epochs": 4,
        "update_every": 10,
        "generic_planned_updates": GENERIC_UPDATES,
        "candidate_budget_per_update": 2,
        "stage_b_budget": 2,
        "proposal_memory_mode": "off",
        "validation_schedule": "post_training_final_state_once",
        "validation_used_for_selection": False,
        "forbidden_settings": ["Full", "Module1", "M20", "M2F"],
        "selection_thresholds": {
            "static_mean_vote_acc_max": 0.65,
            "mean_vote_uplift_min": 0.04,
            "generic_mean_oracle_vote_gap_min": 0.08,
            "generic_vote_win_seed_count_min": 2,
            "terminal_invalid_rate_max": 0.01,
        },
        "extra_seed_allowed": False,
        "full_method_run": False,
        "test_accessed": False,
    }
    args.out.mkdir(parents=True)
    write_json(args.out / "source_freeze.json", freeze)
    write_json(args.out / "preregistration.json", preregistration)
    print(json.dumps({
        "gate": "PASS", "execution_commit": head,
        "source_file_count": len(files), "run_count": 12,
        "full_method_run": False, "test_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
