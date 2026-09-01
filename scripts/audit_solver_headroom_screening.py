from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.solver_headroom_screening_support import (
    ARMS, GENERIC_UPDATES, ROLE_MODEL, RUN_ROOT, SEEDS, accepted_update_count,
    entrant_rows, git, infrastructure_failure_count, read_json, run_dir,
    sha256_file, validation_dir, write_json,
)


def audit_train(freeze: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    rows: list[dict[str, Any]] = []
    entrants = entrant_rows()
    initial_by_seed: dict[int, list[str]] = {}
    for entry in entrants:
        key, model = entry["model_key"], entry["solver_model"]
        for seed in SEEDS:
            for arm, setting in ARMS.items():
                run = run_dir(key, seed, arm)
                required = (
                    "run_meta.json", "final_summary.json",
                    "solver_recovery_summary.json", "comparison_cache_match.json",
                    "training_checkpoint.json",
                )
                missing = [name for name in required if not (run / name).exists()]
                if missing:
                    blockers.append(f"missing:{key}:{seed}:{arm}:{','.join(missing)}")
                    continue
                meta = read_json(run / "run_meta.json")
                final = read_json(run / "final_summary.json")
                recovery = read_json(run / "solver_recovery_summary.json")
                cache = read_json(run / "comparison_cache_match.json")
                cfg = meta["config"]
                expected = 0 if arm == "STATIC" else GENERIC_UPDATES
                checks = {
                    "setting": meta.get("canonical_experiment_setting") == setting,
                    "solver_model": cfg.get("agent_model") == model,
                    "optimizer_model": cfg.get("optimizer_model") == ROLE_MODEL,
                    "evaluator_model": cfg.get("evaluator_model") == ROLE_MODEL,
                    "seed": int(cfg.get("seed", -1)) == seed,
                    "agents": int(cfg.get("agents", -1)) == 5,
                    "train_size": int(cfg.get("train_size", -1)) == 75,
                    "val_size": int(cfg.get("val_size", -1)) == 50,
                    "epochs": int(cfg.get("epochs", -1)) == 4,
                    "update_every": int(cfg.get("update_every", -1)) == 10,
                    "memory_off": cfg.get("proposal_memory_mode") == "off",
                    "planned_updates": int(meta.get("planned_update_count", -1)) == expected,
                    "completed_updates": int(meta.get("completed_update_count", -1)) == expected,
                    "validation_zero": int(meta.get("validation_evaluation_count", -1)) == 0,
                    "test_zero": int(meta.get("test_evaluation_count", -1)) == 0,
                    "test_not_early": meta.get("test_called_before_training_complete") is False,
                    "final_active": meta.get("final_state_selection", {}).get("selected_source") == "final_active_state",
                    "training_complete": meta.get("training_completed") is True,
                    "source_commit": meta.get("run_identity", {}).get("git_commit") == freeze["execution_commit"],
                    "source_clean": meta.get("run_identity", {}).get("git_dirty") is False,
                    "cache_match": cache.get("matched") is True and cache.get("gate") == "PASS",
                    "selected_test_absent": final.get("selected_test") is None,
                    "initial_test_absent": final.get("initial_test") is None,
                    "module_vector": meta.get("module_vector") == ([0, 0] if arm == "GENERIC" else None),
                    "branch_count": int(meta.get("target_branch_count", -1)) == (0 if arm == "STATIC" else 1),
                    "candidate_count": int(meta.get("candidates_per_target_branch", -1)) == (0 if arm == "STATIC" else 2),
                }
                blockers.extend(
                    f"{name}:{key}:{seed}:{arm}" for name, passed in checks.items()
                    if not passed
                )
                initial = list(map(str, meta.get("initial_prompt_hashes", [])))
                if seed not in initial_by_seed:
                    initial_by_seed[seed] = initial
                elif initial_by_seed[seed] != initial:
                    blockers.append(f"cross_solver_or_arm_initialization:{seed}")
                infra = infrastructure_failure_count(run)
                if infra:
                    blockers.append(f"infrastructure_failure:{key}:{seed}:{arm}:{infra}")
                rows.append({
                    "model_key": key, "solver_model": model, "seed": seed,
                    "arm": arm, "setting": setting, "planned_updates": expected,
                    "completed_updates": int(meta["completed_update_count"]),
                    "accepted_updates": accepted_update_count(run),
                    "first_attempt_invalid_count": int(recovery["first_attempt_invalid_count"]),
                    "terminal_invalid_count": int(recovery["terminal_invalid_count"]),
                    "resolved_request_count": int(recovery["unique_resolved_request_count"]),
                    "infrastructure_failure_count": infra,
                })
    expected_count = len(entrants) * len(SEEDS) * len(ARMS)
    return {
        "audit_version": "solver_headroom_train_gate_v1", "phase": "train",
        "gate": "PASS" if not blockers and len(rows) == expected_count else "FAIL",
        "blockers": blockers, "entrant_count": len(entrants),
        "expected_run_count": expected_count, "run_count": len(rows),
        "validation_evaluation_count": 0, "test_evaluation_count": 0,
        "full_method_run": False, "rows": rows,
    }


def audit_validation(_: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    rows: list[dict[str, Any]] = []
    entrants = entrant_rows()
    complete = read_json(RUN_ROOT / "validation_complete.json")
    before = read_json(RUN_ROOT / "training_pre_validation_manifest.json")
    after = read_json(RUN_ROOT / "training_post_validation_manifest.json")
    expected_count = len(entrants) * len(SEEDS) * len(ARMS)
    if before != after or not complete.get("training_tree_unchanged"):
        blockers.append("training_tree_changed")
    if int(complete.get("logical_validation_evaluation_count", -1)) != expected_count:
        blockers.append("validation_count")
    if int(complete.get("test_calls", -1)) != 0:
        blockers.append("test_calls")
    for entry in entrants:
        key, model = entry["model_key"], entry["solver_model"]
        for seed in SEEDS:
            for arm, setting in ARMS.items():
                source = run_dir(key, seed, arm)
                path = validation_dir(key, seed, arm) / "validation_summary_private.json"
                if not path.exists():
                    blockers.append(f"missing_validation:{key}:{seed}:{arm}")
                    continue
                row = read_json(path)
                checks = {
                    "model": row.get("solver_model") == model,
                    "seed": int(row.get("seed", -1)) == seed,
                    "arm": row.get("arm") == arm,
                    "setting": row.get("setting") == setting,
                    "single_validation": int(row.get("logical_validation_evaluation_count", -1)) == 1,
                    "validation_rows": int(row.get("validation_row_count", -1)) == 50,
                    "state_unchanged": row.get("state_mutation") is False,
                    "checkpoint_unchanged": row.get("checkpoint_mutation") is False,
                    "selection_unchanged": row.get("selection_change") is False,
                    "test_zero": int(row.get("test_calls", -1)) == 0,
                    "checkpoint_hash": row.get("checkpoint_sha256") == sha256_file(source / "training_checkpoint.json"),
                }
                blockers.extend(
                    f"{name}:{key}:{seed}:{arm}" for name, passed in checks.items()
                    if not passed
                )
                rows.append(row)
    return {
        "audit_version": "solver_headroom_validation_gate_v1",
        "phase": "validation",
        "gate": "PASS" if not blockers and len(rows) == expected_count else "FAIL",
        "blockers": blockers, "expected_run_count": expected_count,
        "run_count": len(rows), "logical_validation_evaluation_count": len(rows),
        "test_evaluation_count": 0, "training_tree_unchanged": before == after,
        "full_method_run": False, "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "validation"), required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("fresh gate output required")
    freeze = read_json(args.freeze)
    outer: list[str] = []
    if freeze.get("source_freeze_status") != "PASS":
        outer.append("source_freeze")
    if git("rev-parse", "HEAD") != freeze.get("execution_commit"):
        outer.append("execution_commit")
    if git("status", "--porcelain", "--untracked-files=all"):
        outer.append("worktree_dirty")
    result = audit_train(freeze) if args.phase == "train" else audit_validation(freeze)
    result["blockers"] = outer + result["blockers"]
    result["gate"] = "PASS" if not result["blockers"] else "FAIL"
    write_json(args.out, result)
    print(json.dumps({
        "phase": args.phase, "gate": result["gate"],
        "run_count": result["run_count"],
        "blocker_count": len(result["blockers"]),
    }, indent=2))
    raise SystemExit(0 if result["gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
