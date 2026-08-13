from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from multi_dataset_diverse_rl.compatibility_repair import (
    LOSS_BLIND_GENERIC_REVISION_SYSTEM_PROMPT,
    ONLINE_COMPATIBILITY_REPAIR_VERSION,
    REPAIR_SYSTEM_PROMPT,
)
from multi_dataset_diverse_rl.versions import CHECKPOINT_VERSION, METHOD_VERSION
from scripts.v17_formal_support import (
    ARMS, EXPERIMENT_ROOT, ROOT, SEEDS, git, read_json, sha256_bytes,
    sha256_file, sha256_json, source_semantics_diff, split_freeze,
    tracked_source_inventory, write_json,
)


DEFINITIONS = (
    "DESIGN_SPEC.md", "preregistration.json", "formal_matrix.json",
    "success_classifier.json", "dataset_freeze.json", "execution_order.json",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full_tests", required=True)
    parser.add_argument("--smokes", required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() or ROOT.resolve() not in out.parents:
        raise SystemExit("fresh repo-local freeze directory required")
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=all")
    files, source_hash = tracked_source_inventory()
    prereg = read_json(EXPERIMENT_ROOT / "preregistration.json")
    data = split_freeze()
    semantics = source_semantics_diff()
    definitions = {
        f"experiments/v17_formal_5arm_3seed_20260813/{name}": sha256_file(
            EXPERIMENT_ROOT / name
        ) for name in DEFINITIONS
    }
    errors = []
    if dirty:
        errors.append("tracked_or_untracked_worktree_dirty")
    if prereg.get("formal_seeds") != list(SEEDS):
        errors.append("formal_seeds")
    if prereg.get("arms") != list(ARMS):
        errors.append("formal_arms")
    if data["gate"] != "PASS":
        errors.append("dataset_freeze")
    if semantics["gate"] != "PASS":
        errors.append("frozen_v16_semantics")
    mechanism = {
        "M20": sha256_file(ROOT / "multi_dataset_diverse_rl" / "tcs.py"),
        "M2F": sha256_bytes((
            ONLINE_COMPATIBILITY_REPAIR_VERSION + "\n" + REPAIR_SYSTEM_PROMPT
        ).encode("utf-8")),
        "loss_blind_generic_revision": sha256_bytes(
            LOSS_BLIND_GENERIC_REVISION_SYSTEM_PROMPT.encode("utf-8")
        ),
        "Common_Safe": sha256_file(
            ROOT / "multi_dataset_diverse_rl" / "candidate_selection.py"
        ),
        "Module1_selector": sha256_file(
            ROOT / "multi_dataset_diverse_rl" / "responsibility.py"
        ),
    }
    payload = {
        "freeze_version": "v17_formal_source_freeze_v1",
        "source_freeze_status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "git_head": head,
        "repo_dirty": bool(dirty),
        "working_tree_source_hash": source_hash,
        "source_file_count": len(files),
        "files": files,
        "definition_sha256": definitions,
        "method_version": METHOD_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "arm_definitions_hash": sha256_json(read_json(EXPERIMENT_ROOT / "formal_matrix.json")),
        "mechanism_hashes": mechanism,
        "dataset_freeze": data,
        "seeds": list(SEEDS),
        "execution_order_hash": sha256_file(EXPERIMENT_ROOT / "execution_order.json"),
        "classifier_hash": prereg["classifier_hash"],
        "models": {"agent": "qwen3-14b", "optimizer": "qwen3-14b", "evaluator": "qwen3-14b"},
        "thinking": False,
        "source_candidate_budget": "2x2",
        "provider_call_ceiling": 8000,
        "token_ceiling": 3000000,
        "full_test_result": args.full_tests,
        "smoke_result": args.smokes,
        "api_calls": 0,
        "model_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
    }
    out.mkdir(parents=True)
    write_json(out / "source_freeze_manifest.json", payload)
    write_json(out / "phase_a_verification.json", {
        "status": payload["source_freeze_status"],
        "errors": errors,
        "execution_commit": head,
        "formal_cells": 15,
        "api_calls": 0, "model_calls": 0,
        "validation_calls": 0, "test_calls": 0,
    })
    print(json.dumps({
        "status": payload["source_freeze_status"], "errors": errors,
        "execution_commit": head, "source_file_count": len(files),
    }, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
