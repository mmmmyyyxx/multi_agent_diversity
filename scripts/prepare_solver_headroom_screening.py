from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.versions import CHECKPOINT_VERSION, METHOD_VERSION
from scripts.solver_headroom_screening_support import (
    ARMS, CANDIDATES, GENERIC_UPDATES, MANIFEST, ROLE_MODEL, RUN_ROOT,
    SCREENING_VERSION, SEEDS, git, sha256_file, tracked_source_inventory,
    write_json,
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
    execution_commit = git("rev-parse", "HEAD")
    args.out.mkdir(parents=True, exist_ok=False)
    freeze = {
        "source_freeze_status": "PASS",
        "screening_version": SCREENING_VERSION,
        "execution_commit": execution_commit,
        "source_tree_hash": source_hash,
        "source_file_count": len(files),
        "source_files": files,
        "method_version": METHOD_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "manifest_sha256": sha256_file(MANIFEST),
        "candidates": [
            {"model_key": key, "solver_model": model, "priority": index + 1}
            for index, (key, model) in enumerate(CANDIDATES)
        ],
        "excluded_model": "qwen2.5-7b-instruct",
        "role_model": ROLE_MODEL,
        "seeds": list(SEEDS),
        "arms": ARMS,
        "generic_update_count": GENERIC_UPDATES,
        "train_size": 75,
        "validation_size": 50,
        "test_accessed": False,
        "full_method_run": False,
    }
    write_json(args.out / "source_freeze.json", freeze)
    print({
        "gate": "PASS", "execution_commit": execution_commit,
        "candidate_count": len(CANDIDATES), "seed_count": len(SEEDS),
        "test_accessed": False,
    })


if __name__ == "__main__":
    main()
