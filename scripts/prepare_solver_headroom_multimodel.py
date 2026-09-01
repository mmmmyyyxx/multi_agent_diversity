from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.versions import CHECKPOINT_VERSION, METHOD_VERSION
from scripts.solver_headroom_multimodel_support import (
    CANDIDATES, MANIFEST, OLD_ROOT, ROLE_MODEL, RUN_ROOT, git, read_json,
    sha256_file, source_inventory, write_json,
)


def main() -> None:
    if RUN_ROOT.exists():
        raise SystemExit("fresh multimodel root required")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("clean worktree required")
    old_static = OLD_ROOT / "training/model_A/seed65/disambiguation_qa/shared_static_reference_seed65"
    old_generic = OLD_ROOT / "training/model_A/seed65/disambiguation_qa/shared_generic_evolution_seed65"
    for run, expected in ((old_static, 0), (old_generic, 32)):
        meta = read_json(run / "run_meta.json")
        if int(meta.get("completed_update_count", -1)) != expected or not meta.get("training_completed"):
            raise SystemExit("qwen3-8b Seed65 anchor incomplete")
        if int(meta.get("test_evaluation_count", -1)) != 0:
            raise SystemExit("qwen3-8b anchor test contamination")
    old_seed66 = OLD_ROOT / "training/model_A/seed66/disambiguation_qa/shared_generic_evolution_seed66/training_checkpoint.json"
    interrupted = read_json(old_seed66) if old_seed66.exists() else {}
    files, tree_hash = source_inventory()
    freeze = {
        "gate": "PASS", "execution_commit": git("rev-parse", "HEAD"),
        "source_tree_hash": tree_hash, "source_files": files,
        "method_version": METHOD_VERSION, "checkpoint_version": CHECKPOINT_VERSION,
        "manifest_sha256": sha256_file(MANIFEST), "seed": 65,
        "candidates": [{"key": k, "model": m, "priority": i + 1} for i, (k, m) in enumerate(CANDIDATES)],
        "role_model": ROLE_MODEL,
        "qwen3_8b_anchor": {
            "execution_commit": read_json(old_generic / "run_meta.json")["run_identity"]["git_commit"],
            "static_run_hash": sha256_file(old_static / "run_meta.json"),
            "generic_run_hash": sha256_file(old_generic / "run_meta.json"),
            "completed_updates": 32,
        },
        "excluded_seed66": {
            "completed_updates": int(interrupted.get("completed_update_count", -1)),
            "training_complete": bool(interrupted.get("training_completed", False)),
            "status": "INTERRUPTED_EXCLUDED",
        },
        "full_method_run": False, "test_accessed": False,
    }
    write_json(RUN_ROOT / "freeze/source_freeze.json", freeze)
    print({"gate": "PASS", "commit": freeze["execution_commit"], "anchor_updates": 32})


if __name__ == "__main__":
    main()
