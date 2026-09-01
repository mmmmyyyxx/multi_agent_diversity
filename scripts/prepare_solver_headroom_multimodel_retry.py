from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.solver_headroom_multimodel_support import (
    RETRY_FREEZE_ROOT,
    RUN_ROOT,
    entrants,
    git,
    read_json,
    run_dir,
    source_inventory,
    write_json,
)


def main() -> None:
    if RETRY_FREEZE_ROOT.exists():
        raise SystemExit("fresh retry freeze root required")
    if (RUN_ROOT / "validation_retry1").exists() or (RUN_ROOT / "generic_retry1").exists():
        raise SystemExit("fresh retry execution roots required")
    head = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("tracked worktree must be clean")
    if not (RUN_ROOT / "validation").is_dir():
        raise SystemExit("preserved failed validation evidence missing")
    rows = []
    for entry in entrants():
        source = run_dir(entry["key"], "STATIC")
        checkpoint = read_json(source / "training_checkpoint.json")
        summary = read_json(source / "final_summary.json")
        selection = summary["selection_summary"]
        if not checkpoint.get("training_completed") or int(selection.get("test_evaluation_count", -1)) != 0:
            raise RuntimeError(f"incomplete Static source: {entry['key']}")
        rows.append({
            "key": entry["key"],
            "model": entry["model"],
            "checkpoint_sha256": hashlib.sha256((source / "training_checkpoint.json").read_bytes()).hexdigest(),
            "solver_cache_sha256": hashlib.sha256((source / "_solver_cache.sqlite").read_bytes()).hexdigest(),
            "training_complete": True,
            "test_evaluation_count": 0,
        })
    files, tree_hash = source_inventory()
    write_json(RETRY_FREEZE_ROOT / "source_freeze.json", {
        "gate": "PASS",
        "execution_commit": head,
        "source_tree_hash": tree_hash,
        "source_file_count": len(files),
        "static_sources": rows,
        "static_source_count": len(rows),
        "old_validation_failure_preserved": True,
        "validation_retry_root": "validation_retry1",
        "generic_retry_root": "generic_retry1",
        "test_calls": 0,
        "static_reruns": 0,
    })
    print(json.dumps({"gate": "PASS", "execution_commit": head, "static_source_count": len(rows)}))


if __name__ == "__main__":
    main()
