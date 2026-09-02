from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.solver_headroom_multimodel_support import (
    GENERIC_RETRY_FREEZE_ROOT,
    GENERIC_ROOT,
    GENERIC_VALIDATION_ROOT,
    entrants,
    git,
    read_json,
    selected_generic,
    source_inventory,
    validation_dir,
    write_json,
)


def main() -> None:
    if GENERIC_RETRY_FREEZE_ROOT.exists() or GENERIC_ROOT.exists() or GENERIC_VALIDATION_ROOT.exists():
        raise SystemExit("fresh Generic retry4 roots required")
    head = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("tracked worktree must be clean")
    if len(entrants()) != 6:
        raise RuntimeError("Static entrant inventory mismatch")
    for entry in entrants():
        summary = read_json(validation_dir(entry["key"], "STATIC") / "validation_summary_private.json")
        if summary["test_calls"] != 0 or summary["source_artifact_tree_mutation"]:
            raise RuntimeError(f"invalid Static validation evidence: {entry['key']}")
    selected = selected_generic()
    if [row["key"] for row in selected] != ["FLASH", "TURBO"]:
        raise RuntimeError("frozen Static-gate v2 selection mismatch")
    anchor = read_json(validation_dir("Q8", "GENERIC") / "validation_summary_private.json")
    if anchor["test_calls"] != 0:
        raise RuntimeError("anchor test isolation mismatch")
    files, tree_hash = source_inventory()
    write_json(GENERIC_RETRY_FREEZE_ROOT / "source_freeze.json", {
        "gate": "PASS",
        "execution_commit": head,
        "source_tree_hash": tree_hash,
        "source_file_count": len(files),
        "selection_version": "static_headroom_gate_v2_structural_oracle_fix",
        "selected": selected,
        "qwen3_8b_anchor_reused": True,
        "new_generic_training_count": len(selected),
        "generic_retry_root": "generic_retry4",
        "generic_validation_root": "validation_retry4",
        "test_calls": 0,
    })
    print(json.dumps({"gate": "PASS", "execution_commit": head, "selected": selected}))


if __name__ == "__main__":
    main()
