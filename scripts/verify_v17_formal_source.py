from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from scripts.v17_formal_support import (
    ROOT, git, read_json, sha256_file, tracked_source_inventory,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    frozen = read_json(args.freeze)
    errors = []
    if git("rev-parse", "HEAD") != frozen.get("git_head"):
        errors.append("git_head")
    if git("status", "--porcelain", "--untracked-files=all"):
        errors.append("git_dirty")
    files, source_hash = tracked_source_inventory()
    if source_hash != frozen.get("working_tree_source_hash"):
        errors.append("source_hash")
    expected = {row["path"]: row["sha256"] for row in frozen.get("files", [])}
    actual = {row["path"]: row["sha256"] for row in files}
    if actual != expected:
        errors.append("source_inventory")
    for relative, digest in frozen.get("definition_sha256", {}).items():
        if sha256_file(ROOT / relative) != digest:
            errors.append("definition_hash")
    print(json.dumps({
        "gate": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
    }, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
