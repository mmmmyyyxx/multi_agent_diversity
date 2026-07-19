"""Strictly validate repository run records and selected run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_json(path: Path) -> None:
    json.loads(path.read_text(encoding="utf-8"))


def validate_jsonl(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--include-runs", action="store_true")
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    json_files = [workspace / "run_cleanup_plan.json", *sorted((workspace / "run_records").glob("*.json"))]
    jsonl_files = sorted((workspace / "run_records").glob("*.jsonl"))
    if args.include_runs:
        for root in sorted(workspace.glob("runs_*")):
            if root.is_dir() and not root.is_symlink():
                json_files.extend(root.rglob("*.json"))
                jsonl_files.extend(root.rglob("*.jsonl"))
    for path in json_files:
        validate_json(path)
    row_count = sum(validate_jsonl(path) for path in jsonl_files)
    print(f"validated_json={len(json_files)} jsonl_files={len(jsonl_files)} jsonl_rows={row_count}")


if __name__ == "__main__":
    main()
