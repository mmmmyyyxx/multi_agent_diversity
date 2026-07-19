"""Archive compact records, then remove only reviewed repository run roots."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def read_result_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "accuracy_results.jsonl"
    rows = []
    if not path.exists():
        return rows
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    except (OSError, ValueError, TypeError):
        return []
    return rows


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def active_command_lines() -> list[str]:
    try:
        command = [
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process | Select-Object -ExpandProperty CommandLine",
        ]
        return subprocess.run(command, capture_output=True, text=True, check=False).stdout.splitlines()
    except OSError:
        return []


def validate_target(workspace: Path, name: str, command_lines: list[str]) -> Path:
    relative = Path(name)
    if not name.startswith("runs_") or relative.name != name or ".." in relative.parts:
        raise ValueError(f"unsafe run root name: {name}")
    unresolved = workspace / name
    if unresolved.is_symlink():
        raise ValueError(f"refusing symlink run root: {unresolved}")
    target = unresolved.resolve()
    if target.parent != workspace or target == workspace:
        raise ValueError(f"target escapes workspace: {target}")
    if not target.exists() or not target.is_dir():
        raise FileNotFoundError(f"planned run root is missing: {target}")
    normalized_target = str(target).lower()
    if any(normalized_target in line.lower() for line in command_lines if line):
        raise RuntimeError(f"run root appears in an active process: {target}")
    for lock in ("RUNNING.lock", ".lock", "runner.lock"):
        if (target / lock).exists():
            raise RuntimeError(f"run root has active lock {lock}: {target}")
    return target


def compact_record(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    metas = [read_json(path) for path in root.rglob("run_meta.json")]
    costs = [read_json(path) for path in root.rglob("cost_summary.json")]
    accuracy = read_json(root / "accuracy_summary.json")
    result_rows = read_result_rows(root)
    failure = ""
    for path in root.rglob("*.log"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "Traceback" in text or "ERROR" in text:
            failure = text[-1200:]
            break
    return {
        "schema_version": 1,
        "original_path": str(root),
        "deleted_at": datetime.now(timezone.utc).isoformat(),
        "classification": row["classification"],
        "reason": row["reason"],
        "size_bytes": row["size_bytes"],
        "file_count": row["file_count"],
        "versions": row.get("versions", []),
        "settings": row.get("settings", []),
        "tasks": row.get("tasks", []),
        "seeds": row.get("seeds", []),
        "run_meta_summary": [
            {
                "git_commit": meta.get("git_commit"),
                "setting": meta.get("setting") or (meta.get("config") or {}).get("setting"),
                "method_version": meta.get("method_version") or (meta.get("config") or {}).get("method_version"),
                "comparison_task_id": meta.get("comparison_task_id") or (meta.get("config") or {}).get("comparison_task_id"),
            }
            for meta in metas
        ],
        "cost_summary": costs,
        "accuracy_summary": accuracy or {
            "result_count": len(result_rows),
            "results": [
                {
                    key: result.get(key)
                    for key in (
                        "task_id", "setting", "seed", "status", "vote_acc",
                        "mean_individual_acc", "oracle_acc", "elapsed_sec",
                    )
                    if key in result
                }
                for result in result_rows
            ],
        },
        "failure_summary": failure,
    }


def collect_targets(
    workspace: Path,
    plan: dict[str, Any],
    command_lines: list[str],
) -> list[tuple[Path, dict[str, Any]]]:
    if plan.get("schema_version") != 2:
        raise ValueError("unsupported cleanup plan schema; regenerate and review the plan")
    if Path(str(plan.get("workspace", ""))).resolve() != workspace:
        raise ValueError("cleanup plan workspace does not match requested workspace")
    roots = plan.get("roots")
    if not isinstance(roots, list):
        raise ValueError("cleanup plan roots must be a list")
    targets = []
    seen: set[str] = set()
    for row in roots:
        if not isinstance(row, dict):
            raise ValueError("cleanup plan contains a non-object row")
        name = str(row.get("path", ""))
        if name in seen:
            raise ValueError(f"cleanup plan contains duplicate path: {name}")
        seen.add(name)
        if row.get("keep") or row.get("classification") == "AMBIGUOUS":
            continue
        target = validate_target(workspace, name, command_lines)
        if str(row.get("resolved_path", "")).lower() != str(target).lower():
            raise ValueError(f"planned resolved path changed: {name}")
        targets.append((target, row))
    return targets


def apply_plan(
    workspace: Path,
    targets: list[tuple[Path, dict[str, Any]]],
    *,
    command_line_provider: Callable[[], list[str]] = active_command_lines,
) -> list[dict[str, Any]]:
    records_dir = workspace / "run_records"
    records_dir.mkdir(exist_ok=True)
    records = []
    for target, row in targets:
        # Recheck immediately before each destructive action.
        validate_target(workspace, target.name, command_line_provider())
        record = compact_record(target, row)
        atomic_write_json(records_dir / f"{target.name}.json", record)
        records.append(record)
        with (records_dir / "index.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        shutil.rmtree(target)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--plan", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    plan = read_json(Path(args.plan))
    targets = collect_targets(workspace, plan, active_command_lines())
    print(json.dumps({
        "mode": "apply" if args.apply else "dry-run",
        "target_count": len(targets),
        "targets": [str(path) for path, _ in targets],
    }, ensure_ascii=False, indent=2))
    if args.apply:
        apply_plan(workspace, targets)


if __name__ == "__main__":
    main()
