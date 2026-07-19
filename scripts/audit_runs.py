"""Build a conservative, evidence-based retention plan for local run roots."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


VERSIONS = (
    "baseline",
    "v1_reward_redesign",
    "v2_teacher_critic_student",
    "v3_oracle_pareto",
    "v7_vote_oriented",
    "v8_stable_qd_lineage",
)


def read_json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def read_json(path: Path) -> dict[str, Any]:
    value = read_json_value(path)
    return value if isinstance(value, dict) else {}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
    except (OSError, ValueError, TypeError):
        return


def accuracy_rows(root: Path) -> list[dict[str, Any]]:
    jsonl_path = root / "accuracy_results.jsonl"
    rows = list(iter_jsonl(jsonl_path))
    if rows or not (root / "accuracy_results.csv").exists():
        return rows
    try:
        with (root / "accuracy_results.csv").open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError:
        return []


def first_value(meta: dict[str, Any], *names: str) -> Any:
    config = meta.get("config") if isinstance(meta.get("config"), dict) else {}
    for source in (meta, config):
        for name in names:
            value = source.get(name)
            if value not in (None, ""):
                return value
    return None


def truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def infer_meta_versions(meta: dict[str, Any], run_dir_name: str = "") -> set[str]:
    setting = str(first_value(meta, "experiment_setting", "setting") or run_dir_name).lower()
    method = str(first_value(meta, "method_version") or "").lower()
    reward = str(first_value(meta, "reward_mode") or "").lower()
    selector = str(first_value(meta, "candidate_selection_mode", "beam_selection_policy") or "").lower()
    architecture = str(first_value(meta, "optimizer_architecture") or "").lower()
    text = " ".join((setting, method, reward, selector, architecture))
    baseline = truthy(first_value(meta, "baseline_only")) or "baseline" in setting
    if baseline:
        return {"baseline"}
    if "v8_stable_qd_lineage" in method or "stable_qd" in method:
        return {"v8_stable_qd_lineage"}
    if "competence_depth" in text or "v8_2_hybrid" in text:
        return {"v8_historical_progressive"}
    if any(token in text for token in ("residual", "cycle_guard", "vote_error", "v7")):
        return {"v7_vote_oriented"}
    if "oracle_pareto" in text or ("oracle" in text and "pareto" in text):
        return {"v3_oracle_pareto"}
    if "teacher_critic_student" in text or "tcs" in text:
        return {"v2_teacher_critic_student"}
    if any(token in text for token in ("guarded_diversity", "coverage_useful_diversity", "phase_adaptive")):
        return {"v1_reward_redesign"}
    return set()


def history_test_metrics(path: Path) -> tuple[dict[str, Any], bool]:
    value = read_json_value(path)
    if not isinstance(value, list):
        return {}, False
    explicit_final = False
    for record in reversed(value):
        if not isinstance(record, dict):
            continue
        explicit_final = explicit_final or record.get("epoch") == "final"
        test = record.get("test")
        if isinstance(test, dict) and test and any(key in test for key in ("vote_acc", "num_test_samples", "size")):
            return test, explicit_final
    return {}, explicit_final


def is_completed_run_dir(directory: Path) -> tuple[bool, bool]:
    test, explicit_final = history_test_metrics(directory / "history.json")
    complete = all((directory / name).exists() for name in ("run_meta.json", "history.json", "cost_summary.json")) and bool(test)
    return complete, explicit_final


def classify_run_kind(root: Path, run_count: int, completed_count: int) -> tuple[str, str]:
    name = root.name.lower()
    if "smoke" in name or "acceptance" in name:
        return "SMOKE", "SMOKE" if completed_count == run_count and run_count else "FAILED"
    if any(marker in name for marker in ("pilot", "validation", "traincheck")):
        return "PILOT", "PILOT" if completed_count == run_count and run_count else "PARTIAL"
    if "formal" in name or "full" in name:
        return "FORMAL", "FORMAL" if completed_count == run_count and run_count else "PARTIAL"
    if run_count and completed_count == run_count:
        return "UNLABELED", "AMBIGUOUS"
    if run_count:
        return "UNLABELED", "PARTIAL"
    return "UNKNOWN", "FAILED"


def inspect_root(root: Path) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file()]
    meta_paths = list(root.rglob("run_meta.json"))
    meta_pairs = [(path.parent, read_json(path)) for path in meta_paths]
    results = accuracy_rows(root)
    settings = {
        str(value)
        for directory, meta in meta_pairs
        for value in (first_value(meta, "experiment_setting", "setting"), directory.name.rsplit("_seed", 1)[0])
        if value
    }
    settings.update(str(row.get("setting")) for row in results if row.get("setting"))
    methods = {str(first_value(meta, "method_version")) for _, meta in meta_pairs if first_value(meta, "method_version")}
    versions: set[str] = set()
    for directory, meta in meta_pairs:
        versions.update(infer_meta_versions(meta, directory.name))
    completed_count = explicit_final_count = 0
    for directory, _ in meta_pairs:
        complete, explicit_final = is_completed_run_dir(directory)
        completed_count += int(complete)
        explicit_final_count += int(explicit_final)
    tasks = {
        str(value)
        for _, meta in meta_pairs
        for value in (first_value(meta, "comparison_task_id"),)
        if value
    }
    tasks.update(str(row.get("task_id")) for row in results if row.get("task_id"))
    seeds = {
        int(value)
        for _, meta in meta_pairs
        for value in (first_value(meta, "seed"),)
        if value is not None and str(value).lstrip("-").isdigit()
    }
    strict_values = [bool(first_value(meta, "split_integrity_json")) for _, meta in meta_pairs]
    run_kind, classification = classify_run_kind(root, len(meta_pairs), completed_count)
    refill = any(
        int(row.get("refill_round_count", 0) or 0) > 0 or bool(row.get("post_archive_refill_triggered"))
        for path in root.rglob("update_logs.jsonl") for row in iter_jsonl(path)
    )
    resume = any(
        row.get("event") in {"run_resumed", "checkpoint_restored"} or bool(row.get("resumed_from_checkpoint"))
        for path in root.rglob("update_logs.jsonl") for row in iter_jsonl(path)
    )
    unknown_version = bool(meta_pairs) and not versions
    if unknown_version and classification not in {"FAILED", "PARTIAL"}:
        classification = "AMBIGUOUS"
    return {
        "path": root.name,
        "resolved_path": str(root.resolve()),
        "size_bytes": sum(path.stat().st_size for path in files),
        "file_count": len(files),
        "method_versions": sorted(methods),
        "versions": sorted(versions),
        "primary_version": next((version for version in reversed(VERSIONS) if version in versions), ""),
        "settings": sorted(settings),
        "tasks": sorted(tasks),
        "seeds": sorted(seeds),
        "status": classification,
        "has_final_test": completed_count > 0,
        "explicit_final_count": explicit_final_count,
        "has_best_state": any((directory / "best_prompts.json").exists() for directory, _ in meta_pairs),
        "has_run_meta": bool(meta_pairs),
        "strict_split": bool(strict_values and all(strict_values)),
        "run_kind": run_kind,
        "classification": classification,
        "run_count": len(meta_pairs),
        "completed_run_count": completed_count,
        "accuracy_result_count": len(results),
        "refill_evidence": refill,
        "checkpoint_resume_evidence": resume,
        "latest_mtime": max((path.stat().st_mtime for path in files), default=root.stat().st_mtime),
        "keep": classification == "AMBIGUOUS",
        "reason": "AMBIGUOUS defaults to keep" if classification == "AMBIGUOUS" else "unselected",
    }


def formal_score(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["classification"] == "FORMAL"),
        int(row["completed_run_count"] == row["run_count"] and row["run_count"] > 0),
        int(row["strict_split"]),
        len(row["tasks"]),
        len(row["seeds"]),
        row["completed_run_count"],
        row["latest_mtime"],
    )


def append_reason(row: dict[str, Any], reason: str) -> None:
    previous = row.get("reason", "")
    if previous in {"", "unselected"}:
        row["reason"] = reason
    elif reason not in previous:
        row["reason"] = f"{previous}; {reason}"


def choose_retention(rows: list[dict[str, Any]], retention: dict[str, Any] | None = None) -> None:
    retention = retention or {}
    preferred = retention.get("canonical_formal_roots", {})
    if not isinstance(preferred, dict):
        raise ValueError("canonical_formal_roots must be a mapping")
    # Formal roots may contain both baseline and one method, so one root can represent both.
    for version in VERSIONS:
        preferred_path = str(preferred.get(version, "") or "")
        if preferred_path:
            matches = [row for row in rows if row["path"] == preferred_path]
            if len(matches) != 1:
                raise ValueError(f"configured canonical run root is missing: {preferred_path}")
            chosen = matches[0]
            if version not in chosen["versions"]:
                raise ValueError(f"configured canonical root {preferred_path} has no {version} evidence")
            if not chosen["run_count"] or chosen["completed_run_count"] != chosen["run_count"]:
                raise ValueError(f"configured canonical root is incomplete: {preferred_path}")
            if chosen["reason"] == "AMBIGUOUS defaults to keep":
                chosen["reason"] = "unselected"
            chosen.update({"keep": True, "run_kind": "FORMAL", "classification": "FORMAL", "status": "FORMAL"})
            append_reason(chosen, f"reviewed canonical formal run for {version}")
            continue
        eligible = [row for row in rows if version in row["versions"] and row["run_kind"] == "FORMAL"]
        completed = [row for row in eligible if row["classification"] == "FORMAL"]
        if completed:
            chosen = max(completed, key=formal_score)
            chosen["keep"] = True
            append_reason(chosen, f"canonical formal run for {version}")
            for row in completed:
                if row is not chosen and not row["keep"]:
                    row["classification"] = "DUPLICATE"
                    append_reason(row, f"superseded formal run for {version}")

    v8_smokes = [row for row in rows if "v8_stable_qd_lineage" in row["versions"] and row["run_kind"] == "SMOKE"]
    roles = (
        ("latest completed V8 smoke", lambda row: row["classification"] == "SMOKE"),
        ("latest V8 refill evidence", lambda row: row["refill_evidence"]),
        ("latest V8 checkpoint/observability smoke", lambda row: row["checkpoint_resume_evidence"] or "acceptance" in row["path"]),
    )
    kept_paths: set[str] = set()
    for reason, predicate in roles:
        candidates = [row for row in v8_smokes if predicate(row)]
        if candidates:
            chosen = max(candidates, key=lambda row: row["latest_mtime"])
            chosen["keep"] = True
            append_reason(chosen, reason)
            kept_paths.add(chosen["path"])
    for row in v8_smokes:
        if row["path"] not in kept_paths and row["reason"] == "unselected":
            row["reason"] = "older V8 smoke beyond representative limit"

    for row in rows:
        if row["classification"] == "AMBIGUOUS":
            row.update({"keep": True, "reason": "AMBIGUOUS defaults to keep"})
        elif not row["keep"] and row["reason"] == "unselected":
            row["reason"] = f"non-canonical {row['classification'].lower()} run"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output", default="run_cleanup_plan.json")
    parser.add_argument("--retention-config", default="configs/run_retention.yaml")
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    retention_path = workspace / args.retention_config
    retention = yaml.safe_load(retention_path.read_text(encoding="utf-8")) or {}
    if not isinstance(retention, dict):
        raise ValueError("retention config must contain a mapping")
    rows = [inspect_root(path) for path in sorted(workspace.glob("runs_*")) if path.is_dir() and not path.is_symlink()]
    choose_retention(rows, retention)
    payload = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "retention_config": str(retention_path.resolve()),
        "roots": rows,
        "summary": {
            "root_count": len(rows),
            "size_bytes": sum(row["size_bytes"] for row in rows),
            "keep_count": sum(bool(row["keep"]) for row in rows),
            "delete_count": sum(not bool(row["keep"]) for row in rows),
            "ambiguous_keep_count": sum(row["classification"] == "AMBIGUOUS" for row in rows),
            "delete_size_bytes": sum(row["size_bytes"] for row in rows if not row["keep"]),
        },
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
