import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj
    except Exception:
        return


def _run_task_id(run_dir: Path) -> str:
    meta = _read_json(run_dir / "run_meta.json")
    for key in ("comparison_task_id", "task_id", "mars_task_id"):
        value = str(meta.get(key, "") or "").strip()
        if value:
            return value
    return run_dir.parent.name if run_dir.parent != run_dir else run_dir.name


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _is_tcs_student_row(row: Dict[str, Any]) -> bool:
    if str(row.get("optimizer_architecture", "") or "") == "teacher_critic_student":
        return True
    return any(str(key).startswith("student_") for key in row)


def _student_final_failure(row: Dict[str, Any]) -> bool:
    if not _is_tcs_student_row(row):
        return False
    return _int(row.get("student_candidate_count_final", 0)) == 0


def _recovery_status(row: Dict[str, Any]) -> str:
    final_count = _int(row.get("student_candidate_count_final", 0))
    retry_succeeded = _safe_bool(row.get("student_json_retry_succeeded", False))
    repair_succeeded = _safe_bool(row.get("student_json_repair_succeeded", False))
    parse_failed = _safe_bool(row.get("student_json_parse_failed", False))
    retry_attempted = _safe_bool(row.get("student_json_retry_attempted", False))
    repair_attempted = _safe_bool(row.get("student_json_repair_attempted", False))
    raw_empty = _safe_bool(row.get("student_raw_response_empty", False))
    if final_count > 0 and retry_succeeded:
        return "retry_recovered"
    if final_count > 0 and repair_succeeded:
        return "repair_recovered"
    if final_count > 0 and parse_failed:
        return "parse_failed_recovered"
    if final_count == 0 and raw_empty:
        return "raw_empty_final_failure"
    if final_count == 0 and (parse_failed or retry_attempted or repair_attempted):
        return "parse_failed_unrecovered"
    if final_count == 0:
        return "final_student_failure"
    return "ok"


def summarize_run(update_log: Path) -> List[Dict[str, Any]]:
    run_dir = update_log.parent
    task_id = _run_task_id(run_dir)
    rows = [
        row
        for row in _iter_jsonl(update_log)
        if row.get("event") == "beam_update_summary" and _is_tcs_student_row(row)
    ]
    total = len(rows)
    teacher_approved_count = sum(1 for row in rows if _safe_bool(row.get("teacher_question_approved", False)))
    final_failure_count = sum(1 for row in rows if _student_final_failure(row))
    parse_failure_count = sum(1 for row in rows if _safe_bool(row.get("student_json_parse_failed", False)))
    retry_recovery_count = sum(
        1
        for row in rows
        if _safe_bool(row.get("student_json_retry_succeeded", False))
        and _int(row.get("student_candidate_count_final", 0)) > 0
    )
    repair_recovery_count = sum(
        1
        for row in rows
        if _safe_bool(row.get("student_json_repair_succeeded", False))
        and _int(row.get("student_candidate_count_final", 0)) > 0
    )
    target_rows = [
        row
        for row in rows
        if _student_final_failure(row)
        or _safe_bool(row.get("student_json_retry_succeeded", False))
        or _safe_bool(row.get("student_json_repair_succeeded", False))
        or _safe_bool(row.get("student_json_parse_failed", False))
    ]
    grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in target_rows:
        retry_attempted = _safe_bool(row.get("student_json_retry_attempted", False))
        retry_succeeded = _safe_bool(row.get("student_json_retry_succeeded", False))
        repair_attempted = _safe_bool(row.get("student_json_repair_attempted", False))
        repair_succeeded = _safe_bool(row.get("student_json_repair_succeeded", False))
        parse_failed = _safe_bool(row.get("student_json_parse_failed", False))
        recovery_status = _recovery_status(row)
        final_student_failure = _student_final_failure(row)
        key = (
            str(row.get("student_failure_stage", "") or "unknown"),
            _int(row.get("student_candidate_count_raw", 0)),
            _int(row.get("student_candidate_count_final", 0)),
            _safe_bool(row.get("student_raw_response_empty", False)),
            parse_failed,
            _safe_bool(row.get("student_json_has_candidates_key", False)),
            _safe_bool(row.get("student_candidates_is_list", False)),
            _safe_bool(row.get("student_candidates_empty_list", False)),
            _safe_bool(row.get("student_refusal_or_explanation", False)),
            retry_attempted,
            retry_succeeded,
            repair_attempted,
            repair_succeeded,
            str(row.get("student_json_repair_failure_reason", ""))[:500],
            recovery_status,
            final_student_failure,
        )
        if key not in grouped:
            grouped[key] = {
                "run_dir": str(run_dir),
                "task_id": task_id,
                "total_update_summaries": total,
                "teacher_approved_count": teacher_approved_count,
                "student_final_failure_count": final_failure_count,
                "student_json_parse_failure_count": parse_failure_count,
                "student_retry_recovery_count": retry_recovery_count,
                "student_repair_recovery_count": repair_recovery_count,
                "student_candidate_count_raw": key[1],
                "student_candidate_count_final": key[2],
                "failure_stage": key[0],
                "student_raw_response_empty": key[3],
                "student_json_parse_failed": key[4],
                "student_json_has_candidates_key": key[5],
                "student_candidates_is_list": key[6],
                "student_candidates_empty_list": key[7],
                "student_refusal_or_explanation": key[8],
                "student_json_retry_attempted": key[9],
                "student_json_retry_succeeded": key[10],
                "student_json_repair_attempted": key[11],
                "student_json_repair_succeeded": key[12],
                "student_json_repair_failure_reason": key[13],
                "recovery_status": key[14],
                "final_student_failure": key[15],
                "count": 0,
                "rate_within_update_summaries": 0.0,
                "example_raw_response_preview": str(row.get("student_raw_response_preview", ""))[:1000],
                "example_retry_raw_response_preview": str(row.get("student_json_retry_raw_response_preview", ""))[:1000],
                "example_repair_raw_response_preview": str(row.get("student_json_repair_raw_response_preview", ""))[:1000],
                "example_teacher_question": str(row.get("teacher_question", ""))[:500],
            }
        grouped[key]["count"] += 1
    for item in grouped.values():
        item["rate_within_update_summaries"] = float(item["count"] / total) if total else 0.0
    return list(grouped.values())


def find_update_logs(paths: List[str]) -> List[Path]:
    logs: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.name == "update_logs.jsonl":
            logs.append(path)
        elif path.is_dir():
            logs.extend(sorted(path.rglob("update_logs.jsonl")))
    return sorted(set(logs))


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    fields = [
        "run_dir",
        "task_id",
        "total_update_summaries",
        "teacher_approved_count",
        "student_final_failure_count",
        "student_json_parse_failure_count",
        "student_retry_recovery_count",
        "student_repair_recovery_count",
        "failure_stage",
        "student_candidate_count_raw",
        "student_candidate_count_final",
        "count",
        "rate_within_update_summaries",
        "student_raw_response_empty",
        "student_json_parse_failed",
        "student_json_has_candidates_key",
        "student_candidates_is_list",
        "student_candidates_empty_list",
        "student_refusal_or_explanation",
        "student_json_retry_attempted",
        "student_json_retry_succeeded",
        "student_json_repair_attempted",
        "student_json_repair_succeeded",
        "student_json_repair_failure_reason",
        "recovery_status",
        "final_student_failure",
        "example_raw_response_preview",
        "example_retry_raw_response_preview",
        "example_repair_raw_response_preview",
        "example_teacher_question",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize final Student generation failures and JSON retry/repair recovery.")
    parser.add_argument("run_dirs", nargs="+", help="Run directories or update_logs.jsonl files")
    parser.add_argument("--out_csv", type=str, default="student_failure_summary.csv")
    args = parser.parse_args()

    logs = find_update_logs(args.run_dirs)
    all_rows: List[Dict[str, Any]] = []
    for log in logs:
        all_rows.extend(summarize_run(log))
    out_path = Path(args.out_csv)
    write_csv(all_rows, out_path)

    totals = defaultdict(int)
    for row in all_rows:
        totals[(row.get("task_id", ""), row.get("recovery_status", ""), row.get("failure_stage", ""))] += int(row.get("count", 0) or 0)
    print(f"Found {len(logs)} update_logs.jsonl files")
    print(f"Wrote {out_path}")
    if not totals:
        print("No final Student failures or JSON recovery events found.")
        return
    for (task_id, recovery_status, stage), count in sorted(totals.items()):
        print(f"{task_id}\t{recovery_status}\t{stage}\t{count}")


if __name__ == "__main__":
    main()
