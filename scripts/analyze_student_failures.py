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


def summarize_run(update_log: Path) -> List[Dict[str, Any]]:
    run_dir = update_log.parent
    task_id = _run_task_id(run_dir)
    rows = [row for row in _iter_jsonl(update_log) if row.get("event") == "beam_update_summary"]
    total = len(rows)
    teacher_approved = [row for row in rows if _safe_bool(row.get("teacher_question_approved", False))]
    approved_count = len(teacher_approved)
    target_rows = [
        row
        for row in teacher_approved
        if int(row.get("student_candidate_count_raw", 0) or 0) == 0
    ]
    grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in target_rows:
        retry_attempted = _safe_bool(row.get("student_json_retry_attempted", False))
        retry_succeeded = _safe_bool(row.get("student_json_retry_succeeded", False))
        repair_attempted = _safe_bool(row.get("student_json_repair_attempted", False))
        repair_succeeded = _safe_bool(row.get("student_json_repair_succeeded", False))
        parse_failed = _safe_bool(row.get("student_json_parse_failed", False))
        if retry_succeeded:
            recovery_status = "parse_failed_but_retry_recovered"
        elif repair_succeeded:
            recovery_status = "parse_failed_but_repair_recovered"
        elif parse_failed or retry_attempted or repair_attempted:
            recovery_status = "parse_failed_unrecovered"
        else:
            recovery_status = "not_parse_failure"
        final_student_failure = not retry_succeeded and not repair_succeeded
        key = (
            str(row.get("student_failure_stage", "") or "unknown"),
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
                "teacher_approved_count": approved_count,
                "teacher_approved_student_raw_zero_count": len(target_rows),
                "failure_stage": key[0],
                "student_raw_response_empty": key[1],
                "student_json_parse_failed": key[2],
                "student_json_has_candidates_key": key[3],
                "student_candidates_is_list": key[4],
                "student_candidates_empty_list": key[5],
                "student_refusal_or_explanation": key[6],
                "student_json_retry_attempted": key[7],
                "student_json_retry_succeeded": key[8],
                "student_json_repair_attempted": key[9],
                "student_json_repair_succeeded": key[10],
                "student_json_repair_failure_reason": key[11],
                "recovery_status": key[12],
                "final_student_failure": key[13],
                "count": 0,
                "rate_within_teacher_approved": 0.0,
                "example_raw_response_preview": str(row.get("student_raw_response_preview", ""))[:1000],
                "example_retry_raw_response_preview": str(row.get("student_json_retry_raw_response_preview", ""))[:1000],
                "example_repair_raw_response_preview": str(row.get("student_json_repair_raw_response_preview", ""))[:1000],
                "example_teacher_question": str(row.get("teacher_question", ""))[:500],
            }
        grouped[key]["count"] += 1
    for item in grouped.values():
        item["rate_within_teacher_approved"] = (
            float(item["count"] / approved_count) if approved_count else 0.0
        )
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
        "teacher_approved_student_raw_zero_count",
        "failure_stage",
        "count",
        "rate_within_teacher_approved",
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
    parser = argparse.ArgumentParser(description="Summarize Teacher-approved Student generation failures.")
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
        print("No teacher-approved raw-zero Student failures found.")
        return
    for (task_id, recovery_status, stage), count in sorted(totals.items()):
        print(f"{task_id}\t{recovery_status}\t{stage}\t{count}")


if __name__ == "__main__":
    main()
