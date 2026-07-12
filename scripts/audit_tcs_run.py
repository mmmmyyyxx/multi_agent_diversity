import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.system import tcs_metadata_applicable, validate_tcs_candidate_metadata


EPS = 1e-9


def read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    """Read JSONL without hiding corruption from an audit."""
    rows: List[Dict[str, Any]] = []
    malformed: List[Dict[str, Any]] = []
    if not path.exists():
        return rows, 0, malformed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            malformed.append({"path": str(path), "line_number": line_number, "error": str(exc), "line_preview": line[:240]})
            continue
        if isinstance(value, dict):
            rows.append(value)
        else:
            malformed.append({"path": str(path), "line_number": line_number, "error": "JSONL row is not an object", "line_preview": line[:240]})
    return rows, len(malformed), malformed


def _float(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def normalize_stage(row: Dict[str, Any]) -> str:
    value = str(row.get("llm_call_stage", "") or "")
    if value:
        return value
    stage = str(row.get("stage", "") or "").lower()
    if "teacher_rewrite" in stage:
        return "teacher_rewrite"
    if "teacher_critic" in stage:
        return "critic"
    if stage.startswith("teacher_"):
        return "teacher"
    if "student_json_retry" in stage:
        return "student_json_retry"
    if "student_json_repair" in stage:
        return "student_json_repair"
    if "student_" in stage:
        return "student"
    if "solver" in stage:
        return "solver"
    return "one_shot_optimizer" if "optimizer" in stage else stage


def _delta_inconsistent(row: Dict[str, Any], delta: str, candidate: str, baseline: str) -> bool:
    return abs(_float(row, delta) - (_float(row, candidate) - _float(row, baseline))) > EPS


def _successful_nonempty(rows: List[Dict[str, Any]], stage: str) -> List[Dict[str, Any]]:
    return [row for row in rows if normalize_stage(row) == stage and bool(row.get("call_succeeded", row.get("success", False))) and not bool(row.get("response_empty", False))]


def _group_problems(group_id: str, candidates: List[Dict[str, Any]], calls: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    expected = ("parent_id", "agent_id", "epoch", "step")
    for field in expected:
        values = {str(row.get(field)) for row in candidates}
        if len(values) != 1:
            errors.append(f"candidate_{field}_mismatch")
        call_values = {str(row.get(field)) for row in calls if row.get(field) is not None}
        if call_values and values and not call_values.issubset(values):
            errors.append(f"call_{field}_mismatch")
    rounds = {int(row.get("teacher_critic_rounds", 0) or 0) for row in candidates}
    rewrites = {int(row.get("teacher_rewrite_count", 0) or 0) for row in candidates}
    if len(rounds) != 1 or len(rewrites) != 1:
        errors.append("candidate_tcs_metadata_mismatch")
        return errors
    expected_rounds, expected_rewrites = rounds.pop(), rewrites.pop()
    if len(_successful_nonempty(calls, "teacher")) < 1:
        errors.append("missing_successful_teacher")
    if len(_successful_nonempty(calls, "critic")) != expected_rounds:
        errors.append("critic_round_count_mismatch")
    if len(_successful_nonempty(calls, "teacher_rewrite")) != expected_rewrites:
        errors.append("teacher_rewrite_count_mismatch")
    if not (_successful_nonempty(calls, "student") or _successful_nonempty(calls, "student_json_retry") or _successful_nonempty(calls, "student_json_repair")):
        errors.append("missing_successful_student")
    return errors


def audit_run(run_dir: Path) -> Dict[str, Any]:
    update_rows, update_malformed, update_locations = read_jsonl(run_dir / "update_logs.jsonl")
    llm_rows, llm_malformed, llm_locations = read_jsonl(run_dir / "llm_calls.jsonl")
    meta_path = run_dir / "run_meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    except json.JSONDecodeError:
        meta = {}
    config = meta.get("config", {}) if isinstance(meta, dict) else {}
    candidates = [row for row in update_rows if row.get("event") == "candidate_evaluated"]
    applicable = [row for row in candidates if tcs_metadata_applicable(row)]
    invalid = [row for row in applicable if validate_tcs_candidate_metadata(row)]
    existing = [row for row in candidates if str(row.get("candidate_pool_source", "")) == "existing_beam"]
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    missing_group = 0
    for row in applicable:
        group_id = str(row.get("tcs_call_group_id", "") or "")
        if not group_id:
            missing_group += 1
        else:
            groups[group_id].append(row)
    calls_by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    attempted_groups = set()
    for row in llm_rows:
        group_id = str(row.get("tcs_call_group_id", "") or "")
        if group_id:
            attempted_groups.add(group_id)
            calls_by_group[group_id].append(row)
    group_errors = {group_id: _group_problems(group_id, rows, calls_by_group[group_id]) for group_id, rows in groups.items()}
    incomplete = {group_id: errors for group_id, errors in group_errors.items() if errors}
    successful_stage_counts = Counter(normalize_stage(row) for row in llm_rows if bool(row.get("call_succeeded", row.get("success", False))) and not bool(row.get("response_empty", False)))
    delta_inconsistency = sum(
        int(any((
            _delta_inconsistent(row, "accuracy_delta", "candidate_target_accuracy", "baseline_target_accuracy"),
            _delta_inconsistent(row, "diversity_delta", "candidate_embedding_diversity", "baseline_embedding_diversity"),
            _delta_inconsistent(row, "invalid_delta", "candidate_invalid_rate", "baseline_invalid_rate"),
            _delta_inconsistent(row, "vote_delta", "candidate_team_accuracy", "baseline_team_accuracy"),
            _delta_inconsistent(row, "coverage_delta", "candidate_oracle_acc", "baseline_oracle_acc"),
            abs(_float(row, "net_coverage_delta") - _float(row, "coverage_delta")) > EPS,
        )))
        for row in candidates
    )
    malformed_locations = update_locations + llm_locations
    problems = bool(invalid or missing_group or incomplete or delta_inconsistency or malformed_locations)
    return {
        "run_dir": str(run_dir),
        "optimizer_architecture": config.get("optimizer_architecture", meta.get("optimizer_architecture", "")),
        "optimizer_candidate_count": sum(str(row.get("candidate_pool_source", "")) == "optimizer" for row in candidates),
        "existing_beam_candidate_count": len(existing),
        "tcs_applicable_candidate_count": len(applicable),
        "valid_tcs_metadata_count": len(applicable) - len(invalid),
        "invalid_tcs_metadata_count": len(invalid),
        "missing_tcs_call_group_id_count": missing_group,
        "attempted_tcs_group_count": len(attempted_groups),
        "candidate_producing_tcs_group_count": len(groups),
        "completed_tcs_group_count": len(groups) - len(incomplete),
        "failed_tcs_group_count": len(attempted_groups - set(groups)),
        "unexplained_incomplete_tcs_group_count": len(incomplete),
        "incomplete_tcs_groups": incomplete,
        "malformed_jsonl_count": len(malformed_locations),
        "malformed_locations": malformed_locations,
        "teacher_call_count": successful_stage_counts["teacher"],
        "critic_call_count": successful_stage_counts["critic"],
        "teacher_rewrite_call_count": successful_stage_counts["teacher_rewrite"],
        "student_call_count": successful_stage_counts["student"],
        "student_retry_call_count": successful_stage_counts["student_json_retry"],
        "student_repair_call_count": successful_stage_counts["student_json_repair"],
        "delta_inconsistency_count": delta_inconsistency,
        "problems": problems,
    }


def find_run_dirs(path: Path) -> List[Path]:
    if (path / "update_logs.jsonl").exists() or (path / "llm_calls.jsonl").exists():
        return [path]
    return sorted({candidate.parent for candidate in path.rglob("update_logs.jsonl")})


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit TCS group provenance and candidate delta consistency without changing run files.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dirs = find_run_dirs(args.run_dir)
    if not run_dirs:
        print(f"No run directories with update_logs.jsonl found under {args.run_dir}")
        return 2
    reports = [audit_run(run_dir) for run_dir in run_dirs]
    for report in reports:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 1 if any(report["problems"] for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
