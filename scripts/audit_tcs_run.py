import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.system import tcs_metadata_applicable, validate_tcs_candidate_metadata


EPS = 1e-9


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


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


def audit_run(run_dir: Path) -> Dict[str, Any]:
    update_rows = list(read_jsonl(run_dir / "update_logs.jsonl"))
    llm_rows = list(read_jsonl(run_dir / "llm_calls.jsonl"))
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    config = meta.get("config", {}) if isinstance(meta, dict) else {}
    candidates = [row for row in update_rows if row.get("event") == "candidate_evaluated"]
    applicable = [row for row in candidates if tcs_metadata_applicable(row)]
    invalid = [row for row in applicable if validate_tcs_candidate_metadata(row)]
    existing = [row for row in candidates if str(row.get("candidate_source", "")) == "existing_beam"]
    stages = Counter(normalize_stage(row) for row in llm_rows)
    empty_teacher = sum(not str(row.get("teacher_question", "") or "").strip() for row in applicable)
    zero_round = sum(int(row.get("teacher_critic_rounds", 0) or 0) < 1 for row in applicable)
    forced_best = sum(bool(row.get("teacher_question_forced_best_score", False)) for row in applicable)
    delta_inconsistency = 0
    for row in candidates:
        checks = (
            _delta_inconsistent(row, "accuracy_delta", "candidate_target_accuracy", "baseline_target_accuracy"),
            _delta_inconsistent(row, "diversity_delta", "candidate_embedding_diversity", "baseline_embedding_diversity"),
            _delta_inconsistent(row, "invalid_delta", "candidate_invalid_rate", "baseline_invalid_rate"),
            _delta_inconsistent(row, "vote_delta", "candidate_team_accuracy", "baseline_team_accuracy"),
            _delta_inconsistent(row, "coverage_delta", "candidate_oracle_acc", "baseline_oracle_acc"),
            abs(_float(row, "net_coverage_delta") - _float(row, "coverage_delta")) > EPS,
        )
        delta_inconsistency += int(any(checks))
    has_tcs_call_evidence = bool(stages["teacher"] and stages["critic"] and stages["student"])
    has_critic_without_evidence = any(int(row.get("teacher_critic_rounds", 0) or 0) > 0 for row in applicable) and not stages["critic"]
    problems = bool(invalid or delta_inconsistency or (applicable and not has_tcs_call_evidence) or has_critic_without_evidence)
    return {
        "run_dir": str(run_dir),
        "optimizer_architecture": config.get("optimizer_architecture", meta.get("optimizer_architecture", "")),
        "optimizer_candidate_count": sum(str(row.get("candidate_pool_source", "")) == "optimizer" for row in candidates),
        "existing_beam_candidate_count": len(existing),
        "tcs_applicable_candidate_count": len(applicable),
        "valid_tcs_metadata_count": len(applicable) - len(invalid),
        "invalid_tcs_metadata_count": len(invalid),
        "empty_teacher_question_count": empty_teacher,
        "zero_critic_round_count": zero_round,
        "forced_best_count": forced_best,
        "teacher_call_count": stages["teacher"],
        "critic_call_count": stages["critic"],
        "teacher_rewrite_call_count": stages["teacher_rewrite"],
        "student_call_count": stages["student"],
        "student_retry_call_count": stages["student_json_retry"],
        "student_repair_call_count": stages["student_json_repair"],
        "delta_inconsistency_count": delta_inconsistency,
        "has_tcs_call_evidence": has_tcs_call_evidence,
        "problems": problems,
    }


def find_run_dirs(path: Path) -> List[Path]:
    if (path / "update_logs.jsonl").exists() or (path / "llm_calls.jsonl").exists():
        return [path]
    return sorted({candidate.parent for candidate in path.rglob("update_logs.jsonl")})


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit TCS provenance and candidate delta consistency without changing run files.")
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
