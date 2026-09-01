from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "solver_headroom_screening_20260901"
REPORT_ROOT = ROOT / "reports" / "solver_headroom_screening_20260901"
MANIFEST = ROOT / "configs" / "task_level_comparison_strict_bbh_seed42.yaml"
CANDIDATES = (
    ("A", "qwen3-8b"),
    ("B", "qwen3-4b-instruct-2507"),
    ("C", "qwen3-1.7b"),
)
SEEDS = (65, 66, 67)
ARMS = {
    "STATIC": "shared_static_reference",
    "GENERIC": "shared_generic_evolution",
}
ROLE_MODEL = "qwen3.7-flash"
GENERIC_UPDATES = 32
SCREENING_VERSION = "solver_headroom_screening_v1"
SELECTION_VERSION = "solver_headroom_selection_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def tracked_source_inventory() -> tuple[list[dict[str, str]], str]:
    paths = git(
        "ls-files", "multi_dataset_diverse_rl", "scripts", "tests",
        "experiments/solver_headroom_screening_20260901",
    ).splitlines()
    rows: list[dict[str, str]] = []
    combined = hashlib.sha256()
    for relative in sorted(filter(None, paths)):
        normalized = relative.replace("\\", "/")
        digest = sha256_file(ROOT / relative)
        rows.append({"path": normalized, "sha256": digest})
        combined.update(
            normalized.encode() + b"\0" + digest.encode("ascii") + b"\n"
        )
    return rows, combined.hexdigest()


def entrant_rows() -> list[dict[str, Any]]:
    phase_a = read_json(RUN_ROOT / "phase_a" / "availability_smoke_private.json")
    return [row for row in phase_a["candidates"] if row["screening_eligible"]]


def run_dir(model_key: str, seed: int, arm: str) -> Path:
    setting = ARMS[arm]
    return (
        RUN_ROOT / "training" / f"model_{model_key}" / f"seed{seed}"
        / "disambiguation_qa" / f"{setting}_seed{seed}"
    )


def validation_dir(model_key: str, seed: int, arm: str) -> Path:
    return RUN_ROOT / "validation" / f"model_{model_key}" / f"seed{seed}" / arm


def accepted_update_count(run: Path) -> int:
    path = run / "candidate_decisions.jsonl"
    if not path.exists():
        return 0
    return sum(bool(row.get("accepted_prompt_hash")) for row in read_jsonl(path))


def infrastructure_failure_count(run: Path) -> int:
    path = run / "candidate_funnel.json"
    if not path.exists():
        return 0
    return sum(
        int(row.get("infrastructure_failed_updates", 0))
        for row in read_json(path).get("updates", [])
    )


def select_solver(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    evaluations: dict[str, Any] = {}
    for key, row in rows.items():
        criteria = {
            "static_mean_vote_at_most_0_65": row["static_mean_vote_acc"] <= 0.65,
            "mean_vote_uplift_at_least_0_04": row["mean_vote_uplift"] >= 0.04,
            "generic_wins_at_least_2_of_3": row["generic_vote_win_count"] >= 2,
            "generic_mean_oracle_vote_gap_at_least_0_08": (
                row["generic_mean_oracle_vote_gap"] >= 0.08
            ),
            "no_serious_output_instability": not row["serious_output_instability"],
        }
        evaluations[key] = {
            "solver_model": row["solver_model"],
            "criteria": criteria,
            "pass": all(criteria.values()),
        }
    passed = [key for key, row in evaluations.items() if row["pass"]]
    selected = ""
    reason = "no_solver_passes"
    if len(passed) == 1:
        selected, reason = passed[0], "only_passing_solver"
    elif len(passed) > 1:
        best_uplift = max(rows[key]["mean_vote_uplift"] for key in passed)
        finalists = [key for key in passed if rows[key]["mean_vote_uplift"] == best_uplift]
        if len(finalists) > 1:
            best_gap = max(rows[key]["generic_mean_oracle_vote_gap"] for key in finalists)
            finalists = [key for key in finalists if rows[key]["generic_mean_oracle_vote_gap"] == best_gap]
        if len(finalists) > 1:
            lowest_static = min(rows[key]["static_mean_vote_acc"] for key in finalists)
            finalists = [key for key in finalists if rows[key]["static_mean_vote_acc"] == lowest_static]
        if len(finalists) == 1:
            selected, reason = finalists[0], "frozen_lexicographic_tie_break"
        else:
            reason = "exact_tie_after_frozen_criteria"
    return {
        "selection_version": SELECTION_VERSION,
        "decision": "SELECT" if selected else "HOLD",
        "selected_solver_key": selected,
        "selected_solver_model": rows[selected]["solver_model"] if selected else "",
        "reason": reason,
        "solver_evaluations": evaluations,
        "role_model": ROLE_MODEL,
        "full_method_run": False,
        "test_accessed": False,
    }


def sanitization_problems(value: Any, path: str = "$") -> list[str]:
    forbidden = {
        "prompt", "question", "gold_answer", "model_answer", "raw_response",
        "response", "endpoint", "api_key", "checkpoint", "cache_path",
    }
    problems: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in forbidden or normalized.endswith("_prompt"):
                problems.append(f"forbidden_key:{path}.{key}")
            problems.extend(sanitization_problems(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            problems.extend(sanitization_problems(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if "final_answer:" in lowered or "dashscope_" in lowered:
            problems.append(f"forbidden_text:{path}")
        if ":\\" in value or value.startswith("/"):
            problems.append(f"absolute_path:{path}")
    return problems
