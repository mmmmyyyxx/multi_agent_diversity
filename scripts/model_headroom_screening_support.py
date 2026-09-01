from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "model_headroom_screening_20260901"
REPORT_ROOT = ROOT / "reports" / "model_headroom_screening_20260901"
MANIFEST = ROOT / "configs" / "task_level_comparison_strict_bbh_seed42.yaml"
MODELS = {
    "A": "qwen2.5-7b-instruct",
    "B": "qwen3-8b",
}
SEEDS = (62, 63, 64)
ARMS = {
    "STATIC": "shared_static_reference",
    "GENERIC": "shared_generic_evolution",
}
OPTIMIZER_MODEL = "qwen3-14b"
EVALUATOR_MODEL = "qwen3-14b"
GENERIC_UPDATES = 32
SELECTION_VERSION = "task_model_headroom_selection_v1"
SCREENING_VERSION = "task_model_headroom_screening_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
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
        "experiments/model_headroom_screening_20260901",
    ).splitlines()
    rows: list[dict[str, str]] = []
    combined = hashlib.sha256()
    for relative in sorted(filter(None, paths)):
        normalized = relative.replace("\\", "/")
        digest = sha256_file(ROOT / relative)
        rows.append({"path": normalized, "sha256": digest})
        combined.update(
            normalized.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n"
        )
    return rows, combined.hexdigest()


def run_dir(model_key: str, seed: int, arm: str) -> Path:
    setting = ARMS[arm]
    return (
        RUN_ROOT / f"model_{model_key}" / f"seed{seed}" / "disambiguation_qa"
        / f"{setting}_seed{seed}"
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
    payload = read_json(path)
    return sum(
        int(row.get("infrastructure_failed_updates", 0))
        for row in payload.get("updates", [])
    )


def selection_rule(model_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    evaluations: dict[str, dict[str, Any]] = {}
    for model_key in MODELS:
        row = model_rows[model_key]
        criteria = {
            "static_mean_vote_at_most_0_65": row["static_mean_vote_acc"] <= 0.65,
            "mean_vote_uplift_at_least_0_04": row["mean_vote_uplift"] >= 0.04,
            "generic_mean_oracle_vote_gap_at_least_0_08": (
                row["generic_mean_oracle_vote_gap"] >= 0.08
            ),
            "generic_wins_at_least_2_of_3": row["generic_vote_win_count"] >= 2,
            "no_serious_output_instability": not row["serious_output_instability"],
        }
        evaluations[model_key] = {
            "model": MODELS[model_key],
            "criteria": criteria,
            "pass": all(criteria.values()),
        }
    passed = [key for key, row in evaluations.items() if row["pass"]]
    selected = ""
    reason = "both_models_fail"
    if len(passed) == 1:
        selected = passed[0]
        reason = "only_passing_model"
    elif len(passed) == 2:
        a, b = passed
        uplift_delta = (
            model_rows[a]["mean_vote_uplift"] - model_rows[b]["mean_vote_uplift"]
        )
        if abs(uplift_delta) > 0.01:
            selected = a if uplift_delta > 0 else b
            reason = "larger_mean_vote_uplift"
        else:
            gap_delta = (
                model_rows[a]["generic_mean_oracle_vote_gap"]
                - model_rows[b]["generic_mean_oracle_vote_gap"]
            )
            if abs(gap_delta) > 0.01:
                selected = a if gap_delta > 0 else b
                reason = "larger_generic_oracle_vote_gap"
            else:
                static_delta = (
                    model_rows[a]["static_mean_vote_acc"]
                    - model_rows[b]["static_mean_vote_acc"]
                )
                if static_delta != 0:
                    selected = a if static_delta < 0 else b
                    reason = "lower_static_vote_accuracy"
                else:
                    reason = "exact_tie_after_frozen_criteria"
    return {
        "selection_version": SELECTION_VERSION,
        "model_evaluations": evaluations,
        "selected_model_key": selected,
        "selected_task_model": MODELS[selected] if selected else "",
        "decision": "SELECT" if selected else "HOLD",
        "reason": reason,
        "full_method_run": False,
        "test_accessed": False,
    }


def recursive_sanitization_problems(value: Any, path: str = "$") -> list[str]:
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
            problems.extend(recursive_sanitization_problems(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            problems.extend(recursive_sanitization_problems(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if "final_answer:" in lowered or "dashscope_" in lowered:
            problems.append(f"forbidden_text:{path}")
        if ":\\" in value or value.startswith("/"):
            problems.append(f"absolute_path:{path}")
    return problems
