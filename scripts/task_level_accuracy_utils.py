from pathlib import Path
from typing import Any, Dict

try:
    from scripts.experiment_io import read_json
except ModuleNotFoundError:
    from experiment_io import read_json


ACCURACY_RESULT_COLUMNS = [
    "task_id",
    "benchmark",
    "method_id",
    "setting",
    "seed",
    "dataset_format",
    "split_protocol",
    "leakage_warning",
    "num_test_samples",
    "vote_acc",
    "majority_vote_acc",
    "weighted_vote_acc",
    "mean_individual_acc",
    "best_individual_acc",
    "oracle_acc",
    "aggregation_gap",
    "rescue_available_rate",
    "correct_disagreement_rate",
    "mean_useful_diversity",
    "aggregation_mode",
    "solver_calls",
    "optimizer_calls",
    "evaluator_calls",
    "total_llm_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "estimated_cost",
    "latency_seconds",
    "run_dir",
]


def _latest_test_metrics(history: Any) -> Dict[str, Any]:
    if not isinstance(history, list):
        return {}
    for record in reversed(history):
        if isinstance(record, dict) and isinstance(record.get("test"), dict):
            return record["test"]
    return {}


def build_accuracy_result_row(
    *,
    run_dir: Path,
    task_id: str,
    benchmark: str,
    setting: str,
    seed: int,
    dataset_format: str,
    split_protocol: str = "",
    leakage_warning: bool = False,
) -> Dict[str, Any]:
    history = read_json(run_dir / "history.json") or []
    test = _latest_test_metrics(history)
    cost = read_json(run_dir / "cost_summary.json") or {}
    return {
        "task_id": task_id,
        "benchmark": benchmark,
        "method_id": f"mad_{setting}",
        "setting": setting,
        "seed": int(seed),
        "dataset_format": dataset_format,
        "split_protocol": split_protocol,
        "leakage_warning": bool(leakage_warning),
        "num_test_samples": int(test.get("num_test_samples", test.get("size", 0)) or 0),
        "vote_acc": float(test.get("vote_acc", 0.0) or 0.0),
        "majority_vote_acc": float(test.get("majority_vote_acc", test.get("vote_acc", 0.0)) or 0.0),
        "weighted_vote_acc": float(test.get("weighted_vote_acc", 0.0) or 0.0),
        "mean_individual_acc": float(test.get("mean_individual_acc", 0.0) or 0.0),
        "best_individual_acc": float(test.get("best_individual_acc", 0.0) or 0.0),
        "oracle_acc": float(test.get("oracle_acc", 0.0) or 0.0),
        "aggregation_gap": float(test.get("aggregation_gap", 0.0) or 0.0),
        "rescue_available_rate": float(test.get("rescue_available_rate", 0.0) or 0.0),
        "correct_disagreement_rate": float(test.get("correct_disagreement_rate", 0.0) or 0.0),
        "mean_useful_diversity": float(test.get("mean_useful_diversity", 0.0) or 0.0),
        "aggregation_mode": str(test.get("aggregation_mode", "")),
        "solver_calls": int(cost.get("solver_calls", 0) or 0),
        "optimizer_calls": int(cost.get("optimizer_calls", 0) or 0),
        "evaluator_calls": int(cost.get("evaluator_calls", 0) or 0),
        "total_llm_calls": int(cost.get("total_llm_calls", 0) or 0),
        "prompt_tokens": int(cost.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(cost.get("completion_tokens", 0) or 0),
        "total_tokens": int(cost.get("total_tokens", 0) or 0),
        "estimated_cost": float(cost.get("estimated_cost", 0.0) or 0.0),
        "latency_seconds": float(cost.get("latency_seconds", 0.0) or 0.0),
        "run_dir": str(run_dir),
    }


def cost_summary_schema_keys():
    return {
        "solver_calls",
        "optimizer_calls",
        "evaluator_calls",
        "total_llm_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_cost",
        "latency_seconds",
    }
