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
    "per_agent_acc",
    "min_individual_acc",
    "bottom2_mean_acc",
    "bottom3_mean_acc",
    "individual_acc_std",
    "best_minus_worst_gap",
    "best_minus_bottom2_gap",
    "oracle_acc",
    "aggregation_gap",
    "rescue_available_rate",
    "correct_disagreement_rate",
    "mean_useful_diversity",
    "mean_vote_margin",
    "mean_boundary_useful_diversity",
    "vote_tie_rate",
    "mean_pairwise_double_fault",
    "mean_pairwise_error_covariance",
    "same_wrong_pair_rate",
    "triple_joint_error_rate",
    "majority_failure_tail_rate",
    "coverage_depth_c1",
    "coverage_depth_c2",
    "coverage_depth_c3",
    "coverage_depth_c4",
    "coverage_depth_c5",
    "c1_minus_c2",
    "c2_minus_c3",
    "max_minority_rescue_share",
    "minority_rescue_hhi",
    "specialization_strength_final",
    "mean_specialization_strength",
    "prompt_overlength_rejection_count",
    "truncated_prompt_count",
    "correct_agent_count_0",
    "correct_agent_count_1",
    "correct_agent_count_2",
    "correct_agent_count_3",
    "correct_agent_count_4",
    "correct_agent_count_5",
    "mean_boundary_conditional_error",
    "mean_pivotal_fix_rate",
    "mean_pivotal_hold_rate",
    "shared_error_rescue_rate",
    "shared_error_creation_rate",
    "boundary_shared_error_net_gain",
    "dominant_wrong_cluster_size",
    "gold_vs_largest_wrong_margin",
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
        "per_agent_acc": test.get("per_agent_acc", []),
        "oracle_acc": float(test.get("oracle_acc", 0.0) or 0.0),
        "aggregation_gap": float(test.get("aggregation_gap", 0.0) or 0.0),
        "rescue_available_rate": float(test.get("rescue_available_rate", 0.0) or 0.0),
        "correct_disagreement_rate": float(test.get("correct_disagreement_rate", 0.0) or 0.0),
        "mean_useful_diversity": float(test.get("mean_useful_diversity", 0.0) or 0.0),
        "mean_vote_margin": float(test.get("mean_vote_margin", -1.0) if test.get("mean_vote_margin") is not None else -1.0),
        "mean_boundary_useful_diversity": float(test.get("mean_boundary_useful_diversity", 0.0) or 0.0),
        "vote_tie_rate": float(test.get("vote_tie_rate", 0.0) or 0.0),
        **{
            key: float(test.get(key, 0.0) or 0.0)
            for key in (
                "mean_pairwise_double_fault", "mean_pairwise_error_covariance", "same_wrong_pair_rate",
                "triple_joint_error_rate", "majority_failure_tail_rate", "coverage_depth_c1",
                "coverage_depth_c2", "coverage_depth_c3", "coverage_depth_c4", "coverage_depth_c5",
                "mean_boundary_conditional_error", "mean_pivotal_fix_rate", "mean_pivotal_hold_rate",
                "shared_error_rescue_rate", "shared_error_creation_rate", "boundary_shared_error_net_gain",
                "dominant_wrong_cluster_size", "gold_vs_largest_wrong_margin",
                "min_individual_acc", "bottom2_mean_acc", "bottom3_mean_acc", "individual_acc_std",
                "best_minus_worst_gap", "best_minus_bottom2_gap", "c1_minus_c2", "c2_minus_c3",
                "max_minority_rescue_share", "minority_rescue_hhi", "specialization_strength_final",
                "mean_specialization_strength",
            )
        },
        **{key: int(test.get(key, 0) or 0) for key in (
            "prompt_overlength_rejection_count", "truncated_prompt_count",
            "correct_agent_count_0", "correct_agent_count_1", "correct_agent_count_2",
            "correct_agent_count_3", "correct_agent_count_4", "correct_agent_count_5",
        )},
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
