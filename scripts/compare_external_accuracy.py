import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.experiment_io import read_csv_rows, read_jsonl, write_csv
except ModuleNotFoundError:
    from experiment_io import read_csv_rows, read_jsonl, write_csv


FAIRNESS_NOTE = (
    "MAD vote_acc uses multi-agent majority voting; MARS accuracy is single-prompt accuracy. "
    "Cost statistics are reported only and are not used as constraints."
)


OUTPUT_COLUMNS = [
    "task_id",
    "benchmark",
    "mars_method_id",
    "mars_accuracy",
    "mad_method_id",
    "mad_setting",
    "mad_seed",
    "split_protocol",
    "leakage_warning",
    "mad_vote_acc",
    "mad_mean_individual_acc",
    "mad_best_individual_acc",
    "delta_vote_acc_vs_mars",
    "delta_mean_individual_acc_vs_mars",
    "delta_best_individual_acc_vs_mars",
    "num_test_samples",
    "solver_calls",
    "optimizer_calls",
    "evaluator_calls",
    "total_llm_calls",
    "total_tokens",
    "estimated_cost",
    "latency_seconds",
    "fairness_note",
]


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _get_any(row: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    lowered = {str(k).lower().lstrip("\ufeff"): v for k, v in row.items()}
    for key in keys:
        value = lowered.get(str(key).lower())
        if value not in (None, ""):
            return value
    return default


def _read_mars_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    return read_csv_rows(path)


def _task_id(row: Dict[str, Any]) -> str:
    return str(_get_any(row, "task_id", "task", "task_name", "dataset", "subject")).strip()


def build_comparison_rows(mars_rows: List[Dict[str, Any]], mad_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mars_by_task: Dict[str, Dict[str, Any]] = {}
    for row in mars_rows:
        task_id = _task_id(row)
        if not task_id:
            continue
        method_id = str(_get_any(row, "method_id", "method", "model", "system", default="mars") or "mars")
        method_key = method_id.lower()
        if method_key not in {"mars", "mars_official"} and mars_by_task.get(task_id):
            continue
        mars_by_task[task_id] = row

    out = []
    for mad in mad_rows:
        task_id = _task_id(mad)
        if not task_id or task_id not in mars_by_task:
            continue
        mars = mars_by_task[task_id]
        mars_acc = _float(_get_any(mars, "accuracy", "acc", "vote_acc", "exact_match", default=0.0))
        vote_acc = _float(mad.get("vote_acc"))
        mean_acc = _float(mad.get("mean_individual_acc"))
        best_acc = _float(mad.get("best_individual_acc"))
        out.append(
            {
                "task_id": task_id,
                "benchmark": mad.get("benchmark", _get_any(mars, "benchmark", "group", default="")),
                "mars_method_id": _get_any(mars, "method_id", "method", "model", "system", default="mars"),
                "mars_accuracy": mars_acc,
                "mad_method_id": mad.get("method_id", ""),
                "mad_setting": mad.get("setting", ""),
                "mad_seed": mad.get("seed", ""),
                "split_protocol": mad.get("split_protocol", ""),
                "leakage_warning": mad.get("leakage_warning", ""),
                "mad_vote_acc": vote_acc,
                "mad_mean_individual_acc": mean_acc,
                "mad_best_individual_acc": best_acc,
                "delta_vote_acc_vs_mars": vote_acc - mars_acc,
                "delta_mean_individual_acc_vs_mars": mean_acc - mars_acc,
                "delta_best_individual_acc_vs_mars": best_acc - mars_acc,
                "num_test_samples": mad.get("num_test_samples", 0),
                "solver_calls": mad.get("solver_calls", 0),
                "optimizer_calls": mad.get("optimizer_calls", 0),
                "evaluator_calls": mad.get("evaluator_calls", 0),
                "total_llm_calls": mad.get("total_llm_calls", 0),
                "total_tokens": mad.get("total_tokens", 0),
                "estimated_cost": mad.get("estimated_cost", 0.0),
                "latency_seconds": mad.get("latency_seconds", 0.0),
                "fairness_note": FAIRNESS_NOTE,
            }
        )
    return out


def write_markdown(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# External Accuracy Comparison", ""]
    if not rows:
        lines.append("No joined rows. Check that both files share `task_id` values.")
    else:
        columns = [
            "task_id",
            "benchmark",
            "mars_accuracy",
            "mad_method_id",
            "mad_seed",
            "split_protocol",
            "leakage_warning",
            "mad_vote_acc",
            "delta_vote_acc_vs_mars",
            "total_llm_calls",
            "total_tokens",
        ]
        lines.extend(["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"])
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
        lines.extend(["", FAIRNESS_NOTE])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Join external MARS summary.csv with MAD task-level accuracy_results.jsonl.")
    parser.add_argument("--mars_summary", type=str, required=True)
    parser.add_argument("--mad_results", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="comparison/mars_vs_mad_accuracy.csv")
    parser.add_argument("--out_md", type=str, default="comparison/mars_vs_mad_accuracy.md")
    args = parser.parse_args()

    mars_path = Path(args.mars_summary)
    mad_path = Path(args.mad_results)
    mars_rows = _read_mars_rows(mars_path)
    mad_rows = read_jsonl(mad_path)
    rows = build_comparison_rows(mars_rows, mad_rows)
    write_csv(Path(args.out_csv), rows, fieldnames=OUTPUT_COLUMNS)
    write_markdown(rows, Path(args.out_md))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
