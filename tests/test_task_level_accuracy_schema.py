import json

from scripts.compare_external_accuracy import build_comparison_rows
from scripts.task_level_accuracy_utils import ACCURACY_RESULT_COLUMNS, build_accuracy_result_row


def test_accuracy_results_schema(tmp_path):
    run_dir = tmp_path / "boolean_expressions" / "shared_guarded_beam_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "history.json").write_text(
        json.dumps(
            [
                {
                    "test": {
                        "num_test_samples": 2,
                        "vote_acc": 0.5,
                        "mean_individual_acc": 0.4,
                        "best_individual_acc": 0.6,
                    }
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "cost_summary.json").write_text(
        json.dumps({"solver_calls": 10, "optimizer_calls": 1, "evaluator_calls": 2, "total_llm_calls": 13, "total_tokens": 123}),
        encoding="utf-8",
    )
    row = build_accuracy_result_row(
        run_dir=run_dir,
        task_id="boolean_expressions",
        benchmark="BBH",
        setting="shared_guarded_beam",
        seed=42,
        dataset_format="mars",
    )
    for key in ["task_id", "benchmark", "method_id", "vote_acc", "mean_individual_acc", "best_individual_acc", "num_test_samples"]:
        assert key in row
    assert row["method_id"] == "mad_shared_guarded_beam"
    assert set(row).issuperset(set(ACCURACY_RESULT_COLUMNS))


def test_external_comparison_rows_join_on_task_id():
    rows = build_comparison_rows(
        [{"task_id": "boolean_expressions", "method_id": "mars", "accuracy": 0.4}],
        [
            {
                "task_id": "boolean_expressions",
                "benchmark": "BBH",
                "method_id": "mad_shared_guarded_beam",
                "setting": "shared_guarded_beam",
                "seed": 42,
                "vote_acc": 0.5,
                "mean_individual_acc": 0.45,
                "best_individual_acc": 0.55,
                "total_tokens": 123,
            }
        ],
    )
    assert len(rows) == 1
    assert round(rows[0]["delta_vote_acc_vs_mars"], 6) == 0.1
    assert "Cost statistics are reported only" in rows[0]["fairness_note"]
