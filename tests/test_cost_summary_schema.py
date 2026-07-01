import json

from scripts.task_level_accuracy_utils import cost_summary_schema_keys


def test_cost_summary_schema_keys(tmp_path):
    path = tmp_path / "cost_summary.json"
    payload = {
        "solver_calls": 0,
        "optimizer_calls": 0,
        "evaluator_calls": 0,
        "total_llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost": 0.0,
        "latency_seconds": 0.0,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert set(loaded).issuperset(cost_summary_schema_keys())
