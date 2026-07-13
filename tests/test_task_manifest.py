import pytest

from multi_dataset_diverse_rl.task_manifest import ComparisonTask, load_task_manifest, resolve_task_ids
from scripts.run_task_level_accuracy import _task_split_integrity, _task_split_protocol


def test_load_task_level_manifest():
    tasks = load_task_manifest("configs/task_level_comparison.yaml")
    assert "boolean_expressions" in tasks
    assert tasks["boolean_expressions"].benchmark == "BBH"
    assert tasks["marketing"].task_type == "mmlu"
    assert sum(1 for task in tasks.values() if task.benchmark == "BBH") == 6
    assert sum(1 for task in tasks.values() if task.benchmark == "MMLU") == 6


def test_resolve_tasks_by_ids_and_benchmark():
    tasks = load_task_manifest("configs/task_level_comparison.yaml")
    assert resolve_task_ids("boolean_expressions,marketing", tasks) == ["boolean_expressions", "marketing"]
    bbh_tasks = resolve_task_ids("all", tasks, benchmarks="BBH")
    assert "boolean_expressions" in bbh_tasks
    assert "marketing" not in bbh_tasks


def test_strict_bbh_manifest_uses_distinct_task_splits():
    tasks = load_task_manifest("configs/task_level_comparison_strict_bbh_seed42.yaml")
    assert set(tasks) == {
        "boolean_expressions", "formal_fallacies", "disambiguation_qa",
        "geometric_shapes", "ruin_names", "sports_understanding",
    }
    for task in tasks.values():
        assert _task_split_protocol(task) == {
            "split_protocol": "task_manifest_split",
            "leakage_warning": False,
        }
        assert task.train_path.endswith("/opt.csv")
        assert task.val_path.endswith("/val.csv")
        assert task.test_path.endswith("/test.csv")
        integrity = _task_split_integrity(task, "mars", ".")
        assert integrity["opt_val_question_overlap"] == 0
        assert integrity["opt_test_question_overlap"] == 0
        assert integrity["val_test_question_overlap"] == 0


def test_strict_split_integrity_records_hashes_and_rejects_question_overlap(tmp_path):
    train = tmp_path / "opt.csv"
    val = tmp_path / "val.csv"
    test = tmp_path / "test.csv"
    train.write_text("question,answer\nTrain question,A\n", encoding="utf-8")
    val.write_text("question,answer\nValidation question,B\n", encoding="utf-8")
    test.write_text("question,answer\nTest question,C\n", encoding="utf-8")
    task = ComparisonTask("toy", "BBH", "bbh", "option_letter", str(train), str(val), str(test))

    integrity = _task_split_integrity(task, "mars", str(tmp_path))

    assert integrity["opt_val_question_overlap"] == 0
    assert integrity["opt_test_question_overlap"] == 0
    assert integrity["val_test_question_overlap"] == 0
    assert len(integrity["opt_file_sha256"]) == 64

    val.write_text("question,answer\nTrain   question,A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Strict split overlap"):
        _task_split_integrity(task, "mars", str(tmp_path))
