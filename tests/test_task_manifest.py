from multi_dataset_diverse_rl.task_manifest import load_task_manifest, resolve_task_ids


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
