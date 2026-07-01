from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass(frozen=True)
class ComparisonTask:
    task_id: str
    benchmark: str
    task_type: str
    answer_format: str
    train_path: str
    val_path: str
    test_path: str


def load_task_manifest(path: str) -> Dict[str, ComparisonTask]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    tasks: Dict[str, ComparisonTask] = {}
    for task_id, cfg in (data.get("tasks", {}) or {}).items():
        cfg = cfg or {}
        tasks[str(task_id)] = ComparisonTask(
            task_id=str(task_id),
            benchmark=str(cfg.get("benchmark", "")),
            task_type=str(cfg.get("task_type", "auto")),
            answer_format=str(cfg.get("answer_format", "")),
            train_path=str(cfg.get("train_path", "")),
            val_path=str(cfg.get("val_path", "")),
            test_path=str(cfg.get("test_path", "")),
        )
    return tasks


def resolve_task_ids(raw: str, tasks: Dict[str, ComparisonTask], benchmarks: str = "") -> List[str]:
    if raw and raw != "all":
        requested = [x.strip() for x in str(raw).split(",") if x.strip()]
        missing = [task_id for task_id in requested if task_id not in tasks]
        if missing:
            raise ValueError(f"Unknown task ids: {missing}")
        return requested
    if benchmarks:
        wanted = {x.strip().lower() for x in str(benchmarks).split(",") if x.strip()}
        return [tid for tid, spec in tasks.items() if spec.benchmark.lower() in wanted]
    return list(tasks.keys())
