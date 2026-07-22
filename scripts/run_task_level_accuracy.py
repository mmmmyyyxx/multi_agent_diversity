from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.task_manifest import load_task_manifest, resolve_task_ids
from scripts.experiment_config import select_settings


RUNNER_FIELDS = (
    "agent_model", "optimizer_model", "evaluator_model", "agents", "epochs", "update_every",
    "train_size", "val_size", "test_size", "num_candidates_per_parent",
    "candidate_eval_pool_size", "eval_solver_call_concurrency",
    "generation_parent_limit", "stage_a_representative_size",
    "stage_a_coverage_size", "stage_a_conversion_size", "stage_a_preservation_size",
    "stage_a_channel_top_k", "stage_b_candidate_budget", "max_retries", "max_transient_retries",
    "retry_sleep", "max_retry_backoff", "llm_call_timeout", "resume_from_checkpoint", "resume_completed",
)


def _task_split_protocol(task) -> dict[str, Any]:
    paths = {str(task.train_path), str(task.val_path), str(task.test_path)}
    if len(paths) < 3:
        return {"split_protocol": "reused_file", "leakage_warning": True}
    return {"split_protocol": "task_manifest_split", "leakage_warning": False}


def _task_split_integrity(task, dataset_format: str, workspace: str) -> dict[str, Any]:
    def resolve(path: str) -> Path:
        value = Path(path)
        return value if value.is_absolute() else Path(workspace) / value

    paths = {"opt": resolve(task.train_path), "val": resolve(task.val_path), "test": resolve(task.test_path)}
    rows = {
        name: __import__("multi_dataset_diverse_rl.cli", fromlist=["build_dataset"]).build_dataset(
            __import__("multi_dataset_diverse_rl.utils", fromlist=["load_jsonl"]).load_jsonl(str(path), -1),
            dataset_format,
        )
        for name, path in paths.items()
    }

    def question_hash(value: Any) -> str:
        return hashlib.sha256(" ".join(str(value or "").split()).lower().encode("utf-8")).hexdigest()

    hashes = {name: {question_hash(row["question"]) for row in values} for name, values in rows.items()}
    overlaps = {
        "opt_val_question_overlap": len(hashes["opt"] & hashes["val"]),
        "opt_test_question_overlap": len(hashes["opt"] & hashes["test"]),
        "val_test_question_overlap": len(hashes["val"] & hashes["test"]),
    }
    protocol = _task_split_protocol(task)
    if protocol["split_protocol"] == "task_manifest_split" and any(overlaps.values()):
        raise ValueError(
            f"Strict split overlap for task={task.task_id}: "
            f"opt_val={overlaps['opt_val_question_overlap']} "
            f"opt_test={overlaps['opt_test_question_overlap']} "
            f"val_test={overlaps['val_test_question_overlap']}"
        )
    return {
        **protocol, **overlaps,
        **{f"{name}_count": len(rows[name]) for name in paths},
        **{f"{name}_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run task-level peer-state experiments.")
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--benchmarks", default="")
    parser.add_argument("--settings", default="all")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--dataset_format", default="mars")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--resume_completed", type=int, choices=[0, 1], default=0)
    defaults = Config().to_flat_dict()
    for name in RUNNER_FIELDS:
        default = defaults[name]
        arg_type = int if isinstance(default, bool) else type(default)
        parser.add_argument(f"--{name}", type=arg_type, default=None)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _completed_run(run_dir: Path) -> bool:
    required = ("final_summary.json", "history.json", "best_prompts.json", "run_meta.json")
    if not all((run_dir / filename).exists() for filename in required):
        return False
    try:
        metadata = _read_json(run_dir / "run_meta.json")
        summary = _read_json(run_dir / "final_summary.json")
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        metadata.get("method_version") == "peer_state_counterfactual_v1"
        and metadata.get("legacy_compatibility_enabled") is False
        and "plurality_vote_acc" in summary
    )


def main() -> None:
    args = _parser().parse_args()
    workspace = args.workspace.resolve()
    tasks = load_task_manifest(str((workspace / args.manifest).resolve()))
    task_ids = resolve_task_ids(args.tasks, tasks, args.benchmarks)
    settings = select_settings(args.settings)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    root = (workspace / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for task_id in task_ids:
        task = tasks[task_id]
        _task_split_integrity(task, args.dataset_format, str(workspace))
        for setting in settings:
            for seed in seeds:
                run_dir = root / task_id / f"{setting.name}_seed{seed}"
                final_path = run_dir / "final_summary.json"
                if args.resume_completed and _completed_run(run_dir):
                    metrics = _read_json(final_path)
                else:
                    values = {
                        **setting.resolved_overrides(),
                        "task_type": task.task_type,
                        "dataset_format": args.dataset_format,
                        "comparison_task_id": task.task_id,
                        "benchmark": task.benchmark,
                        "answer_format": task.answer_format,
                        "train_path": str((workspace / task.train_path).resolve()),
                        "val_path": str((workspace / task.val_path).resolve()),
                        "test_path": str((workspace / task.test_path).resolve()),
                        "out_dir": str(run_dir),
                        "seed": seed,
                        "experiment_setting": setting.name,
                    }
                    for name in RUNNER_FIELDS:
                        value = getattr(args, name)
                        if value is not None:
                            values[name] = bool(value) if isinstance(Config().to_flat_dict()[name], bool) else value
                    cfg = Config.from_flat(**values)
                    cmd = [sys.executable, "-m", "multi_dataset_diverse_rl.cli"]
                    for name, value in cfg.to_flat_dict().items():
                        cmd.extend([f"--{name}", str(int(value) if isinstance(value, bool) else value)])
                    subprocess.run(cmd, cwd=workspace, check=True)
                    metrics = _read_json(final_path)
                rows.append({
                    "task_id": task_id, "benchmark": task.benchmark, "setting": setting.name, "seed": seed,
                    "vote_acc": metrics.get("plurality_vote_acc", metrics.get("vote_acc", 0.0)),
                    "mean_individual_acc": metrics.get("mean_individual_acc", 0.0),
                    "min_individual_acc": metrics.get("min_individual_acc", 0.0),
                    "mean_soft_vote_utility": metrics.get("mean_soft_vote_utility", 0.0),
                    "mean_invalid_rate": metrics.get("mean_invalid_rate", 0.0),
                })
                (root / "accuracy_results.jsonl").write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
                )
    columns = list(rows[0]) if rows else ["task_id", "benchmark", "setting", "seed", "vote_acc"]
    with (root / "accuracy_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
