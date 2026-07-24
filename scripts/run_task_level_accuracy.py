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
from multi_dataset_diverse_rl.cli import build_dataset
from multi_dataset_diverse_rl.evaluation.output_contract import SOLVER_OUTPUT_CONTRACT_VERSION
from multi_dataset_diverse_rl.persistence.identity import build_run_identity, validate_run_identity
from multi_dataset_diverse_rl.task_manifest import load_task_manifest, resolve_task_ids
from multi_dataset_diverse_rl.utils import load_jsonl
from scripts.experiment_config import select_settings


RUNNER_OWNED_FIELDS = {
    "task_type", "dataset_format", "comparison_task_id", "benchmark", "answer_format",
    "train_path", "val_path", "test_path", "manifest_sha256", "out_dir", "seed",
    "method_version", "experiment_setting",
}
RUNNER_FIELDS = tuple(
    name for name in Config().to_flat_dict() if name not in RUNNER_OWNED_FIELDS
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
    parser = argparse.ArgumentParser(
        description="Run task-level member-aware peer-state experiments."
    )
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


def _completed_run(run_dir: Path, expected_identity) -> bool:
    required = (
        "final_summary.json",
        "history.json",
        "best_prompts.json",
        "run_meta.json",
        "tcs_rounds.jsonl",
        "solver_invalid_outputs.jsonl",
        "cost_summary.json",
    )
    if not all((run_dir / filename).exists() for filename in required):
        return False
    try:
        metadata = _read_json(run_dir / "run_meta.json")
        summary = _read_json(run_dir / "final_summary.json")
    except (OSError, json.JSONDecodeError):
        return False
    if metadata["method_version"] != "member_aware_peer_state_v2":
        raise ValueError(f"Completed run has an incompatible method version: {run_dir}")
    if metadata["legacy_compatibility_enabled"] is not False:
        raise ValueError(f"Completed run enabled legacy compatibility: {run_dir}")
    if metadata.get("solver_output_contract_version") != SOLVER_OUTPUT_CONTRACT_VERSION:
        raise ValueError(f"Completed run has an incompatible solver output contract: {run_dir}")
    if not metadata.get("shared_solver_cache_path"):
        raise ValueError(f"Completed run has no persistent shared solver cache: {run_dir}")
    validate_run_identity(expected_identity, metadata["run_identity"])
    if not {"initial_test", "selected_test", "member_gain", "selection_summary"} <= set(summary):
        raise ValueError(f"Completed run has an incompatible final summary: {run_dir}")
    return True


def main() -> None:
    args = _parser().parse_args()
    workspace = args.workspace.resolve()
    tasks = load_task_manifest(str((workspace / args.manifest).resolve()))
    manifest_sha256 = hashlib.sha256((workspace / args.manifest).resolve().read_bytes()).hexdigest()
    task_ids = resolve_task_ids(args.tasks, tasks, args.benchmarks)
    settings = select_settings(args.settings)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    root = (workspace / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for task_id in task_ids:
        task = tasks[task_id]
        split_integrity = _task_split_integrity(task, args.dataset_format, str(workspace))
        for setting in settings:
            for seed in seeds:
                run_dir = root / task_id / f"{setting.name}_seed{seed}"
                final_path = run_dir / "final_summary.json"
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
                    "manifest_sha256": manifest_sha256,
                    "out_dir": str(run_dir),
                    "shared_solver_cache_path": str(root / "_shared_solver_cache.sqlite"),
                    "seed": seed,
                }
                defaults = Config().to_flat_dict()
                for name in RUNNER_FIELDS:
                    value = getattr(args, name)
                    if value is not None:
                        values[name] = bool(value) if isinstance(defaults[name], bool) else value
                cfg = Config.from_flat(**values)
                split_rows = {
                    "train": build_dataset(load_jsonl(cfg.data.train_path, cfg.data.train_size), cfg.data.dataset_format),
                    "val": build_dataset(load_jsonl(cfg.data.val_path, cfg.data.val_size), cfg.data.dataset_format),
                    "test": build_dataset(load_jsonl(cfg.data.test_path, cfg.data.test_size), cfg.data.dataset_format),
                }
                expected_identity = build_run_identity(
                    cfg,
                    train_rows=split_rows["train"],
                    val_rows=split_rows["val"],
                    test_rows=split_rows["test"],
                    workspace=workspace,
                )
                if args.resume_completed and _completed_run(run_dir, expected_identity):
                    metrics = _read_json(final_path)
                else:
                    cmd = [sys.executable, "-m", "multi_dataset_diverse_rl.cli"]
                    for name, value in cfg.to_flat_dict().items():
                        cmd.extend([f"--{name}", str(int(value) if isinstance(value, bool) else value)])
                    subprocess.run(cmd, cwd=workspace, check=True)
                    metrics = _read_json(final_path)
                rows.append({
                    "task_id": task_id, "benchmark": task.benchmark, "setting": setting.name, "seed": seed,
                    "vote_acc_initial": metrics["initial_test"]["plurality_vote_acc"],
                    "vote_acc_selected": metrics["selected_test"]["plurality_vote_acc"],
                    "vote_gain": (
                        metrics["selected_test"]["plurality_vote_acc"]
                        - metrics["initial_test"]["plurality_vote_acc"]
                    ),
                    "minimum_member_correct_count_gain": metrics["member_gain"][
                        "minimum_member_correct_count_gain"
                    ],
                    "mean_member_correct_count_gain": metrics["member_gain"][
                        "mean_member_correct_count_gain"
                    ],
                    "minimum_member_accuracy_gain": metrics["member_gain"][
                        "minimum_member_accuracy_gain"
                    ],
                    "mean_member_accuracy_gain": metrics["member_gain"][
                        "mean_member_accuracy_gain"
                    ],
                    "improved_agent_count": metrics["member_gain"]["improved_agent_count"],
                    "regressed_agent_count": metrics["member_gain"]["regressed_agent_count"],
                    "all_members_improved": metrics["member_gain"]["all_members_improved"],
                    "selected_mean_individual_acc": metrics["selected_test"]["mean_individual_acc"],
                    "selected_min_individual_acc": metrics["selected_test"]["min_individual_acc"],
                    "selected_mean_soft_vote_utility": metrics["selected_test"]["mean_soft_vote_utility"],
                    "selected_mean_invalid_rate": metrics["selected_test"]["mean_invalid_rate"],
                    "selected_tie_rate": metrics["selected_test"]["tie_rate"],
                    "run_identity": expected_identity.to_dict(),
                    **split_integrity,
                })
                (root / "accuracy_results.jsonl").write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
                )
                (root / "experiment_runs.jsonl").write_text(
                    "".join(json.dumps({
                        "task_id": row["task_id"],
                        "setting": row["setting"],
                        "seed": row["seed"],
                        "run_identity": row["run_identity"],
                    }, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8",
                )
    columns = list(rows[0]) if rows else [
        "task_id", "benchmark", "setting", "seed", "vote_acc_selected"
    ]
    with (root / "accuracy_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
