from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.config import Config, add_config_arguments, config_from_args
from multi_dataset_diverse_rl.evaluation.persistent_solver_cache import PersistentSolverCache
from multi_dataset_diverse_rl.persistence.artifacts import ArtifactWriter


def _cfg_for(base: Config, out_dir: Path, resume: bool) -> Config:
    values = base.to_flat_dict()
    values["out_dir"] = str(out_dir.resolve())
    values["resume_from_checkpoint"] = resume
    return Config.from_flat(**values)


def _command(cfg: Config) -> list[str]:
    command = [sys.executable, "-m", "multi_dataset_diverse_rl.cli"]
    for name, value in cfg.to_flat_dict().items():
        command.extend([f"--{name}", str(int(value) if isinstance(value, bool) else value)])
    return command


def _wait_for_checkpoint(
    process: subprocess.Popen,
    path: Path,
    *,
    expected_epoch: int,
    expected_update: int,
    timeout: float,
) -> dict:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"run exited before checkpoint with code {process.returncode}")
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                position = (int(payload["epoch_index"]), int(payload["update_index"]))
                if position == (expected_epoch, expected_update):
                    return payload
                if position > (expected_epoch, expected_update):
                    raise RuntimeError(
                        f"missed requested checkpoint {expected_epoch}/{expected_update}; saw {position}"
                    )
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                last_error = str(exc)
        time.sleep(0.1)
    raise TimeoutError(f"checkpoint wait timed out: {last_error}")


def _run_to_completion(cfg: Config, workspace: Path) -> None:
    subprocess.run(_command(cfg), cwd=workspace, check=True)


def _read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Controlled real-API checkpoint/resume equivalence smoke.")
    value.add_argument("--workspace", type=Path, default=Path("."))
    value.add_argument("--reference_out_dir", required=True)
    value.add_argument("--interrupted_out_dir", required=True)
    value.add_argument("--checkpoint_epoch", type=int, default=0)
    value.add_argument("--checkpoint_update", type=int, default=1)
    value.add_argument("--checkpoint_timeout", type=float, default=3600.0)
    return add_config_arguments(value)


def main() -> int:
    args = parser().parse_args()
    workspace = args.workspace.resolve()
    base = config_from_args(args)
    if not base.persistence.shared_solver_cache_path:
        raise ValueError("shared_solver_cache_path is required")
    reference_dir = Path(args.reference_out_dir).resolve()
    interrupted_dir = Path(args.interrupted_out_dir).resolve()
    for directory in (reference_dir, interrupted_dir):
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"resume smoke requires an empty output directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)

    interrupted_initial = _cfg_for(base, interrupted_dir, False)
    process = subprocess.Popen(_command(interrupted_initial), cwd=workspace)
    checkpoint_path = interrupted_dir / "training_checkpoint.json"
    checkpoint = _wait_for_checkpoint(
        process,
        checkpoint_path,
        expected_epoch=args.checkpoint_epoch,
        expected_update=args.checkpoint_update,
        timeout=args.checkpoint_timeout,
    )
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)

    reference_checkpoint = reference_dir / "training_checkpoint.json"
    shutil.copy2(checkpoint_path, reference_checkpoint)
    cache = PersistentSolverCache(base.persistence.shared_solver_cache_path)
    cache_before = {
        "count": cache.ready_entry_count(),
        "hash": cache.ready_content_hash(),
    }

    _run_to_completion(_cfg_for(base, reference_dir, True), workspace)
    cache_after_reference = {
        "count": cache.ready_entry_count(),
        "hash": cache.ready_content_hash(),
    }
    _run_to_completion(_cfg_for(base, interrupted_dir, True), workspace)
    cache_after_resume = {
        "count": cache.ready_entry_count(),
        "hash": cache.ready_content_hash(),
    }

    compared = {}
    for filename in (
        "best_prompts.json",
        "candidate_decisions.jsonl",
        "responsibility_assignments.jsonl",
        "history.json",
        "final_summary.json",
    ):
        left = reference_dir / filename
        right = interrupted_dir / filename
        if filename.endswith(".jsonl"):
            equal = left.read_text(encoding="utf-8") == right.read_text(encoding="utf-8")
        else:
            equal = _read(left) == _read(right)
        compared[filename] = equal
    reference_identity = _read(reference_dir / "run_meta.json")["run_identity"]
    resumed_identity = _read(interrupted_dir / "run_meta.json")["run_identity"]
    report = {
        "ok": bool(
            all(compared.values())
            and reference_identity == resumed_identity
            and cache_after_reference == cache_after_resume
        ),
        "checkpoint_position": {
            "epoch_index": checkpoint["epoch_index"],
            "update_index": checkpoint["update_index"],
        },
        "run_identity_equal": reference_identity == resumed_identity,
        "artifact_comparison": compared,
        "cache_before": cache_before,
        "cache_after_reference": cache_after_reference,
        "cache_after_resume": cache_after_resume,
    }
    report_path = interrupted_dir.parent / "resume_smoke_report.json"
    ArtifactWriter(report_path.parent).write_json(report_path.name, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
