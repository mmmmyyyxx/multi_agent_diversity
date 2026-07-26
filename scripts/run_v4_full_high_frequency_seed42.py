from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.cli import run
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.persistence.identity import solver_request_identity
from multi_dataset_diverse_rl.task_manifest import load_task_manifest
from multi_dataset_diverse_rl.utils import normalize_prompt_text


SOURCE_BASELINE = (
    "runs_v4_baseline_full_seed42_20260726_114912/disambiguation_qa/"
    "shared_baseline_seed42"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_baseline(workspace: Path, cfg: Config, task) -> dict:
    baseline_dir = workspace / SOURCE_BASELINE
    meta_path = baseline_dir / "run_meta.json"
    summary_path = baseline_dir / "final_summary.json"
    if not meta_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("frozen seed42 Baseline reference is unavailable")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    baseline_cfg = meta["config"]
    current_paths = {
        "train_path": workspace / task.train_path,
        "val_path": workspace / task.val_path,
        "test_path": workspace / task.test_path,
    }
    checks = {
        "task": baseline_cfg["comparison_task_id"] == task.task_id,
        "seed": int(baseline_cfg["seed"]) == 42,
        "agent_count": int(baseline_cfg["agents"]) == cfg.training.agents == 5,
        "agent_model": baseline_cfg["agent_model"] == cfg.models.agent_model,
        "solver_max_tokens": int(baseline_cfg["solver_max_tokens"]) == cfg.models.solver_max_tokens,
        "solver_invalid_retries": int(baseline_cfg["solver_invalid_max_retries"]) == cfg.models.solver_invalid_max_retries,
        "temperature": float(baseline_cfg["temperature"]) == cfg.models.temperature,
        "parser": baseline_cfg["parser_version"] == cfg.peer_state.parser_version,
        "output_contract": baseline_cfg["solver_output_contract_version"] == cfg.peer_state.solver_output_contract_version,
        "plurality": baseline_cfg["aggregation_mode"] == cfg.peer_state.aggregation_mode,
        "tie": baseline_cfg["vote_tie_break"] == cfg.peer_state.vote_tie_break,
        "initial_prompt_hash": meta["initial_prompt_hashes"] == [
            hashlib.sha256(
                normalize_prompt_text(cfg.training.shared_prompt).encode("utf-8")
            ).hexdigest()
        ] * 5,
        "initial_team_hash": len(set(meta["initial_prompt_hashes"])) == 1,
        "train_split": meta["run_identity"]["train_file_sha256"] == _sha256(current_paths["train_path"]),
        "val_split": meta["run_identity"]["val_file_sha256"] == _sha256(current_paths["val_path"]),
        "test_split": meta["run_identity"]["test_file_sha256"] == _sha256(current_paths["test_path"]),
        "solver_request_contract": meta["prompt_question_evaluator_identity"][1] == solver_request_identity(cfg),
        "frozen_baseline_vote": summary["selected_test"]["vote_correct_count"] == 52,
        "frozen_baseline_members": summary["selected_test"]["per_agent_correct_counts"] == [52] * 5,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("frozen Baseline identity mismatch: " + ", ".join(failed))
    return {"baseline_dir": str(baseline_dir), "checks": checks}


async def main_async(args: argparse.Namespace) -> None:
    workspace = args.workspace.resolve()
    root = (workspace / args.out_root).resolve()
    if root.exists() and not args.preflight_only:
        raise FileExistsError("high-frequency pilot requires a fresh out_root")
    manifest_path = workspace / "configs/task_level_comparison_strict_bbh_seed42.yaml"
    task = load_task_manifest(str(manifest_path))["disambiguation_qa"]
    out_dir = root / task.task_id / "shared_member_aware_full_seed42"
    cfg = Config.from_flat(
        task_type=task.task_type,
        dataset_format="mars",
        comparison_task_id=task.task_id,
        benchmark=task.benchmark,
        answer_format=task.answer_format,
        train_path=str((workspace / task.train_path).resolve()),
        val_path=str((workspace / task.val_path).resolve()),
        test_path=str((workspace / task.test_path).resolve()),
        manifest_sha256=_sha256(manifest_path),
        train_size=75,
        val_size=50,
        test_size=125,
        agent_model="gpt-4o-mini",
        optimizer_model="gpt-4o-mini",
        evaluator_model="gpt-4o-mini",
        method_version="member_aware_peer_state_v4",
        experiment_setting="shared_member_aware_full",
        agents=5,
        initialization_mode="shared_identical",
        seed=42,
        epochs=8,
        update_every=25,
        candidate_eval_pool_size=75,
        num_candidates_per_parent=2,
        stage_a_representative_size=12,
        stage_a_coverage_size=6,
        stage_a_conversion_size=6,
        stage_a_preservation_size=4,
        stage_a_channel_top_k=2,
        stage_b_candidate_budget=2,
        solver_max_tokens=1800,
        solver_invalid_max_retries=3,
        student_invalid_max_retries=3,
        student_upstream_regeneration_max_count=1,
        eval_solver_call_concurrency=8,
        out_dir=str(out_dir),
        shared_solver_cache_path=str(root / "_shared_solver_cache.sqlite"),
        resume_from_checkpoint=False,
    )
    planned = cfg.training.epochs * math.ceil(cfg.data.train_size / cfg.training.update_every)
    if planned != 24:
        raise AssertionError(f"planned_update_count must be 24, got {planned}")
    baseline_audit = _verify_baseline(workspace, cfg, task)
    if args.preflight_only:
        print(json.dumps({
            "ok": True,
            "planned_update_count": planned,
            "baseline_identity": baseline_audit,
            "estimated_solver_calls": {"lower": 4786, "upper": 7180},
            "estimated_role_calls": {"lower": 60, "upper": 288},
            "estimated_total_tokens": {"lower": 1703422, "upper": 2555133},
            "estimated_maximum_student_recovery_calls": 192,
        }, ensure_ascii=False, indent=2))
        return
    root.mkdir(parents=True, exist_ok=False)
    (root / "baseline_identity_audit.json").write_text(
        json.dumps(baseline_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    await run(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--preflight_only", action="store_true")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
