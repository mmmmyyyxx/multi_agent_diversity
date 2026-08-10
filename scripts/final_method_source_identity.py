from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.versions import (
    CANDIDATE_ACCEPTANCE_VERSION,
    CANDIDATE_SELECTION_VERSION,
    CANDIDATE_PROTOCOL_FILTER_VERSION,
    CHECKPOINT_VERSION,
    COMMON_UPDATE_POLICY_VERSION,
    DUAL_TARGET_SEARCH_VERSION,
    EXPERIMENT_MATRIX_VERSION,
    METHOD_VERSION,
    MUTABLE_PROMPT_CONTRACT_VERSION,
    PROTOCOL_RESOLUTION_VERSION,
    REPAIRABILITY_VERSION,
    RCRU_VERSION,
    ROBUST_SUPPORT_VERSION,
    RESPONSIBILITY_VERSION,
    SERVICE_ROUTING_VERSION,
    TARGET_SELECTION_VERSION,
    TCS_CONTEXT_VERSION,
    STUDENT_PROMPT_CONTRACT_VERSION,
)
from multi_dataset_diverse_rl.evaluation.output_contract import (
    SOLVER_REQUEST_TEMPLATE_VERSION,
)
from multi_dataset_diverse_rl.evaluation.persistent_solver_cache import (
    SCHEMA_VERSION as SOLVER_CACHE_SCHEMA_VERSION,
)
from multi_dataset_diverse_rl.persistence.identity import (
    PROMPT_QUESTION_EVALUATOR_VERSION,
)


SOURCE_IDENTITY_VERSION = "final_method_source_identity_v5"
SOURCE_ROOTS = (
    "AGENTS.md",
    "README.md",
    "method.md",
    "configs",
    "multi_dataset_diverse_rl",
    "scripts",
    "tests",
)
EXPERIMENT_SCRIPTS = (
    "scripts/run_task_level_accuracy.py",
    "scripts/audit_final_method_stage.py",
    "scripts/build_final_method_complete_report.py",
    "scripts/build_strict_v2_s345_report.py",
    "scripts/run_v13_search_budget_control.py",
    "scripts/audit_v13_search_budget_control.py",
)


def _git(workspace: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _source_files(workspace: Path) -> list[Path]:
    files: list[Path] = []
    for raw in SOURCE_ROOTS:
        path = workspace / raw
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and "__pycache__" not in candidate.parts
                and candidate.suffix.lower() in {".py", ".ps1", ".yaml", ".yml", ".md"}
            )
    return sorted(set(files), key=lambda path: path.relative_to(workspace).as_posix())


def _tree_hash(workspace: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(workspace).as_posix()
        payload = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def build_source_identity(workspace: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    files = _source_files(workspace)
    diff = _git(workspace, "diff", "--binary", "HEAD")
    status = _git(workspace, "status", "--porcelain=v1").decode("utf-8").splitlines()
    script_hashes = {
        name: hashlib.sha256((workspace / name).read_bytes()).hexdigest()
        if (workspace / name).is_file() else "missing"
        for name in EXPERIMENT_SCRIPTS
    }
    return {
        "source_identity_version": SOURCE_IDENTITY_VERSION,
        "git_commit": _git(workspace, "rev-parse", "HEAD").decode("utf-8").strip(),
        "git_dirty": bool(status),
        "git_diff_hash": hashlib.sha256(diff).hexdigest(),
        "source_tree_hash": _tree_hash(workspace, files),
        "source_file_count": len(files),
        "experiment_script_hashes": script_hashes,
        "method_identifiers": {
            "method_version": METHOD_VERSION,
            "responsibility_version": RESPONSIBILITY_VERSION,
            "service_routing_version": SERVICE_ROUTING_VERSION,
            "target_selection_version": TARGET_SELECTION_VERSION,
            "repairability_version": REPAIRABILITY_VERSION,
            "dual_target_search_version": DUAL_TARGET_SEARCH_VERSION,
            "tcs_context_version": TCS_CONTEXT_VERSION,
            "candidate_acceptance_version": CANDIDATE_ACCEPTANCE_VERSION,
            "candidate_selection_version": CANDIDATE_SELECTION_VERSION,
            "experiment_matrix_version": EXPERIMENT_MATRIX_VERSION,
            "protocol_resolution_version": PROTOCOL_RESOLUTION_VERSION,
            "common_update_policy_version": COMMON_UPDATE_POLICY_VERSION,
            "rcru_version": RCRU_VERSION,
            "robust_support_version": ROBUST_SUPPORT_VERSION,
            "mutable_prompt_contract_version": MUTABLE_PROMPT_CONTRACT_VERSION,
            "student_prompt_contract_version": STUDENT_PROMPT_CONTRACT_VERSION,
            "candidate_protocol_filter_version": CANDIDATE_PROTOCOL_FILTER_VERSION,
            "checkpoint_version": CHECKPOINT_VERSION,
            "exact_request_identity_version": PROMPT_QUESTION_EVALUATOR_VERSION,
            "solver_request_template_version": SOLVER_REQUEST_TEMPLATE_VERSION,
            "solver_cache_schema_version": SOLVER_CACHE_SCHEMA_VERSION,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    out = args.out if args.out.is_absolute() else workspace / args.out
    payload = build_source_identity(workspace)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
