from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import Config
from ..utils import normalize_spaces


@dataclass(frozen=True)
class RunIdentity:
    method_version: str
    experiment_setting: str
    git_commit: str
    git_dirty: bool
    config_fingerprint: str
    manifest_sha256: str
    train_file_sha256: str
    val_file_sha256: str
    test_file_sha256: str
    train_question_set_hash: str
    val_question_set_hash: str
    test_question_set_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: str) -> str:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Run identity input does not exist: {target}")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def question_set_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    questions = sorted(
        hashlib.sha256(normalize_spaces(str(row["question"])).lower().encode("utf-8")).hexdigest()
        for row in rows
    )
    return hashlib.sha256(json.dumps(questions, separators=(",", ":")).encode("utf-8")).hexdigest()


def _git_identity(workspace: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=workspace, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip())
    return commit, dirty


def config_fingerprint(cfg: Config) -> str:
    values = cfg.to_flat_dict()
    for operational in ("out_dir", "resume_from_checkpoint"):
        values.pop(operational, None)
    values["endpoint_identity"] = {
        "solver": os.getenv(cfg.models.solver_base_url_env, "") if cfg.models.solver_base_url_env else os.getenv("OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE", "")),
        "optimizer": os.getenv(cfg.models.optimizer_base_url_env, "") if cfg.models.optimizer_base_url_env else os.getenv("OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE", "")),
        "evaluator": os.getenv(cfg.models.evaluator_base_url_env, "") if cfg.models.evaluator_base_url_env else os.getenv("OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE", "")),
    }
    values["behavior_versions"] = {
        "member_objective": "integer_vote_min_sum_v1",
        "responsibility": "member_need_pareto_seeded_v1",
        "target_selection": "overdue_member_pareto_v1",
        "stage_a": "team_vote_worst_mean_v1",
        "stage_b": "competence_guard_member_pareto_v1",
        "validation": "initial_member_feasible_v1",
        "checkpoint": 5,
    }
    encoded = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def solver_request_identity(cfg: Config) -> str:
    payload = solver_request_components(cfg)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def solver_request_components(cfg: Config) -> dict[str, Any]:
    endpoint = (
        os.getenv(cfg.models.solver_base_url_env, "")
        if cfg.models.solver_base_url_env
        else os.getenv("OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE", ""))
    )
    return {
        "solver_model": cfg.models.agent_model,
        "endpoint_identity": hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
        "max_tokens": cfg.models.max_tokens,
        "output_contract_version": cfg.peer_state.solver_output_contract_version,
        "request_template": "decision_procedure_with_task_contract_v1",
    }


def build_run_identity(
    cfg: Config,
    *,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    workspace: str | Path = ".",
) -> RunIdentity:
    commit, dirty = _git_identity(Path(workspace).resolve())
    return RunIdentity(
        method_version=cfg.training.method_version,
        experiment_setting=cfg.training.experiment_setting,
        git_commit=commit,
        git_dirty=dirty,
        config_fingerprint=config_fingerprint(cfg),
        manifest_sha256=cfg.data.manifest_sha256,
        train_file_sha256=_sha256_file(cfg.data.train_path),
        val_file_sha256=_sha256_file(cfg.data.val_path),
        test_file_sha256=_sha256_file(cfg.data.test_path),
        train_question_set_hash=question_set_hash(train_rows),
        val_question_set_hash=question_set_hash(val_rows),
        test_question_set_hash=question_set_hash(test_rows),
    )


def validate_run_identity(expected: RunIdentity, actual: Mapping[str, Any]) -> None:
    expected_payload = expected.to_dict()
    missing = sorted(set(expected_payload) - set(actual))
    mismatches = {
        key: {"expected": expected_payload[key], "actual": actual[key]}
        for key in expected_payload
        if key in actual and actual[key] != expected_payload[key]
    }
    if missing or mismatches:
        raise ValueError(
            "Run identity mismatch; refusing resume or completed-run reuse: "
            + json.dumps({"missing": missing, "mismatches": mismatches}, sort_keys=True)
        )
