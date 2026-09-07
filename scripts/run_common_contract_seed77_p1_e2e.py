"""Prospective Seed77 Diversity P1 rerun under COMMON_SOLVER_CONTRACT_V1."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from infrastructure.common_solver_contract_v1.contract import (
    COMMON_SOLVER_CONTRACT_ID,
    CONTRACT_SPEC,
    contract_identity,
)
from infrastructure.common_solver_contract_v1.evaluator import (
    CommonSolverEvaluator,
    TransportResponse,
)
from infrastructure.common_solver_contract_v1.system_adapter import CommonContractSolverAdapter
from multi_dataset_diverse_rl import cli
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.governance.authorization import require_api_authorization
from multi_dataset_diverse_rl.persistence.checkpoint import restore_checkpoint
from multi_dataset_diverse_rl.persistence.identity import RunIdentity, build_run_identity
from multi_dataset_diverse_rl.provider_credentials import resolve_api_key, resolve_base_url
from multi_dataset_diverse_rl.termination import assess_trajectory_termination
from multi_dataset_diverse_rl.versions import COMMON_SOLVER_CONTRACT_V1_ID
from scripts import run_shadow_gated_evolution as shadow_runner
from scripts import run_vote_aligned_generic_shadow_pilot as base
from scripts.anti_overfitting_shadow_support import write_json
from scripts.run_shadow_gated_evolution import _backup, _rows


EXPERIMENT_ID = "common_contract_seed77_p1_e2e_v1"
SEED = 77
P0_STATIC = "P0_COMMON_STATIC"
P1 = base.P1
AUTH_ENV = "COMMON_CONTRACT_SEED77_P1_E2E_AUTHORIZED"
MANIFEST = ROOT / "experiments/manifests/common_contract_seed77_p1_e2e_v1.yaml"
DESIGN = ROOT / "experiments/common_contract_seed77_p1_e2e_v1"
CLASSIFIER = DESIGN / "classifier_definition.json"
SOURCE_SPLITS = ROOT / "runs/vote_aligned_confirmatory_seed76_77_v1_prep_authorized1/splits_private"
COMMON_REGISTRY = ROOT / "runs/common_solver_contract_v1_seed77_prep_20260907/private_replay_registry.json"
DEFAULT_PREP = ROOT / "runs/common_contract_seed77_p1_e2e_prep_20260907"
DEFAULT_RUN = ROOT / "runs/common_contract_seed77_p1_e2e_20260907"
DEFAULT_PHASE_A_REPORT = ROOT / "reports/common_contract_seed77_p1_e2e_prep_20260907"
DEFAULT_REPORT = ROOT / "reports/common_contract_seed77_p1_e2e_20260907"
FROZEN_COMMON_REPLAY = ROOT / "reports/common_solver_contract_v1_seed77_replay_20260907/per_state_results.csv"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None and getattr(exc, "response", None) is not None:
        status = getattr(exc.response, "status_code", None)
    if status is not None:
        return int(status) in CONTRACT_SPEC.retry_status_codes
    return isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError))


class CommonContractVoteAlignedSystem(base.VoteAlignedShadowSystem):
    """Unchanged P1 system whose Solver boundary is the shared contract."""

    shared_raw_cache: dict[str, str] = {}
    created: list["CommonContractVoteAlignedSystem"] = []

    def __init__(self, cfg: Config, *args: Any, **kwargs: Any) -> None:
        if cfg.models.solver_contract_id != COMMON_SOLVER_CONTRACT_ID:
            raise ValueError("common Solver contract identity is required")
        _, key = resolve_api_key(cfg.models.solver_api_key_env)
        _, endpoint = resolve_base_url(cfg.models.solver_base_url_env)
        if not key or not endpoint:
            raise RuntimeError("provider credentials unavailable")
        client = AsyncOpenAI(api_key=key, base_url=endpoint)

        async def transport(request: dict[str, Any]) -> TransportResponse:
            response = await client.chat.completions.create(
                **request,
                timeout=CONTRACT_SPEC.timeout_seconds,
            )
            usage = response.usage
            return TransportResponse(
                text=response.choices[0].message.content or "",
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                finish_reason=str(response.choices[0].finish_reason or ""),
            )

        evaluator = CommonSolverEvaluator(
            transport=transport,
            cache=type(self).shared_raw_cache,
            retryable=_retryable,
        )
        self.common_solver_adapter = CommonContractSolverAdapter(evaluator)
        super().__init__(cfg, *args, solver=self.common_solver_adapter.solve, **kwargs)

    async def solve(self, question: str, agent_id: int, prompt: str):
        answer = await super().solve(question, agent_id, prompt)
        if self.llm.calls and self.llm.calls[-1].get("client_role") == "solver":
            row = self.llm.calls[-1]
            row.update(
                {
                    "prompt_tokens": answer.prompt_tokens,
                    "completion_tokens": answer.completion_tokens,
                    "total_tokens": answer.total_tokens,
                    "finish_reason": answer.attempt_finish_reasons[-1] if answer.attempt_finish_reasons else "",
                    "common_solver_contract": COMMON_SOLVER_CONTRACT_ID,
                    "common_request_identity": answer.request_identity,
                    "common_transport_attempts": self.common_solver_adapter.last_transport_attempts,
                    "common_cache_hit": self.common_solver_adapter.last_cache_hit,
                }
            )
        return answer


def _config(
    *, out: Path, optimize: Path, validation: Path, cache: Path, initialization: Path
) -> Config:
    cfg = base._config(
        SEED,
        P1,
        out,
        optimize,
        validation,
        cache,
        initialization,
        False,
    )
    values = cfg.to_flat_dict()
    values.update(
        {
            "manifest_sha256": sha256_file(MANIFEST),
            "solver_contract_id": COMMON_SOLVER_CONTRACT_V1_ID,
            "solver_invalid_max_retries": 0,
            "final_test_enabled": False,
            "test_path": "TEST50_BLOCKED_BY_COMMON_CONTRACT_E2E",
            "test_size": 0,
        }
    )
    return Config.from_flat(**values)


def _copy_split(source: str, target: Path) -> None:
    source_path = SOURCE_SPLITS / source
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    shutil.copyfile(source_path, target)


def _source_paths() -> list[Path]:
    paths = [
        path.relative_to(ROOT)
        for path in (ROOT / "multi_dataset_diverse_rl").rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    paths.extend(
        [
            Path("infrastructure/common_solver_contract_v1/contract.py"),
            Path("infrastructure/common_solver_contract_v1/evaluator.py"),
            Path("infrastructure/common_solver_contract_v1/system_adapter.py"),
            Path("scripts/run_shadow_gated_evolution.py"),
            Path("scripts/run_vote_aligned_generic_shadow_pilot.py"),
            Path("scripts/run_common_contract_seed77_p1_e2e.py"),
            MANIFEST.relative_to(ROOT),
            (DESIGN / "PROTOCOL.md").relative_to(ROOT),
            CLASSIFIER.relative_to(ROOT),
            Path("experiments/anti_overfitting_split_v1/split_manifest.json"),
            Path("experiments/anti_overfitting_split_v1/fold_assignment.json"),
            FROZEN_COMMON_REPLAY.relative_to(ROOT),
        ]
    )
    return sorted(set(paths), key=lambda path: path.as_posix())


def prepare(prep: Path, report: Path) -> dict[str, Any]:
    if prep.exists() or report.exists():
        raise FileExistsError("fresh prep and report roots required")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("tracked worktree must be clean")
    prep.mkdir(parents=True)
    report.mkdir(parents=True)
    split_root = prep / "splits_private"
    split_root.mkdir()
    _copy_split("optimize_seed77.csv", split_root / "optimize_seed77.csv")
    _copy_split("fold_a.csv", split_root / "shadow_seed77.csv")
    _copy_split("validation.csv", split_root / "external_validation50.csv")
    common = read_json(COMMON_REGISTRY)
    p0_state = next(row for row in common["states"] if row["state_id"] == "P0_COMMON")
    p0_prompt = str(p0_state["ordered_prompts"][0])
    cfg = _config(
        out=Path("cell"),
        optimize=split_root / "optimize_seed77.csv",
        validation=split_root / "external_validation50.csv",
        cache=Path("cache.sqlite"),
        initialization=Path("initialization.json"),
    )
    protocol = {
        "experiment_id": EXPERIMENT_ID,
        "seed": SEED,
        "solver_contract_id": cfg.models.solver_contract_id,
        "solver_contract_identity": contract_identity(),
        "solver_model": cfg.models.agent_model,
        "solver_invalid_max_retries": cfg.models.solver_invalid_max_retries,
        "role_models": {
            "teacher": cfg.models.optimizer_model,
            "critic": cfg.models.optimizer_model,
            "student": cfg.models.optimizer_model,
            "evaluator": cfg.models.evaluator_model,
        },
        "thinking": False,
        "seed77_split": {
            "optimize": "fold_b+fold_c",
            "shadow": "fold_a",
            "external_validation": "ExternalValidation50",
        },
        "algorithm": {
            "arm": P1,
            "semantic_critic": True,
            "vote_aligned_scheduler": True,
            "common_safe": True,
            "winner_only_shadow": True,
            "max_update_opportunities": 32,
            "no_commit_patience": 6,
        },
        "final_evaluation": [P0_STATIC, P1],
        "validation_before_training_complete": False,
        "test50_accessed": False,
    }
    write_json(prep / "protocol_freeze.json", protocol)
    write_json(
        prep / "private_initialization.json",
        {
            "seed": SEED,
            "p0_prompt": p0_prompt,
            "p0_prompt_sha256": hashlib.sha256(p0_prompt.encode("utf-8")).hexdigest(),
        },
    )
    write_json(prep / "test_access_registry.json", {"events": [], "test50_calls": 0})
    write_json(
        prep / "source_freeze.json",
        {
            "execution_commit": git("rev-parse", "HEAD"),
            "tracked_worktree_clean": True,
            "files": [
                {"path": path.as_posix(), "sha256": sha256_file(ROOT / path)}
                for path in _source_paths()
            ],
            "split_sha256": {
                path.name: sha256_file(path) for path in sorted(split_root.iterdir())
            },
        },
    )
    phase_a = {
        "phase_a_gate": "PASS",
        "seed": SEED,
        "solver_contract": COMMON_SOLVER_CONTRACT_ID,
        "single_algorithm_change": "optimization_time_solver_contract",
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
    }
    write_json(report / "phase_a_gate.json", phase_a)
    write_json(report / "protocol_freeze.json", protocol)
    write_json(
        report / "provenance.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "source_commit": git("rev-parse", "HEAD"),
            "raw_evidence_tracked": False,
        },
    )
    return phase_a


def _verify_source_freeze(prep: Path, *, current: bool) -> None:
    freeze = read_json(prep / "source_freeze.json")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", freeze["execution_commit"], "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise RuntimeError("source freeze is not an ancestor")
    if current and git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("execution requires clean tracked worktree")
    for row in freeze["files"]:
        if current:
            observed = sha256_file(ROOT / row["path"])
        else:
            content = subprocess.check_output(
                ["git", "show", f"{freeze['execution_commit']}:{row['path']}"], cwd=ROOT
            )
            filtered = subprocess.check_output(
                [
                    "git", "cat-file", "--filters", f"--path={row['path']}",
                    f"{freeze['execution_commit']}:{row['path']}",
                ],
                cwd=ROOT,
            )
            candidates = {
                hashlib.sha256(content).hexdigest(),
                hashlib.sha256(filtered).hexdigest(),
            }
            observed = row["sha256"] if row["sha256"] in candidates else next(iter(candidates))
        if observed != row["sha256"]:
            raise RuntimeError(f"source freeze mismatch: {row['path']}")


def _authorize() -> None:
    if os.environ.get(AUTH_ENV) != "1":
        raise PermissionError(f"{AUTH_ENV}=1 is required")
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for role in ("solver", "teacher", "critic", "student"):
        require_api_authorization(
            manifest, phase="online_trajectory", role=role, explicit_user_authorized=True
        )
    for role in ("solver", "evaluator"):
        require_api_authorization(
            manifest, phase="frozen_validation", role=role, explicit_user_authorized=True
        )


@contextlib.contextmanager
def _patched_system(shadow_rows: Sequence[Mapping[str, Any]]) -> Iterator[None]:
    original_rows = CommonContractVoteAlignedSystem.shadow_rows
    original_enabled = CommonContractVoteAlignedSystem.shadow_enabled
    CommonContractVoteAlignedSystem.shadow_rows = list(shadow_rows)
    CommonContractVoteAlignedSystem.shadow_enabled = True
    CommonContractVoteAlignedSystem.created = []
    original_system = cli.PromptEnsembleOptimizationSystem
    original_load = cli._load

    def blocked_test(path: str, limit: int, fmt: str):
        if path == "TEST50_BLOCKED_BY_COMMON_CONTRACT_E2E":
            return []
        return original_load(path, limit, fmt)

    cli.PromptEnsembleOptimizationSystem = CommonContractVoteAlignedSystem
    cli._load = blocked_test
    try:
        yield
    finally:
        cli.PromptEnsembleOptimizationSystem = original_system
        cli._load = original_load
        CommonContractVoteAlignedSystem.shadow_rows = original_rows
        CommonContractVoteAlignedSystem.shadow_enabled = original_enabled


@contextlib.contextmanager
def _without_shadow() -> Iterator[None]:
    original_rows = CommonContractVoteAlignedSystem.shadow_rows
    original_enabled = CommonContractVoteAlignedSystem.shadow_enabled
    CommonContractVoteAlignedSystem.shadow_rows = []
    CommonContractVoteAlignedSystem.shadow_enabled = False
    try:
        yield
    finally:
        CommonContractVoteAlignedSystem.shadow_rows = original_rows
        CommonContractVoteAlignedSystem.shadow_enabled = original_enabled


async def _freeze_initialization(prep: Path, run_root: Path) -> tuple[Path, Path]:
    root = run_root / "initialization/seed77"
    root.mkdir(parents=True, exist_ok=False)
    manifest = root / "frozen_initialization_manifest.json"
    cache = root / "initial_solver_cache.sqlite"
    stable = root / "initial_solver_cache_frozen.sqlite"
    optimize = prep / "splits_private/optimize_seed77.csv"
    validation = prep / "splits_private/external_validation50.csv"
    cfg = _config(out=root, optimize=optimize, validation=validation, cache=cache, initialization=manifest)
    with _without_shadow():
        system = CommonContractVoteAlignedSystem(cfg)
    train = _rows(optimize)
    system.set_run_identity(
        build_run_identity(cfg, train_rows=train, val_rows=_rows(validation), test_rows=[], workspace=ROOT)
    )
    await system.initialize_fixed_probe(train)
    _backup(cache, stable)
    snapshot = system.frozen_initialization_snapshot()
    expected_prompt_hash = read_json(prep / "private_initialization.json")["p0_prompt_sha256"]
    if set(snapshot["initial_prompt_hashes"]) != {expected_prompt_hash}:
        raise RuntimeError("initial P0 prompt identity mismatch")
    write_json(
        manifest,
        {
            "manifest_version": "common_contract_seed77_initialization_v1",
            "seed": SEED,
            "solver_contract": COMMON_SOLVER_CONTRACT_ID,
            "solver_contract_identity": contract_identity(),
            "initialization_snapshot": snapshot,
            "test_calls": 0,
        },
    )
    return manifest, stable


def _evaluation_payload(system: CommonContractVoteAlignedSystem, metrics: Any, state_id: str) -> dict[str, Any]:
    rows = [
        {
            "case_position": index,
            "g": int(row.gold_vote_count),
            "vote_correct": bool(row.vote_correct),
            "oracle_correct": int(row.gold_vote_count) > 0,
            "top_tie": bool(row.top_tie),
        }
        for index, row in enumerate(metrics.rows)
    ]
    per_agent = [int(value) for value in metrics.per_agent_correct_counts]
    return {
        "state_id": state_id,
        "row_count": len(rows),
        "vote_correct": int(metrics.vote_correct_count),
        "vote_accuracy": float(metrics.plurality_vote_acc),
        "oracle_correct": sum(row["oracle_correct"] for row in rows),
        "oracle_accuracy": sum(row["oracle_correct"] for row in rows) / len(rows),
        "per_agent_correct": per_agent,
        "per_agent_accuracy": [value / len(rows) for value in per_agent],
        "mean_member_accuracy": float(metrics.mean_individual_acc),
        "invalid_rate": float(metrics.mean_invalid_rate),
        "tie_count": int(metrics.tie_count),
        "rows": rows,
        "test_calls": 0,
    }


async def _evaluate_p0(prep: Path, run_root: Path, initialization: Path, stable: Path) -> dict[str, Any]:
    out = run_root / "evaluation/P0_COMMON_STATIC"
    out.mkdir(parents=True, exist_ok=False)
    cache = out / "solver_cache.sqlite"
    _backup(stable, cache)
    optimize = prep / "splits_private/optimize_seed77.csv"
    validation = prep / "splits_private/external_validation50.csv"
    cfg = _config(out=out, optimize=optimize, validation=validation, cache=cache, initialization=initialization)
    with _without_shadow():
        system = CommonContractVoteAlignedSystem(cfg)
    system.set_run_identity(
        build_run_identity(cfg, train_rows=_rows(optimize), val_rows=_rows(validation), test_rows=[], workspace=ROOT)
    )
    await system.initialize_fixed_probe(_rows(optimize))
    state_hash = system.team_prompt_state_hash()
    metrics = await system.evaluate_dataset(_rows(validation))
    if system.team_prompt_state_hash() != state_hash:
        raise RuntimeError("P0 evaluation mutated state")
    payload = _evaluation_payload(system, metrics, P0_STATIC)
    payload["state_hash"] = state_hash
    write_json(out / "evaluation_private.json", payload)
    return payload


async def _evaluate_final(prep: Path, run_root: Path, cell: Path) -> dict[str, Any]:
    out = run_root / f"evaluation/{P1}"
    out.mkdir(parents=True, exist_ok=False)
    cache = out / "solver_cache.sqlite"
    _backup(cell / "solver_cache.sqlite", cache)
    checkpoint_path = cell / "training_checkpoint.json"
    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = read_json(checkpoint_path)
    optimize = prep / "splits_private/optimize_seed77.csv"
    validation = prep / "splits_private/external_validation50.csv"
    initialization = run_root / "initialization/seed77/frozen_initialization_manifest.json"
    cfg = _config(out=out, optimize=optimize, validation=validation, cache=cache, initialization=initialization)
    with _without_shadow():
        system = CommonContractVoteAlignedSystem(cfg)
    system.set_run_identity(RunIdentity(**checkpoint["run_identity"]))
    system.proposal_memory_run_id = str(checkpoint["proposal_memory_run_id"])
    system.fixed_probe = system.build_probe(_rows(optimize))
    restore_checkpoint(system, checkpoint)
    state_hash = system.team_prompt_state_hash()
    metrics = await system.evaluate_dataset(_rows(validation))
    if system.team_prompt_state_hash() != state_hash or sha256_file(checkpoint_path) != checkpoint_hash:
        raise RuntimeError("final evaluation mutated frozen evidence")
    payload = _evaluation_payload(system, metrics, P1)
    payload.update({"state_hash": state_hash, "checkpoint_sha256": checkpoint_hash})
    write_json(out / "evaluation_private.json", payload)
    return payload


async def execute(prep: Path, run_root: Path) -> dict[str, Any]:
    _authorize()
    _verify_source_freeze(prep, current=True)
    if run_root.exists():
        raise FileExistsError("fresh run root required")
    run_root.mkdir(parents=True)
    CommonContractVoteAlignedSystem.shared_raw_cache = {}
    initialization, stable = await _freeze_initialization(prep, run_root)
    cell = run_root / f"seed77/{P1}"
    cell.mkdir(parents=True)
    cache = cell / "solver_cache.sqlite"
    _backup(stable, cache)
    optimize = prep / "splits_private/optimize_seed77.csv"
    shadow = prep / "splits_private/shadow_seed77.csv"
    validation = prep / "splits_private/external_validation50.csv"
    cfg = _config(out=cell, optimize=optimize, validation=validation, cache=cache, initialization=initialization)
    with _patched_system(_rows(shadow)):
        await cli.run(cfg)
    if not (cell / "final_summary.json").is_file():
        raise RuntimeError("P1 trajectory did not finish")
    p0 = await _evaluate_p0(prep, run_root, initialization, stable)
    final = await _evaluate_final(prep, run_root, cell)
    write_json(run_root / "common_raw_cache_private.json", CommonContractVoteAlignedSystem.shared_raw_cache)
    result = {
        "execution_gate": "PASS",
        "seed": SEED,
        "trajectory_count": 1,
        "final_evaluation_count": 2,
        "p0_vote_accuracy": p0["vote_accuracy"],
        "final_vote_accuracy": final["vote_accuracy"],
        "validation_before_training_complete": False,
        "test_calls": 0,
        "common_solver_accounting": {
            "logical_calls": sum(
                created.common_solver_adapter.accounting()["logical_calls"]
                for created in CommonContractVoteAlignedSystem.created
            ),
            "provider_calls": sum(
                created.common_solver_adapter.accounting()["provider_calls"]
                for created in CommonContractVoteAlignedSystem.created
            ),
        },
    }
    write_json(run_root / "execution_summary.json", result)
    return result


def audit(prep: Path, run_root: Path) -> dict[str, Any]:
    _verify_source_freeze(prep, current=False)
    errors: list[str] = []
    cell = run_root / f"seed77/{P1}"
    checkpoint = read_json(cell / "training_checkpoint.json")
    assessment = assess_trajectory_termination(
        planned_update_opportunities=int(checkpoint["planned_update_count"]),
        executed_update_records=checkpoint["candidate_decisions"],
        stored_early_stop_reason=str(checkpoint["early_stop_reason"]),
        completed_update_count=int(checkpoint["completed_update_count"]),
    )
    if not assessment.training_completed or assessment.errors:
        errors.extend(f"termination:{value}" for value in assessment.errors)
        if not assessment.training_completed:
            errors.append("trajectory_not_terminal")
    identity = checkpoint["run_identity"]
    meta = read_json(cell / "run_meta.json")["config"]
    if meta["solver_contract_id"] != COMMON_SOLVER_CONTRACT_ID:
        errors.append("solver_contract")
    if int(meta["solver_invalid_max_retries"]) != 0:
        errors.append("invalid_retry")
    if meta["agent_model"] != "qwen3-8b" or meta["optimizer_model"] != "qwen3.7-flash":
        errors.append("model_identity")
    if identity["git_commit"] != read_json(prep / "source_freeze.json")["execution_commit"]:
        errors.append("execution_commit")
    if any(bool(checkpoint.get(key)) for key in (
        "test_evaluation_count",
        "test_used_for_selection",
        "test_used_for_training",
        "test_called_before_training_complete",
    )):
        errors.append("checkpoint_test_isolation")
    expected_initialization = read_json(
        run_root / "initialization/seed77/frozen_initialization_manifest.json"
    )["initialization_snapshot"]
    run_meta = read_json(cell / "run_meta.json")
    if run_meta.get("initial_prompt_hashes") != expected_initialization["initial_prompt_hashes"]:
        errors.append("frozen_initial_prompt_identity")
    if checkpoint.get("probe_hash") != expected_initialization["probe_hash"]:
        errors.append("frozen_initial_probe_identity")
    if checkpoint.get("training_state", {}).get("initial_team_state_hash") != expected_initialization["initial_train_state_hash"]:
        errors.append("frozen_initial_team_identity")
    final_summary = cell / "final_summary.json"
    if not final_summary.is_file():
        errors.append("final_summary_missing")
    evaluations = []
    for state_id in (P0_STATIC, P1):
        path = run_root / f"evaluation/{state_id}/evaluation_private.json"
        if not path.is_file():
            errors.append(f"evaluation_missing:{state_id}")
            continue
        row = read_json(path)
        evaluations.append(row)
        if int(row["row_count"]) != 50 or int(row["test_calls"]) != 0:
            errors.append(f"evaluation_inventory:{state_id}")
        if final_summary.is_file() and path.stat().st_mtime_ns < final_summary.stat().st_mtime_ns:
            errors.append(f"evaluation_before_training_freeze:{state_id}")
    if read_json(prep / "test_access_registry.json")["events"]:
        errors.append("test_access")
    result = {
        "audit_gate": "PASS" if not errors else "HOLD",
        "errors": sorted(set(errors)),
        "seed": SEED,
        "trajectory_count": 1,
        "evaluation_count": len(evaluations),
        "termination": assessment.to_dict(),
        "solver_contract": COMMON_SOLVER_CONTRACT_ID,
        "validation_before_training_complete": False,
        "test_calls": 0,
    }
    write_json(run_root / "audit_summary.json", result)
    return result


def _g_decomposition(p0: Mapping[str, Any], final: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, int]]]:
    p0_rows = {int(row["case_position"]): row for row in p0["rows"]}
    final_rows = {int(row["case_position"]): row for row in final["rows"]}
    matrix = [[0] * 6 for _ in range(6)]
    new_coverage = lost_coverage = new_votes = lost_votes = 0
    wrong = wrong_to_vote = correct = correct_preserved = 0
    oracle_gain = oracle_gain_to_vote = 0
    for index in range(50):
        left, right = p0_rows[index], final_rows[index]
        g0, g1 = int(left["g"]), int(right["g"])
        matrix[g0][g1] += 1
        new_coverage += int(g0 == 0 and g1 > 0)
        lost_coverage += int(g0 > 0 and g1 == 0)
        new_votes += max(0, g1 - g0)
        lost_votes += max(0, g0 - g1)
        if left["vote_correct"]:
            correct += 1
            correct_preserved += int(bool(right["vote_correct"]))
        else:
            wrong += 1
            wrong_to_vote += int(bool(right["vote_correct"]))
        if not left["oracle_correct"] and right["oracle_correct"]:
            oracle_gain += 1
            oracle_gain_to_vote += int(bool(right["vote_correct"]))
    rows = [
        {"p0_g": g0, "final_g": g1, "case_count": matrix[g0][g1]}
        for g0 in range(6) for g1 in range(6)
    ]
    return {
        "new_coverage_cases": new_coverage,
        "lost_coverage_cases": lost_coverage,
        "new_correct_member_votes": new_votes,
        "lost_correct_member_votes": lost_votes,
        "p0_wrong_to_final_majority": wrong_to_vote,
        "p0_wrong_cases": wrong,
        "p0_correct_preserved": correct_preserved,
        "p0_correct_cases": correct,
        "oracle_gain_cases": oracle_gain,
        "oracle_gain_to_vote_gain": oracle_gain_to_vote,
        "p0_g_distribution": {str(g): sum(matrix[g]) for g in range(6)},
        "final_g_distribution": {str(g): sum(matrix[left][g] for left in range(6)) for g in range(6)},
    }, rows


def classify(p0: Mapping[str, Any], final: Mapping[str, Any]) -> str:
    if final["mean_member_accuracy"] > p0["mean_member_accuracy"] and final["vote_accuracy"] > p0["vote_accuracy"]:
        return "CONTRACT_TRANSFER_FAILURE_DOMINANT"
    if final["mean_member_accuracy"] < p0["mean_member_accuracy"] and final["oracle_accuracy"] - p0["oracle_accuracy"] >= 0.15:
        return "COMPETENCE_REDISTRIBUTION_BIAS_SUPPORTED"
    if final["mean_member_accuracy"] > p0["mean_member_accuracy"] or final["vote_accuracy"] > p0["vote_accuracy"]:
        return "MIXED_CONTRACT_AND_OPTIMIZATION_EFFECTS"
    return "NO_CLEAR_DIAGNOSIS"


def analyze(prep: Path, run_root: Path, report: Path) -> dict[str, Any]:
    gate = audit(prep, run_root)
    if gate["audit_gate"] != "PASS":
        return {"analysis_gate": "NOT_RUN", "audit_gate": gate["audit_gate"]}
    if report.exists():
        raise FileExistsError("fresh report required")
    report.mkdir(parents=True)
    p0 = read_json(run_root / f"evaluation/{P0_STATIC}/evaluation_private.json")
    final = read_json(run_root / f"evaluation/{P1}/evaluation_private.json")
    decomposition, matrix = _g_decomposition(p0, final)
    classifier = classify(p0, final)
    with FROZEN_COMMON_REPLAY.open(encoding="utf-8", newline="") as handle:
        frozen_rows = list(csv.DictReader(handle))
    frozen_final = next(
        row for row in frozen_rows if row["state_id"] == "DIVERSITY_SEED77_P1_FINAL"
    )
    frozen_common = {
        "vote": float(frozen_final["vote_accuracy"]),
        "oracle": float(frozen_final["oracle_accuracy"]),
        "mean_member": float(frozen_final["mean_member_accuracy"]),
    }
    metrics = []
    for row in (p0, final):
        metrics.append({key: row[key] for key in (
            "state_id", "vote_accuracy", "oracle_accuracy", "mean_member_accuracy", "invalid_rate", "tie_count"
        )})
    with (report / "state_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    with (report / "g_transition_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("p0_g", "final_g", "case_count"))
        writer.writeheader(); writer.writerows(matrix)
    summary = {
        "analysis_gate": "PASS",
        "classifier": classifier,
        "p1_minus_p0": {
            "vote": final["vote_accuracy"] - p0["vote_accuracy"],
            "oracle": final["oracle_accuracy"] - p0["oracle_accuracy"],
            "mean_member": final["mean_member_accuracy"] - p0["mean_member_accuracy"],
        },
        "decomposition": decomposition,
        "old_frozen_p1_under_common_contract": frozen_common,
        "new_e2e_minus_old_frozen_p1": {
            "vote": final["vote_accuracy"] - frozen_common["vote"],
            "oracle": final["oracle_accuracy"] - frozen_common["oracle"],
            "mean_member": final["mean_member_accuracy"] - frozen_common["mean_member"],
        },
        "test_calls": 0,
    }
    write_json(report / "summary.json", summary)
    write_json(report / "classifier.json", {"classifier": classifier, "definition": read_json(CLASSIFIER)})
    write_json(report / "audit.json", gate)
    write_json(
        report / "provenance.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "execution_commit": read_json(prep / "source_freeze.json")["execution_commit"],
            "solver_contract_identity": contract_identity(),
            "raw_evidence_tracked": False,
        },
    )
    (report / "README.md").write_text(
        "# Seed77 common-contract end-to-end P1\n\n"
        f"Classifier: `{classifier}`. Validation was evaluated only after training freeze; Test50 calls were zero.\n",
        encoding="utf-8", newline="\n",
    )
    write_json(
        report / "fact_assertions.json",
        {
            "audit_gate": gate["audit_gate"],
            "state_count": 2,
            "validation_rows_per_state": 50,
            "solver_contract": COMMON_SOLVER_CONTRACT_ID,
            "test_calls": 0,
        },
    )
    forbidden_markers = (
        "DASHSCOPE",
        "api_key",
        "raw_response",
        "question_text",
        "gold_answer",
        ".sqlite",
        "checkpoint",
    )
    findings = []
    for path in report.iterdir():
        if path.is_file() and path.name not in {"sanitization_manifest.json", "sha256_manifest.json"}:
            lowered = path.read_text(encoding="utf-8").lower()
            for marker in forbidden_markers:
                if marker.lower() in lowered:
                    findings.append({"file": path.name, "marker": marker})
    write_json(
        report / "sanitization_manifest.json",
        {"status": "PASS" if not findings else "FAIL", "findings": findings},
    )
    if findings:
        raise RuntimeError("sanitization failed")
    write_json(
        report / "sha256_manifest.json",
        {
            path.name: sha256_file(path)
            for path in sorted(report.iterdir())
            if path.is_file() and path.name != "sha256_manifest.json"
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--audit", action="store_true")
    modes.add_argument("--analyze", action="store_true")
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--phase-a-report", type=Path, default=DEFAULT_PHASE_A_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    for path in (args.prep, args.run_root, args.phase_a_report, args.report):
        path.resolve().relative_to(ROOT.resolve())
    if args.prepare:
        result = prepare(args.prep.resolve(), args.phase_a_report.resolve())
    elif args.execute:
        result = asyncio.run(execute(args.prep.resolve(), args.run_root.resolve()))
    elif args.audit:
        result = audit(args.prep.resolve(), args.run_root.resolve())
    else:
        result = analyze(args.prep.resolve(), args.run_root.resolve(), args.report.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
