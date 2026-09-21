"""Seed78 scheduler-only A/B pilot using the official local GEPA backend.

``--prepare`` and ``--preflight`` are strictly zero API.  ``--execute`` is a
separate, fail-closed entry point for the execution-only handoff.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from dataclasses import asdict
from typing import Any, Mapping, Sequence

import yaml
from openai import APIConnectionError, AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from infrastructure.common_solver_contract_v1.contract import (  # noqa: E402
    COMMON_SOLVER_CONTRACT_ID,
    CONTRACT_SPEC,
    canonical_json_bytes,
    contract_identity,
)
from infrastructure.common_solver_contract_v1.evaluator import (  # noqa: E402
    CommonSolverEvaluator,
    TransportResponse,
)
from multi_dataset_diverse_rl.config import Config  # noqa: E402
from multi_dataset_diverse_rl.evaluation.output_contract import (  # noqa: E402
    SOLVER_OUTPUT_CONTRACT_VERSION,
)
from multi_dataset_diverse_rl.evaluation.prompt_question import PromptAnswer  # noqa: E402
from multi_dataset_diverse_rl.evaluation.solver_stage import (  # noqa: E402
    validate_solver_stage_attribution,
)
from multi_dataset_diverse_rl.governance.authorization import require_api_authorization  # noqa: E402
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import (  # noqa: E402
    GEPAOptimizerConfig,
    GEPALocalPromptOptimizer,
    local_gepa_budget_capacity,
    verify_frozen_gepa_engine_contract,
)
from multi_dataset_diverse_rl.persistence.identity import build_run_identity  # noqa: E402
from multi_dataset_diverse_rl.shadow_gate import advance_no_commit_streak  # noqa: E402
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem  # noqa: E402
from multi_dataset_diverse_rl.team_search.candidate_selector import (  # noqa: E402
    CommonSafeTeamCandidateSelector,
)
from multi_dataset_diverse_rl.team_search.controller import TeamSearchController  # noqa: E402
from multi_dataset_diverse_rl.team_search.primary_responsibility_binding import (  # noqa: E402
    PrimaryResponsibilityOnlineBinding,
)
from multi_dataset_diverse_rl.team_search.primary_responsibility_scheduler import (  # noqa: E402
    PrimaryResponsibilityPersistentRealizabilityScheduler,
)
from multi_dataset_diverse_rl.team_search.schemas import (  # noqa: E402
    TeamSearchAssignment,
    TeamSearchRequest,
)
from multi_dataset_diverse_rl.team_search.system_runtime import (  # noqa: E402
    FrozenResponsibilitySnapshot,
    SystemLocalSolverEvaluator,
    SystemResponsibilityAssignmentFactory,
    SystemTeamCandidateEvaluator,
    SystemTeamCommitter,
    freeze_current_responsibility,
)
from multi_dataset_diverse_rl.team_search.task_builder import LocalTaskBuilder  # noqa: E402
from multi_dataset_diverse_rl.vote_aligned_scheduler import (  # noqa: E402
    FALLBACK_RR,
    PURE_COVERAGE,
    select_vote_aligned_targets,
)
from scripts.anti_overfitting_shadow_support import (  # noqa: E402
    construct_assignment,
    export_private_splits,
    metadata,
    sha256_file,
    sha256_json,
    write_json,
)


EXPERIMENT_ID = "seed78_primary_responsibility_ab_v1"
SEED = 78
ARM_A = "A_VOTE_ALIGNED_HIERARCHICAL"
ARM_B = "B_PRIMARY_RESPONSIBILITY_REALIZABILITY"
ARMS = (ARM_A, ARM_B)
SOLVER_MODEL = "qwen3-8b"
ROLE_MODEL = "qwen3.7-flash"
MAX_OPPORTUNITIES = 32
NO_COMMIT_PATIENCE = 6
LOCAL_GEPA_METRIC_BUDGET = 36
AUTH_ENV = "SEED78_PRIMARY_RESPONSIBILITY_AB_AUTHORIZED"
MANIFEST = ROOT / "experiments/manifests/seed78_primary_responsibility_ab_v1.yaml"
DESIGN = ROOT / "experiments/seed78_primary_responsibility_ab_v1"
DEFAULT_PREP = ROOT / "runs/seed78_primary_responsibility_ab_v1_prep_retry1_authorized2"
DEFAULT_RUN = ROOT / "runs/seed78_primary_responsibility_ab_v1_retry1"
DEFAULT_REPORT = ROOT / "reports/seed78_primary_responsibility_ab_v1"
TASK_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "seed78_local_gepa_task", default=None
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None and getattr(exc, "response", None) is not None:
        status = getattr(exc.response, "status_code", None)
    if status is not None:
        return int(status) in CONTRACT_SPEC.retry_status_codes
    return isinstance(
        exc,
        (TimeoutError, ConnectionError, asyncio.TimeoutError, APIConnectionError),
    )


class DurableLedger:
    """Append-only run-local accounting; raw text is never persisted here."""

    def __init__(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError("ledger path must be fresh")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.identities: set[str] = set()

    def append(self, row: Mapping[str, Any]) -> None:
        required = {
            "record_id", "phase", "logical_role", "client_role", "provider_attempts",
            "successful_provider_calls", "cache_hit", "input_tokens", "output_tokens",
            "total_tokens", "seed", "arm", "update_index", "target_member",
        }
        if required - set(row):
            raise ValueError("incomplete Seed78 ledger record")
        if int(row["input_tokens"]) + int(row["output_tokens"]) != int(row["total_tokens"]):
            raise ValueError("Seed78 ledger token arithmetic mismatch")
        identity = str(row["record_id"])
        if identity in self.identities:
            raise ValueError("duplicate Seed78 ledger record identity")
        self.identities.add(identity)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class Seed78System(PromptEnsembleOptimizationSystem):
    """Canonical system state with one per-arm COMMON_SOLVER_CONTRACT_V1 cache."""

    def __init__(
        self,
        cfg: Config,
        *,
        arm: str,
        ledger: DurableLedger,
        raw_cache: dict[str, str],
    ) -> None:
        key = os.environ.get(cfg.models.solver_api_key_env, "")
        endpoint = os.environ.get(cfg.models.solver_base_url_env, "")
        if not key or not endpoint:
            from multi_dataset_diverse_rl.provider_credentials import resolve_api_key, resolve_base_url

            _, key = resolve_api_key(cfg.models.solver_api_key_env)
            _, endpoint = resolve_base_url(cfg.models.solver_base_url_env)
        if not key or not endpoint:
            raise RuntimeError("COMMON_SOLVER_CONTRACT_V1 credentials unavailable")
        client = AsyncOpenAI(api_key=key, base_url=endpoint)

        async def transport(request: dict[str, Any]) -> TransportResponse:
            attempt_guard = getattr(self.ledger, "reserve_provider_attempt", None)
            if attempt_guard is not None:
                attempt_guard("solver")
            try:
                response = await client.chat.completions.create(
                    **request, timeout=CONTRACT_SPEC.timeout_seconds
                )
            except Exception as exc:
                request_identity = hashlib.sha256(
                    canonical_json_bytes(request)
                ).hexdigest()
                attempt_index = (
                    self._solver_failure_attempt_count_by_request.get(
                        request_identity, 0
                    )
                    + 1
                )
                self._solver_failure_attempt_count_by_request[
                    request_identity
                ] = attempt_index
                persist_failed_attempt({
                    "request_identity": request_identity,
                    "attempt_index": attempt_index,
                    "error_type": type(exc).__name__,
                    "status_code": getattr(exc, "status_code", None),
                })
                raise
            usage = response.usage
            return TransportResponse(
                text=response.choices[0].message.content or "",
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                finish_reason=str(response.choices[0].finish_reason or ""),
            )

        self.arm = arm
        self.ledger = ledger
        self._solver_stage: dict[str, Any] | None = None
        self._solver_sequence = 0
        self._solver_failure_attempt_count_by_request: dict[str, int] = {}

        def persist_failed_attempt(event: Mapping[str, object]) -> None:
            if self._solver_stage is None:
                raise RuntimeError("failed Solver attempt lacks frozen stage attribution")
            self._solver_sequence += 1
            stage = dict(self._solver_stage)
            self.ledger.append({
                "record_id": f"{arm}:solver:{self._solver_sequence}",
                "record_kind": "solver_provider_attempt_failure",
                "phase": stage["phase"],
                "logical_role": "solver",
                "client_role": "solver",
                "provider_attempts": 1,
                "successful_provider_calls": 0,
                "cache_hit": False,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "seed": self.cfg.training.seed,
                "arm": arm,
                "update_index": int(stage.get("update_index", -1)),
                "target_member": int(stage.get("target_member", -1)),
                "candidate_id": str(stage.get("candidate_id", "")),
                "request_identity": str(event["request_identity"]),
                "attempt_index": int(event["attempt_index"]),
                "error_type": str(event["error_type"]),
                "status_code": event.get("status_code"),
            })

        self.common = CommonSolverEvaluator(
            transport=transport,
            cache=raw_cache,
            retryable=_retryable,
        )

        async def solver(question: str, agent_id: int, prompt: str) -> PromptAnswer:
            del agent_id
            if self._solver_stage is None:
                raise RuntimeError("Solver call lacks frozen stage attribution")
            result = await self.common.evaluate(
                decision_procedure=prompt, question=question
            )
            parsed = result.response
            raw = self.common.cache[result.request_identity]
            response_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            self._solver_sequence += 1
            stage = dict(self._solver_stage)
            self.ledger.append({
                "record_id": f"{arm}:solver:{self._solver_sequence}",
                "record_kind": "solver_logical_completion",
                "phase": stage["phase"],
                "logical_role": "solver",
                "client_role": "solver",
                "provider_attempts": int(not result.cache_hit),
                "successful_provider_calls": int(not result.cache_hit),
                "cache_hit": result.cache_hit,
                "input_tokens": result.prompt_tokens,
                "output_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
                "seed": self.cfg.training.seed,
                "arm": arm,
                "update_index": int(stage.get("update_index", -1)),
                "target_member": int(stage.get("target_member", -1)),
                "candidate_id": str(stage.get("candidate_id", "")),
                "request_identity": result.request_identity,
                "transport_attempts_for_logical_call": result.transport_attempts,
            })
            return PromptAnswer(
                answer=parsed.answer,
                trace=raw,
                valid=parsed.valid,
                validity_status=parsed.status,
                raw_final_answer_payload=parsed.answer,
                final_answer_line_count=parsed.final_answer_line_count,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
                response_hash=response_hash,
                request_identity=result.request_identity,
                solver_attempt_count=1,
                first_attempt_valid=parsed.valid,
                recovered_from_invalid=False,
                terminal_invalid=not parsed.valid,
                raw_invalid_attempt_count=int(not parsed.valid),
                attempt_validity_statuses=(parsed.status,),
                attempt_finish_reasons=(result.finish_reason,),
                attempt_response_hashes=(response_hash,),
                attempt_prompt_tokens=(result.prompt_tokens,),
                attempt_completion_tokens=(result.completion_tokens,),
                attempt_total_tokens=(result.prompt_tokens + result.completion_tokens,),
            )

        super().__init__(cfg, solver=solver)
        optimizer_attempt_guard = getattr(self.ledger, "reserve_provider_attempt", None)
        if optimizer_attempt_guard is not None:
            self.llm.provider_attempt_guard = lambda: optimizer_attempt_guard("reflection")

    def set_stage(self, stage: Mapping[str, Any] | None) -> None:
        self._solver_stage = (
            validate_solver_stage_attribution(stage) if stage is not None else None
        )

    def optimizer_accounting(self) -> dict[str, int]:
        rows = [row for row in self.llm.calls if row.get("client_role") == "optimizer"]
        return {
            "successful_calls": sum(bool(row.get("success")) for row in rows),
            "input_tokens": sum(int(row.get("prompt_tokens", 0)) for row in rows),
            "output_tokens": sum(int(row.get("completion_tokens", 0)) for row in rows),
        }


class ReflectionLM:
    def __init__(self, system: Seed78System) -> None:
        self.system = system
        self.sequence = 0

    @staticmethod
    def _messages(prompt: str | list[dict[str, Any]]) -> tuple[str, str]:
        if isinstance(prompt, str):
            return (
                "Return only one complete replacement reasoning procedure. Do not discuss output formatting.",
                prompt,
            )
        systems = [str(row.get("content", "")) for row in prompt if row.get("role") == "system"]
        others = [
            f"{row.get('role', 'user')}: {row.get('content', '')}"
            for row in prompt
            if row.get("role") != "system"
        ]
        return "\n".join(systems) or "You improve a reasoning procedure.", "\n\n".join(others)

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        context = TASK_CONTEXT.get()
        if context is None:
            raise RuntimeError("reflection call lacks local task attribution")
        system_prompt, user_prompt = self._messages(prompt)

        async def run() -> str:
            begin = len(self.system.llm.calls)
            try:
                result = await self.system.llm.chat_result(
                    ROLE_MODEL,
                    system_prompt,
                    user_prompt,
                    0.0,
                    1800,
                    "optimizer",
                    "reflection",
                )
                return result.text
            finally:
                for raw in self.system.llm.calls[begin:]:
                    if raw.get("client_role") != "optimizer":
                        continue
                    self.sequence += 1
                    self.system.ledger.append({
                        "record_id": f"{self.system.arm}:optimizer:{self.sequence}",
                        "phase": "local_optimizer_reflection",
                        "logical_role": "reflection",
                        "client_role": "optimizer",
                        "provider_attempts": 1,
                        "successful_provider_calls": int(bool(raw.get("success"))),
                        "cache_hit": False,
                        "input_tokens": int(raw.get("prompt_tokens", 0)),
                        "output_tokens": int(raw.get("completion_tokens", 0)),
                        "total_tokens": int(raw.get("total_tokens", 0)),
                        "seed": self.system.cfg.training.seed,
                        "arm": self.system.arm,
                        "update_index": int(context["update_index"]),
                        "target_member": int(context["target_member"]),
                        "candidate_id": "reflection",
                    })

        return asyncio.run_coroutine_threadsafe(run(), context["loop"]).result()


class ContextualOptimizer:
    def __init__(self, inner: GEPALocalPromptOptimizer, local_solver: SystemLocalSolverEvaluator) -> None:
        self.inner = inner
        self.local_solver = local_solver

    async def optimize(self, task: Any) -> Any:
        member = int(task.task_id.rsplit("member", 1)[1])
        update = int(task.task_id.split("_update", 1)[1].split("_", 1)[0])
        context = {
            "loop": asyncio.get_running_loop(),
            "update_index": update,
            "target_member": member,
            "phase": "local_optimizer_solver_eval",
            "parent_id": f"seed{self.local_solver.system.cfg.seed}_update{update}",
        }
        token = TASK_CONTEXT.set(context)
        self.local_solver.task_context = context
        try:
            return await self.inner.optimize(task)
        finally:
            self.local_solver.task_context = None
            TASK_CONTEXT.reset(token)


def _config(out: Path, *, optimize_path: Path, validation_path: Path) -> Config:
    return Config.from_flat(
        task_type="bbh",
        dataset_format="mars",
        comparison_task_id="disambiguation_qa",
        benchmark="BBH",
        answer_format="option_letter",
        train_path=str(optimize_path.resolve()),
        val_path=str(validation_path.resolve()),
        test_path="TEST50_BLOCKED",
        manifest_sha256=sha256_file(MANIFEST),
        train_size=100,
        val_size=50,
        test_size=0,
        agent_model=SOLVER_MODEL,
        optimizer_model=ROLE_MODEL,
        evaluator_model=ROLE_MODEL,
        temperature=0.0,
        solver_max_tokens=1800,
        solver_invalid_max_retries=0,
        solver_contract_id=COMMON_SOLVER_CONTRACT_ID,
        experiment_setting="experimental_diversity_d2_rr_generic",
        target_scheduler="round_robin",
        agents=5,
        epochs=1,
        update_every=1,
        seed=SEED,
        proposal_memory_mode="off",
        num_candidates_per_parent=2,
        candidate_eval_pool_size=100,
        eval_solver_call_concurrency=8,
        stage_b_candidate_budget=2,
        out_dir=str(out),
        shared_solver_cache_path="",
        provider_call_budget=100000,
        total_token_budget=100_000_000,
        final_test_enabled=False,
        preserve_final_checkpoint=True,
    )


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
            Path("scripts/anti_overfitting_shadow_support.py"),
            Path("scripts/run_seed78_primary_responsibility_ab.py"),
            MANIFEST.relative_to(ROOT),
            (DESIGN / "PROTOCOL.md").relative_to(ROOT),
            (DESIGN / "classifier_definition.json").relative_to(ROOT),
            (DESIGN / "EXPERIMENT_HANDOFF.md").relative_to(ROOT),
            Path("experiments/anti_overfitting_split_v1/split_manifest.json"),
            Path("experiments/anti_overfitting_split_v1/fold_assignment.json"),
        ]
    )
    return sorted(set(paths), key=lambda row: row.as_posix())


def protocol_document() -> dict[str, Any]:
    capacity = asdict(local_gepa_budget_capacity(
        metric_budget=LOCAL_GEPA_METRIC_BUDGET,
        validation_size=12,
        reflection_minibatch_size=GEPAOptimizerConfig().reflection_minibatch_size,
    ))
    return {
        "schema_version": "seed78_primary_responsibility_ab_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "seed": SEED,
        "evidence_type": "single_seed_prospective_mechanism_pilot",
        "split_identity": {
            "optimize100": "anti_overfitting_split_v1/fold_a+fold_b",
            "shadow50": "anti_overfitting_split_v1/fold_c",
            "validation50": "anti_overfitting_split_v1/validation",
            "test50": "blocked_zero_calls",
        },
        "models": {
            "solver": SOLVER_MODEL,
            "solver_thinking": False,
            "reflection": ROLE_MODEL,
            "evaluator_config": ROLE_MODEL,
        },
        "solver_contract": COMMON_SOLVER_CONTRACT_ID,
        "solver_contract_hash": contract_identity(),
        "arms": {
            ARM_A: "direct_flip > near_margin > pure_coverage + deterministic RR",
            ARM_B: "max(4D,2N,C)/(1+persistent_member_failure_count)",
        },
        "only_difference": "target_scheduler",
        "shared": {
            "initial_prompt_team": "shared_identical_P0",
            "official_gepa": asdict(GEPAOptimizerConfig()),
            "local_metric_budget_per_target": LOCAL_GEPA_METRIC_BUDGET,
            "local_validation_size": 12,
            "local_gepa_budget_capacity": capacity,
            "team_minibatch": "strict primary-lane 4 responsibility + global 4 coalition + global 4 preservation",
            "full_team_max_promotions_per_target": 2,
            "common_safe": True,
            "winner_only_shadow": True,
            "always_two_targets": True,
            "max_one_commit_per_opportunity": True,
            "max_opportunities": MAX_OPPORTUNITIES,
            "no_commit_patience": NO_COMMIT_PATIENCE,
        },
        "data_roles": {
            "shadow50": "optimization_time_safety_data_not_generalization",
            "validation50": "once_per_final_arm_after_both_training_trajectories_freeze",
            "test50": "zero_access",
        },
        "primary_mechanism_metrics": [
            "target_count_by_member",
            "commit_count_by_member",
            "target_to_commit_rate_by_member",
            "max_unresolved_target_streak_by_member",
            "tokens_per_commit",
        ],
        "final_metrics": ["Vote", "MeanMember", "Vote-minus-MeanMember", "Oracle", "G0..G5"],
    }


def prepare(prep: Path) -> dict[str, Any]:
    if prep.exists():
        raise FileExistsError("fresh prep root required")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before source freeze")
    items, raw = metadata()
    assignment = construct_assignment(items)
    prep.mkdir(parents=True)
    export_private_splits(raw, assignment, prep / "splits_private")
    optimize = prep / "splits_private/optimize100.csv"
    with optimize.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question", "answer"])
        writer.writeheader()
        for source in ("fold_a.csv", "fold_b.csv"):
            writer.writerows(_rows(prep / "splits_private" / source))
    protocol = protocol_document()
    protocol_hash = sha256_json(protocol)
    write_json(prep / "protocol_freeze.json", protocol)
    write_json(
        prep / "test_access_registry.json",
        {"events": [], "test50_calls": 0, "test50_authorized": False},
    )
    freeze = {
        "execution_commit": _git("rev-parse", "HEAD"),
        "protocol_sha256": protocol_hash,
        "files": [
            {"path": path.as_posix(), "sha256": sha256_file(ROOT / path)}
            for path in _source_paths()
        ],
        "private_split_sha256": {
            path.name: sha256_file(path)
            for path in sorted((prep / "splits_private").glob("*.csv"))
        },
    }
    write_json(prep / "source_freeze.json", freeze)
    handoff = (
        "# Seed78 execution handoff (generated)\n\n"
        f"Execution commit: `{freeze['execution_commit']}`  \n"
        f"Protocol SHA256: `{protocol_hash}`  \n"
        "API calls during preparation: `0`\n\n"
        "Run only after confirming explicit authorization is still active:\n\n"
        "```powershell\n"
        f"$env:{AUTH_ENV}='1'\n"
        "D:\\Anaconda\\envs\\DL\\python.exe scripts\\run_seed78_primary_responsibility_ab.py "
        f"--execute --prep {prep.relative_to(ROOT)} --run {DEFAULT_RUN.relative_to(ROOT)}\n"
        "```\n\n"
        "Do not change the seed, split, arms, budget, method, or output root. On any failure, "
        "preserve evidence and HOLD without retry.\n"
    )
    (prep / "EXPERIMENT_HANDOFF.md").write_text(handoff, encoding="utf-8")
    result = {
        "phase_a_gate": "PASS",
        "protocol_sha256": protocol_hash,
        "execution_commit": freeze["execution_commit"],
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
    }
    write_json(prep / "phase_a_gate.json", result)
    return result


def _verify_freeze(prep: Path) -> None:
    freeze = _read_json(prep / "source_freeze.json")
    if sha256_json(_read_json(prep / "protocol_freeze.json")) != freeze["protocol_sha256"]:
        raise RuntimeError("protocol freeze mismatch")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", freeze["execution_commit"], "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise RuntimeError("execution commit is not an ancestor of HEAD")
    for row in freeze["files"]:
        if sha256_file(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"source freeze mismatch: {row['path']}")
    for name, digest in freeze["private_split_sha256"].items():
        if sha256_file(prep / "splits_private" / name) != digest:
            raise RuntimeError(f"split freeze mismatch: {name}")


def _authorize() -> None:
    if os.environ.get(AUTH_ENV) != "1":
        raise PermissionError(f"{AUTH_ENV}=1 is required")
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for phase, roles in (
        ("online_trajectory", ("solver", "reflection")),
        ("frozen_validation", ("solver",)),
    ):
        for role in roles:
            require_api_authorization(
                manifest,
                phase=phase,
                role=role,
                explicit_user_authorized=True,
            )


def _arm_a_selection(
    snapshot: FrozenResponsibilitySnapshot,
    *,
    update_index: int,
    cursor: Mapping[str, int],
) -> tuple[tuple[int, ...], dict[int, str], dict[str, int], tuple[dict[str, Any], ...]]:
    decision = select_vote_aligned_targets(
        assigned=snapshot.assigned,
        current_margin_by_question=snapshot.current_margin_by_question,
        seed=SEED,
        update_index=update_index,
        cursor_before=cursor,
        target_count=2,
    )
    chosen = list(decision.targets)
    slot_rows = list(decision.slot_decisions)
    if len(chosen) < 2:
        start = (SEED + 2 * update_index) % 5
        for offset in range(5):
            candidate = (start + offset) % 5
            if candidate in chosen:
                continue
            chosen.append(candidate)
            slot_rows.append({
                "slot": len(chosen),
                "lane_selected": FALLBACK_RR,
                "candidate_member_ids": [candidate],
                "selected_target_id": candidate,
                "fallback_reason": "always_two_targets_zero_score_fill",
            })
            if len(chosen) == 2:
                break
    lane_by_member = {
        int(row["selected_target_id"]): (
            "coverage" if row["lane_selected"] == PURE_COVERAGE
            else "fallback" if row["lane_selected"] == FALLBACK_RR
            else str(row["lane_selected"])
        )
        for row in slot_rows
    }
    return tuple(chosen), lane_by_member, decision.cursor_after, tuple(slot_rows)


async def _initialize_system(
    *,
    arm: str,
    root: Path,
    optimize_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    optimize_path: Path,
    validation_path: Path,
    cache: dict[str, str],
    ledger: DurableLedger,
) -> Seed78System:
    cfg = _config(
        root,
        optimize_path=optimize_path,
        validation_path=validation_path,
    )
    system = Seed78System(cfg, arm=arm, ledger=ledger, raw_cache=cache)
    system.set_run_identity(
        build_run_identity(
            cfg,
            train_rows=optimize_rows,
            val_rows=validation_rows,
            test_rows=[],
            workspace=ROOT,
        )
    )
    system.set_stage({
        "phase": "initialization",
        "update_index": -1,
        "target_member": -1,
        "candidate_id": "P0",
    })
    try:
        await system.initialize_fixed_probe(optimize_rows)
    finally:
        system.set_stage(None)
    return system


def _profile_identity(system: Seed78System) -> dict[str, Any]:
    return {
        "team_hash": system.team_prompt_state_hash(),
        "prompt_hashes": [system.prompt_hash(row.current_prompt) for row in system.agents],
        "profile_hash": hashlib.sha256(
            json.dumps(
                [
                    [(answer.answer, answer.valid, answer.response_hash) for answer in profile]
                    for profile in system.active_profiles
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def _public_outcome(outcome: Any, update_index: int, scheduler: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "update_index": update_index,
        "selected_target_ids": list(outcome.audit_metadata["selected_target_ids"]),
        "committed_member_id": outcome.audit_metadata["committed_member_id"],
        "committed_candidate_id": outcome.committed_candidate_id,
        "termination_reason": outcome.termination_reason,
        "funnel": dict(outcome.funnel),
        "cost": asdict(outcome.cost),
        "scheduler": dict(scheduler),
    }


async def _run_arm(
    *,
    arm: str,
    system: Seed78System,
    root: Path,
    shadow_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    task_builder = LocalTaskBuilder()
    snapshot_holder: dict[str, FrozenResponsibilitySnapshot] = {}
    assignment_factory = SystemResponsibilityAssignmentFactory(
        system=system,
        snapshot_reader=lambda: snapshot_holder["value"],
        task_builder=task_builder,
    )
    local_solver = SystemLocalSolverEvaluator(
        system=system,
        loop=loop,
        stage=system.set_stage,
        accounting=system.common.accounting,
        solver_contract_id=COMMON_SOLVER_CONTRACT_ID,
        output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION,
    )
    official = GEPALocalPromptOptimizer(
        evaluator=local_solver,
        reflection_lm=ReflectionLM(system),
        accounting_reader=system.optimizer_accounting,
        run_root=root / "local_gepa",
    )
    contextual = ContextualOptimizer(official, local_solver)
    current_update = {"value": -1}
    shadow_probe = system.build_probe(shadow_rows)
    evaluator = SystemTeamCandidateEvaluator(
        system=system,
        shadow_probe=shadow_probe,
        loop=loop,
        stage=system.set_stage,
        accounting=system.common.accounting,
        update_index_reader=lambda: current_update["value"],
    )
    committer = SystemTeamCommitter(
        system=system,
        evaluator=evaluator,
        update_index_reader=lambda: current_update["value"],
    )
    controller = TeamSearchController(
        responsibility=assignment_factory,
        task_builder=task_builder,
        local_optimizer=contextual,
        evaluator=evaluator,
        selector=CommonSafeTeamCandidateSelector(),
        committer=committer,
    )
    b_scheduler = PrimaryResponsibilityPersistentRealizabilityScheduler()
    b_binding = PrimaryResponsibilityOnlineBinding(
        scheduler=b_scheduler,
        assignment_factory=assignment_factory,
        controller=controller,
    )
    cursor: dict[str, int] = {}
    events: list[dict[str, Any]] = []
    no_commit_streak = 0
    for update_index in range(MAX_OPPORTUNITIES):
        current_update["value"] = update_index
        snapshot = freeze_current_responsibility(system, update_index=update_index)
        snapshot_holder["value"] = snapshot
        request = TeamSearchRequest(
            seed=SEED,
            update_index=update_index,
            team_state_hash=system.team_prompt_state_hash(),
            local_metric_budget=LOCAL_GEPA_METRIC_BUDGET,
            solver_contract_id=COMMON_SOLVER_CONTRACT_ID,
            output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION,
        )
        if arm == ARM_B:
            wrapped = await b_binding.run_opportunity(
                request,
                assigned=snapshot.assigned,
                current_margin_by_question=snapshot.current_margin_by_question,
            )
            outcome = wrapped.team_outcome
            scheduler_payload = {
                "version": wrapped.decision.scheduler_version,
                "summaries": [asdict(row) for row in wrapped.decision.summaries],
                "realizability_transitions": [
                    asdict(row) for row in wrapped.realizability_transitions
                ],
            }
        else:
            targets, lanes, cursor, slot_rows = _arm_a_selection(
                snapshot, update_index=update_index, cursor=cursor
            )
            assignments = tuple(
                assignment_factory.build_from_member(
                    request=request,
                    member_id=target,
                    primary_lane=lanes[target],
                    responsibility_identity="vote_aligned_hierarchical_v1",
                )
                for target in targets
            )
            outcome = await controller.run_frozen_opportunity(request, assignments)
            scheduler_payload = {
                "version": "hierarchical_lane_rr_direct_near_coverage_v1",
                "slot_decisions": list(slot_rows),
            }
        event = _public_outcome(outcome, update_index, scheduler_payload)
        events.append(event)
        no_commit_streak, stop = advance_no_commit_streak(
            no_commit_streak,
            committed=outcome.committed_candidate_id is not None,
        )
        write_json(root / f"update_{update_index:02d}.json", event)
        write_json(
            root / "checkpoint_private.json",
            {
                "arm": arm,
                "completed_updates": len(events),
                "no_commit_streak": no_commit_streak,
                "team_identity": _profile_identity(system),
                "prompts": [row.current_prompt for row in system.agents],
                "scheduler_state": (
                    b_scheduler.state.checkpoint_payload() if arm == ARM_B else {"cursor": cursor}
                ),
            },
        )
        if stop:
            break
    result = {
        "arm": arm,
        "termination": (
            "completed_by_early_stop" if no_commit_streak >= NO_COMMIT_PATIENCE
            else "completed_by_budget"
        ),
        "opportunities": len(events),
        "commits": sum(row["committed_candidate_id"] is not None for row in events),
        "events": events,
        "shadow_events": evaluator.shadow_events,
        "final_team_identity": _profile_identity(system),
        "scheduler_telemetry": (
            b_scheduler.telemetry_summary() if arm == ARM_B else {"cursor": cursor}
        ),
    }
    write_json(root / "trajectory_summary_private.json", result)
    return result


def _metrics_payload(system: Seed78System, metrics: Any) -> dict[str, Any]:
    g_hist = {str(value): 0 for value in range(6)}
    for row in metrics.rows:
        g_hist[str(row.gold_vote_count)] += 1
    oracle = sum(int(row.gold_vote_count > 0) for row in metrics.rows)
    return {
        "row_count": len(metrics.rows),
        "vote_correct": metrics.vote_correct_count,
        "vote_accuracy": metrics.plurality_vote_acc,
        "mean_member_accuracy": metrics.mean_individual_acc,
        "vote_minus_mean_member": metrics.plurality_vote_acc - metrics.mean_individual_acc,
        "oracle_correct": oracle,
        "oracle_accuracy": oracle / len(metrics.rows),
        "per_member_accuracy": list(metrics.per_agent_acc),
        "g_histogram": g_hist,
        "invalid_rate": metrics.mean_invalid_rate,
        "team_identity": _profile_identity(system),
    }


def _bind_paired_validation_cache(
    systems: Mapping[str, Seed78System],
) -> dict[str, str]:
    """Bind every arm to one exact-request realization cache before validation."""
    shared: dict[str, str] = {}
    for system in systems.values():
        system.common.cache = shared
    return shared


async def execute(prep: Path, run_root: Path) -> dict[str, Any]:
    _authorize()
    _verify_freeze(prep)
    if run_root.exists():
        raise FileExistsError("fresh run root required; resume/retry is forbidden")
    run_root.mkdir(parents=True)
    optimize_rows = _rows(prep / "splits_private/optimize100.csv")
    shadow_rows = _rows(prep / "splits_private/fold_c.csv")
    validation_rows = _rows(prep / "splits_private/validation.csv")
    optimize_path = prep / "splits_private/optimize100.csv"
    validation_path = prep / "splits_private/validation.csv"
    init_ledger = DurableLedger(run_root / "initialization/ledger.jsonl")
    init_cache: dict[str, str] = {}
    initial = await _initialize_system(
        arm="INITIALIZATION",
        root=run_root / "initialization",
        optimize_rows=optimize_rows,
        validation_rows=validation_rows,
        optimize_path=optimize_path,
        validation_path=validation_path,
        cache=init_cache,
        ledger=init_ledger,
    )
    initial_identity = _profile_identity(initial)
    systems: dict[str, Seed78System] = {}
    trajectories: dict[str, Any] = {}
    for arm in ARMS:
        cell = run_root / arm
        cell.mkdir()
        ledger = DurableLedger(cell / "ledger.jsonl")
        systems[arm] = await _initialize_system(
            arm=arm,
            root=cell,
            optimize_rows=optimize_rows,
            validation_rows=validation_rows,
            optimize_path=optimize_path,
            validation_path=validation_path,
            cache=dict(init_cache),
            ledger=ledger,
        )
        if _profile_identity(systems[arm]) != initial_identity:
            raise RuntimeError("paired arm initialization mismatch")
    for arm in ARMS:
        trajectories[arm] = await _run_arm(
            arm=arm,
            system=systems[arm],
            root=run_root / arm,
            shadow_rows=shadow_rows,
        )
    if any(row["termination"] not in {"completed_by_early_stop", "completed_by_budget"} for row in trajectories.values()):
        raise RuntimeError("both trajectories must freeze before Validation50")
    _bind_paired_validation_cache(systems)
    validations: dict[str, Any] = {}
    for arm in ARMS:
        system = systems[arm]
        system.set_stage({
            "phase": "final_validation",
            "update_index": -1,
            "target_member": -1,
            "candidate_id": "final_team",
        })
        try:
            metrics = await system.evaluate_dataset(validation_rows)
        finally:
            system.set_stage(None)
        validations[arm] = _metrics_payload(system, metrics)
        write_json(run_root / arm / "validation50_private.json", validations[arm])
    summary = {
        "execution_gate": "PASS",
        "seed": SEED,
        "arms": list(ARMS),
        "initial_identity": initial_identity,
        "trajectories": trajectories,
        "validation50": validations,
        "validation_evaluated_after_both_frozen": True,
        "test50_calls": 0,
    }
    write_json(run_root / "execution_summary_private.json", summary)
    return summary


def _ledger_summary(path: Path) -> dict[str, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    completed = [
        row for row in rows
        if row.get("record_kind") != "solver_provider_attempt_failure"
    ]
    return {
        "logical_calls": len(completed),
        "provider_attempts": sum(int(row["provider_attempts"]) for row in rows),
        "successful_provider_calls": sum(int(row["successful_provider_calls"]) for row in rows),
        "failed_provider_attempts": sum(
            int(row["provider_attempts"]) - int(row["successful_provider_calls"])
            for row in rows
        ),
        "cache_hits": sum(bool(row["cache_hit"]) for row in rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in rows),
    }


def _mechanism_metrics(trajectory: Mapping[str, Any], ledger: Mapping[str, int]) -> dict[str, Any]:
    target_counts = {str(member): 0 for member in range(5)}
    commit_counts = {str(member): 0 for member in range(5)}
    current_streak = {str(member): 0 for member in range(5)}
    max_streak = {str(member): 0 for member in range(5)}
    for event in trajectory["events"]:
        selected = {str(member) for member in event["selected_target_ids"]}
        committed = (
            None if event["committed_member_id"] is None
            else str(event["committed_member_id"])
        )
        for member in selected:
            target_counts[member] += 1
            if member == committed:
                current_streak[member] = 0
                commit_counts[member] += 1
            else:
                current_streak[member] += 1
                max_streak[member] = max(max_streak[member], current_streak[member])
    total_targets = sum(target_counts.values())
    total_commits = sum(commit_counts.values())
    mismatch = None
    if total_targets and total_commits:
        mismatch = 0.5 * sum(
            abs(target_counts[str(member)] / total_targets - commit_counts[str(member)] / total_commits)
            for member in range(5)
        )
    return {
        "target_count_by_member": target_counts,
        "commit_count_by_member": commit_counts,
        "target_to_commit_rate_by_member": {
            member: commit_counts[member] / count if count else None
            for member, count in target_counts.items()
        },
        "target_commit_distribution_mismatch": mismatch,
        "max_unresolved_target_streak_by_member": max_streak,
        "max_unresolved_target_streak": max(max_streak.values()),
        "tokens_per_commit": (
            ledger["total_tokens"] / total_commits if total_commits else None
        ),
    }


def _classify(public_arms: Mapping[str, Any]) -> str:
    a = public_arms[ARM_A]
    b = public_arms[ARM_B]
    required = (
        a["target_commit_distribution_mismatch"],
        b["target_commit_distribution_mismatch"],
        a["tokens_per_commit"],
        b["tokens_per_commit"],
    )
    if any(value is None for value in required):
        return "HOLD"
    mismatch_improved = (
        b["target_commit_distribution_mismatch"]
        < a["target_commit_distribution_mismatch"]
    )
    efficiency_improved = b["tokens_per_commit"] < a["tokens_per_commit"]
    vote_preserved = (
        b["validation50"]["vote_accuracy"]
        >= a["validation50"]["vote_accuracy"]
    )
    if mismatch_improved and efficiency_improved and vote_preserved:
        return "PRIMARY_RESPONSIBILITY_SCHEDULER_SUPPORTED"
    if mismatch_improved or efficiency_improved:
        return "MECHANISM_SIGNAL_WITHOUT_FULL_QUALITY_SUPPORT"
    return "NO_PRIMARY_RESPONSIBILITY_SCHEDULER_SIGNAL"


def audit(prep: Path, run_root: Path) -> dict[str, Any]:
    _verify_freeze(prep)
    errors: list[str] = []
    if not run_root.exists():
        return {"gate": "NOT_RUN", "errors": ["run_root_missing"]}
    summary = _read_json(run_root / "execution_summary_private.json")
    if summary.get("test50_calls") != 0:
        errors.append("test50_access")
    if not summary.get("validation_evaluated_after_both_frozen"):
        errors.append("validation_timing")
    if tuple(summary.get("arms", ())) != ARMS:
        errors.append("arm_identity")
    for arm in ARMS:
        row = summary["trajectories"][arm]
        if row["opportunities"] > MAX_OPPORTUNITIES:
            errors.append(f"budget:{arm}")
        if row["termination"] == "completed_by_early_stop":
            tail = row["events"][-NO_COMMIT_PATIENCE:]
            if len(tail) != NO_COMMIT_PATIENCE or any(event["committed_candidate_id"] for event in tail):
                errors.append(f"early_stop:{arm}")
        if any(len(event["selected_target_ids"]) != 2 for event in row["events"]):
            errors.append(f"target_count:{arm}")
        if any(len(set(event["selected_target_ids"])) != 2 for event in row["events"]):
            errors.append(f"target_distinct:{arm}")
        expected_shadow = sum(
            event["funnel"]["feasible_candidates"] > 0
            for event in row["events"]
        )
        # Common-Safe returns exactly one global winner whenever any feasible
        # candidate exists.  Only that winner may reach the Shadow gate.
        if len(row["shadow_events"]) != expected_shadow:
            errors.append(f"winner_only_shadow:{arm}")
        if len(summary["validation50"][arm].get("g_histogram", {})) != 6:
            errors.append(f"validation_g_histogram:{arm}")
        ledger = _ledger_summary(run_root / arm / "ledger.jsonl")
        if ledger["input_tokens"] + ledger["output_tokens"] != ledger["total_tokens"]:
            errors.append(f"ledger_tokens:{arm}")
    result = {
        "gate": "PASS" if not errors else "HOLD",
        "errors": errors,
        "seed": SEED,
        "trajectories": 2,
        "validation50_evaluations": 2,
        "test50_calls": 0,
        "protocol_sha256": _read_json(prep / "source_freeze.json")["protocol_sha256"],
    }
    write_json(run_root / "audit.json", result)
    return result


def analyze(prep: Path, run_root: Path, report: Path) -> dict[str, Any]:
    gate = audit(prep, run_root)
    if gate["gate"] != "PASS":
        raise RuntimeError("official audit must PASS before analysis")
    if report.exists():
        raise FileExistsError("fresh report root required")
    summary = _read_json(run_root / "execution_summary_private.json")
    report.mkdir(parents=True)
    public_arms: dict[str, Any] = {}
    for arm in ARMS:
        trajectory = summary["trajectories"][arm]
        ledger = _ledger_summary(run_root / arm / "ledger.jsonl")
        public_arms[arm] = {
            "termination": trajectory["termination"],
            "opportunities": trajectory["opportunities"],
            "commits": trajectory["commits"],
            **_mechanism_metrics(trajectory, ledger),
            "ledger": ledger,
            "validation50": summary["validation50"][arm],
        }
    delta = {
        key: public_arms[ARM_B]["validation50"][key] - public_arms[ARM_A]["validation50"][key]
        for key in ("vote_accuracy", "mean_member_accuracy", "vote_minus_mean_member", "oracle_accuracy")
    }
    result = {
        "experiment_id": EXPERIMENT_ID,
        "seed": SEED,
        "evidence_type": "single_seed_prospective_mechanism_pilot",
        "arms": public_arms,
        "B_minus_A_validation50": delta,
        "classifier": _classify(public_arms),
        "shadow50_role": "optimization_time_safety_data_not_generalization",
        "validation50_role": "final_only_external_generalization_endpoint",
        "test50_calls": 0,
    }
    write_json(report / "summary.json", result)
    write_json(report / "audit.json", gate)
    write_json(report / "protocol_freeze.json", _read_json(prep / "protocol_freeze.json"))
    (report / "README.md").write_text(
        "# Seed78 Primary-Responsibility Scheduler A/B\n\n"
        "This report is generated only after the frozen execution and official audit. "
        "Shadow50 is optimization-time safety data; Validation50 is the final-only "
        "external endpoint; Test50 has zero calls. See `summary.json` for results.\n",
        encoding="utf-8",
    )
    return result


def preflight() -> dict[str, Any]:
    verify_frozen_gepa_engine_contract()
    protocol = protocol_document()
    checks = {
        "seed78": protocol["seed"] == 78,
        "seed75_split_reused": protocol["split_identity"]["optimize100"].endswith("fold_a+fold_b"),
        "shadow_is_optimization_data": protocol["data_roles"]["shadow50"].startswith("optimization_time"),
        "validation_final_only": "after_both_training_trajectories_freeze" in protocol["data_roles"]["validation50"],
        "test_zero": protocol["split_identity"]["test50"] == "blocked_zero_calls",
        "only_scheduler_differs": protocol["only_difference"] == "target_scheduler",
        "always_two_targets": protocol["shared"]["always_two_targets"] is True,
        "official_gepa": protocol["shared"]["official_gepa"] == asdict(GEPAOptimizerConfig()),
        "local_gepa_budget_arithmetic": protocol["shared"]["local_gepa_budget_capacity"] == {
            "metric_budget": 36,
            "validation_size": 12,
            "reflection_minibatch_size": 3,
            "seed_evaluation_calls": 12,
            "proposal_attempt_calls": 6,
            "accepted_full_evaluation_calls": 12,
            "max_rejected_proposals": 4,
            "max_accepted_children": 1,
            "max_accepted_generations": 1,
        },
        "budget": protocol["shared"]["max_opportunities"] == 32,
        "early_stop": protocol["shared"]["no_commit_patience"] == 6,
    }
    return {
        "gate": "PASS" if all(checks.values()) else "HOLD",
        "checks": checks,
        "protocol_sha256": sha256_json(protocol),
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.prep)
    elif args.preflight:
        result = preflight()
    elif args.execute:
        result = asyncio.run(execute(args.prep, args.run))
    elif args.audit:
        result = audit(args.prep, args.run)
    else:
        result = analyze(args.prep, args.run, args.report)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
