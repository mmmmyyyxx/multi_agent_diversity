"""Backend-neutral runtime bridge for production local prompt optimization.

This module contains execution mechanics shared by current Layer-2 runners.
It deliberately owns no target allocation, responsibility, GEPA search,
team-level acceptance, or write-back semantics.
"""

from __future__ import annotations

import asyncio
import contextvars
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

from openai import APIConnectionError

from infrastructure.common_solver_contract_v1.contract import (
    CONTRACT_SPEC,
    canonical_json_bytes,
)
from infrastructure.common_solver_contract_v1.evaluator import (
    CommonSolverEvaluator,
    TransportResponse,
)
from ..config import Config
from ..evaluation.prompt_question import PromptAnswer
from ..evaluation.solver_stage import validate_solver_stage_attribution
from ..provider_factory import ProviderClientFactory
from ..system import PromptEnsembleOptimizationSystem
from ..local_optimizers.base import LocalPromptOptimizer
from ..local_optimizers.schemas import LocalOptimizationResult, LocalOptimizationTask


class LocalOptimizerConfigurationError(RuntimeError):
    """The current execution contract cannot safely enter Layer 1."""


@dataclass(frozen=True)
class LocalOptimizerExecutionContext:
    """Frozen execution facts required at the Layer-2 -> Layer-1 boundary."""

    run_seed: int
    provider_profile: str
    solver_model: str
    optimizer_model: str
    evaluator_model: str
    run_identity_sha256: str
    local_no_update_patience: int
    team_no_update_patience: int
    saturation_mode: str
    arm: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_seed, int) or self.run_seed < 0:
            raise LocalOptimizerConfigurationError("local optimizer run_seed is required")
        for name in (
            "provider_profile",
            "solver_model",
            "optimizer_model",
            "evaluator_model",
            "run_identity_sha256",
            "saturation_mode",
            "arm",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise LocalOptimizerConfigurationError(
                    f"local optimizer {name} is required"
                )
        if len(self.run_identity_sha256) != 64:
            raise LocalOptimizerConfigurationError(
                "local optimizer run_identity_sha256 must be a SHA-256 digest"
            )
        if self.local_no_update_patience <= 0 or self.team_no_update_patience <= 0:
            raise LocalOptimizerConfigurationError(
                "local optimizer saturation patience must be positive"
            )


LOCAL_OPTIMIZER_INVOCATION: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("local_optimizer_invocation", default=None)
)


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
            "record_id", "phase", "logical_role", "client_role",
            "provider_attempts", "successful_provider_calls", "cache_hit",
            "input_tokens", "output_tokens", "total_tokens", "seed", "arm",
            "update_index", "target_member",
        }
        if required - set(row):
            raise ValueError("incomplete execution ledger record")
        if int(row["input_tokens"]) + int(row["output_tokens"]) != int(
            row["total_tokens"]
        ):
            raise ValueError("execution ledger token arithmetic mismatch")
        identity = str(row["record_id"])
        if identity in self.identities:
            raise ValueError("duplicate execution ledger record identity")
        self.identities.add(identity)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())


class CappedDurableLedger(DurableLedger):
    """Pre-provider, cross-role attempt/success reservation for a fresh run.

    In-flight calls reserve a potential success slot.  A failed-attempt row
    releases it; a successful row consumes it.  This prevents concurrent
    Solver requests from overshooting a global emergency ceiling.
    """

    def __init__(self, path: Path, *, successful_ceiling: int, attempt_ceiling: int) -> None:
        super().__init__(path)
        if successful_ceiling <= 0 or attempt_ceiling < successful_ceiling:
            raise ValueError("invalid provider emergency ceilings")
        self.successful_ceiling = successful_ceiling
        self.attempt_ceiling = attempt_ceiling
        self._lock = Lock()
        self._reserved_attempts = 0
        self._inflight = 0
        self._successful = 0

    def reserve_provider_attempt(self, role: str) -> None:
        if role not in {"solver", "reflection"}:
            raise ValueError("unrecognized provider role at emergency boundary")
        with self._lock:
            if self._reserved_attempts >= self.attempt_ceiling:
                raise RuntimeError("transport_attempt_emergency_ceiling")
            if self._successful + self._inflight >= self.successful_ceiling:
                raise RuntimeError("successful_provider_emergency_ceiling")
            self._reserved_attempts += 1
            self._inflight += 1

    def append(self, row: Mapping[str, Any]) -> None:
        attempts = int(row.get("provider_attempts", 0))
        successful = int(row.get("successful_provider_calls", 0))
        with self._lock:
            if attempts and (attempts != 1 or successful not in {0, 1} or self._inflight <= 0):
                raise RuntimeError("provider accounting/reservation mismatch")
            super().append(row)
            self._inflight -= attempts
            self._successful += successful


class CommonContractExecutionSystem(PromptEnsembleOptimizationSystem):
    """Five-member execution system using COMMON_SOLVER_CONTRACT_V1."""

    def __init__(
        self,
        cfg: Config,
        *,
        arm: str,
        ledger: DurableLedger,
        raw_cache: dict[str, str],
    ) -> None:
        client = ProviderClientFactory.from_environment(
            provider_profile=cfg.models.provider_profile,
            api_key_env=cfg.models.solver_api_key_env,
            base_url_env=cfg.models.solver_base_url_env,
        )

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
                persist_failed_attempt(
                    {
                        "request_identity": request_identity,
                        "attempt_index": attempt_index,
                        "error_type": type(exc).__name__,
                        "status_code": getattr(exc, "status_code", None),
                    }
                )
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
            self.ledger.append(
                {
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
                }
            )

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
            self.ledger.append(
                {
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
                }
            )
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
    """Synchronous GEPA reflection bridge with durable role attribution."""

    def __init__(self, system: CommonContractExecutionSystem) -> None:
        self.system = system
        self.sequence = 0

    @staticmethod
    def _messages(prompt: str | list[dict[str, Any]]) -> tuple[str, str]:
        if isinstance(prompt, str):
            return (
                "Return only one complete replacement reasoning procedure. "
                "Do not discuss output formatting.",
                prompt,
            )
        systems = [
            str(row.get("content", ""))
            for row in prompt
            if row.get("role") == "system"
        ]
        others = [
            f"{row.get('role', 'user')}: {row.get('content', '')}"
            for row in prompt
            if row.get("role") != "system"
        ]
        return (
            "\n".join(systems) or "You improve a reasoning procedure.",
            "\n\n".join(others),
        )

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        context = LOCAL_OPTIMIZER_INVOCATION.get()
        if context is None:
            raise LocalOptimizerConfigurationError(
                "reflection call lacks local optimizer execution context"
            )
        system_prompt, user_prompt = self._messages(prompt)

        async def run() -> str:
            begin = len(self.system.llm.calls)
            try:
                result = await self.system.llm.chat_result(
                    str(context["optimizer_model"]),
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
                    self.system.ledger.append(
                        {
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
                            "seed": int(context["run_seed"]),
                            "arm": self.system.arm,
                            "update_index": int(context["update_index"]),
                            "target_member": int(context["target_member"]),
                            "candidate_id": "reflection",
                        }
                    )

        return asyncio.run_coroutine_threadsafe(run(), context["loop"]).result()


def _validate_entry(
    task: LocalOptimizationTask,
    runtime: LocalOptimizerExecutionContext,
) -> None:
    if not isinstance(task, LocalOptimizationTask):
        raise LocalOptimizerConfigurationError(
            "ResponsibilityEvidencePacket-derived LocalOptimizationTask is required"
        )
    if not task.parent_prompt:
        raise LocalOptimizerConfigurationError("candidate parent prompt is required")
    if task.run_seed is None or task.run_seed != runtime.run_seed:
        raise LocalOptimizerConfigurationError(
            "local optimizer run seed is missing or mismatched"
        )
    if task.update_index is None or task.update_index < 0:
        raise LocalOptimizerConfigurationError("local optimizer update_index is required")
    if task.target_member is None or not 0 <= task.target_member < 5:
        raise LocalOptimizerConfigurationError("local optimizer target_member is required")


class ContextualLocalPromptOptimizer(LocalPromptOptimizer):
    """Bind typed execution provenance around an otherwise unchanged backend."""

    def __init__(
        self,
        inner: LocalPromptOptimizer,
        local_solver: Any,
        runtime: LocalOptimizerExecutionContext,
    ) -> None:
        self.inner = inner
        self.local_solver = local_solver
        self.runtime = runtime

    async def optimize(self, task: LocalOptimizationTask) -> LocalOptimizationResult:
        _validate_entry(task, self.runtime)
        context = {
            "loop": asyncio.get_running_loop(),
            "run_seed": self.runtime.run_seed,
            "update_index": int(task.update_index),
            "target_member": int(task.target_member),
            "phase": "local_optimizer_solver_eval",
            "parent_id": (
                f"seed{self.runtime.run_seed}_update{task.update_index}"
            ),
            "parent_prompt_sha256": hashlib.sha256(
                task.parent_prompt.encode("utf-8")
            ).hexdigest(),
            "provider_profile": self.runtime.provider_profile,
            "solver_model": self.runtime.solver_model,
            "optimizer_model": self.runtime.optimizer_model,
            "evaluator_model": self.runtime.evaluator_model,
            "run_identity_sha256": self.runtime.run_identity_sha256,
            "local_no_update_patience": self.runtime.local_no_update_patience,
            "team_no_update_patience": self.runtime.team_no_update_patience,
            "saturation_mode": self.runtime.saturation_mode,
            "arm": self.runtime.arm,
        }
        token = LOCAL_OPTIMIZER_INVOCATION.set(context)
        self.local_solver.task_context = context
        try:
            return await self.inner.optimize(task)
        finally:
            self.local_solver.task_context = None
            LOCAL_OPTIMIZER_INVOCATION.reset(token)


def execution_context_from_system(
    system: CommonContractExecutionSystem,
    *,
    run_identity_sha256: str | None = None,
    local_no_update_patience: int,
    team_no_update_patience: int,
    saturation_mode: str,
) -> LocalOptimizerExecutionContext:
    """Build the boundary contract from current structured configuration only."""

    if system.run_identity is None:
        raise LocalOptimizerConfigurationError(
            "system run identity must be frozen before local optimization"
        )
    identity_sha = run_identity_sha256 or hashlib.sha256(
        json.dumps(
            system.run_identity.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    cfg = system.cfg
    return LocalOptimizerExecutionContext(
        run_seed=cfg.training.seed,
        provider_profile=cfg.models.provider_profile,
        solver_model=cfg.models.agent_model,
        optimizer_model=cfg.models.optimizer_model,
        evaluator_model=cfg.models.evaluator_model,
        run_identity_sha256=identity_sha,
        local_no_update_patience=local_no_update_patience,
        team_no_update_patience=team_no_update_patience,
        saturation_mode=saturation_mode,
        arm=system.arm,
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def profile_identity(system: CommonContractExecutionSystem) -> dict[str, Any]:
    return {
        "team_hash": system.team_prompt_state_hash(),
        "prompt_hashes": [
            system.prompt_hash(row.current_prompt) for row in system.agents
        ],
        "profile_hash": hashlib.sha256(
            json.dumps(
                [
                    [
                        (answer.answer, answer.valid, answer.response_hash)
                        for answer in profile
                    ]
                    for profile in system.active_profiles
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def ledger_summary(path: Path) -> dict[str, int]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    completed = [
        row
        for row in rows
        if row.get("record_kind") != "solver_provider_attempt_failure"
    ]
    return {
        "logical_calls": len(completed),
        "provider_attempts": sum(int(row["provider_attempts"]) for row in rows),
        "successful_provider_calls": sum(
            int(row["successful_provider_calls"]) for row in rows
        ),
        "failed_provider_attempts": sum(
            int(row["provider_attempts"]) - int(row["successful_provider_calls"])
            for row in rows
        ),
        "cache_hits": sum(bool(row["cache_hit"]) for row in rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in rows),
    }
