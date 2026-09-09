"""Authorized, non-committing RG-GEPA fixed-parent pilot runner.

This runner is deliberately separate from the canonical v15 trajectory.  It
never mutates a historical team, never evaluates validation/Test50, and writes
raw prompts/questions/responses only below the ignored ``runs/`` root.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infrastructure.common_solver_contract_v1.contract import (  # noqa: E402
    COMMON_SOLVER_CONTRACT_ID, CONTRACT_SPEC, contract_identity,
)
from infrastructure.common_solver_contract_v1.evaluator import (  # noqa: E402
    CommonSolverEvaluator, TransportResponse,
)
from infrastructure.common_solver_contract_v1.system_adapter import (  # noqa: E402
    CommonContractSolverAdapter,
)
from multi_dataset_diverse_rl.candidate_selection import (  # noqa: E402
    common_monotone_safe_key, evaluate_constraints,
)
from multi_dataset_diverse_rl.config import Config  # noqa: E402
from multi_dataset_diverse_rl.evaluation.fixed_probe import (  # noqa: E402
    evaluate_candidate_profile, subset_profiles,
)
from multi_dataset_diverse_rl.evaluation.mutable_prompt_contract import (  # noqa: E402
    validate_mutable_decision_procedure,
)
from multi_dataset_diverse_rl.experimental_rg_gepa import (  # noqa: E402
    CandidateScore, RGGEPAProtocol, RG_GEPA_LEDGER_VERSION, TeamVector, current_selector_winner,
    progressive_promotions, team_pareto_winner, validate_ledger_record,
)
from multi_dataset_diverse_rl.provider_credentials import (  # noqa: E402
    resolve_api_key, resolve_base_url,
)
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem  # noqa: E402
from multi_dataset_diverse_rl.versions import COMMON_SOLVER_CONTRACT_V1_ID  # noqa: E402


AUTH_ENV = "RG_GEPA_FIXED_PARENT_PILOT_AUTHORIZED"
DEFAULT_PREP = ROOT / "runs" / "responsibility_guided_gepa_fixed_parent_pilot_v1_prep_20260909"
DEFAULT_RUN = ROOT / "runs" / "responsibility_guided_gepa_fixed_parent_pilot_v1_20260909"
DEFAULT_REPORT = ROOT / "reports" / "responsibility_guided_gepa_fixed_parent_pilot_v1_execution"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class RGGEPAExecutionLedger:
    """Single append-only durable writer for every RG-GEPA accounting path."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError("RG-GEPA durable ledger must be fresh")
        self.rows: list[dict[str, Any]] = []
        self._attempt_ids: set[str] = set()

    def append(self, record: dict[str, Any]) -> None:
        validate_ledger_record(record)
        attempt_id = str(record["provider_attempt_id"])
        if attempt_id in self._attempt_ids:
            raise ValueError("duplicate RG-GEPA provider-attempt identity")
        self.rows.append(dict(record))
        self._attempt_ids.add(attempt_id)
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            self.rows.pop()
            self._attempt_ids.remove(attempt_id)
            raise


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def _source_hashes() -> list[dict[str, str]]:
    paths = [
        Path("multi_dataset_diverse_rl/experimental_rg_gepa.py"),
        Path("scripts/prepare_responsibility_guided_gepa_fixed_parent_pilot.py"),
        Path("scripts/run_responsibility_guided_gepa_fixed_parent_pilot.py"),
        Path("infrastructure/common_solver_contract_v1/contract.py"),
        Path("infrastructure/common_solver_contract_v1/evaluator.py"),
        Path("infrastructure/common_solver_contract_v1/system_adapter.py"),
    ]
    return [{"path": path.as_posix(), "sha256": _sha((ROOT / path).read_bytes())} for path in paths]


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None and getattr(exc, "response", None) is not None:
        status = getattr(exc.response, "status_code", None)
    return (int(status) in CONTRACT_SPEC.retry_status_codes if status is not None
            else isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError)))


class _CommonSystem(PromptEnsembleOptimizationSystem):
    """Canonical Solver boundary, with a run-wide exact request cache."""

    raw_cache: dict[str, str] = {}

    def __init__(self, cfg: Config, *, ledger_writer: RGGEPAExecutionLedger) -> None:
        _, key = resolve_api_key(cfg.models.solver_api_key_env)
        _, endpoint = resolve_base_url(cfg.models.solver_base_url_env)
        if not key or not endpoint:
            raise RuntimeError("COMMON_SOLVER_CONTRACT_V1 provider credentials unavailable")
        client = AsyncOpenAI(api_key=key, base_url=endpoint)

        async def transport(request: dict[str, Any]) -> TransportResponse:
            response = await client.chat.completions.create(**request, timeout=CONTRACT_SPEC.timeout_seconds)
            usage = response.usage
            return TransportResponse(
                text=response.choices[0].message.content or "",
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                finish_reason=str(response.choices[0].finish_reason or ""),
            )

        self.common = CommonSolverEvaluator(transport=transport, cache=type(self).raw_cache, retryable=_retryable)
        self.common_adapter = CommonContractSolverAdapter(self.common)
        self._ledger_writer = ledger_writer
        self._ledger_meta: dict[str, Any] | None = None
        self._ledger_sequence = 0
        super().__init__(cfg, solver=self.common_adapter.solve)

    def set_solver_stage(self, meta: dict[str, Any] | None) -> None:
        """Attach immutable stage/candidate attribution before Solver work."""
        self._ledger_meta = dict(meta) if meta is not None else None

    def _persist_solver_ledger(self, raw: dict[str, Any]) -> None:
        if self._ledger_meta is None:
            raise RuntimeError("RG-GEPA Solver call occurred without ledger stage attribution")
        self._ledger_sequence += 1
        meta = self._ledger_meta
        record = {
            "ledger_version": RG_GEPA_LEDGER_VERSION,
            "seed": meta["seed"], "parent_id": meta["parent_id"], "update_index": meta["update_index"],
            "candidate_id": meta["candidate_id"], "proposal_engine": meta["proposal_engine"],
            "evaluation_stage": meta["evaluation_stage"], "input_tokens": int(raw.get("prompt_tokens", 0)),
            "output_tokens": int(raw.get("completion_tokens", 0)), "total_tokens": int(raw.get("total_tokens", 0)),
            "provider_attempt_id": f"{meta['parent_id']}:{meta['candidate_id']}:{meta['evaluation_stage']}:solver:{self._ledger_sequence}",
            "logical_call_id": f"{meta['parent_id']}:{meta['candidate_id']}:{meta['evaluation_stage']}:solver:{self._ledger_sequence}",
            "attempt_index": 0,
            "record_kind": "solver_logical_invocation",
            "provider_attempts": int(raw.get("common_transport_attempts", 0)),
            "successful_provider_calls": int(not raw.get("common_cache_hit", False) and bool(raw.get("success", False))),
            "cache_hit": bool(raw.get("common_cache_hit", False)), "logical_role": "solver", "client_role": "solver",
            "success": bool(raw.get("success", False)),
        }
        self._ledger_writer.append(record)

    async def solve(self, question: str, agent_id: int, prompt: str):
        answer = await super().solve(question, agent_id, prompt)
        row = self.llm.calls[-1]
        row.update({
            "prompt_tokens": answer.prompt_tokens,
            "completion_tokens": answer.completion_tokens,
            "total_tokens": answer.total_tokens,
            "common_solver_contract": COMMON_SOLVER_CONTRACT_ID,
            "common_request_identity": answer.request_identity,
            "common_transport_attempts": self.common_adapter.last_transport_attempts,
            "common_cache_hit": self.common_adapter.last_cache_hit,
        })
        self._persist_solver_ledger(row)
        return answer


def _cfg(case: dict[str, Any], out: Path) -> Config:
    # The experimental runner uses the existing generic setting only to obtain
    # fixed five-member/plurality parsing semantics; it does not enter its loop.
    return Config.from_flat(
        task_type="bbh", dataset_format="mars", comparison_task_id="disambiguation_qa",
        benchmark="BBH", answer_format="option_letter", train_path="OPTIMIZE100_PRIVATE",
        val_path="EXTERNAL_VALIDATION_BLOCKED", test_path="TEST50_BLOCKED",
        train_size=100, val_size=0, test_size=0, agent_model="qwen3-8b",
        optimizer_model="qwen3.7-flash", evaluator_model="qwen3.7-flash",
        temperature=0.0, solver_max_tokens=1800, solver_invalid_max_retries=0,
        solver_contract_id=COMMON_SOLVER_CONTRACT_V1_ID,
        experiment_setting="experimental_diversity_d2_rr_generic", target_scheduler="round_robin",
        agents=5, seed=int(case["source_seed"]), proposal_memory_mode="off",
        num_candidates_per_parent=2, candidate_eval_pool_size=100, stage_b_candidate_budget=2,
        eval_solver_call_concurrency=8, initialization_mode="provided_prompt_set",
        provided_prompts_json=json.dumps(case["parent_prompts"], ensure_ascii=False),
        out_dir=str(out), shared_solver_cache_path="", final_test_enabled=False,
        provider_call_budget=10000, total_token_budget=20_000_000,
    )


def _vector(evaluation: Any, parent: Any | None = None) -> TeamVector:
    majority = 0
    if parent is not None:
        majority = sum(bool(before and after) for before, after in zip(
            parent.team_outcome.vote_correct_vector, evaluation.team_outcome.vote_correct_vector, strict=True
        ))
    return TeamVector(
        vote_correct=int(evaluation.team_outcome.vote_correct_count),
        target_correct=int(evaluation.competence.correct_count),
        total_member_correct=sum(int(x) for x in evaluation.member_gain.candidate_correct_counts),
        majority_preservation=majority,
        responsibility_recovery=int(evaluation.marginal.assigned_residual_repair_count),
    )


def _score(candidate_id: str, evaluation: Any, parent: Any, hard: bool) -> CandidateScore:
    return CandidateScore(candidate_id, _vector(evaluation, parent), int(evaluation.competence.invalid_count), hard)


def _hard_gate(prompt: str, questions: list[dict[str, str]]) -> tuple[bool, str]:
    try:
        validate_mutable_decision_procedure(prompt)
    except ValueError:
        return False, "output_contract"
    normalized = prompt.casefold()
    # A full question appearing in a mutable rule is an exact, deterministic
    # sample-copy violation; no model judgement is used here.
    if any(str(row["question"]).casefold() in normalized for row in questions):
        return False, "sample_memorization"
    return True, ""


def _reflection_request(case: dict[str, Any], candidate_index: int) -> tuple[str, str]:
    target = int(case["target_member"])
    evidence = case["minibatch_private"][:4]
    cases = "\n\n".join(
        f"Example {index + 1}:\nQuestion: {row['question']}\nGold: {row['answer']}"
        for index, row in enumerate(evidence)
    )
    system = (
        "You revise a general reasoning decision procedure for multiple-choice questions. "
        "Return only one complete replacement decision procedure, with no JSON, commentary, "
        "answer-format instruction, answer-label rule, example lookup, or peer-copying instruction."
    )
    user = (
        f"Current member procedure:\n{case['parent_prompts'][target]}\n\n"
        f"Responsibility lane: {case['responsibility_type']}.\n"
        "Diagnose a systematic limitation, improve a general rule, preserve broad correct behavior, "
        "and do not mention output formatting or final-answer syntax.\n\n"
        f"Optimization evidence (do not memorize):\n{cases}\n\nMutation index: {candidate_index}"
    )
    return system, user


def _persist_new_optimizer_calls(
    system: _CommonSystem,
    begin: int,
    meta: dict[str, Any],
    ledger_writer: RGGEPAExecutionLedger,
) -> None:
    logical_call_id = f"{meta['parent_id']}:{meta['candidate_id']}:{meta['evaluation_stage']}"
    for offset, raw in enumerate(system.llm.calls[begin:], start=begin):
        if raw.get("client_role") != "optimizer":
            continue
        attempt_index = int(raw.get("attempt", offset - begin + 1))
        record = {
            "ledger_version": RG_GEPA_LEDGER_VERSION,
            "seed": meta["seed"], "parent_id": meta["parent_id"], "update_index": meta["update_index"],
            "candidate_id": meta["candidate_id"], "proposal_engine": meta["proposal_engine"],
            "evaluation_stage": meta["evaluation_stage"], "input_tokens": int(raw.get("prompt_tokens", 0)),
            "output_tokens": int(raw.get("completion_tokens", 0)), "total_tokens": int(raw.get("total_tokens", 0)),
            "provider_attempt_id": f"{logical_call_id}:attempt:{attempt_index}",
            "logical_call_id": logical_call_id,
            "attempt_index": attempt_index,
            "record_kind": "optimizer_provider_attempt",
            "provider_attempts": 1,
            "successful_provider_calls": int(bool(raw.get("success", False))),
            "cache_hit": False, "logical_role": raw.get("role", ""),
            "client_role": raw.get("client_role", ""), "success": bool(raw.get("success", False)),
        }
        ledger_writer.append(record)


async def _profile(system: _CommonSystem, target: int, prompt: str, indices: list[int] | None, meta: dict[str, Any]) -> Any:
    assert system.fixed_probe is not None
    system.set_solver_stage(meta)
    try:
        if indices is None:
            return await system.fixed_probe.evaluate_prompt(target, prompt, system.prompt_hash(prompt), system.solve)
        result = await system.fixed_probe.evaluate_prompt_indices(target, prompt, system.prompt_hash(prompt), indices, system.solve)
        return tuple(result[index] for index in indices)
    finally:
        system.set_solver_stage(None)


def _record_minibatch_parent_reuse(case: dict[str, Any], ledger_writer: RGGEPAExecutionLedger) -> None:
    """Account for 12 paired parent lookups explicitly reused from full state."""
    for offset in range(12):
        record = {
            "ledger_version": RG_GEPA_LEDGER_VERSION,
            "seed": case["source_seed"], "parent_id": case["case_id"], "update_index": case["source_update_index"],
            "candidate_id": "parent", "proposal_engine": "current", "evaluation_stage": "minibatch_parent",
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "provider_attempt_id": f"{case['case_id']}:parent:minibatch_parent:cache:{offset}", "cache_hit": True,
            "logical_call_id": f"{case['case_id']}:parent:minibatch_parent:{offset}",
            "attempt_index": 0,
            "record_kind": "cache_reuse",
            "provider_attempts": 0,
            "successful_provider_calls": 0,
            "logical_role": "solver", "client_role": "solver", "success": True,
        }
        ledger_writer.append(record)


def _evaluation(system: _CommonSystem, target: int, prompt: str, profile: Any, indices: list[int] | None, assigned: set[str]) -> Any:
    assert system.fixed_probe is not None
    examples = system.fixed_probe.examples
    active = system.active_profiles
    initial = system.initial_profiles
    if indices is not None:
        examples, active = subset_profiles(examples, active, indices)
        _, initial = subset_profiles(system.fixed_probe.examples, initial, indices)
    return evaluate_candidate_profile(
        prompt=prompt, prompt_hash=system.prompt_hash(prompt), examples=examples,
        active_profiles=active, initial_profiles=initial, candidate_profile=profile,
        target_agent_id=target, assigned_question_hashes=assigned,
        normalize_answer=system.normalize_answer, match_answer=system.match_answer,
        tie_break=system.cfg.peer_state.vote_tie_break, seed=system.cfg.training.seed,
        tau=system.cfg.peer_state.soft_vote_tau,
    )


def _public_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"prompt", "raw_response"}}


def _runtime_accounting(system: _CommonSystem) -> dict[str, int]:
    optimizer = [row for row in system.llm.calls if row.get("client_role") == "optimizer"]
    solver = system.common.accounting()
    return {
        "solver_logical_calls": int(solver["logical_calls"]),
        "solver_provider_attempts": int(solver["provider_attempts"]),
        "solver_successful_attempts": int(solver["successful_provider_calls"]),
        "solver_failed_attempts": int(solver["failed_provider_attempts"]),
        "solver_cache_hits": int(solver["cache_hits"]),
        "solver_input_tokens": int(solver["prompt_tokens"]),
        "solver_output_tokens": int(solver["completion_tokens"]),
        "optimizer_provider_attempts": len(optimizer),
        "optimizer_successful_attempts": sum(bool(row.get("success", False)) for row in optimizer),
        "optimizer_failed_attempts": sum(not bool(row.get("success", False)) for row in optimizer),
        "optimizer_input_tokens": sum(int(row.get("prompt_tokens", 0)) for row in optimizer),
        "optimizer_output_tokens": sum(int(row.get("completion_tokens", 0)) for row in optimizer),
    }


def _durable_accounting(ledger: list[dict[str, Any]]) -> dict[str, int]:
    solver = [row for row in ledger if row["record_kind"] == "solver_logical_invocation"]
    optimizer = [row for row in ledger if row["record_kind"] == "optimizer_provider_attempt"]
    return {
        "solver_logical_calls": len(solver),
        "solver_provider_attempts": sum(int(row["provider_attempts"]) for row in solver),
        "solver_successful_attempts": sum(int(row["successful_provider_calls"]) for row in solver),
        "solver_failed_attempts": sum(
            int(row["provider_attempts"]) - int(row["successful_provider_calls"])
            for row in solver
        ),
        "solver_cache_hits": sum(bool(row["cache_hit"]) for row in solver),
        "solver_input_tokens": sum(int(row["input_tokens"]) for row in solver),
        "solver_output_tokens": sum(int(row["output_tokens"]) for row in solver),
        "optimizer_provider_attempts": len(optimizer),
        "optimizer_successful_attempts": sum(bool(row["success"]) for row in optimizer),
        "optimizer_failed_attempts": sum(not bool(row["success"]) for row in optimizer),
        "optimizer_input_tokens": sum(int(row["input_tokens"]) for row in optimizer),
        "optimizer_output_tokens": sum(int(row["output_tokens"]) for row in optimizer),
    }


async def _run_case(case: dict[str, Any], root: Path, ledger_writer: RGGEPAExecutionLedger) -> dict[str, Any]:
    case_root = root / case["case_id"]
    case_root.mkdir(parents=True, exist_ok=False)
    protocol = RGGEPAProtocol()
    system = _CommonSystem(_cfg(case, case_root), ledger_writer=ledger_writer)
    system.set_solver_stage({
        "seed": case["source_seed"], "parent_id": case["case_id"], "update_index": case["source_update_index"],
        "candidate_id": "parent", "proposal_engine": "current", "evaluation_stage": "full_team",
    })
    try:
        await system.initialize_fixed_probe(case["questions"])
    finally:
        system.set_solver_stage(None)
    assert system.fixed_probe is not None
    target = int(case["target_member"])
    assigned = set(case["assigned_example_ids"])
    parent_profile = system.active_profiles[target]
    parent = _evaluation(system, target, case["parent_prompts"][target], parent_profile, None, assigned)
    mini_ids = [item["example_id"] for item in case["minibatch_private"]]
    by_hash = {row.question_hash: index for index, row in enumerate(system.fixed_probe.examples)}
    mini_indices = [by_hash[item] for item in mini_ids]
    parent_mini = _evaluation(system, target, case["parent_prompts"][target], tuple(parent_profile[i] for i in mini_indices), mini_indices, assigned)
    _record_minibatch_parent_reuse(case, ledger_writer)
    candidates: list[dict[str, Any]] = []

    for source, pool in (("current", case["current_pool"]),):
        for index, source_row in enumerate(pool):
            prompt = str(source_row["prompt"])
            cid = f"A_{index}"
            hard, reason = _hard_gate(prompt, case["questions"])
            row = {"candidate_id": cid, "proposal_engine": source, "prompt": prompt,
                   "prompt_hash": _sha(prompt.encode()), "hard_gate_passed": hard, "hard_gate_reason": reason,
                   "generated": True, "audit_only": False}
            if hard:
                mini_profile = await _profile(system, target, prompt, mini_indices, {
                    "seed": case["source_seed"], "parent_id": case["case_id"], "update_index": case["source_update_index"],
                    "candidate_id": cid, "proposal_engine": source, "evaluation_stage": "minibatch_candidate"})
                row["mini"] = _vector(_evaluation(system, target, prompt, mini_profile, mini_indices, assigned), parent_mini).__dict__
                full_profile = await _profile(system, target, prompt, None, {
                    "seed": case["source_seed"], "parent_id": case["case_id"], "update_index": case["source_update_index"],
                    "candidate_id": cid, "proposal_engine": source, "evaluation_stage": "full_member"})
                full = _evaluation(system, target, prompt, full_profile, None, assigned)
                constraint = evaluate_constraints(full, parent)
                row["full"] = _vector(full, parent).__dict__
                row["current_selector_key"] = list(common_monotone_safe_key(full, index))
                row["feasible"] = bool(constraint.passed)
                row["constraint_reasons"] = list(constraint.rejection_reasons)
            candidates.append(row)

    for index in range(2):
        cid = f"B_{index}"
        begin = len(system.llm.calls)
        sys_prompt, user_prompt = _reflection_request(case, index)
        reflection_meta = {
            "seed": case["source_seed"], "parent_id": case["case_id"], "update_index": case["source_update_index"],
            "candidate_id": cid, "proposal_engine": "gepa_reflection", "evaluation_stage": "reflection"}
        try:
            response = await system.llm.chat_result("qwen3.7-flash", sys_prompt, user_prompt, 0.3, 1800, "optimizer", "reflection")
        finally:
            _persist_new_optimizer_calls(system, begin, reflection_meta, ledger_writer)
        prompt = response.text.strip()
        hard, reason = _hard_gate(prompt, case["questions"])
        row = {"candidate_id": cid, "proposal_engine": "gepa_reflection", "prompt": prompt,
               "prompt_hash": _sha(prompt.encode()), "hard_gate_passed": hard, "hard_gate_reason": reason,
               "generated": True, "audit_only": case["case_id"] in {"seed76_u0_coverage_target1", "seed77_u0_coverage_target2"},
               "raw_response": response.text}
        if hard:
            mini_profile = await _profile(system, target, prompt, mini_indices, {
                "seed": case["source_seed"], "parent_id": case["case_id"], "update_index": case["source_update_index"],
                "candidate_id": cid, "proposal_engine": "gepa_reflection", "evaluation_stage": "minibatch_candidate"})
            mini_eval = _evaluation(system, target, prompt, mini_profile, mini_indices, assigned)
            row["mini"] = _vector(mini_eval, parent_mini).__dict__
        candidates.append(row)

    b_rows = [row for row in candidates if row["proposal_engine"] == "gepa_reflection" and row["hard_gate_passed"]]
    promotion_input = [CandidateScore(row["candidate_id"], TeamVector(**row["mini"]), 0, True) for row in b_rows]
    promoted = {row.candidate_id for row in progressive_promotions(parent=_vector(parent_mini), candidates=promotion_input)}
    for row in b_rows:
        if row["candidate_id"] not in promoted and not row["audit_only"]:
            row["promoted"] = False
            row["progressive_promoted"] = False
            continue
        row["promoted"] = row["candidate_id"] in promoted
        row["progressive_promoted"] = row["promoted"]
        stage = "full_member" if row["promoted"] else "audit_only"
        full_profile = await _profile(system, target, row["prompt"], None, {
            "seed": case["source_seed"], "parent_id": case["case_id"], "update_index": case["source_update_index"],
            "candidate_id": row["candidate_id"], "proposal_engine": "gepa_reflection", "evaluation_stage": stage})
        full = _evaluation(system, target, row["prompt"], full_profile, None, assigned)
        constraint = evaluate_constraints(full, parent)
        row["full"] = _vector(full, parent).__dict__
        row["current_selector_key"] = list(common_monotone_safe_key(full, index))
        row["feasible"] = bool(constraint.passed)
        row["constraint_reasons"] = list(constraint.rejection_reasons)

    current = [row for row in candidates if row["proposal_engine"] == "current" and row.get("feasible")]
    reflected = [row for row in candidates if row["proposal_engine"] == "gepa_reflection" and row.get("feasible") and row.get("promoted")]
    def winner(rows: list[dict[str, Any]], mode: str) -> tuple[str | None, list[str]]:
        scores = [CandidateScore(row["candidate_id"], TeamVector(**row["full"]), 0, True) for row in rows]
        if mode == "team_pareto":
            got, frontier = team_pareto_winner(scores)
            return (got.candidate_id if got else None, [entry.candidate_id for entry in frontier])
        got = max(rows, key=lambda row: tuple(row["current_selector_key"]), default=None)
        return (got["candidate_id"] if got else None, [])
    a0, _ = winner(current, "current")
    # A1 uses exactly same raw pool; progressive decision is counterfactual.
    a_promoted = {entry.candidate_id for entry in progressive_promotions(
        parent=_vector(parent_mini),
        candidates=[CandidateScore(row["candidate_id"], TeamVector(**row["mini"]), 0, True) for row in candidates if row["proposal_engine"] == "current" and row["hard_gate_passed"]],
    )}
    for row in candidates:
        if row["proposal_engine"] == "current":
            row["progressive_promoted"] = row["candidate_id"] in a_promoted
    a1, _ = winner([row for row in current if row["candidate_id"] in a_promoted], "current")
    b0, _ = winner(reflected, "current")
    b1, frontier = winner(reflected, "team_pareto")
    runtime_accounting = _runtime_accounting(system)
    private = {"case": case, "parent": _vector(parent).__dict__, "parent_mini": _vector(parent_mini).__dict__, "candidates": candidates,
               "winners": {"A0": a0, "A1": a1, "B0": b0, "B1": b1}, "b1_frontier": frontier,
               "actual_commits": 0, "validation_calls": 0, "test_calls": 0, "protocol": asdict(protocol),
               "runtime_accounting": runtime_accounting}
    _write(case_root / "private_result.json", private)
    return {
        "case_id": case["case_id"], "seed": case["source_seed"],
        "responsibility_type": case["responsibility_type"], "parent": _vector(parent).__dict__,
        "candidates": [_public_candidate(row) for row in candidates], "winners": private["winners"],
        "b1_frontier": frontier, "actual_commits": 0, "validation_calls": 0, "test_calls": 0,
        "runtime_accounting": runtime_accounting,
    }


def _summary(results: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for arm, engine, progressive, selection in (("A0", "current", False, "current"), ("A1", "current", True, "current"), ("B0", "gepa_reflection", True, "current"), ("B1", "gepa_reflection", True, "team_pareto")):
        pool = [candidate for result in results for candidate in result["candidates"] if candidate["proposal_engine"] == engine]
        valid = [candidate for candidate in pool if candidate["hard_gate_passed"]]
        promoted_ids: set[tuple[str, str]] = set()
        if progressive:
            for result in results:
                candidates = [candidate for candidate in result["candidates"] if candidate["proposal_engine"] == engine and candidate["hard_gate_passed"]]
                for candidate in candidates:
                    if candidate.get("progressive_promoted", False):
                        promoted_ids.add((result["case_id"], candidate["candidate_id"]))
        else:
            promoted_ids = {(result["case_id"], candidate["candidate_id"]) for result in results for candidate in result["candidates"] if candidate["proposal_engine"] == engine and candidate["hard_gate_passed"]}
        promoted = [candidate for result in results for candidate in result["candidates"] if (result["case_id"], candidate["candidate_id"]) in promoted_ids]
        full = [candidate for result in results for candidate in result["candidates"] if candidate["proposal_engine"] == engine and (not progressive or (result["case_id"], candidate["candidate_id"]) in promoted_ids) and "full" in candidate]
        feasible = [candidate for candidate in full if candidate.get("feasible")]
        rows.append({"arm": arm, "proposal_engine": engine, "evaluation_mode": "progressive" if progressive else "full", "selection_mode": selection,
                     "generated": len(pool), "valid": len(valid), "promoted": len(promoted), "full_evaluated": len(full), "feasible": len(feasible),
                     "would_commit": sum(result["winners"].get(arm) is not None for result in results)})
    calls = defaultdict(lambda: {"logical_calls": 0, "provider_calls": 0, "successful_provider_calls": 0, "failed_provider_attempts": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    logical_ids: dict[str, set[str]] = defaultdict(set)
    for row in ledger:
        stage = row["evaluation_stage"]
        logical_ids[stage].add(str(row["logical_call_id"]))
        calls[stage]["provider_calls"] += int(row["provider_attempts"])
        calls[stage]["successful_provider_calls"] += int(row["successful_provider_calls"])
        calls[stage]["failed_provider_attempts"] += int(row["provider_attempts"]) - int(row["successful_provider_calls"])
        calls[stage]["cache_hits"] += int(row["cache_hit"])
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            calls[stage][key] += int(row[key])
    for stage, ids in logical_ids.items():
        calls[stage]["logical_calls"] = len(ids)
    return {"arms": rows, "cost_by_stage": [{"stage": key, **value} for key, value in sorted(calls.items())],
            "api_calls": sum(int(row["successful_provider_calls"]) for row in ledger), "test_calls": 0}


async def execute(args: argparse.Namespace) -> None:
    if not args.authorize_api or os.environ.get(AUTH_ENV) != "1":
        raise PermissionError(f"--authorize-api and {AUTH_ENV}=1 are required")
    if args.run.exists() or args.report.exists():
        raise FileExistsError("execution run/report roots must be fresh")
    registry = _json(args.prep / "private_registry.json")
    if registry["protocol"] != asdict(RGGEPAProtocol()):
        raise RuntimeError("frozen protocol mismatch")
    # A sanitized freeze/report commit may follow the implementation commit; it
    # is harmless only when the frozen execution source remains an ancestor.
    if subprocess.run(["git", "merge-base", "--is-ancestor", str(registry.get("execution_commit", "")), "HEAD"], cwd=ROOT).returncode:
        raise RuntimeError("frozen execution source is not an ancestor of HEAD")
    args.run.mkdir(parents=True)
    ledger_writer = RGGEPAExecutionLedger(args.run / "api_ledger_private.jsonl")
    results: list[dict[str, Any]] = []
    for case in registry["cases"]:
        case["minibatch_private"] = [
            next(row for row in case["questions"] if row["example_id"] == item["example_id"])
            for item in registry["minibatches"][case["case_id"]]
        ]
        results.append(await _run_case(case, args.run, ledger_writer))
        _write(args.run / "progress.json", {"completed_cases": len(results), "total_cases": 6, "test_calls": 0})
    _write(args.run / "result_sanitized.json", {"results": results, "summary": _summary(results, ledger_writer.rows)})
    print(json.dumps({"status": "PASS", "completed_cases": len(results), "test_calls": 0}, sort_keys=True))


def audit(run: Path) -> dict[str, Any]:
    payload = _json(run / "result_sanitized.json")
    results = payload["results"]
    if len(results) != 6 or any(int(row["actual_commits"]) != 0 or int(row["test_calls"]) != 0 for row in results):
        raise RuntimeError("RG-GEPA audit failed: case count, commits, or test isolation")
    for result in results:
        if len(result["candidates"]) != 4:
            raise RuntimeError("RG-GEPA audit failed: two pools must each contain two candidates")
    ledger_path = run / "api_ledger_private.jsonl"
    if not ledger_path.is_file():
        raise RuntimeError("RG-GEPA audit failed: API ledger is absent")
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
    if not ledger:
        raise RuntimeError("RG-GEPA audit failed: API ledger is empty")
    for row in ledger:
        validate_ledger_record(row)
    expected_cases = {str(row["case_id"]) for row in results}
    full_team = [row for row in ledger if row["evaluation_stage"] == "full_team"]
    full_team_cases = {str(row["parent_id"]) for row in full_team}
    minibatch_parent = [row for row in ledger if row["evaluation_stage"] == "minibatch_parent"]
    if full_team_cases != expected_cases:
        raise RuntimeError("RG-GEPA audit failed: parent full-team ledger coverage is incomplete")
    if any(
        row["record_kind"] != "solver_logical_invocation"
        or row["client_role"] != "solver"
        or row["logical_role"] != "solver"
        for row in full_team
    ):
        raise RuntimeError("RG-GEPA audit failed: parent full-team ledger attribution mismatch")
    if len(minibatch_parent) != 6 * 12 or any(
        not row["cache_hit"]
        or row["record_kind"] != "cache_reuse"
        or row["client_role"] != "solver"
        or row["logical_role"] != "solver"
        for row in minibatch_parent
    ):
        raise RuntimeError("RG-GEPA audit failed: paired minibatch parent reuse is incomplete")
    if len({str(row["provider_attempt_id"]) for row in ledger}) != len(ledger):
        raise RuntimeError("RG-GEPA audit failed: duplicate provider-attempt identity")

    reflection_rows = [row for row in ledger if row["evaluation_stage"] == "reflection"]
    expected_reflections = {(case_id, candidate_id, "reflection") for case_id in expected_cases for candidate_id in ("B_0", "B_1")}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in reflection_rows:
        if row.get("client_role") != "optimizer" or row.get("logical_role") != "reflection":
            raise RuntimeError("RG-GEPA audit failed: reflection role attribution mismatch")
        if row.get("proposal_engine") != "gepa_reflection" or row.get("record_kind") != "optimizer_provider_attempt":
            raise RuntimeError("RG-GEPA audit failed: reflection engine/record-kind mismatch")
        grouped[(str(row["parent_id"]), str(row["candidate_id"]), str(row["evaluation_stage"]))].append(row)
    if set(grouped) != expected_reflections:
        raise RuntimeError("RG-GEPA audit failed: reflection logical-call coverage is incomplete")
    for key, attempts in grouped.items():
        ordered = sorted(attempts, key=lambda row: int(row["attempt_index"]))
        if [int(row["attempt_index"]) for row in ordered] != list(range(1, len(ordered) + 1)):
            raise RuntimeError(f"RG-GEPA audit failed: reflection attempt sequence is invalid: {key}")
        if sum(bool(row["success"]) for row in ordered) != 1:
            raise RuntimeError(f"RG-GEPA audit failed: reflection must have one successful attempt: {key}")
        if not bool(ordered[-1]["success"]):
            raise RuntimeError(f"RG-GEPA audit failed: reflection final attempt must succeed: {key}")
        if len({str(row["logical_call_id"]) for row in ordered}) != 1:
            raise RuntimeError(f"RG-GEPA audit failed: reflection logical identity split: {key}")

    runtime = defaultdict(int)
    for result in results:
        if "runtime_accounting" not in result:
            raise RuntimeError("RG-GEPA audit failed: runtime accounting reconciliation is absent")
        for key, value in result["runtime_accounting"].items():
            runtime[key] += int(value)
    durable = _durable_accounting(ledger)
    if dict(runtime) != durable:
        raise RuntimeError("RG-GEPA audit failed: runtime/durable accounting reconciliation mismatch")

    reflection_successes = sum(bool(row["success"]) for row in reflection_rows)
    reflection_failures = len(reflection_rows) - reflection_successes
    return {"status": "PASS", "case_count": 6, "candidate_count": 24, "actual_commits": 0,
            "test_calls": 0, "ledger_records": len(ledger), "parent_full_team_coverage": 6,
            "minibatch_parent_cache_reuses": len(minibatch_parent),
            "reflection_logical_calls": len(grouped), "reflection_successful_calls": reflection_successes,
            "reflection_provider_attempts": len(reflection_rows), "reflection_failed_attempts": reflection_failures,
            "reflection_input_tokens": sum(int(row["input_tokens"]) for row in reflection_rows),
            "reflection_output_tokens": sum(int(row["output_tokens"]) for row in reflection_rows),
            "reflection_total_tokens": sum(int(row["total_tokens"]) for row in reflection_rows),
            "runtime_durable_reconciliation": "PASS"}


def report(run: Path, report_root: Path, prep: Path) -> None:
    if report_root.exists():
        raise FileExistsError("fresh sanitized report root required")
    payload = _json(run / "result_sanitized.json")
    results = payload["results"]
    ledger = [json.loads(line) for line in (run / "api_ledger_private.jsonl").read_text(encoding="utf-8").splitlines() if line]
    summary = _summary(results, ledger)
    report_root.mkdir(parents=True)
    _write(report_root / "protocol.json", {**_json(prep / "PRE_API_FREEZE.json"), "execution_commit": _git("rev-parse", "HEAD"), "solver_contract_identity": contract_identity(), "actual_commits": 0, "test_calls": 0})
    _jsonl(report_root / "candidate_manifest.jsonl", [candidate for result in results for candidate in result["candidates"]])
    with (report_root / "candidate_eval_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "candidate_id", "proposal_engine", "hard_gate_passed", "promoted", "feasible", "full", "mini"])
        writer.writeheader()
        for result in results:
            for candidate in result["candidates"]:
                writer.writerow({key: (json.dumps(candidate[key], sort_keys=True) if isinstance(candidate.get(key), dict) else candidate.get(key, "")) for key in writer.fieldnames if key != "case_id"} | {"case_id": result["case_id"]})
    _write(report_root / "proposal_quality.json", summary["arms"])
    _write(report_root / "cost_by_stage.json", summary["cost_by_stage"])
    _write(report_root / "api_ledger_summary.json", {"api_calls": summary["api_calls"], "test_calls": 0, "ledger_records": len(ledger)})
    _write(report_root / "selector_disagreements.json", [{"case_id": row["case_id"], "B0": row["winners"]["B0"], "B1": row["winners"]["B1"], "frontier": row["b1_frontier"]} for row in results])
    _write(report_root / "fact_assertions.json", {"status": "PASS", **audit(run), "external_validation_optimization_calls": 0, "historical_artifacts_modified": 0})
    _write(report_root / "provenance.json", {"run_root": "ignored_runtime", "source_commit": _git("rev-parse", "HEAD"), "raw_prompts_questions_responses": "excluded"})
    (report_root / "README.md").write_text(
        "# Responsibility-Guided GEPA fixed-parent pilot\n\n"
        "Six frozen Optimize100 parent states were evaluated analytically. All branches have actual commit = 0; Validation, ExternalValidation and Test50 are excluded. Raw prompts, questions, responses and endpoint details remain only in ignored runtime evidence.\n\n"
        + "| Arm | Generated | Valid | Promoted | Full eval | Feasible | Would commit |\n|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(f"| {x['arm']} | {x['generated']} | {x['valid']} | {x['promoted']} | {x['full_evaluated']} | {x['feasible']} | {x['would_commit']} |" for x in summary["arms"]) + "\n",
        encoding="utf-8",
    )
    forbidden = ("FINAL_ANSWER:", "api_key", "dashscope", "https://", "D:\\\\")
    hashes = []
    for path in sorted(report_root.iterdir()):
        data = path.read_bytes(); text = data.decode("utf-8", errors="ignore").lower()
        if any(value.lower() in text for value in forbidden):
            raise RuntimeError(f"sanitization failure: {path.name}")
        hashes.append({"path": path.name, "sha256": _sha(data), "bytes": len(data)})
    _write(report_root / "sanitization_manifest.json", {"status": "PASS", "files": len(hashes)})
    _write(report_root / "sha256_manifest.json", {"files": hashes})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--authorize-api", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.prep, args.run, args.report = args.prep.resolve(), args.run.resolve(), args.report.resolve()
    if args.audit:
        print(json.dumps(audit(args.run), sort_keys=True))
    elif args.report_only:
        report(args.run, args.report, args.prep)
        print(json.dumps({"status": "PASS", "report": str(args.report)}, sort_keys=True))
    else:
        asyncio.run(execute(args))


if __name__ == "__main__":
    main()
