"""Frozen Phase-B execution: four local GEPA tasks, no Layer-2 evaluation."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from infrastructure.common_solver_contract_v1.contract import (
    COMMON_SOLVER_CONTRACT_ID, CONTRACT_SPEC, canonical_json_bytes,
)
from infrastructure.common_solver_contract_v1.evaluator import CommonSolverEvaluator, TransportResponse
from multi_dataset_diverse_rl.evaluation.output_contract import SOLVER_OUTPUT_CONTRACT_VERSION
from multi_dataset_diverse_rl.governance.authorization import require_api_authorization
from multi_dataset_diverse_rl.governance.manifest import validate_manifest
from multi_dataset_diverse_rl.governance.run_lifecycle import run_with_lifecycle
from multi_dataset_diverse_rl.local_optimizers.gepa_adapter import LocalSolverObservation
from multi_dataset_diverse_rl.parent_acquisition import digest, restore_task
from multi_dataset_diverse_rl.provider_credentials import resolve_api_key, resolve_base_url
from scripts.run_local_gepa_acceptance_rate_pilot import pooled_summary, run_parent

IDENTITY = "local_gepa_acceptance_rate_pilot_phase_b_v2"
MANIFEST = ROOT / "experiments/manifests" / f"{IDENTITY}.yaml"
AUTH_ENV = "LOCAL_GEPA_PHASE_B_V2_AUTHORIZED"
EXPECTED_IDS = [f"seed78_update0_member{i}" for i in range(1, 5)]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


class DurableLedger:
    def __init__(self, path: Path):
        self.path = path
        self.sequence = 0

    def append(self, row: dict[str, Any]) -> None:
        self.sequence += 1
        payload = {"sequence": self.sequence, **row}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def verify_freeze(bundle: Path) -> tuple[dict, dict, dict]:
    freeze = read(bundle / "freeze.json")
    handoff = read(bundle / "HANDOFF.json")
    if not freeze["READY_TO_RUN"] or not handoff["READY_TO_RUN"]:
        raise ValueError("READY_TO_RUN_required")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != freeze["commit"] or head != handoff["source"]["repository_commit"]:
        raise ValueError("frozen_commit_mismatch")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise ValueError("tracked_worktree_dirty")
    for relative, expected in freeze["source_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"source_hash_mismatch:{relative}")
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "infrastructure/experiment_manifest.schema.json").read_text())
    errors = validate_manifest(manifest, schema)
    if errors:
        raise ValueError("manifest_invalid:" + ";".join(errors))
    if sha(MANIFEST) != freeze["manifest_sha256"]:
        raise ValueError("manifest_hash_mismatch")
    if sha(bundle / "selected_parent_tasks_private.json") != freeze["private_task_bundle_sha256"]:
        raise ValueError("private_task_bundle_hash_mismatch")
    parent_freeze_path = ROOT / "reports/local_gepa_acceptance_rate_pilot_phase_b_v2_prep_20260918/PARENT_FREEZE.json"
    if sha(parent_freeze_path) != freeze["parent_freeze_sha256"]:
        raise ValueError("parent_freeze_hash_mismatch")
    parent_freeze = read(parent_freeze_path)
    tasks = read(bundle / "selected_parent_tasks_private.json")
    if sorted(tasks) != EXPECTED_IDS:
        raise ValueError("selected_parent_ids_mismatch")
    for row in parent_freeze["selected_parents"]:
        payload = tasks.get(row["parent_task_id"])
        if payload is None or digest(payload) != row["task_payload_sha256"]:
            raise ValueError("parent_payload_mismatch")
        task = restore_task(payload)
        if (task.task_id != row["parent_task_id"] or task.seed != row["task_seed"]
                or task.budget.max_metric_calls != 205
                or digest([item.example_id for item in task.search_examples]) != row["search_example_identity_hash"]
                or digest([item.example_id for item in task.local_validation_examples]) != row["local_validation_identity_hash"]):
            raise ValueError("parent_identity_mismatch")
    if len({row["source_state_hash"] for row in parent_freeze["selected_parents"]}) != 1:
        raise ValueError("single_baseline_state_required")
    return manifest, tasks, freeze


class Runtime:
    def __init__(self, *, loop, client, ledger, freeze_guard):
        self.loop, self.client, self.ledger, self.freeze_guard = loop, client, ledger, freeze_guard
        self.current_task = ""
        self.solver_attempts = self.reflection_attempts = self.reflection_calls = 0
        self.reflection_successes = self.reflection_input_tokens = self.reflection_output_tokens = 0
        self.solver_evaluator = CommonSolverEvaluator(transport=self._solver_transport, retryable=self._retryable)

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        return (isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError))
                or getattr(exc, "status_code", None) in CONTRACT_SPEC.retry_status_codes)

    async def _provider(self, *, role: str, request: dict, identity: str) -> TransportResponse:
        self.freeze_guard()
        if role == "solver":
            self.solver_attempts += 1
            attempt = self.solver_attempts
        else:
            self.reflection_attempts += 1
            attempt = self.reflection_attempts
        self.ledger.append({"event": "provider_attempt_start", "role": role, "attempt": attempt,
                            "task_id": self.current_task, "request_identity": identity})
        try:
            response = await self.client.chat.completions.create(**request)
            if response.usage is None or len(response.choices) != 1:
                raise ValueError("provider_usage_or_choice_missing")
            choice = response.choices[0]
            result = TransportResponse(choice.message.content or "", int(response.usage.prompt_tokens),
                                       int(response.usage.completion_tokens), str(choice.finish_reason or ""))
            if result.finish_reason != "stop":
                raise ValueError("provider_finish_reason_not_stop")
            self.ledger.append({"event": "provider_attempt_success", "role": role, "attempt": attempt,
                                "task_id": self.current_task, "request_identity": identity,
                                "prompt_tokens": result.prompt_tokens,
                                "completion_tokens": result.completion_tokens,
                                "finish_reason": result.finish_reason})
            return result
        except Exception as exc:
            self.ledger.append({"event": "provider_attempt_failure", "role": role, "attempt": attempt,
                                "task_id": self.current_task, "request_identity": identity,
                                "error_type": type(exc).__name__})
            raise

    async def _solver_transport(self, request: dict) -> TransportResponse:
        identity = hashlib.sha256(canonical_json_bytes(request)).hexdigest()
        return await self._provider(role="solver", request=request, identity=identity)

    def evaluate(self, decision_procedure: str, example) -> LocalSolverObservation:
        result = asyncio.run_coroutine_threadsafe(
            self.solver_evaluator.evaluate(decision_procedure=decision_procedure,
                                           question=example.input_payload), self.loop).result()
        raw = self.solver_evaluator.cache[result.request_identity]
        return LocalSolverObservation(
            parsed_answer=result.response.answer if result.response.valid else None,
            raw_output=raw, correct=bool(result.response.valid and result.response.answer == example.gold),
            valid=result.response.valid, failure_reason=None if result.response.valid else result.response.status,
            input_tokens=result.prompt_tokens, output_tokens=result.completion_tokens,
            provider_called=not result.cache_hit)

    solver_contract_id = COMMON_SOLVER_CONTRACT_ID
    output_contract_id = SOLVER_OUTPUT_CONTRACT_VERSION

    @staticmethod
    def _reflection_messages(prompt) -> tuple[str, str]:
        if isinstance(prompt, str):
            return "Return only one complete replacement reasoning procedure. Do not discuss output formatting.", prompt
        systems = [str(row.get("content", "")) for row in prompt if row.get("role") == "system"]
        others = [f"{row.get('role', 'user')}: {row.get('content', '')}" for row in prompt if row.get("role") != "system"]
        return "\n".join(systems) or "You improve a reasoning procedure.", "\n\n".join(others)

    def reflect(self, prompt) -> str:
        if self.reflection_calls >= 32:
            raise RuntimeError("reflection_logical_call_ceiling_exhausted")
        self.reflection_calls += 1
        system, user = self._reflection_messages(prompt)
        request = {"model": "qwen3.7-flash", "messages": [{"role": "system", "content": system},
                   {"role": "user", "content": user}], "temperature": 0.0, "max_tokens": 1800,
                   "timeout": CONTRACT_SPEC.timeout_seconds, "extra_body": {"enable_thinking": False}}
        identity = hashlib.sha256(canonical_json_bytes(request)).hexdigest()

        async def call() -> str:
            last = None
            for attempt in range(CONTRACT_SPEC.transport_attempt_cap):
                try:
                    result = await self._provider(role="reflection", request=request, identity=identity)
                    self.reflection_successes += 1
                    self.reflection_input_tokens += result.prompt_tokens
                    self.reflection_output_tokens += result.completion_tokens
                    return result.text
                except Exception as exc:
                    last = exc
                    if not self._retryable(exc) or attempt + 1 == CONTRACT_SPEC.transport_attempt_cap:
                        raise
                    await asyncio.sleep(CONTRACT_SPEC.retry_backoff_seconds[attempt])
            raise RuntimeError(f"unreachable reflection failure:{type(last).__name__}")
        return asyncio.run_coroutine_threadsafe(call(), self.loop).result()

    def reflection_accounting(self) -> dict[str, int]:
        return {"successful_calls": self.reflection_successes, "input_tokens": self.reflection_input_tokens,
                "output_tokens": self.reflection_output_tokens}


async def execute(bundle: Path, output: Path) -> None:
    if os.getenv(AUTH_ENV) != "1":
        raise PermissionError(f"{AUTH_ENV}=1_required")
    manifest, payloads, freeze = verify_freeze(bundle)
    runtime_manifest = json.loads(json.dumps(manifest))
    runtime_manifest["status"] = "RUNNING"
    runtime_manifest["git"]["implementation_commit"] = freeze["commit"]
    runtime_manifest["lifecycle_history"].append({"status": "RUNNING", "timestamp": freeze["frozen_at"]})
    runtime_errors = validate_manifest(
        runtime_manifest,
        json.loads((ROOT / "infrastructure/experiment_manifest.schema.json").read_text()),
    )
    if runtime_errors:
        raise ValueError("runtime_manifest_invalid:" + ";".join(runtime_errors))
    for role in ("solver", "reflection"):
        require_api_authorization(runtime_manifest, phase="phase_b_local_optimizer", role=role,
                                  explicit_user_authorized=True)
    if output.exists():
        raise FileExistsError("formal Phase-B run root must be fresh")
    output.mkdir(parents=True)
    summaries: list[dict[str, Any]] = []

    async def operation() -> None:
        ledger = DurableLedger(output / "provider_ledger.jsonl")
        from openai import AsyncOpenAI
        _, key = resolve_api_key("DASHSCOPE_API_KEY")
        _, base = resolve_base_url("DASHSCOPE_BASE_URL")
        if not key or not base:
            raise ValueError("provider_credentials_unavailable")

        def guard():
            for relative, expected in freeze["critical_source_hashes"].items():
                if sha(ROOT / relative) != expected:
                    raise RuntimeError("critical_source_changed_after_freeze")

        try:
            async with AsyncOpenAI(api_key=key, base_url=base, max_retries=0,
                                   timeout=CONTRACT_SPEC.timeout_seconds) as client:
                runtime = Runtime(loop=asyncio.get_running_loop(), client=client,
                                  ledger=ledger, freeze_guard=guard)
                for index, identity in enumerate(EXPECTED_IDS, start=1):
                    runtime.current_task = identity
                    task = restore_task(payloads[identity])
                    summary = await run_parent(task=task, evaluator=runtime,
                        reflection_lm=runtime.reflect, accounting_reader=runtime.reflection_accounting,
                        run_root=output / "local_gepa", proposal_quota=8, skip_allowance=16)
                    if summary["funnel"]["proposal_attempts_observed"] > 8:
                        raise RuntimeError("proposal_event_ceiling_exceeded")
                    write_new(output / "parent_summaries" / f"{identity}.json", summary)
                    summaries.append(summary)
                    print(json.dumps({"parent_completed": index, "task_id": identity,
                          "proposal_end": summary["funnel"]["proposal_attempts_observed"],
                          "accepted": summary["primary"]["numerator"],
                          "denominator": summary["primary"]["denominator"],
                          "solver_metric_rows": runtime.solver_evaluator.logical_calls,
                          "reflection_calls": runtime.reflection_calls}), flush=True)
                if runtime.solver_evaluator.logical_calls > 820 or runtime.reflection_calls > 32:
                    raise RuntimeError("Phase-B hard budget exceeded")
                pooled = pooled_summary(summaries)
                result = {"execution_status": "EXECUTION_COMPLETE", "experiment_id": IDENTITY,
                    "frozen_commit": freeze["commit"], "parent_count": 4,
                    "all_parent_quotas_complete": pooled["complete"], "pooled": pooled,
                    "accounting": {"solver": runtime.solver_evaluator.accounting(),
                        "solver_provider_attempts": runtime.solver_attempts,
                        "reflection_logical_calls": runtime.reflection_calls,
                        "reflection_provider_attempts": runtime.reflection_attempts,
                        "reflection_successful_calls": runtime.reflection_successes,
                        "reflection_input_tokens": runtime.reflection_input_tokens,
                        "reflection_output_tokens": runtime.reflection_output_tokens},
                    "isolation": {"TeamMiniBatch": 0, "Full": 0, "Common_Safe": 0, "Shadow": 0,
                        "write_back": 0, "persistent_realizability_update": 0,
                        "Validation50": 0, "Test50": 0},
                    "team_transfer": "TEAM_TRANSFER_NOT_EVALUATED"}
            verify_freeze(bundle)
            write_new(output / "execution.json", result)
            write_new(output / "completion.json", {"status": "EXECUTION_COMPLETE",
                "source_hashes_verified_after_execution": True})
        except BaseException as exc:
            write_new(output / "abort.json", {"status": "EXECUTION_ABORTED", "error_type": type(exc).__name__,
                "completed_parents": len(summaries)})
            raise

    await run_with_lifecycle(
        output / "run_lifecycle.json",
        identity={"experiment_id": IDENTITY, "frozen_commit": freeze["commit"], "phase_b_only": True},
        operation=operation,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "execute"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, tasks, _ = verify_freeze(args.bundle)
    if args.mode == "preflight":
        print(json.dumps({"status": "PASS", "selected_parent_ids": sorted(tasks),
                          "api_authorized": manifest["api_authorization"]["authorized"],
                          "formal_run_root_absent": not args.output.exists()}))
    else:
        asyncio.run(execute(args.bundle, args.output))


if __name__ == "__main__":
    main()
