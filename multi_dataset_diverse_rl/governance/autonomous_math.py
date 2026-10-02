"""Execution-only MATH reservations and immutable terminal search evidence."""
from __future__ import annotations
from dataclasses import fields, is_dataclass
import hashlib
import json
import os

from .token_accounting import TokenLedger, OperationalAbort, serialized_request
from .unified_execution import consumption_path, inventory
from ..persistence.durable_io import atomic_write_json, append_jsonl, read_json
from ..benchmarks.math_autonomous import MATHAutonomousBinding
from ..benchmarks.math_accounting_prep import ValidationReserve
from ..benchmarks.math import MATHBenchmarkAdapter
from ..search.provider_runtime import RequestBroker, BenchmarkSolver, ReflectionProvider
from ..search.schemas import SearchContractError


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(value):
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [plain(v) for v in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def consume(root, prep, run_root, payload):
    if run_root.exists() or not run_root.resolve().is_relative_to((root / "runs").resolve()):
        raise SearchContractError("FRESH_LOCAL_RUN_ROOT_REQUIRED")
    marker = consumption_path(root, payload["scope"])
    marker.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(scope=payload["scope"], startup_identity_sha256=payload["startup_identity_sha256"])
    with marker.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    atomic_write_json(prep / "authorization_consumed.json", receipt)
    run_root.mkdir(parents=True)
    atomic_write_json(run_root / "consumed_authorization.json", {**read_json(prep / "authorization.json"), "consumed":True})
    atomic_write_json(run_root / "lifecycle.json", dict(status="RUNNING", attempt_id=payload["scope"]["attempt_id"]))


def create_transport(contract):
    from ..provider_factory import ProviderClientFactory
    from openai import OpenAI, APIConnectionError, APITimeoutError
    import httpx
    client = ProviderClientFactory.from_environment(provider_profile=contract["provider"], client_type=OpenAI,
        max_retries=0, timeout=contract["decoding"]["timeout_seconds"])

    def transport(request):
        # Count, hash and send exactly the same UTF-8 body bytes.
        wire = httpx.Request("POST", client.base_url.join("chat/completions"),
            headers={k:v for k,v in client.default_headers.items() if isinstance(v,(str,bytes))},
            content=serialized_request(request))
        try:
            response = client._client.send(wire)
        except httpx.TimeoutException as exc:
            raise APITimeoutError(request=wire) from exc
        except httpx.TransportError as exc:
            raise APIConnectionError(request=wire) from exc
        try:
            try:
                body = response.json()
            except ValueError:
                if response.is_error:
                    raise client._make_status_error_from_response(body=None, response=response)
                raise
            if response.is_error:
                error = client._make_status_error_from_response(body=body, response=response)
                usage = body.get("usage") or {} if isinstance(body, dict) else {}
                error.token_usage = dict(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"))
                raise error
            choice = body["choices"][0]
            usage = body.get("usage") or {}
            return dict(text=choice["message"].get("content"), finish_reason=choice.get("finish_reason"),
                input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"), response_id=body.get("id"))
        finally:
            response.close()
    return transport, client


def initial_prompts(root, contract):
    return tuple(m["prompt"] for m in read_json(root / contract["initial_team_path"])["members"])


async def execute_search(root, prep, run_root, payload):
    c = read_json(root / payload["manifest"]["execution_binding"]["path"])
    binding = MATHAutonomousBinding(root, c)
    if binding.blockers():
        raise SearchContractError("AUTONOMOUS_BINDING_NOT_READY")
    budget = TokenLedger(root / c["token_ledger_directory"], task_sha256=c["task_authorization_sha256"])
    broker = client = None
    try:
        reserve = ValidationReserve(read_json(root / c["validation_accounting_metadata_path"]), initial_prompts(root,c))
        if budget.remaining < reserve.remaining():
            raise OperationalAbort("STOP_TOKEN_BUDGET_INSUFFICIENT_FOR_VALIDATION")
        consume(root,prep,run_root,payload)
        atomic_write_json(run_root / "accounting_start.json",budget.view())
        transport,client = create_transport(c)
        broker = RequestBroker(contract=c,transport=transport,arm="A1",seed=81,token_ledger=budget,
            reserve_reader=reserve.remaining,ledger_writer=lambda r:append_jsonl(run_root / "ledger.jsonl",r),
            raw_writer=lambda r:append_jsonl(run_root / "provider_trace_private.jsonl",r))
        broker.prompt_observer = reserve.observe
        solver = BenchmarkSolver(MATHBenchmarkAdapter(),broker)
        composed = binding.compose(arm="A1",seed=81,solver=solver,reflection=ReflectionProvider(broker),pattern_provider=None,run_root=run_root)
        composed.execution_observer = lambda stage,data:append_jsonl(run_root / "trajectory_private.jsonl",{"stage":stage,**plain(data)})
        composed.state.initialize()
        atomic_write_json(run_root / "initial_state_private.json",plain(composed.state.snapshot()))
        result = await composed.run(max_opportunities=c["provider_bounds"]["max_opportunities"])
        accepted = {"CANARY_PARENT_EPOCH_COMPLETE","NO_FEASIBLE_OPPORTUNITY"} if c["execution_phase"]=="canary" else {"SATURATION_REACHED","NO_FEASIBLE_OPPORTUNITY"}
        if result.stop_reason not in accepted:
            raise OperationalAbort("NONSCIENTIFIC_STOP_"+result.stop_reason)
        final = composed.state.snapshot()
        atomic_write_json(run_root / "final_team_private.json",dict(prompts=final.member_prompts,state_id=final.team_state_id))
        atomic_write_json(run_root / "final_state_private.json",plain(final))
        summary = dict(result=plain(result),ledger=broker.usage,accounting=budget.view(),validation_reserve=reserve.remaining(),
            validation_search_raw_reads=0,validation_calls=0,test_raw_reads=0,test_calls=0,pattern_calls=0,memory_activity=0)
        atomic_write_json(run_root / "execution_summary.json",summary)
        if read_json(run_root / "execution_summary.json")!=summary:
            raise OperationalAbort("EXECUTION_PERSISTENCE_MISMATCH")
        atomic_write_json(run_root / "accounting_end.json",budget.view())
        atomic_write_json(run_root / "lifecycle.json",dict(status="EXECUTION_COMPLETE",attempt_id=c["execution_attempt_id"]))
        atomic_write_json(run_root / "raw_evidence_inventory.json",inventory(run_root))
        if c["execution_phase"]=="pilot":
            atomic_write_json(run_root / "SEARCH_COMPLETE_RECEIPT.json",dict(identity="SEARCH_COMPLETE_RECEIPT",attempt_id=c["execution_attempt_id"],
                source_sha=payload["manifest"]["source_sha"],startup_identity_sha256=payload["startup_identity_sha256"],
                final_team_sha256=file_sha(run_root / "final_team_private.json"),
                trajectory_sha256=file_sha(run_root / "trajectory_private.jsonl") if (run_root / "trajectory_private.jsonl").exists() else None,
                raw_inventory_sha256=file_sha(run_root / "raw_evidence_inventory.json"),execution_summary_sha256=file_sha(run_root / "execution_summary.json"),
                stop_reason=result.stop_reason,search_closed_forever=True,validation_search_raw_reads=0,validation_model_calls=0,test_raw_reads=0,test_model_calls=0))
        return summary
    except BaseException as exc:
        for key in tuple(budget.inflight):
            budget.reconcile(key,None,outcome="ABORT_UNKNOWN_FULL_CHARGE")
        if run_root.exists():
            atomic_write_json(run_root / "accounting_end.json",budget.view())
            atomic_write_json(run_root / "lifecycle.json",dict(status="EXECUTION_ABORTED",attempt_id=c["execution_attempt_id"],
                error_category=type(exc).__name__,stop_category=str(exc) if isinstance(exc,OperationalAbort) else type(exc).__name__,provider_usage=broker.usage if broker else {"attempts":0}))
            atomic_write_json(run_root / "raw_evidence_inventory.json",inventory(run_root))
        raise
    finally:
        if client:
            client.close()
        budget.close()
