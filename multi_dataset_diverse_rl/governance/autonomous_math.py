"""Execution-only MATH reservations and immutable terminal search evidence."""
from __future__ import annotations
from dataclasses import fields, is_dataclass
import hashlib
import json
import os

from .token_accounting import TokenLedger, OperationalAbort, serialized_request, POLICY, POLICY_40M
from .unified_execution import consumption_path, inventory
from ..persistence.durable_io import atomic_write_json, append_jsonl, read_json
from ..benchmarks.math_domain_binding import execution_binding
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
    from openai._models import FinalRequestOptions
    import httpx
    client = ProviderClientFactory.from_environment(provider_profile=contract["provider"], client_type=OpenAI,
        max_retries=0, timeout=contract["decoding"]["timeout_seconds"])

    def transport(request):
        # Count, hash and send exactly the same UTF-8 body bytes.
        options = FinalRequestOptions.construct(method="post", url="/chat/completions", security={"bearer_auth": True})
        wire = client._client.build_request("POST", client._prepare_url("/chat/completions"),
            headers=client._build_headers(options),
            content=serialized_request(request))
        try:
            response = client._client.send(wire, follow_redirects=False)
        except httpx.TimeoutException as exc:
            raise APITimeoutError(request=wire) from exc
        except httpx.TransportError as exc:
            raise APIConnectionError(request=wire) from exc
        try:
            try:
                body = response.json()
            except ValueError as exc:
                if response.is_error or response.is_redirect:
                    error = client._make_status_error_from_response(response)
                    error.provider_evidence = dict(http_status=response.status_code, response_text=response.text)
                    raise error
                if contract.get('runtime_persistence_policy'):
                    exc.provider_evidence = dict(http_status=response.status_code, response_text=response.text)
                raise
            if response.is_error or response.is_redirect:
                error = client._make_status_error_from_response(response)
                usage = body.get("usage") if isinstance(body, dict) else None
                usage = usage if isinstance(usage, dict) else {}
                error.token_usage = dict(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"))
                error.provider_evidence = dict(http_status=response.status_code, response_body=body)
                raise error
            try:
                choice = body["choices"][0]
                content = choice["message"].get("content")
            except (KeyError, IndexError, TypeError, AttributeError) as exc:
                if contract.get('runtime_persistence_policy'):
                    exc.provider_evidence = dict(http_status=response.status_code, response_body=body)
                    usage = body.get('usage') if isinstance(body,dict) else None
                    usage = usage if isinstance(usage,dict) else {}
                    exc.token_usage = dict(input_tokens=usage.get('prompt_tokens'),output_tokens=usage.get('completion_tokens'))
                raise
            usage = body.get("usage")
            usage = usage if isinstance(usage, dict) else {}
            result = dict(text=content, finish_reason=choice.get("finish_reason"),
                input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"), response_id=body.get("id"))
            if contract.get('runtime_persistence_policy'):
                result['provider_http_response_body'] = body
            if 'optimizer_generation_policy' in contract:
                from ..benchmarks.math_optimizer_diagnostics import provider_thinking_indicators
                reasoning = choice['message'].get('reasoning_content')
                result.update(provider_usage_details=usage,
                    provider_reasoning_character_count=len(reasoning) if isinstance(reasoning,str) else None,
                    provider_reasoning_content_present='reasoning_content' in choice['message'],
                    provider_response_accepted=True,provider_metadata_loss_audited=True,
                    provider_thinking_indicators=provider_thinking_indicators(body))
            return result
        finally:
            response.close()
    return transport, client


def initial_prompts(root, contract):
    return tuple(m["prompt"] for m in read_json(root / contract["initial_team_path"])["members"])


def ledger_policy(contract):
    return POLICY_40M


def durable_output_cache(run_root,contract,payload):
    from ..persistence.exact_output_cache import DurableExactOutputCache,digest
    return DurableExactOutputCache(run_root/'resolved_output_cache',dict(
        execution_attempt_id=contract['execution_attempt_id'],cache_namespace=contract['cache_namespace'],
        startup_identity_sha256=payload['startup_identity_sha256'],source_sha=payload['scope']['source_sha'],
        authorization_sha256=digest(read_json(run_root/'consumed_authorization.json')),
        binding_sha256=digest(contract),generation_policy_sha256=digest(contract['solver_decoding_policy']),
        recovery_policy_sha256=digest(contract['invalid_recovery_policy'])))


async def execute_search(root, prep, run_root, payload):
    c = read_json(root / payload["manifest"]["execution_binding"]["path"])
    binding = execution_binding(root, c)
    if binding.blockers():
        raise SearchContractError("AUTONOMOUS_BINDING_NOT_READY")
    from ..persistence.provider_receipts import POLICY as durability_policy, ProviderResponseReceipts
    durable = c.get("runtime_persistence_policy")
    if durable is not None and durable != durability_policy:
        raise SearchContractError("RUNTIME_PERSISTENCE_POLICY_MISMATCH")
    budget = TokenLedger(root / c["token_ledger_directory"], task_sha256=c["task_authorization_sha256"],
        policy=ledger_policy(c), best_effort_snapshots=durable is not None)
    broker = client = None
    try:
        reserve = ValidationReserve(read_json(root / c["validation_accounting_metadata_path"]), initial_prompts(root,c))
        if budget.remaining < reserve.remaining():
            raise OperationalAbort("STOP_TOKEN_BUDGET_INSUFFICIENT_FOR_VALIDATION")
        consume(root,prep,run_root,payload)
        atomic_write_json(run_root / "accounting_start.json",budget.view())
        transport,client = create_transport(c)
        arm=c.get('execution_arm','A1')
        broker = RequestBroker(contract=c,transport=transport,arm=arm,seed=81,token_ledger=budget,
            reserve_reader=reserve.remaining,ledger_writer=lambda r:append_jsonl(run_root / "ledger.jsonl",r),
            raw_writer=lambda r:append_jsonl(run_root / "provider_trace_private.jsonl",r),
            durable_cache=durable_output_cache(run_root,c,payload),
            response_receipts=(ProviderResponseReceipts(run_root/'provider_response_receipts_private',
                attempt_id=c['execution_attempt_id'],startup_identity_sha256=payload['startup_identity_sha256'])
                if durable is not None else None))
        broker.prompt_observer = reserve.observe
        solver = BenchmarkSolver(binding.benchmark(),broker)
        from ..search.textual_gradients import PerExampleGradientProvider, GradientClusterProvider
        pattern_provider=GradientClusterProvider(broker,read_json(root/c['pattern_prompt_path'])['prompt'],
            gradient_provider=PerExampleGradientProvider(broker,read_json(root/c['gradient_prompt_path'])['prompt'],
                numeric_guard_writer=lambda r:append_jsonl(run_root/'numeric_guard_private.jsonl',r),
                recovery_policy=c.get('gradient_recovery_policy'),
                recovery_writer=(lambda r:append_jsonl(run_root/'gradient_recovery_private.jsonl',r))
                    if c.get('gradient_recovery_policy') is not None else None),
            partition_completion_policy=c.get('partition_completion_policy'),
            partition_writer=(lambda r:append_jsonl(run_root/'partition_completion_private.jsonl',r))
                if c.get('partition_completion_policy') is not None else None)
        composed = binding.compose(arm=arm,seed=81,solver=solver,reflection=ReflectionProvider(broker),pattern_provider=pattern_provider,run_root=run_root)
        if any(composed.memory.audit()[k] for k in ('success_writes','failure_writes','shared_writes')):
            raise OperationalAbort('FRESH_MEMORY_STATE_REQUIRED')
        # A naturally empty trajectory still needs a durable hashable receipt.
        with (run_root / "trajectory_private.jsonl").open("xb") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        composed.execution_observer = lambda stage,data:append_jsonl(run_root / "trajectory_private.jsonl",{"stage":stage,**plain(data)})
        if c['execution_phase']=='pilot':
            from .pilot_observation import attach_pilot_observer
            attach_pilot_observer(composed,run_root)
        composed.state.initialize()
        atomic_write_json(run_root / "initial_state_private.json",plain(composed.state.snapshot()))
        if "initial_competence_binding" in c:
            initial = composed.state.snapshot()
            if initial.diagnostics["evaluation_support_identity"] != c["initial_competence_binding"]["support_identity"]:
                raise OperationalAbort("INITIAL_COMPETENCE_SUPPORT_MISMATCH")
            atomic_write_json(run_root / "initial_competence_floor.json",dict(binding=c["initial_competence_binding"],
                state_id=initial.team_state_id, member_scores=initial.member_scores))
        result = await composed.run(max_opportunities=c["provider_bounds"]["max_opportunities"])
        accepted = {"CANARY_PARENT_EPOCH_COMPLETE","CANARY_ONE_PRODUCTION_OPPORTUNITY_COMPLETE","NO_FEASIBLE_OPPORTUNITY"} if c["execution_phase"]=="canary" else {"SATURATION_REACHED","NO_FEASIBLE_OPPORTUNITY"}
        scientific_complete=result.stop_reason in accepted
        operational_truncation=bool(c.get('operational_pilot')) and result.stop_reason in {
            'OPERATIONAL_OPPORTUNITY_CEILING','EMERGENCY_PROVIDER_CALL_CEILING'}
        if not scientific_complete and not operational_truncation:
            raise OperationalAbort("NONSCIENTIFIC_STOP_"+result.stop_reason)
        if c.get("method_identity") == "unified_team_prompt_search_v2_1" and c["execution_phase"] == "canary" and not result.trace:
            raise OperationalAbort("CANARY_NO_COMPLETE_PRODUCTION_OPPORTUNITY")
        if c['execution_phase']=='canary' and not any(t.candidate_ids for t in result.trace):
            raise OperationalAbort('STOP_LAYER1_ZERO_THROUGHPUT')
        final = composed.state.snapshot()
        atomic_write_json(run_root / "final_team_private.json",dict(prompts=final.member_prompts,state_id=final.team_state_id))
        atomic_write_json(run_root / "final_state_private.json",plain(final))
        summary = dict(result=plain(result),ledger=broker.usage,accounting=budget.view(),validation_reserve=reserve.remaining(),
            validation_search_raw_reads=0,validation_calls=0,test_raw_reads=0,test_calls=0,pattern_calls=broker.usage['pattern'],memory_activity=0)
        if c.get('operational_pilot'):
            summary.update(pilot_completed=scientific_complete,
                pilot_status='SCIENTIFIC_COMPLETION' if scientific_complete else 'INCOMPLETE_OPERATIONAL_TRUNCATION',
                operational_pilot=c['operational_pilot'])
        summary['pattern_input_audit']=pattern_provider.input_audit
        if c.get('partition_completion_policy') is not None:
            from ..search.partition_completion import completion_statistics
            summary['partition_completion']=dict(policy=c['partition_completion_policy'],
                **completion_statistics(pattern_provider.partition_audit))
        if durable is not None:
            summary['runtime_persistence'] = dict(policy=durable,
                derived_token_snapshot_detached=budget.snapshot_updates_disabled,
                snapshot_error_category=budget.snapshot_error_category)
        gradients=pattern_provider.gradient_provider
        gradient_multiplier=3 if c.get('gradient_recovery_policy') is not None else 1
        if c.get('gradient_recovery_policy') is not None:
            from ..search.gradient_recovery import statistics
            recovery_path=run_root/'gradient_recovery_private.jsonl'
            recovery_events=[json.loads(line) for line in recovery_path.read_text(encoding='utf-8').splitlines() if line] if recovery_path.exists() else []
            summary['gradient_recovery']=statistics(recovery_events)
            if (summary['gradient_recovery']['physical_gradient_calls']!=gradients.calls
                    or summary['gradient_recovery']['logical_gradient_count']!=summary['gradient_recovery']['accepted_gradient_count']
                    or summary['gradient_recovery']['gradient_three_fail']):
                raise OperationalAbort('GRADIENT_CONTRACT_RECOVERY_ACCOUNTING_INCOMPLETE')
        summary.update(pattern_gradient_calls=broker.usage['pattern_gradient'],
            pattern_cluster_calls=broker.usage['pattern_cluster'],gradient_input_audit=gradients.input_audit)
        if (gradients.calls!=broker.usage['pattern_gradient'] or pattern_provider.calls!=broker.usage['pattern_cluster']
                or broker.usage['pattern']!=gradients.calls+pattern_provider.calls):
            raise OperationalAbort('PATTERN_GRADIENT_ACCOUNTING_INCOMPLETE')
        if c['execution_phase']=='canary' and not 1<=gradients.calls<=gradient_multiplier*c['initial_competence_binding']['count']:
            raise OperationalAbort('PATTERN_GRADIENT_CANARY_ACCOUNTING_INCOMPLETE')
        if c['execution_phase']=='canary' and (pattern_provider.calls!=1 or not composed.evaluation.provider.probed):
            raise OperationalAbort('PATTERN_CANARY_FLOW_INCOMPLETE')
        if c['execution_phase']=='pilot' and (pattern_provider.calls!=len(result.trace)
                or gradients.calls!=len(gradients.input_audit)
                or not len(result.trace)<=gradients.calls<=gradient_multiplier*60*len(result.trace)):
            raise OperationalAbort('PATTERN_PILOT_FLOW_INCOMPLETE')
        summary['memory_activity']=composed.memory.audit()['stateful_write_count']
        summary['memory_audit']=composed.memory.audit()
        from .team_change import team_change_receipt
        summary['deployed_team_change']=team_change_receipt(initial_prompts(root,c),final.member_prompts)
        if c['execution_phase']=='pilot':
            atomic_write_json(run_root/'VALIDATION_DISPOSITION.json',{
                **summary['deployed_team_change'], 'validation_model_calls':0, 'validation_provider_calls':0,
                'validation_status':('DEFERRED_BY_USER_SCOPE' if summary['deployed_team_change']['team_changed'] else 'SKIPPED_NO_TEAM_CHANGE'),
                'VoteAccDelta':'NOT_AVAILABLE', 'OracleAccDelta':'NOT_AVAILABLE', 'bootstrap':'NOT_RUN'})
        summary = plain(summary)
        atomic_write_json(run_root / "execution_summary.json",summary)
        if read_json(run_root / "execution_summary.json")!=summary:
            raise OperationalAbort("EXECUTION_PERSISTENCE_MISMATCH")
        atomic_write_json(run_root / "accounting_end.json",budget.view())
        atomic_write_json(run_root / "lifecycle.json",dict(status="EXECUTION_COMPLETE",attempt_id=c["execution_attempt_id"]))
        atomic_write_json(run_root / "raw_evidence_inventory.json",inventory(run_root))
        if c["execution_phase"]=="pilot":
            receipt_name='SEARCH_COMPLETE_RECEIPT' if scientific_complete else 'SEARCH_CLOSED_RECEIPT'
            atomic_write_json(run_root / (receipt_name+'.json'),dict(identity=receipt_name,attempt_id=c["execution_attempt_id"],
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
            truncation=(bool(c.get('operational_pilot')) and isinstance(exc,OperationalAbort)
                and (str(exc) in {'TOKEN_CEILING','PROVIDER_CALL_CEILING','TRANSPORT_CEILING'}
                    or str(exc).startswith('STOP_TOKEN_BUDGET_')))
            if truncation:
                reason=('TOKEN_CEILING' if str(exc).startswith('STOP_TOKEN_BUDGET_') else str(exc))
                partial=dict(pilot_completed=False,pilot_status='INCOMPLETE_OPERATIONAL_TRUNCATION',
                    stop_reason=reason,operational_pilot=c['operational_pilot'],
                    ledger=broker.usage if broker else {},accounting=budget.view(),
                    validation_calls=0,test_calls=0,partial_opportunity_possible=True)
                if ('composed' in locals() and composed.state.initial_state_id is not None):
                    state=composed.state.snapshot()
                    atomic_write_json(run_root/'final_state_private.json',plain(state))
                    atomic_write_json(run_root/'final_team_private.json',dict(prompts=state.member_prompts,state_id=state.team_state_id))
                    from .pilot_observation import memory_snapshot
                    atomic_write_json(run_root/'terminal_memory_private.json',memory_snapshot(composed.memory))
                atomic_write_json(run_root/'execution_summary.json',plain(partial))
                atomic_write_json(run_root/'accounting_end.json',budget.view())
                atomic_write_json(run_root/'lifecycle.json',dict(status='EXECUTION_COMPLETE',
                    scientific_complete=False,attempt_id=c['execution_attempt_id'],stop_category=reason))
                atomic_write_json(run_root/'raw_evidence_inventory.json',inventory(run_root))
                atomic_write_json(run_root/'SEARCH_CLOSED_RECEIPT.json',dict(identity='SEARCH_CLOSED_RECEIPT',
                    attempt_id=c['execution_attempt_id'],search_closed_forever=True,pilot_completed=False,
                    stop_reason=reason,execution_summary_sha256=file_sha(run_root/'execution_summary.json'),
                    source_sha=payload['manifest']['source_sha'],startup_identity_sha256=payload['startup_identity_sha256'],
                    validation_model_calls=0,test_model_calls=0))
                return partial
            pattern_failure=(isinstance(exc,SearchContractError) and str(exc).startswith(('PATTERN_','STOP_PATTERN_','FOCUSED_')))
            atomic_write_json(run_root / "accounting_end.json",budget.view())
            atomic_write_json(run_root / "lifecycle.json",dict(status="EXECUTION_ABORTED",attempt_id=c["execution_attempt_id"],
                error_category=type(exc).__name__,stop_category=str(exc) if isinstance(exc,OperationalAbort) or pattern_failure else type(exc).__name__,provider_usage=broker.usage if broker else {"attempts":0}))
            atomic_write_json(run_root / "raw_evidence_inventory.json",inventory(run_root))
        raise
    finally:
        if client:
            client.close()
        budget.close()
