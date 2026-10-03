"""Single-use optimizer-only diagnostic; no scoring, archive, or search state."""
import hashlib
import json
import time

from .unified_execution import validate_prep, consumption_path, inventory
from .startup_identity import canonical_sha256
from .autonomous_math import consume, create_transport, initial_prompts
from .token_accounting import TokenLedger, POLICY_40M, OperationalAbort, serialized_request
from ..benchmarks.math_accounting_prep import ValidationReserve
from ..benchmarks.math_optimizer_diagnostics import DIAGNOSTIC_ID, optimizer_response_telemetry, reflection_input_anatomy
from ..search.provider_runtime import RequestBroker
from ..search.schemas import SearchContractError
from ..persistence.durable_io import atomic_write_json, append_jsonl, read_json


def prepare_witness(root,prep,*,messages,attempt_id,request_path,expected_run_root,exact_production_input):
    base=validate_prep(root,prep)
    c=read_json(root/base['manifest']['execution_binding']['path'])
    if c['optimizer_generation_policy']['enable_thinking'] is not False:
        raise SearchContractError('NONTHINKING_WITNESS_POLICY_REQUIRED')
    anatomy=reflection_input_anatomy(messages)
    if anatomy['reflective_examples']!=3:
        raise SearchContractError('WITNESS_REFLECTION_MINIBATCH_MISMATCH')
    broker=RequestBroker(contract=c,transport=None,arm='A1',seed=81)
    request,_=broker._request_identity(role='reflection',split='optimize',messages=messages)
    request_file=(root/request_path).resolve()
    if not request_file.is_relative_to((root/'runs').resolve()) or request_file.exists():
        raise SearchContractError('FRESH_PRIVATE_WITNESS_REQUEST_REQUIRED')
    atomic_write_json(request_file,request)
    scope=dict(identity=DIAGNOSTIC_ID,attempt_id=attempt_id,phase='OPERATIONAL_DIAGNOSTIC_ONLY',
        source_sha=base['scope']['source_sha'],preregistration_identity=base['manifest']['preregistration_identity'],
        binding_sha256=base['scope']['binding_sha256'],base_startup_identity_sha256=base['startup_identity_sha256'],
        user_authorization_sha256=c['continuation_authorization_sha256'],provider=c['provider'],model='qwen3.7-flash',
        roles=['reflection'],successful_provider_call_ceiling=1,
        transport_attempt_ceiling=c['decoding']['transport_retries']+1,
        validation_calls=0,test_calls=0,solver_calls=0,pattern_calls=0,
        optimizer_generation_policy=c['optimizer_generation_policy'],request_path=request_path,
        request_file_sha256=hashlib.sha256(request_file.read_bytes()).hexdigest(),
        serialized_request_sha256=hashlib.sha256(serialized_request(request)).hexdigest(),
        messages_sha256=canonical_sha256(messages),expected_run_root=expected_run_root,
        exact_production_prompt_witness=bool(exact_production_input),output_reuse='FORBIDDEN',accuracy_scoring='FORBIDDEN')
    payload=dict(schema_version='math_optimizer_witness_prep_v1',scope=scope,source_identity=base['source_identity'])
    payload['startup_identity_sha256']=canonical_sha256(payload)
    atomic_write_json(prep/'optimizer_witness_prep.json',payload)
    atomic_write_json(prep/'authorization.json',dict(explicit_user_authorized=False,single_use=True,consumed=False,
        scope=scope,startup_identity_sha256=payload['startup_identity_sha256']))
    return payload


def validate_witness(root,prep,*,require_authorized=False):
    base=validate_prep(root,prep)
    payload=read_json(prep/'optimizer_witness_prep.json')
    scope=payload['scope']
    if payload['startup_identity_sha256']!=canonical_sha256({k:v for k,v in payload.items() if k!='startup_identity_sha256'}):
        raise SearchContractError('WITNESS_STARTUP_IDENTITY_MISMATCH')
    c=read_json(root/base['manifest']['execution_binding']['path'])
    if (payload['schema_version']!='math_optimizer_witness_prep_v1' or scope['identity']!=DIAGNOSTIC_ID
            or payload['source_identity']!=base['source_identity']
            or scope['base_startup_identity_sha256']!=base['startup_identity_sha256']
            or scope['source_sha']!=base['scope']['source_sha']
            or scope['binding_sha256']!=base['scope']['binding_sha256']
            or scope['preregistration_identity']!=base['manifest']['preregistration_identity']
            or scope['user_authorization_sha256']!=c['continuation_authorization_sha256']
            or scope['provider']!=c['provider'] or scope['model']!='qwen3.7-flash'
            or scope['roles']!=['reflection'] or scope['successful_provider_call_ceiling']!=1
            or scope['transport_attempt_ceiling']!=c['decoding']['transport_retries']+1
            or scope['phase']!='OPERATIONAL_DIAGNOSTIC_ONLY'
            or any(scope[k]!=0 for k in ('solver_calls','pattern_calls','validation_calls','test_calls'))
            or scope['optimizer_generation_policy']!=c['optimizer_generation_policy']
            or scope['optimizer_generation_policy']['enable_thinking'] is not False
            or scope['output_reuse']!='FORBIDDEN' or scope['accuracy_scoring']!='FORBIDDEN'):
        raise SearchContractError('WITNESS_SCOPE_MISMATCH')
    request_path=(root/scope['request_path']).resolve()
    if not request_path.is_relative_to((root/'runs').resolve()) or hashlib.sha256(request_path.read_bytes()).hexdigest()!=scope['request_file_sha256']:
        raise SearchContractError('WITNESS_REQUEST_FILE_MISMATCH')
    request=read_json(request_path)
    broker=RequestBroker(contract=c,transport=None,arm='A1',seed=81)
    expected,_=broker._request_identity(role='reflection',split='optimize',messages=request['messages'])
    if (request!=expected or canonical_sha256(request['messages'])!=scope['messages_sha256']
            or hashlib.sha256(serialized_request(request)).hexdigest()!=scope['serialized_request_sha256']
            or reflection_input_anatomy(request['messages'])['reflective_examples']!=3):
        raise SearchContractError('WITNESS_REQUEST_BODY_MISMATCH')
    if require_authorized:
        auth=read_json(prep/'authorization.json')
        if (auth.get('explicit_user_authorized') is not True or auth.get('single_use') is not True
                or auth.get('consumed') is not False or auth.get('scope')!=scope
                or auth.get('startup_identity_sha256')!=payload['startup_identity_sha256']
                or (prep/'authorization_consumed.json').exists() or consumption_path(root,scope).exists()):
            raise SearchContractError('EXACT_SINGLE_USE_AUTHORIZATION_REQUIRED')
    return payload,c,request


async def execute_witness(root,prep,run_root):
    payload,c,request=validate_witness(root,prep,require_authorized=True)
    if run_root.resolve()!=(root/payload['scope']['expected_run_root']).resolve():
        raise SearchContractError('WITNESS_RUN_ROOT_MISMATCH')
    ledger=TokenLedger(root/c['token_ledger_directory'],task_sha256=c['task_authorization_sha256'],policy=POLICY_40M)
    client=None
    try:
        reserve=ValidationReserve(read_json(root/c['validation_accounting_metadata_path']),initial_prompts(root,c)).remaining()
        consume(root,prep,run_root,payload)
        atomic_write_json(run_root/'accounting_start.json',ledger.view())
        transport,client=create_transport(c)
        for retry in range(payload['scope']['transport_attempt_ceiling']):
            reservation_id=ledger.reserve(request,attempt_id=payload['scope']['attempt_id'],
                stage='optimizer_witness',role='reflection',model=request['model'],protected_validation=reserve)
            append_jsonl(run_root/'ledger.jsonl',dict(kind='ATTEMPT',physical_attempt_no=retry+1,
                serialized_request_sha256=payload['scope']['serialized_request_sha256']))
            try:
                result=transport(request)
            except Exception as exc:
                charge=ledger.reconcile(reservation_id,getattr(exc,'token_usage',None),outcome=type(exc).__name__)
                append_jsonl(run_root/'provider_trace_private.jsonl',dict(request=request,
                    error_category=type(exc).__name__,provider_evidence=getattr(exc,'provider_evidence',None),charge=charge))
                from openai import APIConnectionError,APITimeoutError,RateLimitError,InternalServerError
                if not isinstance(exc,(APIConnectionError,APITimeoutError,RateLimitError,InternalServerError)) or retry+1==payload['scope']['transport_attempt_ceiling']:
                    raise OperationalAbort('PROVIDER_TERMINAL_'+type(exc).__name__)
                time.sleep(min(c['decoding']['retry_sleep_seconds']*2**retry,c['decoding']['retry_backoff_ceiling_seconds']))
                continue
            charge=ledger.reconcile(reservation_id,result,outcome='RESPONSE')
            if not isinstance(result,dict):raise OperationalAbort('PROVIDER_RESPONSE_ACCOUNTING_INVALID')
            telemetry=optimizer_response_telemetry(request,result,c['optimizer_generation_policy'])
            append_jsonl(run_root/'provider_trace_private.jsonl',dict(request=request,response=result,charge=charge))
            atomic_write_json(run_root/'witness_response_diagnostics.json',telemetry)
            append_jsonl(run_root/'ledger.jsonl',dict(kind='RESPONSE',**charge))
            atomic_write_json(run_root/'lifecycle.json',dict(status='EXECUTION_COMPLETE',attempt_id=payload['scope']['attempt_id'],
                successful_provider_calls=1,physical_attempts=retry+1,output_reuse='FORBIDDEN',accuracy_scoring='FORBIDDEN'))
            return dict(execution_status='EXECUTION_COMPLETE',provider_calls=1,diagnostics=telemetry,
                solver_calls=0,validation_calls=0,test_calls=0)
        raise OperationalAbort('WITNESS_COMPLETION_MISSING')
    except BaseException as exc:
        if run_root.exists():
            atomic_write_json(run_root/'lifecycle.json',dict(status='EXECUTION_ABORTED',attempt_id=payload['scope']['attempt_id'],
                termination_reason=str(exc) if isinstance(exc,(OperationalAbort,SearchContractError)) else type(exc).__name__))
        raise
    finally:
        if client is not None:client.close()
        if run_root.exists():
            atomic_write_json(run_root/'accounting_end.json',ledger.view())
            atomic_write_json(run_root/'raw_evidence_inventory.json',inventory(run_root))
        ledger.close()
