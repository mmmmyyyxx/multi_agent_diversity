"""Governed independent Stage 1 runner. No optimization graph or held-out access."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .math_baseline_calibration import (digest,native_extract,diagnostic_extract,TRUNCATED)
from .math_baseline_audit import optimize_rows
from .math_calibration_plan import validate_plan,request_for,PLAN_ID
from .math_calibration_accounting import CalibrationLedger,CalibrationAbort,CAP,read_ledger
from ..governance.token_accounting import serialized_request
from ..persistence.durable_io import atomic_write_json,append_jsonl,read_json
from ..persistence.provider_receipts import ProviderResponseReceipts

IDENTITY = 'MATH_BASELINE_STAGE1_EXECUTION_V1'
SPEC_PATH = 'experiments/manifests/math_baseline_calibration_stage1_v1.json'
PLAN_PATH = 'experiments/protocols/math_baseline_calibration_v1/protocol.json'
ATTEMPT = 'math_baseline_stage1_seed81_20261010_attempt1'
RETRYABLE = {'APIConnectionError','APITimeoutError','RateLimitError','InternalServerError'}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root,*args):
    return subprocess.check_output(['git',*args],cwd=root,text=True).strip()


def tracked_clean(root):
    if git(root,'branch','--show-current') != 'main' or git(root,'status','--porcelain','--untracked-files=no'):
        raise CalibrationAbort('CALIBRATION_TRACKED_SOURCE_NOT_CLEAN_MAIN')


def load_selection(root,plan):
    binding=read_json(root/'experiments/execution_bindings/a4_v25_seed81_canary_attempt1.json')
    if sha(root/'experiments/execution_bindings/a4_v25_seed81_canary_attempt1.json') != plan['source_identities']['original_binding_sha256']:
        raise CalibrationAbort('CALIBRATION_DATA_BINDING_MISMATCH')
    rows,_=optimize_rows(root,binding)
    lookup={hashlib.sha256(r['stable_example_id'].encode()).hexdigest():r for r in rows}
    selected=[]
    for m in plan['data']['membership']:
        row=lookup[m['example_id_sha256']]
        if (any(row[k]!=m[k] for k in ('content_sha256','input_sha256'))
                or hashlib.sha256(row['reference_final_answer'].encode()).hexdigest()!=m['reference_sha256']):
            raise CalibrationAbort('CALIBRATION_MEMBERSHIP_DATA_MISMATCH')
        selected.append(row)
    if len(selected)!=60: raise CalibrationAbort('CALIBRATION_STAGE1_N_MISMATCH')
    return selected,binding


def source_inventory(root):
    from ..governance.source_identity import local_imports
    roots=[root/'scripts/run_math_baseline_calibration.py',root/'multi_dataset_diverse_rl/diagnostics/math_calibration_results.py']
    pending=list(roots);seen=set()
    while pending:
        path=pending.pop()
        if path in seen or not path.is_file(): continue
        seen.add(path);pending.extend(local_imports(root,path,initializers=True)-seen)
    seen.update(root/p for p in (SPEC_PATH,PLAN_PATH,'experiments/execution_bindings/a4_v25_seed81_canary_attempt1.json',
        'docs/workflows/MATH_BASELINE_STAGE1_EXECUTION_V1.md',
        'AGENTS.md','experiments/registry.yaml','experiments/lineage.yaml','tests/suite_classification.json','scripts/tooling_classification.json'))
    return [dict(path=p.relative_to(root).as_posix(),sha256=sha(p)) for p in sorted(seen)]


def freeze_private_bundle(root,destination,authorization):
    """Offline freeze after the implementation/preregistration commit is clean."""
    root=Path(root);destination=Path(destination)
    tracked_clean(root)
    if destination.exists() or not destination.resolve().is_relative_to((root/'runs').resolve()):
        raise CalibrationAbort('FRESH_PRIVATE_CALIBRATION_PREP_REQUIRED')
    plan=read_json(root/PLAN_PATH);validate_plan(plan)
    spec=read_json(root/SPEC_PATH)
    if spec != expected_spec(plan): raise CalibrationAbort('CALIBRATION_EXECUTION_SPEC_MISMATCH')
    selected,binding=load_selection(root,plan)
    # The guarded offline environment retains the URL configuration but strips
    # real credential values. Only its digest enters this private frozen scope.
    endpoint=os.environ.get('LWJ_DASHSCOPE_BASE_URL','').rstrip('/')
    if not endpoint: raise CalibrationAbort('CALIBRATION_PROVIDER_CONFIGURATION_MISSING')
    if (authorization.get('exact_user_message')!='授权进行api调用'
            or authorization.get('stage')!='stage1' or authorization.get('cap')!=CAP):
        raise CalibrationAbort('CALIBRATION_USER_AUTHORIZATION_MISMATCH')
    inventory=source_inventory(root)
    import importlib.metadata
    from ..benchmarks.math_worker import PINS
    dependencies={name:importlib.metadata.version(name) for name in (*PINS,'openai','httpx')}
    scope=dict(identity=IDENTITY,attempt_id=ATTEMPT,phase='stage1',roles=['solver'],
        source_sha=git(root,'rev-parse','HEAD'),source_files=inventory,source_files_sha256=digest(inventory),
        preregistration_path=SPEC_PATH,preregistration_sha256=sha(root/SPEC_PATH),
        parent_protocol_sha256=plan['protocol_sha256'],parent_protocol_file_sha256=sha(root/PLAN_PATH),
        membership_sha256=plan['data']['membership_sha256'],model='qwen3-8b',enable_thinking=False,
        solver_concurrency=1,optimizer_concurrency=0,seed=81,n=60,arms=['A','B','C'],members=1,
        provider_profile='lwj',provider_endpoint_sha256=hashlib.sha256(endpoint.encode()).hexdigest(),
        timeout_seconds=120,sdk_retries=0,transport_retries=20,retry_sleep_seconds=1.5,retry_backoff_ceiling_seconds=60,
        python_executable='runs/math_v2_1/runtime/Scripts/python.exe',
        interpreter_sha256=sha(root/'runs/math_v2_1/runtime/Scripts/python.exe'),dependency_versions=dependencies,
        authorized_total=CAP,old_scope_reuse=False,validation_access=False,test_access=False,shadow_access=False,
        selected_rows_sha256=digest(selected),real_api_authorized=True,READY_TO_RUN=True,
        user_authorization_sha256=digest(authorization),run_root='runs/'+ATTEMPT)
    scope['startup_identity_sha256']=digest(scope)
    destination.mkdir(parents=True)
    atomic_write_json(destination/'selected_rows_private.json',selected)
    atomic_write_json(destination/'authorization_private.json',authorization)
    atomic_write_json(destination/'frozen_manifest_private.json',scope)
    atomic_write_json(destination/'handoff_private.json',dict(schema_version='sol_luna_experiment_handoff_v1',
        experiment_id='math_baseline_calibration_stage1_v1',source=dict(repository_commit=scope['source_sha'],tracked_worktree_status='clean'),
        protocol=dict(name=IDENTITY,version='v1',protocol_sha256=plan['protocol_sha256'],
            manifest_path=str(destination.relative_to(root)/'frozen_manifest_private.json').replace('\\','/'),
            manifest_sha256=sha(destination/'frozen_manifest_private.json'),preregistration_path=SPEC_PATH,
            preregistration_sha256=sha(root/SPEC_PATH)),
        data=dict(dataset='math',split_identity=binding['split_version'],split_sha256=binding['split_manifest_sha256']),
        models=dict(solver='qwen3-8b',optimizer_or_reflection='DISABLED',thinking=False,local_optimizer_backend='DISABLED',local_optimizer_version_or_sha256='DISABLED'),
        execution=dict(seeds=[81],opportunity_budget=0,early_stop_rule=spec['stopping'],
            exact_runner_command=scope['python_executable']+' scripts/run_math_baseline_calibration.py --execute --prep '+destination.relative_to(root).as_posix(),
            expected_output_directory=scope['run_root'],expected_checkpoint_path=scope['run_root']+'/checkpoint_private.json',
            expected_ledger_path=scope['run_root']+'/accounting/events.jsonl',allowed_retries=spec['retries']),
        authorization=dict(api_scope=scope['startup_identity_sha256'],validation_access_policy='FORBIDDEN',test_access_policy='FORBIDDEN'),
        fail_closed_conditions=spec['fail_closed_conditions'],READY_TO_RUN=True))
    return scope


def expected_spec(plan):
    return dict(schema_version='independent_math_baseline_stage1_preregistration_v1',
        experiment_id='math_baseline_calibration_stage1_v1',parent_ids=['math_baseline_calibration_v1'],identity=IDENTITY,
        parent_protocol_path=PLAN_PATH,parent_protocol_sha256=plan['protocol_sha256'],membership_sha256=plan['data']['membership_sha256'],
        source_freeze='EXACT_CLEAN_MAIN_HEAD_IN_PRIVATE_FROZEN_MANIFEST',stage='stage1',n=60,arms=['A','B','C'],members=1,
        sampling=plan['sampling'],model=plan['model'],native_parsers={a:plan['arms'][a]['parser'] for a in ('A','B','C')},
        common_diagnostic=plan['secondary']['identity'],pairing=plan['pairing'],retries=plan['retry'],
        provider_policy=dict(profile='lwj',timeout_seconds=120,sdk_retries=0,
            retry_categories=sorted(RETRYABLE),retry_sleep_seconds=1.5,retry_backoff_ceiling_seconds=60,
            exact_serialized_http_body=True,returned_model_alias_required='qwen3-8b',weight_revision='UNKNOWN_UNLESS_RETURNED'),
        authorized_total=CAP,physical_attempts_upper=15120,successful_draws_upper=720,
        validation_access=False,test_access=False,shadow_access=False,optimizer_components=plan['disabled_components'],
        stopping=plan['stopping'],incomplete_stage=plan['incomplete_stage'],stage2_authorized=False,
        scientific_results_used_to_stop=False,authorization='USER_STAGE1_CONTEXT_ACCEPTANCE_FROZEN_TO_SINGLE_FRESH_ATTEMPT',
        fail_closed_conditions=['SOURCE_OR_PROTOCOL_MISMATCH','PROVIDER_OR_IDENTITY_FAILURE','UNTRUSTED_USAGE_OR_OUTPUT_BOUND',
            'PERSISTENCE_OR_LEDGER_FAILURE','BUDGET_EXHAUSTION','HELDOUT_ACCESS','NO_SCIENTIFIC_RESUME_OR_RERUN'])


def verify_bundle(root,prep,*,credential_check=False):
    root=Path(root);prep=Path(prep);tracked_clean(root)
    scope=read_json(prep/'frozen_manifest_private.json')
    payload={k:v for k,v in scope.items() if k!='startup_identity_sha256'}
    plan=read_json(root/PLAN_PATH);validate_plan(plan)
    if (digest(payload)!=scope['startup_identity_sha256'] or scope['source_sha']!=git(root,'rev-parse','HEAD')
            or scope['identity']!=IDENTITY or scope['attempt_id']!=ATTEMPT
            or scope['authorized_total']!=CAP or scope['phase']!='stage1'
            or scope['real_api_authorized'] is not True or scope['READY_TO_RUN'] is not True
            or read_json(root/SPEC_PATH)!=expected_spec(plan)
            or sha(root/SPEC_PATH)!=scope['preregistration_sha256']
            or sha(root/PLAN_PATH)!=scope['parent_protocol_file_sha256']
            or source_inventory(root)!=scope['source_files']):
        raise CalibrationAbort('CALIBRATION_STARTUP_FREEZE_MISMATCH')
    required=dict(roles=['solver'],model='qwen3-8b',enable_thinking=False,solver_concurrency=1,
        optimizer_concurrency=0,seed=81,n=60,arms=['A','B','C'],members=1,provider_profile='lwj',
        timeout_seconds=120,sdk_retries=0,transport_retries=20,retry_sleep_seconds=1.5,retry_backoff_ceiling_seconds=60,
        old_scope_reuse=False,validation_access=False,test_access=False,shadow_access=False,
        python_executable='runs/math_v2_1/runtime/Scripts/python.exe',run_root='runs/'+ATTEMPT)
    if (any(scope.get(k)!=v or type(scope.get(k)) is not type(v) for k,v in required.items())
            or scope['parent_protocol_sha256']!=plan['protocol_sha256']
            or scope['membership_sha256']!=plan['data']['membership_sha256']
            or digest(scope['source_files'])!=scope['source_files_sha256']):
        raise CalibrationAbort('CALIBRATION_STAGE1_SCOPE_MISMATCH')
    import importlib.metadata
    if (Path(sys.executable).resolve()!=(root/scope['python_executable']).resolve()
            or sha(root/scope['python_executable'])!=scope['interpreter_sha256']
            or any(importlib.metadata.version(name)!=version for name,version in scope['dependency_versions'].items())):
        raise CalibrationAbort('CALIBRATION_INTERPRETER_OR_DEPENDENCY_MISMATCH')
    selected=read_json(prep/'selected_rows_private.json')
    if digest(selected)!=scope['selected_rows_sha256']:
        raise CalibrationAbort('CALIBRATION_PRIVATE_SELECTION_MISMATCH')
    if len(selected)!=60 or len({r['stable_example_id'] for r in selected})!=60:
        raise CalibrationAbort('CALIBRATION_PRIVATE_MEMBERSHIP_SHAPE')
    selected_identity=[dict(example_id_sha256=hashlib.sha256(r['stable_example_id'].encode()).hexdigest(),
        content_sha256=r['content_sha256'],input_sha256=r['input_sha256'],
        reference_sha256=hashlib.sha256(r['reference_final_answer'].encode()).hexdigest(),
        subject=r['content']['type'].lower().replace(' ','_').replace('&','and'),level=r['content']['level']) for r in selected]
    if selected_identity!=plan['data']['membership']:
        raise CalibrationAbort('CALIBRATION_PRIVATE_MEMBERSHIP_MISMATCH')
    if digest(read_json(prep/'authorization_private.json'))!=scope['user_authorization_sha256']:
        raise CalibrationAbort('CALIBRATION_AUTHORIZATION_HASH_MISMATCH')
    handoff=read_json(prep/'handoff_private.json')
    if (handoff['source']['repository_commit']!=scope['source_sha'] or handoff['READY_TO_RUN'] is not True
            or handoff['protocol']['manifest_sha256']!=sha(prep/'frozen_manifest_private.json')
            or handoff['execution']['exact_runner_command']!=scope['python_executable']+' scripts/run_math_baseline_calibration.py --execute --prep '+prep.relative_to(root).as_posix()):
        raise CalibrationAbort('CALIBRATION_HANDOFF_MISMATCH')
    # Canonical source bytes are checked at every startup without parsing any
    # held-out rows. The selected private rows were generated by the gated reader.
    binding=read_json(root/'experiments/execution_bindings/a4_v25_seed81_canary_attempt1.json')
    if sha(root/'experiments/execution_bindings/a4_v25_seed81_canary_attempt1.json')!=plan['source_identities']['original_binding_sha256']:
        raise CalibrationAbort('CALIBRATION_DATA_BINDING_MISMATCH')
    split_path=root/binding['split_directory']/'math.json'
    if sha(split_path)!=binding['split_manifest_sha256']:
        raise CalibrationAbort('CALIBRATION_DATA_SPLIT_MANIFEST_MISMATCH')
    split=read_json(split_path)
    for source in {r['source_split'] for r in selected}:
        expected=next(s for s in split['canonical_sources'] if s['name']==source)
        if sha(root/binding['canonical_root']/'raw/math'/ (source+'.jsonl'))!=expected['canonical_sha256']:
            raise CalibrationAbort('CALIBRATION_CANONICAL_BYTES_CHANGED')
    if credential_check:
        endpoint=os.environ.get('LWJ_DASHSCOPE_BASE_URL','').rstrip('/')
        if not os.environ.get('LWJ_DASHSCOPE_API_KEY') or hashlib.sha256(endpoint.encode()).hexdigest()!=scope['provider_endpoint_sha256']:
            raise CalibrationAbort('CALIBRATION_PROVIDER_CONFIGURATION_CHANGED')
    return scope,plan,selected


def create_exact_transport(scope):
    from ..provider_factory import ProviderClientFactory
    from openai import OpenAI,APIConnectionError,APITimeoutError
    from openai._models import FinalRequestOptions
    import httpx
    client=ProviderClientFactory.from_environment(provider_profile='lwj',client_type=OpenAI,
        max_retries=0,timeout=scope['timeout_seconds'])
    def transport(request):
        options=FinalRequestOptions.construct(method='post',url='/chat/completions',security={'bearer_auth':True})
        wire=client._client.build_request('POST',client._prepare_url('/chat/completions'),
            headers=client._build_headers(options),content=serialized_request(request))
        try: response=client._client.send(wire,follow_redirects=False)
        except httpx.TimeoutException as error: raise APITimeoutError(request=wire) from error
        except httpx.TransportError as error: raise APIConnectionError(request=wire) from error
        try:
            try:body=response.json()
            except ValueError as parsing_error:
                error=client._make_status_error_from_response(response) if response.is_error or response.is_redirect else parsing_error
                error.provider_evidence=dict(http_status=response.status_code,response_text=response.text)
                raise error
            if response.is_error or response.is_redirect:
                error=client._make_status_error_from_response(response)
                usage=body.get('usage',{}) if isinstance(body,dict) else {}
                error.token_usage=dict(input_tokens=usage.get('prompt_tokens'),output_tokens=usage.get('completion_tokens'))
                error.provider_evidence=dict(http_status=response.status_code,response_body=body)
                raise error
            try:
                choice=body['choices'][0];usage=body.get('usage') or {}
                return dict(text=choice['message'].get('content'),finish_reason=choice.get('finish_reason'),
                    input_tokens=usage.get('prompt_tokens'),output_tokens=usage.get('completion_tokens'),
                    response_id=body.get('id'),provider_http_response_body=body,http_status=response.status_code)
            except Exception as error:
                error.provider_evidence=dict(http_status=response.status_code,response_body=body)
                usage=body.get('usage') or {} if isinstance(body,dict) else {}
                error.token_usage=dict(input_tokens=usage.get('prompt_tokens'),output_tokens=usage.get('completion_tokens'))
                raise
        finally: response.close()
    return transport,client


def validate_response(response,request):
    body=response.get('provider_http_response_body')
    if (not isinstance(body,dict) or body.get('model')!='qwen3-8b'
            or not isinstance(response.get('text'),(str,type(None)))):
        raise CalibrationAbort('CALIBRATION_PROVIDER_IDENTITY_OR_CONTENT_FAILURE')
    choice=body['choices'][0];usage=body.get('usage') or {}
    if (choice['message'].get('content')!=response['text'] or choice.get('finish_reason')!=response['finish_reason']
            or usage.get('prompt_tokens')!=response.get('input_tokens')
            or usage.get('completion_tokens')!=response.get('output_tokens')):
        raise CalibrationAbort('CALIBRATION_CONTENT_OR_USAGE_LOSS')
    reasoning=choice['message'].get('reasoning_content')
    details=usage.get('completion_tokens_details') or {}
    if reasoning or type(details.get('reasoning_tokens')) is int and details['reasoning_tokens']>0:
        raise CalibrationAbort('CALIBRATION_NONTHINKING_CONTROL_CONTRADICTORY')
    if type(response.get('output_tokens')) is int and response['output_tokens']>request['max_tokens']:
        raise CalibrationAbort('CALIBRATION_OUTPUT_BOUND_NOT_ENFORCED')


def run_logical(arm,row,scope,transport,ledger,receipts,*,sleep=time.sleep,dispatch_guard=lambda:None):
    attempts=[];previous=None;physical_retries=0
    xid=row['stable_example_id'];xhash=hashlib.sha256(xid.encode()).hexdigest()
    for ordinal in range(1,5):
        capacity=6144 if previous in TRUNCATED else 3600
        request=request_for(arm,row['content']['problem'],capacity)
        for physical in range(1,22):
            dispatch_guard()
            lane=dict(stage='stage1',arm=arm,example_id_sha256=xhash,member=0,
                semantic_draw=ordinal,physical_attempt=physical,attempt_id=scope['attempt_id'])
            key=ledger.reserve(request,lane)
            try: response=transport(request)
            except Exception as error:
                category=type(error).__name__
                try:
                    receipt=receipts.persist(key,dict(lane=lane,request=request,error_category=category,
                        provider_evidence=getattr(error,'provider_evidence',None)))
                except BaseException:
                    ledger.charge(key,None,'RECEIPT_PERSISTENCE_FAILURE');raise
                ledger.charge(key,getattr(error,'token_usage',None),category,receipt)
                if category not in RETRYABLE or physical==21:
                    raise CalibrationAbort('PROVIDER_TERMINAL_'+category) from None
                physical_retries+=1
                sleep(min(1.5*2**(physical-1),60));continue
            try: receipt=receipts.persist(key,dict(lane=lane,request=request,response=response))
            except BaseException:
                ledger.charge(key,None,'RECEIPT_PERSISTENCE_FAILURE');raise
            charged,reliable=ledger.charge(key,response,'RESPONSE',receipt)
            if not reliable: raise CalibrationAbort('CALIBRATION_USAGE_NOT_RELIABLE')
            validate_response(response,request)
            break
        native=native_extract(arm,response['text'],response['finish_reason'])
        attempts.append(dict(lane=lane,reservation_id=key,receipt=receipt,request=request,response=response,
            native=native,charged_tokens=charged,output_capacity=capacity))
        previous=response['finish_reason']
        if native['valid']:break
    return dict(example_id=xid,example_id_sha256=xhash,arm=arm,member=0,stage='stage1',
        reference=row['reference_final_answer'],subject=row['content']['type'],level=row['content']['level'],
        first_native=attempts[0]['native'],recovered_native=attempts[-1]['native'],
        semantic_draws=len(attempts),transport_retries=physical_retries,attempts=attempts,
        input_tokens=sum(a['response']['input_tokens'] for a in attempts),
        output_tokens=sum(a['response']['output_tokens'] for a in attempts))


def run_panel(selected,scope,run_root,transport,*,sleep=time.sleep,progress=print,dispatch_guard=lambda:None):
    run_root=Path(run_root)
    ledger=CalibrationLedger(run_root/'accounting',scope['startup_identity_sha256'])
    receipts=ProviderResponseReceipts(run_root/'provider_receipts_private',attempt_id=scope['attempt_id'],
        startup_identity_sha256=scope['startup_identity_sha256'])
    completed=[];status='EXECUTION_ABORTED';reason='UNEXPECTED_PROCESS_OR_PERSISTENCE_FAILURE'
    try:
        atomic_write_json(run_root/'lifecycle.json',dict(status='RUNNING',attempt_id=scope['attempt_id'],source_sha=scope['source_sha']))
        for index,row in enumerate(selected):
            order=('A','B','C')[index%3:]+('A','B','C')[:index%3]
            for arm in order:
                result=run_logical(arm,row,scope,transport,ledger,receipts,sleep=sleep,dispatch_guard=dispatch_guard)
                completed.append(result)
                envelope=dict(scope_sha256=scope['startup_identity_sha256'],completed=completed)
                envelope['integrity_sha256']=digest(envelope)
                atomic_write_json(run_root/'checkpoint_private.json',envelope)
                progress(json.dumps(dict(completed_logicals=len(completed),expected_logicals=3*len(selected),
                    physical_attempts=ledger.state['physical_attempts'],charged_tokens=ledger.state['charged_total'],
                    reserved_tokens=ledger.state['reserved_total']),sort_keys=True))
        status='EXECUTION_COMPLETE';reason='FIXED_PAIRED_PANEL_COMPLETE'
    except BaseException as error:
        reason=str(error) if isinstance(error,CalibrationAbort) else type(error).__name__
    finally:
        try:
            ledger.terminal(status)
            summary=dict(status=status,termination_reason=reason,completed_logicals=len(completed),expected_logicals=3*len(selected),
                accounting=ledger.state,source_sha=scope['source_sha'],startup_identity_sha256=scope['startup_identity_sha256'],
                validation_calls=0,test_calls=0,shadow_calls=0,optimizer_calls=0)
            atomic_write_json(run_root/'execution_summary.json',summary)
            atomic_write_json(run_root/'lifecycle.json',summary)
        finally: ledger.close()
    return summary


def execute_paid(root,prep):
    scope,plan,selected=verify_bundle(root,prep,credential_check=True)
    marker=Path(root)/'runs/math_baseline_authorization_consumed'/ (scope['startup_identity_sha256']+'.json')
    run_root=Path(root)/scope['run_root']
    if marker.exists() or run_root.exists(): raise CalibrationAbort('CALIBRATION_SINGLE_USE_SCOPE_ALREADY_CONSUMED')
    transport,client=create_exact_transport(scope)  # configuration only, no request
    marker.parent.mkdir(parents=True,exist_ok=True)
    try:
        # Exclusive creation precedes every network request and cannot be reset.
        with marker.open('x',encoding='utf-8',newline='\n') as handle:
            json.dump(dict(scope_sha256=scope['startup_identity_sha256'],source_sha=scope['source_sha']),handle)
            handle.flush();os.fsync(handle.fileno())
        run_root.mkdir()
        atomic_write_json(run_root/'consumed_authorization_private.json',scope)
        def guard():
            if (git(root,'rev-parse','HEAD')!=scope['source_sha']
                    or any(sha(Path(root)/f['path'])!=f['sha256'] for f in scope['source_files'])):
                raise CalibrationAbort('CALIBRATION_FROZEN_SOURCE_CHANGED_DURING_EXECUTION')
        summary=run_panel(selected,scope,run_root,transport,dispatch_guard=guard)
        return {k:v for k,v in summary.items() if k!='accounting'} | dict(
            charged_tokens=summary['accounting']['charged_total'],provider_attempts=summary['accounting']['physical_attempts'])
    finally: client.close()
