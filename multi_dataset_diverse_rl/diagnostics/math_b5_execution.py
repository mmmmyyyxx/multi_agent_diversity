"""Separate governed B-five-member diagnostic runner; never constructs search."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from .math_baseline_followup import (
    PLAN_ID, CAP, initial_team, membership_identity, choose_fresh_membership,
)
from .math_baseline_calibration import digest, native_extract, TRUNCATED
from .math_baseline_audit import optimize_rows
from .math_calibration_plan import request_for, SAMPLING
from .math_b5_accounting import CalibrationLedger, CalibrationAbort
# Only the exact-wire provider transport and response checks are shared with
# Stage 1. Its startup, ledger, authorization and logical executor are not used.
from .math_calibration_execution import create_exact_transport, validate_response, sha, git, RETRYABLE
from ..governance.token_accounting import serialized_request
from ..persistence.durable_io import atomic_write_json, read_json
from ..persistence.provider_receipts import ProviderResponseReceipts

SPEC_PATH='experiments/manifests/math_b_five_member_calibration_v1.json'
PLAN_PATH='experiments/protocols/math_b_five_member_calibration_v1/protocol.json'
TEAM_PATH='experiments/initial_teams/math_b_calibrated_team_seed_v1.json'
ATTEMPT='math_b_five_member_seed82_20261010_attempt1'
POLICY=dict(sampling=SAMPLING,
    retries=dict(successful_draws=4,first_capacity=3600,truncation_capacity=6144,
        expansion='IMMEDIATELY_PREVIOUS_SUCCESS_LENGTH_ONLY',trigger='NATIVE_INVALID_ONLY',
        physical_attempts_per_draw=21,same_body_per_transport_retry=True),
    provider=dict(profile='lwj',timeout_seconds=120,sdk_retries=0,retry_categories=sorted(RETRYABLE),
        sleep_seconds=1.5,backoff_ceiling_seconds=60,returned_alias='qwen3-8b',weight_revision='UNKNOWN_UNLESS_RETURNED'),
    order='ROTATE_01234_BY_EXAMPLE_INDEX',solver_concurrency=1,optimizer_concurrency=0,
    cache='FRESH_NAMESPACE_NO_RESPONSE_REUSE_MEMBER_DRAW_PHYSICAL_DISTINCT',
    stability_gate=dict(first_valid_min=0.90,recovered_valid_min=0.95,
        recovered_valid_member_min=0.90,recovery_charge_fraction_max=0.50),
    stopping='FIXED_60_BY_5_OR_NATIVE_INVALID_FOUR_DRAW_RECOVERY_OR_OPERATIONAL_FAIL_CLOSED',
    incomplete='PARTIAL_FACTS_ONLY_NO_EFFICACY_COMPARISON',scientific_results_used_to_stop=False)


def source_inventory(root):
    from ..governance.source_identity import local_imports
    pending=[root/'scripts/run_math_b_five_member_calibration.py'];seen=set()
    while pending:
        path=pending.pop()
        if path in seen or not path.is_file():continue
        seen.add(path);pending.extend(local_imports(root,path,initializers=True)-seen)
    seen.update(root/p for p in (SPEC_PATH,PLAN_PATH,TEAM_PATH,'AGENTS.md',
        'docs/design/CURRENT_SPEC.md','experiments/current_frontier.yaml',
        'docs/failures/registry.yaml','docs/REPOSITORY_MAP.md',
        'docs/workflows/MATH_B_FIVE_MEMBER_CALIBRATION_V1.md','experiments/registry.yaml','experiments/lineage.yaml',
        'scripts/tooling_classification.json','tests/suite_classification.json',
        'experiments/execution_bindings/a4_v25_seed81_canary_attempt1.json'))
    return [dict(path=p.relative_to(root).as_posix(),sha256=sha(p)) for p in sorted(seen)]


def load_plan(root):
    plan=read_json(root/PLAN_PATH)
    if (digest({k:v for k,v in plan.items() if k!='protocol_sha256'})!=plan['protocol_sha256']
            or plan['identity']!=PLAN_ID or plan['n']!=60 or plan['members']!=5 or plan['seed']!=82
            or plan['cap']!=CAP or plan['arms']!=['B']
            or plan['native_parser']!='MATH_EXPLICIT_FINAL_ANSWER_EXTRACTION_V2'
            or plan['aggregation']!='equal_equivalence_plurality_consistency_v1'
            or digest(plan['policy'])!=digest(POLICY)
            or plan['access']!=dict(optimize=True,shadow=False,validation=False,test=False)
            or plan['real_api_authorized'] is not False or plan['READY_TO_RUN'] is not False
            or read_json(root/TEAM_PATH)!=initial_team()
            or sha(root/TEAM_PATH)!=plan['team_file_sha256']
            or digest(plan['membership'])!=plan['membership_sha256']):
        raise CalibrationAbort('B5_PROTOCOL_OR_TEAM_MISMATCH')
    if read_json(root/SPEC_PATH)!=dict(schema_version='independent_math_b5_preregistration_v1',
            experiment_id='math_b_five_member_calibration_v1',parent_ids=['math_baseline_calibration_stage1_v1'],
            protocol_path=PLAN_PATH,protocol_sha256=plan['protocol_sha256'],
            membership_sha256=plan['membership_sha256'],team_sha256=initial_team()['ordered_team_sha256'],
            status='FROZEN_AWAITING_FRESH_AUTHORIZATION',READY_TO_RUN=False,real_api_authorized=False):
        raise CalibrationAbort('B5_MANIFEST_MISMATCH')
    return plan


def load_selection(root,plan):
    binding=read_json(root/'experiments/execution_bindings/a4_v25_seed81_canary_attempt1.json')
    if sha(root/'experiments/execution_bindings/a4_v25_seed81_canary_attempt1.json')!=plan['original_binding_sha256']:
        raise CalibrationAbort('B5_BINDING_MISMATCH')
    rows,_=optimize_rows(root,binding)
    by_hash={hashlib.sha256(r['stable_example_id'].encode()).hexdigest():r for r in rows}
    selected=[by_hash[m['example_id_sha256']] for m in plan['membership']]
    if membership_identity(selected)!=plan['membership']:
        raise CalibrationAbort('B5_SELECTED_DATA_MISMATCH')
    exclusion={r['stable_example_id'] for r in rows if hashlib.sha256(r['stable_example_id'].encode()).hexdigest() in plan['excluded_example_hashes']}
    if len(exclusion)!=72 or choose_fresh_membership(rows,exclusion)[0]!=selected:
        raise CalibrationAbort('B5_DISJOINT_SELECTION_MISMATCH')
    return selected


def freeze_review(root,prep):
    """Freeze a concrete review scope with readiness FALSE and no API authority."""
    root=Path(root);prep=Path(prep)
    if git(root,'branch','--show-current')!='main' or git(root,'status','--porcelain','--untracked-files=no'):
        raise CalibrationAbort('B5_CLEAN_MAIN_REQUIRED')
    if prep.exists() or not prep.resolve().is_relative_to((root/'runs').resolve()):
        raise CalibrationAbort('B5_FRESH_PRIVATE_PREPARATION_REQUIRED')
    if Path(sys.executable).resolve()!=(root/'runs/math_v2_1/runtime/Scripts/python.exe').resolve():
        raise CalibrationAbort('B5_PINNED_INTERPRETER_REQUIRED')
    plan=load_plan(root);selected=load_selection(root,plan)
    endpoint=os.environ.get('LWJ_DASHSCOPE_BASE_URL','').rstrip('/')
    if not endpoint:raise CalibrationAbort('B5_ENDPOINT_CONFIGURATION_MISSING')
    from ..benchmarks.math_worker import PINS
    scope=dict(identity=PLAN_ID,attempt_id=ATTEMPT,phase='b_five_member_development',roles=['solver'],
        source_sha=git(root,'rev-parse','HEAD'),source_files=source_inventory(root),
        protocol_sha256=plan['protocol_sha256'],protocol_file_sha256=sha(root/PLAN_PATH),
        manifest_sha256=sha(root/SPEC_PATH),membership_sha256=plan['membership_sha256'],
        selected_rows_sha256=digest(selected),team_sha256=initial_team()['ordered_team_sha256'],
        model='qwen3-8b',enable_thinking=False,n=60,members=5,seed=82,arms=['B'],
        solver_concurrency=1,optimizer_concurrency=0,authorized_total=CAP,
        provider_endpoint_sha256=hashlib.sha256(endpoint.encode()).hexdigest(),
        timeout_seconds=120,sdk_retries=0,
        interpreter_sha256=sha(root/'runs/math_v2_1/runtime/Scripts/python.exe'),
        dependency_versions={name:importlib.metadata.version(name) for name in (*PINS,'openai','httpx')},
        validation_access=False,test_access=False,shadow_access=False,old_scope_reuse=False,
        run_root='runs/'+ATTEMPT,READY_TO_RUN=False,real_api_authorized=False)
    scope['review_scope_sha256']=digest(scope)
    prep.mkdir(parents=True)
    atomic_write_json(prep/'review_scope_private.json',scope)
    atomic_write_json(prep/'selected_rows_private.json',selected)
    return scope


def authorize_review(root,prep,authorization):
    """Owner-only call after a new exact user authorization; not a CLI bypass."""
    scope,_,_=verify_review(root,prep)
    expected=dict(experiment_id='math_b_five_member_calibration_v1',
        review_scope_sha256=scope['review_scope_sha256'],source_sha=scope['source_sha'],
        protocol_sha256=scope['protocol_sha256'],membership_sha256=scope['membership_sha256'],
        phase=scope['phase'],roles=['solver'],n=60,members=5,cap=CAP)
    if (any(authorization.get(k)!=v for k,v in expected.items())
            or authorization.get('explicit_user_authorization') is not True
            or not isinstance(authorization.get('exact_user_message'),str)
            or not authorization['exact_user_message'].strip()):
        raise CalibrationAbort('B5_FRESH_EXACT_AUTHORIZATION_REQUIRED')
    prep=Path(prep)
    if (prep/'frozen_manifest_private.json').exists():
        raise CalibrationAbort('B5_AUTHORIZATION_ALREADY_FROZEN')
    frozen=scope|dict(READY_TO_RUN=True,real_api_authorized=True,user_authorization_sha256=digest(authorization))
    frozen['startup_identity_sha256']=digest(frozen)
    atomic_write_json(prep/'authorization_private.json',authorization)
    atomic_write_json(prep/'frozen_manifest_private.json',frozen)
    atomic_write_json(prep/'handoff_private.json',expected_handoff(Path(root),prep,frozen))
    return frozen


def expected_handoff(root,prep,frozen):
    binding=read_json(root/'experiments/execution_bindings/a4_v25_seed81_canary_attempt1.json')
    return dict(schema_version='sol_luna_experiment_handoff_v1',experiment_id='math_b_five_member_calibration_v1',
        source=dict(repository_commit=frozen['source_sha'],tracked_worktree_status='clean'),
        protocol=dict(name=PLAN_ID,version='v1',protocol_sha256=frozen['protocol_sha256'],
            manifest_path=(prep.relative_to(root)/'frozen_manifest_private.json').as_posix(),
            manifest_sha256=sha(prep/'frozen_manifest_private.json'),preregistration_path=SPEC_PATH,
            preregistration_sha256=sha(root/SPEC_PATH)),
        data=dict(dataset='math',split_identity=binding['split_version'],split_sha256=binding['split_manifest_sha256']),
        models=dict(solver='qwen3-8b',optimizer_or_reflection='DISABLED',thinking=False,
            local_optimizer_backend='DISABLED',local_optimizer_version_or_sha256='DISABLED'),
        execution=dict(seeds=[82],opportunity_budget=0,early_stop_rule=POLICY['stopping'],
            exact_runner_command='runs/math_v2_1/runtime/Scripts/python.exe scripts/run_math_b_five_member_calibration.py --execute --prep '+prep.relative_to(root).as_posix(),
            expected_output_directory=frozen['run_root'],expected_checkpoint_path=frozen['run_root']+'/checkpoint_private.json',
            expected_ledger_path=frozen['run_root']+'/accounting/events.jsonl',allowed_retries=POLICY['retries']),
        authorization=dict(api_scope=frozen['startup_identity_sha256'],validation_access_policy='FORBIDDEN',test_access_policy='FORBIDDEN'),
        fail_closed_conditions=['SOURCE_DATA_IDENTITY_OR_HANDOFF_MISMATCH','PROVIDER_OR_CREDENTIAL_FAILURE',
            'PERSISTENCE_CHECKPOINT_LEDGER_FAILURE','BUDGET_OR_RESOURCE_CEILING','HELDOUT_ACCESS','NO_RESUME_OR_RERUN'],READY_TO_RUN=True)


def verify_review(root,prep):
    root=Path(root);prep=Path(prep);scope=read_json(prep/'review_scope_private.json')
    if (digest({k:v for k,v in scope.items() if k!='review_scope_sha256'})!=scope['review_scope_sha256']
            or scope['source_sha']!=git(root,'rev-parse','HEAD')
            or git(root,'branch','--show-current')!='main'
            or git(root,'status','--porcelain','--untracked-files=no')
            or source_inventory(root)!=scope['source_files']
            or scope['READY_TO_RUN'] is not False or scope['real_api_authorized'] is not False):
        raise CalibrationAbort('B5_REVIEW_SOURCE_MISMATCH')
    plan=load_plan(root);selected=read_json(prep/'selected_rows_private.json')
    required=dict(identity=PLAN_ID,attempt_id=ATTEMPT,phase='b_five_member_development',roles=['solver'],
        model='qwen3-8b',enable_thinking=False,n=60,members=5,seed=82,arms=['B'],
        solver_concurrency=1,optimizer_concurrency=0,authorized_total=CAP,
        timeout_seconds=120,sdk_retries=0,validation_access=False,test_access=False,shadow_access=False,
        old_scope_reuse=False,run_root='runs/'+ATTEMPT,protocol_sha256=plan['protocol_sha256'],
        membership_sha256=plan['membership_sha256'],team_sha256=initial_team()['ordered_team_sha256'])
    if any(scope.get(k)!=v or type(scope.get(k)) is not type(v) for k,v in required.items()):
        raise CalibrationAbort('B5_REVIEW_CLOSED_SCOPE_MISMATCH')
    if (sha(root/PLAN_PATH)!=scope['protocol_file_sha256'] or sha(root/SPEC_PATH)!=scope['manifest_sha256']
            or digest(selected)!=scope['selected_rows_sha256'] or selected!=load_selection(root,plan)
            or Path(sys.executable).resolve()!=(root/'runs/math_v2_1/runtime/Scripts/python.exe').resolve()
            or sha(Path(sys.executable))!=scope['interpreter_sha256']
            or any(importlib.metadata.version(k)!=v for k,v in scope['dependency_versions'].items())):
        raise CalibrationAbort('B5_FROZEN_DATA_OR_RUNTIME_MISMATCH')
    return scope,plan,selected


def verify_bundle(root,prep,credential_check=False):
    scope,plan,selected=verify_review(root,prep)
    frozen=read_json(Path(prep)/'frozen_manifest_private.json')
    authorization=read_json(Path(prep)/'authorization_private.json')
    required=dict(experiment_id='math_b_five_member_calibration_v1',review_scope_sha256=scope['review_scope_sha256'],
        source_sha=scope['source_sha'],protocol_sha256=scope['protocol_sha256'],membership_sha256=scope['membership_sha256'],
        phase=scope['phase'],roles=['solver'],n=60,members=5,cap=CAP,explicit_user_authorization=True)
    if (any(authorization.get(k)!=v or type(authorization.get(k)) is not type(v) for k,v in required.items())
            or not isinstance(authorization.get('exact_user_message'),str) or not authorization['exact_user_message'].strip()):
        raise CalibrationAbort('B5_FRESH_EXACT_AUTHORIZATION_REQUIRED')
    expected=scope|dict(READY_TO_RUN=True,real_api_authorized=True,user_authorization_sha256=digest(authorization))
    expected['startup_identity_sha256']=digest(expected)
    if frozen!=expected:
        raise CalibrationAbort('B5_AUTHORIZED_STARTUP_MISMATCH')
    if read_json(Path(prep)/'handoff_private.json')!=expected_handoff(Path(root),Path(prep),frozen):
        raise CalibrationAbort('B5_HANDOFF_MISMATCH')
    if credential_check:
        endpoint=os.environ.get('LWJ_DASHSCOPE_BASE_URL','').rstrip('/')
        if not os.environ.get('LWJ_DASHSCOPE_API_KEY') or hashlib.sha256(endpoint.encode()).hexdigest()!=scope['provider_endpoint_sha256']:
            raise CalibrationAbort('B5_PROVIDER_CONFIGURATION_CHANGED')
    return frozen,plan,selected


def execute_paid(root,prep):
    root=Path(root)
    scope,_,selected=verify_bundle(root,prep,credential_check=True)
    marker=root/'runs/math_b5_authorization_consumed'/(scope['startup_identity_sha256']+'.json')
    run_root=root/scope['run_root']
    if marker.exists() or run_root.exists():raise CalibrationAbort('B5_SINGLE_USE_SCOPE_CONSUMED')
    transport,client=create_exact_transport(scope)
    marker.parent.mkdir(parents=True,exist_ok=True)
    try:
        with marker.open('x',encoding='utf-8',newline='\n') as handle:
            json.dump(dict(scope_sha256=scope['startup_identity_sha256'],source_sha=scope['source_sha']),handle)
            handle.flush();os.fsync(handle.fileno())
        run_root.mkdir()
        atomic_write_json(run_root/'consumed_authorization_private.json',scope)
        def guard():
            check_dispatch_source(root,scope)
        return run_panel(selected,scope,run_root,transport,dispatch_guard=guard)
    finally:client.close()


def check_dispatch_source(root,scope):
    if (git(root,'rev-parse','HEAD')!=scope['source_sha']
            or git(root,'status','--porcelain','--untracked-files=no')
            or any(sha(Path(root)/f['path'])!=f['sha256'] for f in scope['source_files'])):
        raise CalibrationAbort('B5_SOURCE_CHANGED_DURING_EXECUTION')


def run_logical(member,row,scope,transport,ledger,receipts,*,sleep=time.sleep,dispatch_guard=lambda:None):
    arm="B"
    attempts=[];previous=None;physical_retries=0
    xid=row['stable_example_id'];xhash=hashlib.sha256(xid.encode()).hexdigest()
    for ordinal in range(1,5):
        capacity=6144 if previous in TRUNCATED else 3600
        request=request_for(arm,row['content']['problem'],capacity)
        for physical in range(1,22):
            dispatch_guard()
            lane=dict(stage='b_five_member_development',arm=arm,example_id_sha256=xhash,member=member,
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
            try:validate_response(response,request)
            except RuntimeError as error:raise CalibrationAbort(str(error)) from None
            break
        native=native_extract(arm,response['text'],response['finish_reason'])
        attempts.append(dict(lane=lane,reservation_id=key,receipt=receipt,request=request,response=response,
            native=native,charged_tokens=charged,output_capacity=capacity))
        previous=response['finish_reason']
        if native['valid']:break
    return dict(example_id=xid,example_id_sha256=xhash,arm=arm,member=member,stage='b_five_member_development',
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
            order=tuple(range(5))[index%5:]+tuple(range(5))[:index%5]
            for member in order:
                result=run_logical(member,row,scope,transport,ledger,receipts,sleep=sleep,dispatch_guard=dispatch_guard)
                completed.append(result)
                envelope=dict(scope_sha256=scope['startup_identity_sha256'],completed=completed)
                envelope['integrity_sha256']=digest(envelope)
                atomic_write_json(run_root/'checkpoint_private.json',envelope)
                progress(json.dumps(dict(completed_logicals=len(completed),expected_logicals=5*len(selected),
                    physical_attempts=ledger.state['physical_attempts'],charged_tokens=ledger.state['charged_total'],
                    reserved_tokens=ledger.state['reserved_total']),sort_keys=True))
        status='EXECUTION_COMPLETE';reason='FIXED_FIVE_MEMBER_PANEL_COMPLETE'
    except BaseException as error:
        reason=str(error) if isinstance(error,CalibrationAbort) else type(error).__name__
    finally:
        try:
            ledger.terminal(status)
            summary=dict(status=status,termination_reason=reason,completed_logicals=len(completed),expected_logicals=5*len(selected),
                accounting=ledger.state,source_sha=scope['source_sha'],startup_identity_sha256=scope['startup_identity_sha256'],
                validation_calls=0,test_calls=0,shadow_calls=0,optimizer_calls=0)
            atomic_write_json(run_root/'execution_summary.json',summary)
            atomic_write_json(run_root/'lifecycle.json',summary)
        finally: ledger.close()
    return summary
