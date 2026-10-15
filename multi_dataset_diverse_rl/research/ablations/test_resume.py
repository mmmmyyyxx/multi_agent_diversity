"""Fresh, test-only continuation; original training and aborted Test are immutable."""
from copy import deepcopy
from pathlib import Path
import hashlib
from ...benchmarks.data_freeze import digest,file_hash
from ...benchmarks.math_domain_v2 import SETTINGS
from ...persistence.durable_io import read_json,atomic_write_json
from ...persistence.provider_receipts import ProviderResponseReceipts
from ...governance.token_accounting import TokenLedger,audit_journal,POLICY_V27_FORMAL_UNCAPPED,serialized_request
from ...search.schemas import SearchContractError
from ...search.system_prompt import SystemPrompt
from ..single_prompt_governance import private_path,require_selected_runtime,StudyHeldoutAccess
from . import governance as gov
from .analysis import summarize
from .test_resume_parent import PARENT,PIN,PIN_SHA256,parent_pin,audit_parent
from .test_resume_runtime import TestResumeBroker

MANIFEST='experiments/manifests/single_prompt_c_test_resume_v8.json'
PROTOCOL='docs/design/SINGLE_PROMPT_C_TEST_RESUME_V8.md'
SCHEMA='single_prompt_c_test_resume_manifest_v8'
PREP_SCHEMA='single_prompt_c_test_resume_prep_v8'


def operation_policy():
    return dict(identity='TEST_ONLY_SAME_ANSWER_LOCAL_EVALUATOR_RETRY_V1',roles=['solver'],phase='test',
        local_retry_errors=['MATH_EVALUATOR_TIMEOUT'],backoff_seconds=[2,4,8,16,30,60],
        local_retry_limit=None,local_retry_new_model_calls=0,mathematical_settings=deepcopy(SETTINGS),
        reference_wrapper='retry_only_if_REFERENCE_UNSCORABLE_caused_by_MATH_EVALUATOR_TIMEOUT',
        successful_semantic_and_inflight_ceiling=3600,inherited_successful_calls=735,new_call_ceiling=2865,
        token_ceiling=None,teacher_replay=False,solver_prefix='exact_cell_stage_complete_request_only',
        inherited_completed_cell='retain_verified_C1_300_predictions',
        inherited_partial_cell='reconstruct_from_paid_outputs_verify_297_terminal_observation_hashes',
        other_failure='abort_preserve_evidence',train_access=False,validation_access=False,
        future_all42_rule='not_a_new_repeated_training_experiment_preserve_parent_memberships')


def manifest_template():
    return dict(schema_version=SCHEMA,experiment_id='single_prompt_c_test_resume_v8',
        parents=['experiments/manifests/single_prompt_pattern_c_only_v7.json'],protocol_path=PROTOCOL,
        parent_root=PARENT,parent_pin_path=PIN,parent_pin_sha256=PIN_SHA256,
        operation_policy=operation_policy(),status='PREPARED_NOT_EXECUTED',real_api_authorized=False)


def validate_manifest(root,value):
    if value!=manifest_template():raise SearchContractError('TEST_RESUME_MANIFEST_CHANGED')


def _scope(root,prep):
    validate_manifest(root,read_json(root/MANIFEST));pin=parent_pin(root)
    parent=read_json(root/PARENT/'prep/prep.json')['scope']
    old_test=read_json(root/PARENT/'test_run/scope.json')
    body=dict(study_id='single_prompt_c_test_resume_v8',phase='test',source_sha=gov._git(root,'rev-parse','HEAD'),
        source_hashes=gov.source_snapshot(root),runtime_identity=gov.runtime_identity(),
        manifest_sha256=file_hash(root/MANIFEST),protocol_sha256=file_hash(root/PROTOCOL),
        parent_pin_sha256=PIN_SHA256,parent_train_terminal_sha256=pin['parent_train_terminal_sha256'],
        plan=parent['plan'],test_plan=parent['test_plan'],
        call_ceiling=2865,token_ceiling=None,local_evaluator_recovery=operation_policy(),
        network_recovery=parent['network_recovery'],
        continuation=dict(request_identity_scope=old_test['scope_identity']),
        execution=dict(prep_path=prep.relative_to(root).as_posix(),run_root=(prep.parent/'test_run').relative_to(root).as_posix()))
    body['execution']['command']=[gov.INTERPRETER,'scripts/run_experiment.py','--prep',body['execution']['prep_path'],
        '--run-root',body['execution']['run_root'],'--execute']
    return dict(body,scope_identity=digest(body))


def authorization_template(scope):
    return dict(scope_identity=scope['scope_identity'],source_sha=scope['source_sha'],approved=False,
        test_access=False,train_access=False,validation_access=False,roles=['solver'],
        user_authorization_evidence_sha256=None,
        authorized_attempt_ids=[scope['scope_identity']+'_test_'+c['cell_id'] for c in scope['plan']['cells']])


def handoff_template(root,prep,scope):
    return dict(schema_version='sol_luna_experiment_handoff_v1',experiment_id=scope['study_id'],
        source=dict(repository_commit=scope['source_sha'],tracked_worktree_status='clean'),
        protocol=dict(name='C-only final Test continuation with same-answer local scorer retry',version='V8',
            protocol_sha256=file_hash(root/PROTOCOL),manifest_path=MANIFEST,manifest_sha256=file_hash(root/MANIFEST),
            preregistration_path=PROTOCOL,preregistration_sha256=file_hash(root/PROTOCOL)),
        data=dict(dataset='math',split_identity=scope['test_plan']['split_manifest_sha256'],
            test_membership_sha256=scope['test_plan']['membership_sha256'],test_count=300),
        models=dict(solver='gpt-4o-mini',optimizer_or_reflection='NOT_AUTHORIZED',thinking=False,
            local_optimizer_backend='none_test_only',local_optimizer_version_or_sha256=scope['manifest_sha256']),
        execution=dict(seeds=[42],repeat_ids=[1,2,3],opportunity_budget=0,
            early_stop_rule='finish_remaining_frozen_test_cells_no_efficacy_stop',
            exact_runner_command=scope['execution']['command'],expected_output_directory=scope['execution']['run_root'],
            expected_checkpoint_path=scope['execution']['run_root']+'/lifecycle.json',
            expected_ledger_path=scope['execution']['run_root']+'/ledger',test_solver_concurrency=16,
            new_successful_semantic_and_inflight_ceiling=2865,total_test_successful_ceiling=3600,
            allowed_retries=dict(local=operation_policy(),network=scope['network_recovery'])),
        authorization=dict(api_scope=scope['scope_identity'],validation_access_policy='FORBIDDEN',
            test_access_policy='exact_parent_test300_all_three_final_prompts'),
        fail_closed_conditions=['identity_or_parent_mutation','ledger_receipt_mismatch','non_timeout_scoring_failure',
            'unexpected_role_or_split','changed_parent_terminal_score','budget_overrun'],
        READY_TO_RUN=False,status='FROZEN_AWAITING_EXACT_USER_AUTHORIZATION')


def prepare(root,prep):
    prep=private_path(root,prep);require_selected_runtime(root)
    if prep.exists() or gov._git(root,'status','--porcelain','--untracked-files=no'):
        raise SearchContractError('TEST_RESUME_FRESH_CLEAN_PREPARATION_REQUIRED')
    audit_parent(root);scope=_scope(root,prep)
    if set(scope['source_hashes'])-set(gov._git(root,'ls-files').splitlines()):
        raise SearchContractError('TEST_RESUME_UNCOMMITTED_FREEZE')
    prep.mkdir(parents=True)
    atomic_write_json(prep/'prep.json',dict(schema_version=PREP_SCHEMA,scope=scope))
    atomic_write_json(prep/'authorization.json',authorization_template(scope))
    atomic_write_json(prep/'execution_plan.json',scope['execution'])
    atomic_write_json(prep/'handoff.json',handoff_template(root,prep,scope))
    return dict(status='FROZEN_NOT_AUTHORIZED',source_sha=scope['source_sha'],scope_identity=scope['scope_identity'],real_api_calls=0)


def validate_prep(root,prep,*,authorized=False):
    prep=private_path(root,prep);require_selected_runtime(root)
    value=read_json(prep/'prep.json')
    if (value.get('schema_version')!=PREP_SCHEMA or gov._git(root,'status','--porcelain','--untracked-files=no')
            or value['scope']!=_scope(root,prep)):
        raise SearchContractError('TEST_RESUME_FROZEN_IDENTITY_CHANGED')
    scope=value['scope'];auth=read_json(prep/'authorization.json');expected=authorization_template(scope)
    if (set(auth)!=set(expected) or any(auth[k]!=expected[k] for k in expected if k not in
            ('approved','test_access','user_authorization_evidence_sha256'))
            or type(auth['approved']) is not bool or type(auth['test_access']) is not bool):
        raise SearchContractError('TEST_RESUME_EXACT_AUTHORIZATION_REQUIRED')
    handoff=read_json(prep/'handoff.json');wanted=handoff_template(root,prep,scope)
    if (read_json(prep/'execution_plan.json')!=scope['execution'] or set(handoff)!=set(wanted)
            or any(handoff[k]!=v for k,v in wanted.items() if k not in ('READY_TO_RUN','status'))):
        raise SearchContractError('TEST_RESUME_HANDOFF_CHANGED')
    if authorized and (not auth['approved'] or not auth['test_access'] or not handoff['READY_TO_RUN']
            or file_hash(prep/'user_authorization_evidence.json')!=auth['user_authorization_evidence_sha256']):
        raise SearchContractError('TEST_RESUME_FRESH_USER_AUTHORIZATION_REQUIRED')
    return scope


def audit_new_run(run,scope,usage):
    view,events=audit_journal(run/'ledger',task_sha256=scope['scope_identity'],policy=POLICY_V27_FORMAL_UNCAPPED)
    reserves={e['reservation_id']:e for e in events if e['kind']=='RESERVE'};charges=[e for e in events if e['kind']=='CHARGE']
    if (len(reserves)!=len(charges) or len(reserves)!=usage['provider_calls'] or
            view['charged_total']!=usage['input_tokens']+usage['output_tokens']):
        raise SearchContractError('TEST_RESUME_NEW_ACCOUNTING_MISMATCH')
    failures=0;allowed={scope['scope_identity']+'_test_'+c['cell_id'] for c in scope['plan']['cells']}
    for e in charges:
        r=reserves[e['reservation_id']];ref=e['response_receipt']
        if r['role']!='solver' or r['attempt_id'] not in allowed:raise SearchContractError('TEST_RESUME_ROLE_SCOPE')
        store=ProviderResponseReceipts.__new__(ProviderResponseReceipts)
        store.directory=run/'provider_receipts';store.attempt_id=r['attempt_id'];store.startup=scope['scope_identity']
        record=store.read(e['reservation_id'])
        if (read_json(store.directory/(e['reservation_id']+'.json'))['integrity_sha256']!=ref['integrity_sha256']
                or ref['attempt_id']!=r['attempt_id'] or ref['reservation_id']!=e['reservation_id']
                or hashlib.sha256(serialized_request(record['request'])).hexdigest()!=r['request_sha256']
                or record['logical_identity']!=dict(seed=42,ordinal=0,role='solver',stage=r['stage'])):
            raise SearchContractError('TEST_RESUME_NEW_RECEIPT_CHANGED')
        failures+=record.get('error_category') in ('APITimeoutError','APIConnectionError')
    if failures!=usage['network_failed_calls'] or len(reserves)-failures>scope.get('call_ceiling',2865):
        raise SearchContractError('TEST_RESUME_SEMANTIC_BUDGET')
    return view


def execute(root,prep,run,*,contract=None):
    if contract is None:contract=__import__(__name__,fromlist=['execute'])
    prep,run=private_path(root,prep),private_path(root,run)
    scope=contract.validate_prep(root,prep,authorized=True)
    if run!=root/scope['execution']['run_root'] or run.exists():
        raise SearchContractError('TEST_RESUME_EXACT_FRESH_RUN_REQUIRED')
    parent=contract.audit_parent(root);gov._consume(root,scope,run);run.mkdir(parents=True)
    atomic_write_json(run/'scope.json',scope);atomic_write_json(run/'authorization.json',read_json(prep/'authorization.json'))
    atomic_write_json(run/'lifecycle.json',dict(status='RUNNING',phase='test',cells_completed=0))
    files={p:(root/p).stat() for p in scope['source_hashes']}
    pins={n:file_hash(prep/n) for n in ('prep.json','authorization.json','handoff.json','execution_plan.json')}
    active=None;ledger=clients=broker=None;completed=[]
    def guard(cell):
        if cell!=active or any(file_hash(prep/n)!=sha for n,sha in pins.items()):
            raise SearchContractError('TEST_RESUME_ACTIVE_SCOPE_CHANGED')
        for path,old in files.items():
            now=(root/path).stat()
            if (old.st_size,old.st_mtime_ns)!=(now.st_size,now.st_mtime_ns):
                raise SearchContractError('TEST_RESUME_SOURCE_CHANGED')
    try:
        transport,clients=gov._transport()
        ledger=TokenLedger(run/'ledger',task_sha256=scope['scope_identity'],policy=POLICY_V27_FORMAL_UNCAPPED)
        broker=contract.TestResumeBroker(plan=scope['plan'],phase='test',scope=scope,transport=transport,ledger=ledger,
            receipts=ProviderResponseReceipts(run/'provider_receipts',attempt_id=scope['scope_identity'],startup_identity_sha256=scope['scope_identity']),
            dispatch_guard=guard,paid_prefix=parent['paid_prefix'],terminal_observations=parent['terminal_observations'])
        reader=gov.validate_manifest(root,read_json(root/gov.MANIFEST))
        rows=broker.retry_local(lambda:reader.heldout_rows(StudyHeldoutAccess('test',reader.manifest_sha256,parent['terminal_sha256'],
            lambda:contract.validate_prep(root,prep,authorized=True))),dict(operation='test_membership_reference_check'))
        if [r['stable_example_id'] for r in rows]!=scope['test_plan']['sample_ids']:
            raise SearchContractError('TEST_RESUME_TEST_MEMBERSHIP_CHANGED')
        atomic_write_json(run/'access.json',dict(test_ids=scope['test_plan']['sample_ids'],optimize_access=False,validation_access=False))
        for index,cell in enumerate(scope['plan']['cells']):
            contract.validate_prep(root,prep,authorized=True);active=cell['cell_id'];broker.bind_cell(cell,scope['plan'])
            folder=run/active;folder.mkdir();before=broker.usage_view()
            atomic_write_json(run/'lifecycle.json',dict(status='RUNNING',phase='test',current_cell=active,cells_completed=len(completed)))
            values=(deepcopy(parent['first_result']['predictions']) if index==0 else
                broker.evaluate(SystemPrompt.from_dict(parent['train_results'][index]['final_prompt']),rows,'test',42,0))
            if set(values)!=set(scope['test_plan']['sample_ids']):raise SearchContractError('TEST_RESUME_INCOMPLETE_CELL')
            result=dict(cell=cell,predictions=values,new_usage={k:broker.usage_view()[k]-v for k,v in before.items()},
                inherited_usage=parent['test_view']['by_attempt'].get(parent['test_scope']['scope_identity']+'_test_'+active,{}),
                completed_parent_cell_reused=index==0)
            atomic_write_json(folder/'result.json',result);completed.append(result)
            print(__import__('json').dumps(dict(cell=active,status='CELL_COMPLETE',new_provider_calls=broker.usage_view()['provider_calls'])),flush=True)
        contract.audit_parent(root);contract.validate_prep(root,prep,authorized=True)
        usage=broker.usage_view();view=audit_new_run(run,scope,usage)
        replay_path=run/'provider_receipts/solver_paid_prefix.jsonl'
        replay_rows=[__import__('json').loads(v) for v in replay_path.read_bytes().splitlines()] if replay_path.exists() else []
        if len(replay_rows)!=usage['replayed_calls'] or len(replay_rows)!=len(broker.used_prefix):
            raise SearchContractError('TEST_RESUME_PREFIX_ACCOUNTING')
        reused=set()
        for row in replay_rows:
            slot=(row['cell_id'],row['stage'],row['provider_request_sha256'])
            entry=parent['paid_prefix'].get(slot)
            if (entry is None or slot in reused or row['parent_reservation_id']!=entry['reservation_id']
                    or row['parent_receipt_sha256']!=entry['receipt_sha256']):
                raise SearchContractError('TEST_RESUME_PREFIX_RECEIPT_LINK')
            reused.add(slot)
        if reused!=broker.used_prefix:raise SearchContractError('TEST_RESUME_PREFIX_RECEIPT_LINK')
        predictions={r['cell']['cell_id']:r['predictions'] for r in completed}
        summary=summarize(scope['plan'],parent['train_results'],predictions)
        summary.update(test_continuation=True,historical_test_previously_exposed=True,
            new_test_tokens=view['charged_total'],inherited_test_tokens=parent.get('inherited_test_tokens',786747),
            combined_train_and_test_tokens=parent.get('inherited_all_tokens',37131963)+view['charged_total'],
            new_test_physical_calls=usage['provider_calls'],inherited_all_physical_calls=parent.get('inherited_all_calls',21586),
            combined_all_physical_calls=parent.get('inherited_all_calls',21586)+usage['provider_calls'],
            local_evaluator_retries=broker.local_retry_count,paid_solver_responses_reused=usage['replayed_calls'],
            completed_parent_test_predictions_reused=len(parent['first_result']['predictions']))
        atomic_write_json(run/'results.json',completed);atomic_write_json(run/'summary.json',summary)
        atomic_write_json(run/'execution_summary.json',dict(usage=usage,token_ledger=view,local_evaluator_retries=broker.local_retry_count,
            inherited_train_tokens=36345216,inherited_test_tokens=parent.get('inherited_test_tokens',786747)))
        ledger.close();ledger=None;clients.close();clients=None
        atomic_write_json(run/'lifecycle.json',dict(status='EXECUTION_COMPLETE',phase='test',cells_completed=3))
        atomic_write_json(run/'terminal.json',dict(status='EXECUTION_COMPLETE',phase='test',source_sha=scope['source_sha'],
            scope_identity=scope['scope_identity'],artifacts=gov.inventory(run)))
        return dict(status='EXECUTION_COMPLETE',cells_completed=3)
    except BaseException as exc:
        if ledger is not None:
            for ident in tuple(ledger.inflight):ledger.reconcile(ident,{},outcome='ABORT_FULL_RESERVATION_FALLBACK')
            atomic_write_json(run/'abort.json',dict(error_type=type(exc).__name__,reason=str(exc),token_ledger=ledger.view(),
                usage=broker.usage_view() if broker else None))
        atomic_write_json(run/'lifecycle.json',dict(status='EXECUTION_ABORTED',phase='test',current_cell=active,cells_completed=len(completed),error_type=type(exc).__name__))
        raise
    finally:
        if ledger is not None:ledger.close()
        if clients is not None:clients.close()
