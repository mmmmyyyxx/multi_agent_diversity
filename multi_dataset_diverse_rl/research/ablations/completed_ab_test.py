"""Fresh Solver-only Test of six completed historical A/B cells, never training."""
from copy import deepcopy
from pathlib import Path

from ...benchmarks.data_freeze import digest,file_hash
from ...governance.token_accounting import TokenLedger,POLICY_V27_FORMAL_UNCAPPED,POLICY_GATED_32_UNCAPPED,audit_journal
from ...persistence.durable_io import read_json,atomic_write_json
from ...persistence.provider_receipts import ProviderResponseReceipts
from ...search.schemas import SearchContractError
from ...search.system_prompt import SystemPrompt
from ..single_prompt_governance import StudyHeldoutAccess
from . import governance as gov
from .config import batch_config,build_plan
from .analysis import summarize
from .test_certified import TestResumeBroker
from .test_resume import audit_new_run

PARENT='runs/ablations/pattern_seed42_repeat3_20261015'
PARENT_SOURCE='241f725797080b1b5fc714834b222b675a09d0e8'
PARENT_SCOPE='624d623e7ef83118f58d953b7b34cdd01110b7d1e07a8bbaba4b001267d6c1d6'
MANIFEST='experiments/manifests/single_prompt_completed_ab_test_v14.json'
PROTOCOL='docs/design/SINGLE_PROMPT_COMPLETED_AB_TEST_V14.md'
PREP_SCHEMA='single_prompt_completed_ab_test_prep_v14'
FAILED_ATTEMPT='runs/ablations/completed_ab_test_v13_20261016'


def require(value,reason='AB_TEST_COMPLETED_PARENT_CHANGED'):
    if not value:raise SearchContractError(reason)


def audit_policy():
    return dict(identity='BOUNDED_ADJACENT_CHECKPOINT_AUDIT_V1',
        live_full_checkpoints=2,initial_check='all_historical_semantics_and_full_file_inventory',
        boundary_check='rehash_all_audited_source_and_parent_files',
        historical_outcomes='unchanged_no_new_guard_or_scoring',persisted_audit_cache=False)


def audit_cell_checkpoints(folder,cell,result,scope):
    """Retain two adjacent checkpoints, never a complete trajectory of answers."""
    previous=None;history=[];metrics=[];first_scope=None
    for n in range(21):
        cp=read_json(folder/f'checkpoint_{n}.json')
        require(cp['integrity_sha256']==digest({k:v for k,v in cp.items() if k!='integrity_sha256'}))
        if n==0:
            first_scope=cp['scope']
            require(SystemPrompt.from_dict(cp['prompt']).prompt_hash==scope['initial_prompt_sha256'])
            require(first_scope['cell']==cell and first_scope['plan_identity']==scope['plan']['plan_identity'])
        require(cp['scope_identity']==result['scope_identity']==digest(first_scope))
        require(set(cp['optimize'])==set(scope['optimize_sample_ids']) and set(cp['validation'])==set(scope['gate_membership']['sample_ids']))
        for key in ('optimize','validation'):
            require(all(type(v['correct']) is bool and type(v['valid']) is bool and (not v['correct'] or v['valid']) for v in cp[key].values()))
        history.append(cp['prompt'])
        if n:
            m=cp['metric'];candidate=cp['candidate_prompt'];metrics.append(m)
            require(cp['parent_prompt']==previous['prompt'] and m['round']==n)
            before=sum(v['correct'] for v in previous['validation'].values())
            require(m['validation_before']==before)
            if candidate is not None:
                require(sum(candidate[k]!=previous['prompt'][k] for k in ('role','strategy','answer'))==1)
                require(set(cp['candidate_optimize'])==set(previous['optimize']) and set(cp['candidate_validation'])==set(previous['validation']))
                require(all(type(v['correct']) is bool and type(v['valid']) is bool and (not v['correct'] or v['valid'])
                    for k in ('candidate_optimize','candidate_validation') for v in cp[k].values()))
                gain=sum(v['correct'] for v in cp['candidate_validation'].values())>before
                require(m['accepted'] is gain and m['validation_candidate']==sum(v['correct'] for v in cp['candidate_validation'].values()))
                require(m['decision']==('ACCEPTED_STRICT_VALIDATION_GAIN' if gain else 'REJECTED_NO_STRICT_VALIDATION_GAIN'))
            else:
                require(not m['accepted'] and m['decision']=='ABANDONED_GENERATION_RETRIES_EXHAUSTED' and m['generation_draws']==4)
            for key in ('prompt','optimize','validation'):
                require(cp[key]==(cp['candidate_'+key] if m['accepted'] else previous[key]))
        previous=cp
    require(result['prompt_history']==history and result['metrics']==metrics)
    require(result['final_prompt']==previous['prompt'])
    require(result['final_optimize_correct']==sum(v['correct'] for v in previous['optimize'].values()))
    require(result['final_validation_correct']==sum(v['correct'] for v in previous['validation'].values()))


def parent_evidence(root):
    """Check stored outcomes, not a new Guard judgment or a new measured score."""
    base=root/PARENT;scope=read_json(base/'prep/prep.json')['scope']
    require(scope['source_sha']==PARENT_SOURCE and scope['scope_identity']==PARENT_SCOPE)
    require(digest({k:v for k,v in scope.items() if k!='scope_identity'})==PARENT_SCOPE)
    require(file_hash(root/gov.MANIFEST)==scope['manifest_sha256'])
    require(read_json(root/gov.MANIFEST)['plan']==scope['plan'])
    require(read_json(base/'prep/authorization.json')['approved'] is True)
    require(read_json(base/'train_run/access.json')['test_access'] is False)
    require(read_json(base/'train_run/scope.json')==scope)
    lifecycle=read_json(base/'train_run/lifecycle.json')
    require(lifecycle['status']=='EXECUTION_ABORTED' and lifecycle['cells_completed']==6)
    require(not (base/'test_run').exists())
    ledger,events=audit_journal(base/'train_run/ledger',task_sha256=PARENT_SCOPE,policy=POLICY_GATED_32_UNCAPPED)
    del events  # The journal is fully verified; do not retain its full event graph.
    require(ledger==read_json(base/'train_run/abort.json')['token_ledger'])
    cells=scope['plan']['cells'][:6]
    require([(c['arm_id'],c['repeat_id'],c['seed']) for c in cells]==[(a,r,42) for a in 'AB' for r in (1,2,3)])
    inventory={};results=[]
    for c in cells:
        folder=base/'train_run'/c['cell_id'];r=read_json(folder/'result.json')
        require(r['cell']==c and r['rounds_completed']==20 and len(r['metrics'])==20)
        audit_cell_checkpoints(folder,c,r,scope)
        charged=ledger['by_attempt'][PARENT_SCOPE+'_train_'+c['cell_id']]
        require(r['usage']['provider_calls']==charged['physical_attempts'])
        require(all(r['usage'][k]==charged[k] for k in ('input_tokens','output_tokens')))
        SystemPrompt.from_dict(r['final_prompt']);results.append(r)
        inventory.update({(Path(PARENT)/'train_run'/c['cell_id']/p).as_posix():h for p,h in gov.inventory(folder).items()})
    for p in ('prep/prep.json','prep/authorization.json','train_run/scope.json','train_run/access.json','train_run/lifecycle.json','train_run/abort.json'):
        inventory[(Path(PARENT)/p).as_posix()]=file_hash(base/p)
    inventory[gov.MANIFEST]=file_hash(root/gov.MANIFEST)
    inventory.update({(Path(PARENT)/'train_run/ledger'/p).as_posix():h for p,h in gov.inventory(base/'train_run/ledger').items()})
    return scope,results,inventory


def test_only_plan():
    c=batch_config();c['arms']=c['arms'][:2];c['experiment_id']='single_prompt_completed_ab_test_v14'
    return build_plan(c)


def operation_policy():
    from .test_certified import operation_policy as certified
    p=certified();p.update(identity='COMPLETED_AB_TEST_CERTIFIED_V14',successful_semantic_and_inflight_ceiling=7200,
        inherited_successful_calls=0,new_call_ceiling=7200,solver_prefix='none_no_prior_AB_Test',
        inherited_completed_cell='none',inherited_partial_cell='none')
    return p


def manifest_template():
    return dict(schema_version='single_prompt_completed_ab_test_manifest_v14',experiment_id='single_prompt_completed_ab_test_v14',
        parents=['experiments/manifests/single_prompt_completed_ab_test_v13.json'],
        protocol_path=PROTOCOL,parent_root=PARENT,parent_source=PARENT_SOURCE,parent_scope=PARENT_SCOPE,
        operation_policy=operation_policy(),audit_policy=audit_policy(),status='PREPARED_NOT_EXECUTED',real_api_authorized=False)


def scope_template(root,prep,*,parent_snapshot=None):
    require(read_json(root/MANIFEST)==manifest_template(),'AB_TEST_MANIFEST_CHANGED')
    old,results,inventory=parent_evidence(root) if parent_snapshot is None else parent_snapshot
    if parent_snapshot is not None:
        verify_parent_inventory(root,inventory)
    plan=test_only_plan()
    # Only these read points affect Solver provider requests/recovery/scoring.
    keys=('provider_routing_policy','solver_decoding_policy','first_solver_capacity','length_recovery_capacity',
          'solver_semantic_draws','trajectory_policy','test_solver_concurrency','provider_timeout_seconds')
    require(all(old['plan']['config']['runtime'][k]==plan['config']['runtime'][k] for k in keys),'AB_TEST_SOLVER_CONTRACT_CHANGED')
    hashes=gov.source_snapshot(root);hashes[MANIFEST]=file_hash(root/MANIFEST)
    failed=root/FAILED_ATTEMPT
    lifecycle=read_json(failed/'test_run/lifecycle.json');abort=read_json(failed/'test_run/abort.json')
    require(lifecycle['status']=='EXECUTION_ABORTED' and lifecycle['error_type']=='MemoryError'
        and lifecycle['cells_completed']==0 and abort['usage']['provider_calls']==0
        and abort['token_ledger']['charged_total']==abort['token_ledger']['reserved_inflight']==0,'AB_TEST_FAILED_PREFIX_NOT_ZERO')
    failure_inventory={(Path(FAILED_ATTEMPT)/p).as_posix():h for p,h in gov.inventory(failed).items()}
    value=dict(study_id='single_prompt_completed_ab_test_v14',phase='test',source_sha=gov._git(root,'rev-parse','HEAD'),
        source_hashes=hashes,runtime_identity=gov.runtime_identity(),plan=plan,
        test_plan=deepcopy(old['test_plan']),call_ceiling=7200,token_ceiling=None,
        network_recovery=deepcopy(old['network_recovery']),local_evaluator_recovery=operation_policy(),audit_policy=audit_policy(),
        parent_inventory=inventory,parent_evidence_sha256=digest(inventory),
        prior_zero_call_failure=dict(root=FAILED_ATTEMPT,artifacts=failure_inventory),
        final_prompt_sha256=[SystemPrompt.from_dict(r['final_prompt']).prompt_hash for r in results],
        manifest_sha256=file_hash(root/MANIFEST),protocol_sha256=file_hash(root/PROTOCOL),
        execution=dict(prep_path=prep.relative_to(root).as_posix(),run_root=(prep.parent/'test_run').relative_to(root).as_posix()))
    value['test_plan']['successful_semantic_call_ceiling']=7200
    value['execution']['command']=[gov.INTERPRETER,'scripts/run_experiment.py','--prep',value['execution']['prep_path'],
        '--run-root',value['execution']['run_root'],'--execute']
    return dict(value,scope_identity=digest(value))


def verify_parent_inventory(root,inventory):
    """Read actual bytes on every boundary; mtime and size are insufficient."""
    require(all((root/p).is_file() and file_hash(root/p)==h for p,h in inventory.items()),'AB_TEST_PARENT_FILES_CHANGED')


def authorization_template(scope):
    return dict(scope_identity=scope['scope_identity'],source_sha=scope['source_sha'],approved=False,
        test_access=False,train_access=False,validation_access=False,roles=['solver'],
        user_authorization_evidence_sha256=None,
        authorized_attempt_ids=[scope['scope_identity']+'_test_'+c['cell_id'] for c in scope['plan']['cells']])


def handoff(root,scope):
    return dict(schema_version='sol_luna_experiment_handoff_v1',experiment_id=scope['study_id'],
        source=dict(repository_commit=scope['source_sha'],tracked_worktree_status='clean'),
        protocol=dict(name='Completed historical A/B Test only, bounded audit',version='V14',manifest_path=MANIFEST,
            manifest_sha256=scope['manifest_sha256'],protocol_sha256=scope['protocol_sha256'],
            preregistration_path=PROTOCOL,preregistration_sha256=scope['protocol_sha256']),
        data=dict(dataset='math',split_identity=scope['test_plan']['split_manifest_sha256'],
            test_membership_sha256=scope['test_plan']['membership_sha256'],test_count=300),
        models=dict(solver=scope['plan']['config']['runtime']['provider_routing_policy']['routes']['solver']['model'],
            optimizer_or_reflection='FORBIDDEN',thinking='unchanged_solver_contract',local_optimizer_backend='not_used',local_optimizer_version_or_sha256='not_used'),
        execution=dict(exact_runner_command=scope['execution']['command'],expected_output_directory=scope['execution']['run_root'],
            expected_checkpoint_path=scope['execution']['run_root']+'/A_seed42_repeat01/result.json',
            expected_ledger_path=scope['execution']['run_root']+'/ledger',seeds=[42],repeat_ids=[1,2,3],opportunity_budget=0,
            early_stop_rule='all_six_complete_or_typed_nonrecoverable_failure',allowed_retries=dict(local=operation_policy(),network=scope['network_recovery']),
            semantic_call_ceiling=7200,token_ceiling=None),
        authorization=dict(api_scope=scope['scope_identity'],validation_access_policy='FORBIDDEN',test_access_policy='six_frozen_final_AB_prompts_Test300'),
        fail_closed_conditions=['identity','source','authorization','parent','receipt','ledger','budget','split'],
        READY_TO_RUN=False,status='FROZEN_NOT_AUTHORIZED')


def prepare(root,prep):
    prep=gov.private_path(root,prep);gov.require_selected_runtime(root)
    require(not prep.exists() and not gov._git(root,'status','--porcelain','--untracked-files=no'),'AB_TEST_FRESH_COMMITTED_SOURCE_REQUIRED')
    scope=scope_template(root,prep)
    require(not set(scope['source_hashes'])-set(gov._git(root,'ls-files').splitlines()),'AB_TEST_UNTRACKED_SOURCE')
    prep.mkdir(parents=True)
    for name,value in {'prep':dict(schema_version=PREP_SCHEMA,scope=scope),'authorization':authorization_template(scope),
        'handoff':handoff(root,scope),'execution_plan':scope['execution']}.items():atomic_write_json(prep/(name+'.json'),value)
    return dict(status='FROZEN_NOT_AUTHORIZED',scope_identity=scope['scope_identity'],source_sha=scope['source_sha'])


def validate_prep(root,prep,*,authorized=False,parent_snapshot=None):
    prep=gov.private_path(root,prep);gov.require_selected_runtime(root)
    require(not gov._git(root,'status','--porcelain','--untracked-files=no'),'AB_TEST_SOURCE_DIRTY')
    value=read_json(prep/'prep.json');require(value.get('schema_version')==PREP_SCHEMA,'AB_TEST_SCHEMA')
    scope=value['scope'];require(scope==scope_template(root,prep,parent_snapshot=parent_snapshot),'AB_TEST_FREEZE_CHANGED')
    auth=read_json(prep/'authorization.json');expected=authorization_template(scope);mutable=('approved','test_access','user_authorization_evidence_sha256')
    require(set(auth)==set(expected) and all(auth[k]==expected[k] for k in expected if k not in mutable),'AB_TEST_AUTH_SCOPE')
    require(type(auth['approved']) is bool and type(auth['test_access']) is bool,'AB_TEST_AUTH_TYPE')
    actual=read_json(prep/'handoff.json');wanted=handoff(root,scope)
    require(set(actual)==set(wanted) and all(actual[k]==v for k,v in wanted.items() if k not in ('READY_TO_RUN','status')),'AB_TEST_HANDOFF')
    require(read_json(prep/'execution_plan.json')==scope['execution'],'AB_TEST_COMMAND')
    if authorized:
        require(auth['approved'] and auth['test_access'] and actual['READY_TO_RUN'] is True,'AB_TEST_UNAUTHORIZED')
        require(file_hash(prep/'user_authorization_evidence.json')==auth['user_authorization_evidence_sha256'],'AB_TEST_HUMAN_EVIDENCE')
    return scope


def execute(root,prep,run):
    prep,run=gov.private_path(root,prep),gov.private_path(root,run)
    auth=read_json(prep/'authorization.json')
    require(auth.get('approved') is True and auth.get('test_access') is True,'AB_TEST_UNAUTHORIZED')
    # One full semantic audit in this process, then byte-for-byte boundary checks.
    # No disk cache, prior process assertion, or unchecked runtime input is trusted.
    parent_snapshot=parent_evidence(root)
    scope=validate_prep(root,prep,authorized=True,parent_snapshot=parent_snapshot)
    require(run==root/scope['execution']['run_root'] and not run.exists(),'AB_TEST_FRESH_EXACT_RUN')
    _,parents,_=parent_snapshot;gov._consume(root,scope,run);run.mkdir(parents=True)
    atomic_write_json(run/'scope.json',scope);atomic_write_json(run/'authorization.json',read_json(prep/'authorization.json'))
    pins={n:file_hash(prep/n) for n in ('prep.json','authorization.json','handoff.json','execution_plan.json')}
    signatures={p:(root/p).stat() for p in scope['source_hashes']};active=None;completed=[];ledger=clients=broker=None
    def guard(cell):
        require(cell==active and all(file_hash(prep/n)==h for n,h in pins.items()),'AB_TEST_ACTIVE_SCOPE')
        require(all(((root/p).stat().st_size,(root/p).stat().st_mtime_ns)==(s.st_size,s.st_mtime_ns) for p,s in signatures.items()),'AB_TEST_SOURCE_MUTATED')
    atomic_write_json(run/'lifecycle.json',dict(status='RUNNING',phase='test',cells_completed=0))
    try:
        transport,clients=gov._transport()
        ledger=TokenLedger(run/'ledger',task_sha256=scope['scope_identity'],policy=POLICY_V27_FORMAL_UNCAPPED)
        broker=TestResumeBroker(plan=scope['plan'],phase='test',scope=scope,transport=transport,ledger=ledger,
            receipts=ProviderResponseReceipts(run/'provider_receipts',attempt_id=scope['scope_identity'],startup_identity_sha256=scope['scope_identity']),
            dispatch_guard=guard,paid_prefix={},terminal_observations={})
        from .population_governance import inputs
        _,reader=inputs(root)
        rows=broker.retry_local(lambda:reader.heldout_rows(StudyHeldoutAccess('test',reader.manifest_sha256,scope['parent_evidence_sha256'],
            lambda:validate_prep(root,prep,authorized=True,parent_snapshot=parent_snapshot))),dict(operation='test_membership_reference_check'))
        require([r['stable_example_id'] for r in rows]==scope['test_plan']['sample_ids'],'AB_TEST_MEMBERSHIP_CHANGED')
        atomic_write_json(run/'access.json',dict(test_ids=scope['test_plan']['sample_ids'],optimize_access=False,validation_access=False))
        for cell,parent in zip(scope['plan']['cells'],parents):
            validate_prep(root,prep,authorized=True,parent_snapshot=parent_snapshot);active=cell['cell_id'];broker.bind_cell(cell,scope['plan']);before=broker.usage_view()
            atomic_write_json(run/'lifecycle.json',dict(status='RUNNING',phase='test',current_cell=active,cells_completed=len(completed)))
            predictions=broker.evaluate(SystemPrompt.from_dict(parent['final_prompt']),rows,'test',42,0)
            require(set(predictions)==set(scope['test_plan']['sample_ids']),'AB_TEST_INCOMPLETE_CELL')
            r=dict(cell=cell,predictions=predictions,usage={k:broker.usage_view()[k]-v for k,v in before.items()})
            atomic_write_json(run/active/'result.json',r);completed.append(r)
            print(__import__('json').dumps(dict(cell=active,status='CELL_COMPLETE',provider_calls=broker.usage_view()['provider_calls'])),flush=True)
        validate_prep(root,prep,authorized=True,parent_snapshot=parent_snapshot);view=audit_new_run(run,scope,broker.usage_view())
        gov.audit_accounting(run,scope,'test',summary=dict(usage=broker.usage_view(),token_ledger=view),results=completed)
        # New Test plan metadata does not relabel historical training identities.
        summary=summarize(dict(scope['plan'],cells=[p['cell'] for p in parents]),parents,{r['cell']['cell_id']:r['predictions'] for r in completed})
        summary.update(test_only=True,historical_test_previously_exposed=True,matched_causal_ablation=False,
            new_test_tokens=view['charged_total'],new_test_physical_calls=broker.usage_view()['provider_calls'],local_evaluator_retries=broker.local_retry_count)
        for name,value in {'results':completed,'summary':summary,'execution_summary':dict(usage=broker.usage_view(),token_ledger=view)}.items():atomic_write_json(run/(name+'.json'),value)
        ledger.close();ledger=None;clients.close();clients=None
        atomic_write_json(run/'lifecycle.json',dict(status='EXECUTION_COMPLETE',phase='test',cells_completed=6))
        atomic_write_json(run/'terminal.json',dict(status='EXECUTION_COMPLETE',phase='test',source_sha=scope['source_sha'],scope_identity=scope['scope_identity'],artifacts=gov.inventory(run)))
        return dict(status='EXECUTION_COMPLETE',cells_completed=6)
    except BaseException as exc:
        if ledger is not None:
            for key in tuple(ledger.inflight):ledger.reconcile(key,{},outcome='ABORT_FULL_RESERVATION_FALLBACK')
            atomic_write_json(run/'abort.json',dict(error_type=type(exc).__name__,usage=broker.usage_view() if broker else None,token_ledger=ledger.view()))
        atomic_write_json(run/'lifecycle.json',dict(status='EXECUTION_ABORTED',phase='test',current_cell=active,cells_completed=len(completed),error_type=type(exc).__name__))
        raise
    finally:
        if ledger is not None:ledger.close()
        if clients is not None:clients.close()
