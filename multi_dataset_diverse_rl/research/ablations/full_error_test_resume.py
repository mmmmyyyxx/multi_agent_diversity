"""Test-only continuation of sealed A2 training and its exact paid Solver draws."""
from copy import deepcopy
from pathlib import Path
import json

from ...benchmarks.data_freeze import digest, file_hash
from ...governance.token_accounting import TokenLedger, audit_journal
from ...persistence.durable_io import read_json, atomic_write_json
from ...persistence.provider_receipts import ProviderResponseReceipts
from ...search.schemas import SearchContractError
from ...search.system_prompt import SystemPrompt
from ..single_prompt_governance import StudyHeldoutAccess
from . import governance as gov, full_error_control as control
from .ab_test_continuation import Broker as PowerBroker
from .test_resume_runtime import draw_key

MANIFEST = 'experiments/manifests/single_prompt_full_error_test_resume_v3.json'
PROTOCOL = 'docs/design/SINGLE_PROMPT_FULL_ERROR_TEST_RESUME_V3.md'
PREP_SCHEMA = 'single_prompt_full_error_test_resume_prep_v3'
PARENT = 'runs/ablations/full_error_two_proposals_seed42_repeat3_20261020'
PARENT_SOURCE = 'ad10b68077bebb4edc93a96e8c41e3104d646b19'
PARENT_SCOPE = 'c9cac6b8c64dc9782c3ad9757f660695c0c772a4e64294f8b3f561e450691a47'


def require(value, reason):
    if not value:
        raise SearchContractError(reason)


def operation_policy():
    from .ab_test_continuation import operation_policy as previous
    p = previous()
    p.update(identity='A2_TEST_PAID_PREFIX_POWER_SCORING_V3',
        successful_semantic_and_inflight_ceiling=3600, inherited_successful_calls=350,
        new_call_ceiling=3250, worker_deadline_seconds=30,
        prior_worker_deadline_seconds=8, local_grading_concurrency=16,
        inherited_completed_cell='none', inherited_partial_cell='all_exact_paid_draws_and_terminal_hashes',
        audit='stream_adjacent_checkpoints_once_then_rehash_full_parent_at_boundaries')
    return p


def validate_manifest(root, value=None):
    m = read_json(root / MANIFEST) if value is None else value
    require(m == dict(schema_version='single_prompt_full_error_test_resume_manifest_v3',
        experiment_id='single_prompt_full_error_test_resume_v3',
        parents=[control.MANIFEST], protocol_path=PROTOCOL, parent_root=PARENT,
        parent_source=PARENT_SOURCE, parent_scope=PARENT_SCOPE,
        operation_policy=operation_policy(), real_api_authorized=False,
        status='PREPARED_NOT_EXECUTED'), 'A2_TEST_MANIFEST_CHANGED')
    return m


def execution_concurrency(scope):
    require(scope.get('test_resume_identity') == PREP_SCHEMA and scope.get('phase') == 'test'
        and scope.get('execution_concurrency') == control.CONCURRENCY
        and scope.get('local_evaluator_recovery') == operation_policy()
        and scope.get('continuation') == dict(request_identity_scope=scope.get('parent_test_scope_identity'))
        and isinstance(scope.get('parent_test_scope_identity'),str) and len(scope['parent_test_scope_identity']) == 64,
        'A2_TEST_EXECUTION_POLICY')
    control.validate_config(scope['plan']['config'])
    return deepcopy(control.CONCURRENCY)


def receipt_record(run, scope, reserve, charge):
    store = ProviderResponseReceipts.__new__(ProviderResponseReceipts)
    store.directory = run / 'provider_receipts'
    store.attempt_id, store.startup = reserve['attempt_id'], scope['scope_identity']
    record = store.read(charge['reservation_id'])
    ref = charge['response_receipt']
    require(ref['attempt_id'] == reserve['attempt_id'] and ref['reservation_id'] == charge['reservation_id']
        and ref['integrity_sha256'] == read_json(store.directory / (charge['reservation_id'] + '.json'))['integrity_sha256'],
        'A2_TEST_RECEIPT_LINK')
    return record


def audit_parent(root):
    """Full semantic audit with two live checkpoints and no trajectory answer graph."""
    from .proposal_audit import audit_initial, audit_transition
    from .config import request_namespace
    base = root / PARENT; train, test = base / 'train_run', base / 'test_run'
    s = read_json(base / 'prep/prep.json')['scope']; auth = read_json(base / 'prep/authorization.json')
    require(s['source_sha'] == PARENT_SOURCE and s['scope_identity'] == PARENT_SCOPE
        and digest({k:v for k,v in s.items() if k != 'scope_identity'}) == PARENT_SCOPE,
        'A2_TEST_PARENT_IDENTITY')
    require(auth['approved'] is True and auth['test_access'] is True
        and auth['scope_identity'] == PARENT_SCOPE
        and file_hash(base / 'prep/user_authorization_evidence.json') == auth['user_authorization_evidence_sha256'],
        'A2_TEST_PARENT_AUTHORIZATION')
    terminal = read_json(train / 'terminal.json'); inventory = gov.inventory(train); inventory.pop('terminal.json')
    require(terminal == dict(status='EXECUTION_COMPLETE', phase='train', source_sha=PARENT_SOURCE,
        scope_identity=PARENT_SCOPE, artifacts=inventory) and read_json(train / 'scope.json') == s
        and read_json(train / 'access.json')['test_access'] is False, 'A2_TEST_TRAINING_SEAL')
    require(read_json(test / 'scope.json') == gov.final_test_scope(s, file_hash(train / 'terminal.json'))
        and read_json(test / 'lifecycle.json')['status'] == 'EXECUTION_ABORTED'
        and read_json(test / 'lifecycle.json')['cells_completed'] == 0
        and not (test / 'terminal.json').exists(), 'A2_TEST_ABORTED_PARENT_REQUIRED')
    plan = s['plan']; trained = []
    for cell in plan['cells']:
        folder = train / cell['cell_id']; read = lambda n: read_json(folder / (n + '.json'))
        prev = read('checkpoint_0')
        target = dict(plan_identity=plan['plan_identity'], cell=cell,
            initial_prompt_sha256=s['initial_prompt_sha256'], optimize_membership_sha256=digest(s['optimize_sample_ids']),
            validation_membership_sha256=s['gate_membership']['membership_sha256'],
            module_binding=plan['module_bindings'][cell['arm_id']],
            repeat_namespace=request_namespace(plan['plan_identity'],42,cell['repeat_id']))
        audit_initial(prev,target,s['optimize_sample_ids'],s['gate_membership']['sample_ids'])
        for n in range(1,21):
            cp = read('checkpoint_' + str(n))
            audit_transition(prev,cp,plan,cell,n,read,optimize_ids=s['optimize_sample_ids']); prev = cp
        trained.append(dict(cell=cell,final_prompt=prev['prompt']))
    gov.audit_accounting(train,s,'train',results=[])
    ts = read_json(test / 'scope.json'); abort = read_json(test / 'abort.json')
    gov.audit_accounting(test,ts,'test',summary=abort,results=[])
    view, events = audit_journal(test / 'ledger',task_sha256=ts['scope_identity'],policy=gov.accounting_policy(ts,'test'))
    reserves = {e['reservation_id']:e for e in events if e['kind']=='RESERVE'}
    prefix = {}; failed = 0
    for e in events:
        if e['kind'] != 'CHARGE': continue
        reserve = reserves[e['reservation_id']]; record = receipt_record(test,ts,reserve,e)
        cell = next((c for c in plan['cells'] if reserve['attempt_id']==ts['scope_identity']+'_test_'+c['cell_id']),None)
        require(cell is not None and reserve['role']=='solver','A2_TEST_PARENT_ROLE')
        if record.get('error_category') in ('APITimeoutError','APIConnectionError'):
            failed += 1; continue
        require(e['outcome']=='PROVIDER_RESPONSE' and 'error_category' not in record,'A2_TEST_PARENT_OUTCOME')
        key = draw_key(cell['cell_id'],reserve['stage'],record['request'])
        require(key not in prefix,'A2_TEST_DUPLICATE_PAID_DRAW')
        prefix[key] = dict(request=record['request'],response=record['response'],reservation_id=e['reservation_id'],
            receipt_sha256=e['response_receipt']['integrity_sha256'])
    require(len(prefix)==350 and failed==9 and len(reserves)==359 and view['charged_total']==523754,
        'A2_TEST_PARENT_COST')
    observations = {}
    for line in (test / 'provider_receipts/solver_cache_access.jsonl').read_bytes().splitlines():
        v=json.loads(line); cell=next((c for c in plan['cells'] if c['cell_id']==v['cell_id']),None)
        require(cell is not None and v['split']=='test' and v['repeat_id']==cell['repeat_id']
            and v['example_id'] in s['test_plan']['sample_ids'],'A2_TEST_PARENT_TERMINAL_SCOPE')
        key=(v['cell_id'],v['request_sha256'])
        require(key not in observations or observations[key]==v['observation_sha256'],'A2_TEST_TERMINAL_CONFLICT')
        observations[key]=v['observation_sha256']
    return dict(train_scope=s,test_scope=ts,trained=trained,paid_prefix=prefix,
        terminal_observations=observations,parent_inventory=gov.inventory(base),
        train_summary=read_json(train / 'summary.json'),train_tokens=86507149,
        inherited_test_tokens=view['charged_total'],inherited_calls=34348)


def verify_snapshot(root, snapshot):
    require(gov.inventory(root / PARENT)==snapshot['parent_inventory'],'A2_TEST_PARENT_BYTES_CHANGED')


def scope_template(root,prep,parent):
    validate_manifest(root); verify_snapshot(root,parent)
    old=parent['train_scope']; hashes=gov.source_snapshot(root)
    hashes[MANIFEST]=file_hash(root/MANIFEST);hashes[PROTOCOL]=file_hash(root/PROTOCOL)
    run=prep.parent/'test_run'
    s=dict(study_id='single_prompt_full_error_test_resume_v3',phase='test',source_sha=gov._git(root,'rev-parse','HEAD'),
        source_hashes=hashes,runtime_identity=gov.runtime_identity(),plan=old['plan'],test_plan=old['test_plan'],
        execution_concurrency=control.CONCURRENCY,call_ceiling=3250,token_ceiling=None,
        network_recovery=old['network_recovery'],local_evaluator_recovery=operation_policy(),
        parent_root=PARENT,parent_inventory=parent['parent_inventory'],
        parent_evidence_sha256=digest(parent['parent_inventory']),
        inherited_successful_calls=350,inherited_test_tokens=523754,inherited_all_calls=34348,
        inherited_train_tokens=86507149,final_prompt_sha256=[SystemPrompt.from_dict(v['final_prompt']).prompt_hash for v in parent['trained']],
        test_resume_identity=PREP_SCHEMA,parent_test_scope_identity=parent['test_scope']['scope_identity'],
        continuation=dict(request_identity_scope=parent['test_scope']['scope_identity']),
        execution=dict(prep_path=prep.relative_to(root).as_posix(),run_root=run.relative_to(root).as_posix(),
            command=[gov.INTERPRETER,'scripts/run_experiment.py','--prep',prep.relative_to(root).as_posix(),
                '--run-root',run.relative_to(root).as_posix(),'--execute']))
    return dict(s,scope_identity=digest(s))


def authorization_template(s):
    return dict(scope_identity=s['scope_identity'],source_sha=s['source_sha'],approved=False,
        test_access=False,train_access=False,validation_access=False,roles=['solver'],
        user_authorization_evidence_sha256=None)


def handoff(s):
    return dict(schema_version='sol_luna_experiment_handoff_v1',experiment_id=s['study_id'],
        source=dict(repository_commit=s['source_sha'],tracked_worktree_status='clean'),
        protocol=dict(name='A2 paid Test continuation with exact power scoring',version='V3',
            manifest_path=MANIFEST,manifest_sha256=s['source_hashes'][MANIFEST],
            preregistration_path=PROTOCOL,preregistration_sha256=s['source_hashes'][PROTOCOL],protocol_sha256=s['source_hashes'][PROTOCOL]),
        data=dict(dataset='math',split_identity=s['test_plan']['identity'],split_sha256=digest(s['test_plan']['sample_ids'])),
        models=dict(solver='gpt-4o-mini',optimizer_or_reflection='disabled',thinking=False,
            local_optimizer_backend='disabled_test_only',local_optimizer_version_or_sha256='not_applicable'),
        execution=dict(seeds=[42],repeat_ids=[1,2,3],opportunity_budget=900,early_stop_rule='none',
            exact_runner_command=s['execution']['command'],expected_output_directory=s['execution']['run_root'],
            expected_checkpoint_path=s['execution']['run_root']+'/A2_seed42_repeat01/result.json',
            expected_ledger_path=s['execution']['run_root']+'/ledger',allowed_retries=dict(network=s['network_recovery'],local=s['local_evaluator_recovery']),
            semantic_call_ceiling=3250,combined_semantic_call_ceiling=3600,cumulative_token_ceiling=None,
            solver_concurrency=96,local_grading_concurrency=16),
        authorization=dict(api_scope=s['scope_identity'],validation_access_policy='forbidden',test_access_policy='exact_final_three_Test300_only'),
        fail_closed_conditions=['source_hash_parent_authorization_ledger_mismatch','non_network_fault','local_scoring_stalled'],
        READY_TO_RUN=False,status='FROZEN_AWAITING_EXACT_USER_AUTHORIZATION')


def prepare(root,prep):
    raise RuntimeError('HISTORICAL_MATH_SCORER_RETIRED_USE_NEW_FROZEN_ATTEMPT')


def validate_prep(root,prep,*,authorized=False,parent=None):
    prep=gov.private_path(root,prep);gov.require_selected_runtime(root)
    require(not gov._git(root,'status','--porcelain','--untracked-files=no'),'A2_TEST_DIRTY_SOURCE')
    v=read_json(prep/'prep.json');s=v['scope']
    parent=audit_parent(root) if parent is None else parent
    require(v['schema_version']==PREP_SCHEMA and s==scope_template(root,prep,parent),'A2_TEST_SCOPE_CHANGED')
    wanted=authorization_template(s);auth=read_json(prep/'authorization.json')
    mutable={'approved','test_access','user_authorization_evidence_sha256'}
    require(set(auth)==set(wanted) and all(auth[k]==wanted[k] for k in wanted if k not in mutable)
        and type(auth['approved']) is bool and type(auth['test_access']) is bool,'A2_TEST_AUTH_SCOPE')
    expected=handoff(s);actual=read_json(prep/'handoff.json')
    require(set(actual)==set(expected) and all(actual[k]==v for k,v in expected.items() if k not in ('READY_TO_RUN','status'))
        and read_json(prep/'execution_plan.json')==s['execution'],'A2_TEST_HANDOFF')
    if authorized:
        require(auth['approved'] and auth['test_access'] and actual['READY_TO_RUN'] is True
            and file_hash(prep/'user_authorization_evidence.json')==auth['user_authorization_evidence_sha256'],'A2_TEST_UNAUTHORIZED')
    return s


class Broker(PowerBroker):
    def classify_response(self,response):
        # Parsing/semantic recovery is unchanged; timeout retries the same response.
        from ..single_prompt_runtime import StudyBroker
        with self.grading_slots:
            return self.retry_local(lambda:StudyBroker.classify_response(self,response),dict(operation='parse_paid_response'))

    def score_resolved(self,resolved,row):
        raise RuntimeError('HISTORICAL_MATH_SCORER_RETIRED_USE_NEW_FROZEN_ATTEMPT')


def audit_reuse(run,s,parent,broker):
    expected=set(parent['paid_prefix']);used=set()
    for line in (run/'provider_receipts/solver_paid_prefix.jsonl').read_bytes().splitlines():
        v=json.loads(line);key=(v['cell_id'],v['stage'],v['provider_request_sha256']);e=parent['paid_prefix'].get(key)
        require(key not in used and e is not None and e['reservation_id']==v['parent_reservation_id']
            and e['receipt_sha256']==v['parent_receipt_sha256'],'A2_TEST_REUSE_LINK');used.add(key)
    require(expected==used==broker.used_prefix,'A2_TEST_REUSE_COVERAGE')
    observed={}
    for line in (run/'provider_receipts/solver_cache_access.jsonl').read_bytes().splitlines():
        v=json.loads(line);key=(v['cell_id'],v['request_sha256'])
        if key in parent['terminal_observations']:
            require(v['observation_sha256']==parent['terminal_observations'][key],'A2_TEST_PRIOR_TERMINAL_CHANGED')
            observed[key]=v['observation_sha256']
    require(observed==parent['terminal_observations'],'A2_TEST_PRIOR_TERMINAL_COVERAGE')
    require(350+broker.usage_view()['provider_calls']-broker.usage_view()['network_failed_calls']<=3600,
        'A2_TEST_COMBINED_CEILING')


def execute(root,prep,run):
    raise RuntimeError('HISTORICAL_MATH_SCORER_RETIRED_USE_NEW_FROZEN_ATTEMPT')
