"""Explicit thirty-second local grading continuation of the stopped V8 Test."""
from copy import deepcopy
import hashlib,json,sys
from ...benchmarks.data_freeze import digest,file_hash
from ...benchmarks.math_domain_v2 import worker_deadline
from ...persistence.durable_io import read_json,atomic_write_json
from ...persistence.provider_receipts import ProviderResponseReceipts
from ...governance.token_accounting import audit_journal,POLICY_V27_FORMAL_UNCAPPED,serialized_request
from ...search.schemas import SearchContractError
from ..single_prompt_governance import private_path,require_selected_runtime
from . import test_resume as prior,governance as gov
from .test_resume_runtime import TestResumeBroker as PriorBroker,draw_key

MANIFEST='experiments/manifests/single_prompt_c_test_deadline_v9.json'
PROTOCOL='docs/design/SINGLE_PROMPT_C_TEST_DEADLINE_V9.md'
SCHEMA='single_prompt_c_test_deadline_manifest_v9'
PREP_SCHEMA='single_prompt_c_test_deadline_prep_v9'
PARENT='runs/ablations/pattern_c_only_test_resume_v8'
PIN='runs/ablations/pattern_c_only_test_deadline_v9/parent_pin.json'
PIN_SHA256='b97d2ec193ce78974e56acd5a1379b41542d6778ef2930e6fa3dc06aa8f8c069'
EXPERIMENT_ID='single_prompt_c_test_deadline_v9'
PARENT_MANIFEST=prior.MANIFEST
VERSION='V9'
HANDOFF_NAME='C-only Test continuation with thirty-second local grading'


def parent_pin(root,*,contract=None):
    c=contract or sys.modules[__name__]
    if file_hash(root/c.PIN)!=c.PIN_SHA256:raise SearchContractError('TEST_DEADLINE_PARENT_PIN_CHANGED')
    return read_json(root/c.PIN)


def operation_policy():
    result=prior.operation_policy()
    result.update(identity='TEST_ONLY_SAME_ANSWER_LOCAL_EVALUATOR_DEADLINE30_V2',
        worker_deadline_seconds=30,prior_worker_deadline_seconds=8,trigger_same_context_timeouts=3,
        inherited_successful_calls=737,new_call_ceiling=2863,
        inherited_partial_cell='merge_exact_V7_and_V8_paid_outputs_verify_all_recorded_terminal_hashes')
    return result


def manifest_template(*,contract=None):
    c=contract or sys.modules[__name__]
    return dict(schema_version=c.SCHEMA,experiment_id=c.EXPERIMENT_ID,
        parents=[c.PARENT_MANIFEST],protocol_path=c.PROTOCOL,parent_root=c.PARENT,
        parent_pin_path=c.PIN,parent_pin_sha256=c.PIN_SHA256,operation_policy=c.operation_policy(),
        status='PREPARED_NOT_EXECUTED',real_api_authorized=False)


def validate_manifest(root,value,*,contract=None):
    if value!=manifest_template(contract=contract):raise SearchContractError('TEST_DEADLINE_MANIFEST_CHANGED')


def audit_parent(root):
    pin=parent_pin(root);base=root/PARENT;run=base/'test_run'
    if gov.inventory(base)!=pin['parent_inventory']:
        raise SearchContractError('TEST_DEADLINE_PARENT_CHANGED')
    scope=read_json(base/'prep/prep.json')['scope']
    if (scope['source_sha']!=pin['parent_source_sha'] or scope['scope_identity']!=pin['parent_scope']
            or digest({k:v for k,v in scope.items() if k!='scope_identity'})!=scope['scope_identity']
            or not read_json(base/'prep/authorization.json')['approved']
            or read_json(run/'scope.json')!=scope or (run/'terminal.json').exists()
            or not pin['no_inflight_at_stop'] or pin['max_same_context_retry']<3):
        raise SearchContractError('TEST_DEADLINE_PARENT_SCOPE')
    parent=prior.audit_parent(root)
    view,events=audit_journal(run/'ledger',task_sha256=scope['scope_identity'],policy=POLICY_V27_FORMAL_UNCAPPED)
    reserves={e['reservation_id']:e for e in events if e['kind']=='RESERVE'}
    usage=dict(provider_calls=len(reserves),input_tokens=view['input_tokens'],output_tokens=view['output_tokens'],network_failed_calls=0)
    prior.audit_new_run(run,scope,usage)
    if len(reserves)!=2 or view['charged_total']!=3012:raise SearchContractError('TEST_DEADLINE_PARENT_COST')
    prefix=parent['paid_prefix'];original=dict(prefix)
    cells=scope['plan']['cells']
    for e in events:
        if e['kind']!='CHARGE':continue
        r=reserves[e['reservation_id']]
        cell=next((c for c in cells if r['attempt_id']==scope['scope_identity']+'_test_'+c['cell_id']),None)
        if cell is None or e['outcome']!='PROVIDER_RESPONSE':raise SearchContractError('TEST_DEADLINE_PARENT_ROLE')
        store=ProviderResponseReceipts.__new__(ProviderResponseReceipts)
        store.directory=run/'provider_receipts';store.attempt_id=r['attempt_id'];store.startup=scope['scope_identity']
        record=store.read(e['reservation_id']);key=draw_key(cell['cell_id'],r['stage'],record['request'])
        if key in prefix or 'error_category' in record:raise SearchContractError('TEST_DEADLINE_PAID_DRAW_CONFLICT')
        prefix[key]=dict(request=record['request'],response=record['response'],reservation_id=e['reservation_id'],
            receipt_sha256=e['response_receipt']['integrity_sha256'])
    reused=set()
    for line in (run/'provider_receipts/solver_paid_prefix.jsonl').read_bytes().splitlines():
        row=json.loads(line);key=(row['cell_id'],row['stage'],row['provider_request_sha256'])
        entry=original.get(key)
        if (entry is None or key in reused or entry['reservation_id']!=row['parent_reservation_id']
                or entry['receipt_sha256']!=row['parent_receipt_sha256']):
            raise SearchContractError('TEST_DEADLINE_REPLAY_LINK')
        reused.add(key)
    if len(reused)!=354:raise SearchContractError('TEST_DEADLINE_REPLAY_COUNT')
    for line in (run/'provider_receipts/solver_cache_access.jsonl').read_bytes().splitlines():
        row=json.loads(line);key=(row['cell_id'],row['request_sha256'])
        if (row['split']!='test' or row['cell_id'] not in {c['cell_id'] for c in cells}
                or row['example_id'] not in scope['test_plan']['sample_ids']
                or key in parent['terminal_observations'] and parent['terminal_observations'][key]!=row['observation_sha256']):
            raise SearchContractError('TEST_DEADLINE_PARENT_SCORE_CHANGED')
        parent['terminal_observations'][key]=row['observation_sha256']
    first=read_json(run/cells[0]['cell_id']/'result.json')
    if first['predictions']!=parent['first_result']['predictions']:
        raise SearchContractError('TEST_DEADLINE_COMPLETE_CELL_CHANGED')
    parent.update(inherited_test_tokens=789759,inherited_all_tokens=37134975,inherited_all_calls=21588)
    return parent


class TestResumeBroker(PriorBroker):
    def retry_local(self,operation,context):
        with worker_deadline(self.scope['local_evaluator_recovery']['worker_deadline_seconds']):
            return super().retry_local(operation,context)


def _scope(root,prep,*,contract=None):
    c=contract or sys.modules[__name__]
    c.validate_manifest(root,read_json(root/c.MANIFEST));pin=c.parent_pin(root)
    scope=prior._scope(root,prep);scope.pop('scope_identity')
    scope.update(study_id=c.EXPERIMENT_ID,manifest_sha256=file_hash(root/c.MANIFEST),
        protocol_sha256=file_hash(root/c.PROTOCOL),parent_pin_sha256=c.PIN_SHA256,
        call_ceiling=c.operation_policy()['new_call_ceiling'],local_evaluator_recovery=c.operation_policy(),
        deadline_parent_scope_identity=pin['parent_scope'])
    return dict(scope,scope_identity=digest(scope))


authorization_template=prior.authorization_template


def handoff_template(root,prep,scope,*,contract=None):
    c=contract or sys.modules[__name__]
    value=prior.handoff_template(root,prep,scope)
    value['experiment_id']=scope['study_id']
    value['protocol'].update(name=c.HANDOFF_NAME,version=c.VERSION,
        protocol_sha256=file_hash(root/c.PROTOCOL),manifest_path=c.MANIFEST,manifest_sha256=file_hash(root/c.MANIFEST),
        preregistration_path=c.PROTOCOL,preregistration_sha256=file_hash(root/c.PROTOCOL))
    value['execution'].update(new_successful_semantic_and_inflight_ceiling=c.operation_policy()['new_call_ceiling'],
        allowed_retries=dict(local=c.operation_policy(),network=scope['network_recovery']))
    return value


def prepare(root,prep,*,contract=None):
    c=contract or sys.modules[__name__]
    prep=private_path(root,prep);require_selected_runtime(root)
    if prep.exists() or gov._git(root,'status','--porcelain','--untracked-files=no'):
        raise SearchContractError('TEST_DEADLINE_FRESH_CLEAN_PREPARATION_REQUIRED')
    c.audit_parent(root);scope=_scope(root,prep,contract=c)
    if set(scope['source_hashes'])-set(gov._git(root,'ls-files').splitlines()):
        raise SearchContractError('TEST_DEADLINE_UNCOMMITTED_FREEZE')
    prep.mkdir(parents=True)
    atomic_write_json(prep/'prep.json',dict(schema_version=c.PREP_SCHEMA,scope=scope))
    atomic_write_json(prep/'authorization.json',authorization_template(scope))
    atomic_write_json(prep/'execution_plan.json',scope['execution'])
    atomic_write_json(prep/'handoff.json',handoff_template(root,prep,scope,contract=c))
    return dict(status='FROZEN_NOT_AUTHORIZED',source_sha=scope['source_sha'],scope_identity=scope['scope_identity'],real_api_calls=0)


def validate_prep(root,prep,*,authorized=False,contract=None):
    c=contract or sys.modules[__name__]
    prep=private_path(root,prep);require_selected_runtime(root);value=read_json(prep/'prep.json')
    if (value.get('schema_version')!=c.PREP_SCHEMA or gov._git(root,'status','--porcelain','--untracked-files=no')
            or value['scope']!=_scope(root,prep,contract=c)):
        raise SearchContractError('TEST_DEADLINE_FROZEN_IDENTITY_CHANGED')
    scope=value['scope'];auth=read_json(prep/'authorization.json');expected=authorization_template(scope)
    mutable=('approved','test_access','user_authorization_evidence_sha256')
    if (set(auth)!=set(expected) or any(auth[k]!=expected[k] for k in expected if k not in mutable)
            or type(auth['approved']) is not bool or type(auth['test_access']) is not bool):
        raise SearchContractError('TEST_DEADLINE_EXACT_AUTHORIZATION_REQUIRED')
    handoff=read_json(prep/'handoff.json');wanted=handoff_template(root,prep,scope,contract=c)
    if (read_json(prep/'execution_plan.json')!=scope['execution'] or set(handoff)!=set(wanted)
            or any(handoff[k]!=v for k,v in wanted.items() if k not in ('READY_TO_RUN','status'))):
        raise SearchContractError('TEST_DEADLINE_HANDOFF_CHANGED')
    if authorized and (not auth['approved'] or not auth['test_access'] or not handoff['READY_TO_RUN']
            or file_hash(prep/'user_authorization_evidence.json')!=auth['user_authorization_evidence_sha256']):
        raise SearchContractError('TEST_DEADLINE_FRESH_USER_AUTHORIZATION_REQUIRED')
    return scope


def execute(root,prep,run):
    return prior.execute(root,prep,run,contract=sys.modules[__name__])
