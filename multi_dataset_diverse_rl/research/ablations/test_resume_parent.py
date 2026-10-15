"""Validate immutable completed training and the exact aborted Test Solver prefix."""
from pathlib import Path
from ...benchmarks.data_freeze import file_hash,digest
from ...persistence.durable_io import read_json
from ...persistence.provider_receipts import ProviderResponseReceipts
from ...governance.token_accounting import audit_journal,POLICY_V27_FORMAL_UNCAPPED,serialized_request
from ...search.schemas import SearchContractError
from .test_resume_runtime import draw_key
from . import governance as gov
import hashlib

PARENT='runs/ablations/pattern_c_only_seed42_repeat3_v7'
PIN='runs/ablations/pattern_c_only_test_resume_v8/parent_preparation_audit.json'
PIN_SHA256='3332ad43ff789933ece0068b7823e637f5664afdf7ec18af6756c03de709c68c'


def parent_pin(root):
    if file_hash(root/PIN)!=PIN_SHA256:raise SearchContractError('TEST_RESUME_PARENT_PIN_CHANGED')
    return read_json(root/PIN)


def audit_parent(root):
    pin=parent_pin(root);base=root/PARENT;prep=base/'prep';train=base/'train_run';test=base/'test_run'
    if (any(file_hash(prep/n)!=v for n,v in pin['parent_prep_hashes'].items())
            or gov.inventory(test)!=pin['parent_test_inventory']):
        raise SearchContractError('TEST_RESUME_PARENT_EVIDENCE_CHANGED')
    scope=read_json(prep/'prep.json')['scope']
    if (scope['source_sha']!=pin['parent_source_sha'] or scope['scope_identity']!=pin['parent_train_scope_identity']
            or digest({k:v for k,v in scope.items() if k!='scope_identity'})!=scope['scope_identity']):
        raise SearchContractError('TEST_RESUME_PARENT_IDENTITY')
    results,terminal=gov.verify_training(train,scope)
    ts=read_json(test/'scope.json')
    if (terminal!=pin['parent_train_terminal_sha256'] or ts!=gov.final_test_scope(scope,terminal)
            or ts['scope_identity']!=pin['parent_test_scope_identity']
            or read_json(test/'lifecycle.json')['status']!='EXECUTION_ABORTED'
            or 'MATH_EVALUATOR_TIMEOUT' not in (base/'runner.stderr.log').read_text(encoding='utf-8')
            or (test/'terminal.json').exists()):
        raise SearchContractError('TEST_RESUME_REQUIRES_SCORER_ABORTED_PREFIX')
    view,events=audit_journal(test/'ledger',task_sha256=ts['scope_identity'],policy=POLICY_V27_FORMAL_UNCAPPED)
    abort=read_json(test/'abort.json')
    reserves={e['reservation_id']:e for e in events if e['kind']=='RESERVE'}
    charges=[e for e in events if e['kind']=='CHARGE']
    if (view!=abort['token_ledger'] or len(charges)!=len(reserves) or len(charges)!=735
            or view['charged_total']!=786747 or abort['usage']['network_failed_calls']!=0):
        raise SearchContractError('TEST_RESUME_PARENT_ACCOUNTING')
    cells=scope['plan']['cells'];prefix={}
    for e in charges:
        r=reserves[e['reservation_id']];ref=e['response_receipt']
        cell=next((c for c in cells if r['attempt_id']==ts['scope_identity']+'_test_'+c['cell_id']),None)
        if cell is None or r['role']!='solver' or e['outcome']!='PROVIDER_RESPONSE':
            raise SearchContractError('TEST_RESUME_PARENT_ROLE_OR_ATTEMPT')
        store=ProviderResponseReceipts.__new__(ProviderResponseReceipts)
        store.directory=test/'provider_receipts';store.attempt_id=r['attempt_id'];store.startup=ts['scope_identity']
        record=store.read(e['reservation_id'])
        if (read_json(store.directory/(e['reservation_id']+'.json'))['integrity_sha256']!=ref['integrity_sha256']
                or hashlib.sha256(serialized_request(record['request'])).hexdigest()!=r['request_sha256']
                or record['logical_identity']!=dict(seed=42,ordinal=0,role='solver',stage=r['stage'])
                or 'error_category' in record):
            raise SearchContractError('TEST_RESUME_PARENT_RECEIPT')
        key=draw_key(cell['cell_id'],r['stage'],record['request'])
        if key in prefix:raise SearchContractError('TEST_RESUME_AMBIGUOUS_PAID_DRAW')
        prefix[key]=dict(request=record['request'],response=record['response'],
            reservation_id=e['reservation_id'],receipt_sha256=ref['integrity_sha256'])
    observations={};members={c['cell_id']:set() for c in cells}
    for line in (test/'provider_receipts/solver_cache_access.jsonl').read_bytes().splitlines():
        row=__import__('json').loads(line)
        if row['split']!='test' or row['cell_id'] not in members or row['example_id'] not in scope['test_plan']['sample_ids']:
            raise SearchContractError('TEST_RESUME_PARENT_CACHE_SCOPE')
        key=(row['cell_id'],row['request_sha256'])
        if key in observations and observations[key]!=row['observation_sha256']:
            raise SearchContractError('TEST_RESUME_PARENT_CACHE_CONFLICT')
        observations[key]=row['observation_sha256'];members[row['cell_id']].add(row['example_id'])
    first=read_json(test/cells[0]['cell_id']/'result.json')
    if (first['cell']!=cells[0] or set(first['predictions'])!=set(scope['test_plan']['sample_ids'])
            or [len(members[c['cell_id']]) for c in cells]!=[300,297,0]):
        raise SearchContractError('TEST_RESUME_PARENT_PROGRESS')
    for value in first['predictions'].values():
        if digest(value)!=observations[(cells[0]['cell_id'],value['request_sha256'])]:
            raise SearchContractError('TEST_RESUME_COMPLETED_RESULT_CHANGED')
    return dict(train_scope=scope,train_results=results,terminal_sha256=terminal,
        test_scope=ts,test_view=view,test_usage=abort['usage'],paid_prefix=prefix,
        terminal_observations=observations,first_result=first)
