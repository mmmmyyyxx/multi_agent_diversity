"""Conditionally authorized final Test300; sealed training is immutable."""
from pathlib import Path
import json
import os
import yaml
from ..benchmarks.data_freeze import digest,file_hash
from ..benchmarks.math_prediction_validity import structured_recovery_contract
from ..benchmarks.math_flexible_answer import validity_contract
from ..governance.token_accounting import TokenLedger,POLICY_V27_FORMAL_UNCAPPED,audit_journal,OperationalAbort
from ..persistence.durable_io import read_json,atomic_write_json
from ..persistence.provider_receipts import ProviderResponseReceipts
from ..search.schemas import SearchContractError
from ..search.system_prompt import SystemPrompt
from .gated_prompt import GatedPromptBroker
from .gated_prompt_policy import policy
from .gated_prompt_network import NetworkRecoveryMixin
from .single_prompt_analysis import analyze_final
from . import gated_prompt_execution as execution_amendment

FORMAL_MANIFEST='experiments/manifests/math_v27_direct_seed86_formal_v1.yaml'


def formal_test_plan(root,reader):
    formal=yaml.safe_load((root/FORMAL_MANIFEST).read_text(encoding='utf-8'))
    p=policy()
    if (formal['split_identity']!=reader.manifest_sha256
            or formal['solver_decoding_policy']!=p['solver_decoding_policy']
            or formal['models']['solver']!='gpt-4o-mini' or formal['models']['solver_thinking'] is not False
            or formal['solver_execution_policy']['first_output_tokens']!=p['first_solver_capacity']
            or formal['solver_execution_policy']['length_recovery_output_tokens']!=p['length_recovery_capacity']
            or formal['solver_execution_policy']['solver_max_concurrency']!=p['solver_concurrency']):
        raise SearchContractError('GATED_FORMAL_TEST_ALIGNMENT_MISMATCH')
    ids=[r['stable_example_id'] for r in reader.members if r['project_split']=='test']
    if len(ids)!=300:raise SearchContractError('GATED_TEST300_MEMBERSHIP_REQUIRED')
    from ..math_scoring.prediction import cache_identity
    return dict(**cache_identity(),identity='GATED_HENDRYCKS_FINAL_TEST300_V1',formal_manifest=FORMAL_MANIFEST,
        historical_manifest_usage='membership_models_and_resource_provenance_only_not_scoring_or_authorization',
        formal_manifest_sha256=file_hash(root/FORMAL_MANIFEST),split_manifest_sha256=reader.manifest_sha256,
        sample_ids=ids,membership_sha256=digest(ids),count=300,roles=['solver'],
        selection='last_validation_retained_prompt_after_all_nine_twenty_opportunity_cells',
        prediction_validity=validity_contract(),invalid_recovery=structured_recovery_contract(),
        solver_decoding=p['solver_decoding_policy'],first_capacity=p['first_solver_capacity'],
        length_recovery_capacity=p['length_recovery_capacity'],solver_concurrency=p['solver_concurrency'],
        physical_call_ceiling=10800,token_ceiling=None,cache='same_seed_same_request_one_realization',
        authorization='conditional_exact_parent_train_scope_and_sealed_terminal_no_new_selection')


class FinalTestBroker(NetworkRecoveryMixin,GatedPromptBroker):
    def legal_role_split(self,role,split):return self.phase=='test' and role=='solver' and split=='test'
    def cache_ordinal(self,split):return 0
    def timeout_policy(self):
        if 'network_recovery' in self.scope:return dict(self.scope['network_recovery'],successful_semantic_call_ceiling=self.scope.get('call_ceiling',10800))
        return dict(policy()['timeout_recovery'],combined_physical_call_ceiling=10800)


def verify_training(directory,scope):
    from .single_prompt_governance import inventory
    terminal=read_json(directory/'terminal.json')
    if (terminal.get('status')!='EXECUTION_COMPLETE' or terminal.get('phase')!='train'
            or terminal.get('scope_identity')!=scope['scope_identity'] or terminal.get('source_sha')!=scope['source_sha']):
        raise SearchContractError('GATED_TEST_REQUIRES_COMPLETE_EXACT_TRAIN')
    actual=inventory(directory);actual.pop('terminal.json',None)
    if terminal['artifacts']!=actual:raise SearchContractError('GATED_TRAIN_TERMINAL_INVENTORY_CHANGED')
    results=read_json(directory/'results.json')
    if len(results)!=(6 if execution_amendment.amendment(scope) else 9) or [r['cell_id'] for r in results]!=[c['cell_id'] for c in scope['cells']]:
        raise SearchContractError('GATED_TEST_NINE_FINALS_REQUIRED')
    for cell,result in zip(scope['cells'],results):
        if (result['seed']!=cell['seed'] or result['arm']!=cell['arm'] or result['rounds_completed']!=20
                or len(result['prompt_history'])!=21 or len(result['metrics'])!=20
                or result['final_prompt']!=result['prompt_history'][-1]
                or result!=read_json(directory/cell['cell_id']/'result.json')):
            raise SearchContractError('GATED_TEST_INCOMPLETE_FINAL_SELECTION')
        last=read_json(directory/cell['cell_id']/'checkpoint_20.json')
        if last['prompt']!=result['final_prompt']:raise SearchContractError('GATED_TEST_FINAL_CHECKPOINT_MISMATCH')
    view,events=audit_journal(directory/'ledger',task_sha256=scope['scope_identity'],policy=execution_amendment.ledger_policy(scope))
    summary=read_json(directory/'execution_summary.json')
    inherited=scope.get('continuation',{}).get('inherited_usage',{})
    if 'continuation' in scope:
        from .gated_prompt_network import frozen_parent
        parent=scope['continuation']
        parent_root=directory.parents[len(Path(scope['execution']['run_root']).parts)-1]
        if frozen_parent(parent_root,parent_root/parent['parent_run_root'],expected_inventory=parent['parent_inventory'],allow_user_stop='execution_amendment' in scope)!=parent:
            raise SearchContractError('GATED_TEST_CONTINUATION_PARENT_CHANGED')
    if (view!=summary['token_ledger'] or summary['cells_completed']!=len(scope['cells'])
            or len([e for e in events if e['kind']=='RESERVE'])+inherited.get('provider_calls',0)!=summary['usage']['provider_calls']
            or view['charged_total']+inherited.get('input_tokens',0)+inherited.get('output_tokens',0)!=summary['usage']['input_tokens']+summary['usage']['output_tokens']):
        raise SearchContractError('GATED_TEST_PARENT_ACCOUNTING_MISMATCH')
    # Every parent request and charge must have its sealed provider receipt.
    reserved={e['reservation_id']:e for e in events if e['kind']=='RESERVE'}
    for event in events:
        receipt=event.get('response_receipt')
        if event['kind']=='CHARGE':
            if (receipt is None or receipt['reservation_id']!=event['reservation_id']
                    or receipt['attempt_id']!=reserved[event['reservation_id']]['attempt_id']):
                raise SearchContractError('GATED_TEST_PARENT_RECEIPT_MISSING_OR_MISMATCHED')
            store=ProviderResponseReceipts(directory/'provider_receipts',attempt_id=receipt['attempt_id'],startup_identity_sha256=scope['scope_identity'])
            store.read(receipt['reservation_id'])
            if read_json(store.directory/(receipt['reservation_id']+'.json'))['integrity_sha256']!=receipt['integrity_sha256']:
                raise SearchContractError('GATED_TEST_PARENT_RECEIPT_SEAL_MISMATCH')
    return results,file_hash(directory/'terminal.json')


def execute_final_test(root,prep,train):
    from . import gated_prompt_governance as gov
    scope=gov.validate_prep(root,prep,authorized=True)
    auth=read_json(prep/'authorization.json')
    if auth['test_access'] is not True:raise SearchContractError('GATED_FINAL_TEST_AUTHORIZATION_REQUIRED')
    if train.resolve()!=(root/scope['execution']['run_root']).resolve():
        raise SearchContractError('GATED_TEST_PARENT_RUN_PATH_MISMATCH')
    results,terminal_sha=verify_training(train,scope)
    test_root=train.parent/'test_run'
    if test_root.exists():raise SearchContractError('GATED_TEST_ALREADY_EXISTS')
    cells=[dict(c,attempt_id=c['attempt_id'].replace('_train_','_test_')) for c in scope['cells']]
    test_scope=dict(identity='GATED_POST_TRAIN_TEST_SCOPE_V1',phase='test',source_sha=scope['source_sha'],
        parent_scope_identity=scope['scope_identity'],parent_terminal_sha256=terminal_sha,
        parent_run_root=train.relative_to(root).as_posix(),
        policy=scope['test_plan'],cells=cells,call_ceiling=scope['test_plan']['physical_call_ceiling'],token_ceiling=None)
    if 'continuation' in scope:test_scope['network_recovery']=scope['continuation']['operation_policy']
    if 'execution_amendment' in scope:test_scope['execution_amendment']=scope['execution_amendment']
    test_scope['scope_identity']=digest(test_scope)
    marker=root/'runs/gated_prompt_consumed'/(test_scope['scope_identity']+'.json')
    with marker.open('x',encoding='utf-8') as stream:
        json.dump(test_scope,stream);stream.flush();os.fsync(stream.fileno())
    test_root.mkdir();atomic_write_json(test_root/'scope.json',test_scope)
    atomic_write_json(test_root/'authorization.json',dict(parent_authorization=auth,conditional_scope_identity=test_scope['scope_identity']))
    atomic_write_json(test_root/'lifecycle.json',dict(status='RUNNING',phase='test'))
    auth_sha=file_hash(prep/'authorization.json');prep_sha=file_hash(prep/'prep.json')
    signatures={p:(root/p).stat() for p in scope['source_hashes']}
    ledger=None;clients=None;current_cell=None
    def guard(cell):
        if (cell!=current_cell or file_hash(prep/'authorization.json')!=auth_sha
                or file_hash(prep/'prep.json')!=prep_sha or file_hash(train/'terminal.json')!=terminal_sha):
            raise SearchContractError('GATED_TEST_ACTIVE_SCOPE_CHANGED')
        for p,old in signatures.items():
            now=(root/p).stat()
            if (now.st_size,now.st_mtime_ns)!=(old.st_size,old.st_mtime_ns):raise SearchContractError('GATED_TEST_SOURCE_MUTATED')
    try:
        from ..governance.autonomous_math import create_transport
        from ..provider_routing import routing_contract,IDENTITY
        from ..benchmarks.math_optimizer_generation import optimizer_generation_contract
        transport,clients=create_transport(dict(provider=IDENTITY,provider_routing_policy=routing_contract(),
            models=dict(solver='gpt-4o-mini',optimizer_reflection='qwen3.7-flash',pattern='qwen3.7-flash',solver_thinking=False),
            decoding=dict(timeout_seconds=120),runtime_persistence_policy=True,optimizer_generation_policy=optimizer_generation_contract()))
        ledger=TokenLedger(test_root/'ledger',task_sha256=test_scope['scope_identity'],policy=POLICY_V27_FORMAL_UNCAPPED)
        broker=FinalTestBroker(phase='test',scope=test_scope,transport=transport,ledger=ledger,
            receipts=ProviderResponseReceipts(test_root/'provider_receipts',attempt_id=test_scope['scope_identity'],startup_identity_sha256=test_scope['scope_identity']),dispatch_guard=guard)
        reader=gov.validate_manifest(root,read_json(root/gov.MANIFEST))
        # Gold-bearing Test content opens only after all parent checks and authorization.
        gov.validate_prep(root,prep,authorized=True)
        if execution_amendment.test_plan(formal_test_plan(root,reader),'execution_amendment' in scope)!=scope['test_plan']:raise SearchContractError('GATED_TEST_POLICY_CHANGED')
        rows=reader.heldout_rows(gov.shared.StudyHeldoutAccess('test',reader.manifest_sha256,terminal_sha,
            lambda:gov.validate_prep(root,prep,authorized=True)))
        if [r['stable_example_id'] for r in rows]!=scope['test_plan']['sample_ids']:raise SearchContractError('GATED_TEST_MEMBERSHIP_CHANGED')
        atomic_write_json(test_root/'access.json',dict(test_ids=scope['test_plan']['sample_ids'],optimize_access=False,validation_access=False))
        predictions={}
        for cell,result in zip(cells,results):
            gov.validate_prep(root,prep,authorized=True);current_cell=cell['cell_id']
            broker.cell_id=current_cell;broker.attempt_id=cell['attempt_id'];broker.seed=cell['seed']
            predictions[current_cell]=broker.evaluate(SystemPrompt.from_dict(result['final_prompt']),rows,'test',cell['seed'],20)
            atomic_write_json(test_root/(current_cell+'.json'),dict(cell=cell,predictions=predictions[current_cell],
                correct=sum(p['correct'] for p in predictions[current_cell].values()),count=300))
            print(json.dumps(dict(cell=current_cell,status='TEST_CELL_COMPLETE',provider_calls=broker.usage_view()['provider_calls'])),flush=True)
        from ..benchmarks.math import MATHBenchmarkAdapter
        atomic_write_json(test_root/'analysis.json',analyze_final(results,predictions,MATHBenchmarkAdapter().equivalent,seeds=execution_amendment.runtime_policy(scope,'test')['seeds']))
        atomic_write_json(test_root/'execution_summary.json',dict(usage=broker.usage_view(),token_ledger=ledger.view(),cells_completed=len(cells)))
        if ledger.inflight:raise SearchContractError('GATED_TEST_UNRESOLVED_RESERVATION')
        gov.validate_prep(root,prep,authorized=True);verify_training(train,scope)
        ledger.close();ledger=None;clients.close();clients=None
        atomic_write_json(test_root/'lifecycle.json',dict(status='EXECUTION_COMPLETE',phase='test'))
        atomic_write_json(test_root/'terminal.json',dict(status='EXECUTION_COMPLETE',phase='test',scope_identity=test_scope['scope_identity'],artifacts=gov.shared.inventory(test_root)))
        return dict(status='TRAIN_AND_TEST_EXECUTION_COMPLETE',cells_completed=len(cells),test_final_prompts=len(cells))
    except BaseException as exc:
        if ledger is not None:
            for key in tuple(ledger.inflight):ledger.reconcile(key,{},outcome='ABORT_FULL_RESERVATION_FALLBACK')
        atomic_write_json(test_root/'lifecycle.json',dict(status='EXECUTION_ABORTED',phase='test',cell_id=current_cell))
        atomic_write_json(test_root/'abort.json',dict(reason=str(exc) if isinstance(exc,(SearchContractError,OperationalAbort)) else type(exc).__name__,cell_id=current_cell))
        raise
    finally:
        if ledger is not None:ledger.close()
        if clients is not None:clients.close()
