"""Fresh corrected-C scopes; controls referenced, never rerun or overwritten."""
import json
import os
import re
import subprocess
from ..benchmarks.data_freeze import digest,file_hash
from ..persistence.durable_io import read_json,atomic_write_json
from ..persistence.provider_receipts import ProviderResponseReceipts
from ..governance.token_accounting import TokenLedger,audit_journal,POLICY_V27_FORMAL_UNCAPPED,OperationalAbort
from ..search.schemas import SearchContractError
from ..search.system_prompt import SystemPrompt
from . import gated_prompt_governance as gov,gated_prompt_largest as focus,gated_prompt_execution as ex
from .gated_prompt_test import FinalTestBroker,formal_test_plan
from .single_prompt_analysis import analyze_final


def cells(root,destination):
    suffix=digest(destination)[:16]
    return [dict(c,attempt_id=c['cell_id']+'_train_largest_'+suffix)
        for c in read_json(root/gov.MANIFEST)['cells'] if c['arm']=='gradient_pattern' and c['seed'] in [88,89]]


def test_plan(root,reader,parent):
    plan=ex.test_plan(formal_test_plan(root,reader),True)
    return dict(plan,identity='LARGEST_PATTERN_TWO_C_FINAL_TEST300_WITH_REFERENCED_CONTROLS_V1',
        selection='last_retained_corrected_C_after_two_twenty_opportunity_cells',
        additional_semantic_call_ceiling=2400,
        physical_call_ceiling=parent['inherited_test_usage']['provider_calls']-parent['inherited_test_usage'].get('network_failed_calls',0)+2400,
        referenced_control_cells=4,old_test_namespace=parent['parent_test_scope_identity'])


def prepare(root,destination,*,parent_root,parent_inventory_pin=None):
    destination=gov.shared.private_path(root,destination)
    if destination.exists():raise SearchContractError('LARGEST_PATTERN_PREP_EXISTS')
    if parent_root is None:raise SearchContractError('LARGEST_PATTERN_COMPLETE_PARENT_REQUIRED')
    focus.validate_manifest(read_json(root/focus.MANIFEST));gov.shared.require_selected_runtime(root)
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip():raise SearchContractError('LARGEST_PATTERN_SOURCE_NOT_CLEAN')
    source=gov.source_snapshot(root)
    if set(source)-set(subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()):raise SearchContractError('LARGEST_PATTERN_SOURCE_NOT_COMMITTED')
    pin=read_json(gov.shared.private_path(root,parent_inventory_pin)) if parent_inventory_pin else None
    parent=focus.frozen_parent(root,parent_root,expected_inventory=pin['parent_inventory'] if pin else None)
    if pin and any(pin[k]!=parent[k] for k in ('parent_run_root','parent_scope_identity','parent_source_sha')):raise SearchContractError('LARGEST_PATTERN_INITIAL_PARENT_PIN')
    reader=gov.validate_manifest(root,read_json(root/gov.MANIFEST));old=read_json(root/parent['parent_run_root']/'scope.json')
    if old['runtime_identity']!=gov.shared.runtime_identity() or old['gate_membership']!=gov.gate_membership(reader):raise SearchContractError('LARGEST_PATTERN_RUNTIME_OR_DATA_CHANGED')
    path=destination.relative_to(root).as_posix();run=(destination.parent/'train_run').relative_to(root).as_posix()
    scope=dict(study_id=focus.STUDY,phase='train',source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        source_hashes=source,runtime_identity=gov.shared.runtime_identity(),policy=gov.policy(),
        method_amendment=focus.contract(),execution_amendment=ex.contract(),manifest_sha256=file_hash(root/focus.MANIFEST),
        gate_membership=gov.gate_membership(reader),cells=cells(root,path),continuation=parent,
        call_ceiling=parent['inherited_usage']['provider_calls']-parent['inherited_usage']['network_failed_calls']+42640,
        token_ceiling=None,test_plan=test_plan(root,reader,parent),
        execution=dict(prep_path=path,run_root=run,command=[gov.policy()['runtime_interpreter'],'scripts/run_experiment.py','--prep',path,'--run-root',run,'--execute']))
    scope['scope_identity']=digest(scope);destination.mkdir(parents=True)
    atomic_write_json(destination/'prep.json',dict(schema_version=focus.PREP_SCHEMA,scope=scope))
    atomic_write_json(destination/'authorization.json',dict(scope_identity=scope['scope_identity'],approved=False,adaptive_validation_access=False,test_access=False,user_authorization_evidence_sha256=None,authorized_attempt_ids=[c['attempt_id'] for c in scope['cells']]))
    return dict(status='FROZEN_NOT_AUTHORIZED',source_sha=scope['source_sha'],scope_identity=scope['scope_identity'],compatible_prefixes=parent['compatible_prefixes'],additional_train_semantic_ceiling=42640,additional_test_semantic_ceiling=2400,real_api_calls=0)


def validate_prep(root,directory,*,authorized=False):
    directory=gov.shared.private_path(root,directory);prep=read_json(directory/'prep.json');scope=prep['scope']
    keys={'study_id','phase','source_sha','source_hashes','runtime_identity','policy','method_amendment','execution_amendment','manifest_sha256','gate_membership','cells','continuation','call_ceiling','token_ceiling','test_plan','execution','scope_identity'}
    if (set(prep)!={'schema_version','scope'} or prep['schema_version']!=focus.PREP_SCHEMA or set(scope)!=keys
            or scope['scope_identity']!=digest({k:v for k,v in scope.items() if k!='scope_identity'})
            or scope['study_id']!=focus.STUDY or scope['phase']!='train' or scope['policy']!=gov.policy()
            or scope['execution_amendment']!=ex.contract() or not focus.enabled(scope)):
        raise SearchContractError('LARGEST_PATTERN_SCOPE_IDENTITY')
    gov.shared.require_selected_runtime(root)
    if (scope['source_sha']!=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
            or scope['source_hashes']!=gov.source_snapshot(root) or scope['runtime_identity']!=gov.shared.runtime_identity()
            or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip()):raise SearchContractError('LARGEST_PATTERN_SOURCE_OR_RUNTIME_CHANGED')
    focus.validate_manifest(read_json(root/focus.MANIFEST));parent=scope['continuation']
    if focus.frozen_parent(root,root/parent['parent_run_root'],expected_inventory=parent['parent_inventory'])!=parent:raise SearchContractError('LARGEST_PATTERN_FROZEN_PARENT_CHANGED')
    reader=gov.validate_manifest(root,read_json(root/gov.MANIFEST));path=directory.relative_to(root).as_posix();run=(directory.parent/'train_run').relative_to(root).as_posix()
    if (scope['manifest_sha256']!=file_hash(root/focus.MANIFEST) or scope['cells']!=cells(root,path)
            or scope['gate_membership']!=gov.gate_membership(reader) or scope['test_plan']!=test_plan(root,reader,parent)
            or scope['call_ceiling']!=parent['inherited_usage']['provider_calls']-parent['inherited_usage']['network_failed_calls']+42640
            or scope['token_ceiling'] is not None or scope['execution']!=dict(prep_path=path,run_root=run,command=[gov.policy()['runtime_interpreter'],'scripts/run_experiment.py','--prep',path,'--run-root',run,'--execute'])):raise SearchContractError('LARGEST_PATTERN_FROZEN_INPUT_OR_COMMAND')
    auth=read_json(directory/'authorization.json')
    if (set(auth)!={'scope_identity','approved','adaptive_validation_access','test_access','user_authorization_evidence_sha256','authorized_attempt_ids'}
            or auth['scope_identity']!=scope['scope_identity'] or auth['authorized_attempt_ids']!=[c['attempt_id'] for c in scope['cells']]
            or type(auth['test_access']) is not bool):raise SearchContractError('LARGEST_PATTERN_AUTHORIZATION_SCOPE')
    if authorized and (auth['approved'] is not True or auth['adaptive_validation_access'] is not True or not re.fullmatch('[0-9a-f]{64}',str(auth['user_authorization_evidence_sha256']))):raise SearchContractError('LARGEST_PATTERN_EXACT_AUTHORIZATION_REQUIRED')
    return scope


def verify_training(root,directory,scope):
    terminal=read_json(directory/'terminal.json');actual=gov.shared.inventory(directory);actual.pop('terminal.json',None)
    if terminal!=dict(status='EXECUTION_COMPLETE',phase='train',source_sha=scope['source_sha'],scope_identity=scope['scope_identity'],artifacts=actual):raise SearchContractError('LARGEST_PATTERN_TRAIN_TERMINAL_CHANGED')
    parent=scope['continuation']
    if focus.frozen_parent(root,root/parent['parent_run_root'],expected_inventory=parent['parent_inventory'])!=parent:raise SearchContractError('LARGEST_PATTERN_PARENT_CHANGED_AT_TEST')
    results=read_json(directory/'results.json')
    if len(results)!=2 or [r['cell_id'] for r in results]!=[c['cell_id'] for c in scope['cells']]:raise SearchContractError('LARGEST_PATTERN_TWO_FINALS_REQUIRED')
    for cell,result in zip(scope['cells'],results):
        folder=directory/cell['cell_id']
        if (result['arm']!='gradient_pattern' or result['seed']!=cell['seed'] or result['rounds_completed']!=20 or len(result['metrics'])!=20
                or len(result['prompt_history'])!=21 or result['final_prompt']!=read_json(folder/'checkpoint_20.json')['prompt']
                or result!=read_json(folder/'result.json')):raise SearchContractError('LARGEST_PATTERN_FINAL_SELECTION_MISMATCH')
        verify_trajectory(folder,result,scope['gate_membership']['sample_ids'])
        for n in range(parent['compatible_prefixes'][cell['cell_id']]+1,21):
            diagnosis=read_json(folder/f'round_{n}_diagnosis.json')['feedback'];feedback,audit=focus.project_feedback(diagnosis,cell['seed'],n)
            if read_json(folder/f'round_{n}_focus.json')!=audit or read_json(folder/f'round_{n}_opportunity.json')['feedback']!=feedback:raise SearchContractError('LARGEST_PATTERN_FOCUS_AUDIT_MISMATCH')
    view,events=audit_journal(directory/'ledger',task_sha256=scope['scope_identity'],policy=ex.ledger_policy(scope));summary=read_json(directory/'execution_summary.json');inherited=parent['inherited_usage']
    if (summary['token_ledger']!=view or summary['cells_completed']!=2
            or summary['usage']['provider_calls']!=inherited['provider_calls']+sum(e['kind']=='RESERVE' for e in events)
            or summary['usage']['input_tokens']+summary['usage']['output_tokens']!=inherited['input_tokens']+inherited['output_tokens']+view['charged_total']):raise SearchContractError('LARGEST_PATTERN_COMBINED_ACCOUNTING')
    reserves={e['reservation_id']:e for e in events if e['kind']=='RESERVE'}
    for e in events:
        if e['kind']!='CHARGE':continue
        store=ProviderResponseReceipts.__new__(ProviderResponseReceipts);store.directory=directory/'provider_receipts';store.attempt_id=reserves[e['reservation_id']]['attempt_id'];store.startup=scope['scope_identity'];store.read(e['reservation_id'])
        if read_json(store.directory/(e['reservation_id']+'.json'))['integrity_sha256']!=e['response_receipt']['integrity_sha256']:raise SearchContractError('LARGEST_PATTERN_TRAIN_RECEIPT_SEAL')
    return results,file_hash(directory/'terminal.json')


def verify_trajectory(folder,result,validation_ids):
    checkpoints=[read_json(folder/f'checkpoint_{n}.json') for n in range(21)]
    score=lambda values:sum(v['correct'] for v in values.values())
    for n,cp in enumerate(checkpoints):
        if set(cp['validation'])!=set(validation_ids) or cp['prompt']!=result['prompt_history'][n]:raise SearchContractError('LARGEST_PATTERN_CHECKPOINT_MEMBERSHIP_OR_HISTORY')
        if not n:continue
        before=checkpoints[n-1];metric=cp['metric'];candidate=cp['candidate_prompt']
        if metric!=result['metrics'][n-1] or cp['parent_prompt']!=before['prompt'] or metric['round']!=n or not 1<=metric['generation_draws']<=4:raise SearchContractError('LARGEST_PATTERN_CHECKPOINT_TRANSITION')
        if candidate is None:
            if (cp['candidate_validation'] is not None or cp['candidate_optimize'] is not None or metric['generation_draws']!=4
                    or metric['candidate_evaluated'] or metric['validation_candidate'] is not None):raise SearchContractError('LARGEST_PATTERN_ABANDONMENT_MISMATCH')
            accepted=False;decision='ABANDONED_GENERATION_RETRIES_EXHAUSTED'
        else:
            if set(cp['candidate_validation'])!=set(validation_ids) or metric['validation_candidate']!=score(cp['candidate_validation']) or not metric['candidate_evaluated']:raise SearchContractError('LARGEST_PATTERN_CANDIDATE_GATE_MEMBERSHIP')
            accepted=score(cp['candidate_validation'])>score(before['validation'])
            decision='ACCEPTED_STRICT_VALIDATION_GAIN' if accepted else 'REJECTED_NO_STRICT_VALIDATION_GAIN'
        if metric['accepted']!=accepted or metric['decision']!=decision or metric['validation_before']!=score(before['validation']):raise SearchContractError('LARGEST_PATTERN_STRICT_GATE_MISMATCH')
        for field in ('prompt','optimize','validation'):
            if cp[field]!=(cp['candidate_'+field] if accepted else before[field]):raise SearchContractError('LARGEST_PATTERN_RETAINED_STATE_MISMATCH')
    if result['final_validation_correct']!=score(checkpoints[-1]['validation']):raise SearchContractError('LARGEST_PATTERN_FINAL_SCORE_MISMATCH')


def execute_final_test(root,prep,train):
    scope=validate_prep(root,prep,authorized=True);auth=read_json(prep/'authorization.json')
    if auth['test_access'] is not True:raise SearchContractError('LARGEST_PATTERN_TEST_AUTHORIZATION_REQUIRED')
    if train.resolve()!=(root/scope['execution']['run_root']).resolve():raise SearchContractError('LARGEST_PATTERN_TEST_TRAIN_PATH_MISMATCH')
    results,terminal_sha=verify_training(root,train,scope);test=train.parent/'test_run'
    if test.exists():raise SearchContractError('LARGEST_PATTERN_TEST_EXISTS')
    parent=scope['continuation'];parent_train=root/parent['parent_run_root'];parent_test=parent_train.parent/'test_run'
    test_cells=[dict(c,attempt_id=c['attempt_id'].replace('_train_','_test_')) for c in scope['cells']]
    ts=dict(identity='LARGEST_PATTERN_SEPARATE_FINAL_TEST_SCOPE_V1',phase='test',source_sha=scope['source_sha'],parent_scope_identity=scope['scope_identity'],parent_terminal_sha256=terminal_sha,
        policy=scope['test_plan'],cells=test_cells,call_ceiling=scope['test_plan']['physical_call_ceiling'],token_ceiling=None,
        execution_amendment=scope['execution_amendment'],network_recovery=parent['operation_policy'],
        continuation=dict(request_identity_scope=parent['parent_test_scope_identity'],inherited_usage=parent['inherited_test_usage']))
    ts['scope_identity']=digest(ts);marker=root/'runs/gated_prompt_consumed'/(ts['scope_identity']+'.json')
    with marker.open('x',encoding='utf-8') as stream:json.dump(ts,stream);stream.flush();os.fsync(stream.fileno())
    test.mkdir();atomic_write_json(test/'scope.json',ts);atomic_write_json(test/'authorization.json',dict(parent_authorization=auth,conditional_scope_identity=ts['scope_identity']));atomic_write_json(test/'lifecycle.json',dict(status='RUNNING',phase='test'))
    signatures={p:(root/p).stat() for p in scope['source_hashes']};prep_sha=file_hash(prep/'prep.json');auth_sha=file_hash(prep/'authorization.json');current=None;ledger=None;clients=None
    def guard(cell):
        if cell!=current or file_hash(prep/'prep.json')!=prep_sha or file_hash(prep/'authorization.json')!=auth_sha or file_hash(train/'terminal.json')!=terminal_sha:raise SearchContractError('LARGEST_PATTERN_TEST_SCOPE_MUTATED')
        for path,old in signatures.items():
            now=(root/path).stat()
            if (now.st_size,now.st_mtime_ns)!=(old.st_size,old.st_mtime_ns):raise SearchContractError('LARGEST_PATTERN_TEST_SOURCE_MUTATED')
    try:
        from ..governance.autonomous_math import create_transport
        from ..provider_routing import routing_contract,IDENTITY
        from ..benchmarks.math_optimizer_generation import optimizer_generation_contract
        transport,clients=create_transport(dict(provider=IDENTITY,provider_routing_policy=routing_contract(),models=dict(solver='gpt-4o-mini',optimizer_reflection='qwen3.7-flash',pattern='qwen3.7-flash',solver_thinking=False),decoding=dict(timeout_seconds=120),runtime_persistence_policy=True,optimizer_generation_policy=optimizer_generation_contract()))
        ledger=TokenLedger(test/'ledger',task_sha256=ts['scope_identity'],policy=POLICY_V27_FORMAL_UNCAPPED)
        broker=FinalTestBroker(phase='test',scope=ts,transport=transport,ledger=ledger,receipts=ProviderResponseReceipts(test/'provider_receipts',attempt_id=ts['scope_identity'],startup_identity_sha256=ts['scope_identity']),dispatch_guard=guard)
        # No parent Test predictions enter training. Open only after its seal and conditional authorization.
        predictions={};control_results=[]
        for c in read_json(parent_train/'results.json'):
            old=read_json(parent_test/(c['cell_id']+'.json'))
            if old['count']!=300 or set(old['predictions'])!=set(scope['test_plan']['sample_ids']):raise SearchContractError('LARGEST_PATTERN_PARENT_TEST_MEMBERSHIP')
            for value in old['predictions'].values():
                key=value['request_sha256']
                if key in broker.cache and broker.cache[key]!=value:raise SearchContractError('LARGEST_PATTERN_TEST_CACHE_CONFLICT')
                broker.cache[key]=deepcopy_value(value)
            if c['arm']!='gradient_pattern':control_results.append(c);predictions[c['cell_id']]=old['predictions']
        reader=gov.validate_manifest(root,read_json(root/gov.MANIFEST));validate_prep(root,prep,authorized=True)
        rows=reader.heldout_rows(gov.shared.StudyHeldoutAccess('test',reader.manifest_sha256,terminal_sha,lambda:validate_prep(root,prep,authorized=True)))
        if [r['stable_example_id'] for r in rows]!=scope['test_plan']['sample_ids']:raise SearchContractError('LARGEST_PATTERN_TEST300_CHANGED')
        atomic_write_json(test/'access.json',dict(test_ids=scope['test_plan']['sample_ids'],optimize_access=False,validation_access=False,referenced_controls=4))
        for cell,result in zip(test_cells,results):
            validate_prep(root,prep,authorized=True);current=cell['cell_id'];broker.cell_id=current;broker.attempt_id=cell['attempt_id'];broker.seed=cell['seed']
            pred=broker.evaluate(SystemPrompt.from_dict(result['final_prompt']),rows,'test',cell['seed'],20);predictions[current]=pred
            atomic_write_json(test/(current+'.json'),dict(cell=cell,predictions=pred,correct=sum(v['correct'] for v in pred.values()),count=300))
        from ..benchmarks.math import MATHBenchmarkAdapter
        atomic_write_json(test/'analysis.json',analyze_final(control_results+results,predictions,MATHBenchmarkAdapter().equivalent,seeds=[88,89]))
        atomic_write_json(test/'execution_summary.json',dict(usage=broker.usage_view(),token_ledger=ledger.view(),cells_completed=2,referenced_control_cells=4))
        if ledger.inflight:raise SearchContractError('LARGEST_PATTERN_TEST_UNRESOLVED')
        validate_prep(root,prep,authorized=True);verify_training(root,train,scope)
        ledger.close();ledger=None;clients.close();clients=None
        atomic_write_json(test/'lifecycle.json',dict(status='EXECUTION_COMPLETE',phase='test'));atomic_write_json(test/'terminal.json',dict(status='EXECUTION_COMPLETE',phase='test',scope_identity=ts['scope_identity'],artifacts=gov.shared.inventory(test)))
        return dict(status='CORRECTED_C_AND_PAIRED_TEST_COMPLETE',new_training_cells=2,referenced_controls=4)
    except BaseException as exc:
        if ledger:
            for key in tuple(ledger.inflight):ledger.reconcile(key,{},outcome='ABORT_FULL_RESERVATION_FALLBACK')
        atomic_write_json(test/'lifecycle.json',dict(status='EXECUTION_ABORTED',phase='test',cell_id=current));atomic_write_json(test/'abort.json',dict(reason=str(exc) if isinstance(exc,(SearchContractError,OperationalAbort)) else type(exc).__name__))
        raise
    finally:
        if ledger:ledger.close()
        if clients:clients.close()


def deepcopy_value(value):
    from copy import deepcopy
    return deepcopy(value)
