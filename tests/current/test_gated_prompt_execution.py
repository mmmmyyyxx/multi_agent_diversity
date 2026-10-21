"""Two-seed execution and accounting proofs with synthetic transport only."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
from pathlib import Path
import threading
import pytest
from multi_dataset_diverse_rl.research import gated_prompt_execution as ex
from multi_dataset_diverse_rl.research import gated_prompt_governance as gov
from multi_dataset_diverse_rl.research.gated_prompt import GatedPromptBroker
from multi_dataset_diverse_rl.research.gated_prompt_test import FinalTestBroker
from multi_dataset_diverse_rl.research.single_prompt_analysis import analyze_final
from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,OperationalAbort,audit_journal,POLICY_GATED_32_UNCAPPED,POLICY_V27_FORMAL_UNCAPPED
from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json
from multi_dataset_diverse_rl.benchmarks.data_freeze import file_hash
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.search.system_prompt import SEED
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('two_seed_fixture',ROOT/'tests/current/test_single_prompt_study.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


def test_six_ordered_cells_exclude90_and_default_stays_nine():
    manifest=gov.manifest_template(ROOT)
    assert [(c['arm'],c['seed']) for c in gov.phase_cells(manifest,'runs/synthetic',True)]==[(a,s) for a in ex.policy()['arms'] for s in [88,89]]
    assert len(gov.phase_cells(manifest,'runs/synthetic'))==9
    ex.validate_manifest(ex.manifest_template())
    changed=ex.manifest_template();changed['execution_amendment']['training_solver_concurrency']=64
    with pytest.raises(SearchContractError,match='MANIFEST_MISMATCH'):ex.validate_manifest(changed)


def test_training32_test16_and_existing_defaults_are_closed():
    s=dict(execution_amendment=ex.contract())
    assert ex.runtime_policy(s,'train')['solver_concurrency']==32
    assert ex.runtime_policy(s,'test')['solver_concurrency']==16
    assert ex.runtime_policy(s,'train')['gradient_concurrency']==8
    assert ex.runtime_policy({},'train')==ex.policy()
    assert ex.ledger_policy(s)==POLICY_GATED_32_UNCAPPED
    assert ex.ledger_policy(s,'test')==POLICY_V27_FORMAL_UNCAPPED
    bad=deepcopy(s);bad['execution_amendment']['seeds']=[88,90]
    with pytest.raises(SearchContractError,match='AMENDMENT_MISMATCH'):ex.runtime_policy(bad,'train')


def test_32_real_broker_requests_overlap_and_have_closed_charged_receipts(tmp_path,monkeypatch):
    import multi_dataset_diverse_rl.research.single_prompt_runtime as runtime
    monkeypatch.setattr(runtime.MATHBenchmarkAdapter,'score_member_output',lambda *_:0.0)
    barrier=threading.Barrier(32);lock=threading.Lock();state=dict(active=0,peak=0)
    class Transport(fixture.ToyTransport):
        def for_role(self,role,request):
            with lock:state['active']+=1;state['peak']=max(state['peak'],state['active'])
            try:
                barrier.wait(timeout=30)
                return super().for_role(role,request)
            finally:
                with lock:state['active']-=1
    scope=dict(policy=ex.policy(),scope_identity='a'*64,call_ceiling=100,token_ceiling=None,execution_amendment=ex.contract())
    ledger=TokenLedger(tmp_path/'ledger',task_sha256='a'*64,policy=ex.ledger_policy(scope))
    broker=GatedPromptBroker(phase='train',scope=scope,transport=Transport(),ledger=ledger,
        receipts=ProviderResponseReceipts(tmp_path/'receipts',attempt_id='fake',startup_identity_sha256='a'*64),dispatch_guard=lambda _:None)
    broker.seed=88;broker.cell_id=broker.attempt_id='fake'
    try:
        results=broker.evaluate(SEED,fixture.toy_rows(32),'optimize',88,0)
        assert len(results)==32 and state['peak']==32 and not ledger.inflight
        assert broker.usage['provider_calls']==32 and len(list((tmp_path/'receipts').glob('*.json')))==32
        view=ledger.view()
    finally:ledger.close()
    checked,_=audit_journal(tmp_path/'ledger',task_sha256='a'*64,policy=POLICY_GATED_32_UNCAPPED);assert checked==view
    with pytest.raises(OperationalAbort,match='CORRUPTION'):audit_journal(tmp_path/'ledger',task_sha256='a'*64,policy=POLICY_V27_FORMAL_UNCAPPED)


def test_final_test_stays_formal16_with_six_semantic_ceiling(tmp_path):
    old=fixture.broker_for(tmp_path)
    broker=FinalTestBroker(phase='test',scope=dict(old.scope,execution_amendment=ex.contract(),call_ceiling=7200,network_recovery={}),transport=old.transport,ledger=old.ledger,receipts=old.receipts,dispatch_guard=lambda _:None)
    try:
        assert broker.study_policy()['solver_concurrency']==16
        assert broker.timeout_policy()['successful_semantic_call_ceiling']==7200
    finally:old.ledger.close()


def test_analysis_requires_all_six_and_reports_fifteen_pairs():
    cells=[dict(fixture.cell(a,s),final_prompt=SEED.to_dict()) for a in ex.policy()['arms'] for s in [88,89]]
    predictions={c['cell_id']:{'toy':fixture.observation(True)} for c in cells}
    result=analyze_final(cells,predictions,lambda a,b:a==b,seeds=[88,89])
    assert len(result['all_15_pairs'])==15 and result['all_six_coverage']['seed_count']==6
    assert all(g['seed_count']==2 for g in result['groups'].values())
    with pytest.raises(SearchContractError):analyze_final(cells[:-1],predictions,lambda a,b:a==b,seeds=[88,89])
    with pytest.raises(SearchContractError):analyze_final(cells,predictions,lambda a,b:a==b)


def test_user_stop_requires_exact_source_and_sealed_receipt(tmp_path):
    run=tmp_path/'runs/parent/train_run';scope=dict(source_sha=ex.contract()['parent_source_sha'],scope_identity='a'*64)
    receipt=run.parent/'user_stop_20261014/closure.json'
    atomic_write_json(receipt,dict(identity='GATED_USER_REQUESTED_EXECUTION_AMENDMENT_STOP_V1',source_sha=scope['source_sha'],scope_identity=scope['scope_identity'],processes_verified_dead=True,model_calls_from_closure=0,orphan_reservations=['4']))
    abort=dict(owner_stop_receipt_sha256=file_hash(receipt))
    assert ex.validate_stop(tmp_path,run,scope,abort)=={'4'}
    scope['source_sha']='b'*40
    with pytest.raises(SearchContractError,match='STOP_EVIDENCE'):ex.validate_stop(tmp_path,run,scope,abort)


def test_study32_mapper_preserves_order_and_default16_guard():
    from multi_dataset_diverse_rl.search.solver_execution import bounded_ordered_map
    assert ex.bounded_training_map(lambda i:i*2,range(40),dict(execution_amendment=ex.contract()))==tuple(i*2 for i in range(40))
    with pytest.raises(SearchContractError,match='SCOPE_REQUIRED'):ex.bounded_training_map(lambda i:i,[1],{})
    with pytest.raises(SearchContractError,match='CONCURRENCY_INVALID'):bounded_ordered_map(lambda i:i,[1],32)


def test_network_continuation_retains_inherited_timeout_counters(tmp_path):
    from multi_dataset_diverse_rl.research.gated_prompt_network import NetworkBroker,operation_contract
    old=fixture.broker_for(tmp_path)
    scope=dict(old.scope,continuation=dict(operation_policy=operation_contract(),inherited_usage=dict(timeout_retries=4,network_failed_calls=7)))
    broker=NetworkBroker(phase='train',scope=scope,transport=old.transport,ledger=old.ledger,receipts=old.receipts,dispatch_guard=lambda _:None)
    try:assert broker.usage['timeout_retries']==4 and broker.usage['network_failed_calls']==7
    finally:old.ledger.close()


@pytest.mark.parametrize('mutation',['seed90','concurrency','null_amendment','test_concurrency','scope_budget','authorization'])
def test_amended_prep_tampering_rejected_before_dispatch(tmp_path,monkeypatch,mutation):
    from multi_dataset_diverse_rl.benchmarks.data_freeze import digest
    from multi_dataset_diverse_rl.research.gated_prompt_policy import gate_membership,call_ceiling
    manifest=gov.manifest_template(ROOT);atomic_write_json(tmp_path/gov.MANIFEST,manifest)
    atomic_write_json(tmp_path/ex.MANIFEST,ex.manifest_template())
    atomic_write_json(tmp_path/gov.network.MANIFEST,__import__('json').loads((ROOT/gov.network.MANIFEST).read_text(encoding='utf-8')))
    class Metadata:members=[dict(stable_example_id=f'v{i}',project_split='validation') for i in range(300)]
    plan=dict(identity='synthetic',solver_concurrency=16,physical_call_ceiling=10800)
    monkeypatch.setattr(gov,'validate_manifest',lambda *_:Metadata())
    monkeypatch.setattr(gov,'final_test_plan',lambda *_:deepcopy(plan))
    monkeypatch.setattr(gov,'source_snapshot',lambda _: {})
    monkeypatch.setattr(gov.shared,'require_selected_runtime',lambda _:None)
    monkeypatch.setattr(gov.shared,'runtime_identity',lambda:dict(identity='synthetic'))
    monkeypatch.setattr(gov.subprocess,'check_output',lambda args,**kwargs:'a'*40 if args[1]=='rev-parse' else '')
    parent=dict(parent_run_root='runs/stopped/train_run',parent_inventory={},operation_policy=gov.network.operation_contract())
    monkeypatch.setattr(gov.network,'frozen_parent',lambda *args,**kwargs:deepcopy(parent))
    path='runs/prep';run='runs/train_run';prep=tmp_path/path
    scope=dict(study_id=gov.STUDY,phase='train',source_sha='a'*40,source_hashes={},runtime_identity=dict(identity='synthetic'),policy=ex.policy(),manifest_sha256=file_hash(tmp_path/gov.MANIFEST),gate_membership=gate_membership(Metadata()),cells=gov.phase_cells(manifest,path,True),call_ceiling=call_ceiling(),token_ceiling=None,test_plan=ex.test_plan(plan,True),continuation=parent,execution_amendment=ex.contract(),execution=dict(prep_path=path,run_root=run,command=[ex.policy()['runtime_interpreter'],'scripts/run_experiment.py','--prep',path,'--run-root',run,'--execute']))
    scope['scope_identity']=digest(scope)
    auth=dict(scope_identity=scope['scope_identity'],approved=True,adaptive_validation_access=True,test_access=True,user_authorization_evidence_sha256='e'*64,authorized_attempt_ids=[c['attempt_id'] for c in scope['cells']])
    def save():
        atomic_write_json(prep/'prep.json',dict(schema_version=ex.PREP_SCHEMA,scope=scope));atomic_write_json(prep/'authorization.json',auth)
    save();assert gov.validate_prep(tmp_path,prep,authorized=True)==scope
    if mutation=='seed90':scope['cells'].append(gov.phase_cells(manifest,path)[2])
    elif mutation=='concurrency':scope['execution_amendment']['training_solver_concurrency']=64
    elif mutation=='null_amendment':scope['execution_amendment']=None
    elif mutation=='test_concurrency':scope['test_plan']['solver_concurrency']=32
    elif mutation=='scope_budget':scope['call_ceiling']+=1
    else:auth['authorized_attempt_ids'].pop()
    scope['scope_identity']=digest({k:v for k,v in scope.items() if k!='scope_identity'});auth['scope_identity']=scope['scope_identity'];save()
    with pytest.raises(SearchContractError):gov.validate_prep(tmp_path,prep,authorized=True)
    assert not (tmp_path/run).exists()
