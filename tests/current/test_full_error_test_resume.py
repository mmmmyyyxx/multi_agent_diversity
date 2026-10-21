"""A2 Test repair has exact paid reuse, bounded local retry and no training path."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import pytest

from multi_dataset_diverse_rl.research.ablations import full_error_test_resume as run
from multi_dataset_diverse_rl.research.ablations import ab_test_continuation as power
from multi_dataset_diverse_rl.research.ablations.synthetic import SyntheticTransport, synthetic_rows
from multi_dataset_diverse_rl.research.ablations.test_resume_runtime import draw_key
from multi_dataset_diverse_rl.benchmarks.data_freeze import digest
from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger, POLICY_PATTERN_96_UNCAPPED
from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
from multi_dataset_diverse_rl.persistence.durable_io import read_json
from multi_dataset_diverse_rl.search.system_prompt import SEED
from multi_dataset_diverse_rl.search.schemas import SearchContractError


@pytest.fixture
def factory(tmp_path,monkeypatch):
    monkeypatch.setattr(power,'require_scorable',lambda _:None)
    monkeypatch.setattr(power,'power_matrix',lambda _:dict(valid=[True,True],equivalence=[[True,True],[True,True]],certified_pair_rejections=[]))
    objects=[]
    def make(prefix=None,observations=None):
        plan=run.control.build_plan(run.control.default_config());scope=dict(scope_identity='a'*64,phase='test',plan=plan,
            call_ceiling=3250,token_ceiling=None,execution_concurrency=deepcopy(run.control.CONCURRENCY),
            network_recovery=deepcopy(plan['config']['runtime']['timeout_recovery']),
            local_evaluator_recovery=run.operation_policy(),continuation=dict(request_identity_scope='b'*64),
            test_resume_identity=run.PREP_SCHEMA,parent_test_scope_identity='b'*64)
        folder=tmp_path/str(len(objects));ledger=TokenLedger(folder/'ledger',task_sha256=scope['scope_identity'],policy=POLICY_PATTERN_96_UNCAPPED)
        transport=SyntheticTransport();broker=run.Broker(plan=plan,phase='test',scope=scope,transport=transport,ledger=ledger,
            receipts=ProviderResponseReceipts(folder/'receipts',attempt_id='fixture',startup_identity_sha256=scope['scope_identity']),
            dispatch_guard=lambda _:None,paid_prefix=prefix or {},terminal_observations=observations or {})
        broker.bind_cell(plan['cells'][0],plan);objects.append((broker,transport,ledger))
        return broker,transport,folder
    yield make
    for b,t,l in objects:
        assert not l.inflight;l.close()


def paid(factory):
    b,t,p=factory();row=synthetic_rows('test',1)[0];value=b.prediction(SEED,row,'test',42,0)
    r=read_json(next((p/'receipts').glob('[0-9]*.json')));rec=r['record']
    key=draw_key(b.cell_id,'test_checkpoint_0_draw_1',rec['request'])
    entry=dict(request=rec['request'],response=rec['response'],reservation_id=r['reservation_id'],receipt_sha256=r['integrity_sha256'])
    return {key:entry},value,row,b.cell_id


def test_exact_paid_draw_and_prior_terminal_have_no_new_call(factory):
    prefix,value,row,cell=paid(factory);b,t,p=factory(prefix,{(cell,value['request_sha256']):digest(value)})
    assert b.prediction(SEED,row,'test',42,0)==value
    assert not t.calls and b.usage_view()['provider_calls']==0 and len(b.used_prefix)==1
    assert b.local_retry_count==0


def test_local_scoring_timeout_retries_same_answer_not_solver(factory,monkeypatch):
    seen=[];delays=[]
    def score(pair):
        seen.append(pair)
        if len(seen)<3:raise SearchContractError('MATH_EVALUATOR_TIMEOUT')
        return dict(valid=[True,True],equivalence=[[True,False],[False,True]],certified_pair_rejections=[[0,1]])
    monkeypatch.setattr(power,'power_matrix',score);monkeypatch.setattr(power.time,'sleep',delays.append)
    b,t,p=factory();monkeypatch.setattr(power,'power_matrix',score)
    v=b.prediction(SEED,synthetic_rows('test',1)[0],'test',42,0)
    assert v['correct'] is False and len(t.calls)==1 and seen[0]==seen[1]==seen[2]
    assert delays==[2,4] and b.local_retry_count==2


def test_repeated_local_stall_preserves_trace_and_does_not_score_false(factory,monkeypatch):
    b,t,p=factory()
    def stall(pair):
        e=SearchContractError('MATH_EVALUATOR_TIMEOUT');e.private_worker_trace='synthetic trace';raise e
    monkeypatch.setattr(power,'power_matrix',stall);monkeypatch.setattr(power.time,'sleep',lambda _:None)
    with pytest.raises(SearchContractError,match='MATH_LOCAL_SCORING_STALLED'):
        b.prediction(SEED,synthetic_rows('test',1)[0],'test',42,0)
    rows=[json.loads(x) for x in (p/'receipts/local_evaluator_recovery.jsonl').read_text().splitlines()]
    assert len(t.calls)==1 and len(rows)==3 and rows[-1]['disposition']=='DIAGNOSE_STALL'
    assert rows[-1]['private_worker_trace']=='synthetic trace'


@pytest.mark.parametrize('mutation',['repeat','capacity','terminal'])
def test_no_cross_repeat_changed_request_or_prior_score_drift(factory,mutation):
    prefix,value,row,cell=paid(factory)
    b,t,p=factory(prefix,{(cell,value['request_sha256']):'f'*64} if mutation=='terminal' else {})
    if mutation=='repeat':
        b.bind_cell(b.plan['cells'][1],b.plan);b.prediction(SEED,row,'test',42,0)
        assert len(t.calls)==1 and not b.used_prefix
    else:
        if mutation=='capacity':b.plan['config']['runtime']['first_solver_capacity']+=1
        with pytest.raises(SearchContractError,match='PREFIX_REQUEST_CHANGED|TERMINAL_SCORE_CHANGED'):
            b.prediction(SEED,row,'test',42,0)
        assert not t.calls


def test_solver_test_only(factory):
    b,t,p=factory()
    for role,split in [('reflection','test'),('solver','optimize'),('pattern_gradient','test')]:
        with pytest.raises(SearchContractError,match='SOLVER_ONLY'):
            b.complete(role=role,split=split,stage='test',messages=[])
    assert not t.calls


def test_all_local_slots_do_not_reenter_bounded_semaphore(factory,monkeypatch):
    from multi_dataset_diverse_rl.research import single_prompt_runtime as runtime
    b,t,p=factory();pred=runtime.classify_prediction('Final answer: 1','stop')
    monkeypatch.setattr(runtime,'classify_prediction',lambda *a:pred)
    with ThreadPoolExecutor(max_workers=16) as pool:
        values=list(pool.map(lambda r:b.prediction(SEED,r,'test',42,0),synthetic_rows('test',16)))
    assert len(values)==len(t.calls)==16


def test_manifest_budget_and_entrypoint_are_explicit():
    from scripts.run_experiment import preflight
    root=Path(__file__).resolve().parents[2];manifest=run.validate_manifest(root)
    assert preflight(manifest)['status']=='A2_TEST_RESUME_PREFLIGHT_PASS_NOT_AUTHORIZED'
    assert 350+run.operation_policy()['new_call_ceiling']==3600
    assert run.operation_policy()['worker_deadline_seconds']==30
    manifest['operation_policy']['new_call_ceiling']+=1
    with pytest.raises(SearchContractError,match='MANIFEST_CHANGED'):run.validate_manifest(root,manifest)


def test_parent_byte_mutation_fails_before_reuse(tmp_path,monkeypatch):
    monkeypatch.setattr(run.gov,'inventory',lambda _:dict(file='new'))
    with pytest.raises(SearchContractError,match='PARENT_BYTES_CHANGED'):
        run.verify_snapshot(tmp_path,dict(parent_inventory=dict(file='old')))
