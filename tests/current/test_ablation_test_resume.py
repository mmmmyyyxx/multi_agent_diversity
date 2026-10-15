"""Same-answer scorer recovery and exact paid Test replay, always under network guard."""
from copy import deepcopy
from pathlib import Path
import json
import pytest
from multi_dataset_diverse_rl.research.ablations import test_resume as run,governance as gov
from multi_dataset_diverse_rl.research.ablations.test_resume_runtime import TestResumeBroker as ResumeBroker,draw_key
from multi_dataset_diverse_rl.research.ablations.config import build_plan,default_config
from multi_dataset_diverse_rl.research.ablations.synthetic import SyntheticTransport,synthetic_rows
from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
from multi_dataset_diverse_rl.persistence.durable_io import read_json,atomic_write_json
from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,POLICY_V27_FORMAL_UNCAPPED
from multi_dataset_diverse_rl.benchmarks.data_freeze import digest,file_hash
from multi_dataset_diverse_rl.search.system_prompt import SEED
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.benchmarks.math import MATHBenchmarkAdapter
ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def factory(tmp_path,monkeypatch):
    monkeypatch.setattr(MATHBenchmarkAdapter,'score_member_output',lambda *a:0.)
    objects=[]
    def make(prefix=None,observations=None):
        config=default_config();config['arms']=[a for a in config['arms'] if a['arm_id']=='C']
        plan=build_plan(config);folder=tmp_path/str(len(objects));folder.mkdir()
        scope=dict(scope_identity=digest(['child',len(objects)]),call_ceiling=2865,token_ceiling=None,
            network_recovery=plan['config']['runtime']['timeout_recovery'],
            continuation=dict(request_identity_scope=digest('same_parent_test')),
            local_evaluator_recovery=run.operation_policy())
        ledger=TokenLedger(folder/'ledger',task_sha256=scope['scope_identity'],policy=POLICY_V27_FORMAL_UNCAPPED)
        transport=SyntheticTransport()
        broker=ResumeBroker(plan=plan,phase='test',scope=scope,transport=transport,ledger=ledger,
            receipts=ProviderResponseReceipts(folder/'receipts',attempt_id='fixture',startup_identity_sha256=scope['scope_identity']),
            dispatch_guard=lambda _:None,paid_prefix=prefix or {},terminal_observations=observations or {})
        broker.bind_cell(plan['cells'][0],plan);objects.append((broker,transport,ledger))
        return broker,transport,folder
    yield make
    for b,t,ledger in objects:
        assert not ledger.inflight;ledger.close()


def paid(factory):
    b,t,folder=factory();row=synthetic_rows('test',1)[0]
    result=b.prediction(SEED,row,'test',42,0)
    receipt=read_json(next((folder/'receipts').glob('[0-9]*.json')));record=receipt['record']
    key=draw_key(b.cell_id,'test_checkpoint_0_draw_1',record['request'])
    entry=dict(request=record['request'],response=record['response'],reservation_id=receipt['reservation_id'],receipt_sha256=receipt['integrity_sha256'])
    return {key:entry},result,row,b.cell_id


@pytest.mark.parametrize('wrapped',[False,True])
def test_local_timeout_retries_same_answer_without_another_model_call(factory,monkeypatch,wrapped):
    b,t,p=factory();seen=[];delays=[]
    def score(self,parsed,gold):
        seen.append((parsed.answer,gold))
        if len(seen)<4:
            try:raise SearchContractError('MATH_EVALUATOR_TIMEOUT')
            except SearchContractError as exc:
                if wrapped:raise SearchContractError('REFERENCE_UNSCORABLE') from exc
                raise
        return 0.
    monkeypatch.setattr(MATHBenchmarkAdapter,'score_member_output',score)
    monkeypatch.setattr('multi_dataset_diverse_rl.research.ablations.test_resume_runtime.time.sleep',delays.append)
    value=b.prediction(SEED,synthetic_rows('test',1)[0],'test',42,0)
    assert value['correct'] is False and len(t.calls)==1
    assert len(set(seen))==1 and delays==[2,4,8] and b.local_retry_count==3
    assert not b.network_abort.is_set()
    assert len((p/'receipts/local_evaluator_recovery.jsonl').read_bytes().splitlines())==3


@pytest.mark.parametrize('error',['MATH_EVALUATOR_FAILURE','REFERENCE_UNSCORABLE','MATH_EVALUATOR_RESPONSE_INVALID'])
def test_other_scoring_failures_are_not_hidden(factory,monkeypatch,error):
    b,t,_=factory()
    def score(*a):raise SearchContractError(error)
    monkeypatch.setattr(MATHBenchmarkAdapter,'score_member_output',score)
    with pytest.raises(SearchContractError,match=error):b.prediction(SEED,synthetic_rows('test',1)[0],'test',42,0)
    assert len(t.calls)==1 and b.local_retry_count==0


def test_exact_parent_solver_response_reuses_namespace_and_terminal_score(factory):
    prefix,value,row,cell=paid(factory)
    b,t,_=factory(prefix,{(cell,value['request_sha256']):digest(value)})
    assert b.prediction(SEED,row,'test',42,0)==value
    assert not t.calls and b.usage_view()['provider_calls']==0 and b.usage_view()['replayed_calls']==1


def test_paid_response_cannot_cross_repeat(factory):
    prefix,_,row,_=paid(factory);b,t,_=factory(prefix)
    b.bind_cell(b.plan['cells'][1],b.plan)
    b.prediction(SEED,row,'test',42,0)
    assert len(t.calls)==1 and b.usage_view()['replayed_calls']==0


def test_paid_slot_request_change_fails_before_api(factory):
    prefix,_,row,_=paid(factory);b,t,_=factory(prefix)
    b.plan['config']['runtime']['first_solver_capacity']+=1
    with pytest.raises(SearchContractError,match='PREFIX_REQUEST_CHANGED'):b.prediction(SEED,row,'test',42,0)
    assert not t.calls


def test_reconstructed_terminal_score_cannot_drift(factory):
    prefix,value,row,cell=paid(factory);b,t,_=factory(prefix,{(cell,value['request_sha256']):'f'*64})
    with pytest.raises(SearchContractError,match='TERMINAL_SCORE_CHANGED'):b.prediction(SEED,row,'test',42,0)
    assert not t.calls


def test_teacher_and_training_replay_forbidden(factory):
    b,t,_=factory()
    with pytest.raises(SearchContractError,match='SOLVER_ONLY'):
        b.complete(role='reflection',split='test',stage='x',messages=[])
    with pytest.raises(SearchContractError,match='SOLVER_ONLY'):
        b.complete(role='solver',split='optimize',stage='x',messages=[])
    assert not t.calls


def test_manifest_budgets_and_math_settings_are_exact():
    from scripts.run_experiment import preflight
    from multi_dataset_diverse_rl.benchmarks.math_domain_v2 import SETTINGS
    value=read_json(ROOT/run.MANIFEST);run.validate_manifest(ROOT,value)
    assert preflight(value)['status']=='TEST_RESUME_PREFLIGHT_PASS_NOT_AUTHORIZED'
    assert run.operation_policy()['mathematical_settings']==SETTINGS
    assert 735+run.operation_policy()['new_call_ceiling']==3600
    value['operation_policy']['new_call_ceiling']+=1
    with pytest.raises(SearchContractError,match='MANIFEST_CHANGED'):run.validate_manifest(ROOT,value)


def test_fresh_prep_cannot_reuse_parent_authorization(tmp_path,monkeypatch):
    original=gov._git
    def fake_git(root,*args):
        if args==('status','--porcelain','--untracked-files=no'):return ''
        if args==('rev-parse','HEAD'):return '0'*40
        return original(root,*args)
    monkeypatch.setattr(gov,'_git',fake_git)
    monkeypatch.setattr(gov,'source_snapshot',lambda root:{})
    monkeypatch.setattr(run,'parent_pin',lambda root:dict(parent_train_terminal_sha256='1'*64))
    old_plan=read_json(ROOT/'experiments/manifests/single_prompt_pattern_c_only_v7.json')['plan']
    old_parent=dict(plan=old_plan,network_recovery=old_plan['config']['runtime']['timeout_recovery'],
        test_plan=dict(split_manifest_sha256='2'*64,membership_sha256='3'*64,count=300))
    native_read=run.read_json
    def fake_read(path):
        if path==ROOT/run.PARENT/'prep/prep.json':return dict(scope=old_parent)
        if path==ROOT/run.PARENT/'test_run/scope.json':return dict(scope_identity=digest('old_test'))
        return native_read(path)
    monkeypatch.setattr(run,'read_json',fake_read)
    prep=ROOT/'runs/ablations/verification_test_resume_v8'/('fake_'+tmp_path.name)/'prep'
    scope=run._scope(ROOT,prep)
    atomic_write_json(prep/'prep.json',dict(schema_version=run.PREP_SCHEMA,scope=scope))
    atomic_write_json(prep/'execution_plan.json',scope['execution'])
    atomic_write_json(prep/'handoff.json',run.handoff_template(ROOT,prep,scope))
    atomic_write_json(prep/'authorization.json',run.authorization_template(scope))
    assert run.validate_prep(ROOT,prep)==scope
    with pytest.raises(SearchContractError,match='FRESH_USER_AUTHORIZATION'):run.validate_prep(ROOT,prep,authorized=True)
    atomic_write_json(prep/'authorization.json',gov.authorization_template(dict(scope_identity=digest('old'),plan=old_plan)))
    with pytest.raises(SearchContractError,match='EXACT_AUTHORIZATION'):run.validate_prep(ROOT,prep,authorized=True)


def test_governed_fake_test_phase_inherits_one_cell_and_finishes_two(tmp_path,monkeypatch):
    config=default_config();config['arms']=[a for a in config['arms'] if a['arm_id']=='C'];plan=build_plan(config)
    rows=synthetic_rows('test',2);prep=tmp_path/'runs/new/prep';target=prep.parent/'test_run'
    train=[]
    for cell in plan['cells']:
        train.append(dict(cell=cell,rounds_completed=20,final_prompt=SEED.to_dict(),
            final_validation_correct=70,validation_count=100,final_optimize_correct=100,optimize_count=150,
            metrics=[dict(decision='REJECTED_NO_STRICT_VALIDATION_GAIN') for _ in range(20)],
            usage=dict(input_tokens=0,output_tokens=0,provider_calls=0)))
    scope=dict(scope_identity=digest('fake_child'),source_sha='0'*40,source_hashes={},plan=plan,
        test_plan=dict(sample_ids=[r['stable_example_id'] for r in rows]),call_ceiling=2865,token_ceiling=None,
        local_evaluator_recovery=run.operation_policy(),network_recovery=plan['config']['runtime']['timeout_recovery'],
        continuation=dict(request_identity_scope=digest('fake_parent')),execution=dict(run_root='runs/new/test_run'))
    predictions={r['stable_example_id']:dict(correct=False) for r in rows}
    parent=dict(paid_prefix={},terminal_observations={},terminal_sha256='a'*64,train_results=train,
        test_scope=dict(scope_identity=digest('fake_parent')),test_view=dict(by_attempt={}),
        first_result=dict(predictions=predictions))
    for name in ('prep.json','authorization.json','handoff.json','execution_plan.json'):atomic_write_json(prep/name,dict(fake=True))
    monkeypatch.setattr(run,'validate_prep',lambda *a,**k:scope)
    monkeypatch.setattr(run,'audit_parent',lambda *a:parent)
    class Reader:
        manifest_sha256='b'*64
        def heldout_rows(self,access):access.require_authorization();return rows
    monkeypatch.setattr(gov,'validate_manifest',lambda *a:Reader())
    atomic_write_json(tmp_path/gov.MANIFEST,dict(fake=True))
    class Clients:
        def close(self):pass
    transport=SyntheticTransport();monkeypatch.setattr(gov,'_transport',lambda:(transport,Clients()))
    monkeypatch.setattr(MATHBenchmarkAdapter,'score_member_output',lambda *a:0.)
    assert run.execute(tmp_path,prep,target)['status']=='EXECUTION_COMPLETE'
    assert len(transport.calls)==4 and all(role=='solver' for role,_ in transport.calls)
    assert read_json(target/'lifecycle.json')['cells_completed']==3
    assert read_json(target/'summary.json')['completed_parent_test_predictions_reused']==2
