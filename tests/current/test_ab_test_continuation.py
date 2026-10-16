"""Paid-prefix continuation and bounded local stall diagnosis, with fake providers."""
import json
from pathlib import Path
import pytest
from tests.current import test_ablation_test_resume as old_tests
from tests.current.test_ablation_test_resume import factory
from multi_dataset_diverse_rl.research.ablations import ab_test_continuation as run
from multi_dataset_diverse_rl.research.ablations.synthetic import synthetic_rows
from multi_dataset_diverse_rl.search.system_prompt import SEED
from multi_dataset_diverse_rl.search.schemas import SearchContractError


def test_policy_keeps_network_recovery_distinct_and_test_only(monkeypatch):
    p=run.operation_policy()
    assert p['local_retry_limit']==3 and p['worker_deadline_seconds']==30
    assert p['local_retry_new_model_calls']==0 and not p['train_access'] and not p['validation_access']
    assert p['scoring_worker']=='math_power_worker' and p['successful_semantic_and_inflight_ceiling']==7200
    monkeypatch.setattr(run.ab,'handoff',lambda *a:dict(protocol={},execution={}))
    s=dict(call_ceiling=5497,inherited_successful_calls=1703,local_evaluator_recovery=p,
        network_recovery=dict(retry_errors=['APITimeoutError','APIConnectionError']))
    bound=run.handoff(None,s)['execution']
    assert bound['allowed_retries']['local']==p and bound['semantic_call_ceiling']==5497
    assert bound['allowed_retries']['network']==s['network_recovery']


def test_repeated_local_timeout_diagnoses_stall_after_three_attempts(factory,monkeypatch):
    monkeypatch.setattr(old_tests,'ResumeBroker',run.Broker)
    monkeypatch.setattr(run,'require_scorable',lambda _:None)
    monkeypatch.setattr(run.time,'sleep',lambda *_:None)
    calls=[]
    def stalled(_):
        calls.append(True);error=SearchContractError('MATH_EVALUATOR_TIMEOUT');error.private_worker_trace='synthetic stack';raise error
    monkeypatch.setattr(run,'power_matrix',stalled)
    b,t,p=factory();b.scope['local_evaluator_recovery']=run.operation_policy()
    with pytest.raises(SearchContractError,match='MATH_LOCAL_SCORING_STALLED'):
        b.prediction(SEED,synthetic_rows('test',1)[0],'test',42,0)
    assert len(calls)==3 and len(t.calls)==1
    records=[json.loads(x) for x in (p/'receipts/local_evaluator_recovery.jsonl').read_bytes().splitlines()]
    assert records[-1]['disposition']=='DIAGNOSE_STALL' and all(v['model_call_added'] is False for v in records)
    assert records[-1]['private_worker_trace']=='synthetic stack'


def test_local_retry_can_recover_without_new_model_call(factory,monkeypatch):
    monkeypatch.setattr(old_tests,'ResumeBroker',run.Broker)
    monkeypatch.setattr(run,'require_scorable',lambda _:None);monkeypatch.setattr(run.time,'sleep',lambda *_:None)
    calls=[]
    def score(_):
        calls.append(True)
        if len(calls)<3:raise SearchContractError('MATH_EVALUATOR_TIMEOUT')
        return dict(valid=[True,True],equivalence=[[True,False],[False,True]],certified_pair_rejections=[])
    monkeypatch.setattr(run,'power_matrix',score)
    b,t,_=factory();b.scope['local_evaluator_recovery']=run.operation_policy()
    assert b.prediction(SEED,synthetic_rows('test',1)[0],'test',42,0)['correct'] is False
    assert len(calls)==3 and len(t.calls)==1


def test_unapproved_continuation_cannot_open_provider(tmp_path,monkeypatch):
    from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json
    p=tmp_path/'runs/a/prep';atomic_write_json(p/'authorization.json',dict(approved=False,test_access=False))
    calls=[];monkeypatch.setattr(run.gov,'_transport',lambda:calls.append(True))
    with pytest.raises(SearchContractError,match='UNAUTHORIZED'):run.execute(tmp_path,p,p.parent/'test_run')
    assert not calls and not (p.parent/'test_run').exists()


def test_exact_paid_replay_preserves_transport_metadata_and_terminal_hash(factory,monkeypatch):
    from multi_dataset_diverse_rl.benchmarks.data_freeze import digest
    monkeypatch.setattr(old_tests,'ResumeBroker',run.Broker)
    monkeypatch.setattr(run,'require_scorable',lambda _:None)
    monkeypatch.setattr(run,'power_matrix',lambda _:dict(valid=[True,True],equivalence=[[True,False],[False,True]],certified_pair_rejections=[]))
    def make(*args):
        b,t,p=factory(*args);b.scope['local_evaluator_recovery']=run.operation_policy();return b,t,p
    prefix,value,row,cell=old_tests.paid(make)
    # A paid successful response records its preceding failed transport attempts.
    next(iter(prefix.values()))['response']['study_transport_timeout_attempts']=2
    value['prediction']['transport_retry_attempts']=2
    b,t,_=factory(prefix,{(cell,value['request_sha256']):digest(value)})
    b.scope['local_evaluator_recovery']=run.operation_policy()
    assert b.prediction(SEED,row,'test',42,0)==value
    assert not t.calls and b.usage_view()['replayed_calls']==1 and b.usage_view()['provider_calls']==0


def test_actual_fake_continuation_inherits_four_cells_and_evaluates_two(tmp_path,monkeypatch):
    from copy import deepcopy
    from multi_dataset_diverse_rl.benchmarks.data_freeze import digest
    from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json,read_json
    from multi_dataset_diverse_rl.research.ablations import population_governance as pg
    plan=run.ab.test_only_plan();prep=tmp_path/'runs/new/prep';folder=prep.parent/'test_run'
    scope=dict(scope_identity=digest('fake continuation'),source_sha='0'*40,source_hashes={},phase='test',plan=plan,
        parent_evidence_sha256=digest('training'),test_plan=dict(sample_ids=['t0','t1']),call_ceiling=7200,token_ceiling=None,
        execution=dict(run_root='runs/new/test_run'),inherited_successful_calls=0,
        inherited_test_ledger=dict(charged_total=0),inherited_physical_calls=0)
    trained=[dict(cell=c,rounds_completed=20,metrics=[dict(decision='REJECTED_NO_STRICT_VALIDATION_GAIN')]*20,
        final_prompt=SEED.to_dict(),final_optimize_correct=0,optimize_count=150,final_validation_correct=0,validation_count=100,
        usage=dict(input_tokens=0,output_tokens=0,provider_calls=0)) for c in plan['cells']]
    completed={};observations={}
    for c in plan['cells'][:4]:
        predictions={i:dict(correct=False,valid=True,request_sha256=digest([c['cell_id'],i])) for i in scope['test_plan']['sample_ids']}
        completed[c['cell_id']]=dict(cell=c,predictions=predictions)
        observations.update({(c['cell_id'],v['request_sha256']):digest(v) for v in predictions.values()})
    parent=dict(paid_prefix={},terminal_observations=observations,completed=completed)
    for n in ('prep','authorization','handoff','execution_plan'):atomic_write_json(prep/(n+'.json'),{})
    atomic_write_json(prep/'authorization.json',dict(approved=True,test_access=True))
    monkeypatch.setattr(run,'validate_prep',lambda *a,**k:scope)
    monkeypatch.setattr(run,'audit_test_parent',lambda *_:parent)
    monkeypatch.setattr(run.ab,'parent_evidence',lambda *_:(None,trained,{}))
    class Clients:
        def close(self):pass
    monkeypatch.setattr(run.gov,'_transport',lambda:(object(),Clients()))
    class Reader:
        manifest_sha256=digest('Test members')
        def heldout_rows(self,access):
            access.require_authorization();return [dict(stable_example_id=i) for i in scope['test_plan']['sample_ids']]
    monkeypatch.setattr(pg,'inputs',lambda *_:(None,Reader()))
    calls=[]
    class FakeBroker:
        def __init__(self,**kwargs):
            assert kwargs['phase']=='test' and kwargs['terminal_observations']==observations
            self.used_prefix=set();self.local_retry_count=0;self.cache={}
        def retry_local(self,fn,context):return fn()
        def bind_cell(self,cell,plan):self.cell=cell
        def usage_view(self):
            roles=('solver','reflection','pattern_gradient','pattern_cluster')
            return dict(provider_calls=0,input_tokens=0,output_tokens=0,network_failed_calls=0,replayed_calls=0,
                **{r:0 for r in roles},**{r+'_'+k:0 for r in roles for k in ('input_tokens','output_tokens')})
        def evaluate(self,prompt,rows,split,seed,ordinal):
            calls.append(self.cell['cell_id']);assert (split,seed,ordinal)==('test',42,0)
            return {r['stable_example_id']:dict(valid=True,correct=False) for r in rows}
    monkeypatch.setattr(run,'Broker',FakeBroker)
    assert run.execute(tmp_path,prep,folder)==dict(status='EXECUTION_COMPLETE',cells_completed=6)
    assert calls==[c['cell_id'] for c in plan['cells'][4:]]
    assert read_json(folder/'lifecycle.json')['cells_completed']==6
    assert sum(v['inherited_complete'] for v in read_json(folder/'results.json'))==4
    assert read_json(folder/'summary.json')['matched_causal_ablation'] is False
