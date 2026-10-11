"""Strict generated membership retries and explicit uncapped accounting."""
import asyncio
from copy import deepcopy
import json
import pytest
from tests.current.test_structured_optimization_evidence import graph,response,HYPOTHESIS,contract
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.search.generation_failures import generation_failure,recoverable_output
from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,POLICY_V27_FORMAL_UNCAPPED,POLICY_V26_FORMAL_30M,OperationalAbort,audit_journal
from multi_dataset_diverse_rl.search.policies import GlobalStopPolicy
from multi_dataset_diverse_rl.governance.precluster_continuation import compatibility,require_policy,PreclusterContinuation,_key

@pytest.mark.parametrize('bad_draws',[1,2,3])
@pytest.mark.parametrize('kind',['duplicate','unknown'])
def test_generated_membership_rejection_is_bounded_whole_response_retry(tmp_path,bad_draws,kind):
    run,broker,_,_,_=graph(tmp_path);original=broker.transport;calls=[]
    def transport(req):
        if req['model']!='gpt-4o-mini' and len(req['messages'])==2:
            packet=json.loads(req['messages'][1]['content'])
            if 'gradients' in packet:
                calls.append(req)
                if len(calls)<=bad_draws:
                    aliases=[g['example_id'] for g in packet['gradients']]
                    supports=aliases+([aliases[0]] if kind=='duplicate' else ['unknown'])
                    return response(json.dumps(dict(patterns=[dict(generalized_gradient=HYPOTHESIS,support_ids=supports)],unassigned_ids=[])))
        return original(req)
    broker.transport=transport;result=asyncio.run(run.run(max_opportunities=1))
    assert len(calls)==min(bad_draws+1,3)
    assert all(r==calls[0] for r in calls)
    assert bool(result.trace[0].candidate_ids)==(bad_draws<3)
    assert (broker.usage['reflection']>0)==(bad_draws<3)
    assert len(result.trace)==1

def test_untyped_internal_membership_and_integrity_failures_remain_fatal():
    category='PATTERN_GRADIENT_CLUSTER_INVALID_MEMBERSHIP'
    assert recoverable_output(generation_failure(category),'cluster')
    assert not recoverable_output(SearchContractError(category),'cluster')
    for category in ('STARTUP_IDENTITY_MISMATCH','PATTERN_HELDOUT_ACCESS','TOKEN_LEDGER_CORRUPT'):
        assert not recoverable_output(SearchContractError(category),'cluster')

def test_uncapped_ledger_exceeds_old_total_and_replays_all_reservations(tmp_path):
    request=dict(model='synthetic',max_tokens=31_000_000,messages=[])
    with TokenLedger(tmp_path/'new',task_sha256='a'*64,policy=POLICY_V27_FORMAL_UNCAPPED) as ledger:
        key=ledger.reserve(request,attempt_id='synthetic',stage='test',role='synthetic',model='synthetic')
        ledger.reconcile(key,None,outcome='SYNTHETIC_UNTRUSTED_USAGE')
        view=ledger.view();assert view['charged_total']>30_000_000 and view['remaining'] is None
        assert view['reserved_inflight']==0
    audited,_=audit_journal(tmp_path/'new',task_sha256='a'*64,policy=POLICY_V27_FORMAL_UNCAPPED)
    assert audited==view
    with TokenLedger(tmp_path/'old',task_sha256='a'*64,policy=POLICY_V26_FORMAL_30M) as ledger:
        with pytest.raises(OperationalAbort,match='STOP_TOKEN_BUDGET_EXHAUSTED'):
            ledger.reserve(request,attempt_id='synthetic',stage='test',role='synthetic',model='synthetic')

def test_four_complete_epochs_and_commit_reset():
    stop=GlobalStopPolicy(4)
    for _ in range(3):assert stop.observe_epoch(committed=False) is None
    assert stop.observe_epoch(committed=False)=='SATURATION_REACHED'
    assert stop.observe_epoch(committed=True) is None
    for _ in range(3):assert stop.observe_epoch(committed=False) is None
    assert stop.observe_epoch(committed=False)=='SATURATION_REACHED'

def test_four_epochs_do_not_mean_four_no_commit_opportunities():
    stop=GlobalStopPolicy(4)
    for epoch in range(4):
        for member in range(5):
            outcome=stop.observe_opportunity(parent_state_id='same',eligible_members=range(5),
                selected_member=member,local_update=False,committed=False)
            assert outcome==('SATURATION_REACHED' if epoch==3 and member==4 else None)

@pytest.mark.parametrize('field',['models','execution_seed','initial_team_sha256','solver_execution_policy','low_cost_subsets_sha256','memory_limits'])
def test_continuation_rejects_scientific_configuration_drift(field):
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    old=json.loads((root/'experiments/execution_bindings/a4_v27_direct_seed86_formal_attempt1.json').read_bytes())
    new=json.loads((root/'experiments/execution_bindings/a4_v27_seed86_continuation_attempt1.json').read_bytes())
    compatibility(old,new)
    new[field]='incompatible'
    with pytest.raises(SearchContractError,match='CONTINUATION_.*MISMATCH'):compatibility(old,new)

def test_exact_optimizer_prefix_replay_once_preserves_draw_position_without_provider(tmp_path):
    from collections import defaultdict,deque
    import threading
    prefix=PreclusterContinuation.__new__(PreclusterContinuation)
    prefix.lock=threading.RLock();prefix.run_root=tmp_path;prefix.policy={'parent_attempt_id':'sealed'}
    prefix.pending=defaultdict(deque);prefix.remaining=1;prefix.imported=0;prefix.imported_by_role=defaultdict(int)
    request={'model':'synthetic','messages':[{'role':'user','content':'exact synthetic input'}]}
    record=dict(response=response('rejected whole response'),physical_attempt_no=7,request_sha256='a'*64)
    prefix.pending[_key('pattern_cluster','optimize','pattern_cluster',request)].append(record)
    class Broker:
        records=[]
        def _request_identity(self,**kwargs):return request,'b'*64
        def _write(self,row):self.records.append(row)
    broker=Broker()
    result=prefix.take(broker,role='pattern_cluster',split='optimize',stage='pattern_cluster',messages=request['messages'])
    assert result['text']==record['response']['text'] and result['provider_called'] is False
    assert prefix.remaining==0 and prefix.imported==1
    assert prefix.take(broker,role='pattern_cluster',split='optimize',stage='pattern_cluster',messages=request['messages']) is None
    assert len(broker.records)==1 and broker.records[0]['new_api_call'] is False
    assert (tmp_path/'precluster_continuation_drained.json').is_file()

def test_prefix_rejects_an_unknown_wire_before_spending_a_request(tmp_path):
    from collections import defaultdict,deque
    import threading
    prefix=PreclusterContinuation.__new__(PreclusterContinuation);prefix.lock=threading.RLock()
    prefix.pending=defaultdict(deque);prefix.remaining=1
    class Broker:
        def _request_identity(self,**kwargs):return {'messages':[]},'b'*64
    with pytest.raises(SearchContractError,match='CONTINUATION_OPTIMIZER_WIRE_MISMATCH'):
        prefix.take(Broker(),role='pattern_gradient',split='optimize',stage='pattern_gradient',messages=[])
    assert prefix.remaining==1
