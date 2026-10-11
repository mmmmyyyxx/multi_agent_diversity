"""Synthetic formal resource, concurrency, monetary and held-out isolation tests."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
import threading
import time
import asyncio

import pytest

from tests.current import test_structured_optimization_evidence as fixture
from tests.current.test_gradient_diagnostic_limits_400 import output
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.search.textual_gradients import GradientExtractor,PerExampleGradientProvider
from multi_dataset_diverse_rl.search.solver_execution import PARALLEL_READABLE_POLICY as GRADIENT_PARALLEL_POLICY,frozen_execution_policy
from multi_dataset_diverse_rl.governance.token_accounting import (
    TokenLedger,POLICY_V25_FORMAL_30M,POLICY_V25_VALIDATION_6M,OperationalAbort,reservation)
from multi_dataset_diverse_rl.governance.cost_accounting import IDENTITY,amount,frozen_policy,TOKEN_ONLY_POLICY,require_resource_control
from multi_dataset_diverse_rl.benchmarks.math_formal_scope import BOUND,build_subsets,require_bound


def test_draft_is_strict_nonexecutable_and_old_default_binding_cannot_be_used():
    from multi_dataset_diverse_rl.governance.registries import load_yaml
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    from multi_dataset_diverse_rl.governance.unified_execution import bound_preflight,preexecution_manifest
    m=load_yaml(fixture.ROOT/'experiments/manifests/math_parallel_readable_preparation_v2.yaml')
    assert not validate_manifest_v2(fixture.ROOT,m)
    assert m['lifecycle']['status']=='DRAFT' and m['cost_policy'] is None
    result=bound_preflight(fixture.ROOT,m)
    assert result['gate']=='HOLD' and not result['ready_for_authorization'] and result['provider_attempts']==0
    with pytest.raises(SearchContractError,match='EXPLICIT_CURRENT_EXECUTION_BINDING'):
        preexecution_manifest(fixture.ROOT,source_sha='0'*40)
    frozen=deepcopy(m);frozen['lifecycle']['status']='PREEXECUTION_FROZEN'
    assert validate_manifest_v2(fixture.ROOT,frozen)


def test_formal_normative_contract_is_in_both_source_closures():
    from multi_dataset_diverse_rl.governance.source_identity import current_scientific_files
    for closure in (False,True):
        assert fixture.ROOT/'docs/design/FORMAL_EXECUTION_AND_PAIRED_VALIDATION_V1.md' in current_scientific_files(
            fixture.ROOT,execution_closure=closure)


def tariff(limit='100.000000'):
    # Explicit synthetic prices. These never become real supplier prices.
    return dict(identity=IDENTITY,currency='CNY',hard_limit=limit,
        price_basis='CONSERVATIVE_NO_CACHE_DISCOUNT_ALL_PHYSICAL_ATTEMPTS',
        pricing_source='SYNTHETIC_OFFLINE_ONLY',tariffs={
            'openlux':dict(model='gpt-4o-mini',input_per_million='2.000000',output_per_million='6.000000',physical_request_fee='0.001000'),
            'lwj':dict(model='qwen3.7-flash',input_per_million='1.000000',output_per_million='3.000000',physical_request_fee='0.001000')})


def test_token_only_is_explicit_exclusive_and_enters_all_role_request_identities():
    from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
    for bad in ({},dict(token_reporting_policy={}),
            dict(token_reporting_policy={**TOKEN_ONLY_POLICY,'monetary_tracking':True}),
            dict(token_reporting_policy=TOKEN_ONLY_POLICY,cost_policy=tariff())):
        with pytest.raises(SearchContractError):require_resource_control(bad)
    c=fixture.contract();c['token_reporting_policy']=deepcopy(TOKEN_ONLY_POLICY)
    require_resource_control(c)
    broker=RequestBroker(contract=c,transport=lambda _:None,arm='A4',seed=81)
    baseline=RequestBroker(contract=fixture.contract(),transport=lambda _:None,arm='A4',seed=81)
    messages=[dict(role='system',content=fixture.BASE.render()),dict(role='user',content='Synthetic input.')]
    for role in ('solver','reflection','pattern_gradient','pattern_cluster'):
        request,key=broker._request_identity(role=role,split='optimize',messages=messages,
            member_slot=0 if role=='solver' else None)
        old_request,old_key=baseline._request_identity(role=role,split='optimize',messages=messages,
            member_slot=0 if role=='solver' else None)
        assert request==old_request and key!=old_key
        assert 'token_reporting_policy' not in request


def test_token_only_manifest_schema_does_not_allow_unfrozen_or_mixed_resource_mode():
    from multi_dataset_diverse_rl.governance.registries import load_yaml
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    m=load_yaml(fixture.ROOT/'experiments/manifests/math_parallel_readable_preparation_v2.yaml')
    m['token_reporting_policy']=deepcopy(TOKEN_ONLY_POLICY)
    assert not validate_manifest_v2(fixture.ROOT,m)
    m['cost_policy']=tariff()
    assert validate_manifest_v2(fixture.ROOT,m)
    m['cost_policy']=None;m['token_reporting_policy']['missing_usage']='NO_CHARGE'
    assert validate_manifest_v2(fixture.ROOT,m)


def test_token_only_canary_entrypoint_keeps_fresh_two_million_token_ledger(tmp_path,monkeypatch):
    from tests.current.test_structured_production_entrypoint import _entrypoint_fixture
    result,_,_,_=_entrypoint_fixture(tmp_path,monkeypatch,False,phase='canary',
        seed=85,execution_policy=GRADIENT_PARALLEL_POLICY,token_only=True)
    assert result['result']['stop_reason']=='CANARY_ONE_PRODUCTION_OPPORTUNITY_COMPLETE'
    assert result['accounting']['authorized_total']==2_000_000
    assert result['accounting']['reserved_inflight']==0 and 'cost' not in result['accounting']
    assert result['validation_calls']==result['test_calls']==0


def test_recovery_membership_uses_contract_minimum_prior_use_when_fresh_pool_is_empty():
    import hashlib
    from multi_dataset_diverse_rl.benchmarks.math_canary_inputs import build_recovery_subsets,build_fresh_canary_subsets
    from multi_dataset_diverse_rl.persistence.durable_io import read_json
    parent=fixture.contract();data=read_json(fixture.ROOT/parent['low_cost_subsets_path'])
    metadata=data['metadata_universe']
    excluded=sorted(hashlib.sha256(r['stable_example_id'].encode()).hexdigest() for r in metadata)
    with pytest.raises(SearchContractError,match='UNSEEN_POOL_INSUFFICIENT'):
        build_fresh_canary_subsets(metadata,parent['split_manifest_sha256'],seed=85,excluded_example_hashes=excluded)
    recovered=build_recovery_subsets(metadata,parent['split_manifest_sha256'],seed=85,phase='canary',excluded_example_hashes=excluded)
    assert recovered['prior_actual_use_counts']==dict(canary_optimize=12,pilot_shadow=40)
    assert all(len(recovered['memberships'][k])==n for k,n in recovered['counts'].items())
    assert recovered==build_recovery_subsets(metadata,parent['split_manifest_sha256'],seed=85,phase='canary',excluded_example_hashes=excluded)


def parallel_graph(tmp_path,monkeypatch):
    original=fixture.contract
    def contract():
        c=original();c['solver_execution_policy']=deepcopy(GRADIENT_PARALLEL_POLICY)
        c['concurrency']=dict(solver=16,optimizer_reflection=8,pattern=8)
        return c
    monkeypatch.setattr(fixture,'contract',contract)
    run,broker,*_=fixture.graph(tmp_path)
    state=run.state.snapshot();diagnosis=run.analyzer.analyze(state,run.history)
    rows=tuple(r for r in run.opportunities.source.for_member(state,diagnosis,0)
        if not r.signals['target_member_correct'])
    return run,broker,rows


def test_parallel_gradient_orders_rows_preserves_per_row_recovery_and_caps(tmp_path,monkeypatch):
    _,broker,rows=parallel_graph(tmp_path,monkeypatch)
    lock=threading.Lock();barrier=threading.Barrier(len(rows));draws=Counter();active=peak=0
    packets=[]
    def transport(request):
        nonlocal active,peak
        packet=json.loads(request['messages'][1]['content']);xid=packet['example']['example_id']
        with lock:
            active+=1;peak=max(peak,active);draws[xid]+=1;draw=draws[xid];packets.append(packet)
        if draw==1:barrier.wait(timeout=10)
        time.sleep(.02 if xid==rows[0].example_id else .005)
        with lock:active-=1
        return fixture.response('invalid JSON' if draw==1 else json.dumps(output(diagnosis='A'*350)))
    broker.transport=transport
    provider=PerExampleGradientProvider(broker);extractor=GradientExtractor(provider)
    results=extractor.extract(fixture.BASE,rows)
    assert peak==len(rows) and active==0
    assert [r['example_id'] for r in results]==[r.example_id for r in rows]
    assert [r['example_id'] for r in extractor.evidence_diagnostics]==[r.example_id for r in rows]
    assert all(r['generation_draws']==2 and r['recovery_succeeded'] for r in extractor.evidence_diagnostics)
    assert provider.calls==len(provider.input_audit)==2*len(rows)
    assert not broker.physical_inflight and not broker.requests_inflight
    for row in rows:
        actual=[p for p in packets if p['example']['example_id']==row.example_id]
        assert len(actual)==2 and actual[0]==actual[1]
        assert actual[0]['current_system_prompt']==fixture.BASE.to_dict()
    old=deepcopy(broker.contract);old['solver_execution_policy'].pop('gradient_max_concurrency')
    with pytest.raises(SearchContractError):frozen_execution_policy(old)


def test_parallel_fatal_provenance_stops_dispatch_and_drains_issued_work(tmp_path,monkeypatch):
    _,broker,rows=parallel_graph(tmp_path,monkeypatch)
    barrier=threading.Barrier(2);issued=[];original=PerExampleGradientProvider.extract
    first={rows[0].example_id,rows[1].example_id}
    def extract(self,packet):
        xid=packet['example']['example_id'];issued.append(xid)
        if xid in first:barrier.wait(timeout=10)
        if xid==rows[0].example_id:raise SearchContractError('SYNTHETIC_FATAL_PROVENANCE')
        time.sleep(.04)
        return original(self,packet)
    monkeypatch.setattr(PerExampleGradientProvider,'extract',extract)
    broker.transport=lambda req:fixture.response(json.dumps(output()))
    provider=PerExampleGradientProvider(broker);extractor=GradientExtractor(provider)
    with pytest.raises(SearchContractError,match='SYNTHETIC_FATAL_PROVENANCE'):
        extractor.extract(fixture.BASE,rows)
    assert first <= set(issued) and len(issued)<=8
    assert provider.calls==len(provider.input_audit)==len(issued)-1
    assert not broker.physical_inflight and not broker.requests_inflight
    assert [d['example_id'] for d in extractor.evidence_diagnostics]==[
        r.example_id for r in rows if r.example_id in set(issued)-{rows[0].example_id}]


@pytest.mark.parametrize('policy',[POLICY_V25_FORMAL_30M,POLICY_V25_VALIDATION_6M])
def test_fresh_phase_token_journals_replay_without_budget_inheritance(tmp_path,policy):
    with TokenLedger(tmp_path,task_sha256='a'*64,policy=policy) as ledger:
        assert ledger.authorized_total==policy['authorized_total']
    with TokenLedger(tmp_path,task_sha256='a'*64,policy=policy) as ledger:
        assert ledger.authorized_total==policy['authorized_total'] and ledger.remaining==ledger.authorized_total
    wrong=POLICY_V25_FORMAL_30M if policy==POLICY_V25_VALIDATION_6M else POLICY_V25_VALIDATION_6M
    with pytest.raises(OperationalAbort):TokenLedger(tmp_path,task_sha256='a'*64,policy=wrong)


def test_money_cap_is_reserved_before_transport_and_crash_recovery_charged(tmp_path):
    request=dict(model='gpt-4o-mini',messages=[],max_tokens=3600)
    bound=reservation(request);prices=tariff('0.040000')
    cost=amount(prices,'solver',bound['input_upper_bound'],bound['output_hard_cap'])
    with TokenLedger(tmp_path,task_sha256='a'*64,policy=POLICY_V25_FORMAL_30M,cost_policy=prices) as ledger:
        key=ledger.reserve(request,attempt_id='synthetic',stage='initial',role='solver',model=request['model'])
        assert ledger.cost_reserved_micros==cost
        with pytest.raises(OperationalAbort,match='COST_CEILING'):
            ledger.reserve(request,attempt_id='synthetic',stage='initial',role='solver',model=request['model'])
        # Simulate a crash with a surviving durable reservation.
    with TokenLedger(tmp_path,task_sha256='a'*64,policy=POLICY_V25_FORMAL_30M,cost_policy=prices) as ledger:
        assert ledger.cost_charged_micros==cost and ledger.cost_reserved_micros==0
        assert ledger.events[-1]['outcome']=='CRASH_RECOVERY_FULL_CHARGE'
    changed=deepcopy(prices);changed['hard_limit']='0.050000'
    with pytest.raises(OperationalAbort):TokenLedger(tmp_path,task_sha256='a'*64,policy=POLICY_V25_FORMAL_30M,cost_policy=changed)


def test_money_usage_and_tariffs_are_role_routed_and_exact(tmp_path):
    prices=tariff();request=dict(model='gpt-4o-mini',messages=[],max_tokens=3600)
    with TokenLedger(tmp_path,task_sha256='a'*64,policy=POLICY_V25_FORMAL_30M,cost_policy=prices) as ledger:
        key=ledger.reserve(request,attempt_id='synthetic',stage='initial',role='solver',model=request['model'])
        ledger.reconcile(key,dict(input_tokens=101,output_tokens=51),outcome='SUCCESS')
        assert ledger.cost_charged_micros==amount(prices,'solver',101,51)
        assert ledger.cost_reserved_micros==0
        with pytest.raises(OperationalAbort,match='MODEL_ROUTE'):
            ledger.reserve(request,attempt_id='synthetic',stage='initial',role='reflection',model=request['model'])
    for wrong in (float('nan'),'-1','Infinity','0.0000001'):
        invalid=deepcopy(prices);invalid['tariffs']['lwj']['input_per_million']=wrong
        with pytest.raises(OperationalAbort):frozen_policy(invalid)


def test_formal_memberships_use_all_optimize_and_metadata_only_shadow():
    metadata=[dict(stable_example_id=role+str(i),subject='synthetic',level='1',project_split=role)
        for role,n in [('optimize',150),('shadow',300),('validation',300)] for i in range(n)]
    one=build_subsets(metadata,'a'*64,seed=86);two=build_subsets(metadata,'a'*64,seed=86)
    assert one==two and one['counts']==dict(formal_optimize=150,formal_shadow=40)
    assert one['memberships']['formal_optimize']==metadata[:150]
    assert all(r['project_split']=='shadow' for r in one['memberships']['formal_shadow'])
    assert one['validation_access']=='not_authorized' and one['test_access']=='sealed'
    require_bound(BOUND)
    for key,value in [('max_opportunities',5),('token_ceiling',2_000_000),('scientific_stopper','VOTE_STREAK')]:
        bad={**BOUND,key:value}
        with pytest.raises(SearchContractError):require_bound(bad)


def validation_contract():
    from multi_dataset_diverse_rl.governance.postfreeze_validation import POLICY,BOUND,ACCOUNTING_SCOPE
    c=fixture.contract();c.update(execution_phase='validation',post_freeze_validation_policy=deepcopy(POLICY),
        operational_pilot=deepcopy(BOUND),accounting_scope_policy=ACCOUNTING_SCOPE,
        validation_scope_identity='a'*64,cost_policy=tariff(),cache_namespace='synthetic_validation',
        execution_attempt_id='synthetic_validation')
    return c


def test_validation_broker_forbids_every_search_role_and_test_and_shares_identical_pairs():
    from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
    c=validation_contract();calls=[]
    broker=RequestBroker(contract=c,transport=lambda req:(calls.append(req) or fixture.response('Final answer: 2')),
        arm='A4',seed=81,validation_only=True)
    messages=[dict(role='system',content=fixture.BASE.render()),dict(role='user',content='Synthetic arithmetic.')]
    for split in ('optimize','shadow','test'):
        with pytest.raises(SearchContractError):broker.complete(role='solver',split=split,stage='forbidden',messages=messages,member_slot=0)
    for role in ('reflection','pattern_gradient','pattern_cluster'):
        with pytest.raises(SearchContractError):broker.complete(role=role,split='validation',stage='forbidden',messages=messages)
    values=[broker.complete(role='solver',split='validation',stage='paired',messages=messages,member_slot=m)
        for _ in range(2) for m in range(5)]
    assert len(calls)==5 and [v['text'] for v in values]==['Final answer: 2']*10
    assert len({v['request_sha256'] for v in values})==5
    with pytest.raises(SearchContractError):RequestBroker(contract=c,transport=lambda _:None,arm='A4',seed=81)
    c=fixture.contract()
    with pytest.raises(SearchContractError):RequestBroker(contract=c,transport=lambda _:None,arm='A4',seed=81,validation_only=True)


def test_paired_statistics_measure_vote_separately_from_oracle_and_reject_partial():
    from multi_dataset_diverse_rl.governance.paired_validation_analysis import analyze
    before=dict(vote=0,oracle=1,members=[1,0,0,0,0],validity=[1]*5)
    after=dict(vote=1,oracle=1,members=[1,1,1,0,0],validity=[1]*5)
    rows=[(before,after)]*10+[(after,before)]*2+[(before,before)]*288
    result=analyze(rows)
    assert result['VoteAccDelta']==8/300 and result['OracleAccDelta']==0
    assert result['paired_vote_gains']==10 and result['paired_vote_losses']==2
    assert result['exact_mcnemar_two_sided_p']==pytest.approx(158/4096)
    assert analyze([(before,before)]*300)['exact_mcnemar_two_sided_p']==1
    with pytest.raises(SearchContractError):analyze(rows[:-1])


@pytest.mark.parametrize('token_only',[False,True])
def test_real_formal_entrypoint_exceeds_pilot_horizon_and_uses_scientific_epoch_stop(tmp_path,monkeypatch,token_only):
    from tests.current.test_structured_production_entrypoint import _entrypoint_fixture
    from multi_dataset_diverse_rl.benchmarks.math_canary_inputs import ARM_B_TEAM_VERSION,ARM_B_SEED
    result,requests,_,_=_entrypoint_fixture(tmp_path,monkeypatch,False,phase='formal',seed=86,
        team_version=ARM_B_TEAM_VERSION,execution_policy=GRADIENT_PARALLEL_POLICY,token_only=token_only)
    assert 5<len(result['result']['trace'])<=50
    assert result['formal_status']=='SCIENTIFIC_COMPLETION' and 'pilot_status' not in result
    assert result['accounting']['authorized_total']==30_000_000
    if token_only:assert 'cost' not in result['accounting']
    else:assert result['accounting']['cost']['reserved_micros']==0
    assert all(r['messages'][0]['content']==ARM_B_SEED.render() for r in requests if r['model']=='gpt-4o-mini')
    assert result['validation_calls']==result['test_calls']==0


def test_closed_search_rejects_changed_terminal_evidence_before_validation(tmp_path,monkeypatch):
    from tests.current.test_structured_production_entrypoint import _entrypoint_fixture
    from multi_dataset_diverse_rl.governance.postfreeze_validation import closed_search
    from multi_dataset_diverse_rl.persistence.durable_io import read_json
    result,_,_,_=_entrypoint_fixture(tmp_path,monkeypatch,False,phase='canary',execution_policy=GRADIENT_PARALLEL_POLICY)
    run=tmp_path/'runs/execution';c=read_json(tmp_path/'runs/synthetic_flexible_binding.json')
    payload=dict(scope=dict(attempt_id=c['execution_attempt_id']),manifest=dict(source_sha='0'*40,
        execution_binding=dict(path=c['binding_path'])),startup_identity_sha256='a'*64)
    receipt=closed_search(tmp_path,'runs/execution',payload)
    assert receipt['scientific_complete'] and receipt['search_closed_forever']
    path=run/'provider_trace_private.jsonl'
    with path.open('a',encoding='utf-8') as stream:stream.write('{}\n')
    with pytest.raises(SearchContractError,match='EVIDENCE_HASH_MISMATCH'):closed_search(tmp_path,'runs/execution',payload)


@pytest.mark.parametrize('token_only',[False,True])
def test_post_search_validation_real_wrapper_is_single_use_and_gold_never_enters_requests(tmp_path,monkeypatch,token_only):
    from tests.current.test_structured_production_entrypoint import _entrypoint_fixture
    from multi_dataset_diverse_rl.governance import autonomous_math,unified_execution,postfreeze_validation as evaluation
    from multi_dataset_diverse_rl.benchmarks import math_domain_binding
    from multi_dataset_diverse_rl.persistence.durable_io import read_json,atomic_write_json
    _entrypoint_fixture(tmp_path,monkeypatch,False,phase='formal',seed=86,execution_policy=GRADIENT_PARALLEL_POLICY,token_only=token_only)
    c=read_json(tmp_path/'runs/synthetic_flexible_binding.json')
    binding=autonomous_math.execution_binding(tmp_path,c)
    search=dict(scope=dict(attempt_id=c['execution_attempt_id']),
        manifest=dict(source_sha='0'*40,execution_binding=dict(path=c['binding_path'])),
        startup_identity_sha256='a'*64,source_identity=dict(synthetic_no_real_source_grant=True))
    # Source governance has separate frozen-entrypoint tests. All questions,
    # answers, prices, credentials and transport in this test are synthetic.
    monkeypatch.setattr(unified_execution,'validate_prep',lambda *args,**kwargs:deepcopy(search))
    monkeypatch.setattr(math_domain_binding,'execution_binding',lambda *args:binding)
    metadata=read_json(tmp_path/c['low_cost_subsets_path'])['metadata_universe']
    memberships=[r for r in metadata if r['project_split']=='validation']
    reads=[];requests=[]
    reader=binding.reader()  # Frozen metadata/integrity only; no held-out rows.
    def validation_rows(access):
        access.require(reader);reads.append('validation')
        return tuple(dict(stable_example_id=m['stable_example_id'],
            content=dict(problem='SYNTHETIC_HELDOUT_PROBLEM_'+str(i)),
            reference_final_answer='2') for i,m in enumerate(memberships))
    monkeypatch.setattr(reader,'validation_rows',validation_rows)
    monkeypatch.setattr(binding,'reader',lambda:reader)
    class Client:
        def close(self):pass
    def transport(req):
        requests.append(deepcopy(req))
        assert req['model']=='gpt-4o-mini' and len(req['messages'])==2
        assert req['messages'][1]['content'].startswith('SYNTHETIC_HELDOUT_PROBLEM_')
        assert 'reference_final_answer' not in json.dumps(req) and 'SYNTHETIC_REFERENCE' not in json.dumps(req)
        return fixture.response('Final answer: 2')
    monkeypatch.setattr(autonomous_math,'create_transport',lambda _: (transport,Client()))
    prep=tmp_path/'runs/validation_prep';run=tmp_path/'runs/validation_execution'
    payload=evaluation.prepare(tmp_path,tmp_path/'runs/prep',prep,attempt='synthetic_validation',
        cost_policy=None if token_only else tariff('1000000.000000'),search_run='runs/execution',
        token_reporting_policy=TOKEN_ONLY_POLICY if token_only else None)
    with pytest.raises(SearchContractError,match='EXACT_SINGLE_USE_VALIDATION'):
        asyncio.run(evaluation.execute(tmp_path,prep,run))
    assert not reads and not requests and not run.exists()
    auth=read_json(prep/'authorization.json');auth['explicit_user_authorized']=True
    atomic_write_json(prep/'authorization.json',auth)
    result=asyncio.run(evaluation.execute(tmp_path,prep,run))
    assert reads==['validation'] and len(requests)==1500
    assert result['analysis']['initial']==result['analysis']['final']
    assert result['analysis']['initial']['vote_correct']==300
    assert result['analysis']['VoteAccDelta']==result['analysis']['OracleAccDelta']==0
    assert result['provider_usage']['validation']==1500 and result['test_calls']==0
    assert result['accounting']['authorized_total']==6_000_000 and result['accounting']['reserved_inflight']==0
    if token_only:assert 'cost' not in result['accounting']
    else:assert result['accounting']['cost']['reserved_micros']==0
    lifecycle=(run/'lifecycle.json').read_bytes()
    with pytest.raises(SearchContractError,match='EXACT_SINGLE_USE_VALIDATION'):
        asyncio.run(evaluation.execute(tmp_path,prep,run))
    assert (run/'lifecycle.json').read_bytes()==lifecycle and len(requests)==1500 and reads==['validation']
