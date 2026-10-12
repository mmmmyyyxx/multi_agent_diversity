"""Readable current role inputs and retained concurrency/timing; zero API.

The interval utility tests below are offline historical representation checks,
not evidence that a model can use an encoded trajectory.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
import threading
import time

import pytest

from tests.current.test_structured_optimization_evidence import contract, response, graph, BASE
from tests.current.test_parallel_solver_execution import broker_fixture, call
from tests.current.test_gradient_diagnostic_limits_400 import output
from multi_dataset_diverse_rl.search.optimizer_encoding import encode, decode, IDENTITY
from multi_dataset_diverse_rl.search.solver_execution import OUTPUT_FEEDBACK_POLICY, PARALLEL_COMPACT_POLICY, bounded_ordered_map
from multi_dataset_diverse_rl.search.textual_gradients import PerExampleGradientProvider, GradientExtractor
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger, POLICY_V26_2M


def parallel_contract():
    c=contract();c['solver_execution_policy']=deepcopy(OUTPUT_FEEDBACK_POLICY)
    c['concurrency']=dict(solver=16,optimizer_reflection=8,pattern=8)
    return c


@pytest.mark.parametrize('view,raw',[
    ('数学生成步骤。'*250,'数学生成步骤。'*250+'\nFinal answer: 2'),
    ('X'*1000,'prefix\n'+'X'*1000+'\nanswer'),
    ('short','short answer'), ('missing'*100,'unrelated'), ('','answer')])
def test_compact_view_is_lossless_only_when_smaller_and_exact(view,raw):
    p={'example':{'solver_trajectory':dict(observed_response=raw,visible_solution=view,
        source={'member_id':2},prediction_valid=True)}}
    before=deepcopy(p);compact=encode(p)
    assert p==before and decode(compact)==p
    assert len(json.dumps(compact))<=len(json.dumps(p))
    assert compact['example']['solver_trajectory']['observed_response']==raw
    if len(view)>500 and view in raw:
        assert compact['example']['solver_trajectory']['visible_solution']['encoding']==IDENTITY


@pytest.mark.parametrize('mutation',[{'start':True},{'end':100000},{'sha256':'0'*64},{'extra':1}])
def test_modified_reference_fails_closed(mutation):
    p=encode(dict(observed_response='X'*1000,visible_solution='X'*1000))
    p['visible_solution'].update(mutation)
    with pytest.raises(SearchContractError,match='OPTIMIZER_TRAJECTORY_REFERENCE_INVALID'):decode(p)


def test_sixteen_actual_solver_requests_and_reservations_drain_with_timing(tmp_path):
    barrier=threading.Barrier(16);lock=threading.Lock();active=peak=0
    def transport(req):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        barrier.wait(timeout=15)
        time.sleep(.01)
        with lock:active-=1
        return response('Synthetic step.\nFinal answer: 2')
    with TokenLedger(tmp_path/'accounting',task_sha256='e'*64,policy=POLICY_V26_2M) as ledger:
        broker=broker_fixture(tmp_path,parallel_contract(),transport,ledger)
        values=bounded_ordered_map(lambda i:call(broker,i),range(16),16)
        assert len(values)==16 and peak==16 and active==0
        assert not ledger.inflight and not broker.physical_inflight and not broker.requests_inflight
        assert ledger.view()['reserved_inflight']==0
        assert len([e for e in ledger.events if e['kind']=='CHARGE'])==16
    receipts=[json.loads(p.read_text()) for p in (tmp_path/'receipts').glob('*.json')]
    # Receipt wrapper remains durable and timing is attached to the original evidence.
    text='\n'.join(json.dumps(r) for r in receipts)
    assert text.count('MONOTONIC_PROVIDER_TIMING_V1')==16
    rows=[json.loads(s) for s in (tmp_path/'observations.jsonl').read_text().splitlines()]
    timings=[r for r in rows if r['kind']=='PROVIDER_REQUEST_TIMING']
    assert len(timings)==16
    assert all(r['request_timing']['transport_seconds']>=.01 for r in timings)
    assert len({r['physical_attempt_no'] for r in timings})==16


def test_eight_gradient_workers_ordered_serial_per_example_recovery_and_readable_packets(tmp_path):
    run,original,_,_,_=graph(tmp_path,long_response=True)
    state=run.state.snapshot();diagnosis=run.analyzer.analyze(state,run.history)
    template=next(r for r in run.opportunities.source.for_member(state,diagnosis,0)
        if not r.signals['target_member_correct'])
    rows=[]
    for i in range(16):
        signals=deepcopy(template.signals);xid='synthetic-gradient-'+str(i)
        signals['solver_trajectory']['source']['example_id']=xid
        rows.append(replace(template,example_id=xid,signals=signals))
    from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
    lock=threading.Lock();barrier=threading.Barrier(8)
    active=peak=0;draws=Counter();inflight=set();packets=[]
    def transport(req):
        nonlocal active,peak
        p=json.loads(req['messages'][1]['content']);xid=p['example']['example_id']
        with lock:
            assert xid not in inflight
            inflight.add(xid);active+=1;peak=max(peak,active);draws[xid]+=1;n=draws[xid]
            packets.append(p)
        if n==1 and int(xid.rsplit('-',1)[1])<8:barrier.wait(timeout=15)
        time.sleep(.01)
        with lock:active-=1;inflight.remove(xid)
        return response('invalid JSON' if n==1 else json.dumps(output()))
    broker=RequestBroker(contract=parallel_contract(),transport=transport,arm='A4',seed=81)
    provider=PerExampleGradientProvider(broker);extractor=GradientExtractor(provider)
    result=extractor.extract(BASE,rows)
    assert peak==8 and active==0 and not inflight
    assert [r['example_id'] for r in result]==[r.example_id for r in rows]
    assert all(n==2 for n in draws.values())
    assert [r['generation_draws'] for r in extractor.evidence_diagnostics]==[2]*16
    assert not broker.physical_inflight
    for p in packets:
        trajectory=p['example']['solver_trajectory']
        assert isinstance(trajectory['visible_solution'],str)
        expected=next(r.signals['solver_trajectory'] for r in rows if r.example_id==p['example']['example_id'])
        assert trajectory==expected


@pytest.mark.parametrize('options', [dict(long_response=True), dict(answer_only=True), dict(format_repair=True)])
def test_actual_gradient_and_reflection_boundaries_have_direct_readable_evidence(tmp_path, options):
    import asyncio
    from multi_dataset_diverse_rl.search.system_prompt import SystemPrompt
    from multi_dataset_diverse_rl.search.readable_evidence import INSTRUCTION
    run,broker,requests,packets,_=graph(tmp_path,**options)
    asyncio.run(run.run(max_opportunities=1))
    gradient_packets=[]
    for request in requests:
        if request['model']=='gpt-4o-mini':
            assert 'Unicode slice' not in json.dumps(request)
            continue
        text=request['messages'][0]['content']
        assert 'Unicode slice' not in text and 'LOSSLESS_OPTIMIZER_TRAJECTORY_ENCODING_V1' not in json.dumps(request)
        if len(request['messages'])==2:
            packet=json.loads(request['messages'][1]['content'])
            if 'example' not in packet:
                assert set(packet)=={'gradients'}
                continue
            assert text.startswith(INSTRUCTION)
            gradient_packets.append(packet)
            example=packet['example'];trajectory=example['solver_trajectory']
            assert isinstance(trajectory['visible_solution'],str)
            assert isinstance(trajectory['observed_response'],str)
            assert trajectory['source']['mutable_prompt_sha256']==SystemPrompt.from_dict(packet['current_system_prompt']).prompt_hash
            assert example['reference_solution']['source']=='dataset_worked_solution'
            assert 'DATASET_WORKED_SOLUTION_' in example['reference_solution']['text']
    assert gradient_packets and packets
    for packet in packets:
        assert packet['selected_pattern']['generalized_gradient']
        assert set(packet['retrieved_memory'])=={'schema','initial_coverage','edit_effects','shared_risk','interpretation'}
        assert packet['retrieved_memory']['schema']=='SCOPED_EDIT_EXPERIENCE_VIEW_V1'
        assert 'not causal' in packet['retrieved_memory']['interpretation']
        for example in packet['current_panel_observations']:
            trajectory=example['solver_trajectory']
            assert isinstance(trajectory['visible_solution'],str)
            assert trajectory['source']['mutable_prompt_sha256']==SystemPrompt.from_dict(packet['current_parent']).prompt_hash
            assert isinstance(trajectory['observed_response'],str) and trajectory['observed_response']
            if trajectory['prediction_valid']:
                assert 'Final answer:' in trajectory['observed_response']
            else:
                assert trajectory['invalid_reason'] and trajectory['solution_status']=='FINAL_BOUNDARY_INVALID'
            if options.get('long_response'):
                assert trajectory['observed_response'].count('Unicode数学核验 π。')==120
                assert trajectory['feedback_truncated'] is True
                assert len(trajectory['observed_response'])>len(trajectory['visible_solution'])==4096
            if options.get('answer_only'):
                assert trajectory['visible_solution']==''
                assert trajectory['solution_status']=='TRAJECTORY_UNAVAILABLE'
    assert broker.usage['validation']==broker.usage['test']==0


def test_current_source_closure_excludes_interval_codec_and_rejects_old_execution_policy():
    from pathlib import Path
    from multi_dataset_diverse_rl.search.solver_execution import frozen_execution_policy
    from multi_dataset_diverse_rl.governance.source_identity import current_scientific_files
    root=Path(__file__).resolve().parents[2]
    for execution_closure in (False,True):
        paths=current_scientific_files(root,execution_closure=execution_closure)
        assert root/'multi_dataset_diverse_rl/search/optimizer_encoding.py' not in paths
        assert root/'multi_dataset_diverse_rl/search/readable_evidence.py' in paths
        assert root/'docs/design/DIRECT_READABLE_OPTIMIZER_EVIDENCE_V2.md' in paths
        assert root/'docs/design/PARALLEL_COMPACT_OPTIMIZER_V1.md' in paths
    with pytest.raises(SearchContractError,match='SOLVER_EXECUTION_POLICY_MISMATCH'):
        frozen_execution_policy({'solver_execution_policy':PARALLEL_COMPACT_POLICY})


def test_old_eight_slot_journal_cannot_be_reopened_as_sixteen_slot(tmp_path):
    from multi_dataset_diverse_rl.governance.token_accounting import POLICY_V25_2M, OperationalAbort
    with TokenLedger(tmp_path,task_sha256='e'*64,policy=POLICY_V25_2M):pass
    with pytest.raises(OperationalAbort,match='TOKEN_LEDGER_CORRUPTION'):
        TokenLedger(tmp_path,task_sha256='e'*64,policy=POLICY_V26_2M)


def test_error_retry_and_slot_queue_timings_are_durable_without_changing_requests(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from openai import APIConnectionError
    import httpx
    lock=threading.Lock();barrier=threading.Barrier(8);draws=Counter();requests=[]
    def transport(req):
        xid=req['messages'][0]['content']
        with lock:
            draws[xid]+=1;n=draws[xid];requests.append(deepcopy(req))
        if int(xid)<8 and n==1:barrier.wait(timeout=15)
        time.sleep(.03)
        if xid=='0' and n==1:
            raise APIConnectionError(request=httpx.Request('POST','https://offline.invalid'))
        return response(json.dumps(output()))
    c=parallel_contract();c['decoding']['retry_sleep_seconds']=.01
    with TokenLedger(tmp_path/'accounting',task_sha256='e'*64,policy=POLICY_V26_2M) as ledger:
        broker=broker_fixture(tmp_path,c,transport,ledger)
        with ThreadPoolExecutor(max_workers=16) as pool:
            values=list(pool.map(lambda i:broker.complete(role='pattern_gradient',split='optimize',
                stage='per_example_textual_gradient',messages=[dict(role='user',content=str(i))]),range(16)))
        assert len(values)==16 and not ledger.inflight
        assert ledger.view()['fallback_charged']>0
    rows=[json.loads(s) for s in (tmp_path/'observations.jsonl').read_text().splitlines()]
    timings=[r for r in rows if r['kind']=='PROVIDER_REQUEST_TIMING']
    assert len(timings)==17
    assert Counter(r['outcome'] for r in timings)=={'RESPONSE':16,'APIConnectionError':1}
    waits=[r for r in rows if r['kind']=='PROVIDER_SLOT_TIMING']
    assert len(waits)==16 and max(r['slot_wait_seconds'] for r in waits)>.01
    assert {r['slot_id'] for r in waits}=={r['request_timing']['slot_id'] for r in timings}
    backoff=next(r for r in rows if r['kind']=='PROVIDER_RETRY_WAIT')
    assert backoff['elapsed_seconds']>=backoff['configured_seconds']==.01
    same=[r for r in requests if r['messages'][0]['content']=='0']
    assert len(same)==2 and same[0]==same[1]
    assert len(list((tmp_path/'receipts').glob('*.json')))==17


def test_old_reservation_capacity_rejected_before_any_provider_call(tmp_path):
    from multi_dataset_diverse_rl.governance.token_accounting import POLICY_V25_2M
    calls=[]
    with TokenLedger(tmp_path,task_sha256='e'*64,policy=POLICY_V25_2M) as ledger:
        from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
        with pytest.raises(SearchContractError,match='EXECUTION_ACCOUNTING_CONCURRENCY_MISMATCH'):
            RequestBroker(contract=parallel_contract(),transport=lambda r:calls.append(r),
                arm='A4',seed=81,token_ledger=ledger)
        assert not calls and not ledger.inflight
