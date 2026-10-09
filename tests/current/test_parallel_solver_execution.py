"""Real fake-transport concurrency, durable costs, ordered observations and failure drain."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import threading
import time

import pytest
from openai import APIConnectionError
import httpx

from tests.current.test_structured_optimization_evidence import contract, response
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
from multi_dataset_diverse_rl.search.solver_execution import POLICY, bounded_ordered_map
from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
from multi_dataset_diverse_rl.persistence.exact_output_cache import DurableExactOutputCache, digest
from multi_dataset_diverse_rl.governance.token_accounting import (
    TokenLedger, POLICY_V25_2M, POLICY_V23_2M, OperationalAbort, reservation)
from multi_dataset_diverse_rl.persistence.durable_io import append_jsonl


def parallel_contract():
    return contract()


def broker_fixture(tmp_path,c,transport,ledger,**kwargs):
    cache=DurableExactOutputCache(tmp_path/'cache',dict(execution_attempt_id=c['execution_attempt_id'],
        cache_namespace=c['cache_namespace'],startup_identity_sha256='b'*64,source_sha='c'*40,
        authorization_sha256='d'*64,binding_sha256=digest(c),generation_policy_sha256=digest(c['solver_decoding_policy']),
        recovery_policy_sha256=digest(c['invalid_recovery_policy'])))
    return RequestBroker(contract=c,transport=transport,arm='A4',seed=81,token_ledger=ledger,
        durable_cache=cache,response_receipts=ProviderResponseReceipts(tmp_path/'receipts',
            attempt_id=c['execution_attempt_id'],startup_identity_sha256='b'*64),
        raw_writer=lambda r:append_jsonl(tmp_path/'provider_trace_private.jsonl',r),
        ledger_writer=lambda r:append_jsonl(tmp_path/'observations.jsonl',r),**kwargs)


def call(broker,i):
    return broker.complete(role='solver',split='optimize',stage='initial',
        messages=[dict(role='user',content='Deterministic synthetic problem '+str(i))],member_slot=i%5)


def measure_fixture(path,concurrency):
    path.mkdir();c=parallel_contract();mutex=threading.Lock()
    active=peak=0;calls=[];completed=[];perkey=Counter()
    first_wave=threading.Barrier(8) if concurrency==8 else None
    def transport(req):
        nonlocal active,peak
        i=int(req['messages'][0]['content'].rsplit(' ',1)[1])
        with mutex:
            active+=1;peak=max(peak,active);calls.append(deepcopy(req));perkey[i]+=1
            ordinal=perkey[i]
        if first_wave is not None and i<8 and ordinal==1:
            first_wave.wait(timeout=15)
        time.sleep(0.025+(i%7)*0.004)
        with mutex:active-=1;completed.append(i)
        if i%53==0 and ordinal==1:
            raise APIConnectionError(request=httpx.Request('POST','https://offline.invalid'))
        return response('Synthetic arithmetic step.\nFinal answer: '+('2' if i%3 else '3'))
    c['decoding']={**c['decoding'],'retry_sleep_seconds':0}
    # 300 logical observations, 270 distinct keys, adjacent in-flight duplicates.
    jobs=tuple(range(8))+tuple(i for i in range(8,30) for _ in range(2))+tuple(range(8))+tuple(range(30,270))
    started=time.monotonic()
    with TokenLedger(path/'accounting',task_sha256='e'*64,policy=POLICY_V25_2M) as ledger:
        broker=broker_fixture(path,c,transport,ledger)
        results=bounded_ordered_map(lambda i:call(broker,i),jobs,concurrency)
        view=ledger.view();events=tuple(ledger.events)
        assert not ledger.inflight and not broker.requests_inflight and not broker.physical_inflight
    assert active==0 and len(results)==300
    assert Counter(perkey.values())==Counter({1:264,2:6})
    assert len(list((path/'receipts').glob('*.json')))==276
    assert len(list((path/'cache').glob('*.json')))==271 # Includes immutable scope.
    raw=[json.loads(line) for line in (path/'provider_trace_private.jsonl').read_text().splitlines()]
    assert len(raw)==276 and len({r['physical_attempt_no'] for r in raw})==276
    assert sum(r['kind']=='RESERVE' for r in events)==276
    assert sum(r['kind']=='CHARGE' for r in events)==276
    assert view['authorized_total']==2_000_000 and view['reserved_inflight']==0
    assert view['fallback_charged']>0 and view['charged_total']>view['provider_reported_actual']
    return dict(jobs=jobs,results=results,requests=Counter(digest(r) for r in calls),peak=peak,
        completed=completed,view=view,elapsed=time.monotonic()-started,
        logical=300,physical=276,unique=270,receipt_count=276)


def test_300_serial_parallel_equivalence_dedup_ledger_receipts_and_actual_inflight(tmp_path):
    serial=measure_fixture(tmp_path/'serial',1);parallel=measure_fixture(tmp_path/'parallel',8)
    assert serial['peak']==1 and parallel['peak']==8
    assert serial['jobs']==parallel['jobs'] and serial['requests']==parallel['requests']
    projection=lambda r:(r['request_sha256'],r['text'],r['resolved_prediction'],
        tuple((x['request_sha256'],x['text'],x['transport_retry_attempts']) for x in r['original_realizations']))
    assert tuple(map(projection,serial['results']))==tuple(map(projection,parallel['results']))
    score=lambda r:r['resolved_prediction']['prediction_valid'] and r['resolved_prediction']['answer']=='2'
    assert sum(map(score,serial['results']))==sum(map(score,parallel['results']))==200
    for key in ('charged_total','provider_reported_actual','fallback_charged','input_tokens','output_tokens'):
        assert serial['view'][key]==parallel['view'][key]
    assert parallel['completed']!=sorted(parallel['completed'])
    # Export only measured counters/identity, never synthetic request content.
    evidence=tmp_path/'concurrency_proof.json'
    evidence.write_text(json.dumps(dict(logical_observations=300,unique_keys=270,
        physical_attempts=276,transport_failures_fully_reserved=6,
        first_parallel_wave_synchronized_for_peak_measurement=True,
        serial_peak=serial['peak'],parallel_peak=parallel['peak'],out_of_order=True,
        serial_seconds=serial['elapsed'],parallel_seconds=parallel['elapsed'],ordered_score=200,
        same_requests=True,same_provenance=True,same_costs=True,
        lost_receipts=0,duplicate_physical_ids=0,inflight_remaining=0),indent=2))
    import os
    target=os.environ.get('V23_CONCURRENCY_EVIDENCE')
    if target:
        from pathlib import Path
        Path(target).write_bytes(evidence.read_bytes())


def test_capacity_only_after_length_and_never_resets_four_draws(tmp_path):
    requests=[];replies=[('length','unfinished'),('stop','missing marker'),('length','again'),('stop','step\nFinal answer: 2')]
    def transport(req):
        requests.append(req);finish,text=replies[len(requests)-1]
        return {**response(text),'finish_reason':finish}
    c=parallel_contract()
    with TokenLedger(tmp_path/'accounting',task_sha256='e'*64,policy=POLICY_V25_2M) as ledger:
        broker=broker_fixture(tmp_path,c,transport,ledger);result=call(broker,1)
        assert [r['max_tokens'] for r in requests]==[3600,6144,3600,6144]
        assert result['resolved_prediction']['semantic_attempt_count']==4
        assert result['resolved_prediction']['prediction_valid']
        assert digest(broker.durable_cache.get(result['request_sha256']))==digest(result)
        assert not call(broker,1)['provider_called']
    assert len(requests)==4


def test_terminal_length_is_invalid_and_no_8192_or_fifth_call(tmp_path):
    requests=[]
    def transport(req):
        requests.append(req);return {**response('step\nFinal answer: 2'),'finish_reason':'length'}
    c=parallel_contract()
    with TokenLedger(tmp_path/'accounting',task_sha256='e'*64,policy=POLICY_V25_2M) as ledger:
        result=call(broker_fixture(tmp_path,c,transport,ledger),1)
    assert [r['max_tokens'] for r in requests]==[3600,6144,6144,6144]
    assert result['resolved_prediction']['terminal_invalid'] and result['resolved_prediction']['answer']==''


def test_batch_fatal_failure_stops_dispatch_drains_and_persists(tmp_path):
    calls=[];active=0;mutex=threading.Lock();c=parallel_contract()
    def transport(req):
        nonlocal active
        i=int(req['messages'][0]['content'].rsplit(' ',1)[1])
        with mutex:calls.append(i);active+=1
        time.sleep(0.01 if i==0 else 0.15)
        with mutex:active-=1
        if i==0:raise RuntimeError('SYNTHETIC_TERMINAL_FAILURE')
        return response('step\nFinal answer: 2')
    with TokenLedger(tmp_path/'accounting',task_sha256='e'*64,policy=POLICY_V25_2M) as ledger:
        broker=broker_fixture(tmp_path,c,transport,ledger)
        with pytest.raises(OperationalAbort,match='PROVIDER_TERMINAL_RuntimeError'):
            bounded_ordered_map(lambda i:call(broker,i),range(300),8)
        assert not ledger.inflight and active==0 and not broker.requests_inflight
        assert len(calls)<=8
        assert len(list((tmp_path/'receipts').glob('*.json')))==len(calls)
        assert ledger.view()['fallback_charged']==reservation(dict(model=c['models']['solver'],
            messages=[dict(role='user',content='Deterministic synthetic problem 0')],
            **__import__('multi_dataset_diverse_rl.benchmarks.math_solver_decoding',fromlist=['generation_request_fields']).generation_request_fields(c,'solver')))['amount']


def test_structured_scope_cannot_reopen_historical_serial_journal(tmp_path):
    with TokenLedger(tmp_path,task_sha256='e'*64,policy=POLICY_V23_2M):pass
    with pytest.raises(OperationalAbort,match='TOKEN_LEDGER_CORRUPTION'):
        TokenLedger(tmp_path,task_sha256='e'*64,policy=POLICY_V25_2M)
