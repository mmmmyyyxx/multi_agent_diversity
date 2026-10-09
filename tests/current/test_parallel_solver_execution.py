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

from tests.current.test_optimization_evidence_v23 import contract, response
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
from multi_dataset_diverse_rl.search.solver_execution import POLICY, bounded_ordered_map
from multi_dataset_diverse_rl.persistence.solver_evidence_reuse import POLICY as REUSE
from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
from multi_dataset_diverse_rl.persistence.exact_output_cache import DurableExactOutputCache, digest
from multi_dataset_diverse_rl.governance.token_accounting import (
    TokenLedger, POLICY_V23_PARALLEL_2M, POLICY_V23_2M, OperationalAbort, reservation)
from multi_dataset_diverse_rl.persistence.durable_io import append_jsonl


def parallel_contract():
    c=contract()
    c.update(solver_execution_policy=deepcopy(POLICY),initial_evidence_reuse_policy=deepcopy(REUSE),
        initial_evidence_reuse_manifest_sha256='a'*64,
        accounting_scope_policy='FRESH_V23_PARALLEL_SINGLE_ARM_2M_V1')
    from multi_dataset_diverse_rl.benchmarks.math_prediction_validity import capacity_recovery_contract
    c['invalid_recovery_policy']=capacity_recovery_contract()
    return c


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
        return response('Synthetic arithmetic step.\nFINAL_ANSWER: '+('2' if i%3 else '3'))
    c['decoding']={**c['decoding'],'retry_sleep_seconds':0}
    # 300 logical observations, 270 distinct keys, adjacent in-flight duplicates.
    jobs=tuple(range(8))+tuple(i for i in range(8,30) for _ in range(2))+tuple(range(8))+tuple(range(30,270))
    started=time.monotonic()
    with TokenLedger(path/'accounting',task_sha256='e'*64,policy=POLICY_V23_PARALLEL_2M) as ledger:
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
    requests=[];replies=[('length','unfinished'),('stop','missing marker'),('length','again'),('stop','step\nFINAL_ANSWER: 2')]
    def transport(req):
        requests.append(req);finish,text=replies[len(requests)-1]
        return {**response(text),'finish_reason':finish}
    c=parallel_contract()
    with TokenLedger(tmp_path/'accounting',task_sha256='e'*64,policy=POLICY_V23_PARALLEL_2M) as ledger:
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
        requests.append(req);return {**response('step\nFINAL_ANSWER: 2'),'finish_reason':'length'}
    c=parallel_contract()
    with TokenLedger(tmp_path/'accounting',task_sha256='e'*64,policy=POLICY_V23_PARALLEL_2M) as ledger:
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
        return response('step\nFINAL_ANSWER: 2')
    with TokenLedger(tmp_path/'accounting',task_sha256='e'*64,policy=POLICY_V23_PARALLEL_2M) as ledger:
        broker=broker_fixture(tmp_path,c,transport,ledger)
        with pytest.raises(OperationalAbort,match='PROVIDER_TERMINAL_RuntimeError'):
            bounded_ordered_map(lambda i:call(broker,i),range(300),8)
        assert not ledger.inflight and active==0 and not broker.requests_inflight
        assert len(calls)<=8
        assert len(list((tmp_path/'receipts').glob('*.json')))==len(calls)
        assert ledger.view()['fallback_charged']==reservation(dict(model=c['models']['solver'],
            messages=[dict(role='user',content='Deterministic synthetic problem 0')],
            **__import__('multi_dataset_diverse_rl.benchmarks.math_solver_decoding',fromlist=['generation_request_fields']).generation_request_fields(c,'solver')))['amount']


def test_parallel_scope_cannot_reopen_serial_journal(tmp_path):
    with TokenLedger(tmp_path,task_sha256='e'*64,policy=POLICY_V23_2M):pass
    with pytest.raises(OperationalAbort,match='TOKEN_LEDGER_CORRUPTION'):
        TokenLedger(tmp_path,task_sha256='e'*64,policy=POLICY_V23_PARALLEL_2M)


def synthetic_reuse_source(root,monkeypatch):
    from types import SimpleNamespace
    from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json
    from multi_dataset_diverse_rl.governance.unified_execution import inventory
    from multi_dataset_diverse_rl.persistence.solver_evidence_reuse import audit_source,sha
    from multi_dataset_diverse_rl.benchmarks.math_v21_interface import MATHV21BenchmarkAdapter
    from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
    from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample
    from multi_dataset_diverse_rl.search.provider_runtime import BenchmarkSolver
    from multi_dataset_diverse_rl.benchmarks import math_domain_binding
    c=contract();c['initial_team_path']='team.json';c['task_authorization_sha256']='e'*64
    old=root/'old';old.mkdir();prep=root/'prep.json'
    atomic_write_json(root/'binding.json',c)
    atomic_write_json(root/'team.json',dict(members=[dict(prompt='Generic synthetic procedure.') for _ in range(5)]))
    payload=dict(manifest=dict(source_sha='c'*40,execution_binding=dict(path='binding.json',sha256=sha(root/'binding.json'))))
    payload['startup_identity_sha256']=digest(payload);atomic_write_json(prep,payload)
    adapter=MATHV21BenchmarkAdapter(c)
    examples=tuple(CorrectnessExample(protocol_input('math',str(i),dict(problem='Synthetic initial example '+str(i)),
        adapter.output_contract,protocol=adapter.protocol),'2') for i in range(60))
    binding=SimpleNamespace(benchmark=lambda:adapter,examples=lambda _:examples)
    monkeypatch.setattr(math_domain_binding,'execution_binding',lambda *_:binding)
    ordinals=Counter()
    def transport(req):
        i=int(req['messages'][1]['content'].rsplit(' ',1)[1]);ordinals[i]+=1
        ordinal=(ordinals[i]-1)%4+1
        if i==1:return response('No final marker.')
        if i==2:return {**response('No final marker.'),'finish_reason':'length' if ordinal==4 else 'stop'}
        if i==0:
            # Each member's first request truncates; its second 3600 draw resolves.
            first=ordinals[i]%2==1
            return {**response('Incomplete.' if first else 'step\nFINAL_ANSWER: 2'),
                'finish_reason':'length' if first else 'stop'}
        return response('step\nFINAL_ANSWER: 2')
    cache=DurableExactOutputCache(old/'resolved_output_cache',dict(execution_attempt_id=c['execution_attempt_id'],
        cache_namespace=c['cache_namespace'],startup_identity_sha256=payload['startup_identity_sha256'],source_sha='c'*40,
        authorization_sha256='d'*64,binding_sha256=digest(c),generation_policy_sha256=digest(c['solver_decoding_policy']),
        recovery_policy_sha256=digest(c['invalid_recovery_policy'])))
    with TokenLedger(root/'old_accounting',task_sha256='e'*64,policy=POLICY_V23_2M) as ledger:
        broker=RequestBroker(contract=c,transport=transport,arm='A4',seed=81,token_ledger=ledger,durable_cache=cache,
            response_receipts=ProviderResponseReceipts(old/'provider_response_receipts_private',
                attempt_id=c['execution_attempt_id'],startup_identity_sha256=payload['startup_identity_sha256']),
            raw_writer=lambda r:append_jsonl(old/'provider_trace_private.jsonl',r))
        solver=BenchmarkSolver(adapter,broker)
        results=solver.solve_team_batch(((m,'Generic synthetic procedure.',e.item) for m in range(5) for e in examples),stage='initial',split='optimize')
        atomic_write_json(old/'accounting_end.json',ledger.view())
    atomic_write_json(old/'lifecycle.json',dict(status='EXECUTION_ABORTED'))
    atomic_write_json(old/'raw_evidence_inventory.json',inventory(old))
    source=audit_source(root,prep_path='prep.json',execution_directory='old',accounting_directory='old_accounting')
    atomic_write_json(root/'reuse.json',source)
    new=parallel_contract();new['initial_evidence_reuse_manifest_path']='reuse.json'
    new['initial_evidence_reuse_manifest_sha256']=sha(root/'reuse.json')
    return c,new,source,examples,results


def test_verified_prefix_reuse_reconstructs_300_without_resetting_invalid_draws(tmp_path,monkeypatch):
    from multi_dataset_diverse_rl.persistence.solver_evidence_reuse import SolverEvidenceReuse,sha
    from multi_dataset_diverse_rl.search.provider_runtime import BenchmarkSolver
    from multi_dataset_diverse_rl.benchmarks.math_v21_interface import MATHV21BenchmarkAdapter
    old,new,manifest,examples,_=synthetic_reuse_source(tmp_path,monkeypatch)
    before={p:sha(p) for p in (tmp_path/'old').rglob('*') if p.is_file()}
    reuse=SolverEvidenceReuse(tmp_path,new);requests=[]
    def transport(req):requests.append(req);return response('step\nFINAL_ANSWER: 2')
    assert manifest['logical_profiles']==300 and manifest['reuse_complete_profiles']==295
    assert manifest['affected_profiles']==5
    with TokenLedger(tmp_path/'new_accounting',task_sha256='e'*64,policy=POLICY_V23_PARALLEL_2M) as ledger:
        path=tmp_path/'new';broker=broker_fixture(path,new,transport,ledger,evidence_reuse=reuse)
        solver=BenchmarkSolver(MATHV21BenchmarkAdapter(new),broker)
        profiles=solver.solve_team_batch(((m,'Generic synthetic procedure.',e.item) for m in range(5) for e in examples),stage='initial',split='optimize')
        assert len(profiles)==300 and len(requests)==5 and all(r['max_tokens']==6144 for r in requests)
        assert sum(p['prediction']['prediction_valid'] for p in profiles)==290
        assert ledger.view()['charged_total']==20
        # Old four-draw invalids are preserved, never silently granted another four.
        assert all(profiles[m*60+1]['prediction']['semantic_attempt_count']==4 for m in range(5))
        assert all(profiles[m*60+2]['prediction']['terminal_invalid'] for m in range(5))
        logs=[json.loads(line) for line in (path/'observations.jsonl').read_text().splitlines()]
        assert sum(r['kind']=='PREDICTION_VALIDITY' for r in logs)==300
        assert sum(r['kind']=='EVIDENCE_REUSE' for r in logs)==manifest['reuse_physical_prefix_draws']
        assert all(p['solver_trajectory']['source']['member_id']==i//60 for i,p in enumerate(profiles))
        from multi_dataset_diverse_rl.governance.canary_review import profile_audit
        audit=profile_audit(new,solver,path,((i//60,'Generic synthetic procedure.',examples[i%60],p)
            for i,p in enumerate(profiles)))
        assert audit['logical_profiles']==300 and audit['new_physical_realizations']==5
        assert audit['reused_physical_realizations']==manifest['reuse_physical_prefix_draws']
        assert audit['output_capacity_counts']['6144']==5
        # Every observed final profile links through its new cache to old original hashes.
        for profile in profiles:
            cached=broker.durable_cache.get(profile['solver_trajectory']['source']['request_sha256'])
            assert cached['original_realizations'][0]['evidence_reuse_source']['source_attempt']==old['execution_attempt_id']
    assert before=={p:sha(p) for p in before}


def test_reuse_rejects_changed_manifest_or_source_before_provider(tmp_path,monkeypatch):
    from multi_dataset_diverse_rl.persistence.solver_evidence_reuse import SolverEvidenceReuse
    from multi_dataset_diverse_rl.search.schemas import SearchContractError
    _,new,_,_,_=synthetic_reuse_source(tmp_path,monkeypatch)
    p=tmp_path/'old_accounting/events.jsonl';original=p.read_bytes();p.write_bytes(original+b'{}\n')
    with pytest.raises(SearchContractError,match='EVIDENCE_REUSE_LEDGER_CORRUPTION|EVIDENCE_REUSE_SOURCE_CHANGED'):
        SolverEvidenceReuse(tmp_path,new)


def test_three_capacity_canary_barriers_require_matching_owner_receipts(tmp_path,monkeypatch):
    from multi_dataset_diverse_rl.governance import canary_review
    from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json
    original=canary_review.atomic_write_json;seen=[]
    def synthetic_owner(path,value):
        original(path,value)
        if path.name.endswith('.review_pending.json'):
            seen.append(value['stage'])
            atomic_write_json(path.parent/(value['stage']+'.owner_review.json'),dict(approved=True,
                scientific_method_changed=False,review_identity_sha256=value['review_identity_sha256'],
                startup_identity_sha256=value['startup_identity_sha256']))
    monkeypatch.setattr(canary_review,'atomic_write_json',synthetic_owner)
    c=dict(canary_review_policy=canary_review.CAPACITY_POLICY,execution_attempt_id='synthetic')
    for stage in canary_review.CAPACITY_POLICY['stages']:
        canary_review.review(stage,dict(actual_evidence='SYNTHETIC_FIXTURE_ONLY'),contract=c,
            payload=dict(startup_identity_sha256='a'*64),run_root=tmp_path)
        assert (tmp_path/(stage+'.review_pass.json')).exists()
    assert seen==['EARLY_SOLVER_CAPACITY','INITIAL_SOLVER_PROFILE','FIRST_COMPLETE_OPPORTUNITY']
    assert c['canary_review_policy']['maximum_wait_seconds']==3600


def test_new_manifest_schema_requires_complete_parallel_policy_bundle():
    from pathlib import Path
    import yaml
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    root=Path(__file__).resolve().parents[2]
    m=yaml.safe_load((root/'experiments/manifests/a4_v23_parallel_reuse_pilot_v1.yaml').read_text())
    assert validate_manifest_v2(root,m)==[]
    for key in ('solver_execution_policy','initial_evidence_reuse_policy','initial_evidence_reuse_manifest_sha256',
            'canary_review_policy','invalid_recovery_policy','accounting_scope_policy'):
        broken=deepcopy(m);broken.pop(key)
        assert validate_manifest_v2(root,broken),key
    broken=deepcopy(m);broken['concurrency']['solver']=9
    assert validate_manifest_v2(root,broken)


def test_optimizer_transport_remains_serial_with_eight_solver_capacity(tmp_path):
    active=peak=0;mutex=threading.Lock()
    def transport(req):
        nonlocal active,peak
        with mutex:active+=1;peak=max(peak,active)
        time.sleep(0.02)
        with mutex:active-=1
        return response('{}')
    c=parallel_contract()
    with TokenLedger(tmp_path/'accounting',task_sha256='e'*64,policy=POLICY_V23_PARALLEL_2M) as ledger:
        broker=broker_fixture(tmp_path,c,transport,ledger)
        results=bounded_ordered_map(lambda i:broker.complete(role='pattern_gradient',split='optimize',
            stage='synthetic_gradient',messages=[dict(role='user',content=str(i))]),range(12),8)
        assert len(results)==12 and peak==1 and active==0 and ledger.view()['reserved_inflight']==0
        assert broker.usage['pattern_gradient']==12
