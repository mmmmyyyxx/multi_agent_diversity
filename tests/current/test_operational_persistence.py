# V2.1 frozen replay assertions; V2.2 current conformance is tested separately.
"""Offline Windows durability, crash accounting, and unchanged request witnesses."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import types

import pytest

from multi_dataset_diverse_rl.governance.token_accounting import (
    TokenLedger, reservation, serialized_request, OperationalAbort,
)
from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json
from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts, POLICY
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker

ROOT = Path(__file__).resolve().parents[2]
TASK = 'a' * 64


def request():
    return dict(model='qwen3-8b', max_tokens=1800, temperature=0.,
        extra_body={'enable_thinking':False},
        messages=[{'role':'user','content':'Synthetic reusable fixture'}])


def historical_contract():
    c = json.loads((ROOT/'experiments/execution_bindings/math_v2_execution_v1_2.json').read_bytes())
    return dict(c,execution_attempt_id='synthetic_attempt',execution_phase='canary')


def test_locked_derived_snapshot_detaches_without_paid_response_loss(tmp_path):
    with TokenLedger(tmp_path/'ledger',task_sha256=TASK,best_effort_snapshots=True) as ledger:
        receipts=ProviderResponseReceipts(tmp_path/'responses',attempt_id='synthetic_attempt',
            startup_identity_sha256=TASK)
        snapshot=ledger.snapshot.open('rb')
        try:
            def reconcile(key,result,**kwargs):
                # Critical raw evidence exists before the charge can be appended.
                assert receipts.read(key)['response']==result
                return original(key,result,**kwargs)
            original=ledger.reconcile
            ledger.reconcile=reconcile
            broker=RequestBroker(contract=historical_contract(),arm='A1',seed=81,
                token_ledger=ledger,response_receipts=receipts,
                transport=lambda _:dict(text='FINAL_ANSWER: 1',input_tokens=23,output_tokens=11))
            result=broker.complete(role='solver',split='optimize',stage='initial',messages=request()['messages'])
            assert result['text']=='FINAL_ANSWER: 1'
            assert ledger.view()['charged_total']==34 and ledger.reserved==0
            if os.name=='nt':
                assert ledger.snapshot_updates_disabled
                assert ledger.snapshot_error_category=='PermissionError'
            charge=ledger.events[-1]
            assert charge['response_receipt']['integrity_sha256']
        finally:
            snapshot.close()
    with TokenLedger(tmp_path/'ledger',task_sha256=TASK,best_effort_snapshots=True) as replay:
        assert replay.view()['charged_total']==34 and replay.reserved==0


def test_critical_receipt_failure_stops_before_charge_or_evaluation(tmp_path,monkeypatch):
    with TokenLedger(tmp_path/'ledger',task_sha256=TASK,best_effort_snapshots=True) as ledger:
        receipts=ProviderResponseReceipts(tmp_path/'responses',attempt_id='synthetic_attempt',startup_identity_sha256=TASK)
        def fail(*_):raise OSError('synthetic critical evidence failure')
        monkeypatch.setattr(receipts,'persist',fail)
        broker=RequestBroker(contract=historical_contract(),arm='A1',seed=81,
            token_ledger=ledger,response_receipts=receipts,
            transport=lambda _:dict(text='FINAL_ANSWER: 1',input_tokens=23,output_tokens=11))
        with pytest.raises(OSError,match='critical evidence'):
            broker.complete(role='solver',split='optimize',stage='initial',messages=request()['messages'])
        assert broker.successes==0 and ledger.view()['charged_total']==0
        assert len(ledger.inflight)==1
    with TokenLedger(tmp_path/'ledger',task_sha256=TASK,best_effort_snapshots=True) as replay:
        assert replay.reserved==0 and replay.view()['fallback_charged']>1800


@pytest.mark.parametrize('crash_after',['response','charge'])
def test_process_death_replays_authority_without_rewriting_prefix(tmp_path,crash_after):
    script=tmp_path/'crash.py'
    script.write_text('''import os,sys
from pathlib import Path
from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger
from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
root=Path(sys.argv[1]); task='a'*64
l=TokenLedger(root/'ledger',task_sha256=task,best_effort_snapshots=True)
r=dict(model='qwen3-8b',max_tokens=1800,messages=[dict(role='user',content='synthetic')])
k=l.reserve(r,attempt_id='crash_fixture',stage='pilot',role='solver',model='qwen3-8b')
receipts=ProviderResponseReceipts(root/'responses',attempt_id='crash_fixture',startup_identity_sha256=task)
result=dict(text='FINAL_ANSWER: 1',input_tokens=17,output_tokens=9)
receipt=receipts.persist(k,dict(request=r,response=result))
if sys.argv[2]=='charge':
    l._snapshot=lambda:os._exit(23)
    l.reconcile(k,result,outcome='RESPONSE',response_receipt=receipt)
os._exit(23)
''',encoding='utf-8')
    child=subprocess.run([sys.executable,str(script),str(tmp_path),crash_after],check=False)
    assert child.returncode==23
    journal=tmp_path/'ledger/events.jsonl';prefix=journal.read_bytes()
    rows=[json.loads(line) for line in prefix.splitlines()]
    reserve=next(r for r in rows if r['kind']=='RESERVE')
    receipts=ProviderResponseReceipts(tmp_path/'responses',attempt_id='crash_fixture',startup_identity_sha256=TASK)
    assert receipts.read(reserve['reservation_id'])['response']['text']=='FINAL_ANSWER: 1'
    with TokenLedger(tmp_path/'ledger',task_sha256=TASK,best_effort_snapshots=True) as ledger:
        expected=26 if crash_after=='charge' else reserve['bound']['amount']
        assert ledger.view()['charged_total']==expected and ledger.reserved==0
        assert journal.read_bytes().startswith(prefix)
    recovered=journal.read_bytes()
    with TokenLedger(tmp_path/'ledger',task_sha256=TASK,best_effort_snapshots=True) as again:
        assert again.view()['charged_total']==expected
    assert journal.read_bytes()==recovered
    assert not (tmp_path/'report.json').exists()


def test_1005_cycles_with_concurrent_journal_report_monitor_readers(tmp_path):
    receipts=ProviderResponseReceipts(tmp_path/'responses',attempt_id='stress',startup_identity_sha256=TASK)
    finished=threading.Event();errors=[];reads=[]
    def reader():
        count=0
        try:
            while not finished.is_set():
                for path in (tmp_path/'ledger/events.jsonl',tmp_path/'monitor.jsonl',tmp_path/'report.jsonl'):
                    if path.exists():
                        with path.open('rb') as stream:
                            data=stream.read()
                        # A live journal reader admits only complete lines.
                        for line in data.split(b'\n')[:-1]:json.loads(line)
                count+=1
        except BaseException as exc:errors.append(type(exc).__name__)
        finally:reads.append(count)
    workers=[threading.Thread(target=reader) for _ in range(3)]
    for worker in workers:worker.start()
    try:
        from multi_dataset_diverse_rl.persistence.durable_io import append_jsonl
        with TokenLedger(tmp_path/'ledger',task_sha256=TASK,best_effort_snapshots=True) as ledger:
            for i in range(1005):
                k=ledger.reserve(request(),attempt_id='stress',stage='pilot',role='solver',model='qwen3-8b')
                result=dict(text='FINAL_ANSWER: 1',input_tokens=17,output_tokens=9)
                receipt=receipts.persist(k,dict(request=request(),response=result))
                ledger.reconcile(k,result,outcome='RESPONSE',response_receipt=receipt)
                # Unique immutable checkpoints tolerate arbitrary live readers.
                checkpoint=tmp_path/'checkpoints'/f'{i}.json'
                atomic_write_json(checkpoint,dict(sequence=i,accounting=ledger.view()))
                with checkpoint.open('rb') as held:
                    assert json.load(held)['sequence']==i
                    append_jsonl(tmp_path/'monitor.jsonl',dict(sequence=i))
                    append_jsonl(tmp_path/'report.jsonl',dict(sequence=i,charged=ledger.view()['charged_total']))
            assert not ledger.snapshot_updates_disabled
            assert ledger.view()['charged_total']==1005*26 and ledger.reserved==0
            assert len(ledger.events)==2011
    finally:
        finished.set()
        for worker in workers:worker.join(timeout=10)
    assert not errors and all(reads)
    with TokenLedger(tmp_path/'ledger',task_sha256=TASK,best_effort_snapshots=True) as replay:
        assert replay.view()['charged_total']==1005*26 and replay.reserved==0
        assert len({r['event_sha256'] for r in replay.events})==2011
    assert len(list((tmp_path/'responses').glob('*.json')))==1005


def test_all_four_role_wire_bytes_and_five_member_lanes_match_frozen_baseline():
    frozen=subprocess.check_output(['git','show',
        'a734e198191e0b1da860084e5cc6f37d17e4a951:multi_dataset_diverse_rl/search/provider_runtime.py'],cwd=ROOT)
    module=types.ModuleType('multi_dataset_diverse_rl.search._frozen_broker_witness')
    module.__package__='multi_dataset_diverse_rl.search'
    exec(compile(frozen,'<frozen provider runtime>','exec'),module.__dict__)
    # Current V6 fixture; frozen broker source witnesses unchanged wire physics.
    c=json.loads((ROOT/'experiments/execution_bindings/math_v2_1_gradient_pattern_seed81_pilot_v5.json').read_bytes())
    old=module.RequestBroker(contract=c,transport=None,arm='A4',seed=81)
    new=RequestBroker(contract=dict(c,runtime_persistence_policy=POLICY),transport=None,arm='A4',seed=81)
    messages=[dict(role='user',content='Synthetic scientific request 数学')]
    keys=[]
    for role in ('solver','reflection','pattern_gradient','pattern_cluster'):
        for slot in range(5) if role=='solver' else (None,):
            a,ka=old._request_identity(role=role,split='optimize',messages=messages,member_slot=slot)
            b,kb=new._request_identity(role=role,split='optimize',messages=messages,member_slot=slot)
            assert serialized_request(a)==serialized_request(b) and ka==kb
            if role=='solver':keys.append(kb)
    assert len(set(keys))==5


def test_receipt_conflict_and_tamper_are_critical(tmp_path):
    receipts=ProviderResponseReceipts(tmp_path,attempt_id='a',startup_identity_sha256=TASK)
    first=receipts.persist('1',dict(response='original'))
    assert receipts.persist('1',dict(response='original'))==first
    with pytest.raises(ValueError,match='IMMUTABLE_CONFLICT'):
        receipts.persist('1',dict(response='different'))
    row=json.loads((tmp_path/'1.json').read_bytes());row['record']['response']='tampered'
    atomic_write_json(tmp_path/'1.json',row)
    with pytest.raises(ValueError,match='CORRUPTION'):receipts.read('1')


@pytest.mark.parametrize('body',[{'unexpected':'malformed','usage':{'prompt_tokens':17,'completion_tokens':9}},
    {'choices':[{'message':{'content':'FINAL_ANSWER: 1'},'finish_reason':'stop'}],
     'usage':{'prompt_tokens':17,'completion_tokens':9},'id':'synthetic-response'}])
def test_current_transport_preserves_exact_http_body_offline(monkeypatch,body):
    import httpx
    from openai import OpenAI
    from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
    from multi_dataset_diverse_rl.governance.autonomous_math import create_transport
    captured=[]
    def handler(wire):
        captured.append(wire.content)
        return httpx.Response(200,json=body)
    client=OpenAI(api_key='synthetic',base_url='https://example.invalid/v1',max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(ProviderClientFactory,'from_environment',lambda **_:client)
    c=dict(historical_contract(),runtime_persistence_policy=POLICY)
    transport,owner=create_transport(c)
    try:
        if 'choices' in body:
            result=transport(request())
            assert result['provider_http_response_body']==body
            assert result['text']==body['choices'][0]['message']['content']
        else:
            with pytest.raises(KeyError) as raised:transport(request())
            assert raised.value.provider_evidence['response_body']==body
            assert raised.value.token_usage==dict(input_tokens=17,output_tokens=9)
        assert captured==[serialized_request(request())]
    finally:owner.close()


@pytest.mark.parametrize('field,value',[('shadow_count',41),('stop_policy','changed'),
    ('initial_team_version','changed'),('pattern_abstraction_guard','changed'),
    ('gradient_prompt_sha256','0'*64),('cache_policy','changed')])
def test_operational_binding_rejects_scientific_delta(field,value):
    from multi_dataset_diverse_rl.benchmarks.legacy.math_gradient_pattern_binding_v21 import MATHGradientPatternBinding
    c=json.loads((ROOT/'experiments/execution_bindings/math_v2_1_gradient_pattern_seed81_pilot_v3.json').read_bytes())
    c[field]=value
    assert MATHGradientPatternBinding(ROOT,c).blockers()==('OPERATIONAL_PILOT_SCIENTIFIC_SETTING_CHANGED',)
