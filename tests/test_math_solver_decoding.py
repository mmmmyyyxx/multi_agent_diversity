"""Exact SDK wire capture and fail-closed controls for frozen Solver sampling."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import httpx
from openai import OpenAI
import pytest

from multi_dataset_diverse_rl.benchmarks.math_solver_decoding import solver_decoding_contract
from multi_dataset_diverse_rl.benchmarks.math_v21_binding import MATHV21Binding
from multi_dataset_diverse_rl.benchmarks.math_accounting_prep import solver_request, ValidationReserve
from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
from multi_dataset_diverse_rl.governance.legacy.autonomous_math import create_transport
from multi_dataset_diverse_rl.governance.token_accounting import serialized_request, reservation
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker, BenchmarkSolver
from multi_dataset_diverse_rl.search.schemas import SearchContractError

ROOT=Path(__file__).resolve().parents[1]


def contract(version=5):
    return json.loads((ROOT/f'experiments/execution_bindings/math_v2_1_canary_v{version}.json').read_bytes())


def assert_wire(body):
    policy=solver_decoding_contract()
    expected={k:v for k,v in policy.items() if k not in {'identity','max_output_tokens'}}
    expected['max_tokens']=policy['max_output_tokens']
    assert {k:body[k] for k in expected}==expected
    assert type(body['enable_thinking']) is bool
    assert 'extra_body' not in body and 'solver_decoding_policy' not in body
    return expected


def test_all_solver_roles_and_evolved_prompts_reach_exact_sdk_wire(monkeypatch):
    c=contract();binding=MATHV21Binding(ROOT,c);adapter=binding.benchmark();wires=[];captures=[]
    def handler(request):
        wires.append(request.content)
        return httpx.Response(200,json={'choices':[{'message':{'content':'FINAL_ANSWER: 1'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':2,'completion_tokens':3}})
    client=OpenAI(api_key='fake-offline',base_url='https://offline.invalid/v1',max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr('multi_dataset_diverse_rl.provider_factory.ProviderClientFactory.from_environment',lambda **_:client)
    transport,_=create_transport(c)
    initial=[m['prompt'] for m in json.loads((ROOT/c['initial_team_path']).read_bytes())['members']]
    evolved=['Check constraints before deriving the result.','Enumerate compatible cases before solving.','Verify substitutions after deriving a candidate result.']
    stages=['initial','optimize','gepa_local','team_probe','full','shadow','validation_initial','validation_final']
    try:
        for stage in stages:
            split='validation' if stage.startswith('validation') else 'shadow' if stage=='shadow' else 'optimize'
            broker=RequestBroker(contract=c,transport=transport,arm='A1',seed=81,validation_only=split=='validation')
            solver=BenchmarkSolver(adapter,broker)
            for i,prompt in enumerate(initial+evolved):
                problem=f'Synthetic arithmetic for {stage} and procedure {i}.'
                item=protocol_input('math',f'{stage}_{i}',{'problem':problem},adapter.output_contract,protocol=adapter.protocol)
                solver.solve(prompt,item,stage=stage,split=split)
                exact=solver_request(c,prompt,problem)
                assert wires[-1]==serialized_request(exact)
                body=json.loads(wires[-1]);fields=assert_wire(body)
                captures.append(dict(stage=stage,split=split,prompt_kind='initial' if i<5 else 'evolved',
                    member_or_prompt_index=i,body_sha256=hashlib.sha256(wires[-1]).hexdigest(),fields=fields))
        optimizer=RequestBroker(contract=c,transport=transport,arm='A1',seed=81)
        optimizer.complete(role='reflection',split='optimize',stage='optimizer',messages=[{'role':'user','content':'Synthetic optimization context.'}])
        body=json.loads(wires[-1])
        assert body['temperature']==0.0 and body['max_tokens']==1800
        assert not set(('enable_thinking','top_p','top_k','min_p','presence_penalty','frequency_penalty'))&body.keys()
        destination=os.environ.get('FORMAL_V3_EVIDENCE_CAPTURE_DIR')
        if destination:
            p=Path(destination);p.mkdir(parents=True,exist_ok=True)
            (p/'solver_decoding_capture.json').write_text(json.dumps(dict(policy=solver_decoding_contract(),
                mapping={'max_output_tokens':'max_tokens'},captures=captures,solver_calls=len(captures),
                optimizer_fields={'temperature':body['temperature'],'max_tokens':body['max_tokens']},
                real_provider_calls=0),sort_keys=True,indent=2)+'\n',encoding='utf-8')
    finally:client.close()


@pytest.mark.parametrize('field',list(solver_decoding_contract()))
@pytest.mark.parametrize('mutation',['missing','different'])
def test_frozen_policy_poison_is_denied_before_transport(field,mutation):
    c=contract()
    if mutation=='missing':del c['solver_decoding_policy'][field]
    else:c['solver_decoding_policy'][field]='poison'
    assert MATHV21Binding(ROOT,c).blockers()
    broker=RequestBroker(contract=c,transport=lambda _:pytest.fail('provider dispatched'),arm='A1',seed=81)
    with pytest.raises(SearchContractError,match='SOLVER_DECODING_POLICY_BINDING_MISMATCH'):
        broker.complete(role='solver',split='optimize',stage='initial',messages=[])


def test_bool_and_numeric_aliases_cannot_poison_policy():
    for field,value in [('enable_thinking',0),('top_k',20.0),('min_p',False),('temperature','0.7')]:
        c=contract();c['solver_decoding_policy'][field]=value
        assert MATHV21Binding(ROOT,c).blockers()


def test_legacy_request_and_cache_do_not_adopt_new_policy():
    old=contract(4);new=contract();records=[]
    transport=lambda r:records.append(r) or dict(text='FINAL_ANSWER: 1',input_tokens=1,output_tokens=1)
    a=RequestBroker(contract=old,transport=transport,arm='A1',seed=81)
    b=RequestBroker(contract={**new,'cache_namespace':old['cache_namespace']},transport=transport,arm='A1',seed=81)
    a.complete(role='solver',split='optimize',stage='initial',messages=[])
    b.cache.update(a.cache)
    result=b.complete(role='solver',split='optimize',stage='initial',messages=[])
    assert result['provider_called'] and len(records)==2
    assert records[0]==dict(model='qwen3-8b',messages=[],temperature=0.0,max_tokens=3600,extra_body={'enable_thinking':False})
    assert len(a.cache)==1 and len(b.cache)==2
    old['solver_decoding_policy']=solver_decoding_contract()
    assert MATHV21Binding(ROOT,old).blockers()


@pytest.mark.parametrize('text,finish',[('Answer: 1','stop'),('FINAL_ANSWER: 1\nFINAL_ANSWER: 2','stop'),('FINAL_ANSWER: 1','length')])
def test_invalid_or_truncated_solver_stops_without_regeneration(text,finish):
    c=contract();adapter=MATHV21Binding(ROOT,c).benchmark();calls=[]
    def transport(req):
        calls.append(req);return dict(text=text,finish_reason=finish,input_tokens=1,output_tokens=1)
    solver=BenchmarkSolver(adapter,RequestBroker(contract=c,transport=transport,arm='A1',seed=81))
    item=protocol_input('math','synthetic',{'problem':'Synthetic arithmetic.'},adapter.output_contract,protocol=adapter.protocol)
    with pytest.raises(SearchContractError,match='STOP_SOLVER_DECODING_POLICY_INSUFFICIENT'):
        solver.solve('Check constraints.',item,stage='initial',split='optimize')
    assert len(calls)==1


def test_policy_bound_reserve_equals_exact_serialized_request_reservations():
    c=contract();problems=['Synthetic unicode α arithmetic.','Synthetic relation.'];prompts=['Check constraints.']*5
    metadata=dict(decoding=c['decoding'],solver_decoding_policy=c['solver_decoding_policy'],
        examples=[dict(blank_prompt_serialized_request_bytes=len(serialized_request(solver_request(c,'',p)))) for p in problems])
    reserve=ValidationReserve(metadata,prompts)
    assert reserve.remaining()==2*sum(reservation(solver_request(c,p,pb))['amount'] for p in prompts for pb in problems)


def test_manifest_and_authorization_bind_exact_solver_policy():
    from multi_dataset_diverse_rl.governance.legacy.unified_execution import preexecution_manifest,execution_scope
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    c=contract();m=preexecution_manifest(ROOT,source_sha='a'*40,binding_path=c['binding_path'],frozen=False)
    assert not validate_manifest_v2(ROOT,m)
    assert m['solver_decoding_policy']==execution_scope(m,c)['solver_decoding_policy']==solver_decoding_contract()
    del m['solver_decoding_policy'];assert validate_manifest_v2(ROOT,m)
