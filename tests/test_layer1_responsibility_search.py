import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl import versions
from multi_dataset_diverse_rl.benchmarks.math_solver_decoding import solver_decoding_contract,generation_request_fields
from multi_dataset_diverse_rl.benchmarks.math_optimizer_generation import optimizer_generation_contract
from multi_dataset_diverse_rl.evaluation.semantic_mutable_contract import candidate_failed_checks,mutation_shape
from multi_dataset_diverse_rl.evaluation.mutable_prompt_contract import mutable_prompt_violation_reasons
from multi_dataset_diverse_rl.local_optimizers.base import LocalSolverObservation
from multi_dataset_diverse_rl.local_optimizers.schemas import LocalEvidenceExample,LocalOptimizationTask,LocalOptimizerBudget
from multi_dataset_diverse_rl.search.layer1_responsibility import ResponsibilityConditionedOptimizer,generation_input,observation_record
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.governance.team_change import team_change_receipt
from multi_dataset_diverse_rl.governance.legacy.math_paired_validation import require_validation_intervention


@pytest.mark.parametrize('text',[
    'Verify the final answer by substituting into the original equation.',
    'Check the final answer against domain restrictions and reconsider invalid roots.',
    'Simplify fractions and use exact symbolic expressions; check units and rationalize denominators.',
    'Ensure the answer matches the mathematical form requested, such as interval notation.',
    'Determine the target output format: exact value, simplified expression, or a set of solutions.',
    'Ensure the answer format matches the requested mathematical style, such as interval notation.',
    'Check equivalent forms through substitution and signs; remove extraneous roots.',
    'Compare the slopes of two lines and verify their intersection.',
])
def test_semantic_math_reasoning_allowed(text):
    assert candidate_failed_checks(text,parent_prompt='Inspect constraints.',examples=())==()


@pytest.mark.parametrize('text',[
    'End your response with FINAL_ANSWER: 7.',
    'Return exactly one line containing the result.',
    'Reply only with the result.',
    'Place the answer on the last line.',
    'Format the response as JSON with an answer field.',
    'Use Markdown and triple-backtick code fences.',
    'Do not include intermediate reasoning in the final output.',
    'Do not include extraneous commentary in the final answer block.',
    'Always return 42 regardless of the problem.',
])
def test_external_protocol_and_fixed_payload_rejected(text):
    assert candidate_failed_checks(text,parent_prompt='Inspect constraints.',examples=())


def test_append_shape_and_normalized_noop():
    p='Inspect constraints.';c=p+' Verify the final answer against the domain.'
    assert candidate_failed_checks(c,parent_prompt=p,examples=())==()
    assert mutation_shape(c,p)=='append_only'
    assert 'parent_no_op' in candidate_failed_checks('  INSPECT   constraints.\n',parent_prompt=p,examples=())
    assert mutable_prompt_violation_reasons(c)==('forbidden_final_answer_marker',)


def test_example_copy_is_not_relaxed():
    row=LocalEvidenceExample('x','One two three four five six seven eight nine ten eleven twelve thirteen.','2')
    assert 'example_copying' in candidate_failed_checks('Check One two three four five six seven eight nine ten eleven twelve.',parent_prompt='Root.',examples=(row,))


def task():
    rows=tuple(LocalEvidenceExample(str(i),'Synthetic problem '+str(i),'2',tags=('repair' if i<3 else 'preservation',)) for i in range(6))
    return LocalOptimizationTask('bounded_test','Inspect constraints.',rows[:3],rows[3:],'target lane direct_flip',
        'COMMON_SOLVER_CONTRACT_V1','test_output',81,LocalOptimizerBudget(36,3,4),target_member=3)


class SyntheticEvaluator:
    solver_contract_id='COMMON_SOLVER_CONTRACT_V1';output_contract_id='test_output'
    def __init__(self):self.calls=[]
    def evaluate(self,prompt,row):
        self.calls.append((prompt,row.example_id))
        # The changed procedure is locally worse, while the root is correct.
        correct=prompt=='Inspect constraints.'
        return LocalSolverObservation('2' if correct else '3','FINAL_ANSWER: '+('2' if correct else '3'),correct,True,input_tokens=4,output_tokens=2)


class SyntheticReflection:
    def __init__(self):self.requests=[]
    def __call__(self,prompt):
        self.requests.append(prompt)
        n=len(self.requests)
        return json.dumps({'decision_procedure':f'Inspect constraints. Verify signs through {n} independent checks.'})
    def accounting(self):return dict(successful_calls=len(self.requests),input_tokens=10*len(self.requests),output_tokens=4*len(self.requests))


def test_negative_nonfrontier_archive_export_and_budget(tmp_path):
    evaluator=SyntheticEvaluator();reflection=SyntheticReflection()
    backend=ResponsibilityConditionedOptimizer(evaluator=evaluator,reflection_lm=reflection,
        accounting_reader=reflection.accounting,run_root=tmp_path)
    result=asyncio.run(backend.search_task(task(),'op:3'))
    assert len(evaluator.calls)==36
    assert len(reflection.requests)==5
    assert len(result.candidates)==4
    assert all(c.backend_details['LOCAL_REJECTED_EXPORTED'] for c in result.candidates)
    assert all(c.backend_details['official_gepa_frontier'] is False for c in result.candidates)
    assert result.search_state['telemetry']['local_solver_reached']==5
    request=json.loads(reflection.requests[1].rsplit('\n',1)[1])
    assert request['recent_candidate_observations'][0]['prediction']=='3'
    assert request['root_observations'][0]['correct'] is True
    assert 'No reusable reasoning trace' not in reflection.requests[0]
    assert 'FINAL_ANSWER:' not in reflection.requests[0]
    assert result.solver_tokens==36*6


def test_observation_feedback_changes_real_proposer_input():
    t=task();row=t.search_examples[0]
    a=observation_record(row,LocalSolverObservation('3','FINAL_ANSWER: 3',False,True))
    b=observation_record(row,LocalSolverObservation(None,'bad',False,False,'MISSING_FINAL_MARKER'))
    assert generation_input(t,t.parent_prompt,[a],[],[],1)!=generation_input(t,t.parent_prompt,[b],[],[],1)
    assert generation_input(t,t.parent_prompt,[a],[],[],1)!=generation_input(replace(t,optimization_context='target lane coverage'),t.parent_prompt,[a],[],[],1)


def contract():
    c=json.loads(Path('experiments/execution_bindings/math_v2_1_low_cost_pilot_v4.json').read_bytes())
    c.update(identity=versions.MATH_LAYER1_EXECUTION_BINDING_VERSION,cache_policy=versions.SOLVER_MEMBER_LANE_CACHE_VERSION,
        solver_decoding_policy=solver_decoding_contract(versions.MATH_SOLVER_DECODING_POLICY_V2_VERSION),
        optimizer_generation_policy=optimizer_generation_contract(versions.MATH_OPTIMIZER_GENERATION_POLICY_V3_VERSION))
    return c


def fake_response(request):
    return dict(text='FINAL_ANSWER: 2' if request['model']=='qwen3-8b' else '{}',finish_reason='stop',input_tokens=10,output_tokens=3,
        provider_response_accepted=True,provider_metadata_loss_audited=True,provider_reasoning_content_present=False,
        provider_reasoning_character_count=None,provider_usage_details={},provider_thinking_indicators=[])


def test_solver_reuses_same_lane_only_and_generation_never_reuses():
    c=contract();calls=[]
    def transport(request):calls.append(request);return fake_response(request)
    broker=RequestBroker(contract=c,transport=transport,arm='A2',seed=81)
    messages=[{'role':'user','content':'same'}]
    kwargs=dict(role='solver',split='optimize',messages=messages)
    a=broker.complete(**kwargs,stage='initial',member_slot=1)
    b=broker.complete(**kwargs,stage='full',member_slot=1)
    d=broker.complete(**kwargs,stage='initial',member_slot=3)
    assert a['provider_called'] and not b['provider_called'] and d['provider_called']
    assert a['request_sha256']!=d['request_sha256']
    assert calls[0]==calls[1] # Slot changes identity, not the provider body.
    for role in ('reflection','pattern'):
        for _ in range(2):assert broker.complete(role=role,split='optimize',stage='generation',messages=messages)['provider_called']
    assert len(calls)==6
    assert broker.usage['reflection']==2 and broker.usage['pattern']==2
    with pytest.raises(SearchContractError,match='LANE_REQUIRED'):
        broker.complete(**kwargs,stage='initial')
    with pytest.raises(SearchContractError,match='FORBIDDEN'):
        broker.complete(role='reflection',split='shadow',stage='bad',messages=messages)
    for heldout in ('validation','test'):
        with pytest.raises(SearchContractError,match='FORBIDDEN'):
            broker.complete(role='solver',split=heldout,stage='bad',messages=messages,member_slot=1)


def test_changed_generation_wire_and_legacy_parity():
    c=contract();s=generation_request_fields(c,'solver');o=generation_request_fields(c,'reflection')
    assert s['temperature']==0.2 and s['max_tokens']==3600
    assert s['extra_body']==dict(enable_thinking=False,top_k=20,min_p=0)
    assert o['temperature']==0.7 and o['top_p']==0.8 and o['presence_penalty']==1.5
    assert o['extra_body']==dict(enable_thinking=False,top_k=20)
    assert o['max_completion_tokens']==1800 and 'max_tokens' not in o
    assert generation_request_fields(c,'pattern')==o
    old=json.loads(Path('experiments/execution_bindings/math_v2_1_low_cost_pilot_v4.json').read_bytes())
    assert generation_request_fields(old,'solver')['temperature']==0.7
    assert generation_request_fields(old,'reflection')['temperature']==0.0


def test_final_provider_body_dispatch_has_no_dropped_extensions(monkeypatch):
    import httpx
    from openai import OpenAI
    from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
    from multi_dataset_diverse_rl.governance.legacy.autonomous_math import create_transport
    captured=[]
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200,json={'id':'synthetic','choices':[{'message':{'content':'{}'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':2,'completion_tokens':2}})
    client=OpenAI(api_key='synthetic-test-value',base_url='https://example.invalid/v1',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(ProviderClientFactory,'from_environment',lambda **kw:client)
    transport,owned=create_transport(contract())
    try:
        for role in ('solver','reflection','pattern'):
            transport(dict(model='qwen3-8b' if role=='solver' else 'qwen3.7-flash',
                messages=[{'role':'user','content':'synthetic'}],**generation_request_fields(contract(),role)))
    finally:owned.close()
    assert captured[0]['temperature']==0.2 and captured[0]['min_p']==0
    for body in captured:
        assert body['enable_thinking'] is False and body['top_k']==20 and body['top_p']==0.8
        assert 'extra_body' not in body
    for body in captured[1:]:
        assert body['presence_penalty']==1.5 and body['frequency_penalty']==0
        assert body['max_completion_tokens']==1800 and 'max_tokens' not in body


def test_deployed_hash_gate_uses_ordered_prompts():
    initial=['p'+str(i) for i in range(5)]
    a=team_change_receipt(initial,list(initial));assert a['validation_status']=='SKIPPED_NO_TEAM_CHANGE'
    assert a['pilot_signal']=='NOT_ESTIMATED' and a['search_outcome']=='NO_INTERVENTION'
    final=list(initial);final[3]='changed';b=team_change_receipt(initial,final)
    assert b['changed_members']==[3] and b['team_changed']
    assert team_change_receipt(initial,list(reversed(initial)))['changed_member_count']==4


def test_validation_preparation_blocks_unchanged_team_before_calls(tmp_path,monkeypatch):
    from multi_dataset_diverse_rl.governance.legacy import math_paired_validation as validation
    c=contract();initial=['procedure '+str(i) for i in range(5)]
    p=tmp_path/c['initial_team_path'];p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'members':[{'prompt':x} for x in initial]}),encoding='utf-8')
    run=tmp_path/'runs/pilot';run.mkdir(parents=True)
    final_path=run/'final_team_private.json';final_path.write_text(json.dumps({'prompts':initial}),encoding='utf-8')
    receipt={'deployed_team_change':team_change_receipt(initial,initial)}
    monkeypatch.setattr(validation,'search_receipt',lambda *a:({},c,receipt))
    destination=tmp_path/'runs/validation'
    with pytest.raises(SearchContractError,match='SKIPPED_NO_TEAM_CHANGE'):
        validation.prepare_validation(tmp_path,tmp_path/'prep',run,destination)
    assert not destination.exists()
    final=list(initial);final[2]='new procedure'
    final_path.write_text(json.dumps({'prompts':final}),encoding='utf-8')
    receipt['deployed_team_change']=team_change_receipt(initial,final)
    validation.require_validation_intervention(tmp_path,c,run,receipt)


@pytest.mark.parametrize('split',['shadow','validation','test'])
def test_layer1_rejects_unsupplied_data_before_materialization(tmp_path,split):
    from multi_dataset_diverse_rl.search.layer1_responsibility import ResponsibilityConditionedEngine
    backend=ResponsibilityConditionedOptimizer(evaluator=SyntheticEvaluator(),reflection_lm=SyntheticReflection(),
        accounting_reader=lambda:{},run_root=tmp_path)
    row=SimpleNamespace(source_split=split)
    op=SimpleNamespace(evidence=SimpleNamespace(mutation_evidence=(row,),search_validation_evidence=()))
    with pytest.raises(SearchContractError,match='LAYER1_HELDOUT_ACCESS'):
        ResponsibilityConditionedEngine(backend,81).make_task(op,None)


def test_retry_diversity_observation_preserves_recovery_limit():
    responses=[];events=[]
    def invalid(request):
        responses.append(request);return {**fake_response(request),'text':'unparseable'}
    broker=RequestBroker(contract=contract(),transport=invalid,arm='A1',seed=81,ledger_writer=events.append)
    result=broker.complete(role='solver',split='optimize',stage='initial',member_slot=0,
        messages=[{'role':'user','content':'synthetic'}])
    assert len(responses)==4 and all(r==responses[0] for r in responses)
    assert result['resolved_prediction']['terminal_invalid'] is True
    assert result['retry_diversity']['identical_retry_response_count']==3
    assert result['retry_diversity']['same_invalid_repeated_all_retries'] is True
    cached=broker.complete(role='solver',split='optimize',stage='full',member_slot=0,
        messages=[{'role':'user','content':'synthetic'}])
    assert cached['provider_called'] is False and len(responses)==4
