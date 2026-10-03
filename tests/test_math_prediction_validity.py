"""Prediction/reference asymmetry and invalid-vote negative controls."""
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import pytest

from multi_dataset_diverse_rl.benchmarks.math_prediction_validity import (
    classify_prediction, prediction_from_persisted, prediction_validity_contract)
from multi_dataset_diverse_rl.benchmarks.math_v21_binding import MATHV21Binding
from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
from multi_dataset_diverse_rl.search.provider_runtime import BenchmarkSolver, RequestBroker
from multi_dataset_diverse_rl.search.scientific_aggregation import EquivalencePluralityAggregation
from multi_dataset_diverse_rl.search.binary_responsibility import observation_from_outputs
from multi_dataset_diverse_rl.search.binary_runtime import BinaryTeamStateStore, CorrectnessExample
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.local_optimizers.schemas import LocalEvidenceExample

ROOT=Path(__file__).resolve().parents[1]
def ports(transport):
    c=json.loads((ROOT/'experiments/execution_bindings/math_v2_1_canary_v6.json').read_bytes())
    benchmark=MATHV21Binding(ROOT,c).benchmark()
    broker=RequestBroker(contract=c,transport=transport,arm='A1',seed=81)
    item=protocol_input('math','synthetic',{'problem':'Synthetic arithmetic.'},benchmark.output_contract,protocol=benchmark.protocol)
    return benchmark,broker,BenchmarkSolver(benchmark,broker),item

@pytest.mark.parametrize('text,finish,reason',[
    (None,'stop','MISSING_FINAL_MARKER'),
    ('Reasoning with a boxed answer \\boxed{1}.','stop','MISSING_FINAL_MARKER'),
    ('FINAL_ANSWER:','stop','EMPTY_FINAL_PAYLOAD'),
    ('FINAL_ANSWER: 1\nFINAL_ANSWER: 1','stop','MULTIPLE_FINAL_MARKERS'),
    ('FINAL_ANSWER: 1','length','OUTPUT_TRUNCATED'),
    ('Reasoning only','length','OUTPUT_TRUNCATED'),
    ('FINAL_ANSWER: 1\ntrailing commentary','stop','OTHER_PREDICTION_CONTRACT_FAILURE'),
    ('FINAL_ANSWER:','content_filter','OTHER_PREDICTION_CONTRACT_FAILURE'),
    ('FINAL_ANSWER: \\frac{','stop','PAYLOAD_PARSE_FAILURE'),
    ('FINAL_ANSWER: (1,2)','stop','PAYLOAD_UNSUPPORTED')])
def test_successful_transport_invalid_prediction_is_wrong_and_continues(text,finish,reason):
    calls=[]
    b,broker,solver,item=ports(lambda r:calls.append(r) or dict(text=text,finish_reason=finish,input_tokens=2,output_tokens=2))
    p=solver.solve('Synthetic procedure.',item,stage='initial',split='optimize')
    assert not p.prediction_valid and p.invalid_reason==reason
    assert b.score_member_output(b.parse_member_output(p,item),'1')==0
    assert prediction_from_persisted(json.loads(json.dumps(asdict(p))))==p
    again=solver.solve('Synthetic procedure.',item,stage='full',split='optimize')
    assert again==p and len(calls)==1 and broker.successes==1
    local=solver.evaluate('Other synthetic procedure.',LocalEvidenceExample('synthetic',item.problem,'1'))
    assert not local.valid and not local.correct and len(calls)==2

def test_reference_failure_remains_hard_even_when_prediction_invalid():
    b,_,_,item=ports(lambda _:None)
    p=classify_prediction('missing')
    for reference in ('',r'\frac{','(1,2)'):
        with pytest.raises(SearchContractError,match='REFERENCE_UNSCORABLE'):
            b.score_member_output(b.parse_member_output(p,item),reference)

def test_reference_timeout_hard_but_prediction_parser_timeout_wrong(monkeypatch):
    from multi_dataset_diverse_rl.benchmarks import math_prediction_validity as prediction, math_domain_v2 as reference
    def timeout(*args,**kwargs):raise subprocess.TimeoutExpired('synthetic',8)
    prediction.classify_payload.cache_clear();reference.domain_matrix.cache_clear()
    monkeypatch.setattr(subprocess,'run',timeout)
    assert prediction.classify_prediction('FINAL_ANSWER: synthetic-timeout').invalid_reason=='PAYLOAD_PARSE_FAILURE'
    with pytest.raises(SearchContractError,match='REFERENCE_UNSCORABLE'):reference.require_scorable('synthetic-timeout')
    prediction.classify_payload.cache_clear();reference.domain_matrix.cache_clear()

def test_systemic_parser_failure_and_provider_failure_are_hard(monkeypatch):
    from multi_dataset_diverse_rl.benchmarks import math_prediction_validity as prediction
    prediction.classify_payload.cache_clear()
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:subprocess.CompletedProcess([],1,'{}',''))
    with pytest.raises(SearchContractError,match='MATH_PREDICTION_WORKER_FAILURE'):prediction.classify_prediction('FINAL_ANSWER: runtime-defect')
    prediction.classify_payload.cache_clear()
    def fail(_):raise ValueError('synthetic serialization failure')
    _,_,solver,item=ports(fail)
    with pytest.raises(ValueError,match='serialization'):solver.solve('Synthetic procedure.',item,stage='initial',split='optimize')

def test_invalid_answers_never_form_a_plurality_or_oracle_class():
    b,_,_,item=ports(lambda _:None);aggregation=EquivalencePluralityAggregation()
    invalid=classify_prediction('FINAL_ANSWER: 1','length')
    wrong=classify_prediction('FINAL_ANSWER: 2');correct=classify_prediction('FINAL_ANSWER: 1')
    for outputs,vote,oracle in [((invalid,)*5,0,False),((invalid,invalid,wrong,correct,correct),1,True),
            ((invalid,invalid,invalid,wrong,correct),0,True)]:
        result=aggregation.aggregate_sync(item=item,member_outputs=outputs,benchmark=b)
        obs=observation_from_outputs(benchmark=b,item=item,member_outputs=outputs,gold='1')
        assert b.score_member_output(result.parsed_output,'1')==vote
        assert any(obs.member_success)==oracle
        assert all(not obs.vote_classes[i] for i,p in enumerate(outputs) if not p.prediction_valid)
        assert all(all(outputs[i].prediction_valid for i in group) for group in result.diagnostics['equivalence_classes'])

def test_initial_floor_binds_all_rows_and_candidate_invalidity_is_just_wrong():
    from multi_dataset_diverse_rl.search.binary_runtime import FixedPeerTeamEvaluationProvider
    from multi_dataset_diverse_rl.search.semantic_contract import InitialCompetenceTransitionV2
    from types import SimpleNamespace
    def transport(r):
        prompt=r['messages'][1]['content'].split('\n\n')[0]
        return dict(text='FINAL_ANSWER: 1',finish_reason='length' if prompt in {'member4','candidate'} else 'stop',input_tokens=1,output_tokens=1)
    b,_,solver,item=ports(transport)
    store=BinaryTeamStateStore(benchmark=b,examples=[CorrectnessExample(item,'1')],prompts=tuple('member'+str(i) for i in range(5)),
        solver=solver,aggregation=EquivalencePluralityAggregation(),freeze_initial_competence=True)
    store.initialize();initial=store.snapshot()
    assert initial.member_scores==(1,1,1,1,0) and store.initial_member_scores==initial.member_scores
    provider=FixedPeerTeamEvaluationProvider(store)
    opportunity=SimpleNamespace(parent_state_id=initial.team_state_id,target_member=0)
    _,states=provider._evaluate(opportunity,SimpleNamespace(prompt='candidate'),{item.input_id},'full')
    assert not states[0].team_validity[0] and not states[0].team_correctness[0]
    assert provider.evaluation(states).member_scores[0]==0
    assert store.initial_member_scores==initial.member_scores

def test_prediction_policy_is_exact_and_rejects_legacy_injection():
    from multi_dataset_diverse_rl.benchmarks.math_prediction_validity import frozen_prediction_policy
    from multi_dataset_diverse_rl.governance.unified_execution import execution_scope
    from multi_dataset_diverse_rl.search.schemas import TeamEvaluation
    from multi_dataset_diverse_rl.search.semantic_contract import InitialCompetenceTransitionV2
    b,broker,_,_=ports(lambda _:None);c=broker.contract
    assert frozen_prediction_policy(c)==prediction_validity_contract()
    legacy={**c,'identity':'MATH_V2_1_EXECUTION_BINDING_V2'}
    with pytest.raises(SearchContractError,match='REQUIRES_FRESH_BINDING'):frozen_prediction_policy(legacy)
    c['prediction_validity_policy']['fallback_extraction']=True
    with pytest.raises(SearchContractError,match='BINDING_MISMATCH'):frozen_prediction_policy(c)
    transition=InitialCompetenceTransitionV2(invalid_predictions_are_incorrect=True)
    transition.bind_initial((0,)*5,'synthetic')
    parent=TeamEvaluation(1,None,(1,)*5)
    child=TeamEvaluation(2,None,(0,1,1,1,1),aggregation_diagnostics={'terminal_invalid_delta':1})
    assert transition.allows(parent,child,0)

def test_cache_and_persisted_validity_corruption_remain_hard():
    _,broker,solver,item=ports(lambda _:dict(text='FINAL_ANSWER: 1',finish_reason='length',input_tokens=1,output_tokens=1))
    p=solver.solve('Synthetic procedure.',item,stage='initial',split='optimize')
    altered=asdict(p);altered['prediction_valid']=True
    with pytest.raises(SearchContractError,match='STATE_CORRUPTION'):prediction_from_persisted(altered)
    altered['prediction_valid']=0
    with pytest.raises(SearchContractError,match='STATE_CORRUPTION'):prediction_from_persisted(altered)
    broker.cache[next(iter(broker.cache))]['finish_reason']='stop'
    with pytest.raises(SearchContractError,match='CACHE_CORRUPTION'):
        solver.solve('Synthetic procedure.',item,stage='full',split='optimize')

def test_invalidity_feedback_is_incorrect_without_format_rewrite_signal():
    from multi_dataset_diverse_rl.search.gepa_v2 import V2ReflectionAdapter
    from multi_dataset_diverse_rl.local_optimizers.gepa_adapter import LocalGEPATrajectory
    from multi_dataset_diverse_rl.local_optimizers.base import LocalSolverObservation
    from gepa.core.adapter import EvaluationBatch
    _,_,solver,_=ports(lambda _:None)
    prompt='Check constraints before deriving the result.'
    example=LocalEvidenceExample('synthetic','Synthetic arithmetic.','1')
    adapter=V2ReflectionAdapter(solver,parent_prompt=prompt,all_examples=[example],optimization_context='',output_contract_id=solver.output_contract_id)
    observation=LocalSolverObservation(None,'FINAL_ANSWER:',False,False,failure_reason='EMPTY_FINAL_PAYLOAD')
    batch=EvaluationBatch(outputs=[],scores=[],trajectories=[LocalGEPATrajectory(example,observation)])
    rows=adapter.make_reflective_dataset({'decision_procedure':prompt},batch,['decision_procedure'])['decision_procedure']
    assert rows[0]['Evaluation Outcome']=='incorrect'
    assert 'FINAL_ANSWER' not in rows[0]['Reasoning Evidence'] and 'EMPTY_FINAL_PAYLOAD' not in str(rows)

def test_policy_numeric_bool_aliases_are_rejected():
    from multi_dataset_diverse_rl.benchmarks.math_prediction_validity import frozen_prediction_policy
    _,broker,_,_=ports(lambda _:None)
    for field,value in [('fallback_extraction',0),('invalid_response_retries',False)]:
        c={**broker.contract,'prediction_validity_policy':{**broker.contract['prediction_validity_policy'],field:value}}
        with pytest.raises(SearchContractError,match='BINDING_MISMATCH'):frozen_prediction_policy(c)
