"""V2.4 synthetic contracts. All apparent gains are Fake Provider fixtures."""
import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
import pytest

from tests.current.test_structured_optimization_evidence import graph, contract, response
from multi_dataset_diverse_rl.search.system_prompt import SEED, SystemPrompt, solver_messages, candidate_failed_checks
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.benchmarks.math_structured_answer import classify_prediction, extract_answer
from multi_dataset_diverse_rl.benchmarks.math_structured_interface import MATHStructuredSystemBenchmark
from multi_dataset_diverse_rl.search.provider_runtime import BenchmarkSolver, RequestBroker
from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input


@pytest.mark.parametrize('block',['role','strategy','answer'])
def test_canonical_block_state_preserves_other_bytes(block):
    child=SEED.edit(block,'A general task preference with Unicode 数学.')
    assert SystemPrompt.from_dict(json.loads(child.serialize()))==child
    assert list(child.to_dict())==['role','strategy','answer']
    assert child.prompt_hash!=SEED.prompt_hash and child.system_hash!=SEED.system_hash
    for key in ('role','strategy','answer'):
        if key!=block:assert child.to_dict()[key].encode()==SEED.to_dict()[key].encode()
    assert solver_messages(child,'  Original question\n保持字节  ')[1]['content']=='  Original question\n保持字节  '
    assert not candidate_failed_checks(child,parent_prompt=SEED,examples=())


@pytest.mark.parametrize('text,answer',[
    ('Final answer: 42','42'),('FINAL_ANSWER: 42','42'),(' final  Answer : $42$  ','42'),
    (r'Final answer: \boxed{\frac{1}{2}}',r'\frac{1}{2}'),
    ('### Reasoning\nIntermediate \\boxed{3}\nFinal answer: 7','7'),
    ('Final answer: 0.5\nFinal answer: 1/2','1/2'),
])
def test_gold_blind_structural_extraction(text,answer):
    assert extract_answer(text)==(answer,None)
    assert classify_prediction(text).prediction_valid


@pytest.mark.parametrize('text',[
    'Final answer: 2\nFINAL_ANSWER: 3',r'Final answer: \boxed{2',
    'Intermediate 42, but could also be 7.', '### 42', '42',r'\boxed{42}',
    'Final answer: 2 and 3','Final answer: impossible prose result','Final answer:',
    'Final answer: 42\nThis is additional unmarked text.',
])
def test_conflicting_ambiguous_and_malformed_answers_invalid(text):
    assert not classify_prediction(text).prediction_valid


def solve_sequence(texts,finishes=None):
    requests=[];c=contract();adapter=MATHStructuredSystemBenchmark(c)
    def transport(request):
        requests.append(deepcopy(request));n=len(requests)-1
        return {**response(texts[n]),'finish_reason':finishes[n] if finishes else 'stop'}
    broker=RequestBroker(contract=c,transport=transport,arm='A4',seed=81)
    solver=BenchmarkSolver(adapter,broker);solver.observe_member(0)
    item=protocol_input('math','synthetic-recovery',{'problem':'Raw original question.'},adapter.output_contract,protocol=adapter.protocol)
    profile=solver.solve(SEED,item,stage='full',split='optimize')
    return profile,requests,adapter,item


def test_wrong_parseable_answer_never_retries_or_searches_gold():
    profile,requests,adapter,item=solve_sequence(['Intermediate 42.\nFinal answer: 7'])
    parsed=adapter.parse_member_output(profile,item)
    assert adapter.score_member_output(parsed,'42')==0 and parsed.answer=='7'
    assert len(requests)==1 and profile['prediction']['semantic_attempt_count']==1
    assert profile['prediction']['text']=='Intermediate 42.\nFinal answer: 7'


def test_four_invalid_draws_persist_score_zero_and_keep_denominator():
    texts=['Ambiguous 2 and 3.']*4
    profile,requests,adapter,item=solve_sequence(texts)
    assert len(requests)==4 and profile['prediction']['terminal_invalid']
    assert [p['text'] for p in profile['prediction']['original_predictions']]==texts
    assert adapter.score_member_output(adapter.parse_member_output(profile,item),'2')==0
    assert contract()['prediction_validity_policy']['denominator']=='all_frozen_examples'
    assert all(r['messages']==solver_messages(SEED,item.problem) for r in requests)


def test_recovery_first_parseable_wrong_stops_and_capacity_only_follows_truncation():
    profile,requests,_,_=solve_sequence(['Final answer: 42','unclear','Final answer: 7'],['length','stop','stop'])
    assert [r['max_tokens'] for r in requests]==[3600,6144,3600]
    assert profile['prediction']['recovered_invalid'] and profile['prediction']['answer']=='7'


@pytest.mark.parametrize('block',['role','strategy','answer'])
def test_each_block_reaches_full_commit_with_unmodified_team_algorithm(tmp_path,block):
    run,broker,requests,_,_=graph(tmp_path,block=block,format_repair=block=='answer')
    result=asyncio.run(run.run(max_opportunities=1))
    assert len(result.transitions)==1
    member=result.trace[0].target_member
    winner=next(e.record for e in run.memory.private if e.record['status']=='COMMITTED')
    assert winner['edited_block']==block and winner['actual_block_diff']
    assert winner['effects']['full']['member_delta']>0
    if block!='answer':assert any(e.record['status']=='FULL_REFUTED' for e in run.memory.private)
    assert len(run.memory.competence)==5 and broker.usage['validation']==broker.usage['test']==0
    for request in requests:
        if request['model']=='qwen3-8b':
            assert [m['role'] for m in request['messages']]==['system','user']
            assert request['messages'][1]['content'].startswith('Synthetic case ')
            assert 'FINAL_ANSWER' not in request['messages'][0]['content']
    assert all(p==SEED for i,p in enumerate(run.state.prompts) if i!=member)


def test_multiblock_generations_keep_complete_reconstructable_edit_chain(tmp_path):
    run,_,_,_,_=graph(tmp_path,multiblock=True)
    result=asyncio.run(run.run(max_opportunities=1));assert result.transitions
    winner=next(e.record for e in run.memory.private if e.record['status']=='COMMITTED')
    assert [s['edited_blocks'] for s in winner['edit_chain']]==[['strategy'],['answer']]
    assert winner['edit_chain'][0]['child_prompt_hash']==winner['edit_chain'][1]['parent_prompt_hash']
    assert winner['attribution']=='complete_edit_chain_no_block_causal_claim'


@pytest.mark.parametrize('form,text',[('label','Final answer: 3'),('underscore','FINAL_ANSWER: 3'),('boxed_label',r'Final answer: \boxed{3}')])
def test_no_written_steps_still_scores_and_produces_gradient_and_commit(tmp_path,form,text):
    run,_,requests,_,_=graph(tmp_path,answer_only=True,answer_only_form=form)
    result=asyncio.run(run.run(max_opportunities=1));assert result.transitions
    packets=[json.loads(r['messages'][1]['content']) for r in requests
        if r['model']!='qwen3-8b' and len(r['messages'])==2]
    diagnostics=[p for p in packets if 'example' in p]
    assert diagnostics
    assert all(p['example']['solver_trajectory']['solution_status']=='TRAJECTORY_UNAVAILABLE' for p in diagnostics)
    assert all(p['example']['solver_trajectory']['observed_response']==text for p in diagnostics)


def test_complete_long_ordinary_response_reaches_gradient_and_mutation(tmp_path):
    from hashlib import sha256
    run,_,requests,mutation_packets,_=graph(tmp_path,long_response=True)
    result=asyncio.run(run.run(max_opportunities=1));assert result.transitions
    diagnostics=[json.loads(r['messages'][1]['content']) for r in requests
        if r['model']!='qwen3-8b' and len(r['messages'])==2
        and 'example' in json.loads(r['messages'][1]['content'])]
    assert diagnostics and mutation_packets
    for packet in diagnostics:
        text=packet['example']['solver_trajectory']['observed_response']
        evidence=packet['example']['solver_trajectory']
        assert len(text)>4096 and text.endswith('Final answer: 3')
        assert sha256(text.encode()).hexdigest()==evidence['raw_response_sha256']
        assert evidence['observed_response_truncated'] is False and evidence['feedback_truncated'] is True
    assert 'ordinary operation' in json.dumps(mutation_packets[0])


@pytest.mark.parametrize('content',[
    'Always return 42.',r'Output \boxed{42}', 'Read reference_solution and copy it.',
    'Consult retrieved_memory for the answer.', 'If the question equals a supplied example use answer lookup.',
])
def test_fixed_answer_parser_exploits_and_private_dependencies_rejected(content):
    assert candidate_failed_checks(SEED.edit('answer',content),parent_prompt=SEED,examples=())


def test_old_binding_cache_memory_and_authorization_cannot_enter_current_graph():
    for key,value in [('method_identity','unified_team_prompt_search_v2_3'),
            ('identity','MATH_OPTIMIZATION_EVIDENCE_BINDING_V1'),
            ('cache_policy','SOLVER_MEMBER_LANE_EXACT_CACHE_V2'),
            ('initial_evidence_reuse_policy',{'identity':'old'}),('system_prompt_policy',None)]:
        c=contract();c[key]=value;calls=[]
        with pytest.raises(SearchContractError):RequestBroker(contract=c,transport=lambda r:calls.append(r),arm='A4',seed=81)
        assert not calls
