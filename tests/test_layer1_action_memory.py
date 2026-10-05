"""Guarded synthetic proofs for bounded feedback, privacy and target-first access."""
import asyncio
from dataclasses import asdict, replace
import json
from types import SimpleNamespace as NS

import pytest

from multi_dataset_diverse_rl.search.action_memory import StructuredActionMemoryV3, LIMITS, edit_action
from multi_dataset_diverse_rl.search.layer1_memory import (MemoryConditionedOptimizer, MemoryConditionedEngine,
    MemoryLocalTask, anchored_panel, bounded_input, PreservationAnchorEvidenceV3)
from multi_dataset_diverse_rl.search.memory import OpportunityOutcome
from multi_dataset_diverse_rl.search.schemas import SearchCandidate, EvaluatedCandidate, SearchContractError
from multi_dataset_diverse_rl.search.runtime_v2 import V2OpportunityBuilder
from multi_dataset_diverse_rl.search.variable_evidence import VariableEvidenceFeasibilityV1
from multi_dataset_diverse_rl.search.policies import TargetPolicyV1
from multi_dataset_diverse_rl.search.history import HistoryState
from multi_dataset_diverse_rl.local_optimizers.schemas import LocalEvidenceExample, LocalOptimizerBudget
from multi_dataset_diverse_rl.local_optimizers.base import LocalSolverObservation
from test_unified_search_v2 import Store, Analyzer, Source, rows

ROOT='Inspect constraints.'


def memory():return StructuredActionMemoryV3(**LIMITS)


def task(n=5,anchor=True):
    panel=tuple(LocalEvidenceExample(str(i),'Synthetic algebra question '+str(i),'2',tags=('repair',) if i<3 else ('preservation','target_correct')) for i in range(n))
    return MemoryLocalTask('memory_test',ROOT,panel[:3],panel[3:], 'direct_flip',
        'COMMON_SOLVER_CONTRACT_V1','synthetic',81,LocalOptimizerBudget(36,3,4),target_member=1,
        anchor_example=panel[-1] if anchor else None,responsibility_lane='direct_flip')


def details(delta=-1,valid=True):
    p=ROOT+' Check signs.'
    return dict(changed=True,contract_valid=valid,solver_evaluated=valid,local_parent_correct_delta=delta,
        local_parent_newly_fixed=0,local_parent_newly_broken=1,local_newly_broken=1,
        preservation_locally_measurable=True,memory_action=edit_action(ROOT,p))


def write(m,*,member=1,lane='direct_flip',delta=-1,valid=True,index=0):
    return m.observe_local_failure(member=member,lane=lane,parent=ROOT,prompt=ROOT+' Check signs.',
        details=details(delta,valid),opportunity_id='synthetic_op_'+str(index))


class Evaluator:
    solver_contract_id='COMMON_SOLVER_CONTRACT_V1';output_contract_id='synthetic'
    def __init__(self,correct=True):self.calls=[];self.correct=correct
    def evaluate(self,prompt,row):
        self.calls.append((prompt,row.example_id))
        correct=prompt==ROOT and self.correct
        return LocalSolverObservation('2' if correct else '3','synthetic',correct,True,provider_called=True)


class Reflection:
    def __init__(self,summary='valid'):self.inputs=[];self.summary=summary
    def __call__(self,text):
        self.inputs.append(text)
        d=dict(decision_procedure=ROOT+f' Check signs with {len(self.inputs)} independent verifications.')
        if self.summary!='absent':d['change_summary']=('Add sign checks.' if self.summary=='valid' else ['malformed'])
        return json.dumps(d)
    def accounting(self):return dict(successful_calls=len(self.inputs),input_tokens=len(self.inputs)*10,output_tokens=len(self.inputs)*4)


def run(tmp_path,n=5,anchor=True,correct=True,summary='valid'):
    e=Evaluator(correct);r=Reflection(summary);b=MemoryConditionedOptimizer(evaluator=e,reflection_lm=r,accounting_reader=r.accounting,run_root=tmp_path)
    b.memory=memory();result=asyncio.run(b.search_task(task(n,anchor),'synthetic_op'))
    return e,r,b,result


@pytest.mark.parametrize('n,candidates,calls',[(5,6,35),(6,5,36)])
def test_dynamic_panel_capacity_and_same_panel(tmp_path,n,candidates,calls):
    e,r,b,result=run(tmp_path,n)
    assert len(e.calls)==calls and len(r.inputs)==candidates
    assert result.search_state['telemetry']['panel_parent_correct_anchor_count']==1
    assert result.search_state['telemetry']['preservation_locally_measurable']
    assert len(result.candidates)==4
    assert all(c.backend_details['LOCAL_REJECTED_EXPORTED'] for c in result.candidates)
    assert len({tuple(e.calls[i+j][1] for j in range(n)) for i in range(0,calls,n)})==1
    assert {xid for _,xid in e.calls}=={str(j) for j in range(n)}


def test_outside_panel_anchor_is_inserted_without_filling_quota():
    t=task(5);external=LocalEvidenceExample('anchor','Synthetic preserved problem','2',tags=('preservation','target_correct'))
    short=replace(t,search_examples=t.search_examples[:2],local_validation_examples=t.local_validation_examples[:1],anchor_example=external)
    assert len(anchored_panel(short))==4 and external in anchored_panel(short)
    full=replace(task(6),anchor_example=external)
    assert len(anchored_panel(full))==6 and external in anchored_panel(full)


def test_no_anchor_no_root_correct_explicitly_unmeasurable(tmp_path):
    *_,result=run(tmp_path,anchor=False,correct=False)
    t=result.search_state['telemetry']
    assert not t['panel_anchor_available'] and not t['preservation_locally_measurable']
    assert all(c.backend_details['local_preservation_loss'] is None for c in result.candidates)


@pytest.mark.parametrize('delta,valid,expected',[(-1,True,True),(0,True,False),(1,True,False),(-1,False,False)])
def test_only_evaluated_negative_writes_semantic_failure(delta,valid,expected):
    m=memory();assert write(m,delta=delta,valid=valid)==expected
    assert len(m.failures)==int(expected)


def test_queue_five_sixth_deterministic_eviction():
    a,b=memory(),memory()
    for m in (a,b):
        for i in range(6):write(m,lane='coverage' if i==0 else 'direct_flip',index=i)
    assert a.failures==b.failures and len(a.failures)==5
    assert not any(e.lane=='coverage' for e in a.failures)
    assert a.audit()['failure_memory_peak_by_member']==[0,5,0,0,0]
    assert a.audit()['failure_memory_evictions']==1


def test_private_failure_cross_opportunity_and_attempt_isolation():
    canary=memory();write(canary)
    assert canary.read_for_member(1,'direct_flip')
    assert not canary.read_for_member(3,'direct_flip')
    pilot=memory();assert not pilot.read_for_member(1,'direct_flip')


def test_intra_opportunity_action_outcome_and_bounded_input_growth(tmp_path):
    e,r,b,result=run(tmp_path)
    body=[json.loads(s.rsplit('\n',1)[1]) for s in r.inputs]
    assert body[0]['retrieved_memory']=={}
    assert body[1]['retrieved_memory']['private_failure'][0]['action']
    assert 'delta -5' in body[1]['retrieved_memory']['private_failure'][0]['outcome']
    assert all(x['current_panel_observations'] and x['current_parent']==ROOT and x['repair_objective'] for x in body)
    for text in r.inputs:
        assert all(k not in text for k in ('candidate_archive','candidate_id','prompt_hash','target_member','eligible_members','evidence_packet_hash','generation'))
    assert all(x['memory_chars']<=1200 for x in result.search_state['telemetry']['input_by_generation'])
    lengths=[]
    for i in range(12):
        write(b.memory,index=i)
        view=b.memory.read_for_member(1,'direct_flip')
        lengths.append(len(bounded_input(task(),ROOT,body[0]['current_panel_observations'],view)))
    assert max(lengths[-6:])-min(lengths[-6:])==0
    assert b.memory.audit()['failure_memory_peak_by_member'][1]==5


@pytest.mark.parametrize('summary',['absent','malformed'])
def test_bad_summary_does_not_kill_valid_candidate_or_enter_solver(tmp_path,summary):
    e,r,b,result=run(tmp_path,summary=summary)
    assert result.search_state['telemetry']['local_solver_reached']==6
    assert all('change_summary' not in prompt for prompt,_ in e.calls)
    assert b.memory.audit()['failure_writes']==6


def make_op(values=None):
    state=Store().snapshot();h=HistoryState();d=Analyzer().analyze(state,h)
    builder=V2OpportunityBuilder(source=Source(values or rows(7)),feasibility=VariableEvidenceFeasibilityV1(),
        target=TargetPolicyV1(),evidence=PreservationAnchorEvidenceV3())
    return state,d,builder,h,builder.build(state=state,diagnosis=d,history=h,update_index=0)


def test_memory_cannot_change_who_but_changes_reflection_context():
    state,d,builder,h,op=make_op();m=memory()
    initial=asdict(op)
    write(m,member=op.target_member)
    again=builder.build(state=state,diagnosis=d,history=h,update_index=0)
    assert asdict(again)==initial # DNC, feasibility, scores, eligible members and WHO all identical.
    before=bounded_input(task(),ROOT,[],{})
    after=bounded_input(task(),ROOT,[],m.read_for_opportunity(op))
    assert before!=after


def test_engine_prefers_mutation_sensitive_legal_anchor(tmp_path):
    vals=list(rows(7,repairs=3))
    vals[4]=replace(vals[4],roles=vals[4].roles|{'target_correct'},signals={**vals[4].signals,'mutation_sensitive':True})
    vals[5]=replace(vals[5],roles=vals[5].roles|{'target_correct'},signals={**vals[5].signals,'team_disagreement':5})
    *_,op=make_op(tuple(vals))
    b=MemoryConditionedOptimizer(evaluator=Evaluator(),reflection_lm=Reflection(),accounting_reader=lambda:{},run_root=tmp_path)
    task_actual=MemoryConditionedEngine(b,81).make_task(op,NS(pattern_view={},memory_view={}))
    assert task_actual.anchor_example.example_id=='e4'
    assert task_actual.anchor_example in anchored_panel(task_actual)


def outcome(op,committed=False,risk=None,shadow=False):
    c=SearchCandidate('synthetic',ROOT+' Check signs and units.')
    evaluated=EvaluatedCandidate(c,None,None,False,False,{'scientific_risk_code':risk} if risk else {})
    return OpportunityOutcome(op,(evaluated,),'synthetic',committed,False if shadow else True,0)


def test_success_only_committed_member_private_and_shared_risk_no_strategy():
    *_,op=make_op();m=memory()
    delta=m.prepare_outcome(outcome(op));m.apply_outcome(delta);assert not m.private
    delta=m.prepare_outcome(outcome(op,committed=True));m.apply_outcome(delta)
    assert m.read_for_member(op.target_member,'direct_flip')['private_success'][0]['action']
    assert not m.read_for_member(1,'direct_flip')
    delta=m.prepare_outcome(outcome(op,shadow=True));m.apply_outcome(delta)
    other=m.read_for_member(1,'direct_flip')
    assert 'shared_risk' in other and 'signs' not in json.dumps(other) and 'units' not in json.dumps(other)
    assert not any(k in json.dumps(other) for k in ('source_candidate','memory_id','owner_member','Synthetic algebra','prediction','reference'))


@pytest.mark.parametrize('split',['validation','test','shadow'])
def test_heldout_never_read_or_write_memory(split):
    *_,op=make_op()
    # Bypass the earlier EvidenceView firewall to test the memory port itself.
    bad=replace(op,evidence=NS(mutation_evidence=tuple(replace(r,source_split=split) for r in op.evidence.mutation_evidence),
        search_validation_evidence=op.evidence.search_validation_evidence))
    m=memory()
    with pytest.raises(SearchContractError,match='HELDOUT'):m.read_for_opportunity(bad)
    with pytest.raises(SearchContractError,match='HELDOUT'):m.prepare_outcome(outcome(bad))
    assert not m.private and not m.shared and not m.failures


def test_memory_on_governed_binding_full_fake_production_graph(tmp_path,monkeypatch):
    from pathlib import Path
    from multi_dataset_diverse_rl import versions
    from multi_dataset_diverse_rl.benchmarks.math_memory_binding import MATHMemoryBinding
    from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
    from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample
    from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker,BenchmarkSolver,ReflectionProvider
    from multi_dataset_diverse_rl.governance.token_accounting import serialized_request
    from multi_dataset_diverse_rl.governance.legacy.unified_execution import preexecution_manifest,bound_preflight,execution_scope
    root=Path(__file__).resolve().parents[1]
    binding_path='experiments/execution_bindings/math_v2_1_memory_canary_v1.json'
    c=json.loads((root/binding_path).read_bytes());binding=MATHMemoryBinding(root,c)
    assert not binding.blockers()
    m=preexecution_manifest(root,source_sha='0'*40,binding_path=binding_path,experiment_id='synthetic_memory',frozen=False)
    assert bound_preflight(root,m)['blockers']==['PREEXECUTION_NOT_FROZEN']
    scope=execution_scope(m,c)
    assert scope['arm']=='A3' and scope['memory_policy_identity']==versions.STRUCTURED_ACTION_MEMORY_VERSION
    assert scope['initial_memory_entries']==0 and scope['validation_calls']==scope['test_calls']==0
    adapter=binding.benchmark()
    prompts=tuple(x['prompt'] for x in json.loads((root/c['initial_team_path']).read_bytes())['members'])
    def examples(role):
        return tuple(CorrectnessExample(protocol_input('math',f'{role}{i}',{'problem':f'Synthetic {role} arithmetic {i}.'},
            adapter.output_contract,protocol=adapter.protocol),'1') for i in range(12 if role=='optimize' else 40))
    monkeypatch.setattr(binding,'examples',examples)
    inputs=[];solver_texts=[]
    def transport(req):
        body=json.loads(serialized_request(req));assert body['enable_thinking'] is False
        if body['model']=='qwen3-8b':
            assert body['temperature']==.2 and body['max_tokens']==3600
            text=req['messages'][1]['content'];solver_texts.append(text)
            prompt,problem=text.split('\n\n',1)
            from multi_dataset_diverse_rl.benchmarks.math_v21_interface import MATH_SOLVER_INTERFACE_V4_USER_SUFFIX
            problem=problem.removesuffix(MATH_SOLVER_INTERFACE_V4_USER_SUFFIX)
            i=int(problem.rsplit(' ',1)[1].rstrip('.'))
            correct=(prompts.index(prompt) in {1,2} or prompts.index(prompt)==0 and i>=3) if prompt in prompts else i>=4
            text='FINAL_ANSWER: '+('1' if correct else '2')
        else:
            assert body['temperature']==.7 and body['max_completion_tokens']==1800 and 'max_tokens' not in body
            assert body['presence_penalty']==1.5 and body['top_k']==20
            data=json.loads(req['messages'][0]['content'].rsplit('\n',1)[1]);inputs.append(data)
            text=json.dumps(dict(decision_procedure=f'Inspect constraints and check signs with {len(inputs)} independent verifications.',change_summary='Add sign checks.'))
        return dict(text=text,input_tokens=2,output_tokens=2,finish_reason='stop',provider_response_accepted=True,
            provider_metadata_loss_audited=True,provider_reasoning_content_present=False,
            provider_reasoning_character_count=None,provider_usage_details={},provider_thinking_indicators=[])
    broker=RequestBroker(contract=c,transport=transport,arm='A3',seed=81)
    run=binding.compose(arm='A3',seed=81,solver=BenchmarkSolver(adapter,broker),reflection=ReflectionProvider(broker),pattern_provider=None,run_root=tmp_path)
    assert run.memory.audit()['stateful_write_count']==0
    run.state.initialize();initial=run.state.snapshot();result=asyncio.run(run.run(max_opportunities=1))
    assert result.trace and result.trace[0].candidate_ids and run.evaluation.provider.probed
    assert result.stop_reason=='CANARY_ONE_PRODUCTION_OPPORTUNITY_COMPLETE'
    assert run.method.memory_policy==versions.STRUCTURED_ACTION_MEMORY_VERSION
    assert run.method.identity()==binding.method('A3').identity()
    assert run.state.initial_member_scores==initial.member_scores
    assert broker.usage['pattern']==broker.usage['test']==broker.usage['validation']==0
    assert all(x['schema']==versions.LAYER1_INPUT_SCHEMA_VERSION and 'candidate_archive' not in x for x in inputs)
    assert all('retrieved_memory' not in x and 'change_summary' not in x for x in solver_texts)
    assert run.memory.audit()['memory_context_chars']['max']<=1200
    assert len(run.memory.failures)<=5


def test_summary_envelope_diagnostics_preserve_legacy_boundary():
    from multi_dataset_diverse_rl import versions
    from multi_dataset_diverse_rl.benchmarks.math_optimizer_diagnostics import generation_diagnostics
    text=json.dumps(dict(decision_procedure='Check signs and units.',change_summary=['bad metadata']))
    old=generation_diagnostics(text,versions.SEMANTIC_MUTABLE_CONTRACT_VERSION)
    new=generation_diagnostics(text,versions.SEMANTIC_MUTABLE_CONTRACT_VERSION,versions.LAYER1_INPUT_SCHEMA_VERSION)
    assert not old['candidate_generation_contract_valid'] and new['candidate_generation_contract_valid']
    assert new['candidate_envelope']=='json_procedure_optional_summary_v2'


def test_new_limits_fail_closed_and_shared_storage_remains_bounded():
    with pytest.raises(SearchContractError,match='LIMITS'):
        StructuredActionMemoryV3(**{**LIMITS,'max_context_chars':1201})
    *_,op=make_op();m=memory()
    for _ in range(55):m.apply_outcome(m.prepare_outcome(outcome(op,shadow=True)))
    assert len(m.shared)==48
    view=m.read_for_member(3,'direct_flip')
    assert len(json.dumps(view,sort_keys=True,separators=(',',':')))<=1200


def test_context_size_synthetic_receipt(tmp_path):
    import os
    from pathlib import Path
    *_,backend,result=run(tmp_path)
    destination=os.environ.get('FORMAL_V3_EVIDENCE_CAPTURE_DIR')
    if destination:
        out=Path(destination);out.mkdir(parents=True,exist_ok=True)
        (out/'memory_input_growth.json').write_bytes((json.dumps(result.search_state['telemetry'],indent=2)+'\n').encode())
