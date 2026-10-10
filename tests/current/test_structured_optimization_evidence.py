"""Synthetic provider-wire conformance, never a real-model efficacy test."""
import asyncio
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import re

import pytest

from multi_dataset_diverse_rl import versions
from multi_dataset_diverse_rl.benchmarks.math_structured_binding import derive_structured_contract
from multi_dataset_diverse_rl.benchmarks.math_structured_interface import MATHStructuredSystemBenchmark
from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
from multi_dataset_diverse_rl.search.binary_runtime import BinaryEvidenceSource, CorrectnessExample
from multi_dataset_diverse_rl.search.current_composition import build_current_team_prompt_search
from multi_dataset_diverse_rl.search.current_layer1 import CurrentOptimizer
from multi_dataset_diverse_rl.search.current_policy import CURRENT_POLICY_BUNDLE
from multi_dataset_diverse_rl.search.optimization_evidence import (POLICY, INPUT, GRADIENT_INPUT,
    executability_checks, actual_diff, coverage_effect, rotated_correct)
from multi_dataset_diverse_rl.search.provider_runtime import BenchmarkSolver, ReflectionProvider, RequestBroker
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.search.scientific_aggregation import EquivalencePluralityAggregation
from multi_dataset_diverse_rl.search.textual_gradients import GradientClusterProvider, PerExampleGradientProvider

ROOT=Path(__file__).resolve().parents[2]
PARENT='experiments/execution_bindings/math_v2_2_gradient_pattern_seed81_pilot_v2.json'
from multi_dataset_diverse_rl.search.system_prompt import SEED, SystemPrompt
BASE=SEED
BAD=BASE.edit('strategy','Use case analysis to check signs.')
GOOD=BASE.edit('strategy','Check signs and substitute the result to verify constraints.')
HYPOTHESIS='Check constraints before transforming intermediate expressions.'


def contract():
    return derive_structured_contract(json.loads((ROOT/PARENT).read_bytes()),attempt='synthetic_evidence',
        binding_path='runs/synthetic_evidence_binding.json',parent_path=PARENT,
        parent_sha256=sha256((ROOT/PARENT).read_bytes()).hexdigest(),
        authorization_path='runs/synthetic_evidence_scope.json',authorization_sha256='a'*64,
        gradient_prompt_path='runs/synthetic_evidence_gradient.json',gradient_prompt_sha256='b'*64,
        validation_metadata_path='runs/synthetic_evidence_metadata.json',validation_metadata_sha256='c'*64,
        initial_team_path='runs/v24_team.json',initial_team_artifact_sha256='d'*64,
        pattern_prompt_path='runs/v24_pattern.json',pattern_prompt_sha256='e'*64,
        accounting_policy_path='runs/v24_accounting.json',accounting_policy_sha256='f'*64)


def response(text):
    return dict(text=text,input_tokens=2,output_tokens=2,finish_reason='stop',
        provider_response_accepted=True,provider_metadata_loss_audited=True,
        provider_reasoning_content_present=False,provider_reasoning_character_count=None,
        provider_usage_details={},provider_thinking_indicators=[])


def graph(tmp_path,*,mode='both',uncertain=False,block='strategy',answer_only=False,format_repair=False,multiblock=False,long_response=False,answer_only_form='markdown'):
    bad_prompt=BASE.edit(block,BAD.strategy);good_prompt=BASE.edit(block,GOOD.strategy)
    if multiblock:good_prompt=bad_prompt.edit('answer','Present the final result in a balanced box.')
    c=contract();adapter=MATHStructuredSystemBenchmark(c);requests=[];packets=[];membership={};danger=set();fixed=set()
    def examples(split):
        return tuple(CorrectnessExample(protocol_input('math',f'{split}{i}',
            {'problem':f'Synthetic case {i}: find the result.'},adapter.output_contract,protocol=adapter.protocol),
            '2',f'DATASET_WORKED_SOLUTION_{i}: apply the reference operation.' if split=='optimize' else None)
            for i in range(18 if split=='optimize' else 3))
    def transport(req):
        requests.append(deepcopy(req))
        if req['model']=='gpt-4o-mini':
            text=req['messages'][1]['content'];i=int(re.search(r'case (\d+):',text)[1])
            prompt=req['messages'][0]['content']
            correct=i>=6
            if (prompt.startswith(bad_prompt.render()) or prompt==good_prompt.render()) and i in fixed and not (mode=='neutral_compound' and prompt==bad_prompt.render()):correct=True
            if prompt.startswith(bad_prompt.render()) and i in danger and mode!='neutral_compound':correct=False
            if format_repair and i<6 and prompt==BASE.render():return response('Ambiguous unmarked calculations: 2 and 3.')
            answer=('2' if correct else '3')
            if answer_only and answer_only_form=='underscore':return response('FINAL_ANSWER: '+answer)
            if answer_only and answer_only_form=='boxed_label':return response('Final answer: \\boxed{'+answer+'}')
            if answer_only and answer_only_form=='natural_confirmation':return response('Final answer: '+answer+'\nTherefore, x = '+answer+'.')
            written='' if answer_only else 'ACTUAL_WRITTEN_OPERATION_'+str(i)+'.\n'
            if long_response:written+='A synthetic ordinary operation is written here.\n'*120
            return response(written+'Final answer: '+('\\boxed{'+answer+'}' if format_repair else answer))
        if len(req['messages'])==2:
            p=json.loads(req['messages'][1]['content'])
            if 'example' in p:
                return response(json.dumps(dict(disposition='UNCERTAIN' if uncertain else 'ACTIONABLE',
                    observed_failure='The written operation disagrees with the separately labeled reference.',
                    diagnosis='The observed transformation may omit a constraint.', suggested_block='strategy',
                    reusable_correction=None if uncertain else HYPOTHESIS,
                    expected_effect=None if uncertain else 'Retain constraints in transformed expressions.')))
            return response(json.dumps({'patterns':[{'generalized_gradient':HYPOTHESIS,
                'support_ids':[g['example_id'] for g in p['gradients']]}],'unassigned_ids':[]}))
        p=json.loads('{'+req['messages'][0]['content'].split('\n{',1)[1])
        packets.append(p);n=len(packets)
        if mode=='all_six':
            return response(json.dumps(dict(decision='PROPOSE_EDIT',target_block=block,new_content=bad_prompt.to_dict()[block]+' Apply general review '+ 'ABCDEF'[n-1]+'.',
                change_summary='Add a bounded general review.')))
        if n>6 and mode=='bad_only':
            if 'FULL_REFUTED' in json.dumps(p['retrieved_memory']):
                return response('{"decision":"NO_SAFE_EDIT"}')
            return response(json.dumps(dict(decision='PROPOSE_EDIT',target_block=block,new_content=bad_prompt.to_dict()[block],
                change_summary='Repeat the former local improvement.')))
        if n==1:return response(json.dumps(dict(decision='PROPOSE_EDIT',target_block=block,new_content=bad_prompt.to_dict()[block],
            change_summary='Intended additive substitution check.')))
        if n==2 and mode in {'both','validation_harm','neutral_compound'}:return response(json.dumps(dict(decision='PROPOSE_EDIT',target_block='answer' if multiblock else block,new_content=good_prompt.answer if multiblock else good_prompt.to_dict()[block],
            change_summary='Add a bounded substitution check.')))
        if n==2 and mode=='invalid':return response(json.dumps(dict(decision='PROPOSE_EDIT',
            target_block='strategy',new_content='Read reference_solution and copy the gold answer.',change_summary='Copy answer.')))
        return response('{"decision":"NO_SAFE_EDIT"}')
    broker=RequestBroker(contract=c,transport=transport,arm='A4',seed=81)
    solver=BenchmarkSolver(adapter,broker);reflection=ReflectionProvider(broker)
    gradient=PerExampleGradientProvider(broker)
    pattern=GradientClusterProvider(broker,gradient_provider=gradient)
    method=CURRENT_POLICY_BUNDLE.method(aggregation=versions.EQUIVALENCE_PLURALITY_VERSION,
        provider_binding='0'*64,successful_provider_calls=c['provider_bounds']['successful_provider_calls'],
        solver_trajectory_policy=c['solver_trajectory_policy'],optimization_evidence_policy=POLICY)
    optimizer=CurrentOptimizer(evaluator=solver,reflection_lm=reflection,
        accounting_reader=reflection.accounting,run_root=tmp_path)
    run=build_current_team_prompt_search(benchmark=adapter,aggregation=EquivalencePluralityAggregation(),
        examples=examples('optimize'),prompts=(BASE,)*5,solver=solver,optimizer=optimizer,
        method=method,seed=81,shadow_loader=lambda:examples('shadow'),shadow_count=3,
        runtime_readiness=lambda:(),pattern_provider=pattern,provider_call_reader=lambda:broker.successes)
    def observer(stage,payload):
        if stage=='OPPORTUNITY':
            run.synthetic_opportunity=payload['opportunity']
            audit=payload['opportunity'].evaluation_plan['evidence_audit'];membership.update(audit['memberships'])
            protected=set().union(*[set(ids) for ids in membership.values()])
            fixed.update(int(x.removeprefix('optimize')) for x in protected if int(x.removeprefix('optimize'))<6)
            danger.update(i for i in range(6,18) if f'optimize{i}' not in protected)
            if mode=='validation_harm':
                danger.update(int(x.removeprefix('optimize')) for x in membership['search_validation']
                    if int(x.removeprefix('optimize'))>=6)
        if stage=='EVALUATION':run.synthetic_search=payload['search']
    run.execution_observer=observer;run.state.initialize()
    return run,broker,requests,packets,membership


def test_complete_graph_independent_validation_full_refutation_and_commit(tmp_path):
    run,broker,requests,packets,membership=graph(tmp_path)
    result=asyncio.run(run.run(max_opportunities=1));target=result.trace[0].target_member
    assert result.trace[0].committed_candidate_id and len(result.transitions)==1
    assert all(not set(a)&set(b) for i,a in enumerate(membership.values()) for b in list(membership.values())[:i])
    assert len(packets)==6
    assert len(packets[0]['current_panel_observations'])==3
    wire=json.dumps(packets[0])
    assert 'initial_coverage' in wire and 'SearchValidation' not in wire
    for xid in membership['search_validation']+membership['team_probe']:
        i=xid.removeprefix('optimize')
        assert f'Synthetic case {i}:' not in wire
    assert not any(k in wire for k in ('responsibility_value','target_score','eligible_members','failure_counts'))
    for req in requests:
        if req['model']=='gpt-4o-mini':
            assert not any(x in json.dumps(req) for x in ('DATASET_WORKED','ACTUAL_WRITTEN','edit_effects','reference_solution'))
        elif len(req['messages'])==2:
            p=json.loads(req['messages'][1]['content'])
            if 'example' in p:
                e=p['example'];i=e['example_id'].removeprefix('optimize')
                assert p['schema']==GRADIENT_INPUT and e['reference_solution']['split']=='optimize'
                assert f'DATASET_WORKED_SOLUTION_{i}' in e['reference_solution']['text']
                assert f'ACTUAL_WRITTEN_OPERATION_{i}' in e['solver_trajectory']['visible_solution']
                assert e['solver_trajectory']['source']['member_id']==target
    own=[e.record for e in run.memory.private if e.owner_member==target]
    bad=next(e for e in own if e['child_prompt_id']==sha256(BAD.encode()).hexdigest())
    good=next(e for e in own if e['child_prompt_id']==sha256(GOOD.encode()).hexdigest())
    assert bad['status_history'][-1]=='FULL_REFUTED'
    assert bad['effects']['search_validation']['member_delta']>=0
    assert bad['effects']['full']['member_delta']<=0 and bad['effects']['full']['broken_ids']
    assert 'substitution' in bad['intended_edit'] and 'substitution' not in json.dumps(bad['actual_diff'])
    assert good['status']=='COMMITTED' and good['effects']['full']['member_delta']>0
    c=run.memory.competence[target]
    assert c['prompt_id']==good['child_prompt_id'] and len(c['current_correct_ids'])==run.state.snapshot().member_scores[target]
    view=run.memory.read_for_member(target,'general',pattern_id=bad['pattern_id'],hypothesis=HYPOTHESIS)
    assert 'FULL_REFUTED' in json.dumps(view) and len(json.dumps(view,separators=(',',':')))<1201
    assert 'DATASET_WORKED' not in json.dumps(view) and 'ACTUAL_WRITTEN' not in json.dumps(view)
    assert broker.usage['validation']==broker.usage['test']==0


def test_full_rejection_with_zero_commit_learns_and_candidate_gains_do_not_change_pool(tmp_path):
    run,_,_,packets,_=graph(tmp_path,mode='bad_only')
    result=asyncio.run(run.run(max_opportunities=1));target=result.trace[0].target_member
    assert not result.transitions
    assert len(run.memory.competence[target]['current_correct_ids'])==12
    assert run.memory.private[0].record['status']=='FULL_REFUTED'
    view=run.memory.read_for_member(target,'general',hypothesis=HYPOTHESIS)
    assert 'FULL_REFUTED' in json.dumps(view)
    assert run.memory.read_for_member((target+1)%5,'general')['edit_effects']==[]
    # A second same-member local search receives the actual Full refutation.
    # The fake mutation provider abstains conditionally on that input.
    from multi_dataset_diverse_rl.search.orchestrator import UnifiedSearchContext
    op=replace(run.synthetic_opportunity,opportunity_id=run.synthetic_opportunity.opportunity_id+':next')
    context=UnifiedSearchContext(run.benchmark,run.aggregation,run.history,op.pattern_context,{})
    searched=asyncio.run(run.engine.search(op,context))
    assert searched.candidates==() and len(packets)==12
    assert 'FULL_REFUTED' in json.dumps(packets[6]['retrieved_memory'])


def test_invalid_dependency_and_no_safe_edit_never_evaluate_or_export(tmp_path):
    run,_,_,_,_=graph(tmp_path,mode='invalid')
    result=asyncio.run(run.run(max_opportunities=1))
    assert len(result.trace[0].candidate_ids)==1
    lineage=[json.loads(x) for p in tmp_path.glob('*.lineage.jsonl') for x in p.read_text().splitlines()]
    assert lineage[1]['status']=='CONTRACT_INVALID' and not lineage[1]['solver_evaluated']
    assert 'optimizer_only_dependency' in lineage[1]['failed_checks']
    assert sum(x['status']=='NO_SAFE_EDIT' for x in lineage)==4


def test_edit_memory_retains_existing_private_and_full_failure_capacity(tmp_path):
    run,_,_,_,_=graph(tmp_path,mode='bad_only')
    outcomes=[];prepare=run.memory.prepare_outcome
    def capture(outcome):
        outcomes.append(outcome)
        return prepare(outcome)
    run.memory.prepare_outcome=capture
    asyncio.run(run.run(max_opportunities=1))
    measured=deepcopy(run.memory.private[0].record);member=measured['member']
    for index in range(29):
        record=deepcopy(measured);record['candidate_id']=f'capacity_fixture_{index}'
        if index<21:
            record['status']='LOCALLY_SUPPORTED'
            record['status_history']=['PROPOSED','LOCALLY_SUPPORTED']
            record['effects'].pop('full')
        run.memory.observe_edit(record)
    assert len([e for e in run.memory.private if e.owner_member==member])==24
    outcome=replace(outcomes[-1],evaluated=(),committed=False,complete=False)
    delta=prepare(outcome);run.memory.validate_delta(delta)
    failures=[e for e in delta.private if e.record['status']=='FULL_REFUTED']
    assert len(failures)==5
    assert [e.record['candidate_id'] for e in failures]==[f'capacity_fixture_{i}' for i in range(24,29)]
    assert len([e for e in delta.private if e.owner_member==member])<=24
    with pytest.raises(SearchContractError,match='EDIT_MEMORY_DELTA_INVALID'):
        run.memory.validate_delta(replace(delta,private=(*delta.private,failures[0])))


def test_uncertain_gradient_is_an_accepted_nonactionable_outcome(tmp_path):
    run,_,requests,packets,_=graph(tmp_path,uncertain=True)
    result=asyncio.run(run.run(max_opportunities=1))
    assert not result.transitions and not result.trace[0].candidate_ids and not packets
    assert len([r for r in requests if r['model']!='gpt-4o-mini'])==6
    assert result.trace[0].evidence_audit['cluster_logical_calls']==0


def test_reference_feedback_split_and_id_tampering_stops_before_provider(tmp_path):
    run,broker,requests,_,_=graph(tmp_path)
    asyncio.run(run.run(max_opportunities=1))
    payload=next(json.loads(r['messages'][1]['content']) for r in requests
        if r['model']!='gpt-4o-mini' and len(r['messages'])==2 and 'example' in json.loads(r['messages'][1]['content']))
    provider=PerExampleGradientProvider(broker);before=broker.successes
    for key,value in [('split','validation'),('example_id','another_example')]:
        bad=deepcopy(payload);bad['example']['reference_solution'][key]=value
        with pytest.raises(SearchContractError,match='REFERENCE_SOLUTION_PROVENANCE'):
            provider.extract(bad)
    assert broker.successes==before


def test_evidence_rich_feedback_distinguishes_observable_operations_answer_only_cannot(tmp_path):
    from multi_dataset_diverse_rl.search.textual_gradients import GradientExtractor
    from multi_dataset_diverse_rl.benchmarks.math_structured_answer import classify_prediction
    from multi_dataset_diverse_rl.benchmarks.math_response_evidence import solver_profile,trajectory_policy
    run,_,_,_,_=graph(tmp_path)
    state=run.state.snapshot();diagnosis=run.analyzer.analyze(state,run.history)
    rows=BinaryEvidenceSource(run.state,run.history).for_member(state,diagnosis,0)[:2]
    observations=['Squared the equation and kept every candidate root.',
        'Converted lengths but used the unconverted unit in the area calculation.']
    references=['REFERENCE_DOMAIN: Substitute candidates into the original equation.',
        'REFERENCE_UNITS: Convert both length dimensions before computing area.']
    rich=[]
    for row,steps,reference in zip(rows,observations,references,strict=True):
        profile=solver_profile(dict(request_sha256='d'*64),classify_prediction(steps+'\nFinal answer: 3'),
            member_id=0,prompt=BASE,example_id=row.example_id,split='optimize',policy=trajectory_policy(),
            problem=row.signals['input_payload'])
        rich.append(replace(row,signals={**row.signals,'solver_trajectory':profile['solver_trajectory'],
            'reference_solution':reference}))
    seen=[]
    def fake_diagnostic(request):
        e=json.loads(request['messages'][1]['content'])['example'];seen.append(e)
        correction='Check original constraints after squaring.' if 'Squared' in e['solver_trajectory']['visible_solution'] else 'Convert all dimensions consistently before computing area.'
        return response(json.dumps(dict(disposition='ACTIONABLE',observed_failure='Written transformation omitted a check.',
            diagnosis='The observable operation suggests a conditional check.',suggested_block='strategy',reusable_correction=correction,
            expected_effect='Retain applicable constraints.')))
    broker=RequestBroker(contract=contract(),transport=fake_diagnostic,arm='A4',seed=81)
    results=GradientExtractor(PerExampleGradientProvider(broker)).extract(BASE,rich)
    assert len(results)==2 and results[0]['gradient']!=results[1]['gradient']
    assert [(e['prediction'],e['reference']) for e in seen]==[('3','2'),('3','2')]
    for i,e in enumerate(seen):
        assert observations[1-i] not in json.dumps(e) and references[1-i] not in json.dumps(e)


def test_existing_private_observer_persists_bootstrap_edit_diff_and_full_status(tmp_path):
    from multi_dataset_diverse_rl.governance.pilot_observation import attach_pilot_observer
    run,_,_,_,_=graph(tmp_path)
    attach_pilot_observer(run,tmp_path)
    asyncio.run(run.run(max_opportunities=1))
    rows=[json.loads(x) for x in (tmp_path/'pilot_observation_private.jsonl').read_text().splitlines()]
    boot=next(r for r in rows if r['stage']=='INITIAL_OPTIMIZE_COMPETENCE_BOOTSTRAPPED')
    assert len(boot['memory_state']['competence'])==5 and boot['data']['llm_calls']==0
    final=next(r for r in reversed(rows) if r['stage']=='MEMORY_AFTER_OPPORTUNITY')
    assert any(e['record']['status']=='FULL_REFUTED' for e in final['memory_state']['private'])
    assert any(r['stage']=='MEMORY_READ' for r in rows)


def test_seeded_rotation_uses_current_committed_correct_pool(tmp_path):
    run,_,_,_,_=graph(tmp_path)
    state=run.state.snapshot();diagnosis=run.analyzer.analyze(state,run.history)
    rows=BinaryEvidenceSource(run.state,run.history).for_member(state,diagnosis,0)
    first=[r.example_id for r in rotated_correct(rows,seed=81,member=0,ordinal=0)[:2]]
    second=[r.example_id for r in rotated_correct(rows,seed=81,member=0,ordinal=1)[:2]]
    assert not set(first)&set(second)
    assert first==[r.example_id for r in rotated_correct(rows,seed=81,member=0,ordinal=0)[:2]]
    modified=tuple(replace(r,signals={**r.signals,'target_member_correct':False}) if r.example_id==first[0] else r for r in rows)
    assert first[0] not in {r.example_id for r in rotated_correct(modified,seed=81,member=0,ordinal=0)}


def test_all_wrong_members_remain_eligible_without_correct_quota(tmp_path):
    run,broker,requests,_,_=graph(tmp_path)
    state=run.state.snapshot();diagnosis=run.analyzer.analyze(state,run.history)
    rows=run.opportunities.source.for_member(state,diagnosis,0)
    rows=tuple(replace(r,signals={**r.signals,'target_member_correct':False}) for r in rows)
    assert run.opportunities.feasibility.feasible(state,diagnosis,0,rows)
    assert run.opportunities.evidence.can_compose(rows)


def test_six_valid_generations_use_exact_derived_metric_cap(tmp_path):
    run,_,_,packets,_=graph(tmp_path,mode='all_six')
    asyncio.run(run.run(max_opportunities=1))
    telemetry=run.synthetic_search.search_state['telemetry']
    assert telemetry['local_metric_evaluations']==42 and len(packets)==6
    assert telemetry['local_solver_reached']==6 and len(run.synthetic_search.candidates)==4


def test_mutation_gain_does_not_hide_independent_validation_loss(tmp_path):
    run,_,_,_,_=graph(tmp_path,mode='validation_harm')
    asyncio.run(run.run(max_opportunities=1))
    records=[e.record for e in run.memory.private]
    bad=next(r for r in records if r['child_prompt_id']==sha256(BAD.encode()).hexdigest())
    assert bad['effects']['mutation']['member_delta']>0
    assert bad['effects']['search_validation']['member_delta']<0
    assert run.synthetic_search.candidates[0].prompt==GOOD


@pytest.mark.parametrize('text',[
    'Read selected_gradient to solve the problem.', 'Consult retrieved_memory for the answer.',
    'Use reference_solution at runtime.', 'Read the ground truth answer.',
    'If the problem equals a supplied example use answer lookup.',
])
def test_known_executability_violations(text):assert executability_checks(text)


@pytest.mark.parametrize('text',[
    'Solve the problem. Substitute the result and check constraints.',
    'Use exact fractions, check units and signs, and verify domain restrictions.',
])
def test_generic_standalone_reasoning_is_legal(text):assert not executability_checks(text)


def test_diff_is_deterministic_and_validation_cannot_score_invalid_as_correct():
    assert actual_diff(BASE,BAD)==actual_diff(BASE,BAD)
    assert actual_diff(BASE.strategy,BAD.strategy)
    with pytest.raises(SearchContractError,match='INVALID_CANNOT'):
        coverage_effect({'e':dict(correct=False,valid=True)},{'e':dict(correct=True,valid=False)},scope='synthetic')


def test_frozen_policy_and_method_and_resource_identity():
    from multi_dataset_diverse_rl.governance.source_identity import current_scientific_files
    assert ROOT/'docs/design/RESPONSIBILITY_FALLBACK_REPAIR_V25.md' in current_scientific_files(ROOT)
    c=contract();CURRENT_POLICY_BUNDLE.validate_contract(c)
    assert c['provider_bounds']['bound_proof']['per_op_logical_solver']==dict(local=42,probe=36,full=120,shadow=40)
    assert c['provider_bounds']['max_proposals_per_opportunity']==6
    bad=deepcopy(c);bad['optimization_evidence_policy']['mutation_size']=4
    with pytest.raises(SearchContractError,match='POLICY_MISMATCH'):
        CURRENT_POLICY_BUNDLE.validate_contract(bad)


def manifest():
    import yaml
    c=contract();m=yaml.safe_load((ROOT/'experiments/manifests/math_v2_2_gradient_pattern_seed81_pilot_v2.yaml').read_bytes())
    method=CURRENT_POLICY_BUNDLE.method(aggregation=c['aggregation'],provider_binding='0'*64,
        successful_provider_calls=c['provider_bounds']['successful_provider_calls'],
        solver_trajectory_policy=c['solver_trajectory_policy'],optimization_evidence_policy=POLICY,
        partition_completion_policy=c.get('partition_completion_policy'),
        pattern_cluster_generation_policy=c.get('pattern_cluster_generation_policy'))
    m.update(experiment_id='synthetic_evidence',method_family=method.method,method_identity=method.method,
        lifecycle={'status':'DRAFT','history':[]},search_engine_identity=method.search_engine,
        evidence_identity=method.evidence_policy,memory_identity=method.memory_policy,pattern_identity=method.pattern_policy,feasibility_identity=method.feasibility_policy,
        solver_output_interface_identity='MATH_FLEXIBLE_ANSWER_SYSTEM_INTERFACE_V9',
        mechanism_config=method.mechanism_config,layer1_search_policy=c['layer1_search_policy'],
        optimization_evidence_policy=POLICY,solver_trajectory_policy=c['solver_trajectory_policy'],
        execution_binding={'identity':c['identity'],'path':c['binding_path'],'sha256':'0'*64})
    for key in ('gradient_recovery_policy','post_search_validation_policy'):m.pop(key,None)
    for key in ('system_prompt_policy','answer_extraction_policy','solver_execution_policy','prediction_validity_policy',
            'invalid_recovery_policy','canary_review_policy','parser_identity','accounting_scope_policy','candidate_contract_identity','pattern_abstraction_guard','partition_completion_policy','repair_probe_policy','generated_output_recovery_policy'):
        m[key]=c['payload_parser_identity'] if key=='parser_identity' else c[key]
    m['cache_policy']=dict(identity=c['cache_policy'],frozen=True)
    m['models']=dict(solver=c['models']['solver'],optimizer=c['models']['optimizer_reflection'],solver_thinking=False)
    m['provider_policy']=dict(identity=c['provider'],frozen=True)
    m['provider_routing_policy']=c['provider_routing_policy']
    m['solver_decoding_policy']=c['solver_decoding_policy']
    return m


def test_new_manifest_is_strict_and_unfrozen_real_profile_holds():
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    from multi_dataset_diverse_rl.governance.unified_execution import execution_scope
    from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
    m=manifest();assert not validate_manifest_v2(ROOT,m)
    scope=execution_scope(m,contract());assert scope['initial_memory_entries']==5
    assert scope['optimization_evidence_policy']==POLICY
    for key in ('optimization_evidence_policy','solver_trajectory_policy','layer1_search_policy','generated_output_recovery_policy'):
        bad=deepcopy(m);bad.pop(key);assert validate_manifest_v2(ROOT,bad)
    bad=deepcopy(m);bad['execution_binding']['identity']='MATH_V2_2_EXECUTION_BINDING_V1'
    assert validate_manifest_v2(ROOT,bad)
    with pytest.raises(SearchContractError,match='CURRENT_V24_EXECUTION_BINDING_NOT_FROZEN'):
        execution_binding(ROOT,{'identity':'MATH_OPTIMIZATION_EVIDENCE_BINDING_V1'})
