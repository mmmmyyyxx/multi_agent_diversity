"""Zero-API semantic promotion, bounded retention and information-flow proofs."""
from dataclasses import asdict, replace
import asyncio
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace as NS
import copy
import json

import pytest

from multi_dataset_diverse_rl import versions
from multi_dataset_diverse_rl.search.action_memory import LIMITS, edit_action, StructuredActionMemoryV3
from multi_dataset_diverse_rl.search.rolling_risk_memory import (POLICY, FailureSignature,
    StructuredRollingRiskMemoryV4, failure_signature)
from multi_dataset_diverse_rl.search.memory import OpportunityOutcome
from multi_dataset_diverse_rl.search.schemas import SearchCandidate, EvaluatedCandidate, SearchContractError
from test_layer1_action_memory import make_op, task, Evaluator, Reflection

ROOT='Inspect constraints.'


def memory():return StructuredRollingRiskMemoryV4(**LIMITS,risk_policy=copy.deepcopy(POLICY))


def details(prompt=ROOT+' Check signs.',fixed=0,broken=1,preservation=True):
    return dict(changed=True,contract_valid=True,solver_evaluated=True,
        local_invalid_count=0,
        local_parent_correct_delta=fixed-broken,local_parent_newly_fixed=fixed,
        local_parent_newly_broken=broken,local_newly_broken=broken if preservation else 0,
        preservation_locally_measurable=preservation,memory_action=edit_action(ROOT,prompt))


def observe(m,member,prompt=ROOT+' Check signs.',lane='direct_flip',**kw):
    return m.observe_local_failure(member=member,lane=lane,parent=ROOT,prompt=prompt,
        details=details(prompt,**kw),opportunity_id='synthetic_'+str(m.sequence))


def op(member=1,lane='direct_flip',split='optimize'):
    return NS(target_member=member,parent_prompt=ROOT,opportunity_id='synthetic_op_'+str(member),
        evidence=NS(mutation_evidence=(NS(source_split=split),),search_validation_evidence=()),
        diagnosis=NS(responsibility={member:NS(primary_lane=lane)}))


def row(prompt=ROOT+' Check signs.',risk='COMMON_SAFE_REJECTION',index=0,backend=True):
    c=SearchCandidate('synthetic_'+str(index),prompt,backend_details=details(prompt) if backend else {})
    return EvaluatedCandidate(c,None,None,False,False,{'scientific_risk_code':risk})


def commit(m,member=1,lane='direct_flip',rows=(),**kwargs):
    out=OpportunityOutcome(op(member,lane),tuple(rows),None,False,None,m.clock,**kwargs)
    delta=m.prepare_outcome(out);m.validate_delta(delta);m.apply_outcome(delta)
    return delta


def test_same_member_three_failures_private_only_and_relevance():
    m=memory()
    for _ in range(3):assert observe(m,1)
    observe(m,1,ROOT+' Check units.')
    commit(m)
    assert not m.shared and m.audit()['failure_to_shared_promotions']==0
    assert m.audit()['failure_family_count']==2 and m.audit()['failure_family_repeated_count']==1
    assert 'sign checks' in m.read_for_member(1,'direct_flip')['private_failure'][0]['action']


def test_two_members_promote_only_at_commit_then_third_updates_one_family():
    m=memory();observe(m,1);commit(m,1)
    observe(m,3,ROOT+' Verify signs before completion.')
    assert not m.shared # private negative feedback is immediate, shared promotion is not.
    commit(m,3)
    assert len(m.shared)==1 and m.shared[0].distinct_member_count==2
    assert m.shared[0].origin=='REPEATED_PRIVATE_FAILURE' and m.shared[0].occurrence_count==2
    ident=m.shared[0].memory_id;seen=m.shared[0].last_seen_update
    observe(m,4);commit(m,4)
    assert len(m.shared)==1 and m.shared[0].memory_id==ident
    assert m.shared[0].distinct_member_count==3 and m.shared[0].occurrence_count==3
    assert m.shared[0].last_seen_update>seen
    assert m.audit()['failure_to_shared_promotions']==1 and m.audit()['shared_risk_updated']==1


def test_hash_and_equal_delta_do_not_define_family():
    signs=details();units=details(ROOT+' Check units.')
    assert signs['local_parent_correct_delta']==units['local_parent_correct_delta']
    assert failure_signature(signs,'direct_flip')!=failure_signature(units,'direct_flip')
    a=ROOT+' Check signs carefully.';b=ROOT+' Verify signs before completion.'
    assert a!=b and failure_signature(details(a),'direct_flip')==failure_signature(details(b),'coverage')


def test_case_mechanism_preservation_and_unmeasured_loss_are_distinct():
    case=failure_signature(details(ROOT+' Enumerate cases.'),'direct_flip')
    preserved=failure_signature(details(),'direct_flip')
    unmeasured=failure_signature(details(preservation=False),'direct_flip')
    assert len({case,preserved,unmeasured})==3
    assert case.category=='CASE_ANALYSIS_PRESERVATION_REGRESSION'
    assert preserved.category=='PRESERVATION_REGRESSION'
    assert unmeasured.category=='LOCAL_COMPETENCE_REGRESSION'
    m=memory()
    for prompt in (ROOT+' Check signs.',ROOT+' Enumerate cases.'):
        for member in (1,3):observe(m,member,prompt);commit(m,member)
    assert len(m.shared)==2


@pytest.mark.parametrize('bad',[
    {'contract_valid':False},{'solver_evaluated':False},{'changed':False},
    {'memory_action':'Change FINAL_ANSWER formatting'},
    {'memory_action':'Fix invalid-output instability'},
    {'local_parent_newly_broken':-1},
    {'local_invalid_count':1},{'local_invalid_count':True},
    {'operational_failure':True},{'duplicate':True},
])
def test_operational_or_unclassified_failures_never_promote(bad):
    m=memory();d={**details(),**bad}
    assert failure_signature(d,'direct_flip') is None
    for member in (1,3):
        m.observe_local_failure(member=member,lane='direct_flip',parent=ROOT,prompt=ROOT+' Check signs.',details=d,opportunity_id='synthetic')
        commit(m,member)
    assert not m.shared


def test_missing_solver_validity_evidence_cannot_promote():
    d=details();d.pop('local_invalid_count')
    assert failure_signature(d,'direct_flip') is None


def test_shared_view_and_storage_have_no_raw_private_material():
    m=memory();raw='Synthetic entity Zephyr, answer 93481, secret sample-ID.'
    for member in (1,3):observe(m,member,ROOT+' Check signs. '+raw);commit(m,member)
    view=m.read_for_member(4,'direct_flip')
    assert view['shared_risk'] and set(view['shared_risk'][0])=={'situation','action','outcome','lesson'}
    text=json.dumps(asdict(m.shared[0]))+json.dumps(view)
    assert all(s not in text for s in (raw,'Zephyr','93481','sample-ID',ROOT,'source_candidate','gold','prediction'))
    assert all('Check signs.' not in json.dumps(e.visible()) for e in m.shared)


@pytest.mark.parametrize('risk',['TEAM_PROBE_REJECTION','COMMON_SAFE_REJECTION','SHADOW_REJECTION'])
def test_existing_structural_sources_direct_and_duplicate_merge(risk):
    m=memory();r=row(risk=risk,backend=False)
    commit(m,rows=(r,))
    assert len(m.shared)==1 and m.shared[0].origin=='TEAM_STRUCTURAL_REJECTION'
    r=replace(r,candidate=replace(r.candidate,candidate_id='synthetic_next'))
    commit(m,member=3,rows=(r,))
    assert len(m.shared)==1 and m.shared[0].occurrence_count==2 and m.shared[0].distinct_member_count==2


def test_mixed_origin_and_observation_acknowledgements_do_not_replay():
    m=memory();observe(m,1);commit(m,1,rows=(row(),))
    assert m.shared[0].origin=='TEAM_STRUCTURAL_REJECTION'
    observe(m,3);commit(m,3)
    assert m.shared[0].origin=='MIXED' and m.shared[0].distinct_member_count==2
    count=m.shared[0].occurrence_count
    for _ in range(6):commit(m)
    assert m.shared[0].occurrence_count==count
    assert m.audit()['failure_to_shared_promotions']==1


def test_history_trim_does_not_recount_promoted_private_observations():
    m=memory()
    for _ in range(60):observe(m,1);observe(m,3)
    commit(m)
    assert m.shared[0].occurrence_count==120
    assert len(m.shared[0].observations)==48
    for _ in range(3):commit(m)
    assert m.shared[0].occurrence_count==120


def risk_prompts():
    phrases=('Check constraints.','Enumerate cases.','Check boundaries.','Verify independently.',
        'Check relational consistency.','Decompose subproblems.','Check assumptions.',
        'Check domain.','Use exact fractions.','Check units.','Check signs.',
        'Substitute values.','Simplify expressions.','Draw a diagram.')
    found={}
    for pair in combinations(phrases,2):
        prompt=ROOT+' '+' '.join(pair);sig=failure_signature(details(prompt),'direct_flip')
        if sig:found.setdefault(sig.key(),prompt)
    assert len(found)>=60
    return list(found.values())[:60]


def test_real_limit_overflow_has_deterministic_low_value_eviction():
    a,b=memory(),memory();prompts=risk_prompts()
    for start in range(0,len(prompts),4):
        batch=tuple(row(p,index=i) for i,p in enumerate(prompts[start:start+4],start))
        for m in (a,b):
            commit(m,rows=batch)
            assert len(m.shared)<=48
            assert len({e.signature.key() for e in m.shared})==len(m.shared)
    assert a.shared==b.shared and len(a.shared)==48
    assert a.audit()['shared_risk_evicted']==12
    assert a.audit()['shared_risk_storage_peak']==48
    # All entries have one observation/same lane/member; the earliest batch
    # has strictly lower retention value than the most recent twelve families.
    assert not any(e.signature==failure_signature(details(p),'direct_flip') for e in a.shared for p in prompts[:12])


def test_recent_strong_risk_displaces_weak_and_old_ten_times_cannot_live_forever():
    m=memory()
    for i in range(10):commit(m,rows=(row(index=i),))
    ident=m.shared[0].memory_id;assert m.shared[0].occurrence_count==10
    for _ in range(48):commit(m)
    assert all(e.memory_id!=ident for e in m.shared)
    assert m.audit()['shared_risk_evicted']>=1
    observe(m,1,ROOT+' Check units.');commit(m,1)
    observe(m,3,ROOT+' Check units.');commit(m,3)
    assert len(m.shared)==1 and m.shared[0].distinct_member_count==2


def test_storage_selection_prefers_recurrence_before_recency():
    m=memory();observe(m,1);commit(m,1);observe(m,3);commit(m,3)
    protected=m.shared[0].memory_id
    prompts=risk_prompts()
    for start in range(0,len(prompts),4):commit(m,rows=tuple(row(p,index=i) for i,p in enumerate(prompts[start:start+4],start)))
    assert protected in {e.memory_id for e in m.shared} and len(m.shared)==48


def test_private_shared_lifecycles_independent_and_attempt_isolation():
    m=memory();observe(m,1);commit(m,1);observe(m,3);commit(m,3)
    shared_id=m.shared[0].memory_id
    for i in range(7):observe(m,1,ROOT+f' Check units in generic step {i}.')
    assert len([e for e in m.failures if e.owner_member==1])==5
    assert shared_id in {e.memory_id for e in m.shared}
    private=tuple(m.failures)
    for _ in range(48):commit(m)
    assert not m.shared and m.failures==private
    fresh=memory();assert not fresh.shared and not fresh.failures and fresh.clock==0


def test_hundred_plus_events_storage_topk_and_complete_context_bounded(tmp_path):
    m=memory();lengths=[];prompts=risk_prompts()
    for i in range(125):
        commit(m,rows=(row(prompts[i%60],index=i),))
        view=m.read_for_member(4,'direct_flip')
        assert len(view.get('shared_risk',()))<=3 and len(m.shared)<=48
        lengths.append(len(json.dumps(view,sort_keys=True,separators=(',',':'),ensure_ascii=True)))
        assert lengths[-1]<=1200
    assert max(lengths)<=1200 and len(m.failure_events)<=240
    import os
    destination=os.environ.get('FORMAL_V3_EVIDENCE_CAPTURE_DIR')
    if destination:
        out=Path(destination);out.mkdir(parents=True,exist_ok=True)
        (out/'rolling_risk_growth.json').write_bytes((json.dumps(dict(events=125,max_context_chars=max(lengths),audit=m.audit()),indent=2)+'\n').encode())


def test_retrieval_same_lane_then_relevant_family():
    m=memory()
    commit(m,1,'coverage',rows=(row(risk='TEAM_PROBE_REJECTION',backend=False),))
    commit(m,1,'direct_flip',rows=(row(risk='SHADOW_REJECTION',backend=False),))
    view=m.read_for_member(4,'coverage')
    assert 'team probe rejection' in view['shared_risk'][0]['outcome']


@pytest.mark.parametrize('split',['validation','test','shadow'])
def test_heldout_reads_writes_fail_closed(split):
    m=memory();out=OpportunityOutcome(op(split=split),(),None,False,None,0)
    with pytest.raises(SearchContractError,match='HELDOUT'):m.read_for_opportunity(out.opportunity)
    with pytest.raises(SearchContractError,match='HELDOUT'):m.prepare_outcome(out)


def test_shadow_sample_contents_and_raw_strategy_are_never_read():
    m=memory();r=row(risk='SHADOW_REJECTION',backend=False)
    r=replace(r,diagnostics={**r.diagnostics,'shadow_question':'Secret Neptune','shadow_answer':'882','shadow_predictions':['raw']})
    commit(m,rows=(r,))
    assert all(x not in json.dumps(asdict(m.shared[0])) for x in ('Neptune','882','predictions','question'))


def test_shared_memory_does_not_change_responsibility_feasibility_or_who():
    state,d,builder,h,first=make_op();m=memory()
    commit(m,rows=(row(backend=False),))
    second=builder.build(state=state,diagnosis=d,history=h,update_index=0)
    assert asdict(first)==asdict(second)
    assert m.read_for_opportunity(second)['shared_risk']


def test_prepare_is_atomic_and_stale_outcome_rejected():
    m=memory();observe(m,1);observe(m,3)
    before=(m.shared,m.clock,dict(m.risk_counts))
    out=OpportunityOutcome(op(),(),None,False,None,0)
    delta=m.prepare_outcome(out)
    assert (m.shared,m.clock,m.risk_counts)==before
    m.validate_delta(delta);m.apply_outcome(delta)
    with pytest.raises(SearchContractError,match='stale'):m.apply_outcome(delta)
    assert len(m.shared)==1


def test_success_prepare_does_not_mutate_sequence_or_private_state():
    m=memory();r=row();out=OpportunityOutcome(op(),(r,),r.candidate.candidate_id,True,True,0)
    a=m.prepare_outcome(out);b=m.prepare_outcome(out)
    assert a==b and m.sequence==0 and not m.private
    m.apply_outcome(a)
    assert m.sequence==1 and len(m.private)==1 and not m.shared


def test_duplicate_structural_observation_does_not_double_count():
    m=memory();r=row();commit(m,rows=(r,r))
    assert m.shared[0].occurrence_count==1


def test_oversized_delta_bank_and_nonmonotone_counters_rejected():
    m=memory();observe(m,1);delta=commit(m)
    out=m.prepare_outcome(OpportunityOutcome(op(),(),None,False,None,1))
    with pytest.raises(SearchContractError,match='DELTA_INVALID'):
        m.validate_delta(replace(out,failure_events=delta.failure_events*241))
    counts=dict(out.risk_counts);counts['shared_risk_created']=-1
    with pytest.raises(SearchContractError,match='DELTA_INVALID'):
        m.validate_delta(replace(out,risk_counts=tuple(counts.items())))


def test_manifest_schema_binds_promotion_and_retention_exactly():
    import yaml
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    root=Path(__file__).resolve().parents[1]
    m=yaml.safe_load((root/'experiments/manifests/math_v2_1_shared_risk_memory_amendment_v1.yaml').read_bytes())
    assert not validate_manifest_v2(root,m)
    for field,value in (('minimum_distinct_members',1),('recent_opportunity_window',100),('failure_evidence_capacity',10000)):
        bad=copy.deepcopy(m);bad['mechanism_config']['shared_risk_policy'][field]=value
        assert validate_manifest_v2(root,bad)
    bad=copy.deepcopy(m);bad['memory_identity']=versions.STRUCTURED_ACTION_MEMORY_VERSION
    bad['execution_binding']=copy.deepcopy(yaml.safe_load((root/'experiments/manifests/math_v2_1_layer1_memory_amendment_v1.yaml').read_bytes())['execution_binding'])
    bad['memory_identity']=versions.STRUCTURED_ROLLING_RISK_MEMORY_VERSION
    assert validate_manifest_v2(root,bad) # V3 execution binding cannot silently become V4.


def test_incomplete_or_operational_outcome_no_shared_promotion():
    for flags in ({'complete':False},{'operational_failure':True}):
        m=memory();observe(m,1);observe(m,3);commit(m,**flags)
        assert not m.shared and m.clock==0


def test_new_policy_requires_explicit_identity_and_frozen_limits():
    assert versions.STRUCTURED_ACTION_MEMORY_VERSION=='structured_action_failure_memory_v3'
    assert memory().identity!=StructuredActionMemoryV3(**LIMITS).identity
    with pytest.raises(SearchContractError,match='POLICY'):StructuredRollingRiskMemoryV4(**LIMITS,risk_policy={})
    with pytest.raises(SearchContractError,match='LIMITS'):StructuredRollingRiskMemoryV4(**{**LIMITS,'shared_storage_limit':49},risk_policy=POLICY)


def test_production_factory_opt_in_identity_and_old_binding_immutable(tmp_path,historical_math_workspace):
    from multi_dataset_diverse_rl.benchmarks.math_memory_binding import MATHMemoryBinding
    from multi_dataset_diverse_rl.search.binary_composition import build_binary_orchestrator
    from multi_dataset_diverse_rl.search.layer1_memory import MemoryConditionedOptimizer
    root=historical_math_workspace
    binding=MATHMemoryBinding(root,json.loads((root/'experiments/execution_bindings/math_v2_1_memory_canary_v1.json').read_bytes()))
    assert not binding.blockers() and binding.method('A3').memory_policy==versions.STRUCTURED_ACTION_MEMORY_VERSION
    method=replace(binding.method('A3'),memory_policy=versions.STRUCTURED_ROLLING_RISK_MEMORY_VERSION,
        mechanism_config={**binding.method('A3').mechanism_config,'shared_risk_policy':copy.deepcopy(POLICY)})
    assert method.identity()!=binding.method('A3').identity()
    # Factory policy rejection happens before any Solver/data/provider initialization.
    optimizer=MemoryConditionedOptimizer(evaluator=Evaluator(),reflection_lm=Reflection(),accounting_reader=lambda:{},run_root=tmp_path)
    bad=replace(method,mechanism_config={k:v for k,v in method.mechanism_config.items() if k!='shared_risk_policy'})
    with pytest.raises(SearchContractError,match='POLICIES_NOT_BOUND'):
        build_binary_orchestrator(benchmark=binding.benchmark(),aggregation=NS(identity=method.aggregation_policy),examples=(),
            prompts=(),solver=None,optimizer=optimizer,method=bad,seed=81,shadow_loader=lambda:(),shadow_count=40,runtime_readiness=lambda:())


def test_opt_in_v4_full_fake_production_graph(tmp_path,monkeypatch,historical_math_workspace):
    # Synthetic in-memory composition, never a real binding or authorization.

    from pathlib import Path
    from multi_dataset_diverse_rl import versions
    from multi_dataset_diverse_rl.benchmarks.math_memory_binding import MATHMemoryBinding
    from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
    from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample
    from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker,BenchmarkSolver,ReflectionProvider
    from multi_dataset_diverse_rl.governance.token_accounting import serialized_request
    from multi_dataset_diverse_rl.governance.legacy.unified_execution import preexecution_manifest,bound_preflight,execution_scope
    root=historical_math_workspace
    binding_path='experiments/execution_bindings/math_v2_1_memory_canary_v1.json'
    c=json.loads((root/binding_path).read_bytes());binding=MATHMemoryBinding(root,c)
    assert not binding.blockers()
    m=preexecution_manifest(root,source_sha='0'*40,binding_path=binding_path,experiment_id='synthetic_memory',frozen=False)
    assert bound_preflight(root,m)['blockers']==['PREEXECUTION_NOT_FROZEN']
    scope=execution_scope(m,c)
    assert scope['arm']=='A3' and scope['memory_policy_identity']==versions.STRUCTURED_ACTION_MEMORY_VERSION
    assert scope['initial_memory_entries']==0 and scope['validation_calls']==scope['test_calls']==0
    old_method=binding.method('A3')
    new_method=replace(old_method,memory_policy=versions.STRUCTURED_ROLLING_RISK_MEMORY_VERSION,
        mechanism_config={**old_method.mechanism_config,'shared_risk_policy':copy.deepcopy(POLICY)})
    monkeypatch.setattr(binding,'method',lambda arm:new_method)
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
    assert run.method.memory_policy==versions.STRUCTURED_ROLLING_RISK_MEMORY_VERSION
    assert run.method.identity()==binding.method('A3').identity()
    assert run.state.initial_member_scores==initial.member_scores
    assert broker.usage['pattern']==broker.usage['test']==broker.usage['validation']==0
    assert all(x['schema']==versions.LAYER1_INPUT_SCHEMA_VERSION and 'candidate_archive' not in x for x in inputs)
    assert all('retrieved_memory' not in x and 'change_summary' not in x for x in solver_texts)
    assert run.memory.audit()['memory_context_chars']['max']<=1200
    assert len(run.memory.failures)<=5

    assert isinstance(run.memory,StructuredRollingRiskMemoryV4)
    assert old_method.memory_policy==versions.STRUCTURED_ACTION_MEMORY_VERSION
