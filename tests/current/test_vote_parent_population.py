"""V2.9 semantics and real production graph with fake providers only."""
import asyncio,json
from copy import deepcopy
from dataclasses import replace
import pytest
from tests.current.population_support import graph,BASE,contract
from multi_dataset_diverse_rl.search.parent_population import ParentPopulation,Fitness,digest
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.search.optimizer_output_contract import validate_reflection_output
from multi_dataset_diverse_rl.search.generation_failures import GeneratedOutputFailure

def pool():return ParentPopulation(scope=dict(method_identity='m',support_sha256='a'*64,support_count=60))
def fit(n,source='TEAM_VOTE',count=60,prompt=BASE,member=0):
    return Fitness(source,n,count,'a'*64,'team',tuple(prompt.prompt_hash if m==member else 'b'*64 for m in range(5)),'observation')
def add(p,m,n,i=0,source='TEAM_VOTE'):
    prompt=BASE.edit('strategy',f'Check the domain under condition {i}.')
    return prompt,p.update(m,prompt,fit(n,source,prompt=prompt,member=m),fit(n,'MEMBER_ACCURACY_PROXY',prompt=prompt,member=m))
def test_capacity_unique_lowest_tie_and_proxy_provenance():
    p=pool();prompts=[add(p,0,n,i)[0] for i,n in enumerate([25,30,10,40])]
    assert len(p.ordered(0))==3 and prompts[2].prompt_hash not in p.members[0]
    p.update(0,prompts[1],fit(1,'MEMBER_ACCURACY_PROXY',prompt=prompts[1]),fit(1,'MEMBER_ACCURACY_PROXY',prompt=prompts[1]))
    assert p.members[0][prompts[1].prompt_hash].fitness.correct==30
    p.update(0,prompts[1],fit(22,prompt=prompts[1]),fit(1,'MEMBER_ACCURACY_PROXY',prompt=prompts[1]))
    assert p.members[0][prompts[1].prompt_hash].fitness.correct==22
    with pytest.raises(SearchContractError):p.update(0,BASE,fit(2,count=3),fit(2,'MEMBER_ACCURACY_PROXY',count=3))
def test_sampling_nonzero_weighted_with_replacement_and_replay():
    p=pool();[add(p,0,n,i) for i,n in enumerate([60,30,0])]
    parents,prob=p.probabilities(0)
    assert all(x>0 for x in prob) and sum(prob)==pytest.approx(1) and prob[0]>prob[1]>prob[2]
    q=pool();q.members=deepcopy(p.members);q.revision=p.revision
    draws,plan=p.select_three(0,seed=85,opportunity_id='one')
    draws2,plan2=q.select_three(0,seed=85,opportunity_id='one')
    assert plan==plan2 and [x.identity for x in draws]==[x.identity for x in draws2]
    with pytest.raises(SearchContractError):p.select_three(0,seed=85,opportunity_id='one')
    one=pool();add(one,0,5)
    ds,_=one.select_three(0,seed=1,opportunity_id='x');assert len({d.identity for d in ds})==1
def test_checkpoint_atomic_scope_and_integrity():
    p=pool();[add(p,m,20,m) for m in range(5)];payload=p.export()
    q=pool();q.restore(payload);assert q.export()==payload
    bad=deepcopy(payload);bad['members']['0'][0]['fitness']['correct']=61
    bad['integrity_sha256']=digest({k:v for k,v in bad.items() if k!='integrity_sha256'})
    before=q.export()
    with pytest.raises(SearchContractError):q.restore(bad)
    assert q.export()==before
    other=ParentPopulation(scope=dict(method_identity='different',support_sha256='a'*64,support_count=60))
    with pytest.raises(SearchContractError):other.restore(payload)
    p.select_three(0,seed=1,opportunity_id='plan');changed=p.export()
    key=next(iter(changed['plans']));changed['plans'][key]['distribution'][0]['probability']=.5
    changed['integrity_sha256']=digest({k:v for k,v in changed.items() if k!='integrity_sha256'})
    with pytest.raises(SearchContractError):q.restore(changed)
def test_abstention_removed_and_illegal_failure_is_bounded(tmp_path):
    with pytest.raises(GeneratedOutputFailure):validate_reflection_output({'decision':'NO_SAFE_EDIT'})
    run,broker,requests,packets=graph(tmp_path,reflection=lambda p,n:'{"decision":"NO_SAFE_EDIT"}')
    result=asyncio.run(run.run(max_opportunities=1))
    assert len(packets)==6 and not result.trace[0].candidate_ids and not result.transitions
    assert all(len(run.population.ordered(m))==1 for m in range(5))
def test_three_independent_slots_complete_vote_and_final_joint(tmp_path):
    run,broker,requests,packets=graph(tmp_path);events={}
    def observe(stage,data):events[stage]=data
    run.execution_observer=observe
    result=asyncio.run(run.run(max_opportunities=1));t=result.trace[0].target_member
    search=events['EVALUATION']['search'];evaluated=events['EVALUATION']['evaluated']
    assert len(packets)==3 and len(search.candidates)==3 and len(run.evaluation.provider.fulled)==3
    assert all(p['selected_parent']==BASE.to_dict() for p in packets)
    assert len({json.dumps(p['repair_direction'],sort_keys=True) for p in packets})==1
    assert len({json.dumps(p['repair_examples'],sort_keys=True) for p in packets})==1
    assert all('preceding_attempts' not in p and 'current_panel_observations' not in p for p in packets)
    for row in evaluated:
        provenance=row.full.aggregation_diagnostics['fitness_provenance']
        assert provenance['actual_vote']['source']=='TEAM_VOTE'
        assert provenance['member_accuracy_proxy']['count']==18
        assert provenance['actual_vote']['team_prompt_hashes'][t]==row.candidate.prompt.prompt_hash
    assert len(run.population.ordered(t))==3
    assembly=result.final_assembly;assert result.final_state_id==assembly['proposed_team_state_id']
    assert assembly['selection_repeated'] is False
    selected=[r['prompt_sha256'] for r in assembly['selection']['selected']]
    assert selected==[p.prompt_hash for p in run.finalizer.proposed_prompts]
    assert assembly['joint_vote']<=18 and assembly['count']==18
    for req in requests:
        if req['model']=='gpt-4o-mini':
            assert not any(k in json.dumps(req) for k in ('REFERENCE_','scoped_experience','repair_direction','solver_trajectory'))
    assert not any('validation' in json.dumps(r) or 'test_raw' in json.dumps(r) for r in requests)
def test_no_actionable_pattern_does_not_invent_repair(tmp_path):
    run,broker,requests,packets=graph(tmp_path,uncertain=True)
    result=asyncio.run(run.run(max_opportunities=1))
    assert not packets and not result.trace[0].candidate_ids
def test_new_cost_proof_has_three_full_children_and_final_shadow():
    c=contract();b=c['provider_bounds']['bound_proof']
    assert b['per_op_logical_solver']['complete_children']==180
    assert b['final_joint_optimize_additional_calls']==0
    assert c['provider_bounds']['max_proposals_per_opportunity']==3
    assert c['parent_population_policy']['maximum_parents_per_member']==3

def test_root_diagnosis_and_selected_parent_actual_observation_are_distinct(tmp_path):
    from multi_dataset_diverse_rl.search.schemas import SearchCandidate
    from multi_dataset_diverse_rl.search.orchestrator import UnifiedSearchContext
    run,broker,requests,packets=graph(tmp_path);run.memory.bootstrap(run.state)
    run.transition.bind_initial(run.state.initial_member_scores,run.state.initial_state_id)
    provider=run.evaluation.provider;provider.bootstrap()
    parent=run.state.snapshot();diagnosis=run.analyzer.analyze(parent,run.history)
    op=run.opportunities.build(state=parent,diagnosis=diagnosis,history=run.history,update_index=0)
    t=op.target_member;other=BASE.edit('strategy','Apply a conditional repair with a domain check.')
    prompts=list(parent.member_prompts);prompts[t]=other;measured=provider.joint_snapshot(tuple(prompts))
    proxy=provider.fitness(measured,t,'MEMBER_ACCURACY_PROXY','earlier');vote=provider.fitness(measured,t,'TEAM_VOTE','earlier')
    # Keep only this earlier Parent to force the source-distinction witness.
    run.population.members[t].clear();run.population.update(t,other,vote,proxy)
    context=UnifiedSearchContext(run.benchmark,run.aggregation,run.history,op.pattern_context,{})
    asyncio.run(run.engine.search(op,context))
    assert all(p['selected_parent']==other.to_dict() for p in packets)
    for p in packets:
        for e in p['repair_examples']:
            assert [o['applies_to'] for o in e['observations']]==[['root'],['selected_parent']]
    # Provenance hashes stay in canonical private evidence, outside the LLM view.
    records=list(tmp_path.glob('group_reflection.private/*/slot_*.json'))
    assert len(records)==3
    for path in records:
        for e in json.loads(path.read_text())['canonical']['repair_examples']:
            assert e['root_observation']['prompt_sha256']==BASE.prompt_hash
            assert e['selected_parent_observation']['prompt_sha256']==other.prompt_hash

def test_rejected_pool_argmax_does_not_silently_become_incumbent(tmp_path):
    run,broker,requests,packets=graph(tmp_path,solver_behavior=lambda prompt,i,req:
        i>=6 and not ('Lose capability' in prompt and i==6))
    provider=run.evaluation.provider;provider.bootstrap()
    run.transition.bind_initial(run.state.initial_member_scores,run.state.initial_state_id)
    baseline=run.state.snapshot()
    child=next(BASE.edit('strategy',f'Lose capability under a conditional domain repair {i}.')
        for i in range(1000) if BASE.edit('strategy',f'Lose capability under a conditional domain repair {i}.').prompt_hash<BASE.prompt_hash)
    prompts=list(baseline.member_prompts);prompts[0]=child;measured=provider.joint_snapshot(tuple(prompts))
    proxy=provider.fitness(measured,0,'MEMBER_ACCURACY_PROXY','synthetic-history')
    vote=provider.fitness(measured,0,'TEAM_VOTE','synthetic-history')
    run.population.update(0,child,vote,proxy)
    result=asyncio.run(run.finalizer.finalize())
    assert result['status']=='FINAL_ASSEMBLY_REJECTED' and not result['admitted']
    assert result['proposed_team_state_id']!=baseline.team_state_id
    assert result['deployed_team_state_id']==baseline.team_state_id
    assert run.finalizer.proposed_prompts[0]==child and result['shadow_passed'] is None

def test_full_boundary_recovery_restores_memory_pool_profiles_and_stop_without_requests(tmp_path):
    from multi_dataset_diverse_rl.persistence.durable_io import read_json,atomic_write_json
    run,broker,requests,packets=graph(tmp_path/'a')
    asyncio.run(run.run(max_opportunities=1));path=run.checkpoint.path
    # Explicit offline witness at the saved pre-final boundary; closed production
    # final-selection marker must first reject any automatic search resumption.
    fresh,b2,req2,_=graph(tmp_path/'b');fresh.checkpoint.path=path
    before=len(req2)
    with pytest.raises(SearchContractError):fresh.checkpoint.restore()
    body=read_json(path)
    atomic_write_json(path.with_suffix('.pending.json'),dict(scope=body['scope'],opportunity_ordinal=body['next_ordinal'],state='BOUNDARY_COMPLETE'))
    receipt=fresh.checkpoint.restore();assert receipt['real_api_authorized'] is False
    assert len(req2)==before and fresh.history.target_counts==run.history.target_counts
    assert fresh.population.export()==body['pool'] and fresh.memory.export()==body['memory']
    assert fresh.state.initial_member_scores==run.state.initial_member_scores
    bad=deepcopy(body);bad['pool']['members']['0'][0]['fitness']['correct']=0
    bad['pool']['integrity_sha256']=digest({k:v for k,v in bad['pool'].items() if k!='integrity_sha256'})
    bad['integrity_sha256']=digest({k:v for k,v in bad.items() if k!='integrity_sha256'})
    atomic_write_json(path,bad);snapshot=fresh.population.export()
    with pytest.raises(SearchContractError):fresh.checkpoint.restore()
    assert fresh.population.export()==snapshot

def test_memory_delivers_whole_scoped_edit_with_actual_parent_effect(tmp_path):
    run,broker,requests,packets=graph(tmp_path);result=asyncio.run(run.run(max_opportunities=1))
    t=result.trace[0].target_member;view=run.memory.read_for_member(t,BASE.prompt_hash)
    assert view['entries']
    for card in view['entries']:
        assert card['scope']=='complete_optimize' and card['count']==18
        assert card['actual_edit_parent_effect']['count']==18
        assert card['selected_parent_matches_edit_parent'] is True
        assert card['edit_instruction'] is None or card['edit_instruction'].endswith('transformations.')

def test_member_accuracy_gain_vote_loss_is_measured_and_retained_without_deployment(tmp_path):
    from multi_dataset_diverse_rl.search.schemas import SearchCandidate,SearchResult
    from multi_dataset_diverse_rl.search.system_prompt import block_lineage
    prompts=tuple(BASE.edit('role',f'Member {m} performs mathematical reasoning.') for m in range(5))
    def answer(prompt,i,req):
        m=int(prompt.split('Member ')[1][0])
        if i==6:return '2' if m==1 or m==0 and 'conditional repair' not in prompt else str(3+m if m else 5)
        if i>=7:return '2'
        return '2' if m==0 and i<2 and 'conditional repair' in prompt else '3'
    run,broker,requests,packets=graph(tmp_path,prompts=prompts,solver_answer=answer)
    p=run.evaluation.provider;p.bootstrap();run.memory.bootstrap(run.state)
    run.transition.bind_initial(run.state.initial_member_scores,run.state.initial_state_id)
    state=run.state.snapshot();diagnosis=run.analyzer.analyze(state,run.history)
    op=run.opportunities.build(state=state,diagnosis=diagnosis,history=run.history,update_index=0)
    op=replace(op,target_member=0,parent_prompt=prompts[0])
    child=prompts[0].edit('strategy','Apply a conditional repair before symbolic transformation.')
    _,plan=run.population.select_three(0,seed=81,opportunity_id=op.opportunity_id)
    c=SearchCandidate('quality-witness',child,backend_details=dict(changed=True,contract_valid=True,
        opportunity_id=op.opportunity_id,parent_state_id=state.team_state_id,
        selected_parent=prompts[0].to_dict(),selected_parent_sha256=prompts[0].prompt_hash,
        intended_change='Conditional transformation repair.',repair_hypothesis='Check domains.',slot=0))
    rows=asyncio.run(run.evaluation.evaluate(op,SearchResult((c,),'synthetic',search_state={'selection_plan':plan})))
    full=rows[0].full
    assert full.member_scores[0]>state.member_scores[0]
    assert full.aggregate_score<state.team_scores['vote_correct_count']
    assert rows[0].diagnostics['population_update']['retained']
    assert run.transition.select(asyncio.run(p.active(op)),rows).candidate is None
    assert run.state.snapshot().team_state_id==state.team_state_id

def test_layer2_positive_priority_failure_discount_and_seeded_zero_selection_unchanged():
    from multi_dataset_diverse_rl.search.schemas import TeamStateSnapshot,Diagnosis
    from multi_dataset_diverse_rl.search.history import HistoryState
    from multi_dataset_diverse_rl.search.policies import ResponsibilitySignal,TargetPolicyV1
    state=TeamStateSnapshot('state',(BASE,)*5);policy=TargetPolicyV1(81)
    signals={m:ResponsibilitySignal(m,int(m==3),0,0,'direct_flip' if m==3 else 'general') for m in range(5)}
    assert policy.select(state,Diagnosis(responsibility=signals),range(5),HistoryState(failure_counts={3:999})).selected_member==3
    zeros=Diagnosis(responsibility={m:ResponsibilitySignal(m,0,0,0,'general') for m in range(5)})
    choices=[]
    for ordinal in range(12):
        h=HistoryState(target_counts={2:ordinal});a=policy.select(state,zeros,range(5),h)
        assert a==policy.select(state,zeros,range(5),h);choices.append(a.selected_member)
    assert len(set(choices))>1

def test_slot_format_feedback_resets_and_truncation_preserves_other_slots(tmp_path):
    from tests.current.population_support import response
    run,broker,requests,packets=graph(tmp_path);original=broker.transport;draws=[]
    def transport(req):
        if req['model']!='gpt-4o-mini' and len(req['messages'])==1 and not req['messages'][0]['content'].startswith('Diagnose only'):
            draws.append(req['messages'][0]['content']);n=len(draws)
            if n==1:return response('{invalid json')
            if n==4:return {**response('{}'),'finish_reason':'length'}
        return original(req)
    broker.transport=transport
    result=asyncio.run(run.run(max_opportunities=1))
    assert len(draws)==5 and len(result.trace[0].candidate_ids)==3
    assert [t.startswith('Previous draw') for t in draws]==[False,True,False,False,True]
    assert 'invalid json' not in draws[1]
    packets=[json.loads(t.splitlines()[-1]) for t in draws]
    assert packets[0]==packets[1] and packets[3]==packets[4]
    records=[json.loads(p.read_text()) for p in tmp_path.glob('group_reflection.private/*/slot_*.json')]
    assert sum(r['outcome']['input_statistics']['accounting']['successful_calls'] for r in records)==5
    assert sum(r['outcome']['input_statistics']['accounting']['input_tokens'] for r in records)==10

def test_joint_team_measurement_cannot_be_replaced_by_high_historical_votes(tmp_path):
    run,broker,requests,packets=graph(tmp_path,solver_behavior=lambda prompt,i,req:
        True if 'supportive peer' in prompt else i>=7 if 'specialized parent' in prompt else i>=6)
    provider=run.evaluation.provider;provider.bootstrap()
    run.transition.bind_initial(run.state.initial_member_scores,run.state.initial_state_id)
    helpers=tuple(BASE.edit('strategy',f'Use a supportive peer rule {m}.') for m in range(5))
    proposed=[]
    for m in range(5):
        child=BASE.edit('strategy',f'Use a specialized parent rule {m}.');proposed.append(child)
        context=list(helpers);context[m]=child;observed=provider.joint_snapshot(tuple(context))
        vote=provider.fitness(observed,m,'TEAM_VOTE',f'historical-context-{m}')
        proxy=provider.fitness(observed,m,'MEMBER_ACCURACY_PROXY',f'historical-context-{m}')
        assert vote.correct==18 and proxy.correct==11
        run.population.update(m,child,vote,proxy)
    result=asyncio.run(run.finalizer.finalize())
    assert run.finalizer.proposed_prompts==tuple(proposed)
    assert result['joint_vote']==11 and result['joint_vote']!=sum(r['fitness']['correct'] for r in result['selection']['selected'])/5
    assert not result['admitted'] and not result['selection_repeated']

def test_pattern_support_is_optimize_only_representative_and_never_correct_filler(tmp_path):
    from multi_dataset_diverse_rl.search.population_evidence import PatternSupportEvidence
    from multi_dataset_diverse_rl.search.schemas import EvidenceItem
    from multi_dataset_diverse_rl.search.optimization_evidence import POLICY
    rows=tuple(EvidenceItem('x'+str(i),'optimize',frozenset({'REPAIR'}),dict(target_member_correct=False,
        task_metadata={'type':'Algebra' if i<5 else 'Geometry'})) for i in range(6))
    context=dict(patterns=[dict(pattern_id='p',support_ids=[r.example_id for r in rows])],focus_mechanism_id='p')
    evidence=PatternSupportEvidence(81,POLICY)
    class State:team_state_id='team';member_prompts=(BASE,)*5
    view,audit=evidence.compose_opportunity(State(),None,0,rows,context,ordinal=0)
    assert len(view.repair_examples)==3 and len({r.signals['task_metadata']['type'] for r in view.repair_examples})==2
    assert evidence.compose_opportunity(State(),None,0,rows,context,ordinal=0)==(view,audit)
    singleton={**context,'patterns':[dict(pattern_id='p',support_ids=['x0'])]}
    assert len(evidence.compose_opportunity(State(),None,0,rows,singleton,ordinal=0)[0].repair_examples)==1
    wrong_split=(replace(rows[0],source_split='validation'),)+rows[1:]
    with pytest.raises(SearchContractError):evidence.compose_opportunity(State(),None,0,wrong_split,context,ordinal=0)

@pytest.mark.parametrize('field',['selected_parent','reference_final_answer','scoped_experience','repair_examples','fitness_provenance'])
def test_prompt_private_context_names_do_not_create_inference_channels(tmp_path,field):
    run,broker,requests,packets=graph(tmp_path,reflection=lambda p,n:json.dumps(dict(target_block='strategy',
        new_content=f'Consult {field} before solving the problem.',change_summary='Use supplied context.')))
    result=asyncio.run(run.run(max_opportunities=1))
    assert len(packets)==3 and len(result.trace[0].candidate_ids)==3
    solver=[r for r in requests if r['model']=='gpt-4o-mini']
    assert any(field in r['messages'][0]['content'] for r in solver)
    assert all(len(r['messages'])==2 and r['messages'][1]['role']=='user' for r in solver)
    assert all(field not in r['messages'][1]['content'] for r in solver)

def test_neutral_safe_final_assembly_is_not_an_extra_optimization_commit(tmp_path):
    run,broker,requests,packets=graph(tmp_path);provider=run.evaluation.provider;provider.bootstrap()
    run.transition.bind_initial(run.state.initial_member_scores,run.state.initial_state_id)
    parent=run.state.snapshot();child=BASE.edit('strategy','Keep a general domain check before transformations.')
    prompts=list(parent.member_prompts);prompts[0]=child;measured=provider.joint_snapshot(tuple(prompts))
    run.population.members[0].clear()
    run.population.update(0,child,provider.fitness(measured,0,'TEAM_VOTE','neutral'),
        provider.fitness(measured,0,'MEMBER_ACCURACY_PROXY','neutral'))
    result=asyncio.run(run.finalizer.finalize())
    assert result['joint_vote']==parent.team_scores['vote_correct_count']
    assert result['admitted'] and result['proposed_team_state_id']!=parent.team_state_id
    assert not run.history.transitions and not run.history.commit_counts
