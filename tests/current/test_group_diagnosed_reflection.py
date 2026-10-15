"""Synthetic root provenance, wire projection, role accounting and noninterference."""
import asyncio,json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import pytest
from tests.current.population_support import graph,contract,BASE,response
from multi_dataset_diverse_rl.search.group_diagnosis import (
    RootTeamEvidence,GroupDiagnoser,GroupOutputFailure,group_view,validate_experience,KINDS,POLICY)
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.search.parent_population import digest

def empty():return dict(status='NO_SUPPORTED_INSIGHT',**{k:None for k in KINDS})
def insight(example,t,peer,kind='wrong_peer_contrast'):
    value=empty();value['status']='SUPPORTED_INSIGHT'
    value[kind]=dict(observation='The compared written approaches omit a domain check.',
        hypothesis='Shared omissions may limit the proposed repair.',
        modification_implication='Check applicability before adopting another approach.',
        support=[dict(example_id=example,member_ids=[t,peer])])
    return value

def synthetic(answers=('3','2','2','4','5'),*,text=None,valid=None):
    from multi_dataset_diverse_rl.benchmarks.math_structured_interface import MATHStructuredSystemBenchmark
    from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
    from multi_dataset_diverse_rl.benchmarks.math_flexible_answer import classify_prediction
    from multi_dataset_diverse_rl.benchmarks.math_response_evidence import solver_profile,trajectory_policy
    from multi_dataset_diverse_rl.search.binary_runtime import BinaryTeamStateStore,CorrectnessExample,BinaryEvidenceSource
    from multi_dataset_diverse_rl.search.scientific_aggregation import EquivalencePluralityAggregation
    from multi_dataset_diverse_rl.search.binary_responsibility import BinaryPluralityResponsibilityAnalyzer
    from multi_dataset_diverse_rl.search.history import HistoryState
    bench=MATHStructuredSystemBenchmark(contract());prompts=(BASE,)*5
    item=protocol_input('math','synthetic',{'problem':'A synthetic calculation asks for a result.'},bench.output_contract,protocol=bench.protocol)
    e=CorrectnessExample(item,'2','The reference calculation checks the domain before computing.')
    store=BinaryTeamStateStore(benchmark=bench,examples=(e,),prompts=prompts,solver=None,
        aggregation=EquivalencePluralityAggregation(),freeze_initial_competence=True)
    profiles={}
    for m,a in enumerate(answers):
        raw=(text or 'Written condition checking is recorded.\n')+'Final answer: '+a
        if valid is not None and not valid[m]:raw='An unfinished output without a final declaration.'
        pred=classify_prediction(raw,'stop')
        profiles[m]=(solver_profile(dict(request_sha256=digest([m,raw])),pred,member_id=m,prompt=prompts[m],
            example_id='synthetic',split='optimize',policy=trajectory_policy(),problem=item.problem),)
    store.profiles=profiles;state=store.snapshot();diag=BinaryPluralityResponsibilityAnalyzer(bench.capabilities).analyze(state,None)
    rows=BinaryEvidenceSource(store,HistoryState()).for_member(state,diag,0)
    pattern=dict(pattern_id='p',generalized_gradient='Check constraints before transforming expressions.',support_ids=['synthetic'])
    op=SimpleNamespace(target_member=0,parent_state_id=state.team_state_id,parent_prompt=BASE,
        evidence=SimpleNamespace(repair_examples=rows))
    return RootTeamEvidence(store).freeze(op,pattern),rows,store,op,pattern

@pytest.mark.parametrize('answers,count,vote,tie',[
    (('3','2','3','3','3'),1,False,False),
    (('3','2','2','3','3'),2,False,False),
    (('3','2','2','3','4'),2,False,True),
    (('3','2','2','4','5'),2,True,False),
    (('3','2','2','2','4'),3,True,False),
    (('3','2','2','2','2'),4,True,False),
    (('3','3','4','5','6'),0,False,False)])
def test_group_classes_use_actual_equivalence_plurality_counterfactual(answers,count,vote,tie):
    canonical,rows,*_=synthetic(answers)
    view=group_view(canonical);e=view['examples'][0]
    assert len(e['correct_members'])==count and e['team_vote']['correct']==vote and e['team_vote']['top_tie']==tie
    assert len(e['members'])==5
    # In particular two correct votes can lose, tie or win depending on wrong classes.
    assert e['team_impact']==canonical['examples'][0]['team_impact']

def test_source_prompt_validity_and_counterfactual_corruption_fail_closed():
    c,rows,store,op,p=synthetic(valid=[True,True,True,False,True])
    assert group_view(c)['examples'][0]['invalid_members']==[3]
    for mutate in (lambda d:d['root_prompts'][0].update(strategy='Different prompt'),
                   lambda d:d['examples'][0]['members'][1]['trajectory']['source'].update(split='shadow'),
                   lambda d:d['examples'][0]['team_impact'].update(vote_flip_gain=99)):
        bad=deepcopy(c);mutate(bad);bad['evidence_sha256']=digest({k:v for k,v in bad.items() if k!='evidence_sha256'})
        with pytest.raises(SearchContractError):group_view(bad)
    badrow=replace(rows[0],source_split='validation');badop=deepcopy(op);badop.evidence.repair_examples=(badrow,)
    with pytest.raises(SearchContractError):RootTeamEvidence(store).freeze(badop,p)

@pytest.mark.parametrize('kind',KINDS)
def test_optional_supported_insights_are_independent_and_scoped(kind):
    c,rows,*_=synthetic()
    peer=1 if kind in ('correct_peer_contrast','positive_experience') else 3
    value=insight('synthetic',0,peer,kind)
    assert validate_experience(value,c,rows)==value
    assert validate_experience(empty(),c,rows)==empty()
    for support in ([dict(example_id='outside',member_ids=[0,peer])],
                    [dict(example_id='synthetic',member_ids=[0,7])],
                    [dict(example_id='synthetic',member_ids=[0,0])]):
        bad=deepcopy(value);bad[kind]['support']=support
        with pytest.raises(GroupOutputFailure):validate_experience(bad,c,rows)

def test_peer_class_mismatch_specific_answer_and_repetition_rejected():
    c,rows,*_=synthetic()
    bad=insight('synthetic',0,3,'positive_experience')
    with pytest.raises(GroupOutputFailure):validate_experience(bad,c,rows)
    for text in ('Return the reference answer.',c['pattern']['generalized_gradient'],'x'*301):
        bad=insight('synthetic',0,3);bad['wrong_peer_contrast']['modification_implication']=text
        with pytest.raises(GroupOutputFailure):validate_experience(bad,c,rows)

def test_bounded_recovery_failure_empty_insight_and_provider_error():
    c,rows,*_=synthetic();calls=[]
    def provider(wire):calls.append(wire);return '{illegal'
    experience,audit=GroupDiagnoser(provider).diagnose(c,rows)
    assert len(calls)==2 and experience['status']=='GENERATION_FAILED' and audit['accepted'] is None
    assert all(experience[k] is None for k in KINDS)
    assert '{illegal' not in calls[1] and audit['attempts'][0]['raw_output']=='{illegal'
    values=iter(['{bad',json.dumps(empty())])
    e,a=GroupDiagnoser(lambda w:next(values)).diagnose(c,rows)
    assert e['status']=='NO_SUPPORTED_INSIGHT' and len(a['attempts'])==2
    def fail(wire):raise SearchContractError('PROVIDER_BROKEN')
    with pytest.raises(SearchContractError,match='PROVIDER_BROKEN'):GroupDiagnoser(fail).diagnose(c,rows)

def test_peer_groups_are_per_example_and_final_only_has_no_invented_steps():
    first,rows,*_=synthetic(text=' ')
    second,*_=synthetic(answers=('3','4','2','2','5'))
    assert first['root_prompts'][0]==first['root_prompts'][1]
    assert first['examples'][0]['members'][1]['trajectory']['solution_status']=='TRAJECTORY_UNAVAILABLE'
    changed=deepcopy(second['examples'][0]);changed['example_id']='another'
    changed['team_vote']['question_hash']='another';changed['team_impact']['question_hash']='another'
    for m in changed['members']:m['trajectory']['source']['example_id']='another'
    first['examples'].append(changed);first['pattern']['support_ids'].append('another')
    first['evidence_sha256']=digest({k:v for k,v in first.items() if k!='evidence_sha256'})
    view=group_view(first)
    assert 1 in view['examples'][0]['correct_members'] and 1 in view['examples'][1]['wrong_members']
    assert 3 in view['examples'][0]['wrong_members'] and 3 in view['examples'][1]['correct_members']

def test_diagnosis_failure_is_labeled_and_reflection_continues(tmp_path):
    run,broker,requests,packets=graph(tmp_path,group=lambda p:'broken JSON')
    out=asyncio.run(run.run(max_opportunities=1))
    assert broker.usage['group_diagnosis']==2 and len(packets)==3
    assert all(p['group_experience']['status']=='GENERATION_FAILED' for p in packets)
    assert out.trace[0].candidate_ids

def test_fixed_answer_text_is_measured_without_a_content_guard(tmp_path):
    run,broker,requests,packets=graph(tmp_path,reflection=lambda p,n:json.dumps(dict(target_block='answer',
        new_content='Always return 3.',change_summary='Copy a wrong peer final answer.')))
    out=asyncio.run(run.run(max_opportunities=1))
    assert out.trace[0].candidate_ids and len(packets)==3
    assert any('Always return 3.' in json.dumps(r) for r in requests if r['model']=='gpt-4o-mini')

def test_peer_keywords_do_not_grant_access_and_group_recovery_is_unchanged(tmp_path):
    run,broker,requests,packets=graph(tmp_path,reflection=lambda p,n:json.dumps(dict(target_block='strategy',
        new_content='Consult peer answers before solving the problem.',change_summary='Use peer context.')))
    original=broker.transport;group_draws=[]
    def transport(req):
        if req['messages'][0]['content'].startswith('Diagnose only'):
            group_draws.append(req)
            if len(group_draws)==1:return {**response('{}'),'finish_reason':'length'}
        return original(req)
    broker.transport=transport
    out=asyncio.run(run.run(max_opportunities=1))
    assert out.trace[0].candidate_ids and len(packets)==3 and len(group_draws)==2
    for req in requests:
        if req['model']=='gpt-4o-mini':
            assert len(req['messages'])==2 and req['messages'][1]['role']=='user'
            assert 'group_experience' not in req['messages'][1]['content']
    search=next(json.loads(p.read_text()) for p in tmp_path.glob('group_reflection.private/*/group.json'))
    assert search['audit']['accounting']==dict(successful_calls=2,input_tokens=4,output_tokens=4)
    assert search['audit']['attempts'][0]['category']=='GENERATED_OUTPUT_TRUNCATED'
    assert len(list(tmp_path.glob('group_reflection.private/*/group_draw_*.json')))==2

def test_context_limit_fails_before_dispatch_without_truncating():
    c,rows,*_=synthetic(text='长数学解答π−√\n'*20000)
    calls=[]
    with pytest.raises(SearchContractError,match='RESOURCE_INFEASIBLE'):
        GroupDiagnoser(lambda w:calls.append(w)).diagnose(c,rows)
    assert not calls

def test_production_shares_group_once_accounts_tokens_and_preserves_parent_draws(tmp_path):
    def group(packet):
        v=packet['evidence'];e=v['examples'][0];t=v['target_member'];peer=next(m for m in e['wrong_members'] if m!=t)
        return json.dumps(insight(e['example_id'],t,peer))
    run,broker,requests,packets=graph(tmp_path,group=group);events={}
    run.execution_observer=lambda stage,data:events.update({stage:data})
    out=asyncio.run(run.run(max_opportunities=1))
    assert broker.usage['group_diagnosis']==1 and len(packets)==3
    shared=packets[0]['group_experience']
    assert shared['status']=='SUPPORTED_INSIGHT' and all(p['group_experience']==shared for p in packets)
    search=events['EVALUATION']['search']
    assert search.search_meta_calls==broker.usage['reflection']+1
    assert search.search_meta_tokens==4*(broker.usage['reflection']+1)
    assert len({o['selected_parent_sha256'] for o in search.search_state['generation_outcomes']})==1
    assert events['GROUP_DIAGNOSIS']['canonical']['root_team_state_id']==events['OPPORTUNITY']['opportunity'].parent_state_id
    assert broker.usage['solver']==len([r for r in requests if r['model']=='gpt-4o-mini'])
    assert all(p['selected_parent']==BASE.to_dict() for p in packets)
    for p in packets:
        for e in p['repair_examples']:
            assert len(e['observations'])==1 and e['observations'][0]['applies_to']==['root','selected_parent']
            assert 'visible_solution' not in json.dumps(e) and 'request_sha256' not in json.dumps(e)
    record=json.loads(next(tmp_path.glob('group_reflection.private/*/slot_0.json')).read_text())
    stats=record['outcome']['input_statistics']
    assert stats['root_parent_deduplications']==3 and stats['preview_removed_characters']>0
    wire=next(r['messages'][0]['content'] for r in requests if r['messages'][0]['content'].startswith('Perform ONE'))
    assert json.loads(wire.splitlines()[-1])==record['reflector_view']
    assert len(wire)==stats['wire_characters_per_draw'][0]

def test_projection_unicode_long_final_only_truncation_and_same_prompt_different_execution(tmp_path):
    c,rows,*_=synthetic(text='An ordinary mathematical operation 数学核验 π.\n'*120)
    e=c['examples'][0];tr=e['members'][0]['trajectory']
    record=dict(canonical=dict(selected_parent=BASE.to_dict(),repair_direction=dict(generalized_gradient=c['pattern']['generalized_gradient']),
        repair_examples=[dict(example_id=e['example_id'],problem=e['problem'],reference_final_answer=e['reference_final_answer'],
            worked_reference=dict(text=e['reference_solution'],truncated=False),root_observation=dict(prompt_sha256=BASE.prompt_hash,trajectory=tr,correct=False),
            selected_parent_observation=dict(prompt_sha256=BASE.prompt_hash,trajectory=deepcopy(tr),correct=False),
            accepted_root_diagnostic=dict(observed_failure='A missing condition.',diagnosis='A condition may be necessary.',reusable_correction='Check applicability.'))],
        scoped_experience={},edit_capacity={},group_experience=empty()))
    from multi_dataset_diverse_rl.search.reflector_view import project_reflector
    old=record['canonical'];t=old['repair_examples'][0]['root_observation']['trajectory']['source']['member_id']
    v,s=project_reflector(old,root_prompt=BASE,selected_parent=BASE,target_member=t)
    response_text=old['repair_examples'][0]['root_observation']['trajectory']['observed_response']
    assert '数学' in response_text and len(response_text)>4096
    assert v['repair_examples'][0]['observations'][0]['observed_response']==response_text
    changed=deepcopy(old)
    # Same Prompt and visible response do not establish the same execution.
    changed['repair_examples'][0]['selected_parent_observation']['trajectory']['source']['request_sha256']='a'*64
    v,s=project_reflector(changed,root_prompt=BASE,selected_parent=BASE,target_member=t)
    assert len(v['repair_examples'][0]['observations'])==2
    # Changing the actual response is also never deduplicated (preserve Unicode).
    tr=changed['repair_examples'][0]['selected_parent_observation']['trajectory']
    from multi_dataset_diverse_rl.benchmarks.math_flexible_answer import classify_prediction
    from multi_dataset_diverse_rl.benchmarks.math_response_evidence import solver_profile,trajectory_policy
    pred=classify_prediction('Final answer: 3','length')
    profile=solver_profile(dict(request_sha256='a'*64),pred,member_id=t,prompt=BASE,
        example_id=changed['repair_examples'][0]['example_id'],split='optimize',policy=trajectory_policy(),
        problem=changed['repair_examples'][0]['problem'])
    changed['repair_examples'][0]['selected_parent_observation']['trajectory']=profile['solver_trajectory']
    v,s=project_reflector(changed,root_prompt=BASE,selected_parent=BASE,target_member=t)
    obs=v['repair_examples'][0]['observations'][1]
    assert obs['observed_response']=='Final answer: 3' and obs['response_truncated'] and not obs['prediction_valid']

def test_group_broker_identity_wire_reservation_and_split_guard(tmp_path):
    from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker,GroupDiagnosisProvider
    from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,POLICY_V25_2M,reservation
    c=contract();requests=[];raw=[];ledger=[]
    def transport(request):requests.append(request);return response(json.dumps(empty()))
    with TokenLedger(tmp_path/'accounting',task_sha256='a'*64,policy=POLICY_V25_2M) as durable:
        broker=RequestBroker(contract=c,transport=transport,arm='A4',seed=81,token_ledger=durable,
            raw_writer=raw.append,ledger_writer=ledger.append)
        GroupDiagnosisProvider(broker)('Synthetic diagnosis input')
        request=requests[0]
        assert request['model']=='qwen3.7-flash' and request['max_completion_tokens']==4096
        assert reservation(request)['generation_policy_identity']=='GROUP_DIAGNOSIS_GENERATION_POLICY_V1'
        assert durable.view()['reserved_inflight']==0
        assert raw[0]['role']=='group_diagnosis' and broker.usage['group_diagnosis']==1
        assert sum(e['charged'] for e in durable.events if e['kind']=='CHARGE')==4
        for split in ('shadow','validation','test'):
            with pytest.raises(SearchContractError):broker.complete(role='group_diagnosis',split=split,stage='forbidden',messages=[])
        assert len(requests)==1
