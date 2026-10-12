"""V2.8 scope/delivery and role-input regressions. Synthetic gains are wiring only."""
import asyncio
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace
import pytest

from tests.current.test_structured_optimization_evidence import graph,BASE,HYPOTHESIS,response
from multi_dataset_diverse_rl.search.system_prompt import block_lineage
from multi_dataset_diverse_rl.search.optimization_evidence import coverage_effect,executability_checks,POLICY
from multi_dataset_diverse_rl.search.scoped_edit_experience import (pack_experience_view,
    actual_concepts,complete_changed_instruction,serialized_size)
from multi_dataset_diverse_rl.search.private_action_memory import LIMITS
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.search.rolling_risk_memory import EditExperience


def experience(index,status='FULL_REFUTED',hypothesis=HYPOTHESIS,rule=None,full=True):
    child=BASE.edit('strategy',rule or 'When transforming a relation, check its domain before substitution.')
    step=block_lineage(BASE,child)
    before={f'e{i}':dict(correct=i>0,valid=True) for i in range(3)}
    after={**before,'e0':dict(correct=True,valid=True)}
    effect=coverage_effect(before,after,scope='search_validation_root')
    r=dict(experience_schema='SCOPED_EDIT_EXPERIENCE_RECORD_V1',opportunity_id='op'+str(index),
      candidate_id='c'+str(index),member=0,parent_state_id='prior-team',
      pattern_id='prior-pattern',repair_hypothesis=hypothesis,
      full_parent_prompt_id=BASE.prompt_hash,child_prompt_id=child.prompt_hash,
      edit_chain=[step],edited_block='strategy',supporting_example_ids=['e0'],
      repair_context=dict(task_types=['Algebra'],source='mutation_optimize_metadata'),
      effects=dict(search_validation=effect,actual_parent_validation=deepcopy(effect)),
      status=status,provenance=dict(split='optimize'))
    if full:
        r['effects']['full']=deepcopy(effect);r['effects']['full']['scope']='full'
        r['effects']['full']['team_delta']=-1 if status=='FULL_REFUTED' else 1
    return EditExperience(0,index,r)


def render(entries,**query):
    context={**dict(pattern_id='new-pattern',hypothesis=HYPOTHESIS,
       root_prompt_id=BASE.prompt_hash,parent_state_id='current-team',task_types=['Algebra']),**query}
    return pack_experience_view(entries,(),dict(source='measured_optimize_coverage',
       original_correct=2,current_correct=2,incorrect=1,invalid=0,repeated_format_failure=0,
       newly_fixed=0,newly_broken=0),context=context,metadata={},limits=LIMITS)


def test_full_card_survives_many_recent_exact_pattern_local_entries():
    full=experience(1);local=[experience(i,'INCONCLUSIVE',full=False) for i in range(2,24)]
    for e in local:e.record['pattern_id']='new-pattern'
    value,chosen,_,audit=render([full,*local])
    assert chosen[0]==full and audit['full_delivered']==1
    assert serialized_size(value)<=1200
    card=value['edit_effects'][0]
    assert card['effects']['full']['vote_delta']==-1
    assert card['opportunity_root_match']['prompt']=='same'
    assert card['opportunity_root_match']['team']=='different'


def test_complementary_full_observations_fit_without_negative_only_bias():
    bad=experience(1);good=experience(2,'COMMITTED')
    value,chosen,_,_=render([bad,good])
    assert {c['status'] for c in value['edit_effects']}=={'FULL_REFUTED','COMMITTED'}
    assert len(chosen)==2 and serialized_size(value)<=1200
    assert 'not causal' in value['interpretation']


def test_exact_observation_duplicates_merge_but_conflicting_evidence_does_not():
    first=experience(1);duplicate=experience(2)
    duplicate.record['effects']=deepcopy(first.record['effects'])
    conflict=experience(3,'COMMITTED');conflict.record['effects']['full']['team_delta']=2
    _,chosen,_,audit=render([first,duplicate,conflict])
    assert audit['duplicate_observations']==1
    assert len(chosen)==2 and {e.record['effects']['full']['team_delta'] for e in chosen}=={-1,2}


def test_actual_action_features_do_not_relabel_inherited_parent_instructions():
    parent=BASE.edit('strategy','Use substitution and case analysis. Check units.')
    child=parent.edit('strategy',parent.strategy+' Review the final claim.')
    e=experience(1);e.record['edit_chain']=[block_lineage(parent,child)]
    assert 'substitution checks' not in actual_concepts(e.record)
    assert 'case_analysis' not in actual_concepts(e.record)
    assert complete_changed_instruction(e.record)=='Review the final claim.'


@pytest.mark.parametrize('rule',['数学'*1200,'A'+ ' long reasoning instruction'*40+'.'])
def test_long_text_is_not_sliced_or_allowed_to_evict_measured_full(rule):
    e=experience(1,rule=rule);e.record['repair_hypothesis']='中文诊断'*100
    value,chosen,_,_=render([e])
    assert chosen== (e,) and serialized_size(value)<=1200
    card=value['edit_effects'][0]
    assert 'last_step_instruction' not in card
    if 'hypothesis' in card:assert card['hypothesis']==e.record['repair_hypothesis']


def test_unrelated_full_is_not_promoted_to_universal_risk():
    e=experience(1,hypothesis='Use geometric projection to relate spatial areas.')
    e.record['repair_context']['task_types']=['Geometry']
    e.record['full_parent_prompt_id']='prior-prompt'
    value,chosen,_,audit=render([e],hypothesis='Use modular arithmetic for divisibility.',task_types=['Number Theory'])
    assert not chosen and not value['edit_effects'] and audit['related_full_available']==0


def test_current_base_success_is_applicable_even_without_topic_overlap():
    e=experience(1,'COMMITTED',hypothesis='Use geometric projection to relate spatial areas.')
    e.record['repair_context']['task_types']=['Geometry']
    _,chosen,_,audit=render([e],hypothesis='Use modular arithmetic for divisibility.',
        root_prompt_id=e.record['child_prompt_id'],task_types=['Number Theory'])
    assert chosen==(e,) and audit['related_full_available']==1


@pytest.mark.parametrize('name',['worked_reference','root_repair_diagnostic','preceding_attempts',
    'edit_capacity','last_step_instruction','delivered_experience_ids','memory_input_sha256'])
def test_new_optimizer_only_fields_cannot_be_solver_dependencies(name):
    assert 'optimizer_only_dependency' in executability_checks(BASE.edit('strategy','Read '+name+' before solving.'))


def test_rich_inputs_match_mutation_only_and_original_diagnostics_are_labeled(tmp_path):
    run,broker,requests,packets,memberships=graph(tmp_path)
    asyncio.run(run.run(max_opportunities=1))
    for packet in packets:
        for obs in packet['current_panel_observations']:
            xid=obs['solver_trajectory']['source']['example_id']
            assert xid in memberships['mutation']
            ref=obs['worked_reference'];assert ref['source']=='dataset_worked_solution' and ref['split']=='optimize'
            assert obs['solver_trajectory']['observed_response']
            if 'root_repair_diagnostic' in obs:
                d=obs['root_repair_diagnostic']
                assert d['basis']=='original_opportunity_prompt_observation_hypothesis'
                assert all(k in d for k in ('observed_failure','diagnosis','reusable_correction','suggested_block','expected_effect'))
        wire=json.dumps(packet)
        for xid in memberships['search_validation']+memberships['team_probe']:
            assert f'Synthetic case {xid.removeprefix("optimize")}:' not in wire
            assert f'DATASET_WORKED_SOLUTION_{xid.removeprefix("optimize")}' not in wire
    for req in requests:
        if req['model']=='gpt-4o-mini':
            assert all(name not in json.dumps(req) for name in ('worked_reference','root_repair_diagnostic','preceding_attempts','edit_effects'))
    assert broker.usage['validation']==broker.usage['test']==0


def test_capacity_and_feedback_are_exact_and_negative_local_edits_remain_explorable(tmp_path):
    run,_,_,packets,_=graph(tmp_path,mode='all_six')
    asyncio.run(run.run(max_opportunities=1))
    assert len(packets)==6
    for p in packets:
        parent=p['current_parent'];cap=p['edit_capacity']
        assert cap['rendered_characters']==sum(map(len,parent.values()))+4
        for block,n in cap['maximum_replacement_block_characters'].items():
            assert n==3000-4-sum(len(v) for k,v in parent.items() if k!=block)
        assert len(p['preceding_attempts'])<=3
    assert packets[0]['preceding_attempts']==[]
    assert any('measured' in x for p in packets[1:] for x in p['preceding_attempts'])
    events=run.synthetic_search.search_state['proposal_diagnostics']['proposal_outcomes']
    assert len(events)==6 and all(e['local_parent_advanced'] for e in events)
    assert run.synthetic_search.search_state['telemetry']['local_metric_evaluations']==42


def test_no_safe_feedback_changes_next_input_without_spending_extra_slots(tmp_path):
    run,broker,_,_,_=graph(tmp_path);original=broker.transport;packets=[]
    def transport(req):
        if req['model']!='gpt-4o-mini' and len(req['messages'])==1:
            packets.append(json.loads(req['messages'][0]['content'].splitlines()[-1]))
            return response('{"decision":"NO_SAFE_EDIT"}')
        return original(req)
    broker.transport=transport;asyncio.run(run.run(max_opportunities=1))
    assert len(packets)==6 and len({json.dumps(p,sort_keys=True) for p in packets})>=4
    assert all(x['outcome']=='NO_SAFE_EDIT' for p in packets for x in p['preceding_attempts'])
    assert not run.synthetic_search.candidates
    assert run.synthetic_search.search_state['telemetry']['local_metric_evaluations']==6


def test_memory_provenance_is_recorded_without_claiming_model_utilization(tmp_path):
    run,_,_,_,_=graph(tmp_path,mode='bad_only');asyncio.run(run.run(max_opportunities=1))
    r=run.memory.private[0].record
    assert r['experience_schema']=='SCOPED_EDIT_EXPERIENCE_RECORD_V1'
    assert r['parent_state_id'] and r['memory_input_sha256']
    assert r['memory_utilization']=='not_inferred_from_delivery'
    old=deepcopy(r);old.pop('experience_schema')
    with pytest.raises(SearchContractError,match='CURRENT_EDIT_EXPERIENCE_RECORD_REQUIRED'):run.memory.observe_edit(old)
    assert run.memory.read_for_member((r['member']+1)%5,'general')['edit_effects']==[]


def test_current_method_disallows_the_old_prefix_policy_before_provider():
    from tests.current.test_structured_optimization_evidence import contract
    from multi_dataset_diverse_rl.search.current_policy import require_current_contract
    c=contract();c['precluster_continuation_policy']={'identity':'AUTHORIZED_V27_PRECLUSTER_PREFIX_CONTINUATION_V1'}
    with pytest.raises(SearchContractError,match='CURRENT_EXPERIENCE_REQUIRES_FRESH_ATTEMPT'):require_current_contract(c)


def test_scoped_experience_amendment_changes_scientific_source_identity(tmp_path):
    from multi_dataset_diverse_rl.governance.source_identity import build_unified_source_identity
    p=tmp_path/'docs/design/SCOPED_EDIT_EXPERIENCE_LAYER1_V28.md'
    p.parent.mkdir(parents=True)
    p.write_text('Scoped evidence contract one',encoding='utf-8')
    before=build_unified_source_identity(tmp_path)
    p.write_text('Scoped evidence contract two',encoding='utf-8')
    after=build_unified_source_identity(tmp_path)
    assert before['scientific_source_hash']!=after['scientific_source_hash']


def test_same_member_second_search_mock_reacts_to_delivered_full_experience(tmp_path):
    # Existing fake provider is conditional on Full-refuted Memory. This proves
    # delivery-to-consumer wiring only, not real LLM quality or team efficacy.
    from dataclasses import replace
    from multi_dataset_diverse_rl.search.orchestrator import UnifiedSearchContext
    run,_,_,packets,_=graph(tmp_path,mode='bad_only');asyncio.run(run.run(max_opportunities=1))
    op=replace(run.synthetic_opportunity,opportunity_id=run.synthetic_opportunity.opportunity_id+':memory-wire')
    context=UnifiedSearchContext(run.benchmark,run.aggregation,run.history,op.pattern_context,{})
    result=asyncio.run(run.engine.search(op,context))
    assert result.candidates==() and len(packets)==12
    assert any(c['status']=='FULL_REFUTED' for c in packets[6]['retrieved_memory']['edit_effects'])
    assert packets[6]['retrieved_memory']['interpretation']
