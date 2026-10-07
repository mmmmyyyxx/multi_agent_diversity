"""Synthetic contracts for target-first textual gradients and bounded mutation."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

from multi_dataset_diverse_rl import versions as v
from multi_dataset_diverse_rl.search.textual_gradients import (
    GradientExtractor, GradientPatternDiscovery, GradientPatternConditionedEvidence,
    PerExampleGradientProvider, GradientClusterProvider, score_gradient_partition,
    validate_gradient, gradient_identity, POLICY)
from multi_dataset_diverse_rl.search.pattern_primitives import wrong_universe
from multi_dataset_diverse_rl.search.schemas import EvidenceItem,Diagnosis,SearchContractError
from multi_dataset_diverse_rl.search.policies import ResponsibilitySignal,TargetPolicyV1
from multi_dataset_diverse_rl.search.history import HistoryState
from multi_dataset_diverse_rl.search.current_opportunity import CurrentOpportunityBuilder as V2OpportunityBuilder
from multi_dataset_diverse_rl.search.variable_evidence import VariableEvidenceFeasibilityV1
from multi_dataset_diverse_rl.search.current_layer1 import (
    GradientPatternMemoryOptimizer,GradientPatternMemoryEngine,gradient_pattern_input,INSTRUCTION)
from multi_dataset_diverse_rl.search.rolling_risk_memory import StructuredRollingRiskMemoryV4,POLICY as RISK
from multi_dataset_diverse_rl.search.private_action_memory import LIMITS

ROOT=Path(__file__).resolve().parents[1]
BP='experiments/execution_bindings/math_v2_1_gradient_pattern_offline_profile_v5.json'
CHECK='Check constraints before transforming intermediate expressions.'
VERIFY='Verify substitution against the original conditions.'


def evidence():
    rows=[]
    for i,lane in enumerate(['direct_flip','direct_flip','near_margin','near_margin','coverage',None,'coverage']+[None]*5):
        correct=i>=7
        rows.append(EvidenceItem(f'e{i}','optimize',frozenset({'PRESERVATION','target_correct'} if correct else {'REPAIR'}),dict(
            input_payload=f'Synthetic {"geometry" if i%2 else "algebra"} example {i}.',gold='2',
            target_output='2' if correct else '3' if i!=6 else None,target_member_valid=i!=6,target_member_correct=correct,
            correctness_signal_identity=v.TARGET_CORRECTNESS_SIGNAL_VERSION,responsibility_labels=() if correct or lane is None else (lane,),
            team_margin=1,team_disagreement=2,mutation_sensitive=False)))
    return tuple(rows)


def partition(*groups,unassigned=()):
    return dict(patterns=[dict(generalized_gradient=g,support_ids=list(ids)) for g,ids in groups],unassigned_ids=list(unassigned))


class Extract:
    def __init__(self,events=None,invalid=None,byid=None):self.inputs=[];self.events=events;self.invalid=invalid;self.byid=byid or {}
    def extract(self,payload):
        self.inputs.append(deepcopy(payload))
        if self.events is not None:self.events.append('gradient.extract')
        return {'gradient':self.byid.get(payload['example']['example_id'],CHECK) if len(self.inputs)!=self.invalid else 'Return FINAL_ANSWER only'}


class Cluster:
    def __init__(self,value=None,events=None):self.inputs=[];self.value=value;self.events=events
    def cluster(self,payload):
        self.inputs.append(deepcopy(payload))
        if self.events is not None:self.events.append('gradient.cluster')
        return deepcopy(self.value or partition((CHECK,[g['example_id'] for g in payload['gradients']])))


def discover(rows=None,value=None,events=None,invalid=None):
    rows=rows or evidence();ex=Extract(events,invalid);cl=Cluster(value,events)
    d=GradientPatternDiscovery(GradientExtractor(ex),cl)
    state=NS(team_state_id='synthetic',member_prompts=('Inspect constraints.',)*5)
    return d,state,ex,cl,rows


def test_seven_wrong_five_correct_exactly_one_single_example_including_zero_invalid():
    d,s,ex,cl,rows=discover();s.member_prompts=('Inspect constraints.',)+('Other member sentinel.',)*4
    result=d.analyze(s,None,0,rows,HistoryState())
    assert len(ex.inputs)==7 and len(cl.inputs)==1
    assert {p['example']['example_id'] for p in ex.inputs}=={f'e{i}' for i in range(7)}
    assert ex.inputs[5]['example']['responsibility_labels']==()
    assert ex.inputs[6]['example']['valid'] is False
    for i,p in enumerate(ex.inputs):
        assert set(p)=={'schema','current_member_procedure','example'}
        assert p['current_member_procedure']==s.member_prompts[0]
        assert 'Other member sentinel' not in json.dumps(p)
        assert p['example']['problem']==rows[i].signals['input_payload']
        assert all(rows[j].signals['input_payload'] not in json.dumps(p) for j in range(12) if i!=j)
    assert set(cl.inputs[0])=={'gradients'}
    assert all(set(g)=={'example_id','gradient'} for g in cl.inputs[0]['gradients'])
    for forbidden in ('problem','reference','prediction','responsibility','margin','disagreement','memory','history','procedure'):
        assert forbidden not in json.dumps(cl.inputs[0])
    assert result['total_pattern_meta_calls']==8 and result['gradient_cache_hits']==result['cluster_cache_hits']==0
    assert result['invalid_wrong_count']==result['zero_responsibility_wrong_count']==1


@pytest.mark.parametrize('text',['Always return 2','Use the given constant 123','Return FINAL_ANSWER only','you are a math solver',
    'Replace the entire procedure','Solve every mathematical problem','Check Alice before substitution','x'*401,'',None])
def test_gradient_guard_rejects_specific_content_interface_full_procedure(text):
    r=evidence()[0];r=replace(r,signals={**r.signals,'input_payload':'Alice calculates a symbolic equation with given constant 123.'})
    with pytest.raises(SearchContractError,match='GRADIENT_EXTRACTION_INVALID'):validate_gradient(text,(r,))


@pytest.mark.parametrize('punctuation', ['', ','])
@pytest.mark.parametrize('activity', ['play together', 'work together', 'travel together',
    'sit together', 'participate with each other'])
@pytest.mark.parametrize('name', ['Clara', 'IVAN'])
def test_coordinated_participation_entities_rejected_in_individual_and_cluster_gradients(punctuation,activity,name):
    r=replace(evidence()[0],signals={**evidence()[0].signals,
        'input_payload':f'Two participants, Clara and Ivan{punctuation} refuse to {activity}.'})
    text=f'Exclude {name} from simultaneous selections.'
    from multi_dataset_diverse_rl.search.pattern_primitives import guard_abstraction
    from multi_dataset_diverse_rl.current_contract import PATTERN_SPECIFIC_CONTENT_GUARD_VERSION
    with pytest.raises(SearchContractError,match='GRADIENT_EXTRACTION_INVALID'):
        validate_gradient(text,(r,))
    with pytest.raises(SearchContractError,match='EXAMPLE_LEAKAGE'):
        guard_abstraction(text,(r,),abstraction_guard_version=PATTERN_SPECIFIC_CONTENT_GUARD_VERSION)


def test_repair_preserves_v3_replay_and_generic_actionable_instruction():
    from multi_dataset_diverse_rl.search.abstraction_content import specific_content_leaked, current_specific_content_leaked
    r=replace(evidence()[0],signals={**evidence()[0].signals,
        'input_payload':'Two participants, Clara and Ivan, decline to play together.'})
    assert not specific_content_leaked('Exclude Clara from simultaneous selections.',(r,))
    assert current_specific_content_leaked('Exclude Clara from simultaneous selections.',(r,))
    validate_gradient('Enforce incompatibility constraints by counting the complement of invalid joint selections.',(r,))
    assert not current_specific_content_leaked('Compare claravel with ivanhoe independently.',(r,))


@pytest.mark.parametrize('word', ['Simplify', 'Exponent', 'Dimension', 'Total'])
def test_current_guard_preserves_generic_capitalized_mathematical_vocabulary(word):
    r=replace(evidence()[0],signals={**evidence()[0].signals,
        'input_payload':f'{word}: consider a symbolic mathematical expression.'})
    validate_gradient(f'Check the {word.lower()} using an independent transformation.',(r,))


def test_hard_character_boundary_and_no_semantic_regeneration_remain_frozen():
    validate_gradient('x'*400,(evidence()[0],))
    with pytest.raises(SearchContractError,match='GRADIENT_EXTRACTION_INVALID'):
        validate_gradient('x'*401,(evidence()[0],))
    assert POLICY['gradient_policy']['max_characters']==400
    assert POLICY['gradient_policy']['logical_calls_per_wrong']==1
    assert POLICY['gradient_policy']['successful_generations_per_wrong']==1
    assert POLICY['gradient_policy']['semantic_regeneration'] is False


def test_prompt_amendment_changes_current_binding_and_rejects_historical_authorization():
    from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
    current=json.loads((ROOT/BP).read_bytes())
    prior=json.loads((ROOT/'experiments/execution_bindings/math_v2_1_gradient_pattern_offline_profile_v1.json').read_bytes())
    assert current['gradient_prompt_sha256']!=prior['gradient_prompt_sha256']
    assert current['pattern_prompt_sha256']==prior['pattern_prompt_sha256']
    assert current['pattern_policy']['gradient_policy']['prompt_identity']=='PER_EXAMPLE_TEXTUAL_GRADIENT_PROMPT_V3'
    assert current['pattern_policy']['gradient_policy']['identity']==prior['pattern_policy']['gradient_policy']['identity']
    with pytest.raises(SearchContractError,match='CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN'):
        execution_binding(ROOT,prior)


@pytest.mark.parametrize('mutation',['outer_guard','inner_guard','old_policy','stale_prompt','regeneration','hard_limit'])
def test_manifest_schema_rejects_mixed_or_relaxed_gradient_treatments(mutation):
    from multi_dataset_diverse_rl.governance.registries import load_yaml
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    manifest=load_yaml(ROOT/'experiments/manifests/math_v2_1_gradient_pattern_seed81_canary_v2.yaml')
    old=load_yaml(ROOT/'experiments/manifests/math_v2_1_gradient_pattern_seed81_canary_v1.yaml')
    assert not validate_manifest_v2(ROOT,old)
    assert not validate_manifest_v2(ROOT,manifest)
    if mutation=='outer_guard':manifest['pattern_abstraction_guard']=old['pattern_abstraction_guard']
    if mutation=='inner_guard':manifest['mechanism_config']['pattern_abstraction_guard']=old['pattern_abstraction_guard']
    if mutation=='old_policy':manifest['mechanism_config']['pattern_policy']=old['mechanism_config']['pattern_policy']
    if mutation=='stale_prompt':manifest['mechanism_config']['pattern_policy']['gradient_policy']['prompt_identity']='PER_EXAMPLE_TEXTUAL_GRADIENT_V1'
    if mutation=='regeneration':manifest['mechanism_config']['pattern_policy']['gradient_policy']['semantic_regeneration']=True
    if mutation=='hard_limit':manifest['mechanism_config']['pattern_policy']['gradient_policy']['max_characters']=600
    assert validate_manifest_v2(ROOT,manifest)


@pytest.mark.parametrize('split',['shadow','validation','test'])
def test_heldout_rows_fail_before_gradient_provider(split):
    d,s,ex,cl,rows=discover();rows=(replace(rows[0],source_split=split),*rows[1:])
    with pytest.raises(SearchContractError,match='HELDOUT'):d.analyze(s,None,0,rows,HistoryState())
    assert not ex.inputs and not cl.inputs


def test_invalid_fourth_has_no_partial_clustering_or_semantic_retry():
    d,s,ex,cl,rows=discover(invalid=4)
    with pytest.raises(SearchContractError,match='PATTERN_GRADIENT_EXTRACTION_INVALID'):d.analyze(s,None,0,rows,HistoryState())
    assert len(ex.inputs)==4 and not cl.inputs
    with pytest.raises(SearchContractError,match='ONE_SUCCESS'):d.analyze(s,None,0,rows,HistoryState())
    assert len(ex.inputs)==4


def test_cross_topic_shared_correction_and_distinct_singleton_are_legal():
    rows=wrong_universe(evidence());gs=GradientExtractor(Extract()).extract('Inspect constraints.',rows)
    p=partition((CHECK,['e0','e1','e4']),(VERIFY,['e2']),unassigned=['e3','e5','e6'])
    result=score_gradient_partition(p,rows,gs)
    assert sorted(len(p['support_ids']) for p in result['patterns'])==[1,3]
    assert result['selected_pattern_responsibility']==8


def test_identical_normalized_generalized_text_merges_without_similarity_merges():
    rows=wrong_universe(evidence());gs=GradientExtractor(Extract()).extract('Inspect constraints.',rows)
    a=score_gradient_partition(partition((CHECK,['e0']),('  '+CHECK.upper()+'  ',['e1']),
        (VERIFY,['e2','e3','e4','e5','e6'])),rows,gs)
    assert len(a['patterns'])==2
    assert gradient_identity(CHECK)==gradient_identity(' '+CHECK.upper()+' ')
    assert gradient_identity(CHECK)!=gradient_identity('Check the feasible domain before transformation.')


def test_shared_F_direct_plus_coverage_ties_two_near_and_stable_identity():
    rows=wrong_universe(evidence());gs=GradientExtractor(Extract()).extract('Inspect constraints.',rows)
    p=partition((CHECK,['e0','e4']),(VERIFY,['e2','e3']),unassigned=['e1','e5','e6'])
    a=score_gradient_partition(p,rows,gs);p['patterns'].reverse();b=score_gradient_partition(p,rows,gs)
    assert all(x['responsibility']['raw_value']==4 for x in a['patterns'])
    assert a['focus_mechanism_id']==b['focus_mechanism_id']==min(gradient_identity(CHECK),gradient_identity(VERIFY))
    assert a['selection_tiebreak_used']


def test_fixed_gradients_label_changes_leave_cluster_request_bytes_identical():
    rows=evidence();d,s,_,cl,_=discover(rows)
    a=d.analyze(s,None,0,rows,HistoryState())
    changed=tuple(replace(r,signals={**r.signals,'responsibility_labels':('coverage',)}) if i<7 else r for i,r in enumerate(rows))
    d2,s2,_,cl2,_=discover(changed);b=d2.analyze(s2,None,0,changed,HistoryState())
    encode=lambda p:json.dumps(p,sort_keys=True,separators=(',',':')).encode()
    assert encode(cl.inputs[0])==encode(cl2.inputs[0])
    assert a['selected_pattern_responsibility']==8 and b['selected_pattern_responsibility']==7


@pytest.mark.parametrize('mutation',['overlap','unknown','duplicate','missing','extra','unassigned_overlap','label','empty','bad_gradient'])
def test_partition_conformance_fails_closed(mutation):
    rows=wrong_universe(evidence());gs=GradientExtractor(Extract()).extract('Inspect constraints.',rows)
    p=partition((CHECK,[r.example_id for r in rows]))
    if mutation=='overlap':p['patterns'].append(deepcopy(p['patterns'][0]))
    if mutation=='unknown':p['patterns'][0]['support_ids'].append('foreign')
    if mutation=='duplicate':p['patterns'][0]['support_ids'].append('e0')
    if mutation=='missing':p['patterns'][0]['support_ids'].pop()
    if mutation=='extra':p['patterns'][0]['confidence']=1
    if mutation=='unassigned_overlap':p['unassigned_ids']=['e0']
    if mutation=='label':p['patterns'][0]['support_ids']=[1]
    if mutation=='empty':p['patterns'][0]['support_ids']=[]
    if mutation=='bad_gradient':p['patterns'][0]['generalized_gradient']='Always return 2'
    with pytest.raises(SearchContractError,match='INVALID'):score_gradient_partition(p,rows,gs)


class WireBroker:
    def __init__(self):self.requests=[]
    def _request_identity(self,**kw):return dict(model='synthetic',messages=kw['messages']),'synthetic'
    def complete(self,**kw):
        self.requests.append(kw);p=json.loads(kw['messages'][1]['content'])
        return {'text':json.dumps({'gradient':CHECK} if kw['role']=='pattern_gradient' else partition((CHECK,[g['example_id'] for g in p['gradients']])))}


@pytest.mark.parametrize('key',['examples','problem','reference','prediction','labels','rank','memory','history','member_prompts'])
def test_cluster_port_rejects_hidden_raw_inputs_before_provider(key):
    b=WireBroker();p=GradientClusterProvider(b)
    with pytest.raises(SearchContractError,match='INPUT_INVALID'):p.cluster({'gradients':[{'example_id':'private-source','gradient':CHECK}],key:[]})
    assert not b.requests


def test_wire_aliases_exact_and_cluster_uses_only_gradients():
    b=WireBroker();p=GradientClusterProvider(b);ids=['opaque-long-id-a','opaque-long-id-b']
    result=p.cluster({'gradients':[{'example_id':i,'gradient':CHECK} for i in ids]})
    assert result['patterns'][0]['support_ids']==ids
    assert [r['example_id'] for r in json.loads(b.requests[0]['messages'][1]['content'])['gradients']]==['e1','e2']
    assert all(i not in json.dumps(b.requests[0]) for i in ids)


def setup_op(events=None,groups=None):
    events=events if events is not None else [];rows=evidence()
    class Source:
        def for_member(self,*args):return rows
    class Target(TargetPolicyV1):
        def select(self,*args):events.append('target.select');return super().select(*args)
    class Discovery(GradientPatternDiscovery):
        def analyze(self,*args):
            value=super().analyze(*args);events.extend(['pattern.score','pattern.select']);return value
    gradient_by_id={x:p['generalized_gradient'] for p in (groups or {}).get('patterns',[]) for x in p['support_ids']}
    ex=Extract(events,byid=gradient_by_id);cl=Cluster(groups,events)
    state=NS(team_state_id='synthetic',member_prompts=('Inspect constraints.',)*5)
    diagnosis=Diagnosis(responsibility={0:ResponsibilitySignal(0,2,2,2,'direct_flip')})
    builder=V2OpportunityBuilder(source=Source(),target=Target(),feasibility=VariableEvidenceFeasibilityV1(),
        patterns=Discovery(GradientExtractor(ex),cl),evidence=GradientPatternConditionedEvidence())
    return builder.build(state=state,diagnosis=diagnosis,history=HistoryState(),update_index=0)


def test_selected_representatives_gradient_firewall_and_memory_order(tmp_path):
    from test_layer1_action_memory import Evaluator,Reflection
    events=[];op=setup_op(events,partition((CHECK,['e0','e1','e4','e6']),(VERIFY,['e2','e3','e5'])))
    repairs=[r for r in op.evidence.mutation_evidence if 'REPAIR' in r.roles]
    assert [r.example_id for r in repairs]==['e0','e1','e4']
    assert any(r.signals['target_member_correct'] for r in op.evidence.mutation_evidence)
    assert all(r.signals['per_example_gradient']==CHECK for r in repairs)
    refl=Reflection();opt=GradientPatternMemoryOptimizer(evaluator=Evaluator(),reflection_lm=refl,accounting_reader=refl.accounting,run_root=tmp_path)
    class Memory(StructuredRollingRiskMemoryV4):
        def read_for_opportunity(self,*args):events.append('memory.read');return super().read_for_opportunity(*args)
    opt.memory=Memory(**LIMITS,risk_policy=RISK)
    assert opt.memory.audit()['stateful_write_count']==0
    context=NS(pattern_view=op.pattern_context,memory_view=opt.memory.read_for_opportunity(op))
    engine=GradientPatternMemoryEngine(opt,81);task=engine.make_task(op,context)
    events.append('layer1.search');result=asyncio.run(engine.search(op,context))
    assert events==['target.select']+['gradient.extract']*7+['gradient.cluster','pattern.score','pattern.select','memory.read','layer1.search']
    value=json.loads(refl.inputs[0][len(INSTRUCTION)+1:])
    assert value['selected_pattern']['generalized_gradient']==CHECK
    assert len(value['representative_failure_trajectories'])==3
    assert VERIFY not in refl.inputs[0] and 'per_example_gradients' not in value
    assert all('candidate_archive' not in x and 'gradient_history' not in x for x in refl.inputs)
    assert result.candidates and result.search_state['telemetry']['local_metric_evaluations']<=36
    assert all(c.backend_details['LOCAL_REJECTED_EXPORTED'] for c in result.candidates)


def test_singleton_preservation_loss_measurable(tmp_path):
    from test_layer1_action_memory import Evaluator,Reflection
    op=setup_op(groups=partition((CHECK,['e0']),unassigned=['e1','e2','e3','e4','e5','e6']))
    refl=Reflection();opt=GradientPatternMemoryOptimizer(evaluator=Evaluator(),reflection_lm=refl,accounting_reader=refl.accounting,run_root=tmp_path)
    opt.memory=StructuredRollingRiskMemoryV4(**LIMITS,risk_policy=RISK)
    context=NS(pattern_view=op.pattern_context,memory_view=opt.memory.read_for_opportunity(op))
    engine=GradientPatternMemoryEngine(opt,81);result=asyncio.run(engine.search(op,context))
    assert result.search_state['telemetry']['preservation_locally_measurable']
    assert any(c.backend_details['local_newly_broken']>0 for c in result.candidates)
    assert len(json.loads(refl.inputs[0][len(INSTRUCTION)+1:])['representative_failure_trajectories'])==1


def test_binding_scope_generation_and_fresh_roles_are_not_cached():
    from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
    from multi_dataset_diverse_rl.governance.unified_execution import preexecution_manifest,execution_scope
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
    c=json.loads((ROOT/BP).read_bytes());b=execution_binding(ROOT,c);assert not b.blockers()
    m=preexecution_manifest(ROOT,source_sha='0'*40,frozen=False,binding_path=BP,experiment_id='synthetic_gradient')
    assert not validate_manifest_v2(ROOT,m),validate_manifest_v2(ROOT,m)
    scope=execution_scope(m,c)
    assert scope['roles']==['solver','reflection','pattern_gradient','pattern_cluster']
    assert scope['pattern_gradient_call_ceiling']==12 and scope['pattern_cluster_call_ceiling']==1
    seen=[];ledger=[]
    def transport(req):
        seen.append(req);return dict(text='{}',input_tokens=2,output_tokens=2,finish_reason='stop')
    broker=RequestBroker(contract=c,transport=transport,arm='A4',seed=81,ledger_writer=ledger.append)
    for role in ('pattern_gradient','pattern_gradient','pattern_cluster'):
        broker.complete(role=role,split='optimize',stage='synthetic',messages=[])
    assert len(seen)==3 and seen[0]==seen[1]
    assert broker.usage['pattern']==3 and broker.usage['pattern_gradient']==2 and broker.usage['pattern_cluster']==1
    assert not any(r['kind']=='CACHE_HIT' for r in ledger)
    for request in seen:
        assert request['temperature']==.7 and request['max_completion_tokens']==1800 and request['extra_body']['enable_thinking'] is False
    with pytest.raises(SearchContractError,match='ROLE_SPLIT'):broker.complete(role='pattern',split='optimize',stage='raw_diagnosis',messages=[])
    with pytest.raises(SearchContractError,match='CEILING'):broker.complete(role='pattern_cluster',split='optimize',stage='duplicate',messages=[])


def test_offline_profile_is_not_ready_and_old_authorization_scope_is_incompatible():
    from multi_dataset_diverse_rl.governance.unified_execution import preexecution_manifest,execution_scope,bound_preflight,canonical_sha256
    c=json.loads((ROOT/BP).read_bytes())
    m=preexecution_manifest(ROOT,source_sha='0'*40,frozen=False,binding_path=BP,experiment_id='synthetic_gradient')
    result=bound_preflight(ROOT,m)
    assert result['gate']=='HOLD' and 'PREEXECUTION_NOT_FROZEN' in result['blockers'] and result['provider_attempts']==0
    old=json.loads((ROOT/'experiments/execution_bindings/math_v2_1_pattern_canary_v4.json').read_bytes())
    from multi_dataset_diverse_rl.governance.legacy.unified_execution import preexecution_manifest as legacy_manifest, execution_scope as legacy_scope
    # The old binding is closed and cannot be materialized against V1_2 bytes.
    # Scope construction is data-only and does not require running its preflight.
    old_manifest=dict(source_sha='0'*40,preregistration_identity='0'*64,
        execution_binding={'sha256':'0'*64})
    assert canonical_sha256(execution_scope(m,c))!=canonical_sha256(legacy_scope(old_manifest,old))
    assert m['authorization']['real_api_authorized'] is False


def test_method_emergency_limit_includes_gradient_and_cluster_costs():
    from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
    from multi_dataset_diverse_rl.search.orchestrator import UnifiedSearchOrchestrator
    c=json.loads((ROOT/BP).read_bytes());method=execution_binding(ROOT,c).method('A4')
    assert method.global_stop.emergency_max_provider_calls==c['provider_bounds']['successful_provider_calls']==1651
    assert method.global_stop.no_commit_patience==2
    for count,expected in ((1638,False),(1650,False),(1651,True)):
        probe=NS(method=method,provider_call_reader=lambda:count)
        assert UnifiedSearchOrchestrator._emergency_provider_limit_reached(probe) is expected


def test_durable_ledger_has_separate_roles_and_same_frozen_generation_reservation(tmp_path):
    from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger
    from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
    c=json.loads((ROOT/BP).read_bytes())
    with TokenLedger(tmp_path/'fake_budget',task_sha256='a'*64) as ledger:
        b=RequestBroker(contract=c,transport=lambda _:dict(text='{}',input_tokens=2,output_tokens=3,finish_reason='stop'),
            arm='A4',seed=81,token_ledger=ledger)
        for role in ('pattern_gradient','pattern_gradient','pattern_cluster'):
            b.complete(role=role,split='optimize',stage='synthetic',messages=[])
        view=ledger.view()
        assert view['by_role']['pattern_gradient']['physical_attempts']==2
        assert view['by_role']['pattern_cluster']['physical_attempts']==1
        assert view['charged_total']==15 and 'pattern' not in view['by_role']
        reservations=[e for e in ledger.events if e['kind']=='RESERVE']
        assert len(reservations)==3
        assert all(e['bound']['output_hard_cap']==1810 and e['bound']['generation_policy_identity']==v.MATH_OPTIMIZER_GENERATION_POLICY_V3_VERSION for e in reservations)


def test_invalid_json_is_one_successful_generation_then_terminal_not_retried():
    from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
    c=json.loads((ROOT/BP).read_bytes());calls=[]
    def transport(req):
        calls.append(req);return dict(text='invalid JSON',input_tokens=2,output_tokens=3,finish_reason='stop')
    b=RequestBroker(contract=c,transport=transport,arm='A4',seed=81)
    d=GradientPatternDiscovery(GradientExtractor(PerExampleGradientProvider(b)),GradientClusterProvider(b))
    with pytest.raises(SearchContractError,match='PATTERN_GRADIENT_EXTRACTION_INVALID'):
        d.analyze(NS(team_state_id='synthetic',member_prompts=('Inspect constraints.',)*5),None,0,evidence(),HistoryState())
    assert len(calls)==b.usage['pattern_gradient']==1 and b.usage['pattern_cluster']==0


@pytest.mark.parametrize('key',['pattern_gradient_calls','pattern_cluster_calls','pattern_calls','successful_provider_calls'])
def test_binding_rejects_stale_single_call_ceilings(key):
    from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
    c=json.loads((ROOT/BP).read_bytes());c['provider_bounds'][key]=1 if key!='pattern_cluster_calls' else 2
    assert execution_binding(ROOT,c).blockers()==('CURRENT_NUMERIC_CALIBRATION_SCIENTIFIC_SETTING_CHANGED',)


def test_full_fake_production_opportunity_preserves_who_memory_layer2(tmp_path,monkeypatch):
    from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
    from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
    from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample
    from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker,BenchmarkSolver,ReflectionProvider
    from multi_dataset_diverse_rl.governance.token_accounting import serialized_request
    c=json.loads((ROOT/BP).read_bytes());binding=execution_binding(ROOT,c);adapter=binding.benchmark()
    prompts=tuple(x['prompt'] for x in json.loads((ROOT/c['initial_team_path']).read_bytes())['members'])
    def examples(role):
        return tuple(CorrectnessExample(protocol_input('math',f'{role}{i}',{'problem':f'Synthetic {role} arithmetic {i}.'},adapter.output_contract,protocol=adapter.protocol),'1') for i in range(12 if role=='optimize' else 40))
    monkeypatch.setattr(binding,'examples',examples)
    gradient_inputs=[];cluster_inputs=[];optimizer_inputs=[];solver_inputs=[];ledger=[]
    def transport(req):
        body=json.loads(serialized_request(req));assert body['enable_thinking'] is False
        if body['model']=='qwen3-8b':
            payload=req['messages'][1]['content'];solver_inputs.append(payload);prompt,problem=payload.split('\n\n',1)
            from multi_dataset_diverse_rl.benchmarks.math_v21_interface import MATH_SOLVER_INTERFACE_V4_USER_SUFFIX
            i=int(problem.removesuffix(MATH_SOLVER_INTERFACE_V4_USER_SUFFIX).rsplit(' ',1)[1].rstrip('.'))
            lane=next(row['member_realization_lane'] for row in reversed(ledger)
                if row['kind']=='ATTEMPT' and row['role']=='solver')
            correct=(lane>=3 or i>=7) if prompt in prompts else True
            text='FINAL_ANSWER: '+('1' if correct else '2')
        elif len(req['messages'])==2:
            p=json.loads(req['messages'][1]['content'])
            if 'example' in p:gradient_inputs.append(p);text=json.dumps({'gradient':CHECK})
            else:cluster_inputs.append(p);text=json.dumps(partition((CHECK,[g['example_id'] for g in p['gradients']])))
        else:
            p=json.loads(req['messages'][0]['content'].rsplit('\n',1)[1]);optimizer_inputs.append(p)
            text=json.dumps(dict(decision_procedure=f'Inspect constraints and check signs with {len(optimizer_inputs)} independent verifications.',change_summary='Add sign checks.'))
        return dict(text=text,input_tokens=2,output_tokens=2,finish_reason='stop',provider_response_accepted=True,
            provider_metadata_loss_audited=True,provider_reasoning_content_present=False,provider_reasoning_character_count=None,
            provider_usage_details={},provider_thinking_indicators=[])
    broker=RequestBroker(contract=c,transport=transport,arm='A4',seed=81,ledger_writer=ledger.append)
    provider=GradientClusterProvider(broker,gradient_provider=PerExampleGradientProvider(broker))
    run=binding.compose(arm='A4',seed=81,solver=BenchmarkSolver(adapter,broker),reflection=ReflectionProvider(broker),pattern_provider=provider,run_root=tmp_path)
    run.state.initialize();initial=run.state.snapshot();result=asyncio.run(run.run(max_opportunities=1))
    assert result.trace[0].candidate_ids and run.evaluation.provider.probed
    assert result.stop_reason=='CANARY_ONE_PRODUCTION_OPPORTUNITY_COMPLETE'
    assert len(gradient_inputs)==7 and len(cluster_inputs)==1 and broker.usage['pattern']==8
    assert all(p['schema']==v.GRADIENT_OPTIMIZER_INPUT_VERSION and p['selected_pattern']['responsibility_value']==28 for p in optimizer_inputs)
    assert all('selected_pattern' not in text and 'retrieved_memory' not in text and 'per_example_gradient' not in text for text in solver_inputs)
    assert run.state.initial_member_scores==initial.member_scores and run.memory.audit()['memory_context_chars']['max']<=1200
    assert broker.usage['test']==broker.usage['validation']==0
    assert all(r['role']!='pattern' for r in ledger)
    assert not any(r['kind']=='CACHE_HIT' and r['role']!='solver' for r in ledger)
