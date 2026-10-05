"""Zero-API WHAT authority, wide discovery, narrow mutation and full graph proofs."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

from multi_dataset_diverse_rl import versions as v
from multi_dataset_diverse_rl.search.pattern_responsibility import (
    wrong_universe,score_partition,discovery_payload,ResponsibilityPatternDiscoveryV3,
    PatternConditionedEvidenceV4,SetLevelPatternProvider,POLICY)
from multi_dataset_diverse_rl.search.responsibility_value import responsibility_value
from multi_dataset_diverse_rl.search.policies import ResponsibilitySignal,TargetPolicyV1
from multi_dataset_diverse_rl.search.schemas import EvidenceItem,Diagnosis,SearchContractError
from multi_dataset_diverse_rl.search.history import HistoryState
from multi_dataset_diverse_rl.search.runtime_v2 import V2OpportunityBuilder
from multi_dataset_diverse_rl.search.variable_evidence import VariableEvidenceFeasibilityV1
from multi_dataset_diverse_rl.search.pattern_layer1 import PatternMemoryEngine,PatternMemoryOptimizer,pattern_input,INSTRUCTION
from multi_dataset_diverse_rl.search.rolling_risk_memory import StructuredRollingRiskMemoryV4,POLICY as RISK
from multi_dataset_diverse_rl.search.action_memory import LIMITS


def evidence(zero=False):
    labels=['direct_flip']*2+['near_margin']*2+['coverage']*3+[None]*2+(['zero'] if zero else [])
    rows=[]
    for i,lane in enumerate(labels):
        correct=lane is None
        roles={'PRESERVATION','target_correct'} if correct else {'REPAIR',lane} if lane!='zero' else {'PRESERVATION','team_correct'}
        rows.append(EvidenceItem(f'e{i}','optimize',frozenset(roles),dict(
            input_payload=f'Synthetic algebra case {i}.',gold='2',target_output='2' if correct else '3',
            correctness_signal_identity=v.TARGET_CORRECTNESS_SIGNAL_VERSION,target_member_correct=correct,target_member_valid=True,
            responsibility_labels=() if correct or lane=='zero' else (lane,),lane=lane or 'preservation',team_margin=1,
            team_disagreement=2,mutation_sensitive=False)))
    return tuple(rows)


def partition(*groups,unassigned=()):
    return dict(patterns=[dict(failure_mechanism=m,update_direction=d,support_ids=list(ids)) for m,d,ids in groups],unassigned_ids=list(unassigned))


def whole(rows):return partition(('Missing domain check','Check feasible domain before simplifying',[r.example_id for r in rows]))


def test_all_wrong_explicit_signals_across_lanes_and_zero_rows():
    rows=evidence();wrong=wrong_universe(rows)
    assert len(wrong)==7 and {r.example_id for r in wrong}=={f'e{i}' for i in range(7)}
    payload=discovery_payload(wrong)
    assert len(payload['examples'])==7 and not {'e7','e8'} & {x['example_id'] for x in payload['examples']}
    assert len(wrong_universe(evidence(True)))==8
    assert not any(k in json.dumps(payload) for k in ['memory','history','parent_prompt','structured_history'])


@pytest.mark.parametrize('split',['shadow','validation','test'])
def test_wrong_universe_firewall(split):
    with pytest.raises(SearchContractError,match='HELDOUT'):wrong_universe([replace(evidence()[0],source_split=split)])


def test_signal_required_not_REPAIR_proxy():
    r=evidence()[0]
    with pytest.raises(SearchContractError,match='SIGNAL'):wrong_universe([replace(r,signals={})])
    assert wrong_universe([replace(r,roles=frozenset({'PRESERVATION'}))])


def test_same_shared_F_cross_lane_and_counts():
    rows=wrong_universe(evidence())
    obj=partition(('Unverified domains','Check feasibility before transformations',['e0','e2','e4']),unassigned=['e1','e3','e5','e6'])
    result=score_partition(obj,rows);signal=result['patterns'][0]['responsibility']
    assert (signal['direct_count'],signal['near_margin_count'],signal['coverage_count'],signal['raw_value'])==(1,1,1,4)
    assert ResponsibilitySignal(0,1,1,1,'direct_flip').raw_value==signal['raw_value']==responsibility_value(1,1,1)
    assert result['Coverage']==3/7


@pytest.mark.parametrize('ids',[['e0'],['e2','e3']])
def test_raw_F_beats_size_confidence_and_importance(ids):
    obj=partition(('Missing feasibility check','Check domains',ids),('Unverified arithmetic','Verify algebra',['e4','e5','e6']),
        unassigned=[f'e{i}' for i in range(7) if f'e{i}' not in ids+['e4','e5','e6']])
    obj['patterns'][1].update(confidence=1,importance=999,priority=999,ranking=1)
    result=score_partition(obj,wrong_universe(evidence()))
    focus=next(p for p in result['patterns'] if p['pattern_id']==result['focus_mechanism_id'])
    assert list(focus['support_ids'])==ids and result['selected_pattern_responsibility']==4
    assert all('importance' not in p and 'confidence' not in p for p in result['patterns'])


def test_stable_tie_no_provider_order_or_confidence():
    rows=wrong_universe(evidence())
    obj=partition(('Missing feasibility check','Check domains',['e0']),('Missing verification','Verify algebra',['e2','e3']),unassigned=['e1','e4','e5','e6'])
    first=score_partition(obj,rows)
    obj['patterns'].reverse();obj['patterns'][0]['confidence']=1
    second=score_partition(obj,rows)
    assert first['focus_mechanism_id']==second['focus_mechanism_id']
    assert first['selection_tiebreak_used'] and first['focus_mechanism_id']==min(p['pattern_id'] for p in first['patterns'])


@pytest.mark.parametrize('mutation',['overlap','unknown','duplicate','missing','unassigned_overlap','type'])
def test_invalid_partitions_fail_closed(mutation):
    rows=wrong_universe(evidence());obj=whole(rows)
    if mutation=='overlap':obj['patterns'].append(dict(obj['patterns'][0]))
    if mutation=='unknown':obj['patterns'][0]['support_ids'].append('foreign')
    if mutation=='duplicate':obj['patterns'][0]['support_ids'].append('e0')
    if mutation=='missing':obj['patterns'][0]['support_ids'].pop()
    if mutation=='unassigned_overlap':obj['unassigned_ids']=['e0']
    if mutation=='type':obj['patterns'][0]['support_ids']=[1]
    with pytest.raises(SearchContractError,match='INVALID'):score_partition(obj,rows)


def test_unassigned_positive_responsibility_failure_no_generic_fallback():
    rows=wrong_universe(evidence(True))
    obj=partition(('Missing verification','Verify algebra',['e9']),unassigned=[f'e{i}' for i in range(7)])
    with pytest.raises(SearchContractError,match='COVERAGE_FAILURE'):score_partition(obj,rows)
    with pytest.raises(SearchContractError,match='NOT_ACTIONABLE'):score_partition(partition(unassigned=[r.example_id for r in rows]),rows)


@pytest.mark.parametrize('guard',[None,v.PATTERN_ABSTRACTION_GUARD_VERSION])
@pytest.mark.parametrize('text',['Return FINAL_ANSWER only','Always return 2','Use constant 123','Alice has a shortcut'])
def test_abstraction_leakage_and_interface_guard(text,guard):
    rows=list(wrong_universe(evidence()));rows[0]=replace(rows[0],signals={**rows[0].signals,'input_payload':'Alice calculates an expression.'})
    obj=whole(rows);obj['patterns'][0]['update_direction']=text
    with pytest.raises(SearchContractError,match='INVALID_ABSTRACTION|EXAMPLE_LEAKAGE'):score_partition(obj,rows,abstraction_guard_version=guard)


def test_math_imperative_is_generic_only_under_explicit_guard_identity():
    rows=list(wrong_universe(evidence()))
    rows[0]=replace(rows[0],signals={**rows[0].signals,'input_payload':'Simplify a symbolic rational expression.'})
    obj=whole(rows);obj['patterns'][0]['update_direction']='Simplify intermediate terms before checking equivalence.'
    with pytest.raises(SearchContractError,match='EXAMPLE_LEAKAGE'):score_partition(obj,rows)
    result=score_partition(obj,rows,abstraction_guard_version=v.PATTERN_ABSTRACTION_GUARD_VERSION)
    assert result['selected_pattern_responsibility']==8 and result['assigned_residual_count']==7
    with pytest.raises(SearchContractError,match='GUARD_NOT_BOUND'):score_partition(obj,rows,abstraction_guard_version='unknown')


def test_generic_command_exception_does_not_allow_copied_example_or_answer():
    from multi_dataset_diverse_rl.search.pattern_responsibility import guard_abstraction
    rows=list(wrong_universe(evidence()))
    rows[0]=replace(rows[0],signals={**rows[0].signals,'input_payload':'Simplify the symbolic expression involving nested radical factors.','gold':'specialvalue'})
    for text in ('Simplify the symbolic expression involving nested radical factors.','Simplify by returning specialvalue'):
        with pytest.raises(SearchContractError,match='EXAMPLE_LEAKAGE'):
            guard_abstraction(text,rows,abstraction_guard_version=v.PATTERN_ABSTRACTION_GUARD_VERSION)


def setup_op(events=None,obj=None,rows=None):
    events=events if events is not None else [];rows=rows or evidence()
    class Provider:
        def diagnose(self,payload):events.append('pattern.diagnose');assert len(payload['examples'])==7;return obj or whole(wrong_universe(rows))
    class Source:
        def for_member(self,*args):return rows
    class Target(TargetPolicyV1):
        def select(self,*args):events.append('target.select');return super().select(*args)
    class Discovery(ResponsibilityPatternDiscoveryV3):
        def analyze(self,*args):value=super().analyze(*args);events.append('pattern.score/select');return value
    d=Diagnosis(responsibility={0:ResponsibilitySignal(0,2,2,3,'direct_flip')})
    state=NS(team_state_id='synthetic',member_prompts=('Inspect constraints.',)*5)
    builder=V2OpportunityBuilder(source=Source(),feasibility=VariableEvidenceFeasibilityV1(),target=Target(),
        evidence=PatternConditionedEvidenceV4(),patterns=Discovery(Provider()))
    return builder.build(state=state,diagnosis=d,history=HistoryState(),update_index=0)


def test_narrow_selected_only_preservation_and_independent_transition():
    rows=evidence();rows=tuple(replace(r,roles=r.roles|{'TRANSITION_FOCUS'}) if r.example_id=='e6' else r for r in rows)
    obj=partition(('Missing feasibility','Check domains',['e0','e1']),('Missing verification','Verify algebra',['e2','e3','e4','e5','e6']))
    op=setup_op(obj=obj,rows=rows)
    assert set(r.example_id for r in op.evidence.mutation_evidence if 'REPAIR' in r.roles)<= {'e0','e1'}
    assert any(r.signals['target_member_correct'] for r in op.evidence.mutation_evidence)
    assert op.pattern_context['selected_pattern_responsibility']==8
    assert all(p['responsibility']['coverage_count']<=3 for p in op.pattern_context['patterns'])
    assert op.evaluation_plan['evidence_audit']['nonfocus_repair_count']==0
    assert any('TRANSITION_FOCUS' in r.roles for r in op.evidence.mutation_evidence)


def test_memory_after_selected_pattern_and_no_WHO_effect(tmp_path):
    events=[];op=setup_op(events);events.append('memory.read')
    from test_layer1_action_memory import Evaluator,Reflection
    r=Reflection();optimizer=PatternMemoryOptimizer(evaluator=Evaluator(),reflection_lm=r,accounting_reader=r.accounting,run_root=tmp_path)
    optimizer.memory=StructuredRollingRiskMemoryV4(**LIMITS,risk_policy=RISK)
    context=NS(pattern_view=op.pattern_context,memory_view=optimizer.memory.read_for_opportunity(op))
    engine=PatternMemoryEngine(optimizer,81);task=engine.make_task(op,context);events.append('layer1.search')
    assert events==['target.select','pattern.diagnose','pattern.score/select','memory.read','layer1.search']
    from multi_dataset_diverse_rl.search.layer1_memory import anchored_panel
    assert task.anchor_example is not None and len(anchored_panel(task))<=6
    result=asyncio.run(engine.search(op,context))
    assert result.candidates and result.search_state['telemetry']['local_metric_evaluations']<=36
    value=json.loads(r.inputs[0][len(INSTRUCTION)+1:])
    assert value['selected_pattern']['responsibility_value']==8 and 'candidate_archive' not in value
    alternate=setup_op(obj=partition(('Missing feasibility','Check domains',['e0']),('Missing verification','Verify algebra',['e1','e2','e3','e4','e5','e6'])))
    assert op.target_member==alternate.target_member and op.objective==alternate.objective
    # A full retrieved view only enters the already-selected mutation input.
    full=pattern_input(task,task.parent_prompt,[],{'private':[{'Lesson':'Avoid lost sign checks'}],'shared':[]})
    assert json.loads(full[len(INSTRUCTION)+1:])['selected_pattern']==value['selected_pattern']


@pytest.mark.parametrize('binding_version',[1,2,3])
def test_new_binding_scope_hashes_and_generation_policy(binding_version):
    from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
    from multi_dataset_diverse_rl.governance.unified_execution import preexecution_manifest,execution_scope
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
    from multi_dataset_diverse_rl.governance.token_accounting import serialized_request
    root=Path(__file__).resolve().parents[1];bp=f'experiments/execution_bindings/math_v2_1_pattern_canary_v{binding_version}.json'
    c=json.loads((root/bp).read_bytes());b=execution_binding(root,c);assert not b.blockers()
    m=preexecution_manifest(root,source_sha='0'*40,frozen=False,binding_path=bp,experiment_id='synthetic_pattern')
    assert not validate_manifest_v2(root,m)
    scope=execution_scope(m,c)
    assert scope['roles']==['solver','reflection','pattern'] and scope['initial_memory_entries']==0
    assert m['pattern_identity']==v.PATTERN_AWARE_DISCOVERY_VERSION and m['memory_identity']==v.STRUCTURED_ROLLING_RISK_MEMORY_VERSION
    if binding_version==3:
        assert m['pattern_abstraction_guard']==scope['pattern_abstraction_guard']==v.PATTERN_ABSTRACTION_GUARD_VERSION
        assert b.method('A4').mechanism_config['pattern_abstraction_guard']==v.PATTERN_ABSTRACTION_GUARD_VERSION
    broker=RequestBroker(contract=c,transport=lambda _:None,arm='A4',seed=81)
    for role in ['solver','reflection','pattern']:
        req,_=broker._request_identity(role=role,split='optimize',member_slot=0 if role=='solver' else None,messages=[])
        body=json.loads(serialized_request(req));assert body['enable_thinking'] is False
        assert body['temperature']==(.2 if role=='solver' else .7)
        assert body['max_tokens']==3600 if role=='solver' else body['max_completion_tokens']==1800 and 'max_tokens' not in body
    bad=replace(b.method('A4'),mechanism_config={**b.method('A4').mechanism_config,'pattern_policy':{}})
    assert bad.identity()!=b.method('A4').identity()
    if binding_version==3:
        old_guard={k:value for k,value in c.items() if k!='pattern_abstraction_guard'}
        other=RequestBroker(contract=old_guard,transport=lambda _:None,arm='A4',seed=81)
        req,key=broker._request_identity(role='pattern',split='optimize',messages=[])
        legacy,legacy_key=other._request_identity(role='pattern',split='optimize',messages=[])
        assert req==legacy and key!=legacy_key


def test_set_level_context_stop_and_fresh_non_solver_no_cache():
    class Broker:
        def __init__(self):self.calls=0
        def _request_identity(self,**kwargs):return dict(model='synthetic',messages=kwargs['messages']), 'synthetic'
        def complete(self,**kwargs):self.calls+=1;return dict(text=json.dumps(whole(wrong_universe(evidence()))))
    b=Broker();p=SetLevelPatternProvider(b)
    p.diagnose(discovery_payload(wrong_universe(evidence())));assert p.calls==b.calls==1
    rows=list(wrong_universe(evidence()));rows[0]=replace(rows[0],signals={**rows[0].signals,'input_payload':'x'*POLICY['input_limit_tokens']})
    with pytest.raises(SearchContractError,match='CONTEXT_LIMIT'):p.diagnose(discovery_payload(rows))
    assert b.calls==1


def test_duplicate_scientific_discovery_fails_before_provider():
    class Provider:
        def __init__(self):self.calls=0
        def diagnose(self,payload):self.calls+=1;return whole(wrong_universe(evidence()))
    p=Provider();analyzer=ResponsibilityPatternDiscoveryV3(p);history=HistoryState()
    state=NS(team_state_id='synthetic');diagnosis=NS()
    analyzer.analyze(state,diagnosis,0,evidence(),history)
    with pytest.raises(SearchContractError,match='ONE_SUCCESS'):analyzer.analyze(state,diagnosis,0,evidence(),history)
    assert p.calls==1
    history.target_counts[0]=1;analyzer.analyze(state,diagnosis,0,evidence(),history);assert p.calls==2


@pytest.mark.parametrize('binding_version',[1,2,3])
def test_pattern_memory_full_fake_production_graph(tmp_path,monkeypatch,binding_version):
    from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
    from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
    from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample
    from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker,BenchmarkSolver,ReflectionProvider
    from multi_dataset_diverse_rl.governance.token_accounting import serialized_request
    root=Path(__file__).resolve().parents[1]
    alias_transport=binding_version>=2
    c=json.loads((root/f'experiments/execution_bindings/math_v2_1_pattern_canary_v{binding_version}.json').read_bytes());binding=execution_binding(root,c)
    adapter=binding.benchmark();prompts=tuple(x['prompt'] for x in json.loads((root/c['initial_team_path']).read_bytes())['members'])
    def examples(role):
        return tuple(CorrectnessExample(protocol_input('math',f'{role}{i}',{'problem':f'Synthetic {role} arithmetic {i}.'},adapter.output_contract,protocol=adapter.protocol),'1') for i in range(12 if role=='optimize' else 40))
    monkeypatch.setattr(binding,'examples',examples)
    pattern_inputs=[];optimizer_inputs=[];solver_inputs=[];ledger=[]
    def transport(req):
        body=json.loads(serialized_request(req));assert body['enable_thinking'] is False
        if body['model']=='qwen3-8b':
            text=req['messages'][1]['content'];solver_inputs.append(text);prompt,problem=text.split('\n\n',1)
            from multi_dataset_diverse_rl.benchmarks.math_v21_interface import MATH_SOLVER_INTERFACE_V4_USER_SUFFIX
            i=int(problem.removesuffix(MATH_SOLVER_INTERFACE_V4_USER_SUFFIX).rsplit(' ',1)[1].rstrip('.'))
            correct=(prompts.index(prompt)>=3 or i>=8) if prompt in prompts else True
            text='FINAL_ANSWER: '+('1' if correct else '2')
        elif len(req['messages'])==2:
            data=json.loads(req['messages'][1]['content']);pattern_inputs.append(data)
            text=json.dumps(partition(('Missing verification','Verify algebra',[r['example_id'] for r in data['examples']])))
        else:
            data=json.loads(req['messages'][0]['content'].rsplit('\n',1)[1]);optimizer_inputs.append(data)
            text=json.dumps(dict(decision_procedure=f'Inspect constraints and check signs with {len(optimizer_inputs)} independent verifications.',change_summary='Add sign checks.'))
        return dict(text=text,input_tokens=2,output_tokens=2,finish_reason='stop',provider_response_accepted=True,
            provider_metadata_loss_audited=True,provider_reasoning_content_present=False,provider_reasoning_character_count=None,
            provider_usage_details={},provider_thinking_indicators=[])
    broker=RequestBroker(contract=c,transport=transport,arm='A4',seed=81,ledger_writer=ledger.append)
    from multi_dataset_diverse_rl.search.pattern_id_transport import AliasSetLevelPatternProvider
    provider=(AliasSetLevelPatternProvider if alias_transport else SetLevelPatternProvider)(broker)
    run=binding.compose(arm='A4',seed=81,solver=BenchmarkSolver(adapter,broker),reflection=ReflectionProvider(broker),pattern_provider=provider,run_root=tmp_path)
    assert run.memory.audit()['stateful_write_count']==0
    run.state.initialize();initial=run.state.snapshot();result=asyncio.run(run.run(max_opportunities=1))
    assert result.trace and result.trace[0].candidate_ids and run.evaluation.provider.probed
    assert result.stop_reason=='CANARY_ONE_PRODUCTION_OPPORTUNITY_COMPLETE'
    assert broker.usage['pattern']==provider.calls==len(pattern_inputs)==1
    assert len(pattern_inputs[0]['examples'])==8
    assert all(x['schema']==v.PATTERN_OPTIMIZER_INPUT_VERSION and x['selected_pattern']['responsibility_value']==32 for x in optimizer_inputs)
    assert not any(r['kind']=='CACHE_HIT' and r['role']!='solver' for r in ledger)
    assert all('selected_pattern' not in text and 'retrieved_memory' not in text for text in solver_inputs)
    assert run.state.initial_member_scores==initial.member_scores and run.memory.audit()['memory_context_chars']['max']<=1200
    assert broker.usage['test']==broker.usage['validation']==0


def test_lossless_alias_wire_preserves_support_counts_and_all_other_fields(monkeypatch):
    from multi_dataset_diverse_rl.search.pattern_id_transport import AliasSetLevelPatternProvider
    rows=tuple(replace(r,example_id='x'*78+f'{i:02}') for i,r in enumerate(wrong_universe(evidence())))
    from copy import deepcopy
    payload=discovery_payload(rows);original=deepcopy(payload);seen=[]
    def response(self,value):
        seen.append(value)
        return partition(('Missing domain check','Verify feasible domain',['e1','e3']),
            ('Missing substitution check','Verify substituted constraints',['e2','e4','e5','e6','e7']))
    monkeypatch.setattr(SetLevelPatternProvider,'diagnose',response)
    decoded=AliasSetLevelPatternProvider(None).diagnose(payload)
    assert payload==original and len(seen)==1
    for index,(wire,actual) in enumerate(zip(seen[0]['examples'],payload['examples'],strict=True),1):
        assert wire['example_id']==f'e{index}'
        assert {k:v for k,v in wire.items() if k!='example_id'}=={k:v for k,v in actual.items() if k!='example_id'}
    expected=partition(('Missing domain check','Verify feasible domain',[rows[0].example_id,rows[2].example_id]),
        ('Missing substitution check','Verify substituted constraints',[rows[i].example_id for i in (1,3,4,5,6)]))
    assert score_partition(decoded,rows)==score_partition(expected,rows)


@pytest.mark.parametrize('aliases,unassigned',[(['e01'],[]),(['e8'],[]),(['e1','e1'],[]),(['e1'],['e1']),(['e1'],[])])
def test_alias_transport_does_not_repair_invalid_membership(monkeypatch,aliases,unassigned):
    from multi_dataset_diverse_rl.search.pattern_id_transport import AliasSetLevelPatternProvider
    rows=wrong_universe(evidence());calls=[]
    def response(self,value):
        calls.append(value);return partition(('Missing domain check','Verify constraints',aliases),unassigned=unassigned)
    monkeypatch.setattr(SetLevelPatternProvider,'diagnose',response)
    with pytest.raises(SearchContractError,match='PATTERN_DISCOVERY_INVALID_MEMBERSHIP'):
        decoded=AliasSetLevelPatternProvider(None).diagnose(discovery_payload(rows));score_partition(decoded,rows)
    assert len(calls)==1
