"""V2.5 behavioral regressions with fake providers and actual production ports."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import re

import pytest

from tests.current.test_structured_optimization_evidence import graph, response, HYPOTHESIS, BASE
from multi_dataset_diverse_rl import current_contract as identities
from multi_dataset_diverse_rl.search.schemas import EvidenceItem, Diagnosis, TeamStateSnapshot, SearchContractError
from multi_dataset_diverse_rl.search.history import HistoryState
from multi_dataset_diverse_rl.search.policies import ResponsibilitySignal, TargetPolicyV1
from multi_dataset_diverse_rl.search.variable_evidence import VariableEvidenceFeasibilityV1
from multi_dataset_diverse_rl.search.selected_evidence import compose_disjoint_evidence
from multi_dataset_diverse_rl.search.textual_gradients import (
    GradientExtractor, PerExampleGradientProvider, GradientClusterProvider, score_gradient_partition)
from multi_dataset_diverse_rl.search.partition_completion import complete_known_alias_partition
from multi_dataset_diverse_rl.search.generation_failures import generation_failure
from multi_dataset_diverse_rl.search.optimization_evidence import coverage_effect
from multi_dataset_diverse_rl.search.rolling_risk_memory import failure_signature


def evidence(correct=0, labels=()):
    return tuple(EvidenceItem(f'e{i}', 'optimize', frozenset({'REPAIR'} if i>=correct else {'PRESERVATION'}),
        dict(target_member_correct=i<correct, target_member_valid=True,
            correctness_signal_identity=identities.TARGET_CORRECTNESS_SIGNAL_VERSION,
            responsibility_labels=labels, input_payload='A synthetic symbolic question.', gold='2')) for i in range(12))


@pytest.mark.parametrize('accuracy', [0,1,3,8,11,12])
def test_no_accuracy_quota_and_exact_disjoint_canary_memberships(accuracy):
    rows=evidence(accuracy)
    diagnosis=Diagnosis(responsibility={0:ResponsibilitySignal(0,0,0,0,'general')})
    feasible=VariableEvidenceFeasibilityV1().feasible(None,diagnosis,0,rows)
    assert feasible is (accuracy<12)
    if not feasible:return
    support=[r.example_id for r in rows if not r.signals['target_member_correct']]
    context=dict(patterns=[dict(pattern_id='p',support_ids=support)],focus_mechanism_id='p')
    view,audit=compose_disjoint_evidence(rows,context,seed=81,member=0,ordinal=0,state_id='state')
    groups=[set(v) for v in audit['memberships'].values()]
    assert list(map(len,groups))==[3,3,6]
    assert len(set.union(*groups))==12 and all(not a&b for i,a in enumerate(groups) for b in groups[:i])
    assert audit['assigned_repair_ids'] and set(audit['assigned_repair_ids'])<=groups[0]
    assert compose_disjoint_evidence(rows,context,seed=81,member=0,ordinal=0,state_id='state')==(view,audit)


def test_member_positive_priority_failure_discount_and_seeded_zero_replay():
    state=TeamStateSnapshot('state',(BASE,)*5)
    signals={m:ResponsibilitySignal(m,int(m==3),0,0,'direct_flip' if m==3 else 'general') for m in range(5)}
    diagnosis=Diagnosis(responsibility=signals)
    policy=TargetPolicyV1(81)
    assert policy.select(state,diagnosis,range(5),HistoryState(failure_counts={3:999})).selected_member==3
    diagnosis=Diagnosis(responsibility={m:ResponsibilitySignal(m,0,0,0,'general') for m in range(5)})
    choices=[]
    for ordinal in range(12):
        history=HistoryState(target_counts={2:ordinal})
        a=policy.select(state,diagnosis,range(5),history)
        assert a==policy.select(state,diagnosis,range(5),history)
        assert a.reason=='SEEDED_ZERO_RESPONSIBILITY_FALLBACK'
        choices.append(a.selected_member)
    assert len(set(choices))>1
    assert policy.select(state,diagnosis,[4],HistoryState()).selected_member==4
    assert policy.select(state,diagnosis,[],HistoryState()).reason=='NO_REPAIR_SIGNAL'


def test_pattern_positive_priority_and_zero_singleton_are_actionable():
    rows=evidence()
    gradients=[dict(example_id=r.example_id,gradient=HYPOTHESIS) for r in rows]
    value=dict(patterns=[dict(generalized_gradient=HYPOTHESIS,support_ids=[r.example_id for r in rows])],unassigned_ids=[])
    a=score_gradient_partition(value,rows,gradients,seed=81,state_id='state',ordinal=0)
    assert a['selection_reason']=='SEEDED_ZERO_RESPONSIBILITY_FALLBACK' and a['support_count']==12
    assert a==score_gradient_partition(value,rows,gradients,seed=81,state_id='state',ordinal=0)
    a=score_gradient_partition(dict(patterns=[dict(generalized_gradient=HYPOTHESIS,support_ids=['e0'])],
        unassigned_ids=[f'e{i}' for i in range(1,12)]),rows,gradients)
    assert a['support_count']==1
    positive=tuple(replace(r,signals={**r.signals,'responsibility_labels':('coverage',) if i==0 else ()}) for i,r in enumerate(rows))
    value=dict(patterns=[dict(generalized_gradient=HYPOTHESIS,support_ids=['e0']),
        dict(generalized_gradient='Verify transformed equations against original constraints.',support_ids=[f'e{i}' for i in range(1,12)])],unassigned_ids=[])
    a=score_gradient_partition(value,positive,gradients)
    assert a['selection_reason']=='POSITIVE_RESPONSIBILITY' and a['selected_pattern_responsibility']==1


def test_member_wrong_team_correct_is_repair_without_fabricated_responsibility(tmp_path):
    run,_,_,_,_=graph(tmp_path)
    state=run.state.snapshot()
    target=0
    rows=list(state.diagnostics['team_states'])
    first=rows[0]
    rows[0]=replace(first,team_correctness=(False,True,True,True,True),vote_correct=True)
    state=replace(state,diagnostics={**state.diagnostics,'team_states':tuple(rows)})
    diagnosis=run.analyzer.analyze(run.state.snapshot(),run.history)
    diagnosis=replace(diagnosis,benchmark_signals={**diagnosis.benchmark_signals,'assigned':{i:() for i in range(5)}})
    row=run.opportunities.source.for_member(state,diagnosis,target)[0]
    assert 'REPAIR' in row.roles and 'PRESERVATION' not in row.roles
    assert row.signals['team_correct'] and row.signals['responsibility_labels']==() and row.signals['lane']=='general'


def output(disposition='ACTIONABLE'):
    return dict(disposition=disposition,observed_failure='An observed transformation omitted a check.',
        diagnosis='A missing constraint may explain the error.',suggested_block='strategy',
        reusable_correction=HYPOTHESIS if disposition=='ACTIONABLE' else None,
        expected_effect='Retain applicable constraints.' if disposition=='ACTIONABLE' else None)


@pytest.mark.parametrize('first',['json','field','enum'])
def test_gradient_structural_recovery_then_valid_stops_and_costs_all_draws(tmp_path,first):
    run,broker,_,_,_=graph(tmp_path)
    state=run.state.snapshot();diagnosis=run.analyzer.analyze(state,run.history)
    rows=run.opportunities.source.for_member(state,diagnosis,0)[:1]
    bad=output()
    if first=='field':bad.pop('diagnosis')
    elif first=='enum':bad['suggested_block']='geometry'
    sequence=['broken JSON' if first=='json' else json.dumps(bad),json.dumps(output())]
    calls=[]
    def transport(request):
        calls.append(request);return response(sequence[len(calls)-1])
    broker.transport=transport
    extractor=GradientExtractor(PerExampleGradientProvider(broker))
    before=broker.usage['input_tokens']+broker.usage['output_tokens']
    assert len(extractor.extract(BASE,rows))==1 and len(calls)==2
    assert extractor.evidence_diagnostics[0]['recovery_succeeded']
    assert broker.usage['input_tokens']+broker.usage['output_tokens']-before==8


def test_uncertain_and_exhausted_one_wrong_preserve_other_gradients(tmp_path):
    run,broker,_,_,_=graph(tmp_path)
    state=run.state.snapshot();diagnosis=run.analyzer.analyze(state,run.history)
    rows=run.opportunities.source.for_member(state,diagnosis,0)[:3]
    calls=[]
    def transport(request):
        xid=json.loads(request['messages'][1]['content'])['example']['example_id'];calls.append(xid)
        return response('broken' if xid==rows[0].example_id else json.dumps(output('UNCERTAIN' if xid==rows[1].example_id else 'ACTIONABLE')))
    broker.transport=transport
    extractor=GradientExtractor(PerExampleGradientProvider(broker))
    result=extractor.extract(BASE,rows)
    assert [r['example_id'] for r in result]==[rows[2].example_id] and len(calls)==5
    assert [d['structural_draws'] for d in extractor.evidence_diagnostics]==[3,1,1]
    assert extractor.evidence_diagnostics[0]['exhausted']


def test_partial_partition_keeps_valid_patterns_without_guessing_and_conflicts_are_hard():
    raw=dict(patterns=[dict(generalized_gradient='valid',support_ids=['e1']),
        dict(generalized_gradient='',support_ids=['e2'])],unassigned_ids=[])
    def validate(text):
        if not text:raise generation_failure('PATTERN_GRADIENT_CLUSTER_INVALID')
    normalized,audit=complete_known_alias_partition(raw,('e1','e2','e3'),validate_generalized=validate)
    assert normalized['patterns']==raw['patterns'][:1]
    assert set(normalized['unassigned_ids'])=={'e2','e3'} and audit['discarded_pattern_count']==1
    for bad_id in ('unknown','e1'):
        bad=deepcopy(raw);bad['patterns'][1]['support_ids']=[bad_id]
        with pytest.raises(SearchContractError,match='MEMBERSHIP'):
            complete_known_alias_partition(bad,('e1','e2','e3'),validate_generalized=validate)


def test_cluster_malformed_json_recovers_and_all_malformed_returns_no_actionable(tmp_path):
    run,broker,_,_,_=graph(tmp_path)
    state=run.state.snapshot();diagnosis=run.analyzer.analyze(state,run.history)
    rows=run.opportunities.source.for_member(state,diagnosis,0)[:1]
    payload=dict(gradients=[dict(example_id=rows[0].example_id,gradient=HYPOTHESIS)])
    for sequence,expected in [(['broken',json.dumps(dict(patterns=[dict(generalized_gradient=HYPOTHESIS,support_ids=['e1'])],unassigned_ids=[]))],2),(['broken']*3,3)]:
        calls=[]
        def transport(req):calls.append(req);return response(sequence[len(calls)-1])
        broker.transport=transport
        provider=GradientClusterProvider(broker,partition_completion_policy=identities.GRADIENT_PARTITION_COMPLETION_VERSION)
        value=provider.cluster(payload,evidence_rows=rows)
        assert len(calls)==expected
        if expected==3:assert value['patterns']==[] and value['unassigned_ids']==[rows[0].example_id]


def test_gradient_leakage_exhausts_bounded_draws_but_provenance_never_recovers(tmp_path):
    run,broker,_,_,_=graph(tmp_path)
    state=run.state.snapshot();diagnosis=run.analyzer.analyze(state,run.history)
    row=run.opportunities.source.for_member(state,diagnosis,0)[0]
    calls=[]
    bad=output();bad['reusable_correction']='Copy the gold answer 2 from the supplied example.'
    broker.transport=lambda r:(calls.append(r) or response(json.dumps(bad)))
    extractor=GradientExtractor(PerExampleGradientProvider(broker))
    assert extractor.extract(BASE,(row,))==()
    assert len(calls)==3 and extractor.evidence_diagnostics[0]['disposition']=='NONACTIONABLE_EXHAUSTED'
    row=replace(row,signals={**row.signals,'solver_trajectory':{**row.signals['solver_trajectory'],
        'source':{**row.signals['solver_trajectory']['source'],'example_id':'wrong-id'}}})
    with pytest.raises(SearchContractError,match='PROVENANCE'):
        GradientExtractor(PerExampleGradientProvider(broker)).extract(BASE,(row,))
    assert len(calls)==3


def test_neutral_strategy_parent_then_answer_compound_edit_is_preserved(tmp_path):
    run,_,_,packets,_=graph(tmp_path,mode='neutral_compound',multiblock=True)
    result=asyncio.run(run.run(max_opportunities=1))
    assert result.transitions
    records=[e.record for e in run.memory.private]
    first=next(e for e in records if len(e['edit_chain'])==1)
    assert first['effects']['actual_parent_validation']['member_delta']==0
    winner=next(e for e in records if e['status']=='COMMITTED')
    assert [s['edited_blocks'] for s in winner['edit_chain']]==[['strategy'],['answer']]
    assert packets[1]['current_parent']==first['structured_lineage']['child_prompt']


@pytest.mark.parametrize('mode',['no_assigned','independent_loss'])
def test_seen_assigned_repair_is_required_and_independent_collapse_rejects(tmp_path,mode):
    run,broker,_,_,membership=graph(tmp_path)
    original=broker.transport
    def transport(req):
        if req['model']!='gpt-4o-mini' or req['messages'][0]['content']==BASE.render():return original(req)
        i=int(re.search(r'case (\d+):',req['messages'][1]['content'])[1]);xid=f'optimize{i}'
        assigned=set(membership['mutation'])
        correct=(xid in membership['team_probe']) if mode=='no_assigned' else xid in assigned
        return response('Final answer: '+('2' if correct else '3'))
    broker.transport=transport
    result=asyncio.run(run.run(max_opportunities=1))
    assert not result.transitions and not run.evaluation.provider.fulled
    reasons={e.record['promotion_reason'] for e in run.memory.private if 'promotion_reason' in e.record}
    assert ('ASSIGNED_REPAIR_FAILED' if mode=='no_assigned' else 'INDEPENDENT_PROBE_REGRESSION') in reasons


def test_full_pass_shadow_reject_never_commits_and_records_distinct_reason(tmp_path):
    run,_,_,_,_=graph(tmp_path)
    async def reject(opportunity,candidate):return False
    run.gate.check=reject
    result=asyncio.run(run.run(max_opportunities=1))
    assert not result.transitions and run.state.prompts==(BASE,)*5
    assert any(e.record.get('promotion_reason')=='FULL_PASS_SHADOW_REJECT' for e in run.memory.private)


def test_assigned_repair_with_neutral_independent_probe_can_reach_full_and_commit(tmp_path):
    run,broker,_,_,membership=graph(tmp_path)
    original=broker.transport
    def transport(req):
        if req['model']!='gpt-4o-mini' or req['messages'][0]['content']==BASE.render():return original(req)
        i=int(re.search(r'case (\d+):',req['messages'][1]['content'])[1])
        correct=i>=6 or f'optimize{i}' in membership['mutation']
        return response('Final answer: '+('2' if correct else '3'))
    broker.transport=transport
    result=asyncio.run(run.run(max_opportunities=1))
    assert result.transitions and run.evaluation.provider.fulled
    winner=next(e.record for e in run.memory.private if e.record['status']=='COMMITTED')
    assert winner['effects']['team_probe']['member_delta']==0
    assert len(winner['effects']['assigned_repair']['fixed_ids'])>0


def test_format_valid_wrong_is_not_repair_and_private_format_loss_has_closed_signature():
    before=dict(a=dict(correct=False,valid=False),b=dict(correct=True,valid=True))
    after=dict(a=dict(correct=False,valid=True),b=dict(correct=False,valid=False,repeated_format_failure=True))
    effect=coverage_effect(before,after,scope='actual_parent')
    assert effect['fixed_ids']==[] and effect['invalid_to_valid_wrong_ids']==['a']
    assert effect['valid_to_invalid_ids']==['b'] and effect['repeated_format_failure_ids']==['b']
    details=dict(changed=True,contract_valid=True,solver_evaluated=True,
        edit_lineage=dict(effects=dict(actual_parent_validation=effect)))
    signature=failure_signature(details,'general')
    assert signature.category=='OUTPUT_CONTRACT_FAILURE' and signature.action_operations==()
