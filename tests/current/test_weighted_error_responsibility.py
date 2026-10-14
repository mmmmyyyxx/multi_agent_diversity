"""V2.10 classification, counterfactual preservation and production delivery."""
from itertools import product
import asyncio
from collections import Counter
import pytest
from multi_dataset_diverse_rl.search.binary_responsibility import (
    BinaryPluralityObservation, BinaryPluralityResponsibilityAnalyzer, binary_plurality_snapshot)
from multi_dataset_diverse_rl.search.weighted_responsibility import responsibility_value, LABELS, WEIGHTS
from multi_dataset_diverse_rl.search.schemas import BenchmarkCapabilities, Diagnosis, EvidenceItem, SearchContractError
from multi_dataset_diverse_rl.search.history import HistoryState
from multi_dataset_diverse_rl.search.policies import ResponsibilitySignal, TargetPolicyV1
from multi_dataset_diverse_rl.search.pattern_primitives import sample_labels, PatternResponsibilitySignal
from multi_dataset_diverse_rl.search.textual_gradients import score_gradient_partition, gradient_identity
from multi_dataset_diverse_rl.vote_aligned_scheduler import classify_opportunity_lane
from tests.current.population_support import graph

CAP = BenchmarkCapabilities(True, True, True)


def snapshot(answers):
    rows = tuple(BinaryPluralityObservation(str(i), tuple(a), tuple(bool(x) for x in a),
        tuple(x == 'g' for x in a)) for i, a in enumerate(answers))
    return binary_plurality_snapshot(state_id='synthetic', member_prompts=('p',)*5,
        observations=rows, capabilities=CAP)


def test_exhaustive_dnc_unchanged_exclusive_and_complete_error_accounting():
    # 4^5: gold, two equivalent wrong classes and an invalid/no-vote answer.
    state = snapshot(product(('g', 'a', 'b', ''), repeat=5))
    diagnosis = BinaryPluralityResponsibilityAnalyzer(CAP).analyze(state, HistoryState())
    for member, signal in diagnosis.responsibility.items():
        assigned = {o.question_hash: o for o in diagnosis.benchmark_signals['assigned'][member]}
        counts = Counter()
        for row in state.diagnostics['team_states']:
            actual = diagnosis.benchmark_signals['responsibility_labels'][member][row.question_hash]
            assert len(actual) <= 1
            old = classify_opportunity_lane(assigned[row.question_hash], diagnosis.benchmark_signals['margins']) if row.question_hash in assigned else None
            old = 'coverage' if old == 'pure_coverage' else old
            assert tuple(x for x in actual if x in LABELS[:3]) == ((old,) if old else ())
            if row.team_correctness[member]:
                assert not actual
            elif row.vote_correct and sum(row.team_correctness) in (3, 4):
                assert actual == ('r3' if sum(row.team_correctness) == 3 else 'r4',)
            if actual:
                counts[actual[0]] += 1
        assert signal.raw_value == sum(counts[k]*WEIGHTS[k] for k in LABELS)
        assert sum(counts.values()) + signal.uncovered_count == sum(not r.team_correctness[member] for r in state.diagnostics['team_states'])


def test_d_and_n_are_not_correct_member_count_categories():
    # Same one-correct count, different wrong-class partition and Vote counterfactual.
    state = snapshot([('a','g','b','c','d'), ('a','g','a','a','a')])
    diagnosis = BinaryPluralityResponsibilityAnalyzer(CAP).analyze(state, HistoryState())
    labels = diagnosis.benchmark_signals['responsibility_labels'][0]
    assert labels['0'] == ('direct_flip',)
    assert labels['1'] == ()
    # N is the abstaining M=0 boundary; one repair does not yet make Vote correct.
    state = snapshot([('a','g','a','a','b')])
    diagnosis = BinaryPluralityResponsibilityAnalyzer(CAP).analyze(state, HistoryState())
    assert diagnosis.benchmark_signals['responsibility_labels'][0]['0'] == ('near_margin',)


def test_team_correct_two_correct_is_explicitly_uncovered():
    state = snapshot([('a','g','g','b','c')])
    diagnosis = BinaryPluralityResponsibilityAnalyzer(CAP).analyze(state, HistoryState())
    assert state.diagnostics['team_states'][0].vote_correct
    assert diagnosis.responsibility[0].uncovered_count == 1
    assert diagnosis.responsibility[0].raw_value == 0
    assert diagnosis.benchmark_signals['uncovered_errors'][0]['0'] == 'TEAM_CORRECT_OTHER_CORRECT_COUNT'


def test_who_sum_and_existing_failure_discount():
    state = snapshot([('a',)*5])
    signals = {m:ResponsibilitySignal(m,0,0,0,'general') for m in range(5)}
    signals[0] = ResponsibilitySignal(0,1,0,0,'direct_flip')
    signals[1] = ResponsibilitySignal(1,0,2,1,'near_margin',r4_count=1)
    diagnosis = Diagnosis(responsibility=signals)
    h = HistoryState()
    assert TargetPolicyV1().select(state,diagnosis,range(5),h).selected_member == 1
    h.failure_counts[1] = 1
    decision = TargetPolicyV1().select(state,diagnosis,range(5),h)
    assert decision.selected_member == 0 and decision.target_scores[1] == 2.75


def row(i, label):
    return EvidenceItem(str(i),'optimize',frozenset({'REPAIR'}),dict(
        correctness_signal_identity='target_member_correctness_validity_v1',
        target_member_correct=False,target_member_valid=True,responsibility_labels=(label,) if label else (),
        input_payload='Synthetic problem without supplied constants.',gold='reference',target_output='other'))


def test_pattern_sum_includes_r3_r4_and_uncovered_without_discount():
    rows = [row(0,'direct_flip'),row(1,'near_margin'),row(2,'near_margin'),
            row(3,'r3'),row(4,'r4'),row(5,None)]
    texts = ('Check the domain before a transformation.', 'Verify the sign under the applicable condition.')
    gradients = [dict(example_id=r.example_id,gradient=texts[0] if i==0 else texts[1]) for i,r in enumerate(rows)]
    partition = dict(patterns=[dict(generalized_gradient=texts[0],support_ids=['0']),
        dict(generalized_gradient=texts[1],support_ids=['1','2','3','4','5'])],unassigned_ids=[])
    result = score_gradient_partition(partition,rows,gradients)
    assert result['focus_mechanism_id'] == gradient_identity(texts[1])
    assert result['selected_pattern_responsibility'] == 5.5
    chosen = next(p['responsibility'] for p in result['patterns'] if p['pattern_id']==result['focus_mechanism_id'])
    assert (chosen['r3_count'],chosen['r4_count'],chosen['uncovered_count']) == (1,1,1)
    assert result['zero_responsibility_wrong_count'] == 1


@pytest.mark.parametrize('labels',[('r3','r4'),('direct_flip','coverage'),('unknown',),('r3','r3')])
def test_overlap_unknown_and_duplicate_labels_fail(labels):
    from dataclasses import replace
    r = row(0,None)
    with pytest.raises(SearchContractError):
        sample_labels(replace(r,signals={**r.signals,'responsibility_labels':labels}))


@pytest.mark.parametrize('count',[True,-1,0.5,float('nan')])
def test_invalid_counts_fail(count):
    with pytest.raises(ValueError):responsibility_value(0,0,0,count,0)


def test_half_weight_and_support_accounting_guard():
    assert responsibility_value(1,2,3,4,5) == 17.5
    with pytest.raises(SearchContractError):PatternResponsibilitySignal('p',1,1,0,1)


def test_old_binding_or_unbound_weights_fail_closed():
    from copy import deepcopy
    from tests.current.population_support import contract
    from multi_dataset_diverse_rl.search.current_policy import CURRENT_POLICY_BUNDLE
    c=contract()
    CURRENT_POLICY_BUNDLE.validate_contract(c)
    from multi_dataset_diverse_rl import current_contract as identities
    from multi_dataset_diverse_rl.search.optimization_evidence import POLICY as EVIDENCE_POLICY
    method=CURRENT_POLICY_BUNDLE.method(aggregation=identities.EQUIVALENCE_PLURALITY_VERSION,
        provider_binding='0'*64,successful_provider_calls=10000,
        solver_trajectory_policy=c['solver_trajectory_policy'],optimization_evidence_policy=EVIDENCE_POLICY)
    assert method.target_policy == TargetPolicyV1.identity == identities.UNIFIED_TARGET_POLICY_VERSION
    stale=deepcopy(method)
    from dataclasses import replace
    stale=replace(stale,target_policy='responsibility_failure_discount_seeded_fallback_v2')
    with pytest.raises(SearchContractError):CURRENT_POLICY_BUNDLE.validate_method(stale)
    for name,value in [('responsibility_policy',None),
                       ('identity','MATH_VOTE_PARENT_POPULATION_BINDING_V12'),
                       ('benchmark_protocol_sha256','0'*64),
                       ('responsibility','binary_plurality_raw_responsibility_v1')]:
        bad=deepcopy(c);bad[name]=value
        with pytest.raises(SearchContractError):CURRENT_POLICY_BUNDLE.validate_contract(bad)
    bad=deepcopy(c);bad['responsibility_policy']['weights']['r4']=1.0
    with pytest.raises(SearchContractError):CURRENT_POLICY_BUNDLE.validate_contract(bad)


def test_r3_r4_reach_production_pattern_and_three_shared_independent_slots(tmp_path):
    # Errors only on team-correct rows; old runtime gave these members zero responsibility.
    def behavior(prompt,index,request):
        # Exact member lane is carried in cache, not Solver text; distinct initial
        # prompts make fixture behaviors reproducible without optimizer information.
        from multi_dataset_diverse_rl.search.system_prompt import SEED
        if 'lane zero' in prompt:return index>=2
        if 'lane one' in prompt:return index!=0
        return True
    from multi_dataset_diverse_rl.search.system_prompt import SEED
    prompts=[SEED.edit('role',f'Synthetic lane {name}.') for name in ('zero','one','two','three','four')]
    run,broker,requests,packets = graph(tmp_path,solver_behavior=behavior,prompts=prompts)
    events={}
    run.execution_observer=lambda stage,data:events.update({stage:data})
    result=asyncio.run(run.run(max_opportunities=1))
    op=events['OPPORTUNITY']['opportunity']
    assert op.target_member == 0
    assert op.diagnosis.responsibility[0].r3_count == 1
    assert op.diagnosis.responsibility[0].r4_count == 1
    assert op.objective['raw_responsibility'] == 1.5
    assert op.pattern_context['selected_pattern_responsibility'] == 1.5
    assert len(packets)==3 and len({str(p['selected_parent']) for p in packets})==1
    assert len({str(p['repair_direction']) for p in packets})==1
    assert len({str(p['repair_examples']) for p in packets})==1
    assert result.final_assembly['selection_repeated'] is False
