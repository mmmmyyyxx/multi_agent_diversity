"""Audit safeguards and toy mechanism witnesses, not optimizer efficacy tests."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts.audit_local_evaluation_promotion import select_probe,select_local,selection_observations,draw_panel,vote_bounds_from_peers


def candidate(name,*,assigned=1,vote=0,risk=True,known=False):
    return dict(candidate_id=name,assigned_repairs=assigned,probe_vote_delta=vote,probe_target_delta=0,
        probe_risk_pass=risk,probe_collateral=0,probe_invalid_delta=0,full_evaluated=known,
        full_pass=False if known else None,full_vote_delta=-2 if known else None,
        validation_delta=0,validation_lost=0,mutation_delta=0,generation=int(name[-1]),
        validation_vote_delta=0,mutation_vote_delta=0)


def test_unknown_full_is_not_a_negative_or_potential_success():
    result=selection_observations([candidate('c1'),candidate('c2',known=True)])
    assert result['unknown_full']==1 and result['known_full']==1
    assert result['known_negative_vote']==1 and result['known_full_pass']==0


def test_selection_cannot_read_future_full_labels():
    rows=[candidate('c1',vote=1),candidate('c2'),candidate('c3',assigned=0,vote=3)]
    before=[[r['candidate_id'] for r in f(rows)] for f in (select_probe,select_local)]
    mutated=deepcopy(rows)
    for r in mutated:r.update(full_vote_delta=10**9,full_pass=True,full_evaluated=True)
    assert before==[[r['candidate_id'] for r in f(mutated)] for f in (select_probe,select_local)]


def test_probe_replay_matches_production_and_floor_removal_retains_risk_bound():
    from multi_dataset_diverse_rl.search.evaluation import FixedPeerPromotion
    from multi_dataset_diverse_rl.search.optimization_evidence import PROBE_POLICY
    rows=[candidate('c1'),candidate('c2',assigned=0,vote=1),candidate('c3',risk=False,vote=5)]
    fixtures=[]
    for r in rows:
        d=dict(probe_policy=PROBE_POLICY,assigned_repair_count=r['assigned_repairs'],risk_pass=r['probe_risk_pass'],
            collateral_loss=0,team_probe_metrics=SimpleNamespace(vote_delta=r['probe_vote_delta'],target_delta=0,invalid_delta=0))
        fixtures.append((SimpleNamespace(candidate_id=r['candidate_id']),SimpleNamespace(aggregation_diagnostics=d)))
    assert list(FixedPeerPromotion().select(fixtures))==[r['candidate_id'] for r in select_probe(rows)]==['c1']
    assert [r['candidate_id'] for r in select_probe(rows,assigned=False)]==['c2','c1']


def test_wrong_cluster_change_can_improve_vote_without_assigned_repair():
    # Two correct peers, two distinct wrong peers; target remains wrong. The
    # old target creates a tied wrong cluster, the new one breaks that tie.
    from collections import Counter
    peers=['correct','correct','wrong_a','wrong_b']
    before=Counter([*peers,'wrong_a']);after=Counter([*peers,'wrong_c'])
    wins=lambda counts:counts['correct']>max(n for k,n in counts.items() if k!='correct')
    assert not wins(before) and wins(after)
    assert select_probe([candidate('c1',assigned=0,vote=1)])==[]
    assert len(select_probe([candidate('c1',assigned=0,vote=1)],assigned=False))==1
    # Combinatorial witness only: no claim such a prompt exists or passes Shadow.


@pytest.mark.parametrize('stratified',[False,True])
def test_panel_uses_only_parent_features_and_preserves_nine_slots(stratified):
    parent={str(i):dict(vote_correct=i<12,plurality_margin=i%3,team_correctness=[i%2==0]*5) for i in range(30)}
    mutation=['0','1','2']
    a=draw_panel(parent,0,mutation,99,stratified=stratified)
    assert a==draw_panel(deepcopy(parent),0,mutation,99,stratified=stratified)
    assert len(a)==len({q for q,_ in a})==9 and not set(mutation)&{q for q,_ in a}
    assert all(w>0 for _,w in a)


def test_sparse_strata_redistribute_budget_without_dropping_population():
    parent={str(i):dict(vote_correct=True,plurality_margin=3,team_correctness=[True]*5) for i in range(20)}
    panel=draw_panel(parent,0,['0','1','2'],123,stratified=True)
    assert len(panel)==9 and all(w==17/9 for _,w in panel)


@pytest.mark.parametrize('correct,answers,bounds',[
    ([True,True,False,False,False],['g','g','a','b','a'],(0,1)),
    ([True,False,False,False,False],['g','a','a','a','b'],(0,0)),
    ([True,True,True,False,False],['g','g','g','a','b'],(1,1)),
    ([False]*5,['a','a','a','a','b'],(0,0)),
])
def test_partial_bounds_use_fixed_peers_without_imputing_target(correct,answers,bounds):
    row=dict(team_correctness=correct,team_answers=answers,team_validity=[True]*5)
    assert vote_bounds_from_peers(row,4)==bounds
