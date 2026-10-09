"""Current V2.4 checks ported from preserved replay tests; zero real APIs."""
import asyncio,json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from tests.current.test_structured_optimization_evidence import contract

from multi_dataset_diverse_rl.search.target_or_team_transition import InitialCompetenceTargetOrTeamProgressV3,progress_path
from multi_dataset_diverse_rl.search.schemas import EvaluatedCandidate,SearchCandidate,TeamEvaluation
def measurement(target=22, vote=22, invalid_delta=0, **metrics):
    return TeamEvaluation(vote, None, (target, 22, 22, 22, 22),
        aggregation_diagnostics={'terminal_invalid_delta': invalid_delta, **metrics})


def policy():
    result = InitialCompetenceTargetOrTeamProgressV3()
    result.bind_initial((22,) * 5, 'initial')
    return result


def row(cid, target, vote=22, soft=0, local=0):
    return EvaluatedCandidate(SearchCandidate(cid, __import__('multi_dataset_diverse_rl.search.system_prompt',fromlist=['SEED']).SEED.edit('strategy','Check signs. '+cid), search_score=local),
        None, measurement(target, vote, mean_soft_vote_utility=soft), True, False, {'target_member': 0})


@pytest.mark.parametrize('target,vote,invalid,expected', [
    (25, 22, 0, True), (22, 23, 0, True), (25, 23, 0, True),
    (21, 23, 0, False), (25, 21, 0, False), (22, 22, 0, False),
    (25, 23, 1, False),
])
def test_transition_reachability_and_safety(target, vote, invalid, expected):
    assert policy().allows(measurement(), measurement(target, vote, invalid), 0) is expected


def test_team_progress_can_lower_incumbent_above_immutable_floor():
    p = policy()
    assert p.allows(measurement(25), measurement(23, 23), 0)
    assert not p.allows(measurement(25), measurement(21, 23), 0)
    with pytest.raises(SearchContractError, match='CANNOT_REBASE'):
        p.bind_initial((25,) * 5, 'child')
    assert p.initial_scores == (22,) * 5 and p.initial_state_id == 'initial'


def test_invalid_prediction_policy_keeps_existing_frozen_behavior():
    p = InitialCompetenceTargetOrTeamProgressV3(invalid_predictions_are_incorrect=True)
    p.bind_initial((22,) * 5, 'initial')
    # The frozen MATH validity policy scores terminal-invalid as incorrect;
    # it already makes invalidity telemetry observation-only in V2.1.
    assert p.allows(measurement(), measurement(25, 22, 1), 0)
    with pytest.raises(SearchContractError, match='GUARD_NOT_MEASURED'):
        p.allows(measurement(), replace(measurement(25), aggregation_diagnostics={}), 0)
    with pytest.raises(SearchContractError, match='NONFINITE'):
        p.allows(measurement(float('nan')), measurement(25), 0)


def test_winner_team_first_target_second_not_local_or_soft_or_input_order():
    a, b = row('A', 23, soft=999, local=999), row('B', 25, soft=0, local=-999)
    for candidates in ((a, b), (b, a)):
        assert policy().select(measurement(), candidates).candidate == b
    c = row('C', 22, vote=23, soft=-1000)
    assert policy().select(measurement(), (b, c)).candidate == c


@pytest.mark.parametrize('team,target,path', [(1, -2, 'TEAM'), (0, 3, 'TARGET'),
    (1, 3, 'TARGET_AND_TEAM'), (0, 0, 'NONE')])
def test_progress_telemetry_paths(team, target, path):
    assert progress_path(team, target) == path


def test_target_commit_resets_scientific_patience_without_using_team_delta():
    from multi_dataset_diverse_rl.search.policies import GlobalStopPolicy
    stop = GlobalStopPolicy(2)
    assert stop.observe_opportunity(parent_state_id='parent', eligible_members=(0,),
        selected_member=0, local_update=False, committed=False) is None
    assert stop.no_commit_epochs == 1
    assert stop.observe_opportunity(parent_state_id='parent', eligible_members=(0,),
        selected_member=0, local_update=False, committed=True) is None
    assert stop.no_commit_epochs == 0
    assert stop.observe_opportunity(parent_state_id='child', eligible_members=(0,),
        selected_member=0, local_update=False, committed=False) is None
    assert stop.observe_opportunity(parent_state_id='child', eligible_members=(0,),
        selected_member=0, local_update=False, committed=False) == 'SATURATION_REACHED'


def test_target_only_teamprobe_signal_reaches_full_with_ceiling_two():
    from multi_dataset_diverse_rl.search.evaluation import CandidateEvaluationPipeline, FixedPeerPromotion
    from multi_dataset_diverse_rl.search.schemas import SearchResult
    from multi_dataset_diverse_rl.team_search.schemas import TeamMiniBatchMetrics
    calls = []
    from multi_dataset_diverse_rl.search.optimization_evidence import PROBE_POLICY
    class Provider:
        async def team_probe(self, opportunity, candidate):
            return replace(measurement(), aggregation_diagnostics={'probe_policy':PROBE_POLICY,
                'assigned_repair_count':1,'risk_pass':True,'collateral_loss':0,'team_probe_metrics':
                TeamMiniBatchMetrics(target_delta=1, vote_delta=0, team_net_vote_delta=0,
                    responsibility_delta=1, broad_delta=1, invalid_delta=0)})
        async def full(self, opportunity, candidate):
            calls.append(candidate.candidate_id)
            return measurement(25)
    candidates = tuple(SearchCandidate(str(i), 'Check constraints. ' + str(i)) for i in range(4))
    result = asyncio.run(CandidateEvaluationPipeline(Provider(), FixedPeerPromotion()).evaluate(
        NS(target_member=0, evaluation_plan={}), SearchResult(candidates, 'frozen_fake')))
    assert len(calls) == sum(r.promoted for r in result) == 2
    assert policy().select(measurement(), result).candidate is not None
