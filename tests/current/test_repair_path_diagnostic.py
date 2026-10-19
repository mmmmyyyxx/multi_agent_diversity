from fractions import Fraction as F
from types import SimpleNamespace
import pytest
from multi_dataset_diverse_rl.diagnostics.repair_path import (
    repair_game,exhaustive_audit,classes_from_record,vote_correct)
from multi_dataset_diverse_rl.search.scientific_aggregation import EquivalencePluralityAggregation
from multi_dataset_diverse_rl.search.schemas import ParsedOutput


@pytest.mark.parametrize('state,expected',[
    ((1,1,1,1,1),(F(1,5),)*5),
    ((0,0,1,1,1),(0,0,F(1,3),F(1,3),F(1,3))),
    ((1,1,2,3,4),(F(3,10),F(3,10),F(2,15),F(2,15),F(2,15))),
    ((0,1,1,2,3),(0,F(5,12),F(5,12),F(1,12),F(1,12))),
    ((0,0,1,2,3),(0,)*5),
    ((None,)*5,(F(1,5),)*5),
])
def test_supplied_examples_and_invalid_repair(state,expected):
    assert repair_game(state)['shapley']==expected


def test_all_five_member_structures():
    assert exhaustive_audit()['labeled_states']==16807


def test_vote_matches_production_with_ties_and_invalids():
    from itertools import product
    benchmark=SimpleNamespace(capabilities=SimpleNamespace(binary_plurality_responsibility=True),
        parse_member_output=lambda raw,item:ParsedOutput(raw,raw!=''),equivalent=lambda a,b:a==b)
    agg=EquivalencePluralityAggregation()
    for c in product((0,1,2,3,4,5,None),repeat=5):
        result=agg.aggregate_sync(item=None,member_outputs=tuple('' if x is None else str(x) for x in c),benchmark=benchmark)
        actual=result.parsed_output.valid and result.parsed_output.answer=='0'
        assert actual==bool(vote_correct(c))


def test_fail_closed_and_no_silent_invalid_votes():
    for bad in ((0,)*4,(True,0,0,0,0),(-1,0,0,0,0)):
        with pytest.raises(ValueError):repair_game(bad)
    row=dict(team_answers=['g','a','a','b','c'],team_validity=[True]*5,
        team_correctness=[True,False,False,False,False],vote_correct=False)
    assert classes_from_record(row)==(0,1,1,2,3)
    with pytest.raises(ValueError):classes_from_record({**row,'vote_correct':True})
    with pytest.raises(ValueError):classes_from_record({**row,'team_correctness':[True,True,False,False,False]})
    with pytest.raises(ValueError):classes_from_record({**row,'team_validity':[False,True,True,True,True]})


def test_correlation_ties_and_input_hash_guard(tmp_path):
    from multi_dataset_diverse_rl.diagnostics.repair_path_replay import correlation,check_inputs,sha256
    assert correlation([1,1,2],[2,2,3])['spearman']==1
    assert correlation([1,1,1],[1,2,3])['spearman'] is None
    path=tmp_path/'evidence';path.write_text('frozen')
    manifest={'input_sha256':{'evidence':sha256(path)}}
    check_inputs(tmp_path,manifest)
    path.write_text('changed')
    with pytest.raises(ValueError,match='HASH_MISMATCH'):check_inputs(tmp_path,manifest)


def test_replay_rejects_incomplete_or_misjoined_events(tmp_path):
    import json
    from multi_dataset_diverse_rl.diagnostics.repair_path_replay import historical_replay
    path=tmp_path/'trace'
    path.write_text(json.dumps({'stage':'EVALUATION','opportunity_id':'orphan'})+'\n')
    with pytest.raises(ValueError,match='JOIN_MISMATCH'):historical_replay(path)
