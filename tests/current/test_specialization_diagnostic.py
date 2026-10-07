from types import SimpleNamespace as NS
import pytest

from multi_dataset_diverse_rl.diagnostics.specialization import (
    FrozenTeamScorer, coverage, enumerate_existing_teams, nondominated)
from multi_dataset_diverse_rl.search.schemas import ParsedOutput


class Benchmark:
    capabilities=NS(binary_plurality_responsibility=True)
    def parse_member_output(self, raw, item):
        return ParsedOutput(raw, raw != 'INVALID')
    def score_member_output(self, parsed, reference):
        return int(parsed.valid and parsed.answer == reference)
    def equivalent(self, a, b):
        return a == b


def scorer(n=1):
    return FrozenTeamScorer(Benchmark(), [NS(item=NS(benchmark_id='math'), reference='G') for _ in range(n)])


def test_frozen_aggregation_abstention_remains_an_incorrect_observation():
    class InconsistentBenchmark(Benchmark):
        def equivalent(self,a,b):
            return a==b or {a,b} in ({'G','W'},{'W','X'})
    s=FrozenTeamScorer(InconsistentBenchmark(),[NS(item=NS(benchmark_id='math'),reference='G')])
    metrics,vote,oracle=s.team([['G'],['W'],['X'],['INVALID'],['INVALID']])
    assert not vote and oracle=={0}
    assert metrics['aggregation_abstention_counts']=={'EQUIVALENCE_RELATION_INCONSISTENT':1}
    assert metrics['mean_disagreement'] is None
    assert metrics['mean_distinct_valid_equivalence_classes'] is None


@pytest.mark.parametrize('votes,minimum', [
    (['G','W','W','W','W'],2),
    (['G','W','W','X','Y'],1),
    (['G','W','X','Y','Z'],1),
    (['G','W','W','X','X'],2),
    (['G','INVALID','INVALID','INVALID','INVALID'],0)])
def test_vote_conversion_uses_real_plurality_and_peer_replacements(votes, minimum):
    analysis=scorer().topology(['W',*votes[1:]],votes,0,0)
    assert analysis['minimum_additional_correct_votes_needed_to_flip']==minimum
    assert analysis['correct_vote_count']==1
    assert analysis['final_plurality_answer_correct']==(minimum==0)


def test_equal_top_counts_abstain_and_invalid_predictions_have_no_vote():
    s=scorer()
    aggregate=s.aggregate(['G','G','W','W','INVALID'],0)
    assert not aggregate.parsed_output.valid
    assert aggregate.diagnostics['tie_abstained']
    assert aggregate.diagnostics['invalid_abstentions']==1


def test_exact_coverage_loss_and_oracle_are_different_coordinates():
    result=coverage(range(22),[*range(18),22,23,24],range(22))
    assert result['retained_old_correct']==18
    assert result['lost_old_correct']==4
    assert result['newly_correct_vs_member']==result['novel_team_coverage']==3
    assert result['target_delta']==-1
    assert result['candidate_oracle']==25 and result['oracle_delta']==3
    assert result['gain_loss_ratio']==0.75
    assert coverage([0],[0,1],[0])['gain_loss_ratio']==float('inf')


def test_existing_compositions_preserve_member_slots_and_reuse_outputs():
    s=scorer(2);baseline=[['G','W'] for _ in range(5)]
    alternatives={1:[('A',['W','G']),('B',['G','G'])],3:[('C',['W','G'])]}
    result=enumerate_existing_teams(s,baseline,alternatives)
    assert len(result)==6
    assert all(row['member_choices'][0]==row['member_choices'][2]==row['member_choices'][4]=='BASELINE' for row in result)
    best=max(row['vote_correct_count'] for row in result)
    assert best==1
    assert max(row['oracle_correct_count'] for row in result)==2
    assert baseline==[['G','W'] for _ in range(5)]


def test_pareto_keeps_accuracy_coverage_tradeoffs_and_equal_points():
    points={'baseline':dict(accuracy=22,novel=0,invalid=0),
            'specialist':dict(accuracy=20,novel=3,invalid=1),
            'inferior':dict(accuracy=19,novel=2,invalid=2),
            'duplicate':dict(accuracy=20,novel=3,invalid=1)}
    result=nondominated(points,maximize=('accuracy','novel'),minimize=('invalid',))
    assert result['baseline']['nondominated'] and result['specialist']['nondominated']
    assert result['duplicate']['nondominated']
    assert result['inferior']['dominated_by']==['specialist','duplicate']
