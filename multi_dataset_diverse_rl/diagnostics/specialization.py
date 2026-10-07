"""Optimize-only post-hoc analysis of already realized five-member outputs."""
from itertools import combinations, product
from math import inf
from collections import Counter

from ..search.scientific_aggregation import EquivalencePluralityAggregation


def coverage(initial_correct, candidate_correct, initial_oracle):
    """Exact sets, with no smoothing or rebaselining."""
    before, after, oracle = map(set, (initial_correct, candidate_correct, initial_oracle))
    retained, lost, new, novel = before & after, before - after, after - before, after - oracle
    delta = len(after) - len(before)
    assert delta == len(new) - len(lost)
    return dict(initial_target_correct=len(before), candidate_target_correct=len(after),
        retained_old_correct=len(retained), lost_old_correct=len(lost),
        newly_correct_vs_member=len(new), novel_team_coverage=len(novel), target_delta=delta,
        initial_oracle=len(oracle), candidate_oracle=len(oracle | after), oracle_delta=len(after - oracle),
        gain_loss_ratio=len(novel) / len(lost) if lost else inf)


def nondominated(points, *, maximize, minimize):
    """Return descriptive Pareto membership and all strict dominators."""
    def dominates(a, b):
        weak = all(a[k] >= b[k] for k in maximize) and all(a[k] <= b[k] for k in minimize)
        strict = any(a[k] > b[k] for k in maximize) or any(a[k] < b[k] for k in minimize)
        return weak and strict
    return {key: dict(nondominated=not (winners := [other for other in points
             if other != key and dominates(points[other], point)]), dominated_by=winners)
            for key, point in points.items()}


class FrozenTeamScorer:
    """Use the real benchmark and equal-weight aggregator, caching observations only."""
    def __init__(self, benchmark, examples):
        if not examples:
            raise ValueError('Diagnostic requires a nonempty frozen Optimize membership')
        self.benchmark, self.examples = benchmark, tuple(examples)
        self.aggregator = EquivalencePluralityAggregation()
        self._profile_cache = {}

    def profile(self, outputs):
        import json
        key = json.dumps(outputs, sort_keys=True, separators=(',', ':'))
        if key not in self._profile_cache:
            if len(outputs) != len(self.examples):
                raise ValueError('Incomplete member output profile')
            parsed = [self.benchmark.parse_member_output(raw, example.item)
                      for raw, example in zip(outputs, self.examples, strict=True)]
            correct = {i for i, (p, example) in enumerate(zip(parsed, self.examples, strict=True))
                       if self.benchmark.score_member_output(p, example.reference) == 1}
            self._profile_cache[key] = dict(correct=correct, invalid={i for i, p in enumerate(parsed) if not p.valid})
        return self._profile_cache[key]

    def aggregate(self, outputs, index):
        return self.aggregator.aggregate_sync(item=self.examples[index].item,
            member_outputs=tuple(outputs), benchmark=self.benchmark)

    def is_correct(self, aggregate, index):
        return self.benchmark.score_member_output(aggregate.parsed_output,
                                                  self.examples[index].reference) == 1

    def team(self, profiles):
        if len(profiles) != 5:
            raise ValueError('Diagnostic teams require exactly five equal members')
        observations = [self.profile(row) for row in profiles]
        vote, disagreement, eligible_rows, distinct = set(), 0, 0, 0
        abstentions=Counter()
        for index in range(len(self.examples)):
            aggregate = self.aggregate([row[index] for row in profiles], index)
            if self.is_correct(aggregate, index):
                vote.add(index)
            groups = aggregate.diagnostics.get('equivalence_classes', ())
            if aggregate.diagnostics.get('abstention_reason'):
                abstentions[aggregate.diagnostics['abstention_reason']]+=1
                continue
            valid = sum(map(len, groups))
            distinct += len(groups)
            if valid >= 2:
                pairs = valid * (valid - 1) // 2
                same = sum(len(group) * (len(group) - 1) // 2 for group in groups)
                disagreement += (pairs - same) / pairs
                eligible_rows += 1
        n = len(self.examples)
        counts = [len(row['correct']) for row in observations]
        oracle = set().union(*(row['correct'] for row in observations))
        return dict(vote_correct_count=len(vote), oracle_correct_count=len(oracle),
            VoteAcc=len(vote)/n, OracleAcc=len(oracle)/n, MeanMemberAcc=sum(counts)/(5*n),
            MinMemberAcc=min(counts)/n, member_correct_counts=counts,
            invalid_count_by_member=[len(row['invalid']) for row in observations],
            mean_disagreement=disagreement/eligible_rows if eligible_rows else None,
            disagreement_definition='Mean unequal-equivalence fraction over valid member pairs; rows with fewer than two valid outputs excluded.',
            aggregation_abstention_counts=dict(abstentions),
            equivalence_statistics_excluded_rows=sum(abstentions.values()),
            mean_distinct_valid_equivalence_classes=(distinct/(n-sum(abstentions.values()))
                if n>sum(abstentions.values()) else None)), vote, oracle

    def topology(self, old_outputs, candidate_outputs, target, index):
        old = self.aggregate(old_outputs, index)
        new = self.aggregate(candidate_outputs, index)
        gold = self.examples[index].reference
        groups = new.diagnostics.get('equivalence_classes', ())
        if new.diagnostics.get('abstention_reason'):
            raise ValueError('Frozen equivalence relation failed')
        correct_groups, wrong_groups = [], []
        for group in groups:
            parsed = self.benchmark.parse_member_output(candidate_outputs[group[0]], self.examples[index].item)
            (correct_groups if self.benchmark.score_member_output(parsed, gold) == 1 else wrong_groups).append(group)
        correct_votes = sum(map(len, correct_groups))
        if not correct_votes:
            raise ValueError('Novel-coverage topology requires a correct candidate vote')
        minimum = None
        peers = [member for member in range(5) if member != target]
        for k in range(5):
            for subset in combinations(peers, k):
                hypothetical = list(candidate_outputs)
                for member in subset:
                    hypothetical[member] = candidate_outputs[target]
                if self.is_correct(self.aggregate(hypothetical, index), index):
                    minimum = k
                    break
            if minimum is not None:
                break
        assert minimum is not None
        before, after = old.parsed_output, new.parsed_output
        changed = before.valid != after.valid or (before.valid and after.valid
            and not self.benchmark.equivalent(before.answer, after.answer))
        return dict(correct_vote_count=correct_votes,
            largest_wrong_vote_count=max(map(len, wrong_groups), default=0),
            number_of_distinct_wrong_answers=len(wrong_groups),
            class_size_topology=sorted(map(len, groups), reverse=True),
            tie_structure=bool(new.diagnostics['tie_abstained']),
            invalid_vote_count=new.diagnostics['invalid_abstentions'],
            final_plurality_answer_correct=self.is_correct(new, index),
            candidate_added_correct_vote_changed_team_decision=bool(changed),
            candidate_added_correct_vote_changed_team_correctness=self.is_correct(old, index) != self.is_correct(new, index),
            minimum_additional_correct_votes_needed_to_flip=minimum,
            additional_votes_interpretation='Minimum peer-vote replacements over all subsets of the four fixed peers; team size remains five.')


def enumerate_existing_teams(scorer, baseline, candidate_profiles):
    """Enumerate exactly the existing alternatives for every actual member slot."""
    choices = [[('BASELINE', baseline[member]), *candidate_profiles.get(member, [])]
               for member in range(5)]
    result = []
    for labels_and_profiles in product(*choices):
        labels, profiles = zip(*labels_and_profiles, strict=True)
        metrics, _, _ = scorer.team(profiles)
        result.append(dict(member_choices=list(labels), **metrics))
    return result
