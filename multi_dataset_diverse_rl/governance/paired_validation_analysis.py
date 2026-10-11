"""Frozen paired, example-level inference; no search read point."""
from math import comb
import random
from ..search.schemas import SearchContractError


def measure(benchmark,aggregation,item,outputs,gold):
    parsed=tuple(benchmark.parse_member_output(value,item) for value in outputs)
    correct=tuple(int(benchmark.score_member_output(value,gold)==1) for value in parsed)
    aggregated=aggregation.aggregate_sync(item=item,member_outputs=outputs,benchmark=benchmark)
    return dict(vote=int(benchmark.score_member_output(aggregated.parsed_output,gold)==1),
        oracle=int(any(correct)),members=list(correct),validity=[int(value.valid) for value in parsed])


def analyze(measurements):
    from .postfreeze_validation import POLICY
    rows=tuple(measurements);n=len(rows)
    if n!=POLICY['count']:raise SearchContractError('PAIRED_VALIDATION_COMPLETE_SUPPORT_REQUIRED')
    for pair in rows:
        if len(pair)!=2:raise SearchContractError('PAIRED_VALIDATION_MEASUREMENT_INVALID')
        for value in pair:
            if (set(value)!={'vote','oracle','members','validity'} or
                    any(type(v) is not int or v not in {0,1} for v in (value['vote'],value['oracle'],*value['members'],*value['validity']))
                    or len(value['members'])!=5 or len(value['validity'])!=5):
                raise SearchContractError('PAIRED_VALIDATION_MEASUREMENT_INVALID')
    deltas=[b['vote']-a['vote'] for a,b in rows]
    gains=sum(d==1 for d in deltas);losses=sum(d==-1 for d in deltas);discordant=gains+losses
    p=min(1.,2*sum(comb(discordant,k) for k in range(min(gains,losses)+1))/2**discordant) if discordant else 1.
    rng=random.Random(POLICY['analysis']['bootstrap_seed'])
    draws=sorted(sum(deltas[rng.randrange(n)] for _ in range(n))/n
        for _ in range(POLICY['analysis']['bootstrap_resamples']))
    def quantile(q):
        index=(len(draws)-1)*q;lo=int(index);hi=min(lo+1,len(draws)-1)
        return draws[lo]+(draws[hi]-draws[lo])*(index-lo)
    def team(a):
        values=[pair[a] for pair in rows]
        return dict(vote_correct=sum(v['vote'] for v in values),oracle_correct=sum(v['oracle'] for v in values),
            member_correct=[sum(v['members'][m] for v in values) for m in range(5)],
            member_valid=[sum(v['validity'][m] for v in values) for m in range(5)])
    before,after=team(0),team(1)
    return dict(count=n,initial=before,final=after,VoteAccDelta=(gains-losses)/n,
        OracleAccDelta=(after['oracle_correct']-before['oracle_correct'])/n,
        paired_vote_gains=gains,paired_vote_losses=losses,exact_mcnemar_two_sided_p=p,
        paired_bootstrap_vote_delta_interval_95=[quantile(.025),quantile(.975)],
        inference_scope=POLICY['analysis']['scope'])
