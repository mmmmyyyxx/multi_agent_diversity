"""Read-only fixed-state responsibility replay; emits no question or answer data."""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from math import sqrt
from pathlib import Path

from .repair_path import classes_from_record, repair_game

ARMS=('count','immediate','shapley','equal_failed')


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def check_inputs(root,manifest):
    for name,expected in manifest['input_sha256'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path)!=expected:
            raise ValueError('DIAGNOSTIC_INPUT_HASH_MISMATCH: '+name)


def ranks(values):
    ordered=sorted(range(len(values)),key=values.__getitem__);out=[0.0]*len(values);start=0
    while start<len(values):
        end=start+1
        while end<len(values) and values[ordered[end]]==values[ordered[start]]:end+=1
        for k in range(start,end):out[ordered[k]]=(start+end-1)/2
        start=end
    return out


def correlation(xs,ys):
    if len(xs)<3 or len(set(xs))<2 or len(set(ys))<2:return dict(n=len(xs),spearman=None)
    a,b=ranks(xs),ranks(ys);ma=sum(a)/len(a);mb=sum(b)/len(b)
    value=sum((x-ma)*(y-mb) for x,y in zip(a,b))/sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))
    return dict(n=len(xs),spearman=value)


def historical_replay(path):
    counters=Counter();structures=Counter();unique={};cache={};opportunities=[];candidates=[]
    current=None;pending_evaluation=False
    by_correct=defaultdict(Counter)
    for line in Path(path).open(encoding='utf-8'):
        event=json.loads(line)
        if event['stage']=='OPPORTUNITY':
            if current is not None:raise ValueError('UNFINISHED_OPPORTUNITY')
            op=event['opportunity'];parent=event['parent'];sid=parent['team_state_id']
            states=parent['diagnostics']['team_states']
            fingerprint=hashlib.sha256(json.dumps(states,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if sid in cache:
                if cache[sid][0]!=fingerprint:raise ValueError('SAME_STATE_CHANGED')
                games,raw,stats=cache[sid][1:]
            else:
                games={};raw={arm:[Fraction(0)]*5 for arm in ARMS};stats=Counter()
                for row in states:
                    q=row['question_hash']
                    if q in games:raise ValueError('DUPLICATE_OPTIMIZE_INPUT')
                    c=classes_from_record(row);g=repair_game(c)
                    m=len(g['wrong']);failed=1-g['values'][0]
                    g['equal_failed']=tuple(Fraction(failed,m) if i in g['wrong'] else Fraction(0) for i in range(5))
                    games[q]=g;stats['rows']+=1;stats['failed_votes']+=failed
                    stats['errors']+=m;stats['zero_shapley_errors']+=m if not failed else 0
                    stats['invalid_members']+=sum(x is None for x in c)
                    stats['all_wrong_rows']+=m==5
                    stats['all_wrong_nonuniform']+=m==5 and len(set(g['shapley']))>1
                    stats['immediate_zero_positive_shapley_errors']+=sum(a==0 and b>0 for a,b in zip(g['immediate'],g['shapley']))
                    stats['count_mass_on_correct_vote']+=sum(g['count']) if not failed else 0
                    for arm in ARMS:
                        for member in range(5):raw[arm][member]+=g[arm][member]
                    if (sid,q) not in unique:
                        unique[sid,q]=(c,g)
                if sum(raw['shapley'])!=stats['failed_votes']:raise ValueError('EFFICIENCY_FAILURE')
                cache[sid]=(fingerprint,games,raw,stats)
            counters.update(stats);counters['opportunities']+=1
            failures={int(k):v for k,v in op['evaluation_plan']['allocation_failure_counts'].items()}
            eligible=sorted(int(k) for k in op['objective']['target_scores'])
            if not eligible or any(type(failures[k]) is not int or failures[k]<0 for k in eligible):
                raise ValueError('INVALID_FAILURES_OR_ELIGIBILITY')
            if set(eligible)!={i for i in range(5) if raw['count'][i]>0}:raise ValueError('INCOMPLETE_MEMBER_ELIGIBILITY')
            scores={a:{m:raw[a][m]/(1+failures[m]) for m in eligible} for a in ARMS}
            winners={};topsets={}
            for a in ARMS:
                top=max(scores[a].values());topsets[a]=[m+1 for m in eligible if scores[a][m]==top]
                winners[a]=min(topsets[a]) if top>0 else None
            target=op['target_member'];patterns=op['pattern_context']['patterns'];pattern_rows=[];seen=set()
            for p in patterns:
                ids=p['support_ids']
                if len(ids)!=len(set(ids)) or seen.intersection(ids):raise ValueError('PATTERN_SUPPORT_NOT_PARTITION')
                if any(q not in games or target not in games[q]['wrong'] for q in ids):raise ValueError('PATTERN_SUPPORT_NOT_TARGET_ERROR')
                seen.update(ids)
                pattern_rows.append(dict(pattern_id=p['pattern_id'],support_count=len(ids),
                    scores={a:float(sum((games[q][a][target] for q in ids),Fraction(0))) for a in ARMS}))
            pw={};pt={}
            for a in ARMS:
                top=max((p['scores'][a] for p in pattern_rows),default=0)
                pt[a]=sorted(p['pattern_id'] for p in pattern_rows if p['scores'][a]==top)
                pw[a]=pt[a][0] if top>0 else None
            row=dict(opportunity=counters['opportunities'],historical_target=target+1,
                prior_failures={str(m+1):failures[m] for m in eligible},
                who=winners,who_top_sets=topsets,
                who_raw_scores={a:[float(v) for v in raw[a]] for a in ARMS},
                who_discounted_scores={a:{str(m+1):float(v) for m,v in scores[a].items()} for a in ARMS},
                pattern=pw,pattern_top_sets=pt,patterns=pattern_rows,
                historical_pattern=op['pattern_context']['focus_mechanism_id'])
            opportunities.append(row)
            current=(op,raw,row);pending_evaluation=True
        elif event['stage']=='EVALUATION':
            if current is None or not pending_evaluation or event['opportunity_id']!=current[0]['opportunity_id']:
                raise ValueError('EVALUATION_JOIN_MISMATCH')
            op,raw,row=current;target=op['target_member'];pending_evaluation=False
            for candidate in event['evaluated']:
                counters['exported_candidates']+=1
                if candidate.get('full') is None:continue
                fd=candidate['full']['aggregation_diagnostics'];effect=fd['edit_effect']
                fixed=effect['fixed_ids'];broken=effect['broken_ids']
                if len(fixed)!=len(set(fixed)) or set(fixed)&set(broken):raise ValueError('INVALID_EDIT_EFFECT')
                games=cache[op['parent_state_id']][1]
                if any(q not in games or target not in games[q]['wrong'] for q in fixed):raise ValueError('REPAIR_JOIN_MISMATCH')
                if any(q not in games or target in games[q]['wrong'] for q in broken):raise ValueError('LOSS_JOIN_MISMATCH')
                if len(fixed)-len(broken)!=fd['target_score_delta']:raise ValueError('TARGET_DELTA_MISMATCH')
                if fd['team_newly_fixed_count']-fd['team_newly_broken_count']!=fd['team_score_delta']:raise ValueError('VOTE_DELTA_MISMATCH')
                pattern=next(p for p in row['patterns'] if p['pattern_id']==row['historical_pattern'])
                candidates.append(dict(opportunity=row['opportunity'],member=target+1,
                    parent_ordinal=list(cache).index(op['parent_state_id'])+1,
                    target_delta=fd['target_score_delta'],vote_delta=fd['team_score_delta'],
                    vote_fixed=fd['team_newly_fixed_count'],vote_lost=fd['team_newly_broken_count'],
                    preedit_raw={a:float(raw[a][target]) for a in ARMS},
                    preedit_discounted={a:row['who_discounted_scores'][a][str(target+1)] for a in ARMS},
                    preedit_pattern=pattern['scores'],
                    repaired_error_mass={a:float(sum((games[q][a][target] for q in fixed),Fraction(0))) for a in ARMS},
                    repaired_errors=len(fixed),lost_correct=len(broken),
                    repairs_on_already_correct_vote=sum(games[q]['values'][0] for q in fixed)))
                counters['full_candidates']+=1
        elif event['stage']=='TRANSITION':
            if current is None or pending_evaluation or event['opportunity_id']!=current[0]['opportunity_id']:
                raise ValueError('TRANSITION_JOIN_MISMATCH')
            current=None
        else:raise ValueError('UNEXPECTED_EVENT_STAGE')
    if current is not None:raise ValueError('TRUNCATED_TRAJECTORY')
    counters['unique_parent_states']=len(cache);counters['unique_state_question_rows']=len(unique)
    for c,g in unique.values():
        key=f"correct_{c.count(0)}__vote_{g['values'][0]}"
        by_correct[key]['rows']+=1;by_correct[key]['errors']+=len(g['wrong'])
        by_correct[key]['count_mass']+=sum(g['count']);by_correct[key]['shapley_mass']+=sum(g['shapley'])
        # Only anonymous multiplicities and invalid count are retained.
        histogram=Counter(x for x in c if x not in (None,0))
        key=f"gold={c.count(0)},wrong_sizes={sorted(histogram.values(),reverse=True)},invalid={c.count(None)}"
        structures[key]+=1
    comparison={}
    for arm in ARMS[1:]:
        comparison[arm]=dict(who_changed_vs_count=sum(r['who'][arm]!=r['who']['count'] for r in opportunities),
            who_zero_fallback_unresolved=sum(r['who'][arm] is None for r in opportunities),
            who_ties=sum(len(r['who_top_sets'][arm])>1 for r in opportunities),
            pattern_changed_vs_count=sum(r['pattern'][arm]!=r['pattern']['count'] for r in opportunities),
            pattern_zero_fallback_unresolved=sum(r['pattern'][arm] is None for r in opportunities),
            pattern_ties=sum(len(r['pattern_top_sets'][arm])>1 for r in opportunities))
        comparison[arm]['who_disjoint_positive_top_sets_vs_count']=sum(
            r['who'][arm] is not None and not set(r['who_top_sets'][arm]) & set(r['who_top_sets']['count']) for r in opportunities)
        comparison[arm]['pattern_disjoint_positive_top_sets_vs_count']=sum(
            r['pattern'][arm] is not None and not set(r['pattern_top_sets'][arm]) & set(r['pattern_top_sets']['count']) for r in opportunities)
    # The primary unit is an opportunity, not its multiple selected candidates.
    grouped=defaultdict(list)
    for c in candidates:grouped[c['opportunity']].append(c)
    grouped_rows=[]
    for oi,cs in grouped.items():
        grouped_rows.append(dict(opportunity=oi,candidate_count=len(cs),
            vote_delta=sum(c['vote_delta'] for c in cs)/len(cs),
            target_delta=sum(c['target_delta'] for c in cs)/len(cs),
            preedit_raw=cs[0]['preedit_raw'],preedit_discounted=cs[0]['preedit_discounted'],
            preedit_pattern=cs[0]['preedit_pattern']))
    correlations={}
    for label,data in [('primary_opportunity_mean',grouped_rows),('descriptive_candidate_rows_nonindependent',candidates)]:
        correlations[label]={}
        for field in ('preedit_raw','preedit_discounted','preedit_pattern'):
            for outcome in ('vote_delta','target_delta'):
                correlations[label][field+'__'+outcome]={a:correlation([c[field][a] for c in data],[c[outcome] for c in data]) for a in ARMS}
    # Repeated opportunities at the same parent/target are correlated as well.
    # This sensitivity calculation is descriptive, with no independent-sample claim.
    contexts=defaultdict(list)
    for c in candidates:contexts[c['parent_ordinal'],c['member']].append(c)
    context_rows=[]
    for (parent,member),cs in contexts.items():
        context_rows.append(dict(parent_ordinal=parent,member=member,candidate_count=len(cs),
            vote_delta=sum(c['vote_delta'] for c in cs)/len(cs),
            preedit_raw=cs[0]['preedit_raw']))
    correlations['parent_target_mean_sensitivity']={a:correlation(
        [c['preedit_raw'][a] for c in context_rows],[c['vote_delta'] for c in context_rows]) for a in ARMS}
    return dict(counters={k:float(v) if isinstance(v,Fraction) else v for k,v in counters.items()},
        selection_comparison=comparison,
        unique_state_correctness_strata={k:{a:float(b) for a,b in v.items()} for k,v in by_correct.items()},
        anonymous_structure_counts=dict(structures),opportunities=opportunities,
        full_candidates=candidates,opportunity_mean_candidates=grouped_rows,
        candidate_correlations=correlations,
        full_candidate_selected_target_contexts=len(contexts),parent_target_mean_candidates=context_rows,
        limits=['Fixed V2.7 parents, observed failures/eligible members and original-target Pattern partitions.',
            'No alternative member, Pattern, candidate generation, search trajectory or held-out estimate.',
            'Candidate evaluations are selected by historical export/Probe/Full gates; missing Full outcomes are unknown.',
            'Correlations are retrospective associations under a historical selector, not out-of-sample predictions.',
            'equal_failed is a diagnostic control: equal 1/m on failed votes; zero on correct votes.',
            'None selection means all-zero fallback requires a separately frozen policy; no arbitrary winner invented.',
            'No feasibility-aware editing distribution is estimated from these selected candidates.'])
