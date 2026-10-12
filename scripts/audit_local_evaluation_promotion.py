"""Read-only V2.7 candidate catalog diagnosis; never a search or provider runner.

Writes counters/hashes only. Missing Full labels stay missing. Alternative ranks
use pre-Full measurements; Full labels only assess fixed-catalog identifiability.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter,defaultdict
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess

FROZEN='a8985eef7f15f82ae29bf8000cccf12b0ad39b2a'
CURRENT='63796dc815f8ce1195cd5a3950c2ad7e1ad59d5e'


def digest_file(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        while chunk:=source.read(4*1024*1024):h.update(chunk)
    return h.hexdigest()


def select_probe(records,*,assigned=True):
    eligible=[r for r in records if r['probe_risk_pass'] and (not assigned or r['assigned_repairs']>=1)]
    return sorted(eligible,key=lambda r:(r['probe_vote_delta'],r['probe_target_delta'],
        -r['probe_collateral'], -r['probe_invalid_delta'],r['candidate_id']),reverse=True)[:2]


def select_local(records,*,team=False):
    def key(r):
        old=(-r['validation_delta'],r['validation_lost'],-r['mutation_delta'],r['generation'])
        return (-r['validation_vote_delta'],-r['mutation_vote_delta'],*old) if team else old
    return sorted(records,key=key)[:4]


def selection_observations(selected):
    # Unknown != bad, safe, possible success, or zero. No extrapolated run score.
    known=[r for r in selected if r['full_evaluated']]
    return dict(selected=len(selected),known_full=len(known),unknown_full=len(selected)-len(known),
        known_full_pass=sum(r['full_pass'] for r in known),
        known_negative_vote=sum(r['full_vote_delta']<0 for r in known),
        known_nonnegative_vote=sum(r['full_vote_delta']>=0 for r in known))


def vote_bounds_from_peers(row,target):
    """Bounds over any unknown target response, fixed four peer realizations."""
    peers=[j for j in range(5) if j!=target]
    gold=sum(row['team_correctness'][j] for j in peers)
    wrong=Counter(row['team_answers'][j] for j in peers if row['team_validity'][j] and not row['team_correctness'][j])
    maximum_wrong=max(wrong.values(),default=0)
    return int(gold>maximum_wrong+1),int(gold+1>maximum_wrong)


def draw_panel(parent,member,mutation,seed,*,stratified):
    """Nine slots; strata use original parent only, never candidate outcomes."""
    rng=random.Random(seed);pool=sorted(set(parent)-set(mutation))
    if not stratified:return [(q,len(pool)/9) for q in rng.sample(pool,9)]
    groups=[[],[],[]]
    for q in pool:
        p=parent[q]
        lane=0 if p['vote_correct'] and p['plurality_margin']<=2 else 1 if not p['vote_correct'] and not p['team_correctness'][member] else 2
        groups[lane].append(q)
    sizes=[min(3,len(g)) for g in groups]
    while sum(sizes)<9:
        for i,g in enumerate(groups):
            if sizes[i]<len(g) and sum(sizes)<9:sizes[i]+=1
    result=[]
    for g,n in zip(groups,sizes,strict=True):
        if n:result.extend((q,len(g)/n) for q in rng.sample(g,n))
    assert len(result)==9
    return result


def audit(root,out,repetitions=500):
    from multi_dataset_diverse_rl.benchmarks.math_domain_v2 import domain_matrix
    root=Path(root).resolve();out=Path(out).resolve()
    assert out.is_relative_to(root/'runs'), 'Private audit output must stay in ignored runs/'
    out.mkdir(parents=True,exist_ok=True)
    run=root/'runs/a4_v27_seed86_continuation_attempt1/execution'
    load=lambda p:json.loads(p.read_bytes())
    baseline=load(root/'reports/math_v27_seed86_performance_forensic_20261012/audit_summary.json')
    from multi_dataset_diverse_rl.governance.source_identity import build_unified_source_identity
    current_hash=build_unified_source_identity(root)['scientific_source_hash']
    assert current_hash==load(root/'reports/scoped_edit_experience_layer1_v28_preparation_20261012/verification.json')['current_scientific_source_hash']
    trajectory_sha=digest_file(run/'trajectory_private.jsonl')
    assert trajectory_sha==baseline['provenance']['trajectory_sha256']
    source_checks=[]
    for p in ['multi_dataset_diverse_rl/search/selected_evidence.py',
        'multi_dataset_diverse_rl/search/evaluation.py','multi_dataset_diverse_rl/search/binary_runtime.py',
        'multi_dataset_diverse_rl/benchmarks/math_domain_v2.py']:
        historical=subprocess.check_output(['git','show',FROZEN+':'+p],cwd=root).replace(b'\r\n',b'\n')
        current=(root/p).read_bytes().replace(b'\r\n',b'\n')
        assert historical==current,p
        source_checks.append(dict(path=p,lf_sha256=hashlib.sha256(current).hexdigest(),frozen_current_equal=True))
    p='multi_dataset_diverse_rl/search/bounded_layer1.py'
    get_rank=lambda body:ast.dump(next(n for n in ast.walk(ast.parse(body)) if isinstance(n,ast.FunctionDef) and n.name=='rank'))
    assert get_rank(subprocess.check_output(['git','show',FROZEN+':'+p],cwd=root).decode())==get_rank((root/p).read_text(encoding='utf-8-sig'))
    source_checks.append(dict(path=p,local_rank_ast_equal=True))
    for body in (subprocess.check_output(['git','show',FROZEN+':'+p],cwd=root).decode(),(root/p).read_text(encoding='utf-8-sig')):
        assert 'selected_parent=proposed;current_mut=child_mut;current_val=child_val' in body
    ops=[];evaluations=[];transitions=[];parents={}
    for line in (run/'trajectory_private.jsonl').open(encoding='utf-8'):
        x=json.loads(line)
        if x['stage']=='OPPORTUNITY':
            ops.append(x['opportunity']);parents.setdefault(x['parent']['team_state_id'],x['parent'])
        elif x['stage']=='EVALUATION':evaluations.append(x)
        elif x['stage']=='TRANSITION':transitions.append({k:v for k,v in x.items() if k not in {'parent','child'}})
    assert len(ops)==len(evaluations)==len(transitions)==50
    index={};physical=Counter();stage_tokens=Counter();semantic=Counter()
    for line in (run/'ledger.jsonl').open(encoding='utf-8'):
        x=json.loads(line)
        if x['kind']=='PREDICTION_VALIDITY' and x['split']=='optimize':
            k=(x['member_id'],x['mutable_prompt_sha256'],x['input_id'])
            assert k not in index or index[k]==x['request_sha256']
            index[k]=x['request_sha256'];semantic[k]=x['semantic_attempt_count']
    for line in (run/'provider_trace_private.jsonl').open(encoding='utf-8'):
        x=json.loads(line);physical[x['stage']]+=1
        stage_tokens[x['stage']]+=x['response']['input_tokens']+x['response']['output_tokens']
    @lru_cache(None)
    def prediction(member,prompt,q):
        if (member,prompt,q) not in index:raise LookupError('MISSING_COMPATIBLE_PREDICTION')
        return load(run/'resolved_output_cache'/(index[member,prompt,q]+'.json'))['result']['resolved_prediction']
    @lru_cache(None)
    def matches(answer,peers):
        exact=tuple(i for i,a in enumerate(peers) if answer==a)
        if exact:return exact
        d=domain_matrix((answer,*peers));assert all(d['valid'])
        return tuple(i for i in range(len(peers)) if d['equivalence'][0][i+1])
    def vote_for(parent,q,target,prompt,correct):
        states=parent['diagnostics']['team_states'];pos=next(i for i,p in enumerate(states) if p['question_hash']==q)
        old=states[pos];pred=prediction(target,prompt,q);valid=pred['prediction_valid']
        if correct:assert valid
        peers=[j for j in range(5) if j!=target];g=sum(old['team_correctness'][j] for j in peers)
        wrong=Counter(old['team_answers'][j] for j in peers if old['team_validity'][j] and not old['team_correctness'][j])
        w=max(wrong.values(),default=0)
        if correct:return g+1>w
        if not valid:return g>0 and g>w
        if g<=1:return False
        if g>=3:return True
        if w>=2:return False
        labels=[];answers=[]
        for j in peers:
            if old['team_validity'][j] and not old['team_correctness'][j] and old['team_answers'][j] not in labels:
                labels.append(old['team_answers'][j]);answers.append(parent['member_outputs'][j][pos]['answer'])
        eq=matches(pred['answer'],tuple(answers)) if answers else ()
        wrong[labels[eq[0]] if eq else 'new_wrong_cluster']+=1
        return g>max(wrong.values())
    initial=load(run/'initial_state_private.json')
    candidates=[];by_op={};full_matrix={};chain_events=[];ranking=Counter();flow=Counter()
    for oi,(op,ev,tr) in enumerate(zip(ops,evaluations,transitions,strict=True),1):
        parent=parents[op['parent_state_id']];pr={x['question_hash']:x for x in parent['diagnostics']['team_states']};m=op['target_member']
        outcomes=ev['search']['search_state']['proposal_diagnostics']['proposal_outcomes']
        exported={x['candidate']['candidate_id']:x for x in ev['evaluated']}
        local=[]
        for z in outcomes:
            if z['status']!='LOCALLY_EVALUATED':continue
            r=z['edit_lineage'];effects=r['effects'];cid=r['candidate_id'];ph=r['child_prompt_id']
            item=exported.get(cid);pd=item['team_probe']['aggregation_diagnostics'] if item else None
            row=dict(opportunity=oi,member=m+1,generation=z['generation'],candidate_id=cid,prompt_sha256=ph,
                mutation_delta=effects['mutation']['member_delta'],validation_delta=effects['search_validation']['member_delta'],
                validation_lost=len(effects['search_validation']['broken_ids']),actual_parent_delta=effects['actual_parent_validation']['member_delta'],
                chain_depth=len(r['edit_chain']),exported=cid in exported,full_evaluated=bool(item and item['full']),
                full_vote_delta=None,full_target_delta=None,full_pass=None,committed=cid==tr['committed'],
                probe_vote_delta=None,probe_target_delta=None,probe_risk_pass=None,assigned_repairs=None,
                probe_collateral=None,probe_invalid_delta=None)
            if pd:
                pm=pd['team_probe_metrics'];row.update(probe_vote_delta=pm['vote_delta'],probe_target_delta=pm['target_delta'],
                    probe_risk_pass=pd['risk_pass'],assigned_repairs=pd['assigned_repair_count'],
                    probe_collateral=pd['collateral_loss'],probe_invalid_delta=pm['invalid_delta'],promotion_reason=item['diagnostics']['promotion_reason'])
                effects={**effects,'team_probe':pd['edit_effect']}
            if row['full_evaluated']:
                fd=item['full']['aggregation_diagnostics'];effects={**effects,'full':fd['edit_effect']}
                row.update(full_vote_delta=fd['team_score_delta'],full_target_delta=fd['target_score_delta'],
                    full_pass=not bool(fd.get('scientific_risk_code')))
            known_cells={};prefull_cells={}
            for scope,e in effects.items():
                if scope not in {'mutation','search_validation','team_probe','full'}:continue
                vals={}
                for q in e['membership']:
                    before=pr[q]['team_correctness'][m]
                    after=(before and q not in e['broken_ids']) or q in e['fixed_ids']
                    assert prediction(m,ph,q)['prediction_valid']==(q not in e['invalid_ids'])
                    newvote=vote_for(parent,q,m,ph,after)
                    vals[q]=dict(target_delta=int(after)-int(before),vote_delta=int(newvote)-int(pr[q]['vote_correct']))
                    if q in known_cells:assert known_cells[q]==vals[q]
                    known_cells[q]=vals[q]
                    if scope!='full':prefull_cells[q]=vals[q]
                row[scope+'_vote_delta']=sum(x['vote_delta'] for x in vals.values())
                if scope=='team_probe':assert row[scope+'_vote_delta']==row['probe_vote_delta']
                if scope=='full':
                    assert row[scope+'_vote_delta']==row['full_vote_delta']
                    assert sum(x['target_delta'] for x in vals.values())==row['full_target_delta']
                    full_matrix[oi,cid]=vals
            row['validation_vote_delta']=row['search_validation_vote_delta']
            available=sum((m,ph,q) in index for q in pr)
            row['compatible_optimize_predictions']=available;row['missing_optimize_predictions']=len(pr)-available
            incumbent_vote=sum(r['vote_correct'] for r in pr.values())
            def bound(cells):
                unknown=set(pr)-set(cells)
                observed_vote=sum(int(pr[q]['vote_correct'])+c['vote_delta'] for q,c in cells.items())
                observed_target=sum(int(pr[q]['team_correctness'][m])+c['target_delta'] for q,c in cells.items())
                peer_bounds=[vote_bounds_from_peers(pr[q],m) for q in unknown]
                vl=observed_vote+sum(a for a,_ in peer_bounds);vu=observed_vote+sum(b for _,b in peer_bounds)
                tu=observed_target+len(unknown)
                cannot_pass=(vu<incumbent_vote or tu<initial['member_scores'][m] or
                    vu<=incumbent_vote and tu<=parent['member_scores'][m])
                return dict(lower_delta=vl-incumbent_vote,upper_delta=vu-incumbent_vote,
                    unmeasured_questions=len(unknown),full_rejection_certified_by_upper_bounds=bool(cannot_pass))
            row['partial_vote_bounds']=bound(known_cells)
            row['pre_full_bounds']=bound(prefull_cells)
            if row['full_evaluated']:
                b=row['partial_vote_bounds'];assert b['lower_delta']==b['upper_delta']==row['full_vote_delta']
                assert b['full_rejection_certified_by_upper_bounds']==(not row['full_pass'])
                assert not row['pre_full_bounds']['full_rejection_certified_by_upper_bounds'] or not row['full_pass']
            local.append(row);candidates.append(row)
            chain_events.append(dict(opportunity=oi,generation=row['generation'],chain_depth=row['chain_depth'],
                actual_parent_delta=row['actual_parent_delta'],validation_delta=row['validation_delta'],
                full_pass=row['full_pass'],committed=row['committed']))
        assert {r['candidate_id'] for r in select_local(local)}==set(exported)
        promoted=select_probe([r for r in local if r['exported']])
        assert {r['candidate_id'] for r in promoted}=={r['candidate_id'] for r in local if r['full_evaluated']}
        by_op[oi]=local
        flow.update(slots=len(outcomes),local=len(local),exports=len(exported),full=len(promoted),commits=bool(tr['committed']))
        flow['all_no_safe_opportunities']+=all(z['status']=='NO_SAFE_EDIT' for z in outcomes)
        for i,a in enumerate(local):
            for b in local[i+1:]:
                ranking['local_pairs']+=1
                ranking['same_validation_net']+=a['validation_delta']==b['validation_delta']
                ranking['same_validation_net_loss_mutation']+=(a['validation_delta'],a['validation_lost'],a['mutation_delta'])==(b['validation_delta'],b['validation_lost'],b['mutation_delta'])
                if a['full_evaluated'] and b['full_evaluated']:
                    ranking['both_full_pairs']+=1
                    ranking['same_validation_net_different_full_vote']+=a['validation_delta']==b['validation_delta'] and a['full_vote_delta']!=b['full_vote_delta']
                    ranking['same_probe_vote_different_full_vote']+=a['probe_vote_delta']==b['probe_vote_delta'] and a['full_vote_delta']!=b['full_vote_delta']
    assert flow['local']==76 and flow['exports']==70 and flow['full']==31 and flow['commits']==5
    fulls=[r for r in candidates if r['full_evaluated']];exports=[r for r in candidates if r['exported']]
    noassigned=[r for r in exports if r['assigned_repairs']==0];capped=[r for r in exports if r['assigned_repairs']>=1 and r['probe_risk_pass'] and not r['full_evaluated']]
    alt_probe=[r for rs in by_op.values() for r in select_probe([r for r in rs if r['exported']],assigned=False)]
    alt_local=[r for rs in by_op.values() for r in select_local(rs,team=True)]
    def keys(rs):return {(r['opportunity'],r['candidate_id']) for r in rs}
    alt=dict(no_assigned_floor=selection_observations(alt_probe),local_vote_first=dict(selected=len(alt_local),
        replaces_original_exports=len(keys(alt_local)-keys(exports)),newly_exported_full_unknown=sum(not r['exported'] and not r['full_evaluated'] for r in alt_local),
        displaced_observed_commits=sum(r['committed'] and (r['opportunity'],r['candidate_id']) not in keys(alt_local) for r in exports)))
    alt['no_assigned_floor']['new_full_slots']=len(keys(alt_probe)-keys(fulls))
    alt['no_assigned_floor']['displaced_known_full_pass']=sum(r['full_pass'] and (r['opportunity'],r['candidate_id']) not in keys(alt_probe) for r in fulls)
    alt['no_assigned_floor']['displaced_observed_commits']=sum(r['committed'] and (r['opportunity'],r['candidate_id']) not in keys(alt_probe) for r in fulls)
    alt['no_assigned_floor']['displaced_commit_opportunities']=[r['opportunity'] for r in fulls if r['committed'] and (r['opportunity'],r['candidate_id']) not in keys(alt_probe)]
    alt['no_assigned_floor']['choices']=[dict(opportunity=r['opportunity'],generation=r['generation'],full_observed=r['full_evaluated']) for r in alt_probe]
    cross_tabs={}
    for scope in ('validation_delta','mutation_delta','probe_vote_delta','validation_vote_delta'):
        cross_tabs[scope]={sign:dict(candidates=sum((r[scope]>0 if sign=='positive' else r[scope]==0 if sign=='zero' else r[scope]<0) for r in fulls),
            full_pass=sum(r['full_pass'] and (r[scope]>0 if sign=='positive' else r[scope]==0 if sign=='zero' else r[scope]<0) for r in fulls),
            full_vote_positive=sum(r['full_vote_delta']>0 and (r[scope]>0 if sign=='positive' else r[scope]==0 if sign=='zero' else r[scope]<0) for r in fulls)) for sign in ('positive','zero','negative')}
    pairs=[]
    for oi,rs in by_op.items():
        fs=[r for r in rs if r['full_evaluated']]
        if len(fs)==2:
            a,b=fs
            pairs.append(dict(opportunity=oi,full_vote=[a['full_vote_delta'],b['full_vote_delta']],
                validation=[a['validation_delta'],b['validation_delta']],local_vote=[a['validation_vote_delta'],b['validation_vote_delta']],
                probe_vote=[a['probe_vote_delta'],b['probe_vote_delta']],full_pass=[a['full_pass'],b['full_pass']]))
    mc=Counter();panel_pairs=Counter();miss_costs={};support=set()
    for policy in ('uniform','parent_stratified'):
        for seed in range(repetitions):
            for oi,rs in by_op.items():
                fs=[r for r in rs if r['full_evaluated']]
                if not fs:continue
                op=ops[oi-1];pr={x['question_hash']:x for x in parents[op['parent_state_id']]['diagnostics']['team_states']}
                mutation=op['evaluation_plan']['evidence_audit']['memberships']['mutation']
                panel=draw_panel(pr,op['target_member'],mutation,17001+seed*100+oi,stratified=policy=='parent_stratified')
                estimates=[]
                for r in fs:
                    matrix=full_matrix[oi,r['candidate_id']]
                    estimate=sum(matrix[q]['vote_delta']*weight for q,weight in panel)+sum(matrix[q]['vote_delta'] for q in mutation)
                    observed_loss=any(matrix[q]['vote_delta']<0 for q in [*mutation,*(q for q,_ in panel)])
                    k=policy+':';mc[k+'candidate_panel_replicates']+=1
                    if r['full_vote_delta']<0:
                        mc[k+'negative_full_replicates']+=1;mc[k+'negative_detected_by_any_vote_loss']+=observed_loss
                        mc[k+'negative_detected_by_negative_estimate']+=estimate<0
                    if r['full_vote_delta']>0:
                        mc[k+'positive_full_replicates']+=1;mc[k+'positive_estimated_nonpositive']+=estimate<=0
                    estimates.append(estimate)
                if len(fs)==2 and fs[0]['full_vote_delta']!=fs[1]['full_vote_delta']:
                    panel_pairs[policy+':non_tied_true_pair_replicates']+=1
                    if estimates[0]==estimates[1]:panel_pairs[policy+':estimate_tie']+=1
                    elif (estimates[0]>estimates[1])==(fs[0]['full_vote_delta']>fs[1]['full_vote_delta']):panel_pairs[policy+':correct_strict_order']+=1
                    else:panel_pairs[policy+':wrong_strict_order']+=1
    for name,rs in [('nonexported',[r for r in candidates if not r['exported']]),('assigned_zero',noassigned),('cap_excluded',capped)]:
        miss_costs[name]=dict(candidates=len(rs),missing_logical_target_predictions=sum(r['missing_optimize_predictions'] for r in rs),
            cache_coverage_histogram=dict(Counter(r['compatible_optimize_predictions'] for r in rs)))
    chains=[]
    for oi,rs in by_op.items():
        for i,r in enumerate(rs):
            assert r['chain_depth']==i+1
            if r['full_pass']:
                chains.append(dict(opportunity=oi,generation=r['generation'],committed=r['committed'],
                    depth=r['chain_depth'],ancestor_actual_parent_deltas=[a['actual_parent_delta'] for a in rs[:i]],
                    ancestor_validation_root_deltas=[a['validation_delta'] for a in rs[:i]]))
    result=dict(schema='local_evaluation_promotion_feasibility_v1',decision='GATHER_EVIDENCE_BEFORE_V29_ALGORITHM_CHANGE',
        provenance=dict(current_runtime_source=CURRENT,frozen_scientific_source=FROZEN,trajectory_sha256=trajectory_sha,
            current_scientific_source_hash=current_hash,analysis_tool_sha256=digest_file(root/'scripts/audit_local_evaluation_promotion.py'),
            source_checks=source_checks,analysis_plan_sha256=digest_file(out/'PLAN.md') if (out/'PLAN.md').exists() else None),
        funnel=dict(flow),candidates=[{k:v for k,v in r.items() if k!='candidate_id'} for r in candidates],
        ranking=dict(ranking),within_op_full_pairs=pairs,alternatives=alt,selected_full_cross_tabs=cross_tabs,
        promotion_censored=dict(assigned_zero=len(noassigned),assigned_zero_probe_vote_positive=sum(r['probe_vote_delta']>0 for r in noassigned),
            assigned_zero_risk_pass=sum(r['probe_risk_pass'] for r in noassigned),cap_excluded=len(capped),
            assigned_zero_full_known=sum(r['full_evaluated'] for r in noassigned),
            censored_with_complete_compatible_cache=sum(not r['full_evaluated'] and r['compatible_optimize_predictions']==150 for r in candidates),
            censored_rejection_certified_by_upper_bounds=sum(not r['full_evaluated'] and r['partial_vote_bounds']['full_rejection_certified_by_upper_bounds'] for r in candidates)),
        observed_full=dict(count=len(fulls),pass_count=sum(r['full_pass'] for r in fulls),
            vote_delta_histogram=dict(Counter(r['full_vote_delta'] for r in fulls)),
            rejection_certified_before_full=sum(r['pre_full_bounds']['full_rejection_certified_by_upper_bounds'] for r in fulls)),
        cache_completion=miss_costs,parent_chains=chain_events,full_pass_ancestry=chains,
        panel_monte_carlo=dict(repetitions=repetitions,counts=dict(mc),pair_counts=dict(panel_pairs),
            scope='SELECTED_FULL_CATALOG_FIXED_OUTPUT_PANEL_RANDOMNESS_NOT_NEW_SOLVER_OR_SEARCH_REALIZATIONS'),
        cost=dict(new_physical_requests=dict(physical),new_charged_tokens=dict(stage_tokens),
            scope='SEALED_CONTINUATION_NEW_TRACE_ONLY_IMPORTED_PREFIX_SEPARATE'),
        checks=dict(export_rule_matches=True,promotion_rule_matches=True,all_known_probe_full_votes_match=True,
            missing_full_not_imputed=True,model_api_calls=0,validation_test_raw_reads=0))
    (out/'analysis.json').write_bytes((json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps({k:result[k] for k in ('funnel','ranking','promotion_censored','alternatives','observed_full','cache_completion','panel_monte_carlo')}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--repetitions',type=int,default=500)
    args=p.parse_args();audit(args.root,args.out,args.repetitions)
