"""Cost-ordered retained-state curves; no intermediate Test access or lookahead."""
from collections import defaultdict


def budget_comparison(results):
    curves={};by_repeat=defaultdict(list)
    for result in results:
        metrics=result['metrics'];cell=result['cell'];count=0
        first=metrics[0]
        points=[dict(candidate_evaluations=0,round=0,substep=None,
            validation_correct=first['validation_before'],optimize_correct=first['optimize_before'])]
        for m in metrics:
            if cell['proposal_mode']=='sequential':
                for p in m['proposals']:
                    if p['generation_status']!='MEASURED':continue
                    count+=1
                    points.append(dict(candidate_evaluations=count,round=m['round'],substep=p['slot'],
                        validation_correct=p['candidate_validation_score'] if p['accepted'] else p['validation_before'],
                        optimize_correct=p['candidate_optimize_score'] if p['accepted'] else p['optimize_before']))
            else:
                count+=sum(p['generation_status']=='MEASURED' for p in m['proposals'])
                chosen=next((p for p in m['proposals'] if p['accepted']),None)
                # The parallel winner is knowable only after all evaluations.
                # No point after just slot zero borrows the future slot-one score.
                points.append(dict(candidate_evaluations=count,round=m['round'],substep=None,
                    validation_correct=m['validation_before'] if chosen is None else chosen['candidate_validation_score'],
                    optimize_correct=m['optimize_before'] if chosen is None else chosen['candidate_optimize_score']))
        for p in points:
            p['logical_solver_metric_observations']=(1+p['candidate_evaluations'])*(result['optimize_count']+result['validation_count'])
            p['validation_accuracy']=p['validation_correct']/result['validation_count']
            p['optimize_accuracy']=p['optimize_correct']/result['optimize_count']
        curves[cell['cell_id']]=points;by_repeat[cell['repeat_id']].append(cell['cell_id'])
    matched=[]
    for repeat,ids in sorted(by_repeat.items()):
        common=min(curves[i][-1]['candidate_evaluations'] for i in ids)
        for budget in sorted({0,common}|{b for b in (5,10,15,20,30,40) if b<=common}):
            matched.append(dict(repeat_id=repeat,budget=budget,
                retained_states={i:next(p for p in reversed(curves[i]) if p['candidate_evaluations']<=budget) for i in ids}))
    return dict(identity='RETAINED_STATE_MEASURED_CANDIDATE_BUDGET_CURVE_V1',curves=curves,matched=matched,
        metric='recorded_Optimize_and_adaptive_Validation_only_no_intermediate_Test',
        policy='last_available_retained_boundary_at_or_below_budget_no_interpolation_no_future_winner',
        limitations=['equal_ceiling_not_equal_realized_candidates_or_Tokens',
            'parallel_updates_only_available_after_whole_round',
            'logical_metric_observations_include_cache_reuse_not_physical_provider_calls',
            'shared_Solver_cost_charged_to_first_dispatcher',
            'adaptive_Validation_is_selection_evidence_not_independent_confirmation'])
