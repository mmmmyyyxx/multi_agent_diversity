"""Sanitized per-repeat metrics; fixed-seed repeats are not distinct algorithm seeds."""
from itertools import combinations
from difflib import SequenceMatcher

from ...search.schemas import SearchContractError
from ...search.system_prompt import SystemPrompt


def summarize(plan, results, final_predictions=None):
    expected = {c['cell_id']: c for c in plan['cells']}
    actual = {r['cell']['cell_id']: r for r in results}
    if len(actual) != len(results) or set(actual) != set(expected):
        raise SearchContractError('ABLATION_ANALYSIS_INCOMPLETE_CELLS')
    if final_predictions is not None and set(final_predictions) != set(expected):
        raise SearchContractError('ABLATION_ANALYSIS_INCOMPLETE_TEST')
    rows = []
    for ident, cell in expected.items():
        r = actual[ident]
        if r['cell'] != cell or r['rounds_completed'] != plan['config']['rounds']:
            raise SearchContractError('ABLATION_ANALYSIS_SCOPE')
        metrics = r['metrics']
        counts = dict(updates=sum(m['decision'] == 'ACCEPTED_STRICT_VALIDATION_GAIN' for m in metrics),
            rejected=sum(m['decision'] == 'REJECTED_NO_STRICT_VALIDATION_GAIN' for m in metrics),
            exhausted=sum(m['decision'] == 'ABANDONED_GENERATION_RETRIES_EXHAUSTED' for m in metrics),
            no_op=sum(m['decision'] == 'SKIPPED_NO_OP' for m in metrics))
        if sum(counts.values()) != plan['config']['rounds']:
            raise SearchContractError('ABLATION_ANALYSIS_OPPORTUNITY_TOTAL')
        row = dict(cell_id=ident, arm_id=cell['arm_id'], seed=cell['seed'], repeat_id=cell['repeat_id'],
            validation_accuracy=r['final_validation_correct'] / r['validation_count'],
            optimize_accuracy=r['final_optimize_correct'] / r['optimize_count'],
            optimization_tokens=r['usage']['input_tokens'] + r['usage']['output_tokens'],
            optimization_provider_calls=r['usage']['provider_calls'], **counts)
        if metrics and 'proposals' in metrics[0]:
            proposals=[p for m in metrics for p in m['proposals']]
            row.update(proposal_count=len(proposals),
                attempted_proposals=sum(p['generation_draws']>0 for p in proposals),
                generation_draws=sum(p['generation_draws'] for p in proposals),
                generation_retries=sum(max(0,p['generation_draws']-1) for p in proposals),
                structurally_legal_proposals=sum(p['candidate_prompt'] is not None for p in proposals),
                measured_proposals=sum(p['generation_status']=='MEASURED' for p in proposals),
                rejected_measured_proposals=sum(p['generation_status']=='MEASURED' and not p['accepted'] for p in proposals),
                exhausted_proposals=sum(p['generation_status']=='EXHAUSTED' for p in proposals),
                no_op_proposals=sum(p['generation_status']=='NO_OP' for p in proposals),
                selected_two_pattern_opportunities=sum(m['selected_distinct_pattern_count']==2 for m in metrics),
                two_measured_pattern_opportunities=sum(m['evaluated_distinct_pattern_count']==2 for m in metrics),
                true_two_pattern_opportunities=sum(m['distinct_pattern_count']==2 for m in metrics),
                second_slot_wins=sum(m['winning_proposal_id'] is not None and
                    m['winning_proposal_id']==m['proposals'][-1]['proposal_id'] and len(m['proposals'])==2 for m in metrics),
                distinct_second_pattern_wins=sum(m['distinct_pattern_count']==2 and
                    m['winning_proposal_id']==m['proposals'][1]['proposal_id'] for m in metrics),
                shared_diagnostic_tokens=sum(m['shared_diagnostic_cost']['input_tokens']+
                    m['shared_diagnostic_cost']['output_tokens'] for m in metrics),
                proposal_optimizer_tokens=sum(p['generation_cost']['input_tokens']+p['generation_cost']['output_tokens'] for p in proposals))
            row.update(accepted_rounds=counts['updates'],updates=sum(m['accepted_update_count'] for m in metrics),
                accepted_updates=sum(m['accepted_update_count'] for m in metrics),
                double_update_opportunities=sum(m['accepted_update_count']==2 for m in metrics),
                skipped_resolved_proposals=sum(p['generation_status']=='SKIPPED_NO_CURRENT_ERRORS' for p in proposals),
                search_wall_clock_seconds=r['search_wall_clock_seconds'])
            row['role_usage']={role:dict(physical_calls=r['usage'][role],
                input_tokens=r['usage'][role+'_input_tokens'],output_tokens=r['usage'][role+'_output_tokens'])
                for role in ('solver','reflection','pattern_gradient','pattern_cluster')}
            row['second_pattern_win_denominator']=row['selected_two_pattern_opportunities'] if cell['proposal_mode']=='parallel' else None
            row['second_pattern_win_rate']=(row['distinct_second_pattern_wins']/row['second_pattern_win_denominator']
                if row['second_pattern_win_denominator'] else None)
        if 'final_admission' in r:
            row.update(final_admitted=r['final_admission']['admitted'],
                final_status=r['final_admission']['status'],
                running_validation_accuracy=sum(v['correct'] for v in r['final_admission']['running_validation'].values())/r['validation_count'],
                measured_children=sum(s['disposition']=='MEASURED' for m in metrics for s in m['slots']),
                exhausted_slots=sum(s['disposition']=='EXHAUSTED' for m in metrics for s in m['slots']),
                no_op_slots=sum(s['disposition']=='NO_OP' for m in metrics for s in m['slots']))
        if final_predictions is not None:
            values = final_predictions[ident]
            row.update(test_count=len(values), test_accuracy=sum(v['correct'] for v in values.values()) / len(values))
        rows.append(row)
    groups = []
    for arm in plan['config']['arms']:
        subset = [r for r in rows if r['arm_id'] == arm['arm_id']]
        group = dict(arm_id=arm['arm_id'], repeats=len(subset),
            mean_validation_accuracy=sum(r['validation_accuracy'] for r in subset) / len(subset),
            optimization_tokens=sum(r['optimization_tokens'] for r in subset),
            **{k: sum(r[k] for r in subset) for k in ('updates', 'rejected', 'exhausted','no_op')})
        if final_predictions is not None:
            group['mean_test_accuracy'] = sum(r['test_accuracy'] for r in subset) / len(subset)
        if all('final_admitted' in r for r in subset):
            group['final_admitted_count']=sum(r['final_admitted'] for r in subset)
            group['final_rejected_count']=len(subset)-group['final_admitted_count']
        if all('proposal_count' in r for r in subset):
            group.update({k:sum(r[k] for r in subset) for k in ('proposal_count','attempted_proposals','generation_draws',
                'generation_retries','structurally_legal_proposals','measured_proposals','rejected_measured_proposals',
                'selected_two_pattern_opportunities','two_measured_pattern_opportunities',
                'exhausted_proposals','no_op_proposals','true_two_pattern_opportunities','second_slot_wins',
                'distinct_second_pattern_wins','shared_diagnostic_tokens','proposal_optimizer_tokens')})
            group.update({k:sum(r[k] for r in subset) for k in ('accepted_rounds','accepted_updates',
                'double_update_opportunities','skipped_resolved_proposals','search_wall_clock_seconds')})
            group['role_usage']={role:{k:sum(r['role_usage'][role][k] for r in subset)
                for k in ('physical_calls','input_tokens','output_tokens')} for role in subset[0]['role_usage']}
            denominator=(sum(r['second_pattern_win_denominator'] for r in subset)
                if all(r['second_pattern_win_denominator'] is not None for r in subset) else None)
            group['second_pattern_win_denominator']=denominator
            group['second_pattern_win_rate']=group['distinct_second_pattern_wins']/denominator if denominator else None
        groups.append(group)
    pairs = []
    for arm in plan['config']['arms']:
        subset = [c for c in plan['cells'] if c['arm_id'] == arm['arm_id']]
        for left, right in combinations(subset, 2):
            a = SystemPrompt.from_dict(actual[left['cell_id']]['final_prompt'])
            b = SystemPrompt.from_dict(actual[right['cell_id']]['final_prompt'])
            pair = dict(arm_id=arm['arm_id'], left_repeat=left['repeat_id'], right_repeat=right['repeat_id'],
                prompt_character_similarity=SequenceMatcher(None, a.render(), b.render(), autojunk=False).ratio(),
                strategy_character_similarity=SequenceMatcher(None, a.strategy, b.strategy, autojunk=False).ratio())
            if final_predictions is not None:
                av, bv = final_predictions[left['cell_id']], final_predictions[right['cell_id']]
                if not av or set(av) != set(bv):
                    raise SearchContractError('ABLATION_ANALYSIS_TEST_MEMBERSHIP')
                pair['correctness_disagreement'] = sum(av[i]['correct'] != bv[i]['correct'] for i in av) / len(av)
            pairs.append(pair)
    summary=dict(identity='ABLATION_SANITIZED_REPEAT_SUMMARY_V1', rows=rows, groups=groups,
        within_arm_repeat_pairs=pairs,
        limitations=['fixed_seed_repeats_not_distinct_seeds', 'lexical_similarity_not_semantic_strategy_diversity'],
        token_allocation='exact_paired_cache_charged_to_first_dispatching_cell_no_double_charge')
    if results and results[0]['metrics'] and 'proposals' in results[0]['metrics'][0]:
        from .proposal_analysis import budget_comparison
        summary['candidate_budget_comparison']=budget_comparison(results)
    return summary
