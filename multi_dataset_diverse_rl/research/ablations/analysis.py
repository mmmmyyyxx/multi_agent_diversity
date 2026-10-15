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
            exhausted=sum(m['decision'] == 'ABANDONED_GENERATION_RETRIES_EXHAUSTED' for m in metrics))
        if sum(counts.values()) != plan['config']['rounds']:
            raise SearchContractError('ABLATION_ANALYSIS_OPPORTUNITY_TOTAL')
        row = dict(cell_id=ident, arm_id=cell['arm_id'], seed=cell['seed'], repeat_id=cell['repeat_id'],
            validation_accuracy=r['final_validation_correct'] / r['validation_count'],
            optimize_accuracy=r['final_optimize_correct'] / r['optimize_count'],
            optimization_tokens=r['usage']['input_tokens'] + r['usage']['output_tokens'],
            optimization_provider_calls=r['usage']['provider_calls'], **counts)
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
            **{k: sum(r[k] for r in subset) for k in ('updates', 'rejected', 'exhausted')})
        if final_predictions is not None:
            group['mean_test_accuracy'] = sum(r['test_accuracy'] for r in subset) / len(subset)
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
    return dict(identity='ABLATION_SANITIZED_REPEAT_SUMMARY_V1', rows=rows, groups=groups,
        within_arm_repeat_pairs=pairs,
        limitations=['fixed_seed_repeats_not_distinct_seeds', 'lexical_similarity_not_semantic_strategy_diversity'],
        token_allocation='exact_paired_cache_charged_to_first_dispatching_cell_no_double_charge')
