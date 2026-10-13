"""Final correctness coverage and equivalence-aware disagreement, no LLM judge."""
from difflib import SequenceMatcher
from collections import Counter
from itertools import combinations
from statistics import mean, pstdev

from ..search.schemas import SearchContractError
from ..search.system_prompt import SystemPrompt
from .single_prompt_policy import ARMS, policy


def pair_metrics(left, right, equivalent):
    if not left or set(left) != set(right):
        raise SearchContractError('STUDY_ANALYSIS_MEMBERSHIP_MISMATCH')
    ids = sorted(left)
    for rows in (left, right):
        if any(type(r[k]) is not bool for r in rows.values() for k in ('correct', 'valid')):
            raise SearchContractError('STUDY_ANALYSIS_BOOLEAN_REQUIRED')
        if any(r['correct'] and not r['valid'] for r in rows.values()):
            raise SearchContractError('STUDY_INVALID_CORRECTNESS')
    co_valid = [i for i in ids if left[i]['valid'] and right[i]['valid']]
    answer_different = sum(not equivalent(left[i]['answer'], right[i]['answer']) for i in co_valid)
    both = sum(left[i]['correct'] and right[i]['correct'] for i in ids)
    either = sum(left[i]['correct'] or right[i]['correct'] for i in ids)
    average = mean(sum(r['correct'] for r in rows.values()) / len(ids) for rows in (left, right))
    return dict(n=len(ids), co_valid_n=len(co_valid), answer_disagreement_count=answer_different,
        answer_disagreement=answer_different / len(co_valid) if co_valid else None,
        validity_disagreement=sum(left[i]['valid'] != right[i]['valid'] for i in ids) / len(ids),
        correctness_disagreement=sum(left[i]['correct'] != right[i]['correct'] for i in ids) / len(ids),
        correctness_overlap=both / len(ids), correctness_jaccard=both / either if either else None,
        double_fault=(len(ids) - either) / len(ids), oracle_accuracy=either / len(ids),
        mean_individual_accuracy=average, oracle_gain=either / len(ids) - average)


def group_coverage(predictions):
    if not predictions or not predictions[0] or any(set(r) != set(predictions[0]) for r in predictions):
        raise SearchContractError('STUDY_ANALYSIS_MEMBERSHIP_MISMATCH')
    n = len(predictions[0])
    accuracy = [sum(r['correct'] for r in rows.values()) / n for rows in predictions]
    oracle = sum(any(rows[i]['correct'] for rows in predictions) for i in predictions[0]) / n
    return dict(individual_accuracy=accuracy, mean_accuracy=mean(accuracy),
        population_seed_sd=pstdev(accuracy), oracle_accuracy=oracle,
        oracle_gain=oracle - mean(accuracy), seed_count=len(predictions))


def analyze_final(cells, predictions, equivalent, *, seeds=None):
    seeds = policy()['seeds'] if seeds is None else seeds
    if seeds not in ([88,89,90],[88,89]):raise SearchContractError('STUDY_ANALYSIS_SEEDS_FORBIDDEN')
    n_cells=len(ARMS)*len(seeds)
    expected = {(a, s) for a in ARMS for s in seeds}
    if ({(c['arm'], c['seed']) for c in cells} != expected or len(cells) != n_cells
            or len({c['cell_id'] for c in cells}) != n_cells or set(predictions) != {c['cell_id'] for c in cells}):
        raise SearchContractError('STUDY_FINAL_NINE_REQUIRED')
    labels = [c['cell_id'] for c in cells]
    pairs = []
    by_id = {c['cell_id']: c for c in cells}
    matrix = {name: [[None] * n_cells for _ in labels] for name in (
        'answer_disagreement', 'correctness_disagreement', 'correctness_overlap', 'double_fault', 'oracle_gain')}
    for i, a in enumerate(labels):
        for j in range(i, len(labels)):
            b = labels[j]
            metric = pair_metrics(predictions[a], predictions[b], equivalent)
            for name, rows in matrix.items():
                rows[i][j] = rows[j][i] = metric[name]
            if i != j:
                text_a = SystemPrompt.from_dict(by_id[a]['final_prompt']).render()
                text_b = SystemPrompt.from_dict(by_id[b]['final_prompt']).render()
                pairs.append(dict(left=a, right=b, same_seed=by_id[a]['seed'] == by_id[b]['seed'],
                    same_arm=by_id[a]['arm'] == by_id[b]['arm'],
                    identical_prompts=text_a == text_b,
                    identical_prompt_answer_variability=text_a == text_b and bool(metric['answer_disagreement']),
                    prompt_character_similarity=SequenceMatcher(None, text_a, text_b, autojunk=False).ratio(),
                    **metric))
    groups = {a: group_coverage([predictions[c['cell_id']] for c in cells if c['arm'] == a]) for a in ARMS}
    comparisons = {}
    for a, b in combinations(ARMS, 2):
        selected = [r for r in pairs if {by_id[r['left']]['arm'], by_id[r['right']]['arm']} == {a, b}]
        paired = [r for r in selected if r['same_seed']]
        comparisons[a + '_vs_' + b] = dict(all_cross_seed_pair_count=len(selected),
            paired_seed_pair_count=len(paired),
            all_cross_seed_pairs=selected, paired_seed_pairs=paired,
            joint_coverage=group_coverage([predictions[c['cell_id']] for c in cells if c['arm'] in (a, b)]))
    within = {a: [r for r in pairs if by_id[r['left']]['arm'] == by_id[r['right']]['arm'] == a] for a in ARMS}
    outcome_distributions = {name: dict(Counter(r.get('outcome_category', 'CORRECT' if r['correct'] else 'VALID_WRONG' if r['valid'] else 'INVALID')
                                               for r in rows.values())) for name, rows in predictions.items()}
    by_subject = {}
    for name, rows in predictions.items():
        subjects = sorted({r.get('task_type', 'UNAVAILABLE') for r in rows.values()})
        by_subject[name] = {s: dict(n=sum(r.get('task_type', 'UNAVAILABLE') == s for r in rows.values()),
            correct=sum(r['correct'] for r in rows.values() if r.get('task_type', 'UNAVAILABLE') == s)) for s in subjects}
    return dict(labels=labels, matrices=matrix, **{('all_36_pairs' if n_cells==9 else 'all_15_pairs'):pairs}, within_arm_pairs=within,
        groups=groups, between_arms=comparisons,
        outcome_category_distribution=outcome_distributions, subject_counts_and_correct=by_subject,
        **{('all_nine_coverage' if n_cells==9 else 'all_six_coverage'):group_coverage([predictions[i] for i in labels])},
        semantic_strategy_analysis='OWNER_REVIEW_REQUIRED_NO_AUTOMATIC_CAUSAL_OR_SEMANTIC_CLAIM',
        answer_metric='pinned_math_equivalence_on_co_valid_pairs_invalidity_reported_separately',
        statistical_scope=('three_seeds_descriptive_no_significance_or_causal_pattern_only_claim' if len(seeds)==3 else 'two_seeds_descriptive_no_significance_or_causal_pattern_only_claim'))
