"""Counterfactual diagnostics only; no production policy or admission read point."""
import ast
from collections import Counter
from copy import deepcopy


LIMITS = (240, 320, 400, 480, 640)


def threshold_tree(source, observed_limit, diagnosis_limit):
    """Change exactly the frozen two-field comparison in an in-memory AST.

    Do not change the historical generation prompt, correction limits, validators,
    provenance or recovery logic. Refuse an unrecognized source shape.
    """
    if any(type(x) is not int or x not in LIMITS for x in (observed_limit, diagnosis_limit)):
        raise ValueError('UNREGISTERED_DIAGNOSTIC_THRESHOLD')
    tree = ast.parse(source)
    original = ast.dump(tree, include_attributes=False)
    wanted = ast.parse('len(value[k]) > 240', mode='eval').body
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.GeneratorExp) and len(node.generators) == 1:
            iterator = node.generators[0]
            if (ast.dump(iterator.iter) == ast.dump(ast.parse(
                    "('observed_failure', 'diagnosis')", mode='eval').body)
                    and isinstance(iterator.target, ast.Name) and iterator.target.id == 'k'
                    and not iterator.ifs and not iterator.is_async):
                matches.extend(n for n in ast.walk(node.elt)
                    if isinstance(n, ast.Compare) and ast.dump(n) == ast.dump(wanted))
    if len(matches) != 1:
        raise ValueError('FROZEN_TWO_FIELD_COMPARISON_MISMATCH')
    comparison = matches[0]
    replacement = ast.parse(
        "{'observed_failure': %d, 'diagnosis': %d}[k]" % (observed_limit, diagnosis_limit),
        mode='eval').body
    comparison.comparators[0] = replacement
    changed = deepcopy(tree)
    comparison.comparators[0] = ast.Constant(value=240)
    assert ast.dump(tree, include_attributes=False) == original
    return ast.fix_missing_locations(changed)


def counterfactual_summary(groups):
    """Stop each recorded local sequence at its earliest full-validator acceptance.

    Savings include only subsequent historical draws, never the accepted draw or
    downstream algorithm costs. Missing ungenerated future draws are not invented.
    """
    summary = Counter()
    summary['logical_diagnoses'] = len(groups)
    for group in groups:
        if not 1 <= len(group) <= 3:
            raise ValueError('INVALID_RECORDED_DRAW_COUNT')
        for i, draw in enumerate(group):
            if any(type(draw[k]) is not bool for k in ('original_accepted', 'accepted')):
                raise ValueError('INVALID_ACCEPTANCE_FLAG')
            if any(type(draw[k]) is not int or draw[k] < 0 for k in ('input_tokens', 'output_tokens')):
                raise ValueError('INVALID_USAGE')
            if draw['original_accepted'] and (i != len(group)-1 or not draw['accepted']):
                raise ValueError('HISTORICAL_ACCEPTANCE_NOT_PRESERVED')
        if not group[-1]['original_accepted'] and len(group) != 3:
            raise ValueError('INCOMPLETE_HISTORICAL_EXHAUSTION')
        summary['physical_responses'] += len(group)
        summary['all_response_full_pass'] += sum(d['accepted'] for d in group)
        summary['newly_full_pass_responses'] += sum(d['accepted'] and not d['original_accepted'] for d in group)
        summary['historical_input_tokens'] += sum(d['input_tokens'] for d in group)
        summary['historical_output_tokens'] += sum(d['output_tokens'] for d in group)
        first = next((i for i, d in enumerate(group) if d['accepted']), None)
        if first is None:
            summary['remaining_exhausted_diagnoses'] += 1
            continue
        summary['accepted_diagnoses'] += 1
        summary['first_pass_diagnoses'] += first == 0
        summary['actionable_diagnoses'] += group[first]['disposition'] == 'ACTIONABLE'
        summary['uncertain_diagnoses'] += group[first]['disposition'] == 'UNCERTAIN'
        summary['rescued_exhausted_diagnoses'] += not group[-1]['original_accepted']
        summary['earlier_acceptance_diagnoses'] += first < len(group)-1
        suffix = group[first+1:]
        summary['avoidable_historical_retry_responses'] += len(suffix)
        summary['avoidable_input_tokens'] += sum(d['input_tokens'] for d in suffix)
        summary['avoidable_output_tokens'] += sum(d['output_tokens'] for d in suffix)
    # Keep zero results explicit, including the 240 baseline.
    keys = ('physical_responses', 'all_response_full_pass', 'newly_full_pass_responses',
        'accepted_diagnoses', 'first_pass_diagnoses', 'actionable_diagnoses', 'uncertain_diagnoses',
        'remaining_exhausted_diagnoses', 'rescued_exhausted_diagnoses', 'earlier_acceptance_diagnoses',
        'avoidable_historical_retry_responses', 'avoidable_input_tokens', 'avoidable_output_tokens')
    for key in keys:
        summary.setdefault(key, 0)
    result = dict(summary)
    result['avoidable_usage_tokens'] = result['avoidable_input_tokens'] + result['avoidable_output_tokens']
    result['historical_usage_tokens'] = result['historical_input_tokens'] + result['historical_output_tokens']
    return result
