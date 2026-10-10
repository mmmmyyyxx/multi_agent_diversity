"""Offline postmortem and fresh B-only baseline design, outside current search."""
from collections import Counter, defaultdict
import hashlib
import re

from .math_baseline_calibration import (
    digest, balanced_box, native_extract, primary_correctness, declarations,
)
from .math_calibration_plan import ROLE, STRATEGY, ANSWERS, SAMPLING, request_for
from ..benchmarks.math_prediction_validity import classify_payload
from ..benchmarks.math_domain_v2 import domain_matrix
from ..benchmarks.experiment_splits import quotas, subject
from ..search.scientific_aggregation import equivalence_classes
from ..search.schemas import SearchContractError
from ..search.system_prompt import SystemPrompt, team_identity
from ..governance.token_accounting import reservation

COMPAT_ID = 'MATH_TERMINAL_DISPLAY_BOX_COMPAT_DIAGNOSTIC_V1'
PLAN_ID = 'MATH_B_FIVE_MEMBER_CALIBRATION_V1'
TEAM_ID = 'MATH_IDENTICAL_CALIBRATED_B_TEAM_SEED_V1'
CAP = 1_000_000

# Owner-reviewed first erroneous step categories, tied to the sealed Stage 1
# checkpoint. These are repair hypotheses, not claims of repair effectiveness.
RESIDUAL_REVIEW = {
    '977cc7a11bcd8ddf3ecc0884e0c534c97b6959df917beae2fb3a46666cb44a0b':
        ('STRUCTURAL_STRATEGY', 'COMBINATORIAL_OVERCOUNT_AND_GEOMETRIC_EXISTENCE'),
    '4a56e1c92933a1e0720f0d6f6775611014530f7eb8ced2c06797e9ed1f8e98a8':
        ('STRUCTURAL_STRATEGY', 'COORDINATE_CONSTRAINT_SUBSTITUTION'),
    '0fbd58c390431d82384d6f7e3b1a88f2a29afd5c3c677532d5893978ddfbf39b':
        ('STRUCTURAL_STRATEGY', 'GEOMETRIC_INCIDENCE_CONSTRAINT'),
    'c195c15966da10f3a6f0c7f0aee05491001e03efdb7afb1906cdf87048690197':
        ('STRUCTURAL_STRATEGY', 'FORMULA_PRECONDITION_AND_NUMERIC_CHECK'),
    '700a20c4576b304b79647999d3b89ed82887a1f73a34417884fb8ffd6378d128':
        ('MIXED_STRATEGY_AND_TRUNCATION', 'INVARIANT_LOSS_AND_REPEATED_REASONING'),
    '114283fec17b8e2a9a5ebcb1e03869f321896f0c7287b532b2d89fed3b3a8702':
        ('STRUCTURAL_STRATEGY', 'QUANTIFIER_AND_NECESSARY_SUFFICIENT_ROOTS'),
    '5a84be8c4572e1841dda1317041d48a3f5437956d249c06e8a23d62eaa86c5d2':
        ('STRUCTURAL_STRATEGY', 'POSITION_DEPENDENT_COUNTING_WITH_LEADING_ZERO'),
    'aa0bbc140215e16e29fc7155b7d16682f4275372538128772fbf8840e5573d3b':
        ('INTERFACE_ONLY', 'UNICODE_RADICAL_SERIALIZATION_COMPATIBILITY'),
    '7c6f7c9cde95a59ec1445e37173f08efc88769fc3dbe6cf685c126344d2a7b18':
        ('LOCAL_VERIFICATION', 'ENUMERATED_SET_CARDINALITY_CHECK'),
    '96074ed7ebfdab2a0c6239c86778783c70d01b0fbabbc2b4b2d22ad143d731f8':
        ('LOCAL_VERIFICATION', 'NUMERIC_FUNCTION_EVALUATION_AND_EXACT_IDENTITY'),
    '455397138b1eb7495bfa285f024b5e097f6a260b88ac15e5a0169eb61e8ad67d':
        ('INTERFACE_ONLY', 'LATEX_MATRIX_ENVIRONMENT_LEXICAL_GUARD'),
}


def display_box_diagnostic(text, finish_reason='stop'):
    """Gold-blind compatibility probe. Cannot supply an official native score.

    Admit one complete terminal display block, or preserve a valid old native
    box. Explicit competing declarations, malformed boundaries and truncation
    remain invalid. No scan of intermediate values for gold matching.
    """
    result = dict(policy_identity=COMPAT_ID, usage_scope='DIAGNOSTIC_ONLY',
                  valid=False, payload=None, reason='NO_COMPLETE_TERMINAL_BOX')
    if finish_reason != 'stop':
        return result | dict(reason='NON_STOP_FINISH')
    lines = [line.strip() for line in (text or '').splitlines() if line.strip()]
    display = None
    fenced = False
    for line in lines:
        if line.startswith('```'):
            fenced = not fenced
        if line == '\\[':
            if display is not None:
                return result | dict(reason='MALFORMED_DISPLAY_BOUNDARY')
            display = '\\['
        elif line == '\\]':
            if display != '\\[':
                return result | dict(reason='MALFORMED_DISPLAY_BOUNDARY')
            display = None
        elif line == '$$':
            if display not in (None, '$$'):
                return result | dict(reason='MALFORMED_DISPLAY_BOUNDARY')
            display = '$$' if display is None else None
    if display is not None or fenced:
        return result | dict(reason='MALFORMED_DISPLAY_BOUNDARY')
    old = native_extract('C', text, finish_reason)
    payload = old['payload'] if old['valid'] else None
    if payload is None:
        if not lines or lines[-1] not in {'\\]', '$$'}:
            return result
        opening = '\\[' if lines[-1] == '\\]' else '$$'
        starts = [i for i, line in enumerate(lines[:-1]) if line == opening]
        if not starts:
            return result
        start = starts[-1]
        inside = lines[start + 1:-1]
        if not inside or any(line in {'\\[', '\\]', '$$'} for line in inside):
            return result
        payload = balanced_box('\n'.join(inside))
    if payload is None:
        return result
    reason = classify_payload(payload)
    if reason:
        return result | dict(reason=reason)
    # Competing explicit declarations are checked independently of any gold.
    _, items = declarations(text)
    values = list(dict.fromkeys([payload] + [i['payload'] for i in items]))
    for value in values[1:]:
        relation = domain_matrix((payload, value))
        if not all(relation['valid']) or not relation['equivalence'][0][1]:
            return result | dict(reason='COMPETING_OR_UNPARSEABLE_DECLARATION')
    return result | dict(valid=True, payload=payload, reason=None)


def grade_compatibility(extracted, reference):
    if extracted.get('policy_identity') != COMPAT_ID or extracted.get('usage_scope') != 'DIAGNOSTIC_ONLY':
        raise ValueError('COMPATIBILITY_DIAGNOSTIC_IDENTITY_REQUIRED')
    if not extracted['valid']:
        return None
    relation = domain_matrix((reference, extracted['payload']))
    if not relation['valid'][0]:
        raise ValueError('REFERENCE_UNSCORABLE')
    return bool(relation['valid'][1] and relation['equivalence'][0][1])


def b_residual_catalog(logicals):
    catalog = []
    for row in logicals:
        if row['arm'] != 'B':
            continue
        final = row['attempts'][-1]
        native = native_extract('B', final['response']['text'], final['response']['finish_reason'])
        if native != final['native']:
            raise ValueError('STAGE1_NATIVE_REPLAY_MISMATCH')
        if primary_correctness(native, row['reference']):
            continue
        key = row['example_id_sha256']
        if key not in RESIDUAL_REVIEW:
            raise ValueError('UNREVIEWED_RESIDUAL')
        group, category = RESIDUAL_REVIEW[key]
        catalog.append(dict(example_id_sha256=key, native_valid=native['valid'],
            native_invalid_reason=native['invalid_reason'], repair_group=group,
            first_error_category=category, subject=row['subject'], level=row['level'],
            evidence='OWNER_REVIEW_OF_SEALED_RESPONSE_AND_REFERENCE',
            repair_effectiveness='UNTESTED'))
    if {r['example_id_sha256'] for r in catalog} != set(RESIDUAL_REVIEW):
        raise ValueError('RESIDUAL_MEMBERSHIP_MISMATCH')
    if sum(r['native_valid'] for r in catalog) != 8:
        raise ValueError('RESIDUAL_NATIVE_COUNTS_MISMATCH')
    return catalog


def choose_fresh_membership(rows, excluded_ids):
    if len({r['stable_example_id'] for r in rows}) != len(rows):
        raise ValueError('DUPLICATE_OPTIMIZE_IDS')
    excluded = set(excluded_ids)
    pool = [r for r in rows if r['stable_example_id'] not in excluded]
    if len(pool) < 60:
        raise ValueError('INSUFFICIENT_DISJOINT_OPTIMIZE_POOL')
    groups = defaultdict(list)
    for row in pool:
        groups[(subject(row), row['content']['level'])].append(row)
    counts = quotas({k: len(v) for k, v in groups.items()}, 60)
    chosen = set()
    for key, group in groups.items():
        ordered = sorted(group, key=lambda r: (digest([PLAN_ID, 82, r['stable_example_id']]), r['stable_example_id']))
        chosen.update(r['stable_example_id'] for r in ordered[:counts[key]])
    selected = [r for r in rows if r['stable_example_id'] in chosen]
    assert len(selected) == 60 and not chosen & excluded
    return selected, dict(seed=82, n=60, available_after_exclusion=len(pool),
        selection='SUBJECT_LEVEL_LARGEST_REMAINDER_LEXICAL_TIES_SHA256_PLAN_SEED_ID',
        order='ORIGINAL_OPTIMIZE_ORDER', outcome_dependent=False,
        subject_counts=dict(Counter(subject(r) for r in selected)),
        level_counts=dict(Counter(r['content']['level'] for r in selected)),
        uncovered_strata=[s+'|'+l for (s,l), n in sorted(counts.items()) if n == 0])


def initial_team():
    prompt = SystemPrompt(ROLE, STRATEGY, ANSWERS['B'])
    return dict(team_version=TEAM_ID, initial_team_data_dependency='NONE',
        ordered_member_ids=list(range(5)), ordered_team_sha256=team_identity([prompt]*5),
        semantic_contract='FIVE_EQUAL_IDENTICAL_B_MEMBERS_INDEPENDENT_REALIZATIONS',
        members=[dict(member_id=i, prompt=prompt.to_dict(),
            block_hashes={k: prompt.block_hash(k) for k in ('role','strategy','answer')},
            prompt_sha256=prompt.prompt_hash, rendered_system_sha256=prompt.system_hash) for i in range(5)])


def membership_identity(selected):
    return [dict(example_id_sha256=hashlib.sha256(r['stable_example_id'].encode()).hexdigest(),
        content_sha256=r['content_sha256'], input_sha256=r['input_sha256'],
        reference_sha256=hashlib.sha256(r['reference_final_answer'].encode()).hexdigest(),
        subject=subject(r), level=r['content']['level']) for r in selected]


def b5_budget(selected):
    bounds = [reservation(request_for('B', r['content']['problem'], 6144)) for r in selected]
    input_bound = 5 * 4 * sum(b['input_upper_bound'] for b in bounds)
    output_bound = 300 * (3600 + 3 * 6144)
    return dict(proposed_charged_plus_reserved_cap=CAP, authorization='PENDING_FRESH_EXACT_SCOPE',
        logical_requests=300, successful_draws_upper=1200, physical_attempts_upper=25200,
        first_draw_output_upper=1080000, successful_output_upper=output_bound,
        successful_input_reservation_upper=input_bound,
        no_transport_failure_reserved_envelope=input_bound+output_bound,
        all_transport_attempts_conservatively_charged_envelope=21*(input_bound+output_bound),
        parent_b_rate_scenario_tokens=71376*5, parent_b_first_draw_scenario_tokens=43144*5,
        derivation='ROUND_2X_PARENT_B_CHARGED_RATE_TO_1M_WITH_RESERVATION_HEADROOM',
        complete_panel_guaranteed=False, currency_cost='UNKNOWN_NO_BILLING_QUERY',
        input_bound='SERIALIZED_UTF8_BYTES_PLUS_4096', old_balance_reusable=False)


def vote_selection(natives):
    """Exactly the frozen equivalence plurality physics; no reference argument."""
    if len(natives) != 5:
        raise ValueError('FIVE_MEMBERS_REQUIRED')
    if any(n.get('usage_scope') != 'ARM_NATIVE' or n.get('policy_identity') != 'MATH_EXPLICIT_FINAL_ANSWER_EXTRACTION_V2' for n in natives):
        raise ValueError('B_NATIVE_ONLY')
    indices = [i for i, n in enumerate(natives) if n['valid']]
    payloads = tuple(natives[i]['payload'] for i in indices)
    try:
        relation = domain_matrix(payloads) if payloads else dict(valid=[],equivalence=[])
        if not all(relation['valid']):
            raise SearchContractError('EQUIVALENCE_RELATION_INCONSISTENT')
        lookup = {p: i for i, p in enumerate(payloads)}
        groups = equivalence_classes(payloads, lambda a,b: relation['equivalence'][lookup[a]][lookup[b]])
    except SearchContractError as error:
        return dict(winner_member=None, reason=str(error), classes=[], invalid_abstentions=5-len(indices))
    ordered = sorted(groups, key=lambda g: (-len(g), g[0]))
    tie = len(ordered)>1 and len(ordered[0])==len(ordered[1])
    winner = indices[ordered[0][0]] if ordered and not tie else None
    return dict(winner_member=winner, reason='TIE' if tie else ('ALL_INVALID' if not ordered else None),
        classes=[[indices[i] for i in g] for g in groups], invalid_abstentions=5-len(indices))


def panel_metrics(natives, reference):
    selection = vote_selection(natives)  # Freeze gold-blind selection first.
    correct = [primary_correctness(n, reference) for n in natives]
    winner = selection['winner_member']
    return dict(selection=selection, member_correct=correct, member_valid=[n['valid'] for n in natives],
        correct_member_count=sum(correct), oracle_correct=any(correct),
        vote_correct=correct[winner] if winner is not None else False,
        unique_cover_member=correct.index(True) if sum(correct)==1 else None)
