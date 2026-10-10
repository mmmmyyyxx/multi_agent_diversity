"""Versioned, deterministic Optimize evidence primitives; no providers or controller."""
from copy import deepcopy
from difflib import SequenceMatcher
from hashlib import sha256
import random
import re

from .schemas import SearchContractError

METHOD = 'unified_team_prompt_search_v2_5_generation_recovery_v1'
IDENTITY = 'GENERATED_OUTPUT_RECOVERY_EVIDENCE_POLICY_V5'
INPUT = 'STRUCTURED_SYSTEM_PATTERN_EDIT_INPUT_V8'
LAYER1 = 'STRUCTURED_EVALUATED_PARENT_SEARCH_V3'
EVIDENCE = 'DISJOINT_QUOTA_FREE_REPAIR_EVIDENCE_V2'
MEMORY = 'BOOTSTRAPPED_FORMAT_BLOCK_EDIT_EFFECT_MEMORY_V3'
GRADIENT_INPUT = 'STRUCTURED_SYSTEM_GRADIENT_INPUT_V5'
GRADIENT_PROMPT_ID = 'STRUCTURED_SYSTEM_GRADIENT_PROMPT_V7'
BINDING = 'MATH_GENERATION_RECOVERY_EVIDENCE_BINDING_V4'

POLICY = dict(identity=IDENTITY, mutation_size=3, search_validation_size=3,
    team_probe_size=6, assigned_repair='selected_pattern_wrong_mutation_ids_seen',
    opportunity_eligibility='any_member_wrong_no_accuracy_quota',
    responsibility_fallback='seed_team_state_opportunity_ordinal_random_legal',
    local_parent='every_contract_legal_evaluated_edit',
    structural_recovery_draws=3, max_generations=6, max_returned_candidates=4,
    metric_limit=42, sampling='seeded_disjoint_no_correct_wrong_quota_v2',
    validation_feedback='measure_after_generation_counts_only_next_iteration',
    selection='validation_net_then_preservation_then_mutation_net_then_generation',
    bootstrap='measured_initial_optimize_profiles_zero_llm',
    edit_memory='actual_diff_and_scope_bound_coverage_v1',
    no_safe_edit='no_candidate_no_success', reference_solution_split='optimize',
    reference_solution_max_chars=4096, candidate_guard='optimizer_dependency_guard_v1',
    prompt_state='STRUCTURED_SYSTEM_PROMPT_V1', mutation='single_block_per_generation',
    editable_blocks=['role','strategy','answer'], response_evidence='observed_content_optional_trajectory',
    answer_extraction='MATH_FLEXIBLE_ANSWER_EXTRACTION_V3',
    generated_output_recovery='BOUNDED_GENERATED_OUTPUT_RECOVERY_V1',
    abstraction='SOURCE_CONTEXT_SYMBOLIC_ABSTRACTION_GUARD_V1')

# Six independent examples: losing at least half of this small panel is a
# catastrophe. Lesser regression stays measurable; unchanged Full/Shadow decide.
PROBE_POLICY = dict(identity='SEEN_ASSIGNED_REPAIR_INDEPENDENT_TEAM_PROBE_V2',
    independent_size=6, minimum_assigned_binary_repairs=1,
    catastrophic_vote_net_loss=3, catastrophic_target_net_loss=3,
    catastrophic_collateral_loss=3, invalid_math='incorrect_observation_only',
    rank=['vote_delta','target_delta','negative_collateral_loss','negative_invalid_delta','candidate_id'],
    max_full=2, seen_repair_inference='SEEN_ASSIGNED_REPAIR_NOT_GENERALIZATION')


def frozen_policy(value):
    if value is None:
        raise SearchContractError('CURRENT_V25_OPTIMIZATION_EVIDENCE_REQUIRED')
    if value != POLICY or any(type(value.get(k)) is not type(v) for k, v in POLICY.items()):
        raise SearchContractError('OPTIMIZATION_EVIDENCE_POLICY_MISMATCH')
    return deepcopy(POLICY)


def prompt_id(text):
    return sha256(text.encode()).hexdigest()


def actual_diff(parent, child):
    """Lossless deterministic character spans, private only. No clause attribution."""
    from .system_prompt import SystemPrompt
    if isinstance(parent, SystemPrompt): parent = parent.serialize()
    if isinstance(child, SystemPrompt): child = child.serialize()
    return [dict(operation=op, parent_span=[a,b], child_span=[c,d],
        removed=parent[a:b], added=child[c:d])
        for op,a,b,c,d in SequenceMatcher(None,parent,child,autojunk=False).get_opcodes()
        if op != 'equal']


def coverage_effect(before, after, *, scope):
    if set(before) != set(after) or not before:
        raise SearchContractError('EDIT_EFFECT_MEMBERSHIP_MISMATCH')
    ids=sorted(before)
    if any(type(r.get(k)) is not bool for rows in (before,after) for r in rows.values()
            for k in ('correct','valid')):
        raise SearchContractError('EDIT_EFFECT_BOOLEAN_REQUIRED')
    if any(r['correct'] and not r['valid'] for rows in (before,after) for r in rows.values()):
        raise SearchContractError('EDIT_EFFECT_INVALID_CANNOT_BE_CORRECT')
    fixed=[x for x in ids if not before[x]['correct'] and after[x]['correct']]
    broken=[x for x in ids if before[x]['correct'] and not after[x]['correct']]
    return dict(scope=scope, membership=ids, fixed_ids=fixed, broken_ids=broken,
        parent_invalid_count=sum(not r['valid'] for r in before.values()),
        child_invalid_count=sum(not r['valid'] for r in after.values()),
        valid_to_invalid_ids=[x for x in ids if before[x]['valid'] and not after[x]['valid']],
        invalid_to_valid_wrong_ids=[x for x in ids if not before[x]['valid'] and after[x]['valid'] and not after[x]['correct']],
        invalid_to_valid_correct_ids=[x for x in ids if not before[x]['valid'] and after[x]['correct']],
        repeated_format_failure_ids=[x for x in ids if after[x].get('repeated_format_failure') is True],
        retained_ids=[x for x in ids if before[x]['correct'] and after[x]['correct']],
        valid_wrong_ids=[x for x in ids if after[x]['valid'] and not after[x]['correct']],
        invalid_ids=[x for x in ids if not after[x]['valid']],
        invalid_delta=sum(not r['valid'] for r in after.values())-sum(not r['valid'] for r in before.values()),
        member_delta=len(fixed)-len(broken))


def rotated_correct(rows, *, seed, member, ordinal):
    """Rotate current committed correct pool. Ordinal counts member opportunities."""
    correct=sorted((r for r in rows if r.signals['target_member_correct']),key=lambda r:r.example_id)
    random.Random(f'{seed}:{member}:preservation-v1').shuffle(correct)
    if not correct:
        return []
    offset=(2*ordinal)%len(correct)
    return correct[offset:]+correct[:offset]


def executability_checks(prompt, *, private_texts=()):
    """Bounded known dependency/leakage checks, not a semantic proof."""
    from .system_prompt import SystemPrompt
    if isinstance(prompt, SystemPrompt): prompt = prompt.render()
    checks=[]
    if re.search(r'\b(?:selected_gradient|solver_trajectory|retrieved_memory|reference_solution|'
            r'ground[_ -]?truth|gold[_ -]?answer)\b|\b(?:read|consult|copy|access|use)\b[^.;\n]{0,55}'
            r'\b(?:gradient|memory|reference answer|correct answer key)\b',prompt,re.I):
        checks.append('optimizer_only_dependency')
    if re.search(r'\b(?:if the (?:problem|question) (?:is|equals)|answer lookup|memorized answers?)\b',prompt,re.I):
        checks.append('example_answer_lookup')
    normalized=' '.join(prompt.casefold().split())
    for text in private_texts:
        # Long verbatim fragments only; individual mathematical constants are legal.
        words=str(text or '').split()
        if any(' '.join(words[i:i+8]).casefold() in normalized for i in range(max(0,len(words)-7))):
            checks.append('private_solution_or_response_copy');break
    return tuple(checks)
