"""Versioned, deterministic Optimize evidence primitives; no providers or controller."""
from copy import deepcopy
from difflib import SequenceMatcher
from hashlib import sha256
import random
import re

from .schemas import SearchContractError

from ..current_contract import CURRENT_METHOD_VERSION as METHOD
from ..current_contract import OPTIMIZATION_EVIDENCE_POLICY_VERSION as IDENTITY
from ..current_contract import CURRENT_OPTIMIZER_INPUT_SCHEMA as INPUT
from ..current_contract import CURRENT_LAYER1_SEARCH_VERSION as LAYER1
EVIDENCE = 'SHARED_PATTERN_REPRESENTATIVE_SUPPORT_V3'
from ..current_contract import CURRENT_EDIT_MEMORY_VERSION as MEMORY
GRADIENT_INPUT = 'STRUCTURED_SYSTEM_GRADIENT_INPUT_V8'
from ..current_contract import GRADIENT_PROMPT_VERSION as GRADIENT_PROMPT_ID
from ..current_contract import MATH_STRUCTURED_SYSTEM_BINDING_VERSION as BINDING

POLICY = dict(identity=IDENTITY, maximum_support_examples=3,
    candidate_evaluation='complete_optimize_member_accuracy_and_fixed_peer_vote',
    opportunity_eligibility='any_member_wrong_no_accuracy_quota',
    responsibility_fallback='seed_team_state_opportunity_ordinal_random_legal',
    local_parent='frozen_opportunity_start_population',
    structural_draws_per_slot=2, independent_slots=3,
    sampling='seeded_task_type_round_robin_pattern_support_up_to_three',
    validation_feedback='none_within_opportunity',
    selection='fitness_weighted_with_replacement',
    bootstrap='measured_initial_optimize_profiles_zero_llm',
    edit_memory='scoped_single_step_actual_complete_optimize_observations',
    abstention='invalid_output_generation_failure', reference_solution_split='optimize',
    reference_solution_max_chars=4096, candidate_guard='STRUCTURAL_SINGLE_BLOCK_CANDIDATE_CONTRACT_V2',
    prompt_state='STRUCTURED_SYSTEM_PROMPT_V2', mutation='single_block_per_generation',
    editable_blocks=['role','strategy','answer'], response_evidence='observed_content_optional_trajectory',
    answer_extraction='cevolve_hendrycks_math_v1/last_boxed',
    generated_output_recovery='BOUNDED_SINGLE_STEP_OUTPUT_RECOVERY_V4',
    optimizer_output_feedback='TYPED_OPTIMIZER_OUTPUT_FEEDBACK_V1',
    abstraction='SOURCE_CONTEXT_SYMBOLIC_ABSTRACTION_GUARD_V1',
    optimizer_input_encoding='DIRECT_READABLE_REFLECTOR_VIEW_V3',
    layer1_diagnosis='accepted_shared_root_and_actual_parent_plus_verified_group_experience',
    reflector_view='DIRECT_READABLE_REFLECTOR_VIEW_V3',
    reflection_instruction='GROUP_DIAGNOSED_REFLECTION_INSTRUCTION_V1',
    layer1_attempt_feedback='typed_structural_recovery_only',
    layer1_prompt_budget='model_context_only_no_character_budget_v2',
    memory_retrieval='exact_parent_then_latest_measured_edit',
    memory_budget='whole_cards_fit_budget',
    memory_whole_instruction_limit=600,
    memory_relevance='explicit_edit_parent_match_and_scope')

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
    """Text type/nonemptiness only; private evidence never filters candidates."""
    from .system_prompt import SystemPrompt
    if isinstance(prompt, SystemPrompt): prompt = prompt.render()
    return () if isinstance(prompt,str) and prompt.strip() else ('empty_or_nonstring_prompt',)
