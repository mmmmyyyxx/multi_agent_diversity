"""Closed recovery categories for rejected model outputs, never integrity faults."""
from copy import deepcopy
from .schemas import SearchContractError

IDENTITY = 'BOUNDED_GENERATED_OUTPUT_RECOVERY_V1'
POLICY = dict(identity=IDENTITY, gradient_draws_per_example=3,
    cluster_draws_per_opportunity=3, reflection_draws_per_opportunity=6,
    rejected_text_reusable=False, exhausted_gradient='NONACTIONABLE_EXHAUSTED',
    invalid_pattern='known_support_to_unassigned',
    integrity_errors='fatal_without_regeneration', unknown_errors='fatal',
    truncated_output='charged_rejected_draw_no_capacity_change')

class GeneratedOutputFailure(SearchContractError):
    """A provider response was accounted for but cannot enter optimizer state."""

_CATEGORIES = {
    'gradient': frozenset({'REFERENCE_GRADIENT_OUTPUT_INVALID',
        'PATTERN_GRADIENT_EXTRACTION_INVALID', 'PATTERN_DISCOVERY_EXAMPLE_LEAKAGE',
        'PATTERN_DISCOVERY_INVALID_ABSTRACTION', 'GENERATED_OUTPUT_TRUNCATED'}),
    'cluster': frozenset({'PATTERN_GRADIENT_CLUSTER_INVALID',
        'GENERATED_OUTPUT_TRUNCATED'}),
    'pattern': frozenset({'PATTERN_GRADIENT_CLUSTER_INVALID',
        'PATTERN_DISCOVERY_EXAMPLE_LEAKAGE', 'PATTERN_DISCOVERY_INVALID_ABSTRACTION'}),
    'reflection': frozenset({'GENERATED_OUTPUT_TRUNCATED'}),
    'block': frozenset({'SYSTEM_PROMPT_BLOCK_INVALID', 'SYSTEM_PROMPT_OVER_LENGTH',
        'SYSTEM_PROMPT_TARGET_BLOCK_INVALID'}),
}

def recoverable_output(error, stage):
    if stage not in _CATEGORIES:
        raise SearchContractError('GENERATION_RECOVERY_STAGE_UNBOUND')
    # The typed marker distinguishes post-response content rejection from an
    # identically worded exception injected by a provider or controller.
    return isinstance(error, GeneratedOutputFailure) and str(error) in _CATEGORIES[stage]

def generation_failure(category):
    if category not in set().union(*_CATEGORIES.values()):
        raise SearchContractError('GENERATION_RECOVERY_CATEGORY_UNBOUND')
    return GeneratedOutputFailure(category)

def frozen_recovery(value):
    if value != POLICY or any(type(value.get(k)) is not type(v) for k,v in POLICY.items()):
        raise SearchContractError('GENERATION_RECOVERY_POLICY_MISMATCH')
    return deepcopy(POLICY)
