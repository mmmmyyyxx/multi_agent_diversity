"""Closed, independently versioned three-arm mechanism study."""
from copy import deepcopy

from ..benchmarks.math_optimizer_generation import (
    optimizer_generation_contract, pattern_cluster_generation_contract,
)
from ..benchmarks.math_solver_decoding import solver_decoding_contract
from ..benchmarks.math_response_evidence import trajectory_policy
from ..provider_routing import routing_contract
from ..search.textual_gradients import GRADIENT_POLICY

IDENTITY = 'SINGLE_PROMPT_GRADIENT_PATTERN_STUDY_V1'
ARMS = ('full_errors', 'batch_errors', 'gradient_pattern')
STUDY = 'single_prompt_gradient_pattern_v1'


def policy():
    return dict(identity=IDENTITY, arms=list(ARMS), seeds=[88, 89, 90],
        rounds=20, error_batch_size=3,
        batch_identity='INDEPENDENT_UNIFORM_WRONG_WITHOUT_REPLACEMENT_V1',
        batch_universe='current_prompt_wrong_optimize',
        batch_order='frozen_optimize_order_after_seeded_selection',
        gradient_policy=deepcopy(GRADIENT_POLICY),
        pattern_feedback='all_legal_patterns_no_team_ranking',
        optimizer_candidates_per_round=1, optimizer_structural_draws=2,
        acceptance='format_and_executability_only_unconditional_replacement',
        final_selection='last_round_prompt', memory=False, population=False,
        no_errors='carry_prompt_and_continue_fixed_rounds',
        invalid_candidate='two_structural_draws_then_carry_and_continue',
        provider_failure='abort_phase_no_transport_retry_no_resume',
        no_actionable_pattern='retain_explicit_empty_diagnostics_generate_one_candidate',
        provider_routing_policy=routing_contract(),
        solver_decoding_policy=solver_decoding_contract(),
        optimizer_generation_policy=optimizer_generation_contract(),
        cluster_generation_policy=pattern_cluster_generation_contract(),
        trajectory_policy=trajectory_policy(),
        solver_concurrency=16, gradient_concurrency=8,
        provider_timeout_seconds=120,
        solver_semantic_draws=4, first_solver_capacity=3600,
        length_recovery_capacity=6144,
        optimizer_input_limit=991808, optimizer_context_limit=1000000,
        input_estimate='serialized_utf8_bytes_plus_1024',
        cumulative_token_ceiling=None,
        runtime_environment='runs/math_v2_1/runtime',
        runtime_interpreter='runs/math_v2_1/runtime/Scripts/python.exe',
        provider_sdk_version='2.37.0',
        token_budget_identity='USER_SELECTED_UNCAPPED_CUMULATIVE_TOKENS_FINITE_CALLS_V1',
        validation='post_search_all_twenty_one_checkpoints_solver_only',
        test='post_search_final_prompt_only_separately_authorized',
        paired_cache='same_seed_split_round_messages_and_sampling_share_realization',
        optimizer_cache=False, semantic_strategy_judge=False)


def call_ceiling(phase):
    p = policy()
    if phase == 'train':
        return (9 * (p['rounds'] + 1) * 150 * 4
                + 9 * p['rounds'] * 2
                + 3 * p['rounds'] * p['error_batch_size'] * 3
                + 3 * p['rounds'] * 3)
    if phase == 'validation':
        return 9 * (p['rounds'] + 1) * 300 * 4
    if phase == 'test':
        return 9 * 300 * 4
    raise ValueError('STUDY_PHASE_INVALID')
