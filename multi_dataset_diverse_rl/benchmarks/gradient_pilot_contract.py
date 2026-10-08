"""V2.2 Pilot is closed until a new finite operational bound is frozen.

The V2.1 integer-Vote proof lives in legacy/gradient_pilot_contract_v21.py.
Scientific no-commit epochs remain unchanged; they do not prove a resource bound.
"""
import math

from ..search.schemas import SearchContractError

OBSERVATION_POLICY = 'PILOT_READ_ONLY_STATE_JOURNAL_V1'
PILOT_BOUND_BLOCKER = 'TARGET_OR_TEAM_PROGRESS_PILOT_BOUND_NOT_FROZEN'


def progress_potential(vote, member_scores, *, optimize_count):
    """Integer potential for complete binary, fixed-peer V2.2 measurements.

    A TEAM commit can lose at most N selected-member successes. Weight N+1
    therefore guarantees an increase even for Vote +1 / target -N. TARGET
    commits strictly increase the unweighted member sum at constant Vote.
    This is a proof utility, not an execution authorization or stopping rule.
    """
    if type(optimize_count) is not int or optimize_count <= 0:
        raise SearchContractError('INVALID_POTENTIAL_SUPPORT')
    scores = tuple(member_scores)
    if len(scores) != 5 or any(isinstance(v, bool) or not math.isfinite(v)
            or v != int(v) or not 0 <= v <= optimize_count for v in (vote, *scores)):
        raise SearchContractError('INVALID_BINARY_POTENTIAL_MEASUREMENT')
    return (optimize_count + 1) * int(vote) + sum(map(int, scores))


def finite_bound_assessment(*, optimize_count=60, initial_vote=None, initial_scores=None):
    """Prove finiteness without treating an impractical ceiling as runnable.

    The failure-discount recurrence is represented symbolically: expanding it
    for V2.2 would exceed Python's default JSON integer conversion limit. This
    function leaves the production binding and authorization gates closed.
    """
    if type(optimize_count) is not int or optimize_count <= 0:
        raise SearchContractError('INVALID_POTENTIAL_SUPPORT')
    n, members, patience = optimize_count, 5, 2
    maximum = (n + 1) * n + members * n
    if (initial_vote is None) != (initial_scores is None):
        raise SearchContractError('INCOMPLETE_INITIAL_POTENTIAL')
    initial = 0 if initial_vote is None else progress_potential(
        initial_vote, initial_scores, optimize_count=n)
    commits = maximum - initial
    segments = patience * (commits + 1)
    factor = 4 * n + 1
    # Total opportunities >= M*a**S for this conservative recurrence. This
    # lower bound describes the *ceiling*, not required actual runtime work.
    ceiling_min_digits = math.floor(math.log10(members) + segments * math.log10(factor)) + 1
    return dict(identity='V2_2_WEIGHTED_PROGRESS_FINITE_BOUND_ASSESSMENT_V1',
        optimize_count=n, members=members, potential='(N+1)*Vote+sum(member_scores)',
        potential_maximum=maximum, initial_potential=initial,
        max_commits=commits, max_epoch_segments=segments,
        responsibility_upper_bound=4*n, failure_cap_factor=factor,
        failure_cap_recurrence='H_next=(4*N+1)*H+(4*N+2); E=5*H_next',
        symbolic_opportunity_ceiling='5*(a+1)/(a-1)*(a*(a**S-1)/(a-1)-S)',
        symbolic_parameters=dict(a=factor, S=segments),
        conservative_opportunity_ceiling_min_decimal_digits=ceiling_min_digits,
        finite_commit_bound_proved=True, finite_epoch_bound_proved_in_exact_arithmetic=True,
        resource_realistic_completion_bound_established=False,
        float_discount_implementation_not_proved_at_symbolic_ceiling=True,
        scientific_stopper_unchanged=True, execution_gate='HOLD', blocker=PILOT_BOUND_BLOCKER)


OPERATIONAL_BOUND_ID = 'V2_2_FIXED_HORIZON_OPERATIONAL_PILOT_V1'


def pilot_provider_bounds(parent, *, max_opportunities=None, token_ceiling=None):
    if max_opportunities is None or token_ceiling is None:
        raise SearchContractError(PILOT_BOUND_BLOCKER)
    from ..search.current_policy import CURRENT_POLICY_BUNDLE
    CURRENT_POLICY_BUNDLE.validate_contract(parent)
    if (type(max_opportunities) is not int or max_opportunities <= 0
            or type(token_ceiling) is not int or not 0 < token_ceiling < 40_000_000
            or parent['low_cost_protocol']['counts']['pilot_optimize'] != 60
            or parent['shadow_count'] != 40 or parent['stop_policy'] != 'team_epoch_no_commit_v1'):
        raise SearchContractError('OPERATIONAL_PILOT_BOUND_INVALID')
    k=max_opportunities; n=60; shadow=40; layer=parent['layer1_search_policy']
    logical=layer['metric_limit']+layer['k_local_return']*layer['panel_size']+2*n+shadow
    recovery=parent['invalid_recovery_policy']['max_semantic_attempts']
    gradient_recovery=parent['gradient_recovery_policy']['max_physical_attempts_per_wrong']
    solver=recovery*(5*(n+shadow)+k*logical)
    gradients=k*n*gradient_recovery; clusters=k; reflections=k*layer['max_generations']
    success=solver+gradients+clusters+reflections
    transport=success*(parent['decoding']['transport_retries']+1)
    return dict(max_opportunities=k,max_proposals_per_opportunity=layer['max_generations'],
        solver_per_opportunity=logical,physical_solver_per_opportunity=recovery*logical,
        solver_calls=solver,reflection_calls=reflections,pattern_gradient_calls=gradients,
        pattern_cluster_calls=clusters,pattern_calls=gradients+clusters,
        shadow_solver_calls=recovery*(5*shadow+k*shadow),successful_provider_calls=success,
        transport_attempts=transport,reservation_peak_upper_bound=token_ceiling,
        gross_reservation_upper_bound=transport*token_ceiling,
        attempt_charged_token_ceiling=token_ceiling,cumulative_charged_token_ceiling=40_000_000,
        bound_proof=dict(identity=OPERATIONAL_BOUND_ID,scientific_stopper_unchanged=True,
            guarantees_saturation=False,bootstrap_logical_solver_calls=500,
            per_op_logical_solver=dict(local=36,probe=24,full=120,shadow=40),
            solver_semantic_multiplier=recovery,gradient_semantic_multiplier=gradient_recovery,
            transport_multiplier=parent['decoding']['transport_retries']+1,
            token_bound='PRE_TRANSPORT_EXACT_RESERVATION_AND_CUMULATIVE_ATTEMPT_ADMISSION',
            gross_reservations_are_not_spending=True))


def derive_current_pilot_contract(parent, *, attempt, binding_path, parent_path,
        parent_sha256, authorization_path, authorization_sha256,
        max_opportunities=None, token_ceiling=None):
    if max_opportunities is None or token_ceiling is None:
        raise SearchContractError(PILOT_BOUND_BLOCKER)
    from copy import deepcopy
    from .. import current_contract as versions
    c=deepcopy(parent)
    if (not attempt.startswith('math_v2_2_gradient_pattern_A4_seed81_pilot_attempt')
            or attempt in (parent['execution_attempt_id'],parent['cache_namespace'])
            or binding_path==parent['binding_path'] or parent['execution_arm']!='A4' or parent['seeds']!=[81]):
        raise SearchContractError('CURRENT_PILOT_FRESH_IDENTITY_REQUIRED')
    c.update(identity=versions.MATH_V2_2_EXECUTION_BINDING_VERSION,
        method_identity=versions.UNIFIED_TEAM_PROMPT_SEARCH_V2_2_VERSION,
        transition_policy=versions.UNIFIED_TARGET_OR_TEAM_TRANSITION_VERSION,
        execution_phase='pilot',execution_attempt_id=attempt,cache_namespace=attempt,
        binding_path=binding_path,canary_attempt_id=None,
        baseline_scientific_settings_path=parent_path,baseline_scientific_settings_sha256=parent_sha256,
        current_user_scope_path=authorization_path,current_user_scope_sha256=authorization_sha256,
        continuation_authorization_sha256=authorization_sha256,
        operational_pilot=dict(identity=OPERATIONAL_BOUND_ID,max_opportunities=max_opportunities,
            token_ceiling=token_ceiling,scientific_stopper='team_epoch_no_commit_v1',guarantees_saturation=False),
        search_only_scope=dict(validation_authorized=False,test_authorized=False,
            raw_diagnostic_authorized=False,llm_judge_authorized=False,push_authorized=True))
    c.pop('method_implementation_sha',None)
    c['provider_bounds']=pilot_provider_bounds(c,max_opportunities=max_opportunities,token_ceiling=token_ceiling)
    return c
