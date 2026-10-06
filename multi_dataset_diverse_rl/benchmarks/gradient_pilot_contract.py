"""Current-only Pilot scope and conservative bounds for the unchanged search graph.

An epoch need not be round-robin. Failure counts survive commits. If all counts
start below H, an unseen feasible member has score >= 1/(H+1). A previously
seen member with count T=(4N+1)(H+1) has score strictly below that value, even
with float rounding. Thus it cannot be selected again while any member is
unseen. An epoch, including a commit-interrupted prefix, takes <= M(T+1)
opportunities and ends with counts <= T+1. Strict integer Vote improvement
permits at most N commits; each parent has at most two epoch segments. This
deliberately loose bound is a safety ceiling, never a scientific stop rule.
"""
from copy import deepcopy

from ..search.current_policy import CURRENT_POLICY_BUNDLE
from ..search.schemas import SearchContractError


OBSERVATION_POLICY = 'PILOT_READ_ONLY_STATE_JOURNAL_V1'


def pilot_provider_bounds(parent):
    CURRENT_POLICY_BUNDLE.validate_contract(parent)
    n = parent['low_cost_protocol']['counts']['pilot_optimize']
    shadow = parent['low_cost_protocol']['counts']['pilot_shadow']
    if n != 60 or shadow != 40 or parent['stop_policy'] != 'team_epoch_no_commit_v1':
        raise SearchContractError('PILOT_BOUND_CONTRACT_MISMATCH')
    members, patience = 5, 2
    failure_cap = opportunities = 0
    for _ in range(patience * (n + 1)):
        threshold = (4 * n + 1) * (failure_cap + 1)
        opportunities += members * (threshold + 1)
        failure_cap = threshold + 1
    layer1 = parent['layer1_search_policy']
    # Local metrics, four fixed-peer probes, two Fulls, and one selected Shadow.
    # Bootstrap Shadow is at most one complete incumbent team: exact requests
    # for unchanged peers and previously committed winners are already resolved.
    per_op = layer1['metric_limit'] + layer1['k_local_return'] * layer1['panel_size'] + 2 * n + shadow
    recovery = parent['invalid_recovery_policy']['max_semantic_attempts']
    solver = recovery * (members * (n + shadow) + opportunities * per_op)
    gradients, clusters = n * opportunities, opportunities
    reflections = layer1['max_generations'] * opportunities
    success = solver + gradients + clusters + reflections
    transport = success * (parent['decoding']['transport_retries'] + 1)
    return dict(max_opportunities=opportunities, max_proposals_per_opportunity=layer1['max_generations'],
        solver_per_opportunity=per_op, solver_calls=solver, reflection_calls=reflections,
        pattern_gradient_calls=gradients, pattern_cluster_calls=clusters, pattern_calls=gradients+clusters,
        shadow_solver_calls=recovery * (members * shadow + opportunities * shadow),
        successful_provider_calls=success, transport_attempts=transport,
        bound_proof=dict(identity='CARRIED_FAILURE_EPOCH_BOUND_V1',max_commits=n,
            max_epoch_segments=patience*(n+1),responsibility_upper_bound=4*n,
            failure_cap_recurrence='T=(4*N+1)*(H+1); E=M*(T+1); H_next=T+1',
            initial_failure_cap=0, final_failure_cap=failure_cap,
            scientific_stopper_unchanged=True),
        # Every admitted reservation is checked against the durable hard budget.
        # Gross reservations include released reservations; they are not spending.
        reservation_peak_upper_bound=40_000_000,
        gross_reservation_upper_bound=transport*40_000_000,
        cumulative_charged_token_ceiling=40_000_000)


def derive_current_pilot_contract(parent, *, attempt, binding_path, parent_path,
        parent_sha256, authorization_path, authorization_sha256):
    CURRENT_POLICY_BUNDLE.validate_contract(parent)
    if parent.get('execution_phase') != 'canary' or parent.get('execution_arm') != 'A4':
        raise SearchContractError('CURRENT_PILOT_PARENT_INVALID')
    c = deepcopy(parent)
    c.update(binding_path=binding_path, execution_attempt_id=attempt,
        canary_attempt_id=None, cache_namespace=attempt, execution_phase='pilot',
        pilot_parent_binding_path=parent_path, pilot_parent_binding_sha256=parent_sha256,
        pilot_execution_authorization_path=authorization_path,
        pilot_execution_authorization_sha256=authorization_sha256,
        pilot_observation_policy=OBSERVATION_POLICY,
        continuation_authorization_sha256=authorization_sha256,
        provider_bounds=pilot_provider_bounds(parent),
        search_only_scope=dict(validation_authorized=False,test_authorized=False,
            raw_diagnostic_authorized=False,llm_judge_authorized=False,push_authorized=False))
    c['initial_competence_binding'].update(count=60,
        support_identity=c['low_cost_protocol']['membership_hashes']['pilot_optimize'])
    return c
