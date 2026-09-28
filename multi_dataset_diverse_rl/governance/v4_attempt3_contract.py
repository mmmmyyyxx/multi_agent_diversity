"""Prospective observation and operational bounds for V4 Seed81 attempt3."""

SAMPLE_UNIT = "LAYER1_RETURNED_STRICT_POSITIVE_CANDIDATE"
SAMPLE_TARGET = 5
MAX_OPPORTUNITIES = 10
PROPOSAL_CEILING = 20
MAX_RETURNED_PER_OPPORTUNITY = 4
# Pinned GEPA checks its 36 metric-call limit between iterations. Every
# iteration evaluates a three-row parent minibatch; an accepted iteration can
# add three child rows and a twelve-row validation. Seed validation costs 12.
MAX_PROPOSALS_PER_OPPORTUNITY = 12
MAX_LOCAL_SOLVER_CALLS_PER_OPPORTUNITY = 12 + 36 + 18
MAX_REFLECTION_CALLS_PER_OPPORTUNITY = 12
TEAM_MINIBATCH_CALLS_PER_OPPORTUNITY = 4 * 12
DIAGNOSTIC_FULL_CALLS_PER_OPPORTUNITY = 4 * 100
ORDINARY_FULL_CALLS_PER_OPPORTUNITY = 2 * 100
SHADOW_CALLS_PER_OPPORTUNITY = 50
INITIALIZATION_CALLS = 5 * 100
MAX_LOGICAL_PROVIDER_CALLS = INITIALIZATION_CALLS + MAX_OPPORTUNITIES * (
    MAX_LOCAL_SOLVER_CALLS_PER_OPPORTUNITY
    + MAX_REFLECTION_CALLS_PER_OPPORTUNITY
    + TEAM_MINIBATCH_CALLS_PER_OPPORTUNITY
    + DIAGNOSTIC_FULL_CALLS_PER_OPPORTUNITY
    + SHADOW_CALLS_PER_OPPORTUNITY
)
MAX_ATTEMPTS_PER_LOGICAL_CALL = 3 + 20
SUCCESSFUL_PROVIDER_CEILING = 7000
TRANSPORT_ATTEMPT_CEILING = 150000

assert MAX_LOGICAL_PROVIDER_CALLS == 6260
assert MAX_LOGICAL_PROVIDER_CALLS < SUCCESSFUL_PROVIDER_CEILING
assert MAX_LOGICAL_PROVIDER_CALLS * MAX_ATTEMPTS_PER_LOGICAL_CALL < TRANSPORT_ATTEMPT_CEILING


def diagnostic_contract() -> dict[str, object]:
    return {
        "sample_unit": SAMPLE_UNIT,
        "returned_candidate_target": SAMPLE_TARGET,
        "within_opportunity_overshoot": "retain_and_evaluate_all_returned_candidates",
        "max_opportunities": MAX_OPPORTUNITIES,
        "reflection_proposal_ceiling": PROPOSAL_CEILING,
        "proposal_preopportunity_guard": MAX_PROPOSALS_PER_OPPORTUNITY,
        "successful_provider_ceiling": SUCCESSFUL_PROVIDER_CEILING,
        "transport_attempt_ceiling": TRANSPORT_ATTEMPT_CEILING,
        "mandatory_full": True,
        "diagnostic_full_is_admission_inert": True,
    }


def resource_upper_bounds() -> dict[str, int]:
    return {
        "returned_candidates_per_opportunity": MAX_RETURNED_PER_OPPORTUNITY,
        "team_minibatch_solver_calls_per_opportunity": TEAM_MINIBATCH_CALLS_PER_OPPORTUNITY,
        "mandatory_full_solver_calls_per_opportunity": DIAGNOSTIC_FULL_CALLS_PER_OPPORTUNITY,
        "ordinary_full_solver_calls_per_opportunity": ORDINARY_FULL_CALLS_PER_OPPORTUNITY,
        "shadow_solver_calls_per_opportunity": SHADOW_CALLS_PER_OPPORTUNITY,
        "local_solver_calls_per_opportunity": MAX_LOCAL_SOLVER_CALLS_PER_OPPORTUNITY,
        "reflection_calls_per_opportunity": MAX_REFLECTION_CALLS_PER_OPPORTUNITY,
        "total_logical_provider_calls": MAX_LOGICAL_PROVIDER_CALLS,
        "max_attempts_per_logical_call": MAX_ATTEMPTS_PER_LOGICAL_CALL,
        "max_transport_attempts": MAX_LOGICAL_PROVIDER_CALLS * MAX_ATTEMPTS_PER_LOGICAL_CALL,
    }
