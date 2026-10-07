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


def pilot_provider_bounds(parent):
    raise SearchContractError(PILOT_BOUND_BLOCKER)


def derive_current_pilot_contract(parent, *, attempt, binding_path, parent_path,
        parent_sha256, authorization_path, authorization_sha256):
    raise SearchContractError(PILOT_BOUND_BLOCKER)
