"""Current exclusive per-error responsibility; historical max primitive is separate."""
from .. import current_contract as identities
from .schemas import SearchContractError
from ..vote_aligned_scheduler import classify_opportunity_lane

LABELS = ('direct_flip', 'near_margin', 'coverage', 'r3', 'r4')
WEIGHTS = dict(zip(LABELS, (4.0, 2.0, 1.0, 1.0, 0.5), strict=True))
POLICY = dict(identity=identities.RAW_RESPONSIBILITY_VALUE_VERSION,
    classification=identities.BINARY_PLURALITY_RESPONSIBILITY_VERSION,
    weights=WEIGHTS, mutually_exclusive=True, count_each_error_once=True,
    dnc='existing_lexicographic_counterfactual_assignment_and_exclusive_lane',
    r3='team_vote_correct_target_wrong_exactly_three_of_five_correct',
    r4='team_vote_correct_target_wrong_exactly_four_of_five_correct',
    uncovered='zero_weight_reported_eligible_for_existing_seeded_fallback',
    who='sum_divided_by_one_plus_failure_count', pattern='undiscounted_support_sum')


def responsibility_value(direct_count, near_margin_count, coverage_count,
                         r3_count=0, r4_count=0):
    counts = (direct_count, near_margin_count, coverage_count, r3_count, r4_count)
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError('responsibility counts must be nonnegative integers')
    return sum(n * WEIGHTS[label] for n, label in zip(counts, LABELS, strict=True))


def classify_error(team_state, member, opportunity, *, assigned, margins):
    """Preserve assigned D/N/C algebra; extend only explicitly defined Vote-correct errors."""
    if team_state.team_correctness[member]:
        return None, None
    if team_state.vote_correct:
        count = sum(team_state.team_correctness)
        if count == 3:
            return 'r3', None
        if count == 4:
            return 'r4', None
        return None, 'TEAM_CORRECT_OTHER_CORRECT_COUNT'
    if not assigned:
        return None, 'TEAM_WRONG_NOT_COUNTERFACTUALLY_ASSIGNED'
    if opportunity is None or not opportunity.member_error:
        raise SearchContractError('RESPONSIBILITY_OPPORTUNITY_REQUIRED')
    lane = classify_opportunity_lane(opportunity, margins)
    if lane == 'pure_coverage':
        lane = 'coverage'
    return (lane, None) if lane in LABELS[:3] else (None, 'ASSIGNED_TEAM_WRONG_NO_DNC_LANE')
