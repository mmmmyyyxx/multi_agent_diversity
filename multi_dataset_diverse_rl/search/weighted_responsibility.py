"""Count-only, exhaustive responsibility for every incorrectly scored member."""
from .. import current_contract as identities
from .schemas import SearchContractError

LABELS = ('direct_flip', 'near_margin', 'coverage', 'r3', 'r4')
WEIGHTS = dict(zip(LABELS, (4.0, 2.0, 1.0, 1.0, 0.5), strict=True))
COUNT_LABELS = {0: 'coverage', 1: 'near_margin', 2: 'direct_flip', 3: 'r3', 4: 'r4'}
POLICY = dict(identity=identities.RAW_RESPONSIBILITY_VALUE_VERSION,
    classification=identities.BINARY_PLURALITY_RESPONSIBILITY_VERSION,
    weights=WEIGHTS, mutually_exclusive=True, count_each_error_once=True,
    label_correct_counts={label: count for count, label in COUNT_LABELS.items()},
    labels='D_N_C_R3_R4_serialized_aliases_not_vote_counterfactual_claims',
    input='five_boolean_member_correctness_results_only',
    assignment='every_wrong_member_no_counterfactual_filter',
    invalid='incorrect_under_unchanged_benchmark_scoring',
    uncovered='impossible_for_valid_five_member_correctness_observations',
    who='sum_divided_by_one_plus_failure_count', pattern='undiscounted_support_sum')


def responsibility_value(direct_count, near_margin_count, coverage_count,
                         r3_count=0, r4_count=0):
    counts = (direct_count, near_margin_count, coverage_count, r3_count, r4_count)
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError('responsibility counts must be nonnegative integers')
    return sum(n * WEIGHTS[label] for n, label in zip(counts, LABELS, strict=True))


def classify_error(team_state, member):
    """Classify using correctness only; answer partitions and Vote are irrelevant."""
    correctness = tuple(team_state.team_correctness)
    if (len(correctness) != 5 or any(type(value) is not bool for value in correctness)
            or type(member) is not int or not 0 <= member < 5):
        raise SearchContractError('RESPONSIBILITY_FIVE_BOOLEAN_RESULTS_REQUIRED')
    if correctness[member]:
        return None, None
    return COUNT_LABELS[sum(correctness)], None
