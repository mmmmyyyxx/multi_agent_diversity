"""One versioned raw evidence value for member and pattern responsibility."""
from .. import versions

IDENTITY = versions.RAW_RESPONSIBILITY_VALUE_VERSION


def responsibility_value(direct_count, near_margin_count, coverage_count):
    counts = (direct_count, near_margin_count, coverage_count)
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError('responsibility counts must be nonnegative integers')
    return max(4 * direct_count, 2 * near_margin_count, coverage_count)
