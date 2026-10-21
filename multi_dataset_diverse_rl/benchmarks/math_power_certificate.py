"""Retired scoring certificates: metadata only, no executable mathematics.

Original implementation is sealed in docs/archive/scoring/pre_hendrycks_20261021.
POLICY remains solely for immutable historical-manifest inspection.
"""
from copy import deepcopy
from .math_interval_certificate import POLICY as INTERVAL_POLICY
from ..math_scoring import RetiredMathScorerError

POLICY=dict(identity='MATH_EXACT_MONOTONE_POWER_REJECTION_V2',interval=deepcopy(INTERVAL_POLICY),
    witness='positive_base_above_one_and_integer_lower_exponent',
    exponent_cap=INTERVAL_POLICY['maximum_integer_power'],
    gap='lower_minus_other_absolute_upper_strictly_above_one',
    relative_gap='99_times_lower_strictly_above_100_times_other_absolute_upper',
    accepted_equality=False,fallback='unchanged_pinned_verifier')


def _retired(*args, **kwargs):
    raise RetiredMathScorerError("HISTORICAL_MATH_CERTIFICATE_RETIRED")


power_lower_witness = _retired
certified_far_unequal = _retired
