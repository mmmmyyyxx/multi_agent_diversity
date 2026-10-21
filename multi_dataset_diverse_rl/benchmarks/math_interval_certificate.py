"""Retired scoring certificates: metadata only, no executable mathematics.

Original implementation is sealed in docs/archive/scoring/pre_hendrycks_20261021.
POLICY remains solely for immutable historical-manifest inspection.
"""
from ..math_scoring import RetiredMathScorerError

POLICY=dict(identity='MATH_EXACT_FAR_INTERVAL_REJECTION_V1',precision_bits=80,
    maximum_nodes=256,maximum_depth=32,maximum_integer_bits=32768,
    maximum_integer_power=10000,minimum_absolute_gap=1,
    relative_gap_denominator=100,supported='Integer_Rational_Add_Mul_integer_powers_nonnegative_square_roots',
    success='certified_far_inequality_only',overlap_or_unsupported='unchanged_pinned_verifier',
    floats_symbols_complex_containers='unchanged_pinned_verifier')


def _retired(*args, **kwargs):
    raise RetiredMathScorerError("HISTORICAL_MATH_CERTIFICATE_RETIRED")


enclosure = _retired
certified_far_unequal = _retired
