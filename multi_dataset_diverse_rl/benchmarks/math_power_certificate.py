"""Exact one-sided bounds for unevaluated positive powers, never equality."""
from copy import deepcopy
from .math_interval_certificate import POLICY as INTERVAL_POLICY,enclosure,certified_far_unequal as interval_rejection

POLICY=dict(identity='MATH_EXACT_MONOTONE_POWER_REJECTION_V2',interval=deepcopy(INTERVAL_POLICY),
    witness='positive_base_above_one_and_integer_lower_exponent',
    exponent_cap=INTERVAL_POLICY['maximum_integer_power'],
    gap='lower_minus_other_absolute_upper_strictly_above_one',
    relative_gap='99_times_lower_strictly_above_100_times_other_absolute_upper',
    accepted_equality=False,fallback='unchanged_pinned_verifier')


def power_lower_witness(expression,other):
    """If x>=L>M>=abs(y), L-M>1, 99L>100M, far inequality is proven.

    For b>1 and e>=n>=1, b**e>=b**n. Only the bounded integer power
    b**n is enclosed; the original possibly enormous exponent is never expanded.
    Unsupported/negative/complex bases and nonpositive exponents defer.
    """
    from sympy import Pow,Integer
    if getattr(expression,'func',None) is not Pow:return False
    base,exponent=expression.args
    b,e,target=enclosure(base),enclosure(exponent),enclosure(other)
    if b is None or e is None or target is None or b[0]<=1 or e[0]<1:return False
    n=min(e[0].numerator//e[0].denominator,POLICY['exponent_cap'])
    magnitude=max(abs(x) for x in target)
    while n:
        bounded=enclosure(Pow(base,Integer(n),evaluate=False))
        if bounded is not None:
            lower=bounded[0]
            return bool(lower-magnitude>1 and 99*lower>100*magnitude)
        n//=2
    return False


def certified_far_unequal(left,right):
    return bool(interval_rejection(left,right) or power_lower_witness(left,right) or power_lower_witness(right,left))
