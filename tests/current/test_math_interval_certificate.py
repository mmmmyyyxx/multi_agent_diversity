"""Exact enclosure boundaries, with independent rational and algebraic witnesses."""
from fractions import Fraction
import pytest
from sympy import Integer,Rational,Add,Mul,Pow,sqrt,Symbol,Float,I,Matrix,Tuple,FiniteSet
from multi_dataset_diverse_rl.benchmarks.math_interval_certificate import enclosure,certified_far_unequal,POLICY
from multi_dataset_diverse_rl.benchmarks.math_certified_domain import certified_matrix


@pytest.mark.parametrize('value',[Fraction(-7,13),Fraction(1,3),Fraction(0),Fraction(9),Fraction(10**30+1)])
def test_rational_values_are_enclosed(value):
    lo,hi=enclosure(Rational(value.numerator,value.denominator))
    assert lo<=value<=hi and hi-lo<=Fraction(1,1<<80)


@pytest.mark.parametrize('value',[Fraction(2),Fraction(1,7),Fraction(0),Fraction(9),Fraction(10**12+1)])
def test_square_root_bounds_are_proven_by_exact_squaring(value):
    lo,hi=enclosure(Pow(Rational(value.numerator,value.denominator),Rational(1,2),evaluate=False))
    assert 0<=lo<=hi and lo*lo<=value<=hi*hi


@pytest.mark.parametrize('base,power,result',[(2,-3,Fraction(1,8)),(-3,-3,Fraction(-1,27)),(-2,4,Fraction(16)),(-2,3,Fraction(-8)),(0,5,Fraction(0))])
def test_integer_power_signs_and_reciprocals(base,power,result):
    lo,hi=enclosure(Pow(Integer(base),Integer(power),evaluate=False));assert lo<=result<=hi


@pytest.mark.parametrize('left,right',[
    (Pow(8,Rational(1,2),evaluate=False),2*sqrt(2)),
    (Mul(sqrt(2)-1,sqrt(2)+1,evaluate=False),Integer(1)),
    (Add(10**18+1,-10**18,evaluate=False),Integer(1)),
    (Pow(sqrt(2),-2,evaluate=False),Rational(1,2)),
    (Rational(1),Rational(100000000000000000001,100000000000000000000)),
    (Integer(1),Integer(2)),
])
def test_equal_near_and_cancellation_cases_cannot_be_rejected(left,right):
    assert not certified_far_unequal(left,right)
    assert not certified_far_unequal(right,left)


@pytest.mark.parametrize('value',[Float('1.2'),Symbol('x'),I,Pow(-1,Rational(1,2),evaluate=False),Pow(2,10001,evaluate=False),Integer(1)<<32768,Matrix([[1,2]]),Tuple(1,2),FiniteSet(1,2)])
def test_unsupported_and_resource_exhaustion_defer_without_a_wrong_score(value):
    assert enclosure(value) is None
    assert not certified_far_unequal(value,Integer(100))


def test_nested_radical_high_power_can_certify_far_inequality_without_expansion():
    expression=Pow(Pow(1+sqrt(1+sqrt(2)),Rational(1,2),evaluate=False),Integer(8000),evaluate=False)
    assert certified_far_unequal(expression,Integer(10))
    assert certified_far_unequal(Integer(10),expression)


@pytest.mark.parametrize('expressions,expected',[
    ((r'\sqrt{8}',r'2\sqrt{2}'),True),
    (('1','1.00000001'),False),
    ((r'\frac{1}{3}','0.333333'),True),
    (('1','100'),False),
    (('x+1','x+2'),False),
    ((r'\begin{pmatrix}1&2\end{pmatrix}',r'\begin{pmatrix}1&2\end{pmatrix}'),True),
    ((r'\begin{pmatrix}1&2\end{pmatrix}',r'\begin{pmatrix}2&1\end{pmatrix}'),False),
])
def test_scoped_worker_preserves_native_equality_and_unsupported_rules(expressions,expected):
    from multi_dataset_diverse_rl.benchmarks.math_domain_v2 import domain_matrix
    result=certified_matrix(expressions)
    assert result['equivalence'][0][1] is expected
    assert result['equivalence']==domain_matrix(expressions)['equivalence']
    if expected:assert [0,1] not in result['certified_pair_rejections']


def test_worker_integrity_failure_is_not_a_negative_certificate(monkeypatch):
    from types import SimpleNamespace
    from multi_dataset_diverse_rl.benchmarks import math_certified_domain as client
    from multi_dataset_diverse_rl.search.schemas import SearchContractError
    client.certified_matrix.cache_clear()
    monkeypatch.setattr(client.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout='{}'))
    with pytest.raises(SearchContractError,match='MATH_EVALUATOR_FAILURE'):
        client.certified_matrix(('fixture-left','fixture-right'))
