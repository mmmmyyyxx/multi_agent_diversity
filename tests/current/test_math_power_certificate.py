"""Synthetic exact witnesses; no historical answers or approximate equality."""
from fractions import Fraction
import pytest
from sympy import Integer,Rational,Add,Pow,Symbol,I
from multi_dataset_diverse_rl.benchmarks.math_power_certificate import power_lower_witness,certified_far_unequal
from multi_dataset_diverse_rl.benchmarks.math_power_domain import power_matrix
from multi_dataset_diverse_rl.benchmarks.math_interval_certificate import enclosure


@pytest.mark.parametrize('base,exponent,other',[(2,17,1),(3,13,-100),(5,11,20),(Rational(3,2),30,2)])
def test_monotone_witness_has_independent_exact_gap(base,exponent,other):
    x=Pow(base,Integer(exponent),evaluate=False);actual=Fraction(base)**exponent;y=Fraction(other)
    assert power_lower_witness(x,Integer(other))
    assert abs(actual-y)>1 and 100*abs(actual-y)>max(abs(actual),abs(y))


def test_unexpanded_huge_integer_exponent_never_needs_expansion():
    exponent=Add(Pow(Integer(3),Integer(90),evaluate=False),Integer(-2),evaluate=False)
    expression=Pow(Integer(5),exponent,evaluate=False)
    assert enclosure(expression) is None
    assert power_lower_witness(expression,Integer(123))
    assert certified_far_unequal(expression,Integer(123)) and certified_far_unequal(Integer(123),expression)


@pytest.mark.parametrize('base,exponent,other',[(2,3,8),(2,3,7),(2,3,9),(1,1000,1),(Rational(1,2),8,1),
    (-2,3,0),(2,-3,0),(2,Rational(1,3),1),(2,Symbol('x'),10),(I,20,2)])
def test_equal_near_sign_complex_and_unknown_cases_defer(base,exponent,other):
    assert not power_lower_witness(Pow(base,exponent,evaluate=False),Integer(other))


def test_cancellation_inside_exponent_cannot_create_a_false_lower_bound():
    huge=Pow(Integer(3),Integer(90),evaluate=False)
    exponent=Add(huge,-huge,Integer(1),evaluate=False)
    assert not power_lower_witness(Pow(Integer(2),exponent,evaluate=False),Integer(1))


@pytest.mark.parametrize('expressions',[(r'\sqrt{8}',r'2\sqrt{2}'),('1','1.00000001'),
    (r'\frac{1}{3}','0.333333'),('x+1','x+2'),('1','100'),
    (r'\begin{pmatrix}1&2\end{pmatrix}',r'\begin{pmatrix}2&1\end{pmatrix}')])
def test_scoped_worker_preserves_existing_supported_and_fallback_scores(expressions):
    from multi_dataset_diverse_rl.benchmarks.math_certified_domain import certified_matrix
    assert power_matrix(expressions)['equivalence']==certified_matrix(expressions)['equivalence']


def test_worker_handles_synthetic_unexpanded_exponent():
    value=power_matrix(('123',r'5^{3^{90}-2}'))
    assert value['valid']==[True,True] and value['equivalence'][0][1] is False
    assert [0,1] in value['certified_pair_rejections']


def test_timeout_carries_trace_and_never_becomes_wrong(monkeypatch):
    import subprocess
    from multi_dataset_diverse_rl.benchmarks import math_power_domain as client
    from multi_dataset_diverse_rl.search.schemas import SearchContractError
    client.power_matrix.cache_clear()
    def timeout(*a,**k):raise subprocess.TimeoutExpired('worker',30,stderr=b'private synthetic stack')
    monkeypatch.setattr(client.subprocess,'run',timeout)
    with pytest.raises(SearchContractError,match='MATH_EVALUATOR_TIMEOUT') as result:client.power_matrix(('fixture','another'))
    assert result.value.private_worker_trace=='private synthetic stack'
