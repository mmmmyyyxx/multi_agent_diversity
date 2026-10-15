"""Bounded exact enclosures certify far inequality; never approximate equality."""
from fractions import Fraction
from math import isqrt

POLICY=dict(identity='MATH_EXACT_FAR_INTERVAL_REJECTION_V1',precision_bits=80,
    maximum_nodes=256,maximum_depth=32,maximum_integer_bits=32768,
    maximum_integer_power=10000,minimum_absolute_gap=1,
    relative_gap_denominator=100,supported='Integer_Rational_Add_Mul_integer_powers_nonnegative_square_roots',
    success='certified_far_inequality_only',overlap_or_unsupported='unchanged_pinned_verifier',
    floats_symbols_complex_containers='unchanged_pinned_verifier')


class UnsupportedInterval(Exception):pass


def enclosure(expression,diagnostics=None):
    from sympy import Integer,Rational,Add,Mul,Pow,Expr
    scale=1<<POLICY['precision_bits'];nodes=0
    def rounded(lo,hi):
        lo=Fraction((lo.numerator*scale)//lo.denominator,scale)
        hi=Fraction(-((-hi.numerator*scale)//hi.denominator),scale)
        if any(abs(x.numerator).bit_length()>POLICY['maximum_integer_bits'] for x in (lo,hi)):
            raise UnsupportedInterval('rounded_integer_bits')
        return lo,hi
    def mul(a,b):
        values=[x*y for x in a for y in b]
        return rounded(min(values),max(values))
    def power(a,n):
        if abs(n)>POLICY['maximum_integer_power']:raise UnsupportedInterval('integer_power_limit')
        if n<0:
            if a[0]<=0<=a[1]:raise UnsupportedInterval('zero_reciprocal')
            a=rounded(1/a[1],1/a[0]);n=-n
        result=(Fraction(1),Fraction(1))
        while n:
            if n&1:result=mul(result,a)
            n//=2
            if n:a=mul(a,a)
        return result
    def root(a):
        if a[0]<0:raise UnsupportedInterval('negative_root')
        def endpoint(x,upper):
            numerator=x.numerator*scale*scale
            value=isqrt(numerator//x.denominator)
            if upper and value*value*x.denominator<numerator:value+=1
            return Fraction(value,scale)
        return rounded(endpoint(a[0],False),endpoint(a[1],True))
    def visit(x,depth=0):
        nonlocal nodes
        nodes+=1
        if nodes>POLICY['maximum_nodes'] or depth>POLICY['maximum_depth']:raise UnsupportedInterval('node_depth_limit')
        if not isinstance(x,Expr):raise UnsupportedInterval('unsupported_type')
        if isinstance(x,Rational):
            a,b=int(x.p),int(x.q)
            if max(abs(a).bit_length(),b.bit_length())>POLICY['maximum_integer_bits']:raise UnsupportedInterval('input_bits')
            return rounded(Fraction(a,b),Fraction(a,b))
        if x.func is Add:
            result=(Fraction(0),Fraction(0))
            for arg in x.args:
                lo,hi=visit(arg,depth+1);result=rounded(result[0]+lo,result[1]+hi)
            return result
        if x.func is Mul:
            result=(Fraction(1),Fraction(1))
            for arg in x.args:result=mul(result,visit(arg,depth+1))
            return result
        if x.func is Pow:
            base,exponent=x.args
            if not isinstance(exponent,Rational) or int(exponent.q) not in (1,2):raise UnsupportedInterval('unsupported_power')
            if abs(int(exponent.p))>POLICY['maximum_integer_power']:raise UnsupportedInterval('integer_power_limit')
            value=visit(base,depth+1)
            if int(exponent.q)==2:value=root(value)
            return power(value,int(exponent.p))
        raise UnsupportedInterval('unsupported_type')
    try:return visit(expression)
    except UnsupportedInterval as exc:
        if diagnostics is not None:diagnostics.append(str(exc))
        return None


def certified_far_unequal(left,right):
    a,b=enclosure(left),enclosure(right)
    if a is None or b is None:return False
    gap=max(a[0]-b[1],b[0]-a[1])
    magnitude=max(abs(x) for x in (*a,*b))
    return bool(gap>POLICY['minimum_absolute_gap'] and
        gap*POLICY['relative_gap_denominator']>magnitude)
