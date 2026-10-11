"""Conservative monetary reservations under a separately frozen supplier tariff.

Amounts use integer millionths of the declared currency, never binary floats.
No supplier price or currency conversion is inferred from the model name.
"""
from decimal import Decimal, InvalidOperation
from copy import deepcopy
import re
from .token_accounting import OperationalAbort

IDENTITY='ROLE_ROUTED_SUPPLIER_COST_RESERVATION_V1'
ROLE_PROFILE=dict(solver='openlux',reflection='lwj',pattern_gradient='lwj',pattern_cluster='lwj')
SCALE=1_000_000


def micros(value):
    try:
        if not isinstance(value,str) or re.fullmatch(r'(0|[1-9][0-9]*)(\.[0-9]{1,6})?',value) is None:raise ValueError
        amount=Decimal(value)
        if not amount.is_finite() or amount<0 or amount*SCALE!=int(amount*SCALE):raise ValueError
        return int(amount*SCALE)
    except (InvalidOperation,ValueError,OverflowError):
        raise OperationalAbort('SUPPLIER_COST_POLICY_INVALID') from None


def frozen_policy(value):
    if value is None:return None
    if (not isinstance(value,dict) or set(value)!={'identity','currency','hard_limit',
            'price_basis','pricing_source','tariffs'} or value['identity']!=IDENTITY
            or value['currency'] not in {'CNY','USD'}
            or value['price_basis']!='CONSERVATIVE_NO_CACHE_DISCOUNT_ALL_PHYSICAL_ATTEMPTS'
            or not isinstance(value['pricing_source'],str) or not value['pricing_source'].strip()
            or not isinstance(value['tariffs'],dict) or set(value['tariffs'])!={'openlux','lwj'}
            or micros(value['hard_limit'])<=0):
        raise OperationalAbort('SUPPLIER_COST_POLICY_INVALID')
    for profile,model in (('openlux','gpt-4o-mini'),('lwj','qwen3.7-flash')):
        tariff=value['tariffs'][profile]
        if (not isinstance(tariff,dict) or set(tariff)!={'model','input_per_million',
                'output_per_million','physical_request_fee'} or tariff['model']!=model):
            raise OperationalAbort('SUPPLIER_COST_POLICY_INVALID')
        for name in ('input_per_million','output_per_million','physical_request_fee'):micros(tariff[name])
    return deepcopy(value)


def amount(policy, role, input_tokens, output_tokens):
    if role not in ROLE_PROFILE or any(type(n) is not int or n<0 for n in (input_tokens,output_tokens)):
        raise OperationalAbort('SUPPLIER_COST_ROLE_OR_USAGE_INVALID')
    tariff=policy['tariffs'][ROLE_PROFILE[role]]
    numerator=input_tokens*micros(tariff['input_per_million'])+output_tokens*micros(tariff['output_per_million'])
    return (numerator+SCALE-1)//SCALE+micros(tariff['physical_request_fee'])
