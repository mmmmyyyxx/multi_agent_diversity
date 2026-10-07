"""V2.2 eligibility precedes historical executors, data or provider construction."""
from ..current_contract import MATH_V2_2_EXECUTION_BINDING_VERSION
from ..search.schemas import SearchContractError
from .math_gradient_pattern_binding import MATHGradientPatternBinding

BINDING_BLOCKER = 'CURRENT_V2_2_EXECUTION_BINDING_NOT_FROZEN'


def execution_binding(root, contract):
    if contract.get('identity') != MATH_V2_2_EXECUTION_BINDING_VERSION:
        raise SearchContractError(BINDING_BLOCKER)
    return MATHGradientPatternBinding(root, contract)
