"""V2.4 eligibility precedes data, cache and provider construction."""
from ..current_contract import MATH_STRUCTURED_SYSTEM_BINDING_VERSION
from ..search.schemas import SearchContractError

BINDING_BLOCKER='CURRENT_V24_EXECUTION_BINDING_NOT_FROZEN'

def execution_binding(root,contract):
    if contract.get('identity')!=MATH_STRUCTURED_SYSTEM_BINDING_VERSION:
        raise SearchContractError(BINDING_BLOCKER)
    from .math_structured_binding import MATHStructuredBinding
    return MATHStructuredBinding(root,contract)
