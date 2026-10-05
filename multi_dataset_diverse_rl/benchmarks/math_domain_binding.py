"""Current execution eligibility, separate from historical schema compatibility."""
from ..current_contract import MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION
from ..search.current_policy import CURRENT_POLICY_BUNDLE
from ..search.schemas import SearchContractError
from .math_gradient_pattern_binding import MATHGradientPatternBinding


def execution_binding(root,contract):
    if contract.get('identity')!=MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION:
        raise SearchContractError('CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN')
    CURRENT_POLICY_BUNDLE.validate_contract(contract)
    return MATHGradientPatternBinding(root,contract)
