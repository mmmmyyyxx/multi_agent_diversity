# Frozen base V2.1 behavior; explicit historical replay only.
"""Current execution eligibility, separate from historical schema compatibility."""
from ...legacy_current_contract_v21 import MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION
from ...search.legacy.current_policy_v21 import CURRENT_POLICY_BUNDLE
from ...search.schemas import SearchContractError
from .math_gradient_pattern_binding_v21 import MATHGradientPatternBinding


def execution_binding(root,contract):
    if contract.get('identity')!=MATH_PATTERN_AWARE_EXECUTION_BINDING_VERSION:
        raise SearchContractError('CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN')
    CURRENT_POLICY_BUNDLE.validate_contract(contract)
    return MATHGradientPatternBinding(root,contract)
