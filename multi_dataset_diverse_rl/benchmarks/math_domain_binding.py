"""V2.2 eligibility precedes historical executors, data or provider construction."""
from ..current_contract import MATH_V2_2_EXECUTION_BINDING_VERSION, MATH_VISIBLE_TRAJECTORY_BINDING_VERSION
from ..search.schemas import SearchContractError
from .math_gradient_pattern_binding import MATHGradientPatternBinding

BINDING_BLOCKER = 'CURRENT_V2_2_EXECUTION_BINDING_NOT_FROZEN'


def execution_binding(root, contract):
    from ..current_contract import MATH_OPTIMIZATION_EVIDENCE_BINDING_VERSION
    if contract.get('identity')==MATH_OPTIMIZATION_EVIDENCE_BINDING_VERSION:
        from .math_evidence_binding import MATHEvidenceBinding
        return MATHEvidenceBinding(root,contract)
    if contract.get('identity') == MATH_VISIBLE_TRAJECTORY_BINDING_VERSION:
        from .math_visible_binding import MATHVisibleTrajectoryBinding
        return MATHVisibleTrajectoryBinding(root, contract)
    if contract.get('identity') != MATH_V2_2_EXECUTION_BINDING_VERSION:
        raise SearchContractError(BINDING_BLOCKER)
    return MATHGradientPatternBinding(root, contract)
