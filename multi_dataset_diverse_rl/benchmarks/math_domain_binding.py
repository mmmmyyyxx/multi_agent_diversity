"""V2.3 eligibility is checked before any data, cache or provider construction."""
from ..current_contract import MATH_OPTIMIZATION_EVIDENCE_BINDING_VERSION
from ..search.schemas import SearchContractError

BINDING_BLOCKER='CURRENT_V23_EXECUTION_BINDING_NOT_FROZEN'

def execution_binding(root,contract):
    if contract.get('identity')!=MATH_OPTIMIZATION_EVIDENCE_BINDING_VERSION:
        raise SearchContractError(BINDING_BLOCKER)
    from .math_evidence_binding import MATHEvidenceBinding
    return MATHEvidenceBinding(root,contract)
