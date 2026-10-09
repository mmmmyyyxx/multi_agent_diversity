"""Technical feasibility only; V2.3 constructs all evidence memberships separately."""
from .. import current_contract as versions
from .policies import ResponsibilitySignal
from .schemas import SearchContractError

class VariableEvidenceFeasibilityV1:
    identity=versions.UNIFIED_VARIABLE_FEASIBILITY_VERSION

    def feasible(self,state,diagnosis,member_id,evidence):
        # Equivalent to the old pre-Pattern builder's existence check. No role
        # view is built here, so feasibility cannot create overlapping panels.
        signal=diagnosis.responsibility.get(member_id)
        if not isinstance(signal,ResponsibilitySignal):raise SearchContractError('MEMBER_RESPONSIBILITY_MISSING')
        unique={}
        for row in evidence:
            if row.source_split!='optimize':raise SearchContractError('PATTERN_HELDOUT_ACCESS')
            if not row.example_id or row.example_id in unique:raise SearchContractError('EVIDENCE_ID_DUPLICATE_OR_MISSING')
            if type(row.signals.get('target_member_correct')) is not bool:raise SearchContractError('TARGET_CORRECTNESS_SIGNAL_REQUIRED')
            unique[row.example_id]=row
        if len(unique)<12:raise SearchContractError('DISJOINT_OPTIMIZE_EVIDENCE_INSUFFICIENT')
        return any(not r.signals['target_member_correct'] for r in unique.values())
