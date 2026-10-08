"""Technical feasibility only; V2.3 constructs all evidence memberships separately."""
from .. import versions
from .policies import ResponsibilitySignal
from .schemas import SearchContractError

class VariableEvidenceFeasibilityV1:
    identity=versions.UNIFIED_VARIABLE_FEASIBILITY_VERSION

    def feasible(self,state,diagnosis,member_id,evidence):
        # Equivalent to the old pre-Pattern builder's existence check. No role
        # view is built here, so feasibility cannot create overlapping panels.
        if not any('REPAIR' in r.roles for r in evidence):return False
        signal=diagnosis.responsibility.get(member_id)
        if not isinstance(signal,ResponsibilitySignal) or signal.raw_value<=0:return False
        unique={}
        for row in evidence:
            if row.source_split!='optimize':return False
            if row.example_id in unique and unique[row.example_id]!=row:return False
            unique[row.example_id]=row
        return len(unique)>=3
