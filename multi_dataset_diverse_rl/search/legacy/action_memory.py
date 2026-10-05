"""Historical Action Memory V3; current uses only its extracted private primitives."""
from ... import versions
from ..private_action_memory import *
from ..memory_records import MemoryDelta, RISK_CODES
from ..schemas import SearchContractError

class StructuredActionMemoryV3(PrivateActionMemory):
    identity=versions.STRUCTURED_ACTION_MEMORY_VERSION

    def prepare_outcome(self, outcome):
        self._search_only(outcome.opportunity)
        if not outcome.complete or outcome.operational_failure:
            return MemoryDelta(self.revision,self.private,self.shared)
        private=list(self.private);shared=list(self.shared)
        op=outcome.opportunity;member=op.target_member
        lane=op.diagnosis.responsibility[member].primary_lane
        for row in outcome.evaluated:
            cid=row.candidate.candidate_id
            if outcome.committed and cid==outcome.selected_candidate_id:
                action=edit_action(op.parent_prompt,row.candidate.prompt)
                if action:
                    d=row.diagnostics
                    private.append(self._entry(member,lane,'SUCCESS',action,
                        f"Committed; team fixed {int(d.get('team_newly_fixed_count',0))}, broken {int(d.get('team_newly_broken_count',0))}.",
                        'Reuse cautiously against current evidence and preserve fixed-peer competence.',
                        op.opportunity_id,row.candidate.prompt))
                continue
            risk=row.diagnostics.get('scientific_risk_code')
            if cid==outcome.selected_candidate_id and outcome.gate_passed is False:risk='SHADOW_REJECTION'
            if risk is not None:
                if risk not in RISK_CODES:raise SearchContractError('NONSTRUCTURAL_MEMORY_RISK')
                # Closed aggregate-only structural categories, never member strategy or Shadow examples.
                lessons=dict(TEAM_PROBE_REJECTION='Check fixed-peer team collateral loss before deployment.',
                    COMMON_SAFE_REJECTION='Require immutable initial competence and strict fixed-peer team gain.',
                    SHADOW_REJECTION='Local improvement does not establish adaptive safety; preserve team and member competence.')
                shared.append(self._entry(None,lane,'RISK','Evaluated a proposed repair against fixed-peer safeguards.',
                    risk.replace('_',' ').lower(),lessons[risk],op.opportunity_id,row.candidate.prompt))
        kept=[]
        for member_id in range(5):
            kept.extend(sorted((e for e in private if e.owner_member==member_id),key=lambda e:e.created_update)[-self.limits['private_storage_limit']:])
        return MemoryDelta(self.revision,tuple(kept),tuple(sorted(shared,key=lambda e:e.created_update)[-self.limits['shared_storage_limit']:]))
