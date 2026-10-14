"""Authoritative experience recording, retrieval and delivery components."""
from copy import deepcopy
import json
from .parent_population import digest
from .schemas import SearchContractError
from .population_memory import POLICY

class ParentScopedMemoryRetrieval:
    identity='EXACT_PARENT_THEN_LATEST_MEMORY_RETRIEVAL_V1'
    def rank(self,entries,parent_hash):
        return sorted(entries,key=lambda r:(r['edit_parent_sha256']!=parent_hash,-r['sequence']))

class WholeMeasuredCardDelivery:
    identity='WHOLE_MEASURED_CARDS_BUDGET_DELIVERY_V1'
    def deliver(self,ranked,parent_hash,limits,identity):
        rows=ranked
        cards=[]
        for row in rows:
            card={k:row.get(k) for k in ('edited_block','edit_instruction','instruction_omitted',
                'intended_change','repair_hypothesis','actual_edit_parent_effect','full_vote','member_accuracy',
                'scope','count','scoring_team_state','full_vote_delta_to_deployed_baseline','admission','committed','shadow_passed')}
            card.update(source_experience_id=digest(row),selected_parent_matches_edit_parent=row['edit_parent_sha256']==parent_hash)
            if len(json.dumps([card],ensure_ascii=True,separators=(',',':')))>limits['max_context_chars']:
                card['repair_hypothesis']=None;card['hypothesis_omitted']=True
            if len(json.dumps([card],ensure_ascii=True,separators=(',',':')))>limits['max_context_chars']:
                card['edit_instruction']=None;card['instruction_omitted']=True
            if len(json.dumps(cards+[card],ensure_ascii=True,separators=(',',':')))>limits['max_context_chars']:continue
            cards.append(card)
            if len(cards)==limits['delivered_entries']:break
        return dict(identity=identity,entries=deepcopy(cards),interpretation=POLICY['interpretation'])

class DisabledMemoryDelivery:
    identity='EXPLICITLY_DISABLED_MEMORY_DELIVERY_V1'
    def deliver(self,ranked,parent_hash,limits,identity):
        return dict(identity=identity,entries=[],interpretation=POLICY['interpretation'],
            status='MODULE_DISABLED',module_identity=self.identity)

class MeasuredEditExperienceRecorder:
    identity='SINGLE_STEP_COMPLETE_MEASUREMENT_RECORDER_V1'
    def records(self,outcome,revision):
        records=[]
        for row in outcome.evaluated:
            if row.full is None:continue
            d=row.candidate.backend_details; parent=d['selected_parent'];child=row.candidate.prompt.to_dict()
            changed=[k for k in child if child[k]!=parent[k]]
            if len(changed)!=1:raise SearchContractError('MEMORY_SINGLE_STEP_EDIT_REQUIRED')
            k=changed[0]; content=child[k]
            # Do not truncate an instruction into a misleading partial rule.
            # Long edits retain an explicit omission plus intent/source hashes.
            records.append(dict(sequence=revision+1+len(records),member=outcome.opportunity.target_member,
                edit_parent_sha256=d['selected_parent_sha256'],child_sha256=row.candidate.prompt.prompt_hash,
                edited_block=k,edit_instruction=content if len(content)<=600 else None,
                instruction_omitted=len(content)>600,intended_change=d['intended_change'],
                full_vote=row.full.aggregate_score,member_accuracy=row.full.member_scores[outcome.opportunity.target_member],
                actual_edit_parent_effect=d.get('selected_parent_member_effect'),
                repair_hypothesis=d.get('repair_hypothesis'),
                scope='complete_optimize',count=row.full.aggregation_diagnostics['evaluation_count'],
                scoring_team_state=row.full.aggregation_diagnostics['fitness_provenance']['actual_vote']['team_state_id'],
                deployment_baseline_state=outcome.opportunity.parent_state_id,
                full_vote_delta_to_deployed_baseline=row.full.aggregation_diagnostics['team_score_delta'],
                admission=row.diagnostics.get('promotion_reason','FULL_MEASURED'),
                selected=row.candidate.candidate_id==outcome.selected_candidate_id,
                committed=outcome.committed and row.candidate.candidate_id==outcome.selected_candidate_id,
                shadow_passed=outcome.gate_passed if row.candidate.candidate_id==outcome.selected_candidate_id else None))
        return records
