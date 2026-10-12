"""Bounded single-step, measured experience; optimizer-only and descriptive."""
from copy import deepcopy
import json
from ..current_contract import CURRENT_EDIT_MEMORY_VERSION
from .parent_population import digest
from .schemas import SearchContractError

LIMITS=dict(entries_per_member=24,delivered_entries=3,max_context_chars=1200,
    maximum_edit_chars=600)
POLICY=dict(identity=CURRENT_EDIT_MEMORY_VERSION,retrieval='exact_parent_then_latest_measured_edit',
    delivery='whole_cards_fit_budget_no_numeric_slices',interpretation='scoped_observation_not_causal',
    source='single_step_complete_optimize_and_aggregate_admission')

class PopulationEditMemory:
    identity=CURRENT_EDIT_MEMORY_VERSION
    def __init__(self):
        self.limits=deepcopy(LIMITS);self.competence={};self.entries={m:[] for m in range(5)}
        self.revision=0;self.delivered=0
    def bootstrap(self,store):
        state=store.snapshot()
        self.competence=dict(initial_state_id=store.initial_state_id,
            initial_member_scores=list(store.initial_member_scores),
            support_sha256=state.diagnostics['evaluation_support_identity'])
    def read_for_member(self, member, parent_hash):
        rows=sorted(self.entries[member],key=lambda r:(r['edit_parent_sha256']!=parent_hash,-r['sequence']))
        cards=[]
        for row in rows:
            card={k:row.get(k) for k in ('edited_block','edit_instruction','instruction_omitted',
                'intended_change','repair_hypothesis','actual_edit_parent_effect','full_vote','member_accuracy',
                'scope','count','scoring_team_state','full_vote_delta_to_deployed_baseline','admission','committed','shadow_passed')}
            card.update(source_experience_id=digest(row),selected_parent_matches_edit_parent=row['edit_parent_sha256']==parent_hash)
            if len(json.dumps([card],ensure_ascii=True,separators=(',',':')))>self.limits['max_context_chars']:
                card['repair_hypothesis']=None;card['hypothesis_omitted']=True
            if len(json.dumps([card],ensure_ascii=True,separators=(',',':')))>self.limits['max_context_chars']:
                card['edit_instruction']=None;card['instruction_omitted']=True
            if len(json.dumps(cards+[card],ensure_ascii=True,separators=(',',':')))>self.limits['max_context_chars']:continue
            cards.append(card)
            if len(cards)==self.limits['delivered_entries']:break
        self.delivered+=len(cards)
        return dict(identity=self.identity,entries=deepcopy(cards),interpretation=POLICY['interpretation'])
    def read_for_opportunity(self,opportunity):
        return dict(identity=self.identity,delivery='selected_parent_at_single_step')
    def prepare_outcome(self,outcome):
        records=[]
        for row in outcome.evaluated:
            if row.full is None:continue
            d=row.candidate.backend_details; parent=d['selected_parent'];child=row.candidate.prompt.to_dict()
            changed=[k for k in child if child[k]!=parent[k]]
            if len(changed)!=1:raise SearchContractError('MEMORY_SINGLE_STEP_EDIT_REQUIRED')
            k=changed[0]; content=child[k]
            # Do not truncate an instruction into a misleading partial rule.
            # Long edits retain an explicit omission plus intent/source hashes.
            records.append(dict(sequence=self.revision+1+len(records),member=outcome.opportunity.target_member,
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
        return dict(revision=self.revision,member=outcome.opportunity.target_member,records=records)
    def validate_delta(self,delta):
        if delta['revision']!=self.revision or delta['member'] not in range(5):
            raise SearchContractError('MEMORY_TRANSACTION_REVISION_MISMATCH')
    def apply_outcome(self,delta):
        self.validate_delta(delta);m=delta['member']
        self.entries[m]=(self.entries[m]+deepcopy(delta['records']))[-24:]
        self.revision+=len(delta['records'])
    def audit(self):
        return dict(identity=self.identity,revision=self.revision,stored=sum(map(len,self.entries.values())),delivered=self.delivered)
    def export(self):
        payload=dict(identity=self.identity,revision=self.revision,competence=self.competence,
            entries={str(k):v for k,v in self.entries.items()},delivered=self.delivered)
        return {**deepcopy(payload),'integrity_sha256':digest(payload)}
    def restore(self,value):
        body={k:v for k,v in value.items() if k!='integrity_sha256'}
        if value.get('integrity_sha256')!=digest(body) or body.get('identity')!=self.identity:
            raise SearchContractError('MEMORY_CHECKPOINT_INVALID')
        if set(body['entries'])!=set(map(str,range(5))) or any(len(v)>24 for v in body['entries'].values()):
            raise SearchContractError('MEMORY_CHECKPOINT_INVALID')
        if (type(body['revision']) is not int or body['revision']<0 or type(body['delivered']) is not int or body['delivered']<0
                or any(row.get('member')!=int(m) or row.get('scope')!='complete_optimize'
                    or type(row.get('count')) is not int or row['count']<=0
                    or type(row.get('sequence')) is not int or not 1<=row['sequence']<=body['revision']
                    for m,rows in body['entries'].items() for row in rows)):
            raise SearchContractError('MEMORY_CHECKPOINT_INVALID')
        self.competence=deepcopy(body['competence']);self.revision=body['revision']
        self.entries={int(k):deepcopy(v) for k,v in body['entries'].items()};self.delivered=body['delivered']
