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
    def __init__(self,*,recorder=None,retrieval=None,delivery=None):
        from .memory_components import MeasuredEditExperienceRecorder,ParentScopedMemoryRetrieval,WholeMeasuredCardDelivery
        self.recorder=recorder or MeasuredEditExperienceRecorder()
        self.retrieval=retrieval or ParentScopedMemoryRetrieval()
        self.delivery=delivery or WholeMeasuredCardDelivery()
        self.limits=deepcopy(LIMITS);self.competence={};self.entries={m:[] for m in range(5)}
        self.revision=0;self.delivered=0
    def bootstrap(self,store):
        state=store.snapshot()
        self.competence=dict(initial_state_id=store.initial_state_id,
            initial_member_scores=list(store.initial_member_scores),
            support_sha256=state.diagnostics['evaluation_support_identity'])
    def read_for_member(self, member, parent_hash):
        ranked=self.retrieval.rank(self.entries[member],parent_hash)
        view=self.delivery.deliver(ranked,parent_hash,self.limits,self.identity)
        self.delivered+=len(view['entries'])
        return view
    def read_for_opportunity(self,opportunity):
        return dict(identity=self.identity,delivery='selected_parent_at_single_step')
    def prepare_outcome(self,outcome):
        records=self.recorder.records(outcome,self.revision)
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
    def detached(self):
        return PopulationEditMemory(recorder=self.recorder,retrieval=self.retrieval,delivery=self.delivery)
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
