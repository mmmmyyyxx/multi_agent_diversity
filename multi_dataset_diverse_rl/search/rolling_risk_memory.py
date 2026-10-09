"""Opt-in cross-member risk promotion; no raw private strategies are shared.

A bounded bank of generic observations supports promotion
at the opportunity transaction boundary. Retention uses recent evidence, never
permanent occurrence totals or future outcomes, and makes no provider calls.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from hashlib import sha256
import json

from .. import current_contract as versions
from .private_action_memory import PrivateActionMemory, ActionExperience, LABELS, LIMITS
from .memory_records import MemoryDelta, RISK_CODES
from .schemas import SearchContractError


POLICY = dict(signature=versions.SEMANTIC_FAILURE_SIGNATURE_VERSION,
    retention=versions.SHARED_RISK_RETENTION_VERSION,minimum_distinct_members=2,
    recent_opportunity_window=LIMITS['shared_storage_limit'],
    failure_evidence_capacity=5*LIMITS['shared_storage_limit'],
    shared_event_capacity=LIMITS['shared_storage_limit'],promotion_timing='opportunity_commit',
    retention_order=['recent_recurrence','recent_distinct_members','same_lane','last_seen','semantic_key'],
    retrieval_order=['same_lane','compatible_family','last_seen','recent_recurrence','semantic_key'])
LANES = frozenset({'direct_flip','near_margin','coverage','general'})
ORIGINS = frozenset({'TEAM_STRUCTURAL_REJECTION','REPEATED_PRIVATE_FAILURE'})


@dataclass(frozen=True, order=True)
class FailureSignature:
    category: str
    action_operations: tuple[tuple[str,str], ...]
    repair_pattern: str

    def key(self):
        return json.dumps([self.category,self.action_operations,self.repair_pattern],separators=(',',':'))


def failure_signature(details, lane):
    """Exact closed semantic concepts, not hashes, raw prose or equal deltas."""
    if lane not in LANES or any(details.get(k) is not True for k in ('changed','contract_valid','solver_evaluated')):
        return None
    effect=details.get('edit_lineage',{}).get('effects',{}).get('actual_parent_validation',{})
    if (effect.get('valid_to_invalid_ids') or effect.get('repeated_format_failure_ids')):
        return FailureSignature('OUTPUT_CONTRACT_FAILURE',(),'observed_invalid_output_no_mathematical_success')
    # Procedure conformance is distinct from Solver response validity. Terminal
    # parser/format failures must not become a reasoning-risk family.
    if (type(details.get('local_invalid_count')) is not int or details['local_invalid_count']!=0
            or details.get('operational_failure') is True or details.get('duplicate') is True):
        return None
    fixed,broken,delta=(details.get(k) for k in ('local_parent_newly_fixed','local_parent_newly_broken','local_parent_correct_delta'))
    if any(type(x) is not int for x in (fixed,broken,delta)) or min(fixed,broken)<0 or delta>=0 or fixed-broken!=delta:
        return None
    action=details.get('memory_action')
    if not isinstance(action,str):return None
    by_label={v:k for k,v in LABELS.items()};ops=[]
    for clause in action.split('; '):
        matched=False
        for prefix,verb in (('Add ','ADD'),('Remove ','REMOVE'),('Revise instructions for ','REVISE')):
            if clause.startswith(prefix):
                labels=clause[len(prefix):].split(', ')
                if not labels or any(v not in by_label for v in labels):return None
                ops.extend((verb,by_label[v]) for v in labels);matched=True;break
        if not matched:return None
    if not ops:return None
    measured=details.get('preservation_locally_measurable') is True
    loss=details.get('local_newly_broken')
    if type(loss) is not int or loss<0:return None
    preservation=measured and loss>0
    case=any(verb in {'ADD','REVISE'} and concept=='case_analysis' for verb,concept in ops)
    category=('CASE_ANALYSIS_PRESERVATION_REGRESSION' if case else 'PRESERVATION_REGRESSION') if preservation else 'LOCAL_COMPETENCE_REGRESSION'
    return FailureSignature(category,tuple(sorted(set(ops))),
        'repair_gain_with_collateral_loss' if fixed else 'loss_without_observed_repair_gain')


@dataclass(frozen=True)
class RiskObservation:
    signature: FailureSignature
    member: int
    lane: str
    update: int
    observation_id: str
    origin: str
    shared_consumed: bool = False


@dataclass(frozen=True)
class SharedRiskEntry:
    signature: FailureSignature
    observations: tuple[RiskObservation,...]
    occurrence_count: int
    members: tuple[int,...]
    observed_lanes: tuple[str,...]
    origins: tuple[str,...]
    last_seen_update: int

    @property
    def memory_id(self):return sha256((versions.STRUCTURED_ROLLING_RISK_MEMORY_VERSION+self.signature.key()).encode()).hexdigest()
    @property
    def created_update(self):return self.last_seen_update
    @property
    def origin(self):return 'MIXED' if len(self.origins)>1 else self.origins[0]
    @property
    def distinct_member_count(self):return len(self.members)

    def visible(self):
        # Closed risk categories/principles only; no member IDs, provenance,
        # individual outcomes, Action strings, prompts or provider summaries.
        category=self.signature.category.replace('_',' ').lower()
        concepts=sorted({LABELS[k] for _,k in self.signature.action_operations})
        mechanism=('Edits involving '+', '.join(concepts)+'.') if concepts else 'Fixed-peer safeguard rejection.'
        return dict(situation='Responsibility repair; verify applicability to current evidence.',
            action=mechanism,outcome=category+'.',
            lesson='Observed risk, not causal proof. Check preserved competence and fixed-peer safeguards before reuse.')


@dataclass(frozen=True)
class RollingMemoryDelta(MemoryDelta):
    clock: int
    sequence: int
    failure_events: tuple[RiskObservation,...]
    risk_counts: tuple[tuple[str,int],...]


@dataclass(frozen=True)
class EditExperience:
    owner_member: int
    created_update: int
    record: dict

    @property
    def memory_id(self):
        return sha256((self.record['opportunity_id']+':'+self.record['candidate_id']).encode()).hexdigest()

    @property
    def lane(self):return 'general'

    def visible(self):
        effects={name:dict(fixed=len(e['fixed_ids']),broken=len(e['broken_ids']),
            invalid=len(e['invalid_ids']),delta=e['member_delta'],
            parent_invalid=e['parent_invalid_count'],valid_to_invalid=len(e['valid_to_invalid_ids']),
            invalid_to_valid_wrong=len(e['invalid_to_valid_wrong_ids']),invalid_to_valid_correct=len(e['invalid_to_valid_correct_ids']),
            repeated_format_failure=len(e['repeated_format_failure_ids']),
            **({'team_delta':e['team_delta']} if 'team_delta' in e else {}))
            for name,e in self.record['effects'].items() if name in {'search_validation','full'}}
        return dict(hypothesis=self.record['repair_hypothesis'][:100],status=self.record['status'],
            edited_block=self.record['edited_block'],edit_chain_length=len(self.record['edit_chain']),
            edited_block_chain=[s['edited_blocks'][0] for s in self.record['edit_chain']],
            actual_edit=(self.record['action'] or 'Changed instructions; semantic action unclassified.')[:110],
            diff_operations=sorted({d['operation'] for d in self.record['actual_diff']}),
            effects=effects,lesson='Compound edit; scope-specific, no causal claim.')


@dataclass(frozen=True)
class EvidenceMemoryDelta(RollingMemoryDelta):
    competence: tuple[dict,...]


class StructuredRollingRiskMemoryV4(PrivateActionMemory):
    identity=versions.STRUCTURED_ROLLING_RISK_MEMORY_VERSION

    def __init__(self, *, risk_policy, optimization_evidence_policy=None, **limits):
        if risk_policy!=POLICY:raise SearchContractError('ROLLING_RISK_POLICY_NOT_FROZEN')
        super().__init__(**limits)
        self.risk_policy=json.loads(json.dumps(POLICY))
        from .optimization_evidence import frozen_policy, MEMORY
        self.optimization_evidence_policy=frozen_policy(optimization_evidence_policy)
        self.competence=()
        self.identity=MEMORY
        self.clock=0;self.failure_events=()
        self.observation_observer=None
        self.risk_counts=dict(failure_to_shared_promotions=0,shared_risk_created=0,
            shared_risk_updated=0,shared_risk_evicted=0,shared_risk_storage_peak=0)

    def _recent(self, events, clock):
        window=self.risk_policy['recent_opportunity_window']
        return tuple(e for e in events if 0<=clock-e.update<window)

    def observe_local_failure(self, **kwargs):
        member=kwargs['member'];lane=kwargs['lane']
        if member not in range(5) or lane not in LANES:raise SearchContractError('ROLLING_RISK_SCOPE_INVALID')
        wrote=super().observe_local_failure(**kwargs)
        if wrote:
            signature=failure_signature(kwargs['details'],lane)
            if signature is not None:
                entry=max((e for e in self.failures if e.owner_member==member),key=lambda e:e.created_update)
                event=RiskObservation(signature,member,lane,self.clock+1,entry.memory_id,'REPEATED_PRIVATE_FAILURE')
                capacity=self.risk_policy['failure_evidence_capacity']
                self.failure_events=(*self._recent(self.failure_events,self.clock+1),event)[-capacity:]
        return wrote

    @staticmethod
    def _groups(events):
        grouped=defaultdict(list)
        for e in events:grouped[e.signature.key()].append(e)
        return grouped

    def _retention_key(self, entry, clock, lane):
        recent=self._recent(entry.observations,clock)
        return (-len(recent),-len({e.member for e in recent}),
            -int(lane in {e.lane for e in recent}),-entry.last_seen_update,entry.signature.key())

    def prepare_outcome(self,outcome):
        if len(self.competence)!=5:raise SearchContractError('INITIAL_MEMORY_NOT_BOOTSTRAPPED')
        return self._prepare_evidence_outcome(outcome)

    def _prepare_shared_risk_outcome(self, outcome):
        self._search_only(outcome.opportunity)
        if not outcome.complete or outcome.operational_failure:
            return RollingMemoryDelta(self.revision,self.private,self.shared,self.clock,self.sequence,
                self.failure_events,tuple(sorted(self.risk_counts.items())))
        op=outcome.opportunity;member=op.target_member
        lane=('general' if op.pattern_context.get('selection_reason')=='SEEDED_ZERO_RESPONSIBILITY_FALLBACK' else op.diagnosis.responsibility[member].primary_lane)
        if member not in range(5) or lane not in LANES:raise SearchContractError('ROLLING_RISK_SCOPE_INVALID')
        clock=self.clock+1;sequence=self.sequence;private=list(self.private)
        failure_events=self._recent(self.failure_events,clock)
        grouped=self._groups(failure_events)
        shared={e.signature.key():e for e in self.shared};before=dict(shared)
        incoming=defaultdict(list)
        for key,events in grouped.items():
            if len({e.member for e in events})>=self.risk_policy['minimum_distinct_members']:
                fresh=[e for e in events if not e.shared_consumed]
                if fresh:incoming[key].extend(fresh if key in shared else events)
        # A bounded acknowledgement bank prevents trimmed shared histories from
        # replaying the same observation into permanent occurrence counters.
        eligible={key for key,events in grouped.items() if len({e.member for e in events})>=2}
        failure_events=tuple(replace(e,shared_consumed=True) if e.signature.key() in eligible else e for e in failure_events)
        for row in outcome.evaluated:
            cid=row.candidate.candidate_id
            if outcome.committed and cid==outcome.selected_candidate_id:
                # Private committed-success contract.
                from .private_action_memory import edit_action
                action=edit_action(op.parent_prompt,row.candidate.prompt)
                if action:
                    d=row.diagnostics
                    sequence+=1
                    oid=sha256(op.opportunity_id.encode()).hexdigest()
                    prompt_id=sha256(row.candidate.prompt.encode()).hexdigest()
                    mid=sha256(f'{sequence}:{oid}:{prompt_id}:SUCCESS'.encode()).hexdigest()
                    measured_outcome = f"Committed; team fixed {int(d.get('team_newly_fixed_count',0))}, broken {int(d.get('team_newly_broken_count',0))}."
                    if outcome.progress_path is not None:
                        from .target_or_team_transition import progress_path
                        import math
                        if (outcome.realized_team_gain is None or outcome.realized_target_gain is None
                                or not all(math.isfinite(v) for v in (outcome.realized_team_gain, outcome.realized_target_gain))
                                or outcome.realized_team_gain < 0
                                or outcome.progress_path != progress_path(outcome.realized_team_gain, outcome.realized_target_gain)
                                or outcome.progress_path not in {'TARGET','TEAM','TARGET_AND_TEAM'}):
                            raise SearchContractError('MEMORY_COMMITTED_PROGRESS_MISMATCH')
                        measured_outcome += (f" realized_team_gain={outcome.realized_team_gain:g};"
                            f" realized_target_gain={outcome.realized_target_gain:g}; progress_path={outcome.progress_path}.")
                    private.append(ActionExperience(mid,member,'SUCCESS',lane,sequence,lane+' repair',action,
                        measured_outcome,
                        'Reuse cautiously against current evidence and preserve fixed-peer competence.',oid,prompt_id))
                continue
            risk=row.diagnostics.get('scientific_risk_code')
            if cid==outcome.selected_candidate_id and outcome.gate_passed is False:risk='SHADOW_REJECTION'
            if risk is None:continue
            if risk not in RISK_CODES:raise SearchContractError('NONSTRUCTURAL_MEMORY_RISK')
            if row.diagnostics.get('operational_failure'):continue
            if row.candidate.backend_details.get('contract_valid') is False:continue
            signature=failure_signature(row.candidate.backend_details,lane)
            if signature is None:signature=FailureSignature(risk,(),'measured_structural_rejection')
            # Sample-level Shadow fields and raw strategy text are never read.
            oid=sha256((op.opportunity_id+':'+cid+':'+risk).encode()).hexdigest()
            incoming[signature.key()].append(RiskObservation(signature,member,lane,clock,oid,'TEAM_STRUCTURAL_REJECTION'))
        for key,events in incoming.items():
            old=shared.get(key);known={e.observation_id for e in old.observations} if old else set()
            new=tuple({e.observation_id:e for e in events if e.observation_id not in known}.values())
            if not new:continue
            combined=(*(old.observations if old else ()),*new)
            combined=tuple({e.observation_id:e for e in combined}.values())
            observations=self._recent(combined,clock)[-self.risk_policy['shared_event_capacity']:]
            shared[key]=SharedRiskEntry(new[0].signature,observations,
                (old.occurrence_count if old else 0)+len(new),
                tuple(sorted(set(old.members if old else ())|{e.member for e in new})),
                tuple(sorted(set(old.observed_lanes if old else ())|{e.lane for e in new})),
                tuple(sorted(set(old.origins if old else ())|{e.origin for e in new})),
                max(e.update for e in combined))
        shared={k:replace(e,observations=self._recent(e.observations,clock)) for k,e in shared.items()
            if self._recent(e.observations,clock)}
        retained=tuple(sorted(shared.values(),key=lambda e:self._retention_key(e,clock,lane))[:self.limits['shared_storage_limit']])
        kept={e.signature.key():e for e in retained};counts=dict(self.risk_counts)
        counts['shared_risk_created']+=sum(k not in before for k in kept)
        counts['shared_risk_updated']+=sum(k in before and kept[k]!=before[k] for k in kept)
        counts['shared_risk_evicted']+=len(set(before)-set(kept))
        counts['failure_to_shared_promotions']+=sum('REPEATED_PRIVATE_FAILURE' in e.origins and
            (k not in before or 'REPEATED_PRIVATE_FAILURE' not in before[k].origins) for k,e in kept.items())
        counts['shared_risk_storage_peak']=max(counts['shared_risk_storage_peak'],len(kept))
        private=tuple(e for i in range(5) for e in sorted((e for e in private if e.owner_member==i),
            key=lambda e:e.created_update)[-self.limits['private_storage_limit']:])
        return RollingMemoryDelta(self.revision,private,retained,clock,sequence,failure_events,tuple(sorted(counts.items())))

    def validate_delta(self, delta):
        super().validate_delta(delta)
        if (not isinstance(delta,EvidenceMemoryDelta) or len(delta.competence)!=5
                or any(not isinstance(e,EditExperience) or e.owner_member not in range(5)
                    or e.record['provenance']['split']!='optimize' for e in delta.private)
                or any(sum(e.owner_member==i for e in delta.private)>24 for i in range(5))
                or any(sum(e.owner_member==i and e.record['status']=='FULL_REFUTED' for e in delta.private)
                    >self.limits['max_failure_entries_per_member'] for i in range(5))
                or any(set(c['current_correct_ids'])-set(c['all_ids']) or
                    set(c['original_correct_ids'])-set(c['all_ids']) for c in delta.competence)):
            raise SearchContractError('EDIT_MEMORY_DELTA_INVALID')
        if (not isinstance(delta,RollingMemoryDelta) or delta.clock not in {self.clock,self.clock+1}
                or type(delta.sequence) is not int or delta.sequence<self.sequence
                or len(delta.failure_events)>self.risk_policy['failure_evidence_capacity']
                or len(delta.shared)>self.limits['shared_storage_limit']
                or len({e.signature.key() for e in delta.shared})!=len(delta.shared)):
            raise SearchContractError('ROLLING_RISK_DELTA_INVALID')
        events=(*delta.failure_events,*(x for e in delta.shared for x in e.observations))
        if (any(not isinstance(x,RiskObservation) or x.member not in range(5) or x.lane not in LANES
                or x.origin not in ORIGINS or type(x.update) is not int
                or not 0<=delta.clock+1-x.update<=self.risk_policy['recent_opportunity_window'] for x in events)
                or any(not e.observations or len(e.observations)>self.risk_policy['shared_event_capacity']
                    or e.occurrence_count<len(e.observations) or e.last_seen_update>delta.clock
                    or any(x.signature!=e.signature for x in e.observations) for e in delta.shared)
                or set(dict(delta.risk_counts))!=set(self.risk_counts)
                or any(type(v) is not int or v<self.risk_counts[k] for k,v in delta.risk_counts)):
            raise SearchContractError('ROLLING_RISK_DELTA_INVALID')

    def apply_outcome(self, delta):
        self.validate_delta(delta)
        prior_revision=self.revision
        super().apply_outcome(delta)
        if self.revision==prior_revision and delta.clock!=self.clock:
            self.revision+=1;self.write_count+=1
        self.clock=delta.clock;self.sequence=delta.sequence;self.failure_events=delta.failure_events
        self.risk_counts=dict(delta.risk_counts)
        self.competence=delta.competence

    def read_for_member(self,member,lane,*,pattern_id=None,hypothesis=None,edited_block=None):
        return self._read_evidence(member,pattern_id,hypothesis,edited_block)

    def audit(self):
        result=super().audit();groups=self._groups(self._recent(self.failure_events,self.clock+1))
        result.update(failure_family_count=len(groups),failure_family_repeated_count=sum(len(v)>1 for v in groups.values()),
            **self.risk_counts,shared_risk_storage_final=len(self.shared),shared_risk_reads=self.counts['shared_reads'],
            failure_evidence_events=len(self.failure_events),risk_clock=self.clock,
            promoted_families=[dict(family_id=e.memory_id,category=e.signature.category,origin=e.origin,
                occurrence_count=e.occurrence_count,distinct_member_count=e.distinct_member_count,
                recent_distinct_member_count=len({x.member for x in self._recent(e.observations,self.clock)}),
                severity_summary=e.signature.category,
                last_seen_update=e.last_seen_update,observed_lanes=e.observed_lanes)
                for e in self.shared if 'REPEATED_PRIVATE_FAILURE' in e.origins])
        result.update(initial_memory_entries=len(self.competence),
            private_success_count_by_member=[sum(e.owner_member==i and e.record['status']=='COMMITTED' for e in self.private) for i in range(5)],
            private_failure_count_by_member=[sum(e.owner_member==i and e.record['status']=='FULL_REFUTED' for e in self.private) for i in range(5)],
            private_edit_count_by_member=[sum(e.owner_member==i for e in self.private) for i in range(5)],
            edit_status_counts=dict(Counter(e.record['status'] for e in self.private)),
            memory_policy_identity=self.identity)
        return result

    def bootstrap(self,store):
        """Existing measured profiles only; no solve, data loader or LLM call."""
        if self.competence:
            raise SearchContractError('INITIAL_MEMORY_BOOTSTRAP_ONCE')
        from .optimization_evidence import prompt_id
        state=store.snapshot();records=[]
        for member in range(5):
            correct=[e.item.input_id for i,e in enumerate(store.examples)
                if state.diagnostics['team_states'][i].team_correctness[member]]
            invalid=[e.item.input_id for i,e in enumerate(store.examples)
                if not state.diagnostics['team_states'][i].team_validity[member]]
            records.append(dict(member=member,prompt_id=prompt_id(state.member_prompts[member]),
                initial_prompt_id=prompt_id(state.member_prompts[member]),
                evaluation_identity=state.diagnostics['evaluation_support_identity'],
                all_ids=[e.item.input_id for e in store.examples],original_correct_ids=correct,
                current_correct_ids=correct,invalid_ids=invalid,newly_fixed_ids=[],newly_broken_ids=[],
                repeated_format_failure_ids=[e.item.input_id for e,p in zip(store.examples,store.profiles[member],strict=True)
                    if p['solver_trajectory']['retry_summary']['repeated_format_failure']],
                task_metadata={e.item.input_id:dict(e.task_metadata) for e in store.examples},
                source='measured_initial_optimize_profiles'))
        self.competence=tuple(records)
        if self.observation_observer:
            self.observation_observer('INITIAL_OPTIMIZE_COMPETENCE_BOOTSTRAPPED',dict(
                member_records=5,source='measured_initial_optimize_profiles',llm_calls=0))

    def observe_edit(self,record):
        if record['member'] not in range(5):
            raise SearchContractError('EDIT_MEMORY_SCOPE_INVALID')
        if record['provenance']['split']!='optimize' or not record['actual_diff']:
            raise SearchContractError('EDIT_MEMORY_PROVENANCE_INVALID')
        from .system_prompt import SystemPrompt, block_lineage
        chain=record.get('edit_chain')
        if not chain:raise SearchContractError('STRUCTURED_EDIT_CHAIN_REQUIRED')
        expected_parent=record['full_parent_prompt_id']
        for step in chain:
            parent=SystemPrompt.from_dict(step['parent_prompt']);child=SystemPrompt.from_dict(step['child_prompt'])
            if (step!=block_lineage(parent,child) or len(step['edited_blocks'])!=1
                    or parent.prompt_hash!=expected_parent):
                raise SearchContractError('STRUCTURED_EDIT_CHAIN_MISMATCH')
            expected_parent=child.prompt_hash
        if (expected_parent!=record['child_prompt_id'] or chain[-1]!=record['structured_lineage']
                or record['actual_block_diff']!=chain[-1]['block_edits'][0]['actual_block_diff']):
            raise SearchContractError('STRUCTURED_EDIT_CHAIN_MISMATCH')
        self.sequence+=1
        entry=EditExperience(record['member'],self.sequence,json.loads(json.dumps(record)))
        others=tuple(e for e in self.private if e.owner_member!=entry.owner_member)
        own=tuple(e for e in self.private if e.owner_member==entry.owner_member)
        self.private=(*others,*(*own,entry)[-self.limits['private_storage_limit']:])
        e=record['effects']['actual_parent_validation']
        if e['valid_to_invalid_ids'] or e['repeated_format_failure_ids']:
            event=RiskObservation(FailureSignature('OUTPUT_CONTRACT_FAILURE',(),'observed_invalid_output_no_mathematical_success'),
                entry.owner_member,'general',self.clock+1,entry.memory_id,'REPEATED_PRIVATE_FAILURE')
            self.failure_events=(*self._recent(self.failure_events,self.clock+1),event)[-self.risk_policy['failure_evidence_capacity']:]
        self.revision+=1;self.write_count+=1

    def _prepare_evidence_outcome(self,outcome):
        from copy import deepcopy
        base=self._prepare_shared_risk_outcome(outcome)
        records={e.memory_id:e for e in self.private if isinstance(e,EditExperience)}
        competence=deepcopy(self.competence)
        if outcome.complete and not outcome.operational_failure:
            for row in outcome.evaluated:
                lineage=row.candidate.backend_details.get('edit_lineage')
                if lineage is None or row.diagnostics.get('operational_failure'):continue
                entry=EditExperience(outcome.opportunity.target_member,0,lineage)
                old=records.get(entry.memory_id)
                if old is None:raise SearchContractError('EDIT_MEMORY_LINEAGE_MISSING')
                record=deepcopy(old.record)
                if row.team_probe and 'edit_effect' in row.team_probe.aggregation_diagnostics:
                    record['effects']['team_probe']=deepcopy(row.team_probe.aggregation_diagnostics['edit_effect'])
                    record['effects']['assigned_repair']=deepcopy(row.team_probe.aggregation_diagnostics.get('assigned_repair_effect'))
                    record['promotion_reason']=row.team_probe.aggregation_diagnostics['promotion_reason']
                if row.full is not None:
                    effect=row.full.aggregation_diagnostics.get('edit_effect')
                    if not effect:raise SearchContractError('FULL_EDIT_EFFECT_MISSING')
                    record['effects']['full']=deepcopy(effect)
                    record['full_evaluation_parent_state_id']=outcome.opportunity.parent_state_id
                    record['status']=('FULL_REFUTED' if row.full.aggregation_diagnostics.get('scientific_risk_code')
                        else 'FULL_SUPPORTED')
                    record['status_history'].append(record['status'])
                    record['promotion_reason']=row.full.aggregation_diagnostics['promotion_reason']
                if row.candidate.candidate_id==outcome.selected_candidate_id and outcome.gate_passed is False:
                    record['promotion_reason']='FULL_PASS_SHADOW_REJECT'
                if outcome.committed and row.candidate.candidate_id==outcome.selected_candidate_id:
                    if row.full is None:raise SearchContractError('COMMITTED_EDIT_REQUIRES_FULL')
                    record['status']='COMMITTED';record['status_history'].append('COMMITTED')
                    record['promotion_reason']='COMMIT'
                    current=competence[entry.owner_member]
                    if current['prompt_id']!=record['full_parent_prompt_id']:
                        raise SearchContractError('COMMITTED_COVERAGE_PARENT_MISMATCH')
                    effect=record['effects']['full']
                    if set(effect['membership'])!=set(current['all_ids']):
                        raise SearchContractError('COMMITTED_COVERAGE_MEMBERSHIP_MISMATCH')
                    correct=(set(current['current_correct_ids'])|set(effect['fixed_ids']))-set(effect['broken_ids'])
                    original=set(current['original_correct_ids'])
                    current.update(prompt_id=record['child_prompt_id'],current_correct_ids=sorted(correct),
                        invalid_ids=effect['invalid_ids'],newly_fixed_ids=sorted(correct-original),
                        newly_broken_ids=sorted(original-correct))
                records[entry.memory_id]=replace(old,record=record)
        fields={name:getattr(base,name) for name in RollingMemoryDelta.__dataclass_fields__}
        retained=[]
        for member in range(5):
            own=sorted((e for e in records.values() if e.owner_member==member),key=lambda e:e.created_update)
            failures=[e for e in own if e.record['status']=='FULL_REFUTED'][-self.limits['max_failure_entries_per_member']:]
            other=[e for e in own if e.record['status']!='FULL_REFUTED'][-(self.limits['private_storage_limit']-len(failures)):]
            retained.extend((*failures,*other))
        fields['private']=tuple(retained)
        return EvidenceMemoryDelta(**fields,competence=tuple(competence))

    def _read_evidence(self,member,pattern_id,hypothesis,edited_block=None):
        if member not in range(5):raise SearchContractError('EDIT_MEMORY_OWNER_REQUIRED')
        if len(self.competence)!=5:raise SearchContractError('INITIAL_MEMORY_NOT_BOOTSTRAPPED')
        c=self.competence[member]
        initial=dict(source='measured_optimize_coverage',original_correct=len(c['original_correct_ids']),
            current_correct=len(c['current_correct_ids']),incorrect=len(c['all_ids'])-len(c['current_correct_ids']),
            invalid=len(c['invalid_ids']),repeated_format_failure=len(c['repeated_format_failure_ids']),
            newly_fixed=len(c['newly_fixed_ids']),newly_broken=len(c['newly_broken_ids']))
        own=sorted((e for e in self.private if e.owner_member==member),key=lambda e:(
            e.record['pattern_id']!=pattern_id,e.record['repair_hypothesis']!=hypothesis,
            edited_block is not None and e.record['edited_block']!=edited_block,
            e.record['status']!='FULL_REFUTED',-e.created_update,e.memory_id))
        selected=own[:self.limits['top_k_private']]
        shared=sorted(self.shared,key=lambda e:-e.last_seen_update)[:self.limits['top_k_shared']]
        def view():return dict(initial_coverage=[initial],edit_effects=[e.visible() for e in selected],
            shared_risk=[e.visible() for e in shared])
        def size():return len(json.dumps(view(),sort_keys=True,separators=(',',':'),ensure_ascii=True))
        while size()>1200:
            if shared:shared.pop()
            elif len(selected)>1:selected.pop()
            elif selected:
                # Keep the most relevant scope and Full outcome within the old bound.
                v=selected[0].visible();v['effects']={k:e for k,e in v['effects'].items()
                    if k in {'search_validation','full'}}
                result=dict(initial_coverage=[initial],edit_effects=[v],shared_risk=[])
                if len(json.dumps(result,separators=(',',':'),ensure_ascii=True))<=1200:
                    return self._evidence_read_record(member,result,selected,shared)
                selected.pop()
            else:raise SearchContractError('EDIT_MEMORY_CONTEXT_LIMIT')
        result=view()
        return self._evidence_read_record(member,result,selected,shared)

    def _evidence_read_record(self,member,result,selected,shared):
        self.context_lengths.append(len(json.dumps(result,sort_keys=True,separators=(',',':'),ensure_ascii=True)))
        self.read_private_count=len(selected)+1;self.read_shared_count=len(shared)
        self.counts['failure_reads']+=sum(e.record['status']=='FULL_REFUTED' for e in selected)
        self.counts['success_reads']+=sum(e.record['status']=='COMMITTED' for e in selected)
        self.counts['shared_reads']+=len(shared)
        if self.observation_observer:
            self.observation_observer('MEMORY_READ',dict(member=member,lane='general',visible=result,
                entries=dict(edit_effects=[dict(memory_id=e.memory_id,visible=e.visible()) for e in selected]),
                context_chars=self.context_lengths[-1]))
        return result
