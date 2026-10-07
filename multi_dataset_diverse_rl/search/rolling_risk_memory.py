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


class StructuredRollingRiskMemoryV4(PrivateActionMemory):
    identity=versions.STRUCTURED_ROLLING_RISK_MEMORY_VERSION

    def __init__(self, *, risk_policy, **limits):
        if risk_policy!=POLICY:raise SearchContractError('ROLLING_RISK_POLICY_NOT_FROZEN')
        super().__init__(**limits)
        self.risk_policy=json.loads(json.dumps(POLICY))
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

    def prepare_outcome(self, outcome):
        self._search_only(outcome.opportunity)
        if not outcome.complete or outcome.operational_failure:
            return RollingMemoryDelta(self.revision,self.private,self.shared,self.clock,self.sequence,
                self.failure_events,tuple(sorted(self.risk_counts.items())))
        op=outcome.opportunity;member=op.target_member
        lane=op.diagnosis.responsibility[member].primary_lane
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

    def read_for_member(self, member, lane):
        if member not in range(5) or lane not in LANES:raise SearchContractError('ROLLING_RISK_SCOPE_INVALID')
        events=self._recent(self.failure_events,self.clock+1)
        recurrence=Counter(e.signature.key() for e in events if e.member==member)
        # Private entries remain member-private; recurrence strengthens relevance.
        own=[e for e in self.failures if e.owner_member==member]
        by_id={e.observation_id:e.signature.key() for e in events}
        own.sort(key=lambda e:(e.lane!=lane,-recurrence[by_id.get(e.memory_id,'')],-e.created_update,e.memory_id))
        success=sorted((e for e in self.private if e.owner_member==member),key=lambda e:(e.lane!=lane,-e.created_update,e.memory_id))[:self.limits['top_k_private']]
        relevant={e.signature.key() for e in events if e.member==member and e.lane==lane}
        shared=sorted((e for e in self.shared if self._recent(e.observations,self.clock)),key=lambda e:(
            lane not in e.observed_lanes,e.signature.key() not in relevant,-e.last_seen_update,
            -len(self._recent(e.observations,self.clock)),e.signature.key()))[:self.limits['top_k_shared']]
        groups=dict(private_failure=own,private_success=success,shared_risk=shared)
        def view():return {k:[e.visible() for e in values] for k,values in groups.items() if values}
        def size():return len(json.dumps(view(),sort_keys=True,separators=(',',':'),ensure_ascii=True))
        while any(groups.values()) and size()>self.limits['max_context_chars']:
            for k in ('shared_risk','private_success','private_failure'):
                if groups[k]:groups[k].pop();break
        visible=view();self.context_lengths.append(size())
        for group,kind in (('private_failure','failure'),('private_success','success'),('shared_risk','shared')):
            self.counts[kind+'_reads']+=len(groups[group])
        self.read_private_count=len(groups['private_failure'])+len(groups['private_success'])
        self.read_shared_count=len(groups['shared_risk'])
        if self.observation_observer:
            self.observation_observer('MEMORY_READ',dict(member=member,lane=lane,
                entries={k:[dict(memory_id=e.memory_id,visible=e.visible()) for e in values] for k,values in groups.items()},
                visible=visible,context_chars=size()))
        return visible

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
        return result
