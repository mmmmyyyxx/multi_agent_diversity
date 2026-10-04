"""Attempt-local actionable experience, without questions, predictions or lineage in views.

The closed action vocabulary is extracted deterministically from the real edit.
Proposal summaries are metadata only: an absent or unsafe summary never vetoes
an otherwise admissible procedure. No summarizer or held-out access exists here.
"""
from dataclasses import dataclass
import hashlib
import json
import re

from .. import versions
from .experience import STRATEGIES, abstract_actions
from .memory import MemoryDelta, RISK_CODES, StructuredLongTermMemoryProviderV1
from .schemas import SearchContractError

LIMITS = dict(top_k_private=3, top_k_shared=3, max_context_chars=1200,
              private_storage_limit=24, shared_storage_limit=48,
              max_failure_entries_per_member=5)
EXTRA_ACTIONS = {
    'domain checks': r'\bdomain\b|\bextraneous\b',
    'exact symbolic forms': r'\bexact\b|\bfractions?\b|\bsymbolic\b',
    'unit checks': r'\bunits?\b|\bdimension\b',
    'sign checks': r'\bsigns?\b',
    'substitution checks': r'\bsubstitut\w*\b',
    'algebraic simplification': r'\bsimplif\w*\b|\brearrang\w*\b',
    'geometric representation': r'\bdiagram\w*\b|\bgeometr\w*\b',
}
LABELS = {k:k.replace('_',' ') for k in STRATEGIES}
LABELS.update({k:k for k in EXTRA_ACTIONS})


def action_features(text):
    found=set(abstract_actions(text))
    # Reuse the negation exclusion used by the existing action extractor.
    positive=re.sub(r'\b(?:do not|never|avoid|without|omit|skip)\b[^.;\n]*', '', text.casefold())
    found.update(k for k,p in EXTRA_ACTIONS.items() if re.search(p,positive))
    return found


def edit_action(parent, candidate):
    old,new=action_features(parent),action_features(candidate)
    added,removed=new-old,old-new
    parts=[]
    if added:parts.append('Add '+', '.join(LABELS[k] for k in sorted(added)))
    if removed:parts.append('Remove '+', '.join(LABELS[k] for k in sorted(removed)))
    if not parts and parent!=candidate and new:
        parts.append('Revise instructions for '+', '.join(LABELS[k] for k in sorted(new)))
    text='; '.join(parts)
    return text if text and len(text)<=240 else None


@dataclass(frozen=True)
class ActionExperience:
    memory_id: str
    owner_member: int | None
    kind: str
    lane: str
    created_update: int
    situation: str
    action: str
    outcome: str
    lesson: str
    source_opportunity_sha256: str
    source_candidate_sha256: str

    def visible(self):
        return {k:getattr(self,k) for k in ('situation','action','outcome','lesson')}


class StructuredActionMemoryV3(StructuredLongTermMemoryProviderV1):
    identity=versions.STRUCTURED_ACTION_MEMORY_VERSION

    def __init__(self, **limits):
        if limits!=LIMITS:raise SearchContractError('ACTION_MEMORY_FROZEN_LIMITS_MISMATCH')
        super().__init__(**{k:v for k,v in limits.items() if k!='max_failure_entries_per_member'})
        self.limits=dict(limits)
        self.failures=()
        self.sequence=0
        self.counts={f'{kind}_{verb}':0 for kind in ('success','failure','shared') for verb in ('reads','writes')}
        self.peak=[0]*5;self.evictions=0;self.context_lengths=[]

    def _entry(self, member, lane, kind, action, outcome, lesson, opportunity_id, prompt):
        self.sequence+=1
        oid=hashlib.sha256(opportunity_id.encode()).hexdigest()
        cid=hashlib.sha256(prompt.encode()).hexdigest()
        mid=hashlib.sha256(f'{self.sequence}:{oid}:{cid}:{kind}'.encode()).hexdigest()
        return ActionExperience(mid,member,kind,lane,self.sequence,
            lane+' repair',action,outcome,lesson,oid,cid)

    def read_for_member(self, member, lane):
        if member not in range(5):raise SearchContractError('ACTION_MEMORY_OWNER_REQUIRED')
        key=lambda e:(e.lane!=lane,-e.created_update,e.memory_id)
        # Failures first; then same-member success, then generic shared risk.
        groups=dict(private_failure=sorted((e for e in self.failures if e.owner_member==member),key=key),
            private_success=sorted((e for e in self.private if e.owner_member==member),key=key)[:self.limits['top_k_private']],
            shared_risk=sorted(self.shared,key=key)[:self.limits['top_k_shared']])
        def view():return {k:[e.visible() for e in entries] for k,entries in groups.items() if entries}
        def size():return len(json.dumps(view(),sort_keys=True,separators=(',',':'),ensure_ascii=True))
        while any(groups.values()) and size()>self.limits['max_context_chars']:
            for name in ('shared_risk','private_success','private_failure'):
                if groups[name]:groups[name].pop();break
        visible=view();self.context_lengths.append(size() if visible else 2)
        for group,kind in (('private_failure','failure'),('private_success','success'),('shared_risk','shared')):
            self.counts[kind+'_reads']+=len(groups[group])
        self.read_private_count=len(groups['private_failure'])+len(groups['private_success'])
        self.read_shared_count=len(groups['shared_risk'])
        return visible

    def read_for_opportunity(self, opportunity):
        self._search_only(opportunity)
        lane=opportunity.diagnosis.responsibility[opportunity.target_member].primary_lane
        return self.read_for_member(opportunity.target_member,lane)

    @staticmethod
    def _search_only(opportunity):
        rows=(*opportunity.evidence.mutation_evidence,*opportunity.evidence.search_validation_evidence)
        if not rows or any(r.source_split!='optimize' for r in rows):
            raise SearchContractError('MEMORY_HELDOUT_ACCESS_FORBIDDEN')

    def observe_local_failure(self, *, member, lane, parent, prompt, details, opportunity_id):
        if (not all(details.get(k) for k in ('changed','contract_valid','solver_evaluated'))
                or details.get('local_parent_correct_delta',0)>=0):return False
        action=details.get('memory_action')
        if action!=edit_action(parent,prompt) or action is None:return False
        measured=bool(details['preservation_locally_measurable'])
        outcome=(f"fixed {details['local_parent_newly_fixed']}, broken {details['local_parent_newly_broken']}, "
                 f"delta {details['local_parent_correct_delta']}; preservation "+
                 (str(details['local_newly_broken']) if measured else 'unmeasurable'))
        entry=self._entry(member,lane,'FAILURE',action,outcome,
            'Avoid repeating this edit without addressing the observed losses.',opportunity_id,prompt)
        own=[e for e in self.failures if e.owner_member==member]+[entry]
        if len(own)>self.limits['max_failure_entries_per_member']:
            # Keep this lane before unrelated lanes, then newer entries. No future outcome.
            own=sorted(own,key=lambda e:(e.lane!=lane,-e.created_update,e.memory_id))[:self.limits['max_failure_entries_per_member']]
            self.evictions+=1
        self.failures=tuple(e for e in self.failures if e.owner_member!=member)+tuple(own)
        self.peak[member]=max(self.peak[member],len(own))
        self.counts['failure_writes']+=1;self.revision+=1;self.write_count+=1
        return True

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

    def apply_outcome(self, delta):
        self.validate_delta(delta)
        prior={e.memory_id for e in (*self.private,*self.shared)}
        self.counts['success_writes']+=sum(e.memory_id not in prior for e in delta.private)
        self.counts['shared_writes']+=sum(e.memory_id not in prior for e in delta.shared)
        super().apply_outcome(delta)

    def audit(self):
        return dict(memory_policy_identity=self.identity,initial_memory_entries=0,
            private_success_count_by_member=[sum(e.owner_member==i for e in self.private) for i in range(5)],
            private_failure_count_by_member=[sum(e.owner_member==i for e in self.failures) for i in range(5)],
            shared_risk_count=len(self.shared),failure_memory_peak_by_member=list(self.peak),
            failure_memory_evictions=self.evictions,**self.counts,
            memory_context_chars=dict(min=min(self.context_lengths,default=0),
                mean=sum(self.context_lengths)/len(self.context_lengths) if self.context_lengths else 0,
                max=max(self.context_lengths,default=0)),
            memory_ids=[e.memory_id for e in (*self.private,*self.failures,*self.shared)],
            stateful_write_count=self.write_count,memory_llm_calls=0)
