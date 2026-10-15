"""One shared Pattern support view; no iterative validation/probe partitions."""
from dataclasses import dataclass
from .schemas import SearchContractError
from .parent_population import digest
from .optimization_evidence import EVIDENCE, frozen_policy

@dataclass(frozen=True)
class PatternRepairEvidence:
    repair_examples: tuple
    full_evaluation_scope: str
    adaptive_gate_scope: str
    def __post_init__(self):
        if any(r.source_split!='optimize' for r in self.repair_examples):
            raise SearchContractError('PATTERN_HELDOUT_ACCESS')

class PatternSupportEvidence:
    identity=EVIDENCE
    def __init__(self, seed, policy):
        self.seed=seed; self.policy=frozen_policy(policy)
    def can_compose(self, rows):
        if any(r.source_split!='optimize' for r in rows):raise SearchContractError('PATTERN_HELDOUT_ACCESS')
        return bool(rows) and any(not r.signals['target_member_correct'] for r in rows)
    def compose_opportunity(self,state,diagnosis,member_id,rows,context,*,ordinal):
        if any(r.source_split!='optimize' for r in rows):raise SearchContractError('PATTERN_HELDOUT_ACCESS')
        if len({r.example_id for r in rows})!=len(rows):raise SearchContractError('PATTERN_SUPPORT_SOURCE_MISMATCH')
        focus=next((p for p in context.get('patterns',())
            if p['pattern_id']==context.get('focus_mechanism_id')),None)
        supports=set(focus['support_ids']) if focus else set()
        chosen=sample_pattern_support(rows,supports,seed=self.seed,state_id=state.team_state_id,
            member_id=member_id,ordinal=ordinal)
        view=PatternRepairEvidence(tuple(chosen),'optimize_complete','shadow_private')
        return view,dict(identity=self.identity,selected_pattern_id=focus['pattern_id'] if focus else None,
            pattern_support_ids=sorted(supports),repair_example_ids=[r.example_id for r in chosen],
            sampling='seeded_task_type_round_robin_up_to_three',root_prompt_sha256=state.member_prompts[member_id].prompt_hash,
            gradient_physical_calls=context.get('gradient_physical_calls',0),
            cluster_physical_calls=context.get('cluster_physical_calls',0))


def sample_pattern_support(rows, support_ids, *, seed, state_id, member_id, ordinal):
    """Production representative sampler, shared by team and single-Prompt adapters."""
    rows=tuple(rows)
    if (any(r.source_split!='optimize' for r in rows)
            or len({r.example_id for r in rows})!=len(rows)):
        raise SearchContractError('PATTERN_SUPPORT_SOURCE_MISMATCH')
    supports=set(support_ids)
    eligible=[r for r in rows if r.example_id in supports]
    if any(r.signals['target_member_correct'] for r in eligible) or len(eligible)!=len(supports):
        raise SearchContractError('PATTERN_SUPPORT_SOURCE_MISMATCH')
    # Stable seeded ordering, round-robin across observed task types. No
    # efficacy lookahead or correct-example preservation quota is used.
    groups={}
    for r in eligible:
        key=dict(r.signals.get('task_metadata',{})).get('type','unspecified')
        groups.setdefault(key,[]).append(r)
    salt=[seed,state_id,member_id,ordinal]
    for group in groups.values(): group.sort(key=lambda r:digest([salt,r.example_id]))
    keys=sorted(groups,key=lambda k:digest([salt,k])); chosen=[]
    while len(chosen)<3 and any(groups.values()):
        for k in keys:
            if groups[k] and len(chosen)<3:chosen.append(groups[k].pop(0))
    return tuple(chosen)
