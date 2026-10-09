"""Deterministic selected-support, preservation and transition slots."""
from dataclasses import replace
from .. import current_contract as versions
from .pattern_primitives import sample_labels
from .responsibility_value import responsibility_value
from .schemas import SearchContractError, EvidenceView


def compose_disjoint_evidence(rows, context, *, seed, member, ordinal, state_id=''):
    """Fixed disjoint Optimize memberships without accuracy or role quotas."""
    import random
    rows=tuple(rows)
    if any(r.source_split!='optimize' for r in rows):raise SearchContractError('PATTERN_HELDOUT_ACCESS')
    if len({r.example_id for r in rows})!=len(rows):raise SearchContractError('EVIDENCE_ID_DUPLICATE_OR_MISSING')
    if len(rows)<12:raise SearchContractError('DISJOINT_OPTIMIZE_EVIDENCE_INSUFFICIENT')
    focus=next((p for p in context.get('patterns',()) if p['pattern_id']==context.get('focus_mechanism_id')),None)
    support=set(focus['support_ids']) if focus else set()
    by_id={r.example_id:r for r in rows}
    if not support<=set(by_id) or any(by_id[x].signals['target_member_correct'] for x in support):
        raise SearchContractError('ASSIGNED_REPAIR_PROVENANCE_MISMATCH')
    def shuffled(group,lane):
        group=sorted(group,key=lambda r:r.example_id)
        random.Random(f'{seed}:{state_id}:{member}:{ordinal}:{lane}-v2').shuffle(group)
        return group
    selected=shuffled([by_id[x] for x in support],'assigned')
    mutation=selected[:3]
    used={r.example_id for r in mutation}
    mutation+=shuffled([r for r in rows if r.example_id not in used],'mutation-fill')[:3-len(mutation)]
    used={r.example_id for r in mutation}
    validation=shuffled([r for r in rows if r.example_id not in used],'search-validation')[:3]
    used|={r.example_id for r in validation}
    probe=shuffled([r for r in rows if r.example_id not in used],'independent-probe')[:6]
    groups=[{r.example_id for r in group} for group in (mutation,validation,probe)]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)):raise SearchContractError('OPTIMIZE_ROLE_OVERLAP')
    def project(group,role):
        return tuple(replace(r,roles=frozenset({role}|({'REPAIR'} if r.example_id in support else {'PRESERVATION'} if r.signals['target_member_correct'] else {'VALIDATION_REPAIR'})),
            signals={**r.signals,'legacy_tags':('repair',) if r.example_id in support else ('preservation',) if r.signals['target_member_correct'] else ('validation_repair',),
                'feedback':'Measure mathematical correctness and output validity; preserve competence.'}) for r in group)
    memberships=dict(mutation=sorted(groups[0]),search_validation=sorted(groups[1]),team_probe=sorted(groups[2]))
    assigned=sorted(groups[0]&support)
    return EvidenceView(project(mutation,'MUTATION'),project(validation,'SEARCH_VALIDATION'),
        project(probe,'TEAM_PROBE'),'optimize_full','shadow_adaptive'),dict(
        memberships=memberships,assigned_repair_ids=assigned,assigned_repair_scope='SEEN_ASSIGNED_REPAIR',
        seed=seed,state_id=state_id,member_ordinal=ordinal,disjoint=True,
        validation_correct_anchors=[r.example_id for r in validation if r.signals['target_member_correct']],
        validation_selected_support_count=sum(r.example_id in support for r in validation),
        validation_repair_scope='independently_sampled_remaining_optimize',
        mutation_count=3,search_validation_count=3,team_probe_count=6)
