"""Deterministic selected-support, preservation and transition slots."""
from dataclasses import replace
from .. import current_contract as versions
from .pattern_primitives import sample_labels
from .responsibility_value import responsibility_value
from .schemas import SearchContractError, EvidenceView


def compose_disjoint_evidence(rows, context, *, seed, member, ordinal):
    """Internal Optimize roles; no claim of held-out generalization."""
    from .optimization_evidence import rotated_correct
    if any(r.source_split!='optimize' for r in rows):
        raise SearchContractError('PATTERN_HELDOUT_ACCESS')
    focus=next((p for p in context.get('patterns',()) if p['pattern_id']==context.get('focus_mechanism_id')),None)
    support=set(focus['support_ids']) if focus else set()
    correct=rotated_correct(rows,seed=seed,member=member,ordinal=ordinal)
    wrong=sorted((r for r in rows if not r.signals['target_member_correct']),
        key=lambda r:(r.example_id not in support,r.example_id))
    mutation=[r for r in wrong if r.example_id in support][:2] if focus else wrong[:2]
    # Mutation's preservation example is separate from rotating validation anchors.
    mutation+=correct[2:3]
    used={r.example_id for r in mutation}
    mutation+=([r for r in correct[3:] if r.example_id not in used])[:3-len(mutation)]
    used={r.example_id for r in mutation}
    validation=[r for r in wrong if r.example_id not in used][:1]+[r for r in correct if r.example_id not in used][:2]
    used|={r.example_id for r in validation}
    probe=[r for r in wrong if r.example_id not in used][:1]+[r for r in correct if r.example_id not in used][:2]
    if any(len(group)!=3 for group in (mutation,validation,probe)):
        raise SearchContractError('DISJOINT_OPTIMIZE_EVIDENCE_INSUFFICIENT')
    groups=[{r.example_id for r in group} for group in (mutation,validation,probe)]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)):
        raise SearchContractError('OPTIMIZE_ROLE_OVERLAP')
    def project(group, role):
        return tuple(replace(r,roles=frozenset({role}|({'REPAIR'} if r.example_id in support and not r.signals['target_member_correct'] else {'PRESERVATION'} if r.signals['target_member_correct'] else {'VALIDATION_REPAIR'})),
            signals={**r.signals,'legacy_tags':('repair',) if r.example_id in support and not r.signals['target_member_correct'] else ('preservation',) if r.signals['target_member_correct'] else ('validation_repair',),
                'feedback':'Test the repair hypothesis while retaining correct behavior.'}) for r in group)
    membership=dict(mutation=sorted(groups[0]),search_validation=sorted(groups[1]),team_probe=sorted(groups[2]))
    return EvidenceView(project(mutation,'MUTATION'),project(validation,'SEARCH_VALIDATION'),
        project(probe,'TEAM_PROBE'),'optimize_full','shadow_adaptive'),dict(
        memberships=membership,seed=seed,member_ordinal=ordinal,disjoint=True,
        validation_correct_anchors=[r.example_id for r in validation if r.signals['target_member_correct']],
        validation_selected_support_count=sum(r.example_id in support for r in validation),
        validation_repair_scope='selected_support' if any(r.example_id in support for r in validation) else 'other_current_wrong',
        mutation_count=3,search_validation_count=3,team_probe_count=3)
