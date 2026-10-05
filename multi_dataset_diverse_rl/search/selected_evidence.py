"""Deterministic selected-support, preservation and transition slots."""
from dataclasses import replace
from .. import current_contract as versions
from .pattern_primitives import sample_labels
from .responsibility_value import responsibility_value
from .schemas import SearchContractError, EvidenceView

def compose_selected_evidence(rows,pattern_context,minimum=3):
    if any(r.source_split!='optimize' for r in rows):raise SearchContractError('PATTERN_HELDOUT_ACCESS')
    context=pattern_context or {}
    focus=next((p for p in context.get('patterns',()) if p['pattern_id']==context.get('focus_mechanism_id')),None)
    if focus is None:raise SearchContractError('PATTERN_DISCOVERY_NOT_ACTIONABLE')
    support=set(focus['support_ids']); byid={r.example_id:r for r in rows}
    if not support or not support<=set(byid):raise SearchContractError('PATTERN_SUPPORT_MAPPING_INVALID')
    ordered=sorted((byid[x] for x in support),key=lambda r:(-responsibility_value(*(int(k in sample_labels(r)) for k in ('direct_flip','near_margin','coverage'))),r.example_id))
    repairs=[replace(r,roles=frozenset({'REPAIR','FOCUS_REPAIR'}|set(sample_labels(r))|({'TRANSITION_FOCUS','TRANSITION_ANCHOR'}&r.roles)),
        signals={**r.signals,'legacy_tags':('repair','focus_repair_v2'),
            'feedback':'Repair only the selected semantic mechanism.'}) for r in ordered]
    safety=[r for r in rows if r.example_id not in support and
        (r.signals.get('target_member_correct') is True or {'TRANSITION_FOCUS','TRANSITION_ANCHOR'} & r.roles)]
    safety=sorted(safety,key=lambda r:(not r.signals.get('target_member_correct'),
        not bool(r.signals.get('mutation_sensitive')),-r.signals.get('team_disagreement',0),r.signals.get('team_margin',0),r.example_id))
    safety=[replace(r,roles=(r.roles-{'REPAIR','TEAM_HARD','direct_flip','near_margin','coverage','pure_coverage'})|{'SAFETY_BOUNDARY'},
        signals={**r.signals,'legacy_tags':('safety_boundary_v2',),'feedback':'Safety only: preserve competence; no second repair objective.'}) for r in safety]
    # The preservation anchor and recent transition safeguards have distinct
    # slots. Extra correct examples cannot displace transition context.
    chosen=repairs[:3]
    preservation=next((r for r in safety if r.signals.get('target_member_correct')),None)
    if preservation is not None:chosen.append(preservation)
    for role in ('TRANSITION_FOCUS','TRANSITION_ANCHOR'):
        boundary=next((r for r in safety if role in r.roles),None)
        if boundary is not None and boundary.example_id not in {r.example_id for r in chosen}:chosen.append(boundary)
    if len(chosen)<minimum:
        chosen+=[r for r in repairs[3:]+safety if r.example_id not in {x.example_id for x in chosen}][:minimum-len(chosen)]
    if len(chosen)<minimum:
        raise SearchContractError('FOCUSED_BACKEND_MINIMUM_WITHOUT_LEGAL_BOUNDARIES')
    chosen=tuple(chosen[:6])
    view=EvidenceView(tuple(replace(r,roles=r.roles|{'MUTATION'}) for r in chosen),
        tuple(replace(r,roles=r.roles|{'SEARCH_VALIDATION'}) for r in chosen),
        tuple(replace(r,roles=r.roles|{'TEAM_PROBE'}) for r in chosen),'optimize_full','shadow_adaptive')
    return view,dict(pattern_enabled=True,focus_pattern_id=focus['pattern_id'],
        representative_pattern_examples=sum('REPAIR' in r.roles for r in chosen),
        preservation_anchor_count=sum(r.signals.get('target_member_correct') is True for r in chosen),
        transition_evidence_count=sum(bool({'TRANSITION_FOCUS','TRANSITION_ANCHOR'}&r.roles) for r in chosen),
        nonfocus_repair_count=0,generic_backfill_count=0,mutation_count=len(chosen),
        search_validation_count=len(chosen),team_probe_count=len(chosen),pattern_selection_authority=versions.RAW_RESPONSIBILITY_VALUE_VERSION)
