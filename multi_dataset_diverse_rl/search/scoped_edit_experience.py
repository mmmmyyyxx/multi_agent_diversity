"""Deterministic same-member experience cards; no provider or causal attribution.

Packing reserves measured scopes before optional complete text. Raw records and
edit chains stay private. Shared observations use the existing closed risk view.
"""
from copy import deepcopy
import json
import re
from .private_action_memory import action_features, LABELS
from .schemas import SearchContractError

SCHEMA = 'SCOPED_EDIT_EXPERIENCE_VIEW_V1'
CAUTION = 'Earlier Optimize observations describe complete edit chains, not causal rules. Recheck applicability; missing scopes are unmeasured.'
STOPWORDS = frozenset('the a an of to and or in for on with from by is are be as this that it its use check before after when ensure each any all should must not then more only'.split())


def serialized_size(value):
    return len(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True))


def terms(text):
    return frozenset(x for x in re.findall(r'[a-z][a-z_]{3,}',text.casefold()) if x not in STOPWORDS)


def actual_concepts(record):
    """Changed spans only: inherited parent instructions are not a new action."""
    concepts=set()
    for step in record['edit_chain']:
        for block in step['block_edits']:
            for edit in block['actual_block_diff']:
                concepts.update(action_features(edit['added']))
                concepts.update(action_features(edit['removed']))
    return sorted(concepts)


def task_types(record,metadata):
    context=record.get('repair_context',{})
    if 'task_types' in context:return sorted(set(context['task_types']))
    # Archive inspection may lack the newer field. This derives labels only;
    # it never makes historical records eligible for a current runtime import.
    return sorted({metadata[x]['type'] for x in record.get('supporting_example_ids',())
        if x in metadata and isinstance(metadata[x].get('type'),str)})


def relation(record,context):
    prompt=context.get('root_prompt_id')
    team=context.get('parent_state_id')
    return dict(prompt='unknown' if prompt is None else 'same' if prompt==record['full_parent_prompt_id'] else 'different',
        child='unknown' if prompt is None else 'same' if prompt==record['child_prompt_id'] else 'different',
        team='unknown' if team is None or record.get('parent_state_id') is None else
            'same' if team==record['parent_state_id'] else 'different')


def relevance(record,context,metadata):
    old_types=set(task_types(record,metadata));new_types=set(context.get('task_types',()))
    text=record['repair_hypothesis'];hypothesis=context.get('hypothesis') or ''
    common=terms(text)&terms(hypothesis)
    concepts=set(actual_concepts(record))&action_features(hypothesis)
    rel=relation(record,context)
    exact=bool(context.get('pattern_id') and context['pattern_id']==record['pattern_id']) or bool(hypothesis and hypothesis==text)
    # Topic alone is an applicability cue, never a learned mathematical cause.
    match=exact or bool(common) or bool(concepts) or bool(old_types&new_types) or rel['team']=='same' or rel['prompt']=='same' or rel['child']=='same'
    rank=(-int(exact),-len(common),-len(concepts),-len(old_types&new_types),
        -int(context.get('edited_block')==record['edited_block']),
        -int(rel['team']=='same'),-int(rel['prompt']=='same'),-int(rel['child']=='same'))
    return match,rank


def effect_view(effect):
    return dict(n=len(effect['membership']),fixed=len(effect['fixed_ids']),
        lost=len(effect['broken_ids']),invalid=len(effect['invalid_ids']),
        target_delta=effect['member_delta'],
        **({'vote_delta':effect['team_delta']} if 'team_delta' in effect else {}))


def compact_card(entry,context,metadata):
    r=entry.record
    scope='full' if 'full' in r['effects'] else 'search_validation'
    effects={scope:effect_view(r['effects'][scope])}
    operations=sorted({d['operation'] for step in r['edit_chain'] for b in step['block_edits'] for d in b['actual_block_diff']})
    card=dict(id=entry.memory_id[:12],status=r['status'],chain=len(r['edit_chain']),
        edit=dict(blocks=sorted({b for step in r['edit_chain'] for b in step['edited_blocks']}),
            operations=operations),effects=effects,
        opportunity_root_match=relation(r,context))
    if scope=='full':card['local_root']=dict(n=len(r['effects']['search_validation']['membership']),
        target_delta=r['effects']['search_validation']['member_delta'])
    if r.get('promotion_reason')=='FULL_PASS_SHADOW_REJECT':card['admission']='SHADOW_REJECTED'
    return card


def complete_changed_instruction(record):
    """A complete sentence intersecting the last actual child edit; never a slice."""
    step=record['edit_chain'][-1];block=step['edited_blocks'][0]
    text=step['child_prompt'][block]
    spans=step['block_edits'][0]['actual_block_diff']
    # Sentence boundaries need whitespace, avoiding decimal-point splitting.
    bounds=[0]+[m.end() for m in re.finditer(r'(?<=[.!?;])\s+|\n+',text)]+[len(text)]
    for a,b in zip(bounds,bounds[1:]):
        raw=text[a:b];clause=raw.strip()
        content_start=a+len(raw)-len(raw.lstrip())
        content_end=b-len(raw)+len(raw.rstrip())
        if clause and any(content_start<d['child_span'][1] and content_end>d['child_span'][0] for d in spans if d['added']):
            from .optimization_evidence import POLICY
            if len(clause)<=POLICY['memory_whole_instruction_limit']:return clause
    return None


def observation_key(entry):
    r=entry.record
    return json.dumps([r['full_parent_prompt_id'],r['child_prompt_id'],r.get('parent_state_id'),r['effects']],sort_keys=True,separators=(',',':'))


def pack_experience_view(entries,shared,initial,*,context,metadata,limits):
    """Protect related Full evidence; do not let optional prose evict its card."""
    entries=tuple(entries);related=[];unrelated=[];groups={}
    for entry in entries:
        match,rank=relevance(entry.record,context,metadata)
        (related if match else unrelated).append(((*rank,-entry.created_update,entry.memory_id),entry))
    ranked=[e for _,e in sorted(related)]
    # Calls without a query are inspection/initial reads, not a claimed match.
    if not any(context.get(k) for k in ('pattern_id','hypothesis','task_types','parent_state_id')):
        ranked=[e for _,e in sorted((*related,*unrelated))]
    unique=[]
    for entry in ranked:
        key=observation_key(entry)
        if key in groups:continue
        groups[key]=entry;unique.append(entry)
    full=[e for e in unique if 'full' in e.record['effects']]
    local=[e for e in unique if 'full' not in e.record['effects']]
    candidates=[]
    if full:
        candidates.append(full[0])
        positive=full[0].record['status'] in {'COMMITTED','FULL_SUPPORTED'}
        other=next((e for e in full[1:] if (e.record['status'] in {'COMMITTED','FULL_SUPPORTED'})!=positive),None)
        if other:candidates.append(other)
    if local:candidates.append(local[0])
    candidates.extend(e for e in unique if e not in candidates)
    candidates=candidates[:limits['top_k_private']]
    value=dict(schema=SCHEMA,initial_coverage=[initial],edit_effects=[],shared_risk=[],interpretation=CAUTION)
    if serialized_size(value)>limits['max_context_chars']:raise SearchContractError('EDIT_MEMORY_CONTEXT_LIMIT')
    selected=[]
    for entry in candidates:
        card=compact_card(entry,context,metadata)
        if not selected and full:
            instruction=complete_changed_instruction(entry.record)
            if instruction:card['last_step_instruction']=instruction
        value['edit_effects'].append(card)
        if serialized_size(value)>limits['max_context_chars']:
            card.pop('last_step_instruction',None)
            if serialized_size(value)>limits['max_context_chars']:
                value['edit_effects'].pop();continue
        selected.append(entry)
    # A bounded card already contains actual operations and measured scopes.
    # Enrich with complete instructions/context, never truncate mathematical prose.
    for entry,card in zip(selected,value['edit_effects'],strict=True):
        r=entry.record
        optional=[('last_step_instruction',complete_changed_instruction(r)),('concepts',actual_concepts(r)),
            ('task_types',task_types(r,metadata)),('hypothesis',r['repair_hypothesis']),
            ('actual_parent_validation',effect_view(r['effects']['actual_parent_validation']))]
        for key,content in optional:
            if not content:continue
            if key in card:continue
            card[key]=content
            if serialized_size(value)>limits['max_context_chars']:card.pop(key)
    # Closed shared risks never evict delivered edit evidence or whole rules.
    chosen_shared=[]
    for entry in shared[:limits['top_k_shared']]:
        value['shared_risk'].append(entry.visible())
        if serialized_size(value)>limits['max_context_chars']:
            value['shared_risk'].pop();continue
        chosen_shared.append(entry)
    if full and (not selected or 'full' not in selected[0].record['effects']):
        raise SearchContractError('RELATED_FULL_MEMORY_NOT_DELIVERED')
    audit=dict(bank_entries=len(entries),related_entries=len(ranked),duplicate_observations=len(ranked)-len(unique),
        related_full_available=len(full),full_delivered=sum('full' in e.record['effects'] for e in selected),
        delivered_ids=[e.memory_id for e in selected],
        instruction_delivered=sum('last_step_instruction' in c for c in value['edit_effects']),
        omitted_candidate_ids=[e.memory_id for e in candidates if e not in selected],
        serialized_characters=serialized_size(value),causal_utilization_claim=False)
    return value,tuple(selected),tuple(chosen_shared),audit
