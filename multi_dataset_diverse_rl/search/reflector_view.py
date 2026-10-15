"""Verified readable projection; canonical trajectories remain privately persisted."""
from copy import deepcopy
import json
from .group_diagnosis import readable_observation
from .schemas import SearchContractError
from .parent_population import digest

POLICY=dict(identity='DIRECT_READABLE_REFLECTOR_VIEW_V3',
    duplicate_rule='complete_identical_validated_trajectory_and_correctness',
    preview='canonical_only',audit_metadata='canonical_only_except_memory_scope',
    diagnostics=['observed_failure','diagnosis','reusable_correction'],
    reference='existing_4096_prefix_explicit_truncation',memory='unchanged_scoped_cards')

def serialized(value,*,ensure_ascii=False):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=ensure_ascii)

def project_reflector(packet,*,root_prompt,selected_parent,target_member):
    from ..benchmarks.math_response_evidence import validate_adaptive_trajectory
    examples=[];duplicates=0;preview_chars=0;raw_chars=0;actual_chars=0
    for e in packet['repair_examples']:
        root=e['root_observation'];parent=e['selected_parent_observation']
        if root['prompt_sha256']!=root_prompt.prompt_hash or parent['prompt_sha256']!=selected_parent.prompt_hash:
            raise SearchContractError('REFLECTOR_PROMPT_SOURCE_MISMATCH')
        observations=[]
        for label,obs,prompt in [('root',root,root_prompt),('selected_parent',parent,selected_parent)]:
            tr=validate_adaptive_trajectory(obs['trajectory'],example_id=e['example_id'],
                member_id=target_member,prompt=prompt,problem=e['problem'])
            raw_chars+=len(tr['observed_response']);preview_chars+=len(tr['visible_solution'])
            observations.append(dict(applies_to=[label],**readable_observation(tr,correct=obs['correct'])))
        same=(root['correct']==parent['correct'] and digest(root['trajectory'])==digest(parent['trajectory']))
        if same:
            observations=observations[:1];observations[0]['applies_to']=['root','selected_parent'];duplicates+=1
        actual_chars+=sum(len(o['observed_response']) for o in observations)
        diagnostic=e['accepted_root_diagnostic']
        examples.append(dict(example_id=e['example_id'],problem=e['problem'],
            reference_final_answer=e['reference_final_answer'],worked_reference=deepcopy(e['worked_reference']),
            observations=observations,root_diagnostic={k:diagnostic[k] for k in POLICY['diagnostics']}))
    direction=dict(generalized_gradient=packet['repair_direction']['generalized_gradient'],source='root_diagnostics')
    view=dict(schema='GROUP_DIAGNOSED_READABLE_REFLECTION_INPUT_V6',selected_parent=deepcopy(packet['selected_parent']),
        repair_direction=direction,repair_examples=examples,scoped_experience=deepcopy(packet['scoped_experience']),
        group_experience=deepcopy(packet['group_experience']))
    # The old comparison excludes the new channel; it is a wire-template size
    # comparison on these same canonical records, never an old model replay.
    old={k:v for k,v in packet.items() if k!='group_experience'}
    return view,dict(original_observation_body_characters=raw_chars+preview_chars,
        original_full_response_characters=raw_chars,deduplicated_response_characters=actual_chars,
        preview_removed_characters=preview_chars,root_parent_deduplications=duplicates,
        root_parent_response_removed_characters=raw_chars-actual_chars,
        old_packet_serialized_characters=len(serialized(old,ensure_ascii=True)),new_packet_serialized_characters=len(serialized(view)),
        group_experience_characters=len(serialized(view['group_experience'])))
