"""Group-diagnostic source contract, never a Prompt candidate legality port.

Group hypotheses are not editable Solver instructions. Preserve their existing
abstraction/source validation without importing it into the candidate pipeline.
"""
import re

_UNAVAILABLE=re.compile(r'\b(?:selected_gradient|solver_trajectory|retrieved_memory|reference_solution|output_contract_feedback|'
    r'root_repair_diagnostic|worked_reference|preceding_attempts|edit_capacity|'
    r'selected_parent_observation|root_observation|repair_direction|scoped_experience|'
    r'selected_parent|selected_parent_sha256|repair_examples|reference_final_answer|accepted_root_diagnostic|'
    r'group_experience|root_diagnostic|applies_to|positive_experience|negative_experience|'
    r'fitness_provenance|selection_plan|parent_population|actual_edit_parent_effect|source_experience_id|'
    r'last_step_instruction|delivered_experience_ids|memory_input_sha256|'
    r'ground[_ -]?truth|gold[_ -]?answer)\b|\b(?:read|consult|copy|access|use)\b[^.;\n]{0,55}'
    r'\b(?:gradient|memory|reference answer|correct answer key)\b',re.I)
_LOOKUP=re.compile(r'\b(?:if the (?:problem|question) (?:is|equals)|answer lookup|memorized answers?)\b',re.I)

def group_diagnostic_source_rejected(text,*,private_texts=()):
    """The unchanged Group narrative check; accepts diagnostic strings only."""
    if not isinstance(text,str):raise TypeError('group diagnostic text required')
    if _UNAVAILABLE.search(text) or _LOOKUP.search(text):return True
    normalized=' '.join(text.casefold().split())
    for source in private_texts:
        words=str(source or '').split()
        if any(' '.join(words[i:i+8]).casefold() in normalized for i in range(max(0,len(words)-7))):return True
    return False
