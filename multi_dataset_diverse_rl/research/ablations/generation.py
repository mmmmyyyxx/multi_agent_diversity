"""One independent single-Prompt optimizer, shared by every module arm."""
import json
import re

from ...search.system_prompt import candidate_failed_checks
from ...search.schemas import SearchContractError
from ...search.optimizer_output_contract import JSON_INSTRUCTION, parse_output
from ...search.generation_failures import GeneratedOutputFailure
from ...search.candidate_contract import parse_candidate_edit
from ...search.reflection_steps import worked_reference

INSTRUCTION = '''Improve the supplied editable Role/Strategy/Answer System Prompt
to solve future problems accurately. Analyze the supplied Optimize errors freely.
Each reference_solution is the formal bounded dataset worked-solution view.
Its text is Optimize feedback, not hidden Solver reasoning; source, split,
original_characters and truncated explicitly describe its provenance. Structured
diagnoses, patterns and optional module_context are hypotheses or scoped observations,
not proven causes. Generate exactly one genuinely changed modification to exactly
one block. All three blocks are editable. You may extract useful domain knowledge
and strategies from training feedback. At inference the Solver receives only the
edited Prompt and the new problem; optimizer-side inputs are not supplied. Return
{"block":"role or strategy or answer","content":"complete replacement text"}.
There is no character budget. Test outcomes and team feedback are unavailable.
Output-contract feedback contains only the last structural error, not task evidence.
''' + JSON_INSTRUCTION


def optimizer_error_view(error):
    from copy import deepcopy
    value=deepcopy(error)
    value['reference_solution']=worked_reference(error['reference_solution'])
    return value


def optimizer_packet(prompt,errors,structured_feedback,module_context,parent_context=None):
    packet=dict(current_system_prompt=prompt.to_dict(),errors=[optimizer_error_view(e) for e in errors],
        structured_feedback=structured_feedback,
        objective='Improve future accuracy with one block replacement')
    if module_context is not None:packet['module_context']=module_context
    if parent_context is not None:packet['parent_context']=parent_context
    return packet


def retry_feedback(errors):
    return dict(stage='reflection',errors=errors,
        instruction='Return one named block and its nonempty string replacement as an unambiguous object.')


def candidate_checks(child, *, parent, errors, structured_feedback=None):
    """All arms share the structural contract; evidence never filters text."""
    return candidate_failed_checks(child,parent_prompt=parent)


def generate_candidate(prompt, prepared, broker, ordinal, draws, persist,*,parent_context=None):
    packet=optimizer_packet(prompt,prepared.errors,prepared.structured_feedback,prepared.module_context,parent_context)
    instruction=INSTRUCTION if parent_context is None else INSTRUCTION+POPULATION_CONTEXT_INSTRUCTION
    for draw in range(1, draws + 1):
        response = broker.complete(role='reflection', split='optimize',
            stage=f'round_{ordinal}_legal_candidate_draw_{draw}',
            messages=[dict(role='system', content=instruction),
                      dict(role='user', content=json.dumps(packet, sort_keys=True, ensure_ascii=True))])
        child = None
        errors = []
        try:
            edit = parse_candidate_edit(response['text'],parent=prompt,
                finish_reason=response.get('finish_reason'))
        except GeneratedOutputFailure as exc:
            errors = [str(exc)]
        else:
            child = edit.prompt
            errors = list(candidate_checks(child,parent=prompt,errors=prepared.errors))
        persist(f'round_{ordinal}_generation_{draw}', dict(draw=draw, errors=errors,
            request_sha256=response['request_sha256'],
            candidate=child.to_dict() if child is not None and not errors else None,
            disposition='NO_OP' if child==prompt and not errors else 'GENERATED' if not errors else 'FORMAT_RETRY'))
        if not errors:
            return child, draw
        packet['output_contract_feedback']=retry_feedback(errors)
    return None, draws



POPULATION_CONTEXT_INSTRUCTION='''
Errors and structured diagnoses belong to the running Root Prompt. You are editing
the selected Parent in current_system_prompt. Its actual observations are labeled
in parent_context and may already solve a Root error. Check applicability using
these distinct observations; do not treat a Root diagnosis as a Parent diagnosis.
The other candidates from this round, pool fitness and Validation are unavailable.
'''


def audit_optimizer_request(directory,cell_id,record):
    """Bind the actual paid Optimizer request to its private opportunity evidence."""
    from ...persistence.durable_io import read_json
    logical=record.get('logical_identity',{})
    match=re.fullmatch(r'round_(\d+)(?:_slot_([0-2]))?_legal_candidate_draw_(\d+)',logical.get('stage',''))
    if match is None:
        raise SearchContractError('ABLATION_OPTIMIZER_REQUEST_STAGE')
    ordinal,slot,draw=match.groups();ordinal,draw=int(ordinal),int(draw)
    prefix=f'round_{ordinal}'+(f'_slot_{slot}' if slot is not None else '')
    folder=directory/cell_id
    opportunity=read_json(folder/f'{prefix}_opportunity.json')
    from ...search.system_prompt import SystemPrompt
    parent=SystemPrompt.from_dict(opportunity['parent_prompt'])
    expected=optimizer_packet(parent,opportunity['errors'],opportunity['structured_feedback'],opportunity['module_context'],opportunity.get('parent_context'))
    if draw>1:
        prior=read_json(folder/f'{prefix}_generation_{draw-1}.json')
        expected['output_contract_feedback']=retry_feedback(prior['errors'])
    messages=record['request']['messages']
    if (logical.get('ordinal')!=ordinal or not 1<=draw<=4 or len(messages)!=2
            or messages[0]!=dict(role='system',content=INSTRUCTION+(POPULATION_CONTEXT_INSTRUCTION if slot is not None else ''))
            or messages[1].get('role')!='user' or json.loads(messages[1]['content'])!=expected):
        raise SearchContractError('ABLATION_OPTIMIZER_REQUEST_EVIDENCE_MISMATCH')
