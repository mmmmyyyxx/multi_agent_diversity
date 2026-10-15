"""One independent single-Prompt optimizer, shared by every module arm."""
import json
import re

from ...local_optimizers.schemas import LocalEvidenceExample
from ...search.system_prompt import candidate_failed_checks, replacement_budgets
from ...search.schemas import SearchContractError
from ...search.optimizer_output_contract import JSON_INSTRUCTION, parse_output
from ...search.generation_failures import GeneratedOutputFailure
from ...search.optimization_evidence import executability_checks
from ...search.reflection_steps import inference_dependency_checks, worked_reference

INSTRUCTION = '''Improve the supplied editable Role/Strategy/Answer System Prompt
to solve future problems accurately. Analyze the supplied Optimize errors freely.
Each reference_solution is the formal bounded dataset worked-solution view.
Its text is Optimize feedback, not hidden Solver reasoning; source, split,
original_characters and truncated explicitly describe its provenance. Structured
diagnoses, patterns and optional module_context are hypotheses or scoped observations,
not proven causes. Generate exactly one genuinely changed modification to exactly
one block. All three blocks are editable. Do not copy questions or example-specific
answers, and never depend on optimizer-only inputs at inference. Return only
{"block":"role or strategy or answer","content":"complete replacement text"}.
The entire rendered Prompt must remain within 3000 characters. Supplied
block_character_limits are exact complete replacement budgets: rewrite or shorten
when space is tight. Test outcomes and team feedback are unavailable.
Output-contract feedback contains only the last structural error, not task evidence.
''' + JSON_INSTRUCTION


def optimizer_error_view(error):
    from copy import deepcopy
    value=deepcopy(error)
    value['reference_solution']=worked_reference(error['reference_solution'])
    return value


def optimizer_packet(prompt,errors,structured_feedback,module_context):
    packet=dict(current_system_prompt=prompt.to_dict(),errors=[optimizer_error_view(e) for e in errors],
        structured_feedback=structured_feedback,block_character_limits=replacement_budgets(prompt),
        full_prompt_characters=len(prompt.render()),maximum_prompt_characters=3000,
        objective='Improve future accuracy with one genuinely changed legal block')
    if module_context is not None:packet['module_context']=module_context
    return packet


def retry_feedback(errors):
    return dict(stage='reflection',errors=errors,maximum_prompt_characters=3000,
        instruction='Return a changed complete replacement; shorten or rewrite to fit the block budget.')


def candidate_checks(child, *, parent, errors, structured_feedback=None):
    """Apply the actual formal guards to all private evidence available here."""
    examples=tuple(LocalEvidenceExample(e['example_id'],e['problem'],e['reference']) for e in errors)
    failed=list(candidate_failed_checks(child,parent_prompt=parent,examples=examples))
    if not failed:
        private=[e.get(k,'') for e in errors for k in ('reference_solution','solver_response')]
        for diagnostic in (structured_feedback or {}).get('diagnostics',[]):
            private.extend(diagnostic.get(k,'') for k in ('observed_failure','diagnosis'))
        failed.extend(executability_checks(child.render(),private_texts=private))
        failed.extend(inference_dependency_checks(child.render()))
    if re.search(r'\bmodule_context\b',child.render(),re.I):
        failed.append('optimizer_only_dependency')
    return tuple(dict.fromkeys(failed))


def generate_candidate(prompt, prepared, broker, ordinal, draws, persist):
    packet=optimizer_packet(prompt,prepared.errors,prepared.structured_feedback,prepared.module_context)
    for draw in range(1, draws + 1):
        response = broker.complete(role='reflection', split='optimize',
            stage=f'round_{ordinal}_legal_candidate_draw_{draw}',
            messages=[dict(role='system', content=INSTRUCTION),
                      dict(role='user', content=json.dumps(packet, sort_keys=True, ensure_ascii=True))])
        child = None
        errors = []
        try:
            value = parse_output(response['text'], stage='reflection')
        except GeneratedOutputFailure as exc:
            errors = [str(exc)]
        else:
            if response.get('finish_reason') != 'stop':
                errors = ['output_truncated']
            elif not isinstance(value, dict) or set(value) != {'block', 'content'}:
                errors = ['invalid_output_schema']
            elif value['block'] not in ('role', 'strategy', 'answer'):
                errors = ['invalid_block']
            else:
                try:
                    child = prompt.edit(value['block'], value['content'])
                except SearchContractError as exc:
                    if str(exc) not in ('SYSTEM_PROMPT_BLOCK_INVALID', 'SYSTEM_PROMPT_OVER_LENGTH'):
                        raise
                    errors = [str(exc)]
                if not errors:
                    errors = list(candidate_checks(child,parent=prompt,errors=prepared.errors,
                        structured_feedback=prepared.structured_feedback))
        persist(f'round_{ordinal}_generation_{draw}', dict(draw=draw, errors=errors,
            request_sha256=response['request_sha256'],
            candidate=child.to_dict() if child is not None and not errors else None))
        if not errors:
            return child, draw
        packet['output_contract_feedback']=retry_feedback(errors)
    return None, draws



def audit_optimizer_request(directory,cell_id,record):
    """Bind the actual paid Optimizer request to its private opportunity evidence."""
    from ...persistence.durable_io import read_json
    logical=record.get('logical_identity',{})
    match=re.fullmatch(r'round_(\d+)_legal_candidate_draw_(\d+)',logical.get('stage',''))
    if match is None:
        raise SearchContractError('ABLATION_OPTIMIZER_REQUEST_STAGE')
    ordinal,draw=map(int,match.groups())
    folder=directory/cell_id
    opportunity=read_json(folder/f'round_{ordinal}_opportunity.json')
    from ...search.system_prompt import SystemPrompt
    parent=SystemPrompt.from_dict(opportunity['parent_prompt'])
    expected=optimizer_packet(parent,opportunity['errors'],opportunity['structured_feedback'],opportunity['module_context'])
    if draw>1:
        prior=read_json(folder/f'round_{ordinal}_generation_{draw-1}.json')
        expected['output_contract_feedback']=retry_feedback(prior['errors'])
    messages=record['request']['messages']
    if (logical.get('ordinal')!=ordinal or not 1<=draw<=4 or len(messages)!=2
            or messages[0]!=dict(role='system',content=INSTRUCTION)
            or messages[1].get('role')!='user' or json.loads(messages[1]['content'])!=expected):
        raise SearchContractError('ABLATION_OPTIMIZER_REQUEST_EVIDENCE_MISMATCH')
