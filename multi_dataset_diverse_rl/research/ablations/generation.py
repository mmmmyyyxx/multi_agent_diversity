"""One independent single-Prompt optimizer, shared by every module arm."""
import json
import re

from ...local_optimizers.schemas import LocalEvidenceExample
from ...search.system_prompt import candidate_failed_checks
from ...search.schemas import SearchContractError
from ...search.optimizer_output_contract import JSON_INSTRUCTION, parse_output
from ...search.generation_failures import GeneratedOutputFailure

INSTRUCTION = '''Improve the supplied editable Role/Strategy/Answer System Prompt
to solve future problems accurately. Analyze the supplied Optimize errors freely.
The reference_solution is dataset feedback, not hidden Solver reasoning. Structured
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


def replacement_budgets(prompt):
    return {k: 3000 - 4 - sum(len(v) for b, v in prompt.to_dict().items() if b != k)
            for k in ('role', 'strategy', 'answer')}


def generate_candidate(prompt, prepared, broker, ordinal, draws, persist):
    packet = dict(current_system_prompt=prompt.to_dict(), errors=prepared.errors,
        structured_feedback=prepared.structured_feedback,
        block_character_limits=replacement_budgets(prompt),
        full_prompt_characters=len(prompt.render()), maximum_prompt_characters=3000,
        objective='Improve future accuracy with one genuinely changed legal block')
    if prepared.module_context is not None:
        packet['module_context'] = prepared.module_context
    examples = tuple(LocalEvidenceExample(e['example_id'], e['problem'], e['reference'])
                     for e in prepared.errors)
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
                    errors = list(candidate_failed_checks(child, parent_prompt=prompt, examples=examples))
                    # Extend the production known-field guard for this new wire
                    # field; modules must not become Solver-time dependencies.
                    if re.search(r'\bmodule_context\b', child.render(), re.I):
                        errors.append('optimizer_only_dependency')
        persist(f'round_{ordinal}_generation_{draw}', dict(draw=draw, errors=errors,
            request_sha256=response['request_sha256'],
            candidate=child.to_dict() if child is not None and not errors else None))
        if not errors:
            return child, draw
        packet['output_contract_feedback'] = dict(stage='reflection', errors=errors,
            maximum_prompt_characters=3000,
            instruction='Return a changed complete replacement; shorten or rewrite to fit the block budget.')
    return None, draws
