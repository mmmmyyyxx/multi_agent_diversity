"""One unconditional single-Prompt update per round; no team optimization."""
from copy import deepcopy
import hashlib
import json
import random

from ..benchmarks.math_response_evidence import adaptive_trajectory
from ..local_optimizers.schemas import LocalEvidenceExample
from ..search.schemas import EvidenceItem, SearchContractError
from ..search.system_prompt import SystemPrompt, candidate_failed_checks, block_lineage
from ..search.optimizer_output_contract import JSON_INSTRUCTION
from ..search.optimization_evidence import coverage_effect
from ..search.textual_gradients import GradientExtractor, gradient_identity
from .single_prompt_policy import policy, canary_contract, ARMS

OPTIMIZER_PROMPT = '''Improve the supplied editable Role/Strategy/Answer System Prompt
to solve future problems accurately. Analyze the supplied Optimize errors freely,
including any common causes or useful patterns you discover. The reference_solution
is dataset feedback, not hidden Solver reasoning. Use supplied structured diagnoses
and patterns when present; they are hypotheses, not proven causes. Generate exactly
one modification to exactly one block. All three blocks are editable. Do not copy
questions, example-specific answers, or depend on optimizer-only inputs at inference.
Return only {"block":"role or strategy or answer","content":"complete replacement
text for that block"}. The rendered complete Prompt must remain within 3000 characters.
No historical scores, best Prompt, memory, held-out outcomes or team feedback exist.
Output-contract feedback, when present, contains only the last structural error.
''' + JSON_INSTRUCTION


def batch_ids(wrong_ids, seed, ordinal, size):
    if len(set(wrong_ids)) != len(wrong_ids):
        raise SearchContractError('STUDY_DUPLICATE_ERROR_ID')
    key = hashlib.sha256(json.dumps([seed, ordinal, 'error_batch_v1']).encode()).hexdigest()
    selected = set(random.Random(int(key, 16)).sample(tuple(wrong_ids), min(size, len(wrong_ids))))
    return tuple(x for x in wrong_ids if x in selected)


def raw_error(row, observation):
    return dict(example_id=row['stable_example_id'], problem=row['content']['problem'],
        reference=row['reference_final_answer'], reference_solution=row['content']['solution'],
        solver_response=observation['prediction']['text'], prediction=observation['answer'],
        valid=observation['valid'], correct=False,
        invalid_reason=observation['prediction']['invalid_reason'],
        semantic_attempt_count=observation['prediction']['semantic_attempt_count'])


def structured_feedback(prompt, rows, observations, broker):
    evidence = []
    for row in rows:
        ident = row['stable_example_id']
        observed = observations[ident]
        evidence.append(EvidenceItem(ident, 'optimize', frozenset({'pattern'}), dict(
            input_payload=row['content']['problem'], gold=row['reference_final_answer'],
            reference_solution=row['content']['solution'], target_output=observed['answer'],
            target_member_correct=False, target_member_valid=observed['valid'],
            correctness_signal_identity='target_member_correctness_validity_v1',
            responsibility_labels=(),
            solver_trajectory=adaptive_trajectory(observed['profile'], member_id=0,
                prompt=prompt, example_id=ident, problem=row['content']['problem']))))
    if not evidence:
        return dict(gradients=[], diagnostics=[], patterns=[], unassigned_ids=[], partition_audit=[])
    from .single_prompt_runtime import StudyGradientProvider, StudyClusterProvider
    extractor = GradientExtractor(StudyGradientProvider(broker))
    gradients = extractor.extract(prompt, evidence)
    patterns = dict(patterns=[], unassigned_ids=[])
    cluster = StudyClusterProvider(broker)
    if gradients:
        actionable = {g['example_id'] for g in gradients}
        patterns = cluster.cluster(dict(gradients=list(gradients)),
            evidence_rows=tuple(e for e in evidence if e.example_id in actionable))
        # Preserve the formal normalized corrective identity and merge rule,
        # without computing team responsibility or selecting a focus.
        merged = {}
        for item in patterns['patterns']:
            ident = gradient_identity(item['generalized_gradient'])
            if ident not in merged:
                merged[ident] = dict(pattern_id=ident, generalized_gradient=item['generalized_gradient'], support_ids=[])
            merged[ident]['support_ids'] = sorted(set(merged[ident]['support_ids']) | set(item['support_ids']))
        patterns['patterns'] = [merged[i] for i in sorted(merged)]
    return dict(gradients=list(gradients), diagnostics=extractor.evidence_diagnostics,
        **patterns, partition_audit=cluster.partition_audit,
        cluster_structural_errors=getattr(cluster, 'structural_errors', []))


def generate_candidate(prompt, feedback, broker, round_index):
    p = policy()
    payload = dict(current_system_prompt=prompt.to_dict(), errors=feedback['errors'],
        structured_feedback=feedback['structured_feedback'],
        objective='Improve future task accuracy using exactly one block edit')
    audits = []
    examples = tuple(LocalEvidenceExample(e['example_id'], e['problem'], e['reference'])
                     for e in feedback['errors'])
    for draw in range(1, p['optimizer_structural_draws'] + 1):
        response = broker.complete(role='reflection', split='optimize',
            stage=f'round_{round_index}_single_candidate_draw_{draw}',
            messages=[dict(role='system', content=OPTIMIZER_PROMPT),
                      dict(role='user', content=json.dumps(payload, ensure_ascii=True, sort_keys=True))])
        # Provider/ledger/persistence failures propagate; only generated output is local.
        errors = []
        try:
            value = json.loads(response['text'])
        except (json.JSONDecodeError, TypeError):
            errors = ['invalid_json']
        if not errors:
            if response['finish_reason'] != 'stop':
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
                    errors = [r for r in candidate_failed_checks(child, parent_prompt=prompt, examples=examples)
                              if r != 'parent_no_op']
        audits.append(dict(draw=draw, errors=errors, request_sha256=response['request_sha256']))
        if not errors:
            return child, audits
        payload['output_contract_feedback'] = dict(stage='reflection', errors=errors)
    return prompt, audits


def run_cell(cell, rows, evaluator, broker, persist):
    """Ports make fake-provider tests exercise the exact production update loop."""
    p = policy()
    if cell['arm'] not in ARMS or cell['seed'] not in p['seeds']:
        raise SearchContractError('STUDY_CELL_INVALID')
    rounds = p['rounds']
    if broker.phase == 'canary':
        c = canary_contract()
        if cell['seed'] != c['seed'] or len(rows) != c['optimize_count']:
            raise SearchContractError('STUDY_CANARY_CELL_INVALID')
        rounds = c['rounds']
    elif broker.phase != 'train':
        raise SearchContractError('STUDY_SEARCH_PHASE_FORBIDDEN')
    prompt = SystemPrompt.from_dict(cell['initial_prompt'])
    start_usage = broker.usage_view()
    history = [prompt.to_dict()]
    predictions = evaluator(prompt, rows, 'optimize', cell['seed'], 0)
    initial_usage = broker.usage_view()
    metrics = []
    persist('checkpoint_0', dict(prompt=prompt.to_dict(), observations=predictions))
    for ordinal in range(1, rounds + 1):
        broker.round_index = ordinal
        before_usage = broker.usage_view()
        old = prompt
        wrong = tuple(r['stable_example_id'] for r in rows if not predictions[r['stable_example_id']]['correct'])
        selected = wrong if cell['arm'] == 'full_errors' else batch_ids(wrong, cell['seed'], ordinal, p['error_batch_size'])
        selected_rows = tuple(r for r in rows if r['stable_example_id'] in set(selected))
        structure = (structured_feedback(prompt, selected_rows, predictions, broker)
                     if cell['arm'] == 'gradient_pattern' else None)
        feedback = dict(errors=[raw_error(r, predictions[r['stable_example_id']]) for r in selected_rows],
                        structured_feedback=structure)
        candidate_audit = []
        if wrong:
            prompt, candidate_audit = generate_candidate(prompt, feedback, broker, ordinal)
        next_predictions = evaluator(prompt, rows, 'optimize', cell['seed'], ordinal)
        ids = [r['stable_example_id'] for r in rows]
        previous_correct = {i for i in ids if predictions[i]['correct']}
        current_correct = {i for i in ids if next_predictions[i]['correct']}
        after_usage = broker.usage_view()
        metric = dict(round=ordinal, parent_prompt=old.to_dict(), new_prompt=prompt.to_dict(),
            lineage=block_lineage(old, prompt), accuracy_before=len(previous_correct) / len(ids),
            accuracy_after=len(current_correct) / len(ids),
            repaired_ids=sorted(current_correct - previous_correct),
            newly_wrong_ids=sorted(previous_correct - current_correct),
            correct_ids=sorted(current_correct), error_count=len(wrong),
            correctness_transitions=coverage_effect(predictions, next_predictions, scope='independent_complete_optimize'),
            optimizer_example_count=len(selected), optimizer_example_ids=list(selected),
            optimizer_feedback_bytes=len(json.dumps(feedback, ensure_ascii=True).encode()),
            candidate_draws=candidate_audit,
            update_status='REPLACED' if prompt != old else 'NO_ERRORS_CARRY' if not wrong
                else 'UNCHANGED_LEGAL_CANDIDATE' if candidate_audit and not candidate_audit[-1]['errors']
                else 'INVALID_CANDIDATE_CARRY',
            structured_feedback=structure,
            cost={k: after_usage[k] - before_usage[k] for k in after_usage})
        metrics.append(metric)
        history.append(prompt.to_dict())
        persist(f'checkpoint_{ordinal}', dict(prompt=prompt.to_dict(), observations=next_predictions,
            metric=metric, optimizer_feedback=feedback))
        predictions = next_predictions
    result = dict(cell_id=cell['cell_id'], arm=cell['arm'], seed=cell['seed'],
        prompt_history=history, metrics=metrics, final_prompt=prompt.to_dict(),
        final_accuracy=sum(x['correct'] for x in predictions.values()) / len(rows),
        initial_cost={k: initial_usage[k] - start_usage[k] for k in start_usage},
        total_cost={k: broker.usage_view()[k] - start_usage[k] for k in start_usage},
        rounds_completed=len(metrics), final_selection=p['final_selection'])
    persist('result', result)
    return result
