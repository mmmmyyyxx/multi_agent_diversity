"""Module-independent evaluation, generation, strict admission and checkpoints."""
from copy import deepcopy
import json

from ...benchmarks.data_freeze import digest
from ...search.schemas import SearchContractError
from ...search.system_prompt import SystemPrompt, block_lineage
from ..single_prompt import batch_ids, raw_error
from .config import validate_plan, request_namespace
from .interfaces import OptimizeContext, OptimizeOutcome, PreparedFeedback, ModuleProvider
from .generation import generate_candidate


def _observations(rows, values):
    ids = [r['stable_example_id'] for r in rows]
    if (not ids or len(set(ids)) != len(ids) or set(values) != set(ids)
            or any(type(v.get('correct')) is not bool or type(v.get('valid')) is not bool
                   or (v['correct'] and not v['valid']) for v in values.values())):
        raise SearchContractError('ABLATION_EVALUATION_MEMBERSHIP_OR_VALIDITY')
    return deepcopy(values)


def _state(value, binding):
    try:
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise SearchContractError('ABLATION_MODULE_STATE_NOT_JSON') from exc
    if len(encoded.encode()) > binding['maximum_state_bytes']:
        raise SearchContractError('ABLATION_MODULE_STATE_BUDGET')
    return json.loads(encoded)


def _checkpoint(persist, name, value):
    persist(name, dict(value, integrity_sha256=digest(value)))


def run_cell(plan, cell, initial_prompt, optimize_rows, validation_rows, broker, persist,
             *, registry=None, resume=None):
    from .proposal_config import SCHEMA as proposal_schema
    if plan['config']['schema_version']==proposal_schema:
        from .pattern_proposals import run_cell as proposal_run
        return proposal_run(plan,cell,initial_prompt,optimize_rows,validation_rows,broker,persist,
            registry=registry,resume=resume)
    from .config import MULTI_SCHEMA
    if plan['config']['schema_version']==MULTI_SCHEMA:
        from .multi_parent import run_cell as population_run
        return population_run(plan,cell,initial_prompt,optimize_rows,validation_rows,broker,persist,
            registry=registry,resume=resume)
    from .modules import default_registry
    from .config import SCHEMA
    registry = registry or default_registry(include_legacy=plan['config']['schema_version']==SCHEMA)
    validate_plan(plan, registry)
    if cell not in plan['cells']:
        raise SearchContractError('ABLATION_CELL_NOT_BOUND')
    optimize_rows, validation_rows = tuple(optimize_rows), tuple(validation_rows)
    oi = [r['stable_example_id'] for r in optimize_rows]
    vi = [r['stable_example_id'] for r in validation_rows]
    if not oi or not vi or len(set(oi)) != len(oi) or len(set(vi)) != len(vi) or set(oi) & set(vi):
        raise SearchContractError('ABLATION_SPLIT_MEMBERSHIP')
    p = plan['config']['runtime']
    if broker.phase != 'train':
        raise SearchContractError('ABLATION_TRAIN_PHASE_REQUIRED')
    broker.bind_cell(cell, plan)
    initial = SystemPrompt.from_dict(initial_prompt)
    binding = plan['module_bindings'][cell['arm_id']]
    module = registry.create(cell['module'])  # A fresh instance and state for each cell.
    state = _state(module.initial_state(), binding)
    scope = dict(plan_identity=plan['plan_identity'], cell=cell,
        initial_prompt_sha256=initial.prompt_hash, optimize_membership_sha256=digest(oi),
        validation_membership_sha256=digest(vi), module_binding=binding,
        repeat_namespace=request_namespace(plan['plan_identity'], cell['seed'], cell['repeat_id']))
    scope_identity = digest(scope)
    begin = broker.usage_view()
    metrics = []
    history = [initial.to_dict()]
    prompt = initial
    if resume is None:
        optimize = _observations(optimize_rows, broker.evaluate(prompt, optimize_rows, 'optimize', cell['seed'], 0))
        validation = _observations(validation_rows, broker.evaluate(prompt, validation_rows, 'validation', cell['seed'], 0))
        _checkpoint(persist, 'checkpoint_0', dict(scope_identity=scope_identity, scope=scope, prompt=prompt.to_dict(),
            optimize=optimize, validation=validation, module_state=state))
    else:
        checkpoints = resume['checkpoints']
        if not checkpoints or checkpoints[0].get('scope') != scope or len(checkpoints) > p['rounds'] + 1:
            raise SearchContractError('ABLATION_RESUME_IDENTITY')
        if checkpoints[0]['prompt'] != initial_prompt:
            raise SearchContractError('ABLATION_RESUME_INITIAL_PROMPT')
        for n, cp in enumerate(checkpoints):
            if cp.get('integrity_sha256') != digest({k: v for k, v in cp.items() if k != 'integrity_sha256'}):
                raise SearchContractError('ABLATION_RESUME_CHECKPOINT_INTEGRITY')
            if cp.get('scope_identity') != scope_identity or (n and cp['metric']['round'] != n):
                raise SearchContractError('ABLATION_RESUME_NONCONTIGUOUS')
            _observations(optimize_rows, cp['optimize'])
            _observations(validation_rows, cp['validation'])
            _state(cp['module_state'], binding)
            if n:
                previous = checkpoints[n - 1]
                metric = cp['metric']
                accepted = metric['decision'] == 'ACCEPTED_STRICT_VALIDATION_GAIN'
                before_score = sum(v['correct'] for v in previous['validation'].values())
                after_score = sum(v['correct'] for v in cp['validation'].values())
                if (metric['accepted'] is not accepted or metric['validation_before'] != before_score
                        or (accepted and (after_score <= before_score or metric['validation_candidate'] != after_score
                                          or cp.get('candidate_prompt') != cp['prompt']))
                        or (not accepted and any(cp[k] != previous[k] for k in ('prompt', 'optimize', 'validation')))):
                    raise SearchContractError('ABLATION_RESUME_GATE_MISMATCH')
        last = checkpoints[-1]
        prompt = SystemPrompt.from_dict(last['prompt'])
        optimize, validation = deepcopy(last['optimize']), deepcopy(last['validation'])
        state = _state(last['module_state'], binding)
        metrics = [deepcopy(cp['metric']) for cp in checkpoints[1:]]
        history = [deepcopy(cp['prompt']) for cp in checkpoints]
    score = lambda values: sum(v['correct'] for v in values.values())
    for ordinal in range(len(metrics) + 1, p['rounds'] + 1):
        broker.round_index = ordinal
        before = broker.usage_view()
        wrong = tuple(i for i in oi if not optimize[i]['correct'])
        selected = wrong if cell['error_source'] == 'all_wrong' else batch_ids(wrong, cell['seed'], ordinal, p['error_batch_size'])
        selected_rows = tuple(r for r in optimize_rows if r['stable_example_id'] in set(selected))
        context = OptimizeContext(cell['cell_id'], cell['seed'], cell['repeat_id'], ordinal,
            prompt, deepcopy(selected_rows), {i: deepcopy(optimize[i]) for i in selected})
        provider = ModuleProvider(broker, binding['semantic_calls_per_opportunity'])
        prepared = module.prepare(context, deepcopy(state), provider)
        if getattr(broker, 'scope', {}).get('generation_recovery'):
            prepared = PreparedFeedback(prepared.errors, prepared.structured_feedback, prepared.module_context,
                dict(prepared.audit, diagnostic_draw_counts=provider.draw_counts()), prepared.state)
        if not isinstance(prepared, PreparedFeedback):
            raise SearchContractError('ABLATION_MODULE_FEEDBACK_TYPE')
        originals = {r['stable_example_id']: raw_error(r, optimize[r['stable_example_id']]) for r in selected_rows}
        supplied = [e['example_id'] for e in prepared.errors]
        if (len(set(supplied)) != len(supplied)
                or any(e != originals.get(e['example_id']) for e in prepared.errors)):
            raise SearchContractError('ABLATION_MODULE_ERROR_PROVENANCE')
        state = _state(prepared.state, binding)
        persist(f'round_{ordinal}_module', dict(module_binding=binding, audit=prepared.audit, module_state=state))
        persist(f'round_{ordinal}_opportunity', dict(parent_prompt=prompt.to_dict(),
            errors=prepared.errors, structured_feedback=prepared.structured_feedback,
            module_context=prepared.module_context, retained_validation_correct=score(validation)))
        candidate, draws = generate_candidate(prompt, prepared, broker, ordinal, p['optimizer_structural_draws'], persist)
        candidate_optimize = candidate_validation = None
        candidate_score = None
        accepted = False
        decision = 'ABANDONED_GENERATION_RETRIES_EXHAUSTED'
        no_op=candidate==prompt
        if no_op:decision='SKIPPED_NO_OP'
        if candidate is not None and not no_op:
            candidate_optimize = _observations(optimize_rows,
                broker.evaluate(candidate, optimize_rows, 'optimize', cell['seed'], ordinal))
            candidate_validation = _observations(validation_rows,
                broker.evaluate(candidate, validation_rows, 'validation', cell['seed'], ordinal))
            candidate_score = score(candidate_validation)
            accepted = candidate_score > score(validation)
            decision = 'ACCEPTED_STRICT_VALIDATION_GAIN' if accepted else 'REJECTED_NO_STRICT_VALIDATION_GAIN'
        retained = candidate if accepted else prompt
        event = OptimizeOutcome(ordinal, prompt, candidate, retained, deepcopy(optimize),
            deepcopy(candidate_optimize), decision)
        next_state = _state(module.observe(event, deepcopy(state)), binding)
        metric = dict(round=ordinal, accepted=accepted, decision=decision, candidate_evaluated=candidate is not None and not no_op,
            generation_draws=draws, optimizer_example_count=len(prepared.errors),no_op=no_op,
            validation_before=score(validation), validation_candidate=candidate_score,
            optimize_before=score(optimize), optimize_candidate=score(candidate_optimize) if candidate_optimize is not None else None,
            lineage=block_lineage(prompt, candidate) if candidate is not None else None,
            cost={k: broker.usage_view()[k] - v for k, v in before.items()})
        parent = prompt
        if accepted:
            prompt, optimize, validation = candidate, candidate_optimize, candidate_validation
        state = next_state
        _checkpoint(persist, f'checkpoint_{ordinal}', dict(scope_identity=scope_identity, prompt=prompt.to_dict(),
            parent_prompt=parent.to_dict(), optimize=optimize, validation=validation, module_state=state,
            candidate_prompt=candidate.to_dict() if candidate is not None else None,
            candidate_optimize=candidate_optimize, candidate_validation=candidate_validation, metric=metric))
        metrics.append(metric)
        history.append(prompt.to_dict())
    inherited = {} if resume is None else resume['historical_usage']
    result = dict(cell=cell, scope_identity=scope_identity, rounds_completed=len(metrics),
        module_binding=binding, final_prompt=prompt.to_dict(), prompt_history=history, metrics=metrics,
        final_optimize_correct=score(optimize), optimize_count=len(optimize_rows),
        final_validation_correct=score(validation), validation_count=len(validation_rows),
        module_state=state, usage={k: broker.usage_view()[k] - v + inherited.get(k, 0) for k, v in begin.items()})
    persist('result', result)
    return result
