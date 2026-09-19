"""Frozen Phase A only. No GEPA execution path exists in this runner."""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infrastructure.common_solver_contract_v1.contract import CONTRACT_SPEC, canonical_json_bytes, contract_identity, request_identity
from infrastructure.common_solver_contract_v1.evaluator import CommonSolverEvaluator, TransportResponse
from multi_dataset_diverse_rl.parent_acquisition import build_catalog, digest, text_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(bundle):
    freeze = read(bundle / 'freeze.json')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != freeze['commit']:
        raise ValueError('frozen_commit_mismatch')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
        raise ValueError('tracked_worktree_dirty')
    for relative, expected in freeze['source_hashes'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('source_hash_mismatch')
    if sha(bundle / 'inputs_private.json') != freeze['inputs_sha256'] or freeze['contract_identity'] != contract_identity():
        raise ValueError('input_or_contract_hash_mismatch')
    inputs = read(bundle / 'inputs_private.json')
    if len(inputs['examples']) != 100 or len({r['example_id'] for r in inputs['examples']}) != 100:
        raise ValueError('Optimize100_required')
    if any(text_hash(r['question']) != r['example_id'] for r in inputs['examples']):
        raise ValueError('input_identity_mismatch')
    if freeze['authorization'] != 'PHASE_A_OPTIMIZE100_BASELINE_ONLY_MAX_100_SUCCESSFUL' or not freeze['READY_TO_RUN']:
        raise ValueError('phase_a_not_authorized')
    return inputs, freeze


async def acquire(inputs, output, transport, retryable=None):
    output.mkdir(parents=True, exist_ok=False)
    allowed = {request_identity(decision_procedure=inputs['parent_prompt'], question=r['question']) for r in inputs['examples']}
    if len(allowed) != 100:
        raise ValueError('exactly_100_distinct_requests_required')
    counts = {'attempts': 0, 'successes': 0}
    responses = {}
    def event(row):
        with (output / 'provider_ledger.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
    async def guarded(request):
        identity = hashlib.sha256(canonical_json_bytes(request)).hexdigest()
        if identity not in allowed or identity in responses or counts['successes'] >= 100 or counts['attempts'] >= 400:
            raise ValueError('acquisition_request_or_budget_violation')
        counts['attempts'] += 1
        attempt = counts['attempts']
        event({'event': 'start', 'attempt': attempt, 'request_identity': identity, 'stage': 'baseline', 'split': 'Optimize100'})
        try:
            response = await transport(request)
        except Exception as exc:
            event({'event': 'failure', 'attempt': attempt, 'error_type': type(exc).__name__})
            raise
        counts['successes'] += 1
        event({'event': 'success', 'attempt': attempt, 'request_identity': identity,
               'prompt_tokens': response.prompt_tokens, 'completion_tokens': response.completion_tokens,
               'finish_reason': response.finish_reason})
        write(output / 'responses_private' / (identity + '.json'), response.__dict__)
        responses[identity] = response
        if response.finish_reason != 'stop' or response.prompt_tokens <= 0 or response.completion_tokens < 0:
            raise ValueError('provider_response_incomplete')
        return response
    evaluator = CommonSolverEvaluator(transport=guarded, retryable=retryable)
    try:
        for index, example in enumerate(inputs['examples']):
            result = await evaluator.evaluate(decision_procedure=inputs['parent_prompt'], question=example['question'])
            response = responses[result.request_identity]
            write(output / 'observations_private' / f'{index:03d}.json', {
                'example_id': example['example_id'], 'request_identity': result.request_identity,
                'raw_output': response.text, 'response_sha256': text_hash(response.text)})
            print(json.dumps({'baseline_completed': index + 1, 'provider_attempts': counts['attempts']}), flush=True)
    except BaseException as exc:
        write(output / 'execution.json', {'status': 'EXECUTION_ABORTED', 'error_type': type(exc).__name__, **counts})
        raise
    write(output / 'execution.json', {'status': 'EXECUTION_COMPLETE', **counts, 'accounting': evaluator.accounting()})


def audit(bundle, output, public):
    inputs, freeze = verify(bundle)
    execution = read(output / 'execution.json')
    if execution['status'] != 'EXECUTION_COMPLETE':
        raise ValueError('incomplete_baseline_no_parent_freeze')
    observations = [read(output / 'observations_private' / f'{i:03d}.json') for i in range(100)]
    ledger = [json.loads(line) for line in (output / 'provider_ledger.jsonl').read_text().splitlines()]
    successes = [r for r in ledger if r['event'] == 'success']
    starts = [r for r in ledger if r['event'] == 'start']
    failures = [r for r in ledger if r['event'] == 'failure']
    expected = {r['request_identity'] for r in observations}
    if len(successes) != 100 or {r['request_identity'] for r in successes} != expected or len(starts) != len(successes) + len(failures):
        raise ValueError('ledger_integrity_failure')
    if any(r['split'] != 'Optimize100' or r['stage'] != 'baseline' or r['request_identity'] not in expected for r in starts):
        raise ValueError('split_or_stage_violation')
    if [r['attempt'] for r in starts] != list(range(1, len(starts) + 1)) or len(starts) > 400:
        raise ValueError('attempt_identity_or_budget_mismatch')
    terminals = {r['attempt']: r for r in successes + failures}
    if len(terminals) != len(starts) or set(terminals) != {r['attempt'] for r in starts}:
        raise ValueError('attempt_terminal_pairing_mismatch')
    if any(sum(r['request_identity'] == identity for r in starts) > 4 for identity in expected):
        raise ValueError('per_request_retry_limit_exceeded')
    for start in starts:
        terminal = terminals[start['attempt']]
        if terminal['event'] == 'success' and terminal['request_identity'] != start['request_identity']:
            raise ValueError('attempt_request_pairing_mismatch')
    for row in observations:
        response = read(output / 'responses_private' / (row['request_identity'] + '.json'))
        if response['text'] != row['raw_output']:
            raise ValueError('response_observation_mismatch')
        entry = next(r for r in successes if r['request_identity'] == row['request_identity'])
        if any(response[key] != entry[key] for key in ('prompt_tokens', 'completion_tokens', 'finish_reason')):
            raise ValueError('response_ledger_usage_mismatch')
    account = execution['accounting']
    if (execution['successes'] != 100 or execution['attempts'] != len(starts)
        or account['successful_provider_calls'] != 100 or account['provider_attempts'] != len(starts)
        or account['failed_provider_attempts'] != len(failures) or account['cache_hits'] != 0
        or account['logical_calls'] != 100
        or account['prompt_tokens'] != sum(r['prompt_tokens'] for r in successes)
        or account['completion_tokens'] != sum(r['completion_tokens'] for r in successes)):
        raise ValueError('execution_accounting_mismatch')
    result = build_catalog(inputs, observations, seed=78, local_metric_budget=205)
    if result != build_catalog(inputs, observations, seed=78, local_metric_budget=205):
        raise ValueError('nondeterministic_reconstruction')
    write(output / 'parent_tasks_private.json', result.pop('tasks_private'))
    public.mkdir(parents=True, exist_ok=False)
    for key in ('parent_catalog', 'eligible_parent_catalog', 'selected_parent_ids'):
        write(public / (key + '.json'), result[key])
    write(public / 'parent_task_integrity_audit.json', {'status': result['status'], 'checks': result['integrity_audit'], 'deterministic_reconstruction': True, 'source_commit': freeze['commit']})
    write(public / 'split_isolation_audit.json', {'optimize_baseline_successes': 100, 'validation50_calls': 0, 'test50_calls': 0, 'reflection_calls': 0, 'gepa_proposals': 0, 'team_evaluation_calls': 0, 'writebacks': 0, 'persistent_realizability_updates': 0, 'request_allowlist_verified': True})
    cost = {'cost_category': 'Parent acquisition cost', 'provider_attempts': len(starts), 'successful_solver_requests': len(successes), 'failed_provider_attempts': len(failures), 'prompt_tokens': sum(r['prompt_tokens'] for r in successes), 'completion_tokens': sum(r['completion_tokens'] for r in successes), 'failed_attempt_token_cost': 'UNKNOWN' if failures else 0, 'acceptance_denominator_contribution': 0, 'layer1_optimization_cost': 0}
    cost['total_tokens'] = cost['prompt_tokens'] + cost['completion_tokens']
    write(public / 'acquisition_cost.json', cost)
    selected = [r for r in result['eligible_parent_catalog'] if r['parent_task_id'] in result['selected_parent_ids']]
    write(public / 'phase_b_preregistration_inputs.json', {'identity': 'local_gepa_acceptance_rate_pilot_phase_b_v2', 'status': 'DRAFT_REQUIRES_SEPARATE_AUTHORIZATION' if len(selected) == 4 else 'HOLD', 'api_authorized': False, 'planned_proposals': 32 if len(selected) == 4 else 0, 'selected_parents': selected, 'source_state_count': len({r['source_state_hash'] for r in selected}), 'selection_rule': 'exact lexicographic maximum source/target/lane coverage; minimum stable subset SHA256 tie', 'parent_acquisition_cost_separate': True})
    write(public / 'sha256_manifest.json', {p.name: sha(p) for p in sorted(public.glob('*.json'))})
    print(json.dumps({'status': result['status'], 'eligible': len(result['eligible_parent_catalog']), 'selected': len(selected), 'cost': cost}), flush=True)


async def real_run(inputs, output):
    from openai import AsyncOpenAI, APIConnectionError
    from multi_dataset_diverse_rl.provider_credentials import resolve_api_key, resolve_base_url
    _, key = resolve_api_key('DASHSCOPE_API_KEY')
    _, url = resolve_base_url('DASHSCOPE_BASE_URL')
    if not key or not url:
        raise ValueError('provider_credentials_unavailable')
    async with AsyncOpenAI(api_key=key, base_url=url, max_retries=0, timeout=CONTRACT_SPEC.timeout_seconds) as client:
        async def transport(request):
            response = await client.chat.completions.create(**request)
            if response.usage is None or len(response.choices) != 1:
                return TransportResponse('', -1, -1, 'missing_usage_or_choice')
            choice = response.choices[0]
            return TransportResponse(choice.message.content or '', response.usage.prompt_tokens, response.usage.completion_tokens, choice.finish_reason)
        def retryable(exc):
            return isinstance(exc, (TimeoutError, ConnectionError, APIConnectionError)) or getattr(exc, 'status_code', None) in CONTRACT_SPEC.retry_status_codes
        await acquire(inputs, output, transport, retryable)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight', 'execute', 'audit'])
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--public', type=Path)
    args = parser.parse_args()
    inputs, _ = verify(args.bundle)
    if args.mode == 'execute':
        if os.getenv('PARENT_ACQUISITION_API_AUTHORIZED') != '1':
            raise ValueError('explicit_phase_a_authorization_required')
        asyncio.run(real_run(inputs, args.output))
        verify(args.bundle)
    elif args.mode == 'audit':
        audit(args.bundle, args.output, args.public)
    else:
        print('PHASE_A_PREFLIGHT_PASS')


if __name__ == '__main__':
    main()
