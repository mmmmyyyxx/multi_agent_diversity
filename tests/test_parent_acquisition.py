import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from infrastructure.common_solver_contract_v1.contract import request_identity
from infrastructure.common_solver_contract_v1.evaluator import TransportResponse
from multi_dataset_diverse_rl.parent_acquisition import build_catalog, select_parents, text_hash

spec = importlib.util.spec_from_file_location('parent_runner', Path(__file__).resolve().parents[1] / 'scripts/run_local_gepa_parent_acquisition.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def inputs():
    return {'parent_prompt': 'Resolve pronouns by checking grammatical agreement and contextual evidence.',
            'examples': [{'example_id': text_hash(q), 'question': q, 'gold': 'A'}
                         for i in range(100) for q in [f'Example {i}: Who acted?\nOptions:\n(A) Alice\n(B) Bob\n(C) Unknown']]}


def observations(data, wrong=50):
    return [{'example_id': row['example_id'], 'request_identity': request_identity(decision_procedure=data['parent_prompt'], question=row['question']),
             'raw_output': out, 'response_sha256': text_hash(out)}
            for i, row in enumerate(data['examples']) for out in ['FINAL_ANSWER: ' + ('B' if i < wrong else 'A')]]


@pytest.mark.skip(reason="superseded routed V1 parent-acquisition design; not a production Layer-2 contract")
def test_complete_tasks_and_deterministic_selection():
    data = inputs()
    result = build_catalog(data, observations(data), seed=78, local_metric_budget=205)
    assert len(result['eligible_parent_catalog']) == 5
    assert len(result['selected_parent_ids']) == 4
    assert select_parents(result['parent_catalog'][::-1]) == result['selected_parent_ids']
    assert all(all(row['checks'].values()) for row in result['integrity_audit'])
    assert len({r['task_payload_sha256'] for r in result['eligible_parent_catalog']}) == 5


@pytest.mark.parametrize('wrong', [0, 100, 3])
def test_insufficient_is_hold(wrong):
    data = inputs()
    result = build_catalog(data, observations(data, wrong), seed=78, local_metric_budget=205)
    assert result['status'] == 'PARENT_CATALOG_INSUFFICIENT'
    assert result['selected_parent_ids'] == []


def test_tampering_fails():
    data = inputs()
    obs = observations(data)
    obs[0]['raw_output'] = 'FINAL_ANSWER: C'
    with pytest.raises(ValueError, match='response_identity'):
        build_catalog(data, obs, seed=78, local_metric_budget=205)


def test_acquisition_stops_at_100_and_cannot_resume(tmp_path):
    calls = []
    async def transport(request):
        calls.append(request)
        return TransportResponse('FINAL_ANSWER: A', 20, 5, 'stop')
    output = tmp_path / 'run'
    asyncio.run(runner.acquire(inputs(), output, transport))
    assert len(calls) == 100
    assert runner.read(output / 'execution.json')['successes'] == 100
    assert len(list((output / 'observations_private').glob('*.json'))) == 100
    with pytest.raises(FileExistsError):
        asyncio.run(runner.acquire(inputs(), output, transport))
    assert len(calls) == 100


def test_truncation_aborts_without_retry(tmp_path):
    calls = []
    async def transport(request):
        calls.append(request)
        return TransportResponse('partial', 20, 1800, 'length')
    output = tmp_path / 'run'
    with pytest.raises(ValueError, match='incomplete'):
        asyncio.run(runner.acquire(inputs(), output, transport))
    assert len(calls) == 1
    assert runner.read(output / 'execution.json')['status'] == 'EXECUTION_ABORTED'
    assert runner.read(output / 'execution.json')['successes'] == 1


def test_provider_failure_is_preserved(tmp_path):
    async def transport(request):
        raise ValueError('synthetic provider failure')
    output = tmp_path / 'run'
    with pytest.raises(ValueError):
        asyncio.run(runner.acquire(inputs(), output, transport))
    assert runner.read(output / 'execution.json')['attempts'] == 1
    assert len((output / 'provider_ledger.jsonl').read_text().splitlines()) == 2


def test_retry_cap_is_four_not_sdk_multiplied(tmp_path, monkeypatch):
    import infrastructure.common_solver_contract_v1.evaluator as evaluator
    async def no_wait(seconds):
        pass
    monkeypatch.setattr(evaluator.asyncio, 'sleep', no_wait)
    calls = []
    async def transport(request):
        calls.append(request)
        raise TimeoutError('synthetic')
    output = tmp_path / 'run'
    with pytest.raises(TimeoutError):
        asyncio.run(runner.acquire(inputs(), output, transport))
    assert len(calls) == 4
    assert runner.read(output / 'execution.json')['successes'] == 0


@pytest.mark.skip(reason="superseded routed V1 parent-acquisition design; archived experiment only")
def test_audit_complete_outputs_without_any_api(tmp_path, monkeypatch):
    data = inputs()
    counter = 0
    async def transport(request):
        nonlocal counter
        counter += 1
        return TransportResponse('FINAL_ANSWER: ' + ('B' if counter <= 50 else 'A'), 20, 5, 'stop')
    output = tmp_path / 'run'
    asyncio.run(runner.acquire(data, output, transport))
    monkeypatch.setattr(runner, 'verify', lambda bundle: (data, {'commit': 'a' * 40}))
    public = tmp_path / 'public'
    runner.audit(tmp_path, output, public)
    assert counter == 100
    assert len(runner.read(public / 'selected_parent_ids.json')) == 4
    assert runner.read(public / 'phase_b_preregistration_inputs.json')['api_authorized'] is False
    assert runner.read(public / 'acquisition_cost.json')['acceptance_denominator_contribution'] == 0
    from multi_dataset_diverse_rl.governance.artifacts import scan_sanitized_artifacts
    assert scan_sanitized_artifacts(public) == []
