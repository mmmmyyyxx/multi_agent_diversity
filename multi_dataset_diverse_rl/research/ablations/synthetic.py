"""Complete guarded smoke with fake responses and synthetic data only."""
from copy import deepcopy
import json
from pathlib import Path
import socket

from ...benchmarks.data_freeze import digest
from ...governance.token_accounting import TokenLedger, POLICY_GATED_32_UNCAPPED
from ...persistence.provider_receipts import ProviderResponseReceipts
from ...persistence.durable_io import atomic_write_json
from ...search.schemas import SearchContractError
from ...search.system_prompt import SystemPrompt
from .offline_runtime import OfflineAblationBroker
from .engine import run_cell
from .analysis import summarize


class SyntheticTransport:
    def __init__(self):
        self.calls = []

    def for_role(self, role, request):
        self.calls.append((role, deepcopy(request)))
        if role == 'solver':
            number = int(request['messages'][1]['content'].split(':')[-1])
            text = f'Final answer: {number + 1}'
        elif role == 'reflection':
            text = json.dumps(dict(block='strategy', content='Check calculations carefully and verify each transformation.'))
        elif role == 'pattern_gradient':
            text = json.dumps(dict(disposition='ACTIONABLE', observed_failure='The final value disagrees with the reference.',
                diagnosis='A calculation step may require checking.', reusable_correction='Check arithmetic consistency before finalizing.',
                suggested_block='strategy', expected_effect='Reduce calculation errors.'))
        elif role == 'pattern_cluster':
            packet = json.loads(request['messages'][1]['content'])
            text = json.dumps(dict(patterns=[dict(generalized_gradient='Check arithmetic consistency before finalizing.',
                support_ids=[g['example_id'] for g in packet['gradients']])], unassigned_ids=[]))
        else:
            raise AssertionError(role)
        return dict(text=text, finish_reason='stop', input_tokens=80, output_tokens=40,
            provider_thinking_indicators=[], provider_reasoning_character_count=0,
            provider_reasoning_content_present=False,
            provider_usage_details=dict(completion_tokens_details=dict(reasoning_tokens=0)))


def synthetic_rows(split, count):
    return tuple(dict(stable_example_id=f'{split}_fixture_{i}', reference_final_answer=str(i),
        content=dict(problem=f'Synthetic fixture value: {i}', solution=f'The reference value is {i}.'))
        for i in range(count))


def smoke(plan, destination):
    import sitecustomize
    if socket.socket.connect.__module__ != 'sitecustomize' or sitecustomize.network_attempt_count():
        raise SearchContractError('ABLATION_OFFLINE_GUARD_REQUIRED')
    destination = Path(destination)
    if destination.exists():
        raise SearchContractError('ABLATION_FRESH_SMOKE_DESTINATION_REQUIRED')
    destination.mkdir(parents=True)
    scope_id = digest(['SYNTHETIC_ONLY', plan['plan_identity']])
    ledger = TokenLedger(destination / 'ledger', task_sha256=scope_id, policy=POLICY_GATED_32_UNCAPPED)
    broker = OfflineAblationBroker(plan=plan, phase='train', scope=dict(scope_identity=scope_id,
        token_ceiling=None, call_ceiling=plan['training_semantic_call_ceiling'], network_recovery=True),
        transport=SyntheticTransport(), ledger=ledger,
        receipts=ProviderResponseReceipts(destination / 'receipts', attempt_id='SYNTHETIC', startup_identity_sha256=scope_id),
        dispatch_guard=lambda _: None)
    initial = SystemPrompt('You are a helpful assistant.', 'Answer the question.',
        'On the last line, write "Final answer:" followed by your mathematical answer.')
    results = []
    try:
        for cell in plan['cells']:
            folder = destination / cell['cell_id']
            folder.mkdir()
            results.append(run_cell(plan, cell, initial.to_dict(), synthetic_rows('optimize', 5),
                synthetic_rows('validation', 3), broker,
                lambda name, value: atomic_write_json(folder / (name + '.json'), value)))
        # Synthetic Test is only introduced after every cell completes. A real
        # final phase additionally needs terminal seals and its own authorization.
        broker.phase = 'test'
        predictions = {}
        for result in results:
            cell = result['cell']
            broker.cell_id = cell['cell_id']
            broker.seed, broker.repeat_id = cell['seed'], cell['repeat_id']
            broker.attempt_id = 'SYNTHETIC_TEST_' + cell['cell_id']
            predictions[cell['cell_id']] = broker.evaluate(SystemPrompt.from_dict(result['final_prompt']),
                synthetic_rows('test', 4), 'test', cell['seed'], plan['config']['rounds'])
        summary = summarize(plan, results, predictions)
        summary.update(status='SYNTHETIC_SMOKE_COMPLETE_NOT_EFFICACY', real_api_calls=0,
            actual_validation_content_reads=0, actual_test_content_reads=0,
            network_attempts=sitecustomize.network_attempt_count(), unresolved_reservations=len(ledger.inflight))
        atomic_write_json(destination / 'summary.json', summary)
        return summary
    finally:
        ledger.close()
