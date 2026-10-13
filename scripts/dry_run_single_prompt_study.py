"""Guard-required synthetic nine-cell pipeline, never a real experiment."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    if os.environ.get('FORMAL_ZERO_API_GUARD_REQUIRED') != '1':
        raise SystemExit('Use tests/formal_zero_api_runner.py --offline-command')
    import sitecustomize
    import socket
    if socket.socket.connect.__module__ != 'sitecustomize' or sitecustomize.network_attempt_count():
        raise SystemExit('Pre-import network guard required')
    from multi_dataset_diverse_rl.research.single_prompt_governance import private_path
    from multi_dataset_diverse_rl.research.single_prompt_policy import ARMS, policy
    from multi_dataset_diverse_rl.research.single_prompt import run_cell
    from multi_dataset_diverse_rl.research.single_prompt_analysis import analyze_final
    from multi_dataset_diverse_rl.search.system_prompt import SystemPrompt
    from multi_dataset_diverse_rl.benchmarks.math import MATHBenchmarkAdapter
    from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json
    spec = importlib.util.spec_from_file_location('study_toy_fixture', ROOT / 'tests/current/test_single_prompt_study.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    destination = private_path(ROOT, args.destination)
    if destination.exists():
        raise SystemExit('Fresh synthetic destination required')
    destination.mkdir(parents=True)
    rows = fixture.toy_rows(5)
    broker = fixture.broker_for(destination)
    results = []
    final_predictions = {}
    validation_count = 0
    try:
        for arm in ARMS:
            for seed in policy()['seeds']:
                cell = fixture.cell(arm, seed)
                cell_root = destination / cell['cell_id']
                cell_root.mkdir()
                broker.cell_id, broker.seed, broker.attempt_id = cell['cell_id'], seed, 'SYNTHETIC_' + cell['cell_id']
                broker.round_index = 0
                result = run_cell(cell, rows, broker.evaluate, broker,
                    lambda name, value: atomic_write_json(cell_root / (name + '.json'), value))
                results.append(result)
        atomic_write_json(destination / 'search_results_SYNTHETIC.json', results)
        # Held-out fixtures are synthetic, and are introduced only after every
        # search is complete. No production held-out dataset content is opened.
        broker.phase = 'validation'
        for result in results:
            broker.cell_id, broker.seed = result['cell_id'], result['seed']
            broker.attempt_id = 'SYNTHETIC_VALIDATION_' + result['cell_id']
            values = []
            for ordinal, prompt in enumerate(result['prompt_history']):
                observations = broker.evaluate(SystemPrompt.from_dict(prompt), rows, 'validation', result['seed'], ordinal)
                values.append(dict(checkpoint=ordinal, observations=observations))
                validation_count += 1
            atomic_write_json(destination / result['cell_id'] / 'validation_SYNTHETIC.json', values)
        broker.phase = 'test'
        for result in results:
            broker.cell_id, broker.seed = result['cell_id'], result['seed']
            broker.attempt_id = 'SYNTHETIC_TEST_' + result['cell_id']
            observations = broker.evaluate(SystemPrompt.from_dict(result['final_prompt']), rows, 'test', result['seed'], policy()['rounds'])
            final_predictions[result['cell_id']] = observations
            atomic_write_json(destination / result['cell_id'] / 'test_SYNTHETIC.json', observations)
        analysis = analyze_final(results, final_predictions, MATHBenchmarkAdapter().equivalent)
        atomic_write_json(destination / 'diversity_SYNTHETIC.json', analysis)
        summary = dict(status='DRY_RUN_SYNTHETIC_COMPLETE_NOT_REAL_RESULTS', cells_completed=9,
            rounds_per_cell=20, prompt_checkpoints_per_cell=21,
            validation_checkpoints=validation_count, test_final_prompts=9,
            pairwise_pairs=len(analysis['all_36_pairs']), fake_provider_usage=broker.usage_view(),
            unresolved_reservations=len(broker.ledger.inflight), real_api_calls=0,
            actual_validation_content_reads=0, actual_test_content_reads=0,
            network_attempts=sitecustomize.network_attempt_count())
        atomic_write_json(destination / 'dry_run_summary.json', summary)
        print(json.dumps(summary, indent=2))
    finally:
        broker.ledger.close()


if __name__ == '__main__':
    main()
