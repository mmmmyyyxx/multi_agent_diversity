"""Offline analysis of sealed private search, Validation and Test evidence."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from multi_dataset_diverse_rl.research.single_prompt_governance import private_path, verify_search, audit_complete_phase
    from multi_dataset_diverse_rl.research.single_prompt_analysis import analyze_final
    from multi_dataset_diverse_rl.benchmarks.math import MATHBenchmarkAdapter
    from multi_dataset_diverse_rl.persistence.durable_io import read_json, atomic_write_json
    parser = argparse.ArgumentParser()
    parser.add_argument('--search-root', type=Path, required=True)
    parser.add_argument('--validation-root', type=Path, required=True)
    parser.add_argument('--test-root', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    destination = private_path(ROOT, args.destination)
    if destination.exists():
        raise SystemExit('Fresh private analysis destination required')
    search = private_path(ROOT, args.search_root)
    validation = private_path(ROOT, args.validation_root)
    test = private_path(ROOT, args.test_root)
    audits = [audit_complete_phase(path, phase) for path, phase in ((search, 'train'), (validation, 'validation'), (test, 'test'))]
    results, terminal_hash = verify_search(search)
    if len({a['source_sha'] for a in audits}) != 1:
        raise SystemExit('Source mismatch between study phases')
    for path in (validation, test):
        predecessor = read_json(path / 'scope.json')['search_predecessor']
        if predecessor['terminal_sha256'] != terminal_hash or private_path(ROOT, predecessor['path']) != search:
            raise SystemExit('Held-out search predecessor mismatch')
    predictions = {r['cell_id']: read_json(test / r['cell_id'] / 'checkpoint_20.json')['observations'] for r in results}
    for r in results:
        stored = read_json(test / r['cell_id'] / 'checkpoint_20.json')
        if stored['prompt'] != r['final_prompt']:
            raise SystemExit('Test did not use the last-round Prompt')
    analysis = analyze_final(results, predictions, MATHBenchmarkAdapter().equivalent)
    destination.mkdir(parents=True)
    atomic_write_json(destination / 'phase_integrity_audit.json', audits)
    atomic_write_json(destination / 'diversity_metrics.json', analysis)
    atomic_write_json(destination / 'prompt_evolution_private.json', results)
    atomic_write_json(destination / 'validation_trajectories.json', read_json(validation / 'results.json'))
    print(json.dumps(dict(status='SEALED_EVIDENCE_ANALYZED_OWNER_INTERPRETATION_PENDING', final_pairs=36,
        destination=destination.relative_to(ROOT).as_posix(), model_calls=0)))


if __name__ == '__main__':
    main()
