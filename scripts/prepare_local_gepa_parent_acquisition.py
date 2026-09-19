"""Zero-API preparation of the explicitly amended Phase-A acquisition."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from multi_dataset_diverse_rl.parent_acquisition import digest, text_hash
from multi_dataset_diverse_rl.local_optimizers.gepa_adapter import validate_complete_compact_prompt
from multi_dataset_diverse_rl.governance.manifest import preregistration_hash, validate_manifest
from multi_dataset_diverse_rl.versions import METHOD_VERSION

IDENTITY = 'local_gepa_parent_acquisition_v1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimize-csv', type=Path, required=True)
    parser.add_argument('--source-candidates', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    csv_sha = hashlib.sha256(args.optimize_csv.read_bytes()).hexdigest()
    if csv_sha != 'd308dcf239d42e949775fff2b8fe7540add825e20c767dee789955e7243b2b45':
        raise ValueError('frozen_Optimize100_csv_mismatch')
    prompt = json.loads(args.source_candidates.read_text(encoding='utf-8'))[0]['decision_procedure']
    if text_hash(prompt) != '549bc93c03f703faf5aa1bd56b557135fb6e65d0cf6c055b8fad6a15e7c87a63':
        raise ValueError('frozen_shared_P0_mismatch')
    validate_complete_compact_prompt(prompt, parent_prompt=prompt, examples=(), max_chars=3000)
    with args.optimize_csv.open(encoding='utf-8-sig', newline='') as handle:
        examples = [{'example_id': text_hash(row['question']), 'question': row['question'], 'gold': row['answer'].strip().strip('()')}
                    for row in csv.DictReader(handle)]
    if len(examples) != 100 or len({r['example_id'] for r in examples}) != 100 or any(r['gold'] not in ('A', 'B', 'C') for r in examples):
        raise ValueError('invalid_Optimize100_input')
    inputs = {'parent_prompt': prompt, 'examples': examples, 'source_csv_sha256': csv_sha,
              'source_split': 'anti_overfitting_split_v1_fold_a_plus_b'}
    args.bundle.mkdir(parents=True, exist_ok=False)
    (args.bundle / 'inputs_private.json').write_text(json.dumps(inputs, ensure_ascii=False, sort_keys=True, indent=2), encoding='utf-8')
    protocol = ROOT / 'experiments' / IDENTITY / 'PROTOCOL.md'
    manifest = {
        'schema_version': 'experiment_manifest_v1', 'experiment_id': IDENTITY,
        'title': 'Optimize-only parent acquisition, Phase A', 'status': 'IMPLEMENTED', 'legacy_index': False,
        'lifecycle_history': [{'status': status, 'timestamp': datetime.now(timezone.utc).isoformat()} for status in ['DRAFT', 'PREREGISTERED', 'IMPLEMENTED']],
        'lineage': {'parents': ['local_gepa_acceptance_rate_pilot_v1'], 'derives_from': 'local_gepa_acceptance_rate_pilot_v1'},
        'scientific_question': 'Can one authorized baseline initialization produce four complete current-contract parent tasks?',
        'hypotheses': ['Eligibility may remain insufficient; no claim of optimization efficacy is tested.'],
        'method_identity': METHOD_VERSION, 'runtime_version': IDENTITY,
        'data': {'task': 'BBH disambiguation_qa', 'formal': False,
                 'split_ids': {'optimize100': inputs['source_split']},
                 'split_hashes': {'optimize100_csv_sha256': csv_sha, 'ordered_question_ids_sha256': digest([r['example_id'] for r in examples])},
                 'validation_policy': 'prohibited; Validation50 calls=0', 'test_policy': 'prohibited; Test50 calls=0'},
        'model': {'solver': 'qwen3-8b', 'optimizer_roles': {}, 'thinking': False, 'temperatures': {'solver': 0.0}, 'max_tokens': {'solver': 1800}},
        'seeds': [78],
        'design': {'changed': ['independent Optimize-only baseline acquisition amendment'],
                   'unchanged': ['current task definition', 'production responsibility and routing', 'strict Solver contract'],
                   'forbidden_changes': ['GEPA', 'Reflection', 'team evaluation', 'writeback', 'persistent realizability', 'heldout access'],
                   'parent_prompt_sha256': text_hash(prompt), 'selection_rule': 'exact_max_coverage_source_target_lane_then_min_subset_sha256_v1',
                   'target_parent_count': 4, 'local_task_metric_budget_identity_only': 205,
                   'source_state_count': 1, 'baseline_realization_shared_by_five_identical_members': True,
                   'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
                   'private_input_sha256': hashlib.sha256((args.bundle / 'inputs_private.json').read_bytes()).hexdigest()},
        'api_authorization': {'authorized': True, 'allowed_roles': ['Solver'], 'allowed_phases': ['PHASE_A_BASELINE'], 'phase_b_authorized': False, 'authorization_scope': 'user explicit Optimize100 baseline only, max 100 successful Solver requests'},
        'budget': {'type': 'hard_successful_request_and_transport_attempt_cap', 'frozen_before_run': True,
                   'limit': {'successful_solver_requests': 100, 'transport_attempts': 400, 'per_request_attempts': 4, 'Reflection': 0, 'Validation50': 0, 'Test50': 0, 'GEPA_proposals': 0}},
        'selection': {'primary_metric': 'complete_eligible_parent_count', 'frozen_rule': 'maximum source-state/target/lane coverage then stable subset hash; insufficient below four', 'validation_used_for_selection': False, 'test_used_for_selection': False},
        'artifacts': {'preregistration': {'path': protocol.relative_to(ROOT).as_posix()}, 'report': 'reports/' + IDENTITY, 'provenance': 'reports/' + IDENTITY + '/provenance.json'},
        'git': {'design_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'implementation_commit': None, 'result_commit': None},
        'result': {'classifier': 'NOT_YET_RUN', 'conclusion': 'Phase B not authorized', 'evidence_type': 'not_yet_available'}}
    manifest['artifacts']['preregistration']['sha256'] = preregistration_hash(manifest)
    errors = validate_manifest(manifest, json.loads((ROOT / 'infrastructure/experiment_manifest.schema.json').read_text()))
    if errors:
        raise ValueError(errors)
    path = ROOT / 'experiments/manifests' / (IDENTITY + '.yaml')
    with path.open('x', encoding='utf-8') as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    print(json.dumps({'status': 'PREPARED_ZERO_API', 'input_count': len(examples), 'manifest_valid': True}))


if __name__ == '__main__':
    main()
