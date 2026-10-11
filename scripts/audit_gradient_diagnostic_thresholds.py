"""Zero-API two-field sensitivity replay; never a production experiment runner."""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import sitecustomize


def main():
    assert os.environ.get('FORMAL_ZERO_API_GUARD_REQUIRED') == '1'
    assert sitecustomize.network_attempt_count() == 0
    root = Path(__file__).resolve().parents[1]
    parent_sha = '7982871850756457b092a5c83ae60347871cdc46'
    previous = root/'scripts/audit_openlux_pilot_diagnostics.py'
    blob = subprocess.check_output(['git', 'show', parent_sha+':scripts/audit_openlux_pilot_diagnostics.py'])
    assert previous.read_bytes().replace(b'\r\n', b'\n') == blob
    # Original audit checks all sealed inventory hashes and reproduces every
    # original logical disposition before any counterfactual is evaluated.
    original = runpy.run_path(str(previous))
    helper = root/'multi_dataset_diverse_rl/diagnostics/gradient_threshold_replay.py'
    spec = importlib.util.spec_from_file_location('threshold_diagnostic_only', helper)
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    frozen = original['frozen']
    native_source = (frozen/'multi_dataset_diverse_rl/search/textual_gradients.py').read_text(encoding='utf-8')
    from multi_dataset_diverse_rl.search import textual_gradients as native
    from multi_dataset_diverse_rl.search.optimization_evidence import POLICY, GRADIENT_INPUT
    from multi_dataset_diverse_rl.benchmarks.math_response_evidence import trajectory_policy
    from multi_dataset_diverse_rl.search.system_prompt import SystemPrompt
    from multi_dataset_diverse_rl.search.generation_failures import generation_failure
    from multi_dataset_diverse_rl.search.schemas import SearchContractError
    from multi_dataset_diverse_rl import current_contract as versions

    modules = {}
    for observed in diagnostic.LIMITS:
        for diagnosis in diagnostic.LIMITS:
            name = 'multi_dataset_diverse_rl.search._offline_threshold_%d_%d' % (observed, diagnosis)
            module = ModuleType(name)
            module.__package__ = 'multi_dataset_diverse_rl.search'
            # An observation-only generated module is explicitly not current runtime.
            module.__file__ = str(frozen/'counterfactual_threshold_module.py')
            sys.modules[name] = module
            exec(compile(diagnostic.threshold_tree(native_source, observed, diagnosis),
                '<offline-two-field-counterfactual>', 'exec'), module.__dict__)
            modules[observed, diagnosis] = module

    def replay(module, packet, response):
        ex = packet['example']
        evidence = SimpleNamespace(example_id=ex['example_id'], source_split='optimize', signals=dict(
            input_payload=ex['problem'], gold=ex['reference'], target_output=ex['prediction'],
            target_member_correct=False, target_member_valid=ex['valid'], solver_trajectory=ex['solver_trajectory'],
            reference_solution=ex['reference_solution']['text'], responsibility_labels=[],
            correctness_signal_identity=versions.TARGET_CORRECTNESS_SIGNAL_VERSION))
        provider = native.PerExampleGradientProvider.__new__(native.PerExampleGradientProvider)
        provider.optimization_evidence_policy = POLICY
        provider.solver_trajectory_policy = trajectory_policy()
        provider.input_schema = GRADIENT_INPUT
        provider.numeric_guard_writer = None
        calls = 0

        def decode(actual_packet):
            nonlocal calls
            assert actual_packet == packet
            calls += 1
            if response['finish_reason'] != 'stop':
                raise generation_failure('GENERATED_OUTPUT_TRUNCATED')
            try:
                return json.loads(response['text'])
            except (ValueError, TypeError):
                raise generation_failure('PATTERN_GRADIENT_EXTRACTION_INVALID') from None

        # Keep the original provider's input schema / trajectory / reference
        # provenance validation. Only the actual network _call becomes decoding.
        provider._call = decode
        extractor = module.GradientExtractor(provider)
        gradients = extractor.extract(SystemPrompt.from_dict(packet['current_system_prompt']), (evidence,))
        got = extractor.evidence_diagnostics[0]
        accepted = not got['exhausted']
        assert calls == (1 if accepted else 3)
        if gradients:
            assert got['disposition'] == 'ACTIONABLE'
            assert gradients[0]['gradient'] == json.loads(response['text'])['reusable_correction'].strip()
        return dict(accepted=accepted, disposition=got['disposition'],
            native_error=got['structural_errors'][0] if not accepted else None)

    facts = dict(schema_version='gradient_two_field_counterfactual_v1', parent_diagnostic_commit=parent_sha,
        historical_source_sha=original['source'], historical_result_commit=original['published'],
        fields_varied=['observed_failure', 'diagnosis'], limits=list(diagnostic.LIMITS), phases={},
        correction_limit_unchanged=400, expected_effect_limit_unchanged=240,
        original_prompt_preserved=True, schema_and_content_guards_preserved=True,
        production_policy_changed=False, historical_results_changed=False,
        real_api_calls=0, validation_raw_reads=0, test_raw_reads=0,
        provenance=dict(frozen_package_archive_sha256=original['facts']['frozen_package_archive_sha256'],
            historical_report_eol_projection=original['facts']['historical_report_eol_projection'],
            native_validator_ast_sha256=hashlib.sha256(ast.dump(ast.parse(native_source)).encode()).hexdigest(),
            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            diagnostic_helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest()))
    controls = None
    for phase in ('canary', 'pilot'):
        run = root/('runs/a4_v25_arm_b_openlux_seed84_'+phase+'_attempt1/execution')
        trace = [r for r in original['rows'](run/'provider_trace_private.jsonl')
            if r['role'] == 'pattern_gradient' and 'response' in r]
        groups = []
        cursor = 0
        lengths = {'observed_failure': [], 'diagnosis': [], 'reusable_correction': [], 'expected_effect': []}
        raw_errors = Counter()
        for row in original['rows'](run/'trajectory_private.jsonl'):
            if row['stage'] != 'OPPORTUNITY':
                continue
            for expected in row['opportunity']['pattern_context']['gradient_diagnostics']:
                group = []
                draws = trace[cursor:cursor+expected['structural_draws']]
                cursor += len(draws)
                assert len(draws) == expected['structural_draws']
                for i, draw in enumerate(draws):
                    packet = json.loads(draw['request']['messages'][1]['content'])
                    response = draw['response']
                    assert draw['split'] == 'optimize' and packet['example']['example_id'] == expected['example_id']
                    try:
                        value = json.loads(response['text'])
                    except (ValueError, TypeError):
                        value = None
                    if isinstance(value, dict):
                        for field in lengths:
                            if isinstance(value.get(field), str):
                                lengths[field].append(len(value[field]))
                    baseline = replay(native, packet, response)
                    accepted = i >= len(expected['structural_errors'])
                    assert baseline['accepted'] == accepted
                    if not accepted:
                        assert baseline['native_error'] == expected['structural_errors'][i]
                        raw_errors[baseline['native_error']] += 1
                    elif baseline['disposition'] == 'ACTIONABLE' and controls is None:
                        controls = packet, response
                    results = {key: replay(module, packet, response) for key, module in modules.items()}
                    assert results[240, 240] == baseline
                    # Increasing either cap cannot remove an original acceptance.
                    for (a, b), result in results.items():
                        assert not baseline['accepted'] or result['accepted']
                        if result['accepted']:
                            assert all(results[x, y]['accepted'] for x in diagnostic.LIMITS for y in diagnostic.LIMITS
                                if x >= a and y >= b)
                    group.append(dict(original_accepted=accepted, baseline=baseline, results=results,
                        input_tokens=response['input_tokens'], output_tokens=response['output_tokens']))
                groups.append(group)
        assert cursor == len(trace)
        table = []
        for (observed, diagnosis) in modules:
            cost_groups = [[dict(original_accepted=d['original_accepted'], input_tokens=d['input_tokens'],
                output_tokens=d['output_tokens'], **d['results'][observed, diagnosis]) for d in g] for g in groups]
            summary = diagnostic.counterfactual_summary(cost_groups)
            rejected = Counter(d['results'][observed, diagnosis]['native_error'] for g in groups for d in g
                if not d['results'][observed, diagnosis]['accepted'])
            newly_exposed = Counter(d['results'][observed, diagnosis]['native_error'] for g in groups for d in g
                if d['baseline']['native_error'] == 'REFERENCE_GRADIENT_OUTPUT_INVALID'
                and d['results'][observed, diagnosis]['native_error'] not in (None, 'REFERENCE_GRADIENT_OUTPUT_INVALID'))
            table.append(dict(observed_limit=observed, diagnosis_limit=diagnosis, **summary,
                all_response_native_rejections=dict(rejected), newly_exposed_correction_rejections=dict(newly_exposed)))
        original_cost = original['facts']['phases'][phase]['gradient']
        baseline_row = table[0]
        assert baseline_row['avoidable_usage_tokens'] == 0
        assert baseline_row['accepted_diagnoses'] == original_cost['accepted']
        assert baseline_row['historical_usage_tokens'] == original_cost['total_usage_tokens']
        assert baseline_row['all_response_full_pass'] == original_cost['accepted']
        distributions = {}
        for field, data in lengths.items():
            ordered = sorted(data)
            distributions[field] = dict(string_responses=len(data), maximum=max(data, default=0),
                over_limit_counts={str(cap): sum(n > cap for n in data) for cap in diagnostic.LIMITS},
                percentiles={str(p): ordered[min(len(ordered)-1, (len(ordered)*p+99)//100-1)]
                    for p in (50, 75, 90, 95, 100)} if ordered else {})
        facts['phases'][phase] = dict(matrix=table, equal_cap_comparison=[r for r in table
            if r['observed_limit'] == r['diagnosis_limit']], length_distributions=distributions,
            historical_extra_draws=original_cost['extra_draws'], historical_extra_usage_tokens=original_cost['extra_usage_tokens'],
            historical_phase_usage_tokens=original['facts']['phases'][phase]['phase_usage_tokens'],
            raw_inventory_sha256=original['facts']['phases'][phase]['raw_inventory_sha256'],
            original_native_rejections=dict(raw_errors))
        print(json.dumps(dict(phase=phase, status='THRESHOLD_REPLAY_PASS', responses=len(trace))))

    # Negative controls run complete native validation, with only synthetic edits
    # to private in-memory copies; no edited response is saved or accepted.
    assert controls is not None
    packet, response = controls
    base = json.loads(response['text'])
    negative = dict(correction_over_400={'reusable_correction': 'a'*401},
        expected_effect_over_240={'expected_effect': 'a'*241},
        illegal_block={'suggested_block': 'both'},
        unsafe_correction={'reusable_correction': 'Copy the reference solution.'},
        uncertain_nonnull={'disposition': 'UNCERTAIN'}, extra_schema_key={'extra': 'synthetic'})
    control_results = {}
    for name, change in negative.items():
        altered = dict(response, text=json.dumps(base | change))
        result = replay(modules[640, 640], packet, altered)
        assert not result['accepted']
        control_results[name] = 'REJECTED_BY_UNCHANGED_NATIVE_VALIDATOR'
    malformed = replay(modules[640, 640], packet, dict(response, text='{"diagnosis":"bad\\escape"}'))
    assert not malformed['accepted']
    control_results['malformed_json'] = 'REJECTED_BY_UNCHANGED_NATIVE_DECODER'
    provider = native.PerExampleGradientProvider.__new__(native.PerExampleGradientProvider)
    provider.optimization_evidence_policy = POLICY
    provider.solver_trajectory_policy = trajectory_policy()
    provider.input_schema = GRADIENT_INPUT
    def must_not_decode(_):
        raise AssertionError('CORRUPT_INPUT_REACHED_RESPONSE_DECODER')
    provider._call = must_not_decode
    corrupt = deepcopy(packet)
    corrupt['example']['reference_solution']['source'] = 'synthetic_untrusted_source'
    try:
        provider.extract(corrupt)
    except SearchContractError as exc:
        assert str(exc) == 'OPTIMIZE_REFERENCE_SOLUTION_PROVENANCE_REQUIRED'
    else:
        raise AssertionError('CORRUPT_SOURCE_PROVENANCE_ACCEPTED')
    control_results['reference_source_provenance'] = 'REJECTED_BEFORE_RESPONSE_DECODING'
    facts['negative_controls'] = control_results
    facts['replay_import_source_verified'] = all(Path(m.__file__).resolve().is_relative_to(frozen.resolve())
        for name, m in sys.modules.items() if name.startswith('multi_dataset_diverse_rl') and getattr(m, '__file__', None))
    assert facts['replay_import_source_verified'] and sitecustomize.network_attempt_count() == 0
    out = root/'runs/gradient_threshold_replay_20261011'
    out.mkdir(exist_ok=True)
    (out/'threshold_facts.json').write_bytes((json.dumps(facts, indent=2)+'\n').encode())
    print(json.dumps(dict(status='EXACT_SOURCE_TWO_FIELD_THRESHOLD_REPLAY_PASS', real_api_calls=0)))


if __name__ == '__main__':
    main()
