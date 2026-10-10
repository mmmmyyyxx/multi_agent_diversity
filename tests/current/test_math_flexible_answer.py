"""Presentation compatibility does not turn mathematical mistakes into repairs."""
import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.benchmarks.math_flexible_answer import (
    IDENTITY, POLICY, boundaries, classify_prediction, extract_answer, prediction_from_persisted,
)
from multi_dataset_diverse_rl.benchmarks.math_structured_answer import classify_prediction as historical
from multi_dataset_diverse_rl.search.current_policy import require_current_contract
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from tests.current.test_structured_system_contract import solve_sequence
from tests.current.test_structured_optimization_evidence import contract
from tests.current.test_structured_optimization_evidence import graph


@pytest.mark.parametrize('text,expected', [
    ('Final answer: 5\nTherefore, x = 5.', '5'),
    ('Therefore, x = 5.', '5'),
    (r'Therefore, x=\theta.', r'x=\theta'),
    (r'Therefore, x=\alpha+1.', r'x=\alpha+1'),
    ('Thus x = 5', '5'),
    ('Hence, 5.', '5'),
    ('So, the answer is 5.', '5'),
    ('The final answer is: 5.', '5'),
    ('Answer: 5', '5'),
    ('Answer = 5', '5'),
    ('**Final answer:** `5`', '5'),
    ('### Final answer: **5**', '5'),
    ('**Final answer: 5**.', '5'),
    ('- **Answer:** $5$', '5'),
    ('Final answer:\n5', '5'),
    ('### Final answer:\nFinal answer: 5', '5'),
    ('Final answer:\nTherefore, x=5.', '5'),
    (r'Final answer: (a,b,c)=(2,3,4)', '(2,3,4)'),
    ('**Final answer:** (a,b,c)=(2,3,4)\nFinal answer: (2,3,4)', '(2,3,4)'),
    ('Final answer: 5\nA concluding explanation follows.', '5'),
    ('Final answer: 5\nTherefore, this completes the proof.', '5'),
    ('Final answer: 5\nTherefore, x=6.\nFinal answer: 5', '5'),
    ('Final answer: 0.5\nThus x = 1/2.', '0.5'),
    ('Final answer: x=5\nTherefore, x=5.', 'x=5'),
    (r'\boxed{5}', '5'),
    ('\\[\n\\boxed{\\frac{1}{2}}\n\\]', r'\frac{1}{2}'),
    ('Final answer:\n$$\n\\boxed{5}\n$$', '5'),
    ('Answer:\n```latex\n\\frac{1}{2}\n```', r'\frac{1}{2}'),
    (r'Therefore, x = \boxed{5}.', '5'),
    ('Intermediate \\boxed{99}.\nTherefore, x=3.\nFinal answer: 5\nThus x=5.', '5'),
    ('Final answer: 12\u221a2', r'12\sqrt{2}'),
    ('Answer: \u2212\u03c0', r'-\pi'),
    (r'Final answer: \begin{pmatrix}1\\2\end{pmatrix}', r'\begin{pmatrix}1\\2\end{pmatrix}'),
])
def test_clear_presentation_forms(text, expected):
    assert extract_answer(text) == (expected, None)
    prediction = classify_prediction(text)
    assert prediction.prediction_valid and prediction.answer == expected
    assert prediction_from_persisted(asdict(prediction)) == prediction


@pytest.mark.parametrize('text', [
    'Final answer: 5\nTherefore, x=6.',
    'Answer: 5\nFinal answer: 6',
    'Final answer: (a,b,c)=(2,3,4)\nFinal answer: (2,3,5)',
    'Final answer: 5\nTherefore, x=5.\nThus x=6.',
    'Final answer: 5 or 6',
    'Answer: 5 and 6',
    'Therefore, maybe 5.',
    'Final answer: \\boxed{5',
    'Final answer:\n\\[\n5',
    'Answer:\n```latex\n5',
    'Answer: `5',
    'Answer: $5',
    'Answer: 12\u221a2x',
    'Final answer:',
    '42',
    '### 42',
    'Intermediate value 42; another possibility is 5.',
    '\\boxed{5}\nContinue working: x=6.',
    'Answer: \\boxed{5} or \\boxed{6}',
])
def test_conflicts_ambiguity_and_incomplete_wrappers_remain_invalid(text):
    assert not classify_prediction(text).prediction_valid


@pytest.mark.parametrize('finish', ['length', 'max_tokens', 'max_output_tokens', 'content_filter', None])
def test_incomplete_provider_response_never_counts(finish):
    prediction = classify_prediction('Final answer: 5\nTherefore, x=5.', finish)
    assert not prediction.prediction_valid


def test_clear_wrong_answer_stops_recovery_and_scores_wrong():
    profile, requests, adapter, item = solve_sequence(['Final answer: 5\nTherefore, x=5.'])
    assert len(requests) == 1 and profile['prediction']['semantic_attempt_count'] == 1
    parsed = adapter.parse_member_output(profile, item)
    assert parsed.valid and adapter.score_member_output(parsed, '6') == 0
    assert adapter.score_member_output(parsed, '5') == 1
    assert profile['prediction']['raw_invalid_count'] == 0


def test_answer_region_starts_at_label_before_confirmation():
    text = 'Check the constraint.\nFinal answer: 5\nTherefore, x=5.'
    primary = next(r for r in boundaries(text) if r[3])
    assert text[:primary[1]] == 'Check the constraint.\n'


def test_duplicate_heading_is_not_fictitious_written_reasoning():
    profile, requests, _, _ = solve_sequence(['### Final answer:\nFinal answer: 5'])
    assert len(requests) == 1
    assert profile['solver_trajectory']['solution_status'] == 'TRAJECTORY_UNAVAILABLE'


@pytest.mark.parametrize('text', ['Final answer: (a,a,c)=(2,3,4)',
                                 'Final answer: (a,b,c)=(2,3,4,5)'])
def test_coordinate_constraints_are_not_discarded(text):
    prediction=classify_prediction(text)
    assert not prediction.prediction_valid or prediction.answer != text.split('=')[1]


def test_five_member_vote_uses_equivalence_across_presentations():
    _, _, adapter, item = solve_sequence(['Answer: 0.5'])
    from multi_dataset_diverse_rl.search.scientific_aggregation import EquivalencePluralityAggregation
    result = EquivalencePluralityAggregation().aggregate_sync(item=item, benchmark=adapter,
        member_outputs=['Answer: 0.5', r'\boxed{\frac{1}{2}}',
                        'Therefore, x = 1/2.', 'Final answer: 7', 'unclear'])
    assert result.parsed_output.valid
    assert result.diagnostics['equivalence_classes'] == ((0, 1, 2), (3,))
    assert result.diagnostics['invalid_abstentions'] == 1


def test_natural_confirmation_reaches_gradient_and_full_commit_without_format_failure(tmp_path):
    run, _, requests, _, _ = graph(tmp_path, answer_only=True, answer_only_form='natural_confirmation')
    result = asyncio.run(run.run(max_opportunities=1))
    assert result.transitions
    gradients = []
    for request in requests:
        if request['model'] == 'gpt-4o-mini' or len(request['messages']) != 2:
            continue
        packet = json.loads(request['messages'][1]['content'])
        if 'example' in packet:
            gradients.append(packet['example']['solver_trajectory'])
    assert gradients and all(row['prediction_valid'] for row in gradients)
    assert all(row['invalid_reason'] is None and not row['retry_summary']['repeated_format_failure'] for row in gradients)
    assert all(row['solution_status'] == 'TRAJECTORY_UNAVAILABLE' for row in gradients)


def test_new_current_identity_rejects_old_parser_and_cache():
    value = contract()
    assert value['answer_extraction_policy'] == POLICY
    require_current_contract(value)
    for key, old in [('cache_policy', 'STRUCTURED_SYSTEM_MEMBER_LANE_CACHE_V3'),
                     ('method_identity', 'unified_team_prompt_search_v2_5_responsibility_fallback_repair_probe'),
                     ('answer_extraction_policy', {'identity': 'MATH_EXPLICIT_FINAL_ANSWER_EXTRACTION_V2'})]:
        changed = deepcopy(value)
        changed[key] = old
        with pytest.raises(SearchContractError):
            require_current_contract(changed)
    assert IDENTITY == 'MATH_FLEXIBLE_ANSWER_EXTRACTION_V3'


def test_historical_parser_is_immutable_and_persisted_state_is_revalidated():
    text = 'Final answer: 5\nTherefore, x=5.'
    assert not historical(text).prediction_valid
    data = asdict(classify_prediction(text))
    data['answer'] = '6'
    with pytest.raises(SearchContractError, match='CORRUPTION'):
        prediction_from_persisted(data)
    with pytest.raises(SearchContractError, match='CORRUPTION'):
        prediction_from_persisted(asdict(historical(text)))


def test_declaration_resource_limits_are_invalidity_not_worker_integrity_failures():
    assert classify_prediction('Answer: '+('1+'*1100)+'1\nAnswer: 5').invalid_reason == 'PAYLOAD_UNSUPPORTED'
    assert not classify_prediction('\n'.join(['Answer: 5']*17)).prediction_valid
    working='\n'.join('Therefore, x='+str(i)+'.' for i in range(30))
    assert classify_prediction(working+'\nFinal answer: 5').answer == '5'


def test_amended_manifest_schema_binds_actual_parser_and_requires_frozen_binding():
    import yaml
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    root=Path(__file__).resolve().parents[2]
    manifest=yaml.safe_load((root/'experiments/manifests/math_flexible_answer_parser_v3.yaml').read_text(encoding='utf-8'))
    assert not validate_manifest_v2(root,manifest)
    changed=deepcopy(manifest)
    changed['answer_extraction_policy']={'identity':'MATH_EXPLICIT_FINAL_ANSWER_EXTRACTION_V2'}
    assert validate_manifest_v2(root,changed)
    changed=deepcopy(manifest)
    changed['lifecycle']['status']='PREEXECUTION_FROZEN'
    assert validate_manifest_v2(root,changed)


def test_normative_amendment_enters_scientific_identity():
    from multi_dataset_diverse_rl.governance.source_identity import current_scientific_files
    root=Path(__file__).resolve().parents[2]
    assert root/'docs/design/FLEXIBLE_ANSWER_EXTRACTION_V3.md' in current_scientific_files(root)


def test_offline_adapter_exposes_the_same_new_protocol_identity():
    from multi_dataset_diverse_rl.benchmarks.math import MATHBenchmarkAdapter
    adapter=MATHBenchmarkAdapter()
    assert adapter.parser_identity == adapter.protocol.parser_contract_id == IDENTITY
    assert adapter.output_contract == adapter.protocol.system_contract_id
    assert adapter.final_payload('Final answer: 5\nTherefore, x=5.') == '5'
