"""Current V2.3 checks ported from preserved replay tests; zero real APIs."""
import asyncio,json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from tests.current.test_optimization_evidence_v23 import contract

from multi_dataset_diverse_rl.benchmarks.math_domain_v2 import final_payload
from multi_dataset_diverse_rl.benchmarks.math_prediction_validity import classify_prediction
from multi_dataset_diverse_rl.benchmarks.math_v21_interface import MATHV21BenchmarkAdapter
from multi_dataset_diverse_rl.benchmarks.math_visible_trajectory import (
    adaptive_trajectory,profile_prediction,solver_profile,trajectory_policy)
from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
from multi_dataset_diverse_rl.search.provider_runtime import BenchmarkSolver,RequestBroker
visible_contract=contract
ROOT_PROMPT='Solve the problem.'
def response(text, finish='stop'):
    return dict(text=text, input_tokens=2, output_tokens=2, finish_reason=finish,
        provider_response_accepted=True, provider_metadata_loss_audited=True,
        provider_reasoning_content_present=False, provider_reasoning_character_count=None,
        provider_usage_details={}, provider_thinking_indicators=[])


def item(adapter, example_id='synthetic', problem='Synthetic equation: x+1=3.'):
    return protocol_input('math', example_id, {'problem':problem},
        adapter.output_contract, protocol=adapter.protocol)


@pytest.mark.parametrize('answer,reference,correct', [
    ('2', '2', True), ('3', '2', False), (r'\frac{3}{2}', '3/2', True),
    ('x+1', '1+x', True), (r'\{1,2,3\}', r'\{3,2,1\}', True),
])
def test_multiline_final_only_scoring_and_lossless_profile(answer, reference, correct):
    c = visible_contract(); adapter = MATHV21BenchmarkAdapter(c)
    text = 'First simplify the expression.\nThen check the resulting equation.\n\nFINAL_ANSWER: ' + answer
    prediction = classify_prediction(text)
    profile = solver_profile(dict(request_sha256='c' * 64), prediction, member_id=2,
        prompt=ROOT_PROMPT, example_id='synthetic', split='optimize', policy=trajectory_policy(), problem='Synthetic equation: x+1=3.')
    persisted = json.loads(json.dumps(profile))
    assert profile_prediction(persisted).text == text
    assert final_payload(text) == answer
    parsed = adapter.parse_member_output(persisted, item(adapter))
    assert parsed.valid and parsed.answer == answer
    assert adapter.score_member_output(parsed, reference) == float(correct)
    assert adapter.score_member_output(adapter.parse_member_output('FINAL_ANSWER: ' + answer, item(adapter)), reference) == float(correct)
    trajectory = adaptive_trajectory(persisted, member_id=2, prompt=ROOT_PROMPT, example_id='synthetic')
    assert trajectory['visible_solution'] == 'First simplify the expression.\nThen check the resulting equation.'
    assert trajectory['solution_status'] == 'WRITTEN_SOLUTION_PRESENT'
    assert not trajectory['feedback_truncated'] and not trajectory['response_truncated']
    assert trajectory['source']['request_sha256'] == 'c' * 64


@pytest.mark.parametrize('text,finish,reason', [
    ('The reasoning mentions the correct answer 2.', 'stop', 'MISSING_FINAL_MARKER'),
    ('Steps.\nFINAL_ANSWER: 2\nFINAL_ANSWER: 3', 'stop', 'MULTIPLE_FINAL_MARKERS'),
    ('Steps.\nFINAL_ANSWER: 2\nMore explanation.', 'stop', 'OTHER_PREDICTION_CONTRACT_FAILURE'),
    ('Steps.\nFINAL_ANSWER:', 'stop', 'EMPTY_FINAL_PAYLOAD'),
    ('Steps.\nFINAL_ANSWER: 2', 'length', 'OUTPUT_TRUNCATED'),
    ('Steps without an answer.', 'max_tokens', 'OUTPUT_TRUNCATED'),
    (None, 'stop', 'MISSING_FINAL_MARKER'),
])
def test_invalid_boundaries_and_truncation_never_guess_answer(text, finish, reason):
    prediction = classify_prediction(text, finish)
    assert not prediction.prediction_valid and prediction.answer == '' and prediction.invalid_reason == reason
    profile = solver_profile(dict(request_sha256='c' * 64), prediction, member_id=0,
        prompt=ROOT_PROMPT, example_id='synthetic', split='optimize', policy=trajectory_policy(), problem='Synthetic equation: x+1=3.')
    assert not profile_prediction(profile).prediction_valid
    trajectory = profile['solver_trajectory']
    assert trajectory['response_truncated'] is (reason == 'OUTPUT_TRUNCATED')
    assert trajectory['solution_status'] in {'RESPONSE_TRUNCATED', 'FINAL_BOUNDARY_INVALID'}


def test_bounded_projection_explicitly_marks_missing_or_truncated_feedback():
    long_solution = 'Written intermediate operation.\n' * 500
    text = long_solution + '\nFINAL_ANSWER: 2'
    profile = solver_profile(dict(request_sha256='c' * 64), classify_prediction(text), member_id=0,
        prompt=ROOT_PROMPT, example_id='synthetic', split='optimize', policy=trajectory_policy(), problem='Synthetic equation: x+1=3.')
    trajectory = profile['solver_trajectory']
    assert len(trajectory['visible_solution']) == 4096 and trajectory['feedback_truncated']
    assert trajectory['original_solution_characters'] == len(long_solution.rstrip())
    assert not trajectory['response_truncated'] and profile['prediction']['text'] == text
    missing = solver_profile(dict(request_sha256='c' * 64), classify_prediction('FINAL_ANSWER: 2'), member_id=0,
        prompt=ROOT_PROMPT, example_id='synthetic', split='optimize', policy=trajectory_policy(), problem='Synthetic equation: x+1=3.')
    assert missing['solver_trajectory']['visible_solution'] == ''
    assert missing['solver_trajectory']['solution_status'] == 'WRITTEN_SOLUTION_MISSING'
    assert profile_prediction(missing).prediction_valid  # No new scoring/validity penalty.


@pytest.mark.parametrize('changed', ['member', 'prompt', 'example', 'heldout', 'visible_text'])
def test_adaptive_provenance_rejects_cross_member_prompt_example_and_heldout(changed):
    profile = solver_profile(dict(request_sha256='c' * 64), classify_prediction('Steps.\nFINAL_ANSWER: 3'),
        member_id=0, prompt=ROOT_PROMPT, example_id='synthetic', split='optimize', policy=trajectory_policy(), problem='Synthetic equation: x+1=3.')
    kwargs = dict(member_id=0, prompt=ROOT_PROMPT, example_id='synthetic')
    if changed == 'member': kwargs['member_id'] = 1
    if changed == 'prompt': kwargs['prompt'] = 'Different procedure.'
    if changed == 'example': kwargs['example_id'] = 'other'
    if changed == 'heldout': profile['solver_trajectory']['source']['split'] = 'validation'
    if changed == 'visible_text': profile['solver_trajectory']['visible_solution'] = 'Forged steps.'
    with pytest.raises(SearchContractError): adaptive_trajectory(profile, **kwargs)


@pytest.mark.parametrize('kind', ['missing', 'truncated'])
def test_existing_four_attempt_recovery_preserves_all_original_visible_text(kind):
    c = visible_contract(); adapter = MATHV21BenchmarkAdapter(c); requests = []; raw = []
    text = 'Observed incomplete solution.' if kind == 'missing' else 'Observed steps.\nFINAL_ANSWER: 2'
    def transport(request):
        requests.append(deepcopy(request)); return response(text, 'stop' if kind == 'missing' else 'length')
    broker = RequestBroker(contract=c, transport=transport, arm='A4', seed=81, raw_writer=raw.append)
    solver = BenchmarkSolver(adapter, broker); solver.observe_member(3)
    profile = solver.solve(ROOT_PROMPT, item(adapter), stage='initial', split='optimize')
    prediction = profile_prediction(profile)
    assert len(requests) == 4 and all(req == requests[0] for req in requests)
    assert prediction.terminal_invalid and prediction.semantic_attempt_count == 4
    assert len(prediction.original_predictions) == 4 and all(p.text == text for p in prediction.original_predictions)
    assert len(raw) == 4 and all(row['response']['text'] == text for row in raw)
    assert solver.solve(ROOT_PROMPT, item(adapter), stage='full', split='optimize') == profile
    assert len(requests) == 4
