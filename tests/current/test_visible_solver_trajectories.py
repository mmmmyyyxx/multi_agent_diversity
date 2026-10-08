"""Ordinary visible solutions reach both current feedback paths, without APIs."""
import asyncio
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from multi_dataset_diverse_rl import versions
from multi_dataset_diverse_rl.benchmarks.math_domain_v2 import final_payload
from multi_dataset_diverse_rl.benchmarks.math_prediction_validity import classify_prediction
from multi_dataset_diverse_rl.benchmarks.math_v21_interface import (
    MATHV21BenchmarkAdapter, MATH_SOLVER_INTERFACE_V3, MATH_SOLVER_INTERFACE_V4_USER_SUFFIX,
    MATH_SOLVER_INTERFACE_V6, interface_for_contract, solver_user_content,
    v3_interface_contract, v4_interface_contract, v5_interface_contract, v6_interface_contract,
)
from multi_dataset_diverse_rl.benchmarks.math_visible_binding import (
    derive_visible_trajectory_contract, visible_gradient_prompt_artifact,
)
from multi_dataset_diverse_rl.benchmarks.math_visible_trajectory import (
    adaptive_trajectory, frozen_trajectory_policy, profile_prediction, solver_profile,
    text_hash, trajectory_policy, validate_adaptive_trajectory,
)
from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
from multi_dataset_diverse_rl.local_optimizers.schemas import LocalEvidenceExample
from multi_dataset_diverse_rl.search.binary_runtime import BinaryEvidenceSource, CorrectnessExample
from multi_dataset_diverse_rl.search.current_composition import build_current_team_prompt_search
from multi_dataset_diverse_rl.search.current_layer1 import CurrentOptimizer
from multi_dataset_diverse_rl.search.current_policy import CURRENT_POLICY_BUNDLE
from multi_dataset_diverse_rl.search.provider_runtime import BenchmarkSolver, ReflectionProvider, RequestBroker
from multi_dataset_diverse_rl.search.scientific_aggregation import EquivalencePluralityAggregation
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.search.textual_gradients import (
    GradientClusterProvider, PerExampleGradientProvider, VISIBLE_GRADIENT_PROMPT,
)

ROOT = Path(__file__).resolve().parents[2]
PARENT_PATH = 'experiments/execution_bindings/math_v2_2_gradient_pattern_seed81_pilot_v2.json'
ROOT_PROMPT = 'Solve the problem.'
CORRECTION = 'Check constraints before transforming intermediate expressions.'


def parent_contract():
    return json.loads((ROOT / PARENT_PATH).read_bytes())


def visible_contract():
    return derive_visible_trajectory_contract(parent_contract(), attempt='synthetic_visible_v6',
        binding_path='runs/synthetic_visible_binding.json', parent_path=PARENT_PATH,
        parent_sha256=sha256((ROOT / PARENT_PATH).read_bytes()).hexdigest(),
        authorization_path='runs/synthetic_visible_scope.json', authorization_sha256='a' * 64,
        gradient_prompt_path='runs/synthetic_visible_gradient.json',
        gradient_prompt_sha256=text_hash(json.dumps(visible_gradient_prompt_artifact())),
        validation_metadata_path='runs/synthetic_visible_validation_metadata.json',
        validation_metadata_sha256='b' * 64)


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


def test_interface_version_cache_separation_and_unchanged_wire_decoding():
    old, new = parent_contract(), visible_contract()
    from multi_dataset_diverse_rl.benchmarks.math_solver_decoding import generation_request_fields
    for role in ('solver', 'reflection', 'pattern_gradient', 'pattern_cluster'):
        assert generation_request_fields(old, role) == generation_request_fields(new, role)
    assert generation_request_fields(new, 'solver')['extra_body']['enable_thinking'] is False
    assert solver_user_content(new, ROOT_PROMPT, 'Public synthetic problem.') == ROOT_PROMPT + '\n\nPublic synthetic problem.'
    assert interface_for_contract(new) == (MATH_SOLVER_INTERFACE_V6, v6_interface_contract())
    assert 'Do not output reasoning' not in MATH_SOLVER_INTERFACE_V6
    for interface in (v3_interface_contract(), v4_interface_contract(), v5_interface_contract()):
        assert interface_for_contract(dict(solver_output_interface=interface))[0] == MATH_SOLVER_INTERFACE_V3
    assert solver_user_content(old, ROOT_PROMPT, 'Public synthetic problem.').endswith(MATH_SOLVER_INTERFACE_V4_USER_SUFFIX)
    a = RequestBroker(contract=old, transport=lambda _:None, arm='A4', seed=81)
    b = RequestBroker(contract=new, transport=lambda _:None, arm='A4', seed=81)
    kw = dict(role='solver', split='optimize', messages=[{'role':'user', 'content':'Same synthetic bytes.'}], member_slot=0)
    assert a._request_identity(**kw)[1] != b._request_identity(**kw)[1]
    new['solver_trajectory_policy']['max_visible_solution_characters'] = 4095
    with pytest.raises(SearchContractError): RequestBroker(contract=new, transport=lambda _:None, arm='A4', seed=81)
    old['solver_output_interface'] = v6_interface_contract()
    with pytest.raises(SearchContractError): frozen_trajectory_policy(old)


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


def make_graph(tmp_path, *, visible):
    c = visible_contract() if visible else parent_contract()
    c = deepcopy(c)
    c['execution_attempt_id'] = c['cache_namespace'] = 'synthetic_graph_visible' if visible else 'synthetic_graph_historical'
    adapter = MATHV21BenchmarkAdapter(c)
    requests, outputs, mutations = [], {}, []
    def examples(role):
        return tuple(CorrectnessExample(item(adapter, f'{role}{i}',
            f'Synthetic {role} equation case {i}: x+1=3.'), '2') for i in range(4))
    def transport(req):
        requests.append(deepcopy(req))
        if req['model'] == 'qwen3-8b':
            prompt, problem = req['messages'][1]['content'].split('\n\n', 1)
            i = int(problem.split('case ')[1].split(':')[0])
            correct = i == 3 or prompt != ROOT_PROMPT and i == 0
            answer = '2' if correct else '3'
            steps = f'Synthetic written step request {len(requests)}: subtract one from both sides.\nThe written intermediate result is x={answer}.'
            text = (steps + '\n\n' if visible else '') + 'FINAL_ANSWER: ' + answer
        elif len(req['messages']) == 2:
            payload = json.loads(req['messages'][1]['content'])
            if 'example' in payload:
                text = json.dumps({'gradient':CORRECTION})
            else:
                text = json.dumps({'patterns':[{'generalized_gradient':CORRECTION,
                    'support_ids':[g['example_id'] for g in payload['gradients']]}], 'unassigned_ids':[]})
        else:
            mutations.append(req)
            suffix = 'ABCDEF'[len(mutations)-1]
            text = json.dumps({'decision_procedure':f'Solve the problem. Check signs and verify constraints using review {suffix}.',
                'change_summary':'Add sign and constraint checks.'})
        return response(text)
    def raw_writer(row):
        if row['role'] == 'solver' and 'response' in row:
            outputs[row['request_sha256']] = row['response']['text']
    broker = RequestBroker(contract=c, transport=transport, arm='A4', seed=81, raw_writer=raw_writer)
    solver = BenchmarkSolver(adapter, broker); reflection = ReflectionProvider(broker)
    gradient = PerExampleGradientProvider(broker)
    pattern = GradientClusterProvider(broker, gradient_provider=gradient)
    method = CURRENT_POLICY_BUNDLE.method(aggregation=versions.EQUIVALENCE_PLURALITY_VERSION,
        provider_binding='0'*64, successful_provider_calls=c['provider_bounds']['successful_provider_calls'],
        solver_trajectory_policy=trajectory_policy() if visible else None)
    optimizer = CurrentOptimizer(evaluator=solver, reflection_lm=reflection,
        accounting_reader=reflection.accounting, run_root=tmp_path)
    run = build_current_team_prompt_search(benchmark=adapter, aggregation=EquivalencePluralityAggregation(),
        examples=examples('optimize'), prompts=(ROOT_PROMPT,)*5, solver=solver, optimizer=optimizer,
        method=method, seed=81, shadow_loader=lambda:examples('shadow'), shadow_count=4,
        runtime_readiness=lambda:(), pattern_provider=pattern, provider_call_reader=lambda:broker.successes)
    run.state.initialize()
    return run, broker, requests, outputs


def test_current_graph_exact_member_gradient_and_next_generation_candidate_feedback(tmp_path):
    run, broker, requests, outputs = make_graph(tmp_path, visible=True)
    initial = run.state.snapshot()
    result = asyncio.run(run.run(max_opportunities=1))
    target = result.trace[0].target_member
    gradient_requests = [json.loads(req['messages'][1]['content']) for req in requests
        if req['model'] != 'qwen3-8b' and len(req['messages']) == 2
        and 'example' in json.loads(req['messages'][1]['content'])]
    assert len(gradient_requests) == 3  # One independent request per wrong example.
    for payload in gradient_requests:
        assert payload['schema'] == versions.GRADIENT_VISIBLE_INPUT_VERSION
        example = payload['example']; trajectory = example['solver_trajectory']; source = trajectory['source']
        i = int(example['example_id'].removeprefix('optimize'))
        profile = initial.diagnostics['raw_profiles'][target][i]
        assert source['member_id'] == target and source['example_id'] == example['example_id']
        assert payload['current_member_procedure'] == ROOT_PROMPT
        assert trajectory == profile['solver_trajectory']
        assert example['prediction'] == '3' and example['reference'] == '2' and example['valid']
        assert not any(f'case {j}:' in json.dumps(payload) for j in range(4) if j != i)
    cluster = [json.loads(req['messages'][1]['content']) for req in requests
        if req['model'] != 'qwen3-8b' and len(req['messages']) == 2
        and 'gradients' in json.loads(req['messages'][1]['content'])]
    assert len(cluster) == 1 and set(cluster[0]) == {'gradients'}
    assert 'solver_trajectory' not in json.dumps(cluster)
    mutations = [json.loads('{' + req['messages'][0]['content'].split('\n{', 1)[1])
        for req in requests if req['model'] != 'qwen3-8b' and len(req['messages']) == 1]
    assert len(mutations) == 6
    for index, payload in enumerate(mutations):
        assert payload['schema'] == versions.GRADIENT_VISIBLE_OPTIMIZER_INPUT_VERSION
        parent = payload['current_parent']
        if index == 1: assert parent != ROOT_PROMPT
        for row in payload['current_panel_observations']:
            trajectory = row['solver_trajectory']
            source = trajectory['source']
            assert source['member_id'] == target and source['mutable_prompt_sha256'] == text_hash(parent)
            assert source['split'] == 'optimize'
            problem = row['problem']['problem'] if isinstance(row['problem'], dict) else row['problem']
            actual_output = outputs[source['request_sha256']]
            assert trajectory['raw_response_sha256'] == text_hash(actual_output)
            assert trajectory['visible_solution'] in actual_output
        assert all('solver_trajectory' in row for row in payload['representative_failure_trajectories'])
    assert len(result.transitions) == 1 and run.gate.winner_count == 1
    assert broker.usage['validation'] == broker.usage['test'] == 0
    for req in requests:
        if req['model'] == 'qwen3-8b':
            assert req['extra_body']['enable_thinking'] is False
            assert set(req['messages'][0]) == {'role', 'content'}
            assert req['messages'][0]['content'] == MATH_SOLVER_INTERFACE_V6
            assert 'Reference answer' not in req['messages'][1]['content']
    assert 'Synthetic written step' not in json.dumps(run.memory.audit())
    assert 'Synthetic written step' not in str(run.memory.private) + str(run.memory.shared)
    for state in (initial, run.state.snapshot()):
        for lane in state.diagnostics['raw_profiles']:
            for profile in lane:
                assert profile_prediction(json.loads(json.dumps(profile))).text == profile['prediction']['text']


def test_visible_evidence_noninterference_with_current_search_and_team_policies(tmp_path):
    old, a, _, _ = make_graph(tmp_path / 'old', visible=False)
    new, b, _, _ = make_graph(tmp_path / 'new', visible=True)
    before = asyncio.run(old.run(max_opportunities=1))
    after = asyncio.run(new.run(max_opportunities=1))
    assert a.usage == b.usage
    assert before.stop_reason == after.stop_reason == 'CANARY_ONE_PRODUCTION_OPPORTUNITY_COMPLETE'
    x, y = before.trace[0], after.trace[0]
    assert x.target_member == y.target_member
    assert x.committed_candidate_id == y.committed_candidate_id
    assert x.allocation_audit == y.allocation_audit
    assert old.state.snapshot().member_scores == new.state.snapshot().member_scores
    assert old.state.snapshot().team_scores == new.state.snapshot().team_scores
    assert old.transition.initial_scores == new.transition.initial_scores
    assert old.memory.audit() == new.memory.audit()
    assert old.memory.private == new.memory.private and old.memory.shared == new.memory.shared
    assert old.evaluation.provider.probed == new.evaluation.provider.probed
    assert old.evaluation.provider.fulled == new.evaluation.provider.fulled
    for key in ('identity_version', 'metric_limit', 'panel_size', 'max_generations', 'k_local_return',
            'max_prompt_chars', 'candidate_contract', 'official_gepa_fidelity', 'panel_policy'):
        assert getattr(old.engine.optimizer.config, key) == getattr(new.engine.optimizer.config, key)


def test_state_to_feedback_rejects_another_members_realization(tmp_path):
    run, _, _, _ = make_graph(tmp_path, visible=True)
    snapshot = run.state.snapshot()
    bad = deepcopy(snapshot.diagnostics)
    bad['raw_profiles'] = (bad['raw_profiles'][1], *bad['raw_profiles'][1:])
    with pytest.raises(SearchContractError, match='PROVENANCE'):
        BinaryEvidenceSource(run.state, run.history).for_member(
            replace(snapshot, diagnostics=bad), run.analyzer.analyze(snapshot, run.history), 0)


def test_gradient_and_local_feedback_cannot_accept_heldout_trajectory(tmp_path):
    run, broker, _, _ = make_graph(tmp_path, visible=True)
    snapshot = run.state.snapshot()
    row = BinaryEvidenceSource(run.state, run.history).for_member(
        snapshot, run.analyzer.analyze(snapshot, run.history), 0)[0]
    from multi_dataset_diverse_rl.search.pattern_primitives import single_failure_example
    example = single_failure_example(row)
    example['solver_trajectory']['source']['split'] = 'validation'
    provider = PerExampleGradientProvider(broker)
    before = broker.usage['attempts']
    with pytest.raises(SearchContractError, match='PROVENANCE'):
        provider.extract(dict(schema=versions.GRADIENT_VISIBLE_INPUT_VERSION,
            current_member_procedure=ROOT_PROMPT, example=example))
    with pytest.raises(SearchContractError, match='PROVENANCE'):
        validate_adaptive_trajectory(example['solver_trajectory'], example_id=example['example_id'])
    assert broker.usage['attempts'] == before


def test_local_observation_retains_normal_content_and_ignores_private_reasoning_field():
    c = visible_contract(); adapter = MATHV21BenchmarkAdapter(c)
    text = 'Actual written subtraction gives x=3.\nFINAL_ANSWER: 3'
    def transport(req): return dict(response(text), reasoning_content='PRIVATE_REASONING_SENTINEL')
    broker = RequestBroker(contract=c, transport=transport, arm='A4', seed=81)
    solver = BenchmarkSolver(adapter, broker); solver.observe_member(4)
    observation = solver.evaluate(ROOT_PROMPT, LocalEvidenceExample('synthetic', 'Synthetic equation: x+1=3.', '2'))
    assert observation.valid and not observation.correct and observation.parsed_answer == '3'
    assert observation.raw_output == text
    assert observation.solver_trajectory['visible_solution'] == 'Actual written subtraction gives x=3.'
    assert 'PRIVATE_REASONING_SENTINEL' not in json.dumps(observation.solver_trajectory)


def visible_manifest():
    import yaml
    manifest = yaml.safe_load((ROOT / 'experiments/manifests/math_v2_2_gradient_pattern_seed81_pilot_v2.yaml').read_bytes())
    c = visible_contract()
    method = CURRENT_POLICY_BUNDLE.method(aggregation=c['aggregation'], provider_binding='0'*64,
        successful_provider_calls=c['provider_bounds']['successful_provider_calls'],
        partition_completion_policy=c.get('partition_completion_policy'),
        gradient_recovery_policy=c.get('gradient_recovery_policy'),
        pattern_cluster_generation_policy=c.get('pattern_cluster_generation_policy'),
        solver_trajectory_policy=trajectory_policy())
    manifest.update(experiment_id='synthetic_visible', lifecycle={'status':'DRAFT', 'history':[]},
        solver_output_interface_identity=versions.MATH_SOLVER_INTERFACE_V6_VERSION,
        solver_trajectory_policy=trajectory_policy(), layer1_search_policy=c['layer1_search_policy'],
        mechanism_config=method.mechanism_config,
        execution_binding={'identity':c['identity'], 'path':c['binding_path'], 'sha256':'0'*64})
    return manifest


def test_fresh_manifest_schema_and_authorization_scope_bind_feedback_identities():
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    from multi_dataset_diverse_rl.governance.unified_execution import execution_scope
    manifest = visible_manifest()
    assert not validate_manifest_v2(ROOT, manifest)
    scope = execution_scope(manifest, visible_contract())
    assert scope['solver_trajectory_policy'] == trajectory_policy()
    assert scope['solver_output_interface'] == v6_interface_contract()
    for change in ('remove_policy', 'wrong_interface', 'old_binding', 'old_schema', 'budget', 'missing_mechanism_policy'):
        bad = deepcopy(manifest)
        if change == 'remove_policy': bad.pop('solver_trajectory_policy')
        elif change == 'wrong_interface': bad['solver_output_interface_identity'] = 'MATH_SOLVER_INTERFACE_V5'
        elif change == 'old_binding': bad['execution_binding']['identity'] = 'MATH_V2_2_EXECUTION_BINDING_V1'
        elif change == 'old_schema': bad['mechanism_config']['optimizer_input_schema'] = versions.GRADIENT_OPTIMIZER_INPUT_VERSION
        elif change == 'budget': bad['layer1_search_policy']['max_generations'] = 7
        else: bad['mechanism_config'].pop('solver_trajectory_policy')
        assert validate_manifest_v2(ROOT, bad), change


def test_existing_manifest_schema_acceptance_is_unchanged_for_historical_corpus():
    import subprocess
    import jsonschema
    import yaml
    old_schema = json.loads(subprocess.check_output(['git', 'show',
        'ee65f9ea:experiments/schema/experiment_manifest_v2.schema.json'], cwd=ROOT))
    new_schema = json.loads((ROOT/'experiments/schema/experiment_manifest_v2.schema.json').read_bytes())
    old_validator = jsonschema.Draft202012Validator(old_schema)
    new_validator = jsonschema.Draft202012Validator(new_schema)
    count = 0
    for path in sorted((ROOT/'experiments/manifests').glob('*.yaml')):
        manifest = yaml.safe_load(path.read_bytes())
        if isinstance(manifest, dict) and manifest.get('schema_version') == 'experiment_manifest_v2':
            count += 1
            assert old_validator.is_valid(manifest) == new_validator.is_valid(manifest), path.name
    assert count >= 10


@pytest.mark.parametrize('problem', ['Plain synthetic input.', '含 Unicode 的合成题面。', 'Quote " and newline\n and slash \\.'])
def test_accounting_rebinding_is_exact_without_opening_heldout_content(problem):
    from multi_dataset_diverse_rl.benchmarks.math_visible_binding import derive_visible_accounting_metadata
    from multi_dataset_diverse_rl.benchmarks.math_accounting_prep import solver_request
    from multi_dataset_diverse_rl.benchmarks.math_solver_decoding import generation_request_fields
    from multi_dataset_diverse_rl.governance.token_accounting import serialized_request
    old, new = parent_contract(), visible_contract()
    def old_request(text):
        return dict(model=old['models']['solver'], **generation_request_fields(old, 'solver'),
            messages=[{'role':'system','content':interface_for_contract(old)[0]},
                {'role':'user','content':solver_user_content(old, '', text)}])
    metadata = {'solver_output_interface':old['solver_output_interface'],
        **{k:old[k] for k in ('decoding','solver_decoding_policy','prediction_validity_policy',
            'invalid_recovery_policy','low_cost_protocol')},
        'examples':[{'example_id':'synthetic','input_sha256':'0'*64,
            'blank_prompt_serialized_request_bytes':len(serialized_request(old_request(problem)))}]}
    rebound = derive_visible_accounting_metadata(metadata, old, new)
    assert rebound['examples'][0]['blank_prompt_serialized_request_bytes'] == len(serialized_request(solver_request(new, '', problem)))
    assert rebound['visible_interface_length_rebinding']['new_heldout_content_reads'] == 0
    assert rebound['solver_output_interface'] == v6_interface_contract()
    assert metadata['solver_output_interface'] == v5_interface_contract()
    assert not any(key in rebound for key in ('problem','reference','solution'))


def test_unfrozen_visible_profile_stops_before_binding_data_or_provider():
    from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
    from multi_dataset_diverse_rl.governance.unified_execution import bound_preflight
    c = {'identity':versions.MATH_VISIBLE_TRAJECTORY_BINDING_VERSION,
        'status':'HOLD', 'real_api_authorized':False}
    assert execution_binding(ROOT, c).blockers() == ('MATH_VISIBLE_TRAJECTORY_FRESH_FREEZE_REQUIRED',)
    result = bound_preflight(ROOT, {'execution_binding':{'identity':c['identity'],
        'path':'runs/does_not_exist_synthetic.json','sha256':'0'*64}})
    assert result['gate'] == 'HOLD' and result['provider_attempts'] == 0


def test_visible_gradient_requires_the_versioned_prompt_before_provider():
    broker = RequestBroker(contract=visible_contract(), transport=lambda _: pytest.fail('provider called'), arm='A4', seed=81)
    with pytest.raises(SearchContractError, match='MATH_VISIBLE_GRADIENT_PROMPT_MISMATCH'):
        PerExampleGradientProvider(broker, 'Historical answer-only evidence prompt.')
