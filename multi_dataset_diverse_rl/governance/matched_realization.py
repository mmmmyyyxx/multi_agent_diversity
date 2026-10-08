"""Explicit paired Solver realizations and initial operational review.

No optimizer inputs, scoring, search choices or admission rules change here.
Historical attempts remain isolated unless this complete policy is frozen.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from ..search.schemas import SearchContractError
from ..persistence.durable_io import atomic_write_json, read_json

from ..current_contract import MATH_MATCHED_REALIZATION_POLICY_VERSION as IDENTITY


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def inference_identity(c):
    keys = ('provider', 'solver_output_interface', 'solver_decoding_policy',
        'prediction_validity_policy', 'invalid_recovery_policy', 'benchmark_protocol_sha256',
        'low_cost_protocol', 'cache_policy', 'initial_team_sha256', 'split_manifest_sha256',
        'low_cost_subsets_sha256', 'verify_settings_sha256')
    return digest({k: c[k] for k in keys} | {'solver': c['models']['solver'],
        'solver_thinking': c['models']['solver_thinking'], 'seeds': c['seeds']})


def paired_policy(c, *, group_id, cell, attempts, binding_paths, protocol_sha256):
    return dict(identity=IDENTITY, group_id=group_id, cell=cell, attempts=dict(attempts),
        binding_paths=dict(binding_paths),
        protocol_path='experiments/protocols/a4_v23_matched_comparison_v1/protocol.json',
        protocol_sha256=protocol_sha256,
        inference_identity=inference_identity(c), max_opportunities_per_cell=5,
        charged_tokens_per_cell=2_000_000, charged_tokens_total=4_000_000,
        cache_directory=f'runs/{group_id}/paired_solver_cache',
        authorization_file=f'runs/{group_id}/paired_authorization.json',
        canary_review='OWNER_INITIAL_PROFILE_REVIEW_BEFORE_OPTIMIZATION',
        canary_review_timeout_seconds=900)


def group_scope(policy):
    return {k: v for k, v in policy.items() if k != 'cell'}


def validate_policy(c):
    p = c.get('paired_realization_policy')
    if p is None:
        return None
    try:
        attempts = p['attempts']
        expected = paired_policy(c, group_id=p['group_id'], cell=p['cell'], attempts=attempts,
            binding_paths=p['binding_paths'], protocol_sha256=p['protocol_sha256'])
        if (p != expected or p['cell'] not in {'A', 'B'} or set(attempts) != {'A', 'B'}
                or len(set(attempts.values())) != 2
                or any(not isinstance(v, str) or not v for v in attempts.values())
                or set(p['binding_paths']) != {'A', 'B'}
                or len(set(p['binding_paths'].values())) != 2
                or any(not isinstance(v, str) or not v.startswith('experiments/execution_bindings/')
                    or '..' in Path(v).parts for v in p['binding_paths'].values())
                or c['binding_path'] != p['binding_paths'][p['cell']]
                or c['execution_attempt_id'] != attempts[p['cell']]
                or not isinstance(p['protocol_sha256'], str) or len(p['protocol_sha256']) != 64
                or any(ch not in '0123456789abcdef' for ch in p['protocol_sha256'])
                or not isinstance(p['group_id'], str) or not p['group_id']
                or any(x not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for x in p['group_id'])
                or c['seeds'] != [81] or c['models']['solver'] != 'qwen3-8b'
                or c['models']['solver_thinking'] is not False
                or c['solver_output_interface']['identity'] != 'MATH_SOLVER_INTERFACE_V6'
                or c['identity'] != {'A': 'MATH_V2_2_VISIBLE_TRAJECTORY_EXECUTION_BINDING_V1',
                    'B': 'MATH_OPTIMIZATION_EVIDENCE_BINDING_V1'}[p['cell']]
                or bool(c.get('optimization_evidence_policy')) != (p['cell'] == 'B')
                or c['operational_pilot']['max_opportunities'] != 5
                or c['provider_bounds']['max_opportunities'] != 5
                or c['operational_pilot']['token_ceiling'] != 2_000_000):
            raise ValueError('paired settings mismatch')
    except (KeyError, TypeError, ValueError) as exc:
        raise SearchContractError('MATCHED_REALIZATION_POLICY_MISMATCH') from exc
    return deepcopy(p)


def validate_bindings(root, c):
    p = validate_policy(c)
    if p is None:
        return
    if hashlib.sha256((root / p['protocol_path']).read_bytes()).hexdigest() != p['protocol_sha256']:
        raise SearchContractError('MATCHED_PROTOCOL_HASH_MISMATCH')
    peer = read_json(root / p['binding_paths']['B' if p['cell'] == 'A' else 'A'])
    other = validate_policy(peer)
    if other is None or group_scope(p) != group_scope(other):
        raise SearchContractError('MATCHED_PEER_BINDING_MISMATCH')
    from ..benchmarks.math_domain_binding import execution_binding
    blockers = execution_binding(root, peer).blockers()
    if blockers:
        raise SearchContractError('MATCHED_PEER_BINDING_INVALID: ' + blockers[0])


def local_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to((root / 'runs').resolve()):
        raise SearchContractError('MATCHED_REALIZATION_PATH_ESCAPE')
    return path


def validate_authorization(root, c, payload):
    p = validate_policy(c)
    if p is None:
        return None
    try:
        auth = read_json(local_path(root, p['authorization_file']))
        startups = auth['approved_startups']
        if (auth.get('explicit_user_authorized') is not True
                or auth.get('single_use_per_cell') is not True
                or auth.get('scope') != group_scope(p)
                or set(startups) != set(p['attempts'].values())
                or any(not isinstance(v, str) or len(v) != 64 for v in startups.values())
                or any(any(ch not in '0123456789abcdef' for ch in v) for v in startups.values())
                or startups[c['execution_attempt_id']] != payload['startup_identity_sha256']
                or auth.get('source_sha') != payload['manifest']['source_sha']):
            raise ValueError('paired authorization mismatch')
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SearchContractError('EXACT_PAIRED_AUTHORIZATION_REQUIRED') from exc
    return auth


def cache_context(root, c, payload):
    p = validate_policy(c)
    auth = validate_authorization(root, c, payload)
    return dict(execution_attempt_id=p['group_id'], cache_namespace=p['group_id'],
        startup_identity_sha256=digest(auth['approved_startups']),
        source_sha=payload['manifest']['source_sha'], authorization_sha256=digest(auth),
        binding_sha256=digest(group_scope(p)),
        generation_policy_sha256=digest(c['solver_decoding_policy']),
        recovery_policy_sha256=digest(c['invalid_recovery_policy']))


def solver_identity(c, *, request, split, member_slot, seed):
    p = validate_policy(c)
    if seed != 81 or type(member_slot) is not int or member_slot not in range(5):
        raise SearchContractError('MATCHED_REALIZATION_LANE_MISMATCH')
    # Only inference information determines the common realization. The complete
    # treatment still binds each manifest, prep, authorization and optimizer key.
    return dict(policy=group_scope(p), provider=c['provider'], role='solver', split=split,
        request=request, seed=seed, member_realization_lane=member_slot)


def freeze_initial_and_review(root, c, payload, run_root, snapshot):
    p = validate_policy(c)
    if p is None:
        return
    validate_authorization(root, c, payload)
    from .autonomous_math import plain
    profiles = plain(snapshot.diagnostics['raw_profiles'])
    rows = snapshot.diagnostics['team_states']
    value = dict(scope=group_scope(p), source_sha=payload['manifest']['source_sha'],
        support_identity=snapshot.diagnostics['evaluation_support_identity'],
        member_prompt_hashes=[hashlib.sha256(v.encode()).hexdigest() for v in snapshot.member_prompts],
        member_scores=list(snapshot.member_scores), team_scores=dict(snapshot.team_scores),
        oracle=sum(any(row.team_correctness) for row in rows), profile_sha256=digest(profiles))
    sealed = {**value, 'integrity_sha256': digest(value)}
    baseline = local_path(root, p['cache_directory']).parent / 'paired_initial_baseline.json'
    if p['cell'] == 'A':
        # Independently parse and score B's initial state through the current
        # composition, using the common cache exclusively. No new provider call
        # is possible in this verification. Both floors precede A's search.
        from ..benchmarks.math_domain_binding import execution_binding
        from ..search.provider_runtime import RequestBroker, BenchmarkSolver, ReflectionProvider
        from ..search.textual_gradients import PerExampleGradientProvider, GradientClusterProvider
        from ..persistence.exact_output_cache import DurableExactOutputCache
        peer = read_json(root / p['binding_paths']['B'])
        auth = validate_authorization(root, c, payload)
        peer_payload = dict(manifest=payload['manifest'],
            startup_identity_sha256=auth['approved_startups'][peer['execution_attempt_id']])
        def forbidden(_):
            raise SearchContractError('MATCHED_INITIAL_PEER_CACHE_MISS')
        cache = DurableExactOutputCache(local_path(root, p['cache_directory']), cache_context(root, peer, peer_payload))
        broker = RequestBroker(contract=peer, transport=forbidden, arm='A4', seed=81, durable_cache=cache)
        binding = execution_binding(root, peer)
        pattern = GradientClusterProvider(broker, read_json(root / peer['pattern_prompt_path'])['prompt'],
            gradient_provider=PerExampleGradientProvider(broker, read_json(root / peer['gradient_prompt_path'])['prompt']))
        composed = binding.compose(arm='A4', seed=81, solver=BenchmarkSolver(binding.benchmark(), broker),
            reflection=ReflectionProvider(broker), pattern_provider=pattern, run_root=run_root / 'initial_peer')
        composed.state.initialize()
        peer_state = composed.state.snapshot()
        if (list(peer_state.member_scores) != value['member_scores']
                or dict(peer_state.team_scores) != value['team_scores']
                or digest(plain(peer_state.diagnostics['raw_profiles'])) != value['profile_sha256']
                or broker.usage['attempts'] != 0):
            raise SearchContractError('MATCHED_INITIAL_PEER_MEASUREMENT_MISMATCH')
        atomic_write_json(run_root / 'initial_peer_state_private.json', plain(peer_state))
        if baseline.exists():
            raise SearchContractError('MATCHED_INITIAL_BASELINE_ALREADY_FROZEN')
        atomic_write_json(baseline, sealed)
    elif not baseline.is_file() or read_json(baseline) != sealed:
        raise SearchContractError('MATCHED_INITIAL_BASELINE_MISMATCH')
    state_file = run_root / 'initial_state_private.json'
    state_sha = hashlib.sha256(state_file.read_bytes()).hexdigest()
    receipt = run_root / 'initial_canary_review.json'
    atomic_write_json(run_root / 'INITIAL_CANARY_REVIEW_REQUIRED.json', dict(
        initial_state_sha256=state_sha, paired_baseline_sha256=digest(sealed),
        startup_identity_sha256=payload['startup_identity_sha256'],
        timeout_seconds=p['canary_review_timeout_seconds'], provider_calls_for_review=0))
    deadline = time.monotonic() + p['canary_review_timeout_seconds']
    while not receipt.is_file():
        if time.monotonic() >= deadline:
            raise SearchContractError('INITIAL_CANARY_REVIEW_TIMEOUT')
        time.sleep(1)
    review = read_json(receipt)
    expected = dict(initial_state_sha256=state_sha, paired_baseline_sha256=digest(sealed),
        startup_identity_sha256=payload['startup_identity_sha256'], approved_for_search=True,
        visible_solution_useful=True, final_payload_parser_confirmed=True,
        solver_wire_confirmed=True, thinking_control_confirmed=True)
    if (any(review.get(k) != v for k, v in expected.items())
            or any(review.get(k) is not True for k in ('approved_for_search',
                'visible_solution_useful', 'final_payload_parser_confirmed',
                'solver_wire_confirmed', 'thinking_control_confirmed'))
            or not isinstance(review.get('reviewed_example_ids'), list)
            or not review['reviewed_example_ids']):
        raise SearchContractError('INITIAL_CANARY_REVIEW_MISMATCH')
