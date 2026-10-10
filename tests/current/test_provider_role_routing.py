"""Offline HTTP routing, wire compatibility and credential isolation."""
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest

from multi_dataset_diverse_rl.benchmarks.math_solver_decoding import generation_request_fields
from multi_dataset_diverse_rl.governance.autonomous_math import create_transport
from multi_dataset_diverse_rl.provider_factory import ProviderClientFactory
from multi_dataset_diverse_rl.provider_routing import route_for_role
from multi_dataset_diverse_rl.search.current_policy import CURRENT_POLICY_BUNDLE
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from tests.current.test_structured_optimization_evidence import contract


def test_all_roles_use_their_own_http_client_and_solver_wire(monkeypatch):
    c = contract()
    calls = []
    clients = []
    creations = []
    def factory(cls, **kwargs):
        profile = kwargs['provider_profile']
        creations.append(kwargs)
        def handler(request):
            calls.append((profile, request.url.host, request.headers['authorization'], json.loads(request.content)))
            return httpx.Response(200, json={'choices':[{'message':{'content':'Synthetic result'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':2,'completion_tokens':3}})
        http = httpx.Client(transport=httpx.MockTransport(handler))
        client = SimpleNamespace(_client=http, close=http.close,
            _prepare_url=lambda path: 'https://' + profile + '.invalid' + path,
            _build_headers=lambda options: {'Authorization':'Bearer synthetic-' + profile})
        clients.append(client)
        return client
    monkeypatch.setattr(ProviderClientFactory, 'from_environment', classmethod(factory))
    transport, owner = create_transport(c)
    try:
        for role in ('solver','reflection','pattern_gradient','pattern_cluster'):
            request = dict(model=route_for_role(c,role)['model'], messages=[], **generation_request_fields(c,role))
            result = transport.for_role(role, request)
            assert result['output_tokens'] == 3
        assert len(creations) == 2
        assert [v['provider_profile'] for v in creations] == ['openlux','lwj']
        assert all(v['max_retries'] == 0 for v in creations)
        assert [v[0] for v in calls] == ['openlux','lwj','lwj','lwj']
        for profile, host, credential, body in calls:
            assert host == profile + '.invalid' and credential == 'Bearer synthetic-' + profile
            assert 'extra_body' not in body
            if profile == 'openlux':
                assert body['model'] == 'gpt-4o-mini' and body['max_tokens'] == 3600
                assert not {'top_k','min_p','enable_thinking'} & body.keys()
            else:
                assert body['model'] == 'qwen3.7-flash' and body['enable_thinking'] is False and body['top_k'] == 20
        with pytest.raises(SearchContractError, match='ROLE_FORBIDDEN'):
            transport.for_role('unknown', {'model':'gpt-4o-mini'})
        with pytest.raises(SearchContractError, match='MODEL_ROUTE_MISMATCH'):
            transport.for_role('solver', {'model':'qwen3.7-flash'})
        assert len(calls) == 4
    finally:
        owner.close()
    assert all(client._client.is_closed for client in clients)


def test_partial_startup_closes_first_provider(monkeypatch):
    closed = []
    def factory(cls, **kwargs):
        if kwargs['provider_profile'] == 'lwj':
            raise ValueError('SYNTHETIC_MISSING_CREDENTIAL')
        return SimpleNamespace(close=lambda:closed.append('openlux'))
    monkeypatch.setattr(ProviderClientFactory,'from_environment',classmethod(factory))
    with pytest.raises(ValueError,match='SYNTHETIC_MISSING_CREDENTIAL'):
        create_transport(contract())
    assert closed == ['openlux']


@pytest.mark.parametrize('mutation', ['model','provider','route','decoding','old_binding'])
def test_route_mutations_fail_before_runtime(mutation):
    c = deepcopy(contract())
    if mutation == 'model':c['models']['solver']='qwen3-8b'
    elif mutation == 'provider':c['provider']='lwj'
    elif mutation == 'route':c['provider_routing_policy']['routes']['solver']['provider_profile']='lwj'
    elif mutation == 'decoding':c['solver_decoding_policy']['top_k']=20
    else:c['identity']='MATH_GENERATION_RECOVERY_EVIDENCE_BINDING_V4'
    with pytest.raises(SearchContractError):CURRENT_POLICY_BUNDLE.validate_contract(c)


def test_private_openlux_config_respects_environment_and_offline_guard(tmp_path,monkeypatch):
    from multi_dataset_diverse_rl import provider_credentials as credentials
    private = tmp_path/'deployment.json'
    private.write_text(json.dumps(dict(OPENLUX_API_KEY='synthetic-private',OPENLUX_BASE_URL='https://synthetic.invalid/v1')))
    monkeypatch.setattr(credentials,'PRIVATE_CONFIG_PATH',private)
    monkeypatch.delenv('OPENLUX_API_KEY',raising=False)
    monkeypatch.delenv('OPENLUX_BASE_URL',raising=False)
    monkeypatch.setenv('LWJ_DASHSCOPE_API_KEY','synthetic-lwj')
    monkeypatch.setenv('FORMAL_ZERO_API_GUARD_REQUIRED','1')
    assert credentials.resolve_api_key(provider_profile='openlux')[1] == ''
    assert credentials.resolve_base_url(provider_profile='openlux')[1] == ''
    monkeypatch.delenv('FORMAL_ZERO_API_GUARD_REQUIRED')
    assert credentials.resolve_api_key(provider_profile='openlux')[1] == 'synthetic-private'
    assert credentials.resolve_api_key(provider_profile='lwj')[1] == 'synthetic-lwj'
    monkeypatch.setenv('OPENLUX_API_KEY','synthetic-env')
    assert credentials.resolve_api_key(provider_profile='openlux')[1] == 'synthetic-env'
    monkeypatch.delenv('OPENLUX_API_KEY')
    private.unlink()
    assert credentials.resolve_api_key(provider_profile='openlux')[1] == ''


def test_request_identity_and_scope_bind_each_actual_route(monkeypatch):
    from multi_dataset_diverse_rl.search import provider_runtime
    from multi_dataset_diverse_rl.governance.unified_execution import execution_scope
    c = contract()
    identities = []
    original_dumps = json.dumps
    def capture(value, *args, **kwargs):
        if isinstance(value, dict) and 'provider_route' in value:
            identities.append(deepcopy(value))
        return original_dumps(value, *args, **kwargs)
    monkeypatch.setattr(provider_runtime.json, 'dumps', capture)
    broker = provider_runtime.RequestBroker(contract=c, transport=lambda _: None, arm='A4', seed=81)
    hashes = []
    for role in ('solver', 'reflection', 'pattern_gradient', 'pattern_cluster'):
        request, key = broker._request_identity(role=role, split='optimize',
            messages=[dict(role='user', content='Synthetic request')], member_slot=0)
        hashes.append(key)
        identity = identities[-1]
        assert identity['role'] == role
        assert identity['provider_route'] == route_for_role(c, role)
        assert identity['request'] == request and request['model'] == identity['provider_route']['model']
        assert identity['provider_route']['provider_profile'] == ('openlux' if role == 'solver' else 'lwj')
    assert len(set(hashes)) == 4
    manifest = dict(source_sha='0'*40, preregistration_identity='a'*64,
        execution_binding=dict(sha256='b'*64))
    scope = execution_scope(manifest, c)
    assert scope['provider_routing_policy'] == c['provider_routing_policy']
    assert scope['models']['solver'] == 'gpt-4o-mini'
    poisoned = deepcopy(c)
    poisoned['provider_routing_policy']['routes']['pattern_cluster']['provider_profile'] = 'openlux'
    with pytest.raises(SearchContractError, match='ROUTING_MISMATCH'):
        provider_runtime.RequestBroker(contract=poisoned, transport=lambda _: None, arm='A4', seed=81)


def _usage_http_transport(monkeypatch, usage):
    requests = []
    def factory(cls, **kwargs):
        def handler(request):
            requests.append(request.content)
            return httpx.Response(200, json=dict(choices=[dict(message=dict(content='Final answer: 5'),
                finish_reason='stop')], usage=usage))
        http = httpx.Client(transport=httpx.MockTransport(handler))
        return SimpleNamespace(_client=http, close=http.close,
            _prepare_url=lambda path: 'https://synthetic.invalid' + path,
            _build_headers=lambda options: {'Authorization': 'Bearer synthetic'})
    monkeypatch.setattr(ProviderClientFactory, 'from_environment', classmethod(factory))
    transport, owner = create_transport(contract())
    return transport, owner, requests


@pytest.mark.parametrize('usage,reliable', [
    ({'prompt_tokens': 11, 'completion_tokens': 7, 'total_tokens': 18}, True),
    ({'prompt_tokens': 11, 'completion_tokens': 3600}, True),
    ({'prompt_tokens': 11, 'completion_tokens': 7,
      'prompt_tokens_details': {'cached_tokens': 8},
      'completion_tokens_details': {'reasoning_tokens': 0}}, True),
    ({}, False), (None, False),
    ({'prompt_tokens': '11', 'completion_tokens': 7}, False),
    ({'prompt_tokens': True, 'completion_tokens': 7}, False),
    ({'prompt_tokens': -1, 'completion_tokens': 7}, False),
    ({'prompt_tokens': 10**9, 'completion_tokens': 7}, False),
])
def test_openlux_http_usage_reconciles_without_leaks_or_rebilling(tmp_path, monkeypatch, usage, reliable):
    from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger, POLICY_V25_2M, reservation
    from tests.current.test_parallel_solver_execution import broker_fixture
    transport, owner, requests = _usage_http_transport(monkeypatch, usage)
    c = contract()
    messages = [dict(role='system', content='Synthetic instructions'),
        dict(role='user', content='Synthetic Unicode: \u03c0 \u6d4b\u8bd5')]
    try:
        with TokenLedger(tmp_path/'accounting', task_sha256='e'*64, policy=POLICY_V25_2M) as ledger:
            broker = broker_fixture(tmp_path, c, transport, ledger)
            request, _ = broker._request_identity(role='solver', split='optimize', messages=messages, member_slot=0)
            bound = reservation(request)
            result = broker.complete(role='solver', split='optimize', stage='initial', messages=messages, member_slot=0)
            assert result['usage_reliable'] is reliable
            assert len(requests) == 1 and len(requests[0]) == bound['serialized_request_bytes']
            expected = usage['prompt_tokens'] + usage['completion_tokens'] if reliable else bound['amount']
            assert ledger.totals['charged_total'] == expected and ledger.reserved == 0
            assert ledger.totals['fallback_charged'] == (0 if reliable else bound['amount'])
            cached = broker.complete(role='solver', split='optimize', stage='full', messages=messages, member_slot=0)
            assert cached['provider_called'] is False and cached['input_tokens'] == cached['output_tokens'] == 0
            assert len(requests) == 1 and ledger.totals['charged_total'] == expected
            assert len(list((tmp_path/'receipts').glob('*.json'))) == 1
            assert sum(e['kind'] == 'RESERVE' for e in ledger.events) == 1
            assert sum(e['kind'] == 'CHARGE' for e in ledger.events) == 1
    finally:
        owner.close()


def test_openlux_output_cap_violation_is_charged_and_fails_closed(tmp_path, monkeypatch):
    from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger, POLICY_V25_2M, OperationalAbort
    from tests.current.test_parallel_solver_execution import broker_fixture
    transport, owner, requests = _usage_http_transport(monkeypatch, dict(prompt_tokens=11, completion_tokens=3601))
    try:
        with TokenLedger(tmp_path/'accounting', task_sha256='e'*64, policy=POLICY_V25_2M) as ledger:
            broker = broker_fixture(tmp_path, contract(), transport, ledger)
            with pytest.raises(OperationalAbort, match='OUTPUT_CAP_NOT_ENFORCED'):
                broker.complete(role='solver', split='optimize', stage='initial',
                    messages=[dict(role='user', content='Synthetic request')], member_slot=0)
            assert len(requests) == 1 and ledger.reserved == 0
            assert ledger.totals['charged_total'] == ledger.totals['fallback_charged'] > 0
            assert len(list((tmp_path/'receipts').glob('*.json'))) == 1
            assert not broker.cache and not broker.requests_inflight
    finally:
        owner.close()


def test_five_identical_prompts_have_independent_lanes_and_exact_same_attempt_cache(tmp_path):
    from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger, POLICY_V25_2M
    from tests.current.test_parallel_solver_execution import broker_fixture
    from tests.current.test_structured_optimization_evidence import response
    c = contract()
    calls = []
    def transport(request):
        calls.append(deepcopy(request))
        return response('Final answer: ' + str(len(calls)))
    messages = [dict(role='user', content='Same synthetic problem')]
    def complete(broker, member, split='optimize', stage='initial'):
        return broker.complete(role='solver', split=split, stage=stage, messages=messages, member_slot=member)
    with TokenLedger(tmp_path/'accounting', task_sha256='e'*64, policy=POLICY_V25_2M) as ledger:
        broker = broker_fixture(tmp_path, c, transport, ledger)
        results = [complete(broker, m) for m in range(5)]
        assert len(calls) == 5 and all(request == calls[0] for request in calls)
        assert len({r['request_sha256'] for r in results}) == 5
        assert len({r['text'] for r in results}) == 5
        charged = ledger.totals['charged_total']
        gate_broker = broker.private_capability()
        for m in range(5):
            assert complete(gate_broker, m, stage='full')['provider_called'] is False
        assert len(calls) == 5 and ledger.totals['charged_total'] == charged
        shadows = [complete(gate_broker, m, split='shadow', stage='adaptive_gate') for m in range(5)]
        assert len(calls) == 10 and len({r['request_sha256'] for r in shadows + results}) == 10
        assert ledger.reserved == 0
        fresh = deepcopy(c)
        fresh['execution_attempt_id'] = fresh['cache_namespace'] = 'synthetic_other_attempt'
        from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
        with pytest.raises(SearchContractError, match='DURABLE_CACHE_PROVIDER_BINDING_MISMATCH'):
            RequestBroker(contract=fresh, transport=transport, arm='A4', seed=81,
                durable_cache=broker.durable_cache, token_ledger=ledger)
