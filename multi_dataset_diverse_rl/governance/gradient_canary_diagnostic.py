"""One authorized post-hoc request over an already closed Canary's raw failures.

This tool loads the frozen historical prompt as data. It cannot compose an old
search treatment, generate candidates, or mutate the active run or its Memory.
"""
from hashlib import sha256
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unicodedata

from .autonomous_math import create_transport, initial_prompts, inventory
from .token_accounting import TokenLedger, POLICY_40M, OperationalAbort, serialized_request
from .startup_identity import canonical_sha256
from ..benchmarks.math_accounting_prep import ValidationReserve
from ..benchmarks.math_solver_decoding import generation_request_fields
from ..benchmarks.math_optimizer_diagnostics import optimizer_response_telemetry
from ..persistence.durable_io import read_json, atomic_write_json
from ..search.schemas import SearchContractError

ROLE = 'diagnostic_raw_pattern'
MODE = 'POST_HOC_MATCHED_DIAGNOSTIC_ONLY'


def matched_examples(active_run):
    summary = read_json(active_run / 'execution_summary.json')
    if read_json(active_run / 'lifecycle.json')['status'] != 'EXECUTION_COMPLETE':
        raise SearchContractError('DIAGNOSTIC_REQUIRES_CLOSED_ACTIVE_CANARY')
    records = [json.loads(line) for line in (active_run / 'provider_trace_private.jsonl').read_text(encoding='utf-8').splitlines()]
    gradients = [r for r in records if r['role'] == 'pattern_gradient' and 'response' in r]
    examples = [json.loads(r['request']['messages'][1]['content'])['example'] for r in gradients]
    procedures = {json.loads(r['request']['messages'][1]['content'])['current_member_procedure'] for r in gradients}
    ids = [e['example_id'] for e in examples]
    if not ids or len(procedures) != 1 or len(set(ids)) != len(ids) or len(ids) != summary['pattern_gradient_calls']:
        raise SearchContractError('DIAGNOSTIC_WRONG_UNIVERSE_MISMATCH')
    return examples


def partition_metrics(patterns, unassigned, universe_size):
    sizes = [len(p['support_ids']) for p in patterns]
    assigned = sum(sizes)
    shared = sum(n for n in sizes if n >= 2)
    weights = [n / assigned for n in sizes] if assigned else []
    return dict(pattern_count=len(sizes), singleton_count=sizes.count(1),
        singleton_fraction=sizes.count(1) / len(sizes) if sizes else 0,
        non_singleton_pattern_count=sum(n >= 2 for n in sizes), shared_support_count=shared,
        shared_support_fraction=shared / universe_size, max_pattern_support=max(sizes, default=0),
        assigned_fraction=assigned / universe_size, unassigned_fraction=len(unassigned) / universe_size,
        normalized_entropy=-sum(w * math.log(w) for w in weights) / math.log(len(weights)) if len(weights) > 1 else 0,
        support_distribution=sorted(sizes, reverse=True))


def parse_raw_partition(value, aliases, examples=None):
    # Frozen V3 structural identity merges only byte-normalized descriptions,
    # never speculative paraphrases. No focus is selected in this diagnostic.
    if not isinstance(value, dict) or set(value) != {'patterns', 'unassigned_ids'} or not isinstance(value['patterns'], list):
        raise SearchContractError('DIAGNOSTIC_PARTITION_SCHEMA_INVALID')
    used = set()
    grouped = {}
    def decode(ids):
        if not isinstance(ids, list) or any(not isinstance(x, str) or x not in aliases for x in ids) or len(set(ids)) != len(ids):
            raise SearchContractError('DIAGNOSTIC_PARTITION_MEMBERSHIP_INVALID')
        return [aliases[x] for x in ids]
    for item in value['patterns']:
        required = {'failure_mechanism', 'update_direction', 'support_ids'}
        optional = {'confidence', 'importance', 'priority', 'ranking'}
        if not isinstance(item, dict) or not required <= set(item) or set(item) - required - optional:
            raise SearchContractError('DIAGNOSTIC_PARTITION_SCHEMA_INVALID')
        support = decode(item['support_ids'])
        if not support or used.intersection(support):
            raise SearchContractError('DIAGNOSTIC_PARTITION_MEMBERSHIP_INVALID')
        used.update(support)
        text = [item[k] for k in ('failure_mechanism', 'update_direction')]
        if any(not isinstance(v, str) or not v.strip() or len(v) > 600 for v in text):
            raise SearchContractError('DIAGNOSTIC_PARTITION_ABSTRACTION_INVALID')
        if examples is not None:
            from ..search.pattern_primitives import guard_abstraction
            from ..current_contract import PATTERN_SPECIFIC_CONTENT_GUARD_VERSION
            rows = [SimpleNamespace(example_id=e['example_id'], signals=dict(input_payload=e['problem'], gold=e['reference'])) for e in examples]
            for description in text:
                guard_abstraction(description, rows, abstraction_guard_version=PATTERN_SPECIFIC_CONTENT_GUARD_VERSION)
        normalized = [' '.join(unicodedata.normalize('NFKC', v).casefold().split()) for v in text]
        key = sha256(json.dumps(normalized, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        if key not in grouped:
            grouped[key] = dict(pattern_id=key, failure_mechanism=text[0].strip(),
                update_direction=text[1].strip(), support_ids=[])
        grouped[key]['support_ids'].extend(support)
    unassigned = decode(value['unassigned_ids'])
    if used.intersection(unassigned) or used | set(unassigned) != set(aliases.values()):
        raise SearchContractError('DIAGNOSTIC_PARTITION_MEMBERSHIP_INVALID')
    patterns = sorted(grouped.values(), key=lambda p: p['pattern_id'])
    return dict(patterns=patterns, unassigned_ids=unassigned,
        metrics=partition_metrics(patterns, unassigned, len(aliases)))


def execute_diagnostic(root, active_run, destination, authorization_path):
    root, active_run, destination = map(lambda p: Path(p).resolve(), (root, active_run, destination))
    auth = read_json(authorization_path)
    consumed = read_json(active_run / 'consumed_authorization.json')
    scope = auth.get('scope', {})
    if (auth.get('explicit_user_authorized') is not True or auth.get('single_use') is not True
            or auth.get('consumed') is not False or scope.get('role') != ROLE or scope.get('mode') != MODE
            or scope.get('physical_call_ceiling') != 1 or scope.get('cache_hits') != 0
            or scope.get('active_attempt_id') != consumed['scope']['attempt_id']
            or scope.get('active_startup_identity_sha256') != consumed['startup_identity_sha256']
            or destination.exists() or not destination.is_relative_to(root / 'runs')
            or destination.is_relative_to(active_run) or active_run.is_relative_to(destination)):
        raise SearchContractError('EXACT_FRESH_POSTHOC_DIAGNOSTIC_AUTHORIZATION_REQUIRED')
    binding = read_json(root / scope['binding_path'])
    if sha256((root / scope['binding_path']).read_bytes()).hexdigest() != scope['binding_sha256']:
        raise SearchContractError('DIAGNOSTIC_BINDING_HASH_MISMATCH')
    if scope.get('source_sha') != consumed['scope']['source_sha']:
        raise SearchContractError('DIAGNOSTIC_SOURCE_MISMATCH')
    from .unified_execution import execution_identity, verify_source_commit
    verify_source_commit(root, scope['source_sha'], execution_identity(root, binding))
    if sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != scope['diagnostic_module_sha256']:
        raise SearchContractError('DIAGNOSTIC_SOURCE_MISMATCH')
    prompt_path = root / scope['prompt_path']
    if sha256(prompt_path.read_bytes()).hexdigest() != scope['prompt_sha256']:
        raise SearchContractError('DIAGNOSTIC_ARCHIVED_PROMPT_HASH_MISMATCH')
    prompt = read_json(prompt_path)
    if prompt['identity'] != 'set_level_wrong_pattern_discovery_v3':
        raise SearchContractError('DIAGNOSTIC_ARCHIVED_PROMPT_IDENTITY_MISMATCH')
    before = inventory(active_run)
    examples = matched_examples(active_run)
    aliases = {f'e{i}': row['example_id'] for i, row in enumerate(examples, 1)}
    wire_examples = [dict(row, example_id=f'e{i}') for i, row in enumerate(examples, 1)]
    payload = dict(schema=scope['raw_partition_schema'], examples=wire_examples)
    request = dict(model=binding['models']['pattern'],
        messages=[dict(role='system', content=prompt['prompt']),
                  dict(role='user', content=json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True))],
        **generation_request_fields(binding, 'pattern'))
    if scope['model'] != request['model'] or scope['generation_policy'] != binding['optimizer_generation_policy']:
        raise SearchContractError('DIAGNOSTIC_GENERATION_SCOPE_MISMATCH')
    if len(serialized_request(request)) + 1024 + 1810 > 1_000_000:
        raise SearchContractError('DIAGNOSTIC_CONTEXT_CEILING')
    marker = root / 'runs/posthoc_diagnostic_consumption' / (canonical_sha256(scope) + '.json')
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(dict(scope=scope, consumed=True), stream, sort_keys=True)
    destination.mkdir(parents=True)
    atomic_write_json(destination / 'consumed_authorization.json', dict(auth, consumed=True))
    atomic_write_json(destination / 'request_private.json', dict(request=request, aliases=aliases))
    summary = dict(mode=MODE, role=ROLE, physical_calls=0, cache_hits=0, active_run_unchanged=False,
        validation_calls=0, test_calls=0, status='NOT_COMPARABLE', wrong_universe_size=len(examples))
    transport = client = ledger = None
    reservation_id = None
    try:
        ledger = TokenLedger(root / binding['token_ledger_directory'], task_sha256=binding['task_authorization_sha256'], policy=POLICY_40M)
        atomic_write_json(destination / 'accounting_start.json', ledger.view())
        reserve = ValidationReserve(read_json(root / binding['validation_accounting_metadata_path']), initial_prompts(root, binding))
        transport, client = create_transport(binding)
        reservation_id = ledger.reserve(request, attempt_id=binding['execution_attempt_id'], stage='posthoc_diagnostic',
            role=ROLE, model=request['model'], protected_validation=reserve.remaining())
        summary['physical_calls'] = 1
        response = transport(request)  # Exactly one physical call, including on failure.
        charge = ledger.reconcile(reservation_id, response, outcome='DIAGNOSTIC_RESPONSE')
        reservation_id = None
        atomic_write_json(destination / 'response_private.json', response)
        telemetry = optimizer_response_telemetry(request, response, binding['optimizer_generation_policy'], binding['optimizer_nonthinking_evidence_policy'])
        summary.update(charge=charge, generation_telemetry=telemetry)
        if (telemetry['nonthinking_evidence_level'] == 'CONTRADICTORY'
                or response.get('finish_reason') != 'stop' or not charge['usage_reliable']):
            raise OperationalAbort('DIAGNOSTIC_PROVIDER_CONFORMANCE_FAILURE')
        partition = parse_raw_partition(json.loads(response['text']), aliases, examples)
        atomic_write_json(destination / 'partition_private.json', partition)
        summary.update(status='COMPARABLE', metrics=partition['metrics'])
    except BaseException as exc:
        summary['error_category'] = type(exc).__name__
        if isinstance(exc, (OperationalAbort, SearchContractError)):
            summary['stop_category'] = str(exc)
        if reservation_id is not None:
            summary['charge'] = ledger.reconcile(reservation_id, getattr(exc, 'token_usage', None), outcome='DIAGNOSTIC_FAILURE_NO_RETRY')
    finally:
        if client is not None:
            client.close()
        if ledger is not None:
            atomic_write_json(destination / 'accounting_end.json', ledger.view())
            ledger.close()
        summary['active_run_unchanged'] = inventory(active_run) == before
        atomic_write_json(destination / 'execution_summary.json', summary)
    if not summary['active_run_unchanged']:
        raise OperationalAbort('DIAGNOSTIC_ACTIVE_EVIDENCE_MUTATED')
    return summary
