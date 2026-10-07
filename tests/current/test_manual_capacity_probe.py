# V2.1 frozen replay assertions; V2.2 current conformance is tested separately.
"""A Solver-only diagnostic must preserve wire semantics and isolate realizations."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_dataset_diverse_rl.diagnostics import manual_capacity as probe
from multi_dataset_diverse_rl.benchmarks.math_solver_decoding import generation_request_fields
from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker, BenchmarkSolver
from multi_dataset_diverse_rl.search.schemas import SearchContractError

ROOT = Path(__file__).resolve().parents[2]
PROFILE = 'experiments/execution_bindings/math_v2_1_gradient_pattern_pilot_offline_profile_v5.json'


def parent():
    return json.loads((ROOT / PROFILE).read_text(encoding='utf8'))


def manifest(c):
    texts = ['Solve the problem.'] + [f'Solve carefully using reusable behavior {name}.' for name in 'ABCDE']
    return dict(identity=probe.IDENTITY, attempt_id='synthetic_manual_attempt1', cache_namespace='synthetic/manual/v1',
        logical_evaluations=720, replicates=[1, 2], Optimize_count=60,
        manual_team=list(probe.PROMPTS[1:]), allowed_roles=['solver'], allowed_splits=['optimize'],
        scientific_method_changed=False, prompt_edit_after_results=False, scope_expansion=False,
        evaluation_order='replicate_then_prompt_then_frozen_optimize_order',
        replacement_member_slots=dict(zip(probe.PROMPTS,(0,0,1,2,3,4))),
        prompts=[dict(prompt_id=pid,prompt_text=text,prompt_sha256=sha256(text.encode()).hexdigest())
            for pid,text in zip(probe.PROMPTS,texts)], model='qwen3-8b',
        solver_request_fields=generation_request_fields(c,'solver'), solver_output_interface=c['solver_output_interface'],
        prediction_validity_policy=c['prediction_validity_policy'], invalid_recovery_policy=c['invalid_recovery_policy'],
        transport_decoding=c['decoding'], successful_provider_call_ceiling=2880,
        transport_attempt_ceiling=2880*(c['decoding']['transport_retries']+1), token_charge_ceiling=20_000_000)


def broker(c, replicate=1, transport=lambda req: None):
    return probe.SolverDiagnosticBroker(contract=c,transport=transport,arm='MANUAL_PROBE',seed=81,
        diagnostic_identity=dict(identity=probe.IDENTITY,prompt_id='P0_BASELINE',replicate=replicate))


def test_wire_is_parent_identical_but_replicate_request_keys_are_independent():
    c=parent();messages=[dict(role='system',content='Frozen interface'),dict(role='user',content='Solve the problem.\n\nSynthetic input')]
    original=RequestBroker(contract=c,transport=lambda r:None,arm='A4',seed=81)
    baseline,_=original._request_identity(role='solver',split='optimize',messages=messages,member_slot=0)
    first,key1=broker(c,1)._request_identity(role='solver',split='optimize',messages=messages,member_slot=0)
    second,key2=broker(c,2)._request_identity(role='solver',split='optimize',messages=messages,member_slot=0)
    assert first==second==baseline
    assert key1!=key2


@pytest.mark.parametrize('role,split',[('reflection','optimize'),('pattern_gradient','optimize'),
    ('pattern_cluster','optimize'),('solver','shadow'),('solver','validation'),('solver','test')])
def test_forbidden_role_or_split_fails_before_transport(role,split):
    calls=[];b=broker(parent(),transport=lambda req:calls.append(req))
    with pytest.raises(SearchContractError,match='ROLE_OR_SPLIT_FORBIDDEN'):
        b.complete(role=role,split=split,stage='forbidden',messages=[])
    assert not calls
    with pytest.raises(SearchContractError,match='CAPABILITY_EXPANSION'):
        b.private_capability()
    with pytest.raises(SearchContractError,match='REBIND'):
        b.with_contract(parent())


@pytest.mark.parametrize('responses,expected',[
    (['FINAL_ANSWER: 2','FINAL_ANSWER: 1'],1),
    (['missing interface','FINAL_ANSWER: 2'],2),
    (['missing interface']*5,4)])
def test_only_frozen_invalid_recovery_can_add_solver_realizations(responses,expected):
    calls=[]
    def transport(req):
        calls.append(deepcopy(req))
        return dict(text=responses[len(calls)-1],input_tokens=2,output_tokens=2,finish_reason='stop')
    b=broker(parent(),transport=transport)
    b.complete(role='solver',split='optimize',stage='synthetic',messages=[dict(role='user',content='Synthetic')],member_slot=0)
    assert len(calls)==expected
    assert all(req==calls[0] for req in calls)


@pytest.mark.parametrize('field,value',[
    ('logical_evaluations',721),('model','other'),('allowed_splits',['optimize','shadow']),
    ('manual_team',list(probe.PROMPTS[:5])),('successful_provider_call_ceiling',2881),
    ('token_charge_ceiling',40_000_001),('prompt_edit_after_results',True),
    ('evaluation_order','adaptive')])
def test_protocol_tampering_fails_closed(field,value):
    c=parent();m=manifest(c);m[field]=value
    with pytest.raises(SearchContractError):probe.validate_manifest(m,c)


def test_prompt_hash_and_parent_generation_semantics_are_bound():
    c=parent();m=manifest(c);probe.validate_manifest(m,c)
    m['prompts'][1]['prompt_text']+=' changed'
    with pytest.raises(SearchContractError,match='PROMPT_HASH'):probe.validate_manifest(m,c)
    m=manifest(c);m['solver_request_fields']['temperature']=0.9
    with pytest.raises(SearchContractError,match='SOLVER_SEMANTICS'):probe.validate_manifest(m,c)


def test_finite_token_ceiling_checks_before_durable_reservation():
    calls=[];ledger=SimpleNamespace(view=lambda:dict(charged_total=10),reserve=lambda *a,**kw:calls.append(1))
    cap=probe.DiagnosticTokenBudget(ledger,start_charge=0,ceiling=11)
    with pytest.raises(probe.OperationalAbort,match='CEILING_PRE_TRANSPORT'):
        cap.reserve(dict(model='qwen3-8b',messages=[],max_tokens=1))
    assert not calls


@pytest.mark.parametrize('mutation',[None,'user_task','source_worktree','scope_budget','auth_source','parent_binding','examples'])
def test_preflight_binds_authority_source_data_prompts_and_exact_budget(tmp_path,monkeypatch,mutation):
    c=parent();m=manifest(c);prep=tmp_path/'runs/prep';prep.mkdir(parents=True)
    parent_path=tmp_path/'parent.json';parent_path.write_text(json.dumps(c),encoding='utf8')
    task=tmp_path/'task.md';task.write_text('Synthetic explicit authority',encoding='utf8')
    source=tmp_path/'runtime.py';source.write_text('pass\n',encoding='utf8')
    (prep/'examples_private.json').write_text('[]',encoding='utf8')
    (prep/'manifest_private.json').write_text(json.dumps(m),encoding='utf8')
    payload=dict(source_sha='0'*40,parent_binding_path='parent.json',parent_binding_sha256=sha256(parent_path.read_bytes()).hexdigest(),
        user_task_path='task.md',user_task_sha256=sha256(task.read_bytes()).hexdigest(),
        manifest_sha256=probe.digest(m),source_file_hashes={'runtime.py':probe.source_hash(source)},
        examples_sha256=sha256((prep/'examples_private.json').read_bytes()).hexdigest())
    scope=dict(attempt_id=m['attempt_id'],source_sha=payload['source_sha'],manifest_sha256=payload['manifest_sha256'],
        parent_binding_sha256=payload['parent_binding_sha256'],examples_sha256=payload['examples_sha256'],
        model=m['model'],cache_namespace=m['cache_namespace'],token_charge_ceiling=m['token_charge_ceiling'],
        successful_provider_call_ceiling=m['successful_provider_call_ceiling'],transport_attempt_ceiling=m['transport_attempt_ceiling'],
        allowed_roles=['solver'],allowed_splits=['optimize'],logical_evaluations=720,user_task_sha256=payload['user_task_sha256'],
        prompt_hashes={p['prompt_id']:p['prompt_sha256'] for p in m['prompts']})
    payload['scope']=scope
    auth=dict(scope=scope,source_sha=payload['source_sha'],manifest_sha256=payload['manifest_sha256'])
    if mutation=='user_task':task.write_text('different authority',encoding='utf8')
    if mutation=='source_worktree':source.write_text('changed\n',encoding='utf8')
    if mutation=='scope_budget':scope['successful_provider_call_ceiling']+=1
    if mutation=='auth_source':auth['source_sha']='1'*40
    if mutation=='parent_binding':parent_path.write_text('{}',encoding='utf8')
    if mutation=='examples':(prep/'examples_private.json').write_text('[1]',encoding='utf8')
    (prep/'prep.json').write_text(json.dumps(payload),encoding='utf8')
    (prep/'authorization.json').write_text(json.dumps(auth),encoding='utf8')
    def frozen_git(args,**kwargs):
        return 'commit\n' if args[1]=='cat-file' else b'pass\n'
    monkeypatch.setattr(probe.subprocess,'check_output',frozen_git)
    if mutation is None:
        assert probe.validate_prep(tmp_path,prep)[0]==payload
    else:
        with pytest.raises(SearchContractError):probe.validate_prep(tmp_path,prep)


@pytest.mark.parametrize('fail_after',[None,1])
def test_complete_fresh_diagnostic_or_operational_abort_preserves_evidence(tmp_path,monkeypatch,fail_after):
    from multi_dataset_diverse_rl.benchmarks.legacy.current_math_domain_binding_v21 import execution_binding
    from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
    from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample
    from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,POLICY
    c=parent();binding=execution_binding(ROOT,c);adapter=binding.benchmark()
    examples=tuple(CorrectnessExample(protocol_input('math',f'synthetic{i}',
        {'problem':f'Synthetic arithmetic {i}.'},adapter.output_contract,protocol=adapter.protocol),'SECRET_GOLD') for i in range(60))
    binding.contract=c;monkeypatch.setattr(binding,'blockers',lambda:())
    def allowed(role):
        assert role=='optimize'
        return examples
    monkeypatch.setattr(binding,'examples',allowed)
    monkeypatch.setattr(probe,'MATHGradientPatternBinding',lambda root,contract:binding)
    c['token_ledger_directory']='runs/synthetic_ledger'
    with TokenLedger(tmp_path/c['token_ledger_directory'],task_sha256=c['task_authorization_sha256'],policy=POLICY) as ledger:
        ledger.amend_authorization_40m(authorization_sha256='0'*64,expected_charged_total=0)
    m=manifest(c);prep=tmp_path/'runs/prep';prep.mkdir(parents=True)
    run=tmp_path/'runs/synthetic_manual_attempt1'
    payload=dict(source_sha='0'*40,manifest_sha256=probe.digest(m),parent_binding_sha256='0'*64,
        scope=dict(run_root='runs/synthetic_manual_attempt1'))
    auth=dict(explicit_user_authorized=True,consumed=False,closed=False,source_sha=payload['source_sha'],
        manifest_sha256=payload['manifest_sha256'],scope=payload['scope'])
    (prep/'examples_private.json').write_text(json.dumps([dict(input_id=e.item.input_id,
        problem=adapter.format_input(e.item),reference=e.reference) for e in examples]),encoding='utf8')
    monkeypatch.setattr(probe,'validate_prep',lambda root,p:(payload,m,c,auth))
    calls=[];closed=[]
    def transport(req):
        if fail_after is not None and len(calls)==fail_after:
            raise ValueError('synthetic operational failure')
        calls.append(deepcopy(req))
        assert 'SECRET_GOLD' not in json.dumps(req)
        return dict(text='FINAL_ANSWER: 2',input_tokens=2,output_tokens=2,finish_reason='stop',
            provider_response_accepted=True,provider_metadata_loss_audited=True,
            provider_reasoning_content_present=False,provider_reasoning_character_count=None,
            provider_usage_details={},provider_thinking_indicators=[])
    monkeypatch.setattr(probe,'create_transport',lambda contract:(transport,SimpleNamespace(close=lambda:closed.append(True))))
    if fail_after is None:
        result=probe.execute(tmp_path,prep,run)
        assert result['logical_evaluations']==result['solver']==result['successes']==len(calls)==720
        profiles=json.loads((run/'profiles_complete_private.json').read_text(encoding='utf8'))
        assert len(profiles)==12
        assert calls[:360]==calls[360:]
        keys=[json.loads(line)['request_sha256'] for line in (run/'ledger.jsonl').read_text(encoding='utf8').splitlines()
              if json.loads(line).get('kind')=='SUCCESS']
        assert len(keys)==len(set(keys))==720
        end=json.loads((run/'accounting_end.json').read_text(encoding='utf8'))
        assert end['charged_total']==2880 and end['reserved_inflight']==0
        assert result['reflection']==result['pattern']==result['validation']==result['test']==0
        assert json.loads((run/'lifecycle.json').read_text(encoding='utf8'))['status']=='EXECUTION_COMPLETE'
    else:
        with pytest.raises(probe.OperationalAbort,match='PROVIDER_TERMINAL_ValueError'):probe.execute(tmp_path,prep,run)
        assert len(calls)==fail_after
        assert json.loads((run/'lifecycle.json').read_text(encoding='utf8'))['status']=='EXECUTION_ABORTED'
    assert closed==[True]
    assert (run/'raw_evidence_inventory.json').is_file()
    assert (prep/'authorization_consumed.json').is_file()
    with pytest.raises(SearchContractError,match='FRESH_RUN_REQUIRED'):probe.execute(tmp_path,prep,run)
