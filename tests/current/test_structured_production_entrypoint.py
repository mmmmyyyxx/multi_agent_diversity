"""Final execution wrapper and frozen memberships with synthetic data/provider.

Only Optimize/Shadow are opened. Paid credentials and network are absent.
All accounting and execution writes use a separate temporary workspace.
"""
import asyncio,json
from pathlib import Path
import pytest
from tests.current.test_structured_optimization_evidence import response,HYPOTHESIS
from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
from multi_dataset_diverse_rl.governance import autonomous_math,canary_review
from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json,read_json
from multi_dataset_diverse_rl.benchmarks.math_low_cost import read_subsets
from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample

ROOT=Path(__file__).resolve().parents[2]
FRESH='experiments/execution_bindings/a4_v25_seed81_canary_attempt1.json'

def _entrypoint_fixture(tmp_path,monkeypatch,uncertain,phase='pilot',all_correct=False,structural=False,
        seed=81,team_version=None,development_subsets=None,structural_kind='syntax'):
    from hashlib import sha256
    from multi_dataset_diverse_rl.benchmarks.math_structured_binding import derive_structured_contract
    from multi_dataset_diverse_rl.search.optimization_evidence import POLICY
    old=read_json(ROOT/FRESH)
    fresh='runs/synthetic_flexible_binding.json'
    attempt='synthetic_flexible_'+phase
    scope_path='runs/synthetic_flexible_scope.json'
    scope=dict(schema_version='structured_system_preparation_scope_v1',attempt_id=attempt,
        one_attempt_only=True,preparation_only=True,real_api_authorized=False,
        validation_authorized=False,test_authorized=False,optimization_evidence_policy=POLICY,
        parent_binding_sha256=old['trajectory_parent_binding_sha256'])
    atomic_write_json(tmp_path/scope_path,scope)
    c=derive_structured_contract(read_json(ROOT/old['trajectory_parent_binding_path']),
        attempt=attempt,binding_path=fresh,parent_path=old['trajectory_parent_binding_path'],
        parent_sha256=old['trajectory_parent_binding_sha256'],authorization_path=scope_path,
        authorization_sha256=sha256((tmp_path/scope_path).read_bytes()).hexdigest(),
        gradient_prompt_path=old['gradient_prompt_path'],gradient_prompt_sha256=old['gradient_prompt_sha256'],
        pattern_prompt_path=old['pattern_prompt_path'],pattern_prompt_sha256=old['pattern_prompt_sha256'],
        validation_metadata_path=old['validation_accounting_metadata_path'],validation_metadata_sha256=old['validation_accounting_metadata_sha256'],
        initial_team_path=('experiments/initial_teams/math_arm_b_structured_seed_v3.json' if team_version else old['initial_team_path']),
        initial_team_artifact_sha256=(sha256((ROOT/'experiments/initial_teams/math_arm_b_structured_seed_v3.json').read_bytes()).hexdigest() if team_version else old['initial_team_artifact_sha256']),
        accounting_policy_path=old['accounting_policy_path'],accounting_policy_sha256=old['accounting_policy_sha256'],
        execution_phase=phase,max_opportunities=5 if phase=='pilot' else 1,seed=seed,
        **({'team_version':team_version} if team_version else {}),development_subsets=development_subsets)
    binding=execution_binding(ROOT,c)
    original_path=binding.path
    monkeypatch.setattr(binding,'path',lambda relative: tmp_path/relative if relative==scope_path else original_path(relative))
    assert not binding.blockers()
    subsets=read_subsets(ROOT,c)['memberships'];adapter=binding.benchmark()
    def synthetic_examples(role):
        return tuple(CorrectnessExample(protocol_input('math',r['stable_example_id'],
            dict(problem=f'Synthetic {role} case {i}: find the result.'),adapter.output_contract,protocol=adapter.protocol),
            '2',f'Synthetic reference operation for case {i}.' if role=='optimize' else None)
            for i,r in enumerate(subsets[(phase if role=='optimize' else 'pilot')+'_'+role]))
    monkeypatch.setattr(binding,'examples',synthetic_examples)
    optimize=binding.examples('optimize');shadow=binding.examples('shadow')
    answer_by_problem={binding.benchmark().format_input(e.item):(e.reference,i,split)
        for split,examples in [('optimize',optimize),('shadow',shadow)] for i,e in enumerate(examples)}
    for key in ('initial_team_path','pattern_prompt_path','gradient_prompt_path','validation_accounting_metadata_path'):
        p=tmp_path/c[key];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/c[key]).read_bytes())
    atomic_write_json(tmp_path/fresh,c)
    requests=[]
    from collections import Counter
    gradient_draws=Counter();cluster_draws=0
    def transport(req):
        nonlocal cluster_draws
        requests.append(req)
        if req['model']=='qwen3-8b':
            problem=req['messages'][1]['content']
            answer,i,split=answer_by_problem[problem]
            return response('Synthetic visible calculation and constraint check.\nFinal answer: '+(answer if all_correct or i>=6 else '999999997'))
        if len(req['messages'])==2:
            p=json.loads(req['messages'][1]['content'])
            if 'example' in p:
                gradient_draws[p['example']['example_id']]+=1
                if structural and gradient_draws[p['example']['example_id']]==1:return response(None if structural_kind=='null' else 'malformed JSON')
                return response(json.dumps(dict(disposition='UNCERTAIN' if uncertain else 'ACTIONABLE',
                    observed_failure='The actual written operation differs from the labeled reference.',
                    diagnosis='The observed transformation may omit a constraint.',suggested_block='strategy',
                    reusable_correction=None if uncertain else HYPOTHESIS,
                    expected_effect=None if uncertain else 'Retain constraints during transformations.')))
            cluster_draws+=1
            if structural and cluster_draws==1:return response(None if structural_kind=='null' else 'malformed JSON')
            return response(json.dumps(dict(patterns=[dict(generalized_gradient=HYPOTHESIS,
                support_ids=[g['example_id'] for g in p['gradients']])],unassigned_ids=[])))
        return response('{"decision":"NO_SAFE_EDIT"}')
    class Client:
        closed=False
        def close(self):self.closed=True
    client=Client()
    monkeypatch.setattr(autonomous_math,'create_transport',lambda _: (transport,client))
    # Relocate writes while retaining the actual final binding's read-only
    # production data/initial-team ports and conformance checks.
    monkeypatch.setattr(autonomous_math,'execution_binding',lambda root,contract:binding)
    prep=tmp_path/'runs/prep';prep.mkdir(parents=True)
    payload=dict(manifest=dict(source_sha='0'*40,execution_binding=dict(path=fresh)),
        scope=dict(attempt_id=c['execution_attempt_id'],source_sha='0'*40),startup_identity_sha256='a'*64)
    atomic_write_json(prep/'authorization.json',dict(explicit_user_authorized=True,single_use=True,consumed=False,
        scope=payload['scope'],startup_identity_sha256=payload['startup_identity_sha256']))
    original=canary_review.atomic_write_json;reviews=[]
    def reviewed_write(path,value):
        original(path,value)
        if path.name.endswith('.review_pending.json'):
            reviews.append(value)
            original(path.parent/(value['stage']+'.owner_review.json'),dict(approved=True,
                scientific_method_changed=False,review_identity_sha256=value['review_identity_sha256'],
                startup_identity_sha256=value['startup_identity_sha256']))
    monkeypatch.setattr(canary_review,'atomic_write_json',reviewed_write)
    run_root=tmp_path/'runs/execution'
    result=asyncio.run(autonomous_math.execute_search(tmp_path,prep,run_root,payload))
    assert [r['stage'] for r in reviews]==(['INITIAL_SOLVER_PROFILE'] if all_correct else
        ['INITIAL_SOLVER_PROFILE','FIRST_COMPLETE_OPPORTUNITY'])
    initial_profiles=300 if phase=='pilot' else 60
    assert reviews[0]['audit']['logical_profiles']==initial_profiles and reviews[0]['audit']['initial_memory_entries']==5
    if not all_correct:
        assert reviews[1]['audit']['gradient_diagnostics']==6
        assert reviews[1]['audit']['branch_observation']['full']=='NOT_OBSERVED'
    opportunity_count=0 if all_correct else (5 if phase=='pilot' else 1)
    assert len(result['result']['trace'])==opportunity_count
    assert result['result']['stop_reason']==('NO_REPAIR_SIGNAL' if all_correct else
        'OPERATIONAL_OPPORTUNITY_CEILING' if phase=='pilot' else 'CANARY_ONE_PRODUCTION_OPPORTUNITY_COMPLETE')
    if phase=='pilot':
        assert result['pilot_status']=='INCOMPLETE_OPERATIONAL_TRUNCATION'
    assert result['accounting']['authorized_total']==2_000_000 and not result['accounting']['reserved_inflight']
    assert result['validation_calls']==result['test_calls']==0 and result['memory_audit']['initial_memory_entries']==5
    assert len([r for r in requests if r['model']=='qwen3-8b'])==initial_profiles
    assert result['pattern_gradient_calls']==6*opportunity_count*(2 if structural else 1)
    assert result['pattern_cluster_calls']==(0 if uncertain else opportunity_count+(1 if structural else 0))
    assert client.closed and read_json(run_root/'lifecycle.json')['status']=='EXECUTION_COMPLETE'
    if phase=='pilot':
        assert (run_root/'SEARCH_CLOSED_RECEIPT.json').exists()
    with pytest.raises(Exception):asyncio.run(autonomous_math.execute_search(tmp_path,prep,run_root,payload))
    return result,requests,client,reviews


@pytest.mark.parametrize('uncertain',[False,True])
@pytest.mark.parametrize('phase,all_correct',[('pilot',False),('canary',False),('canary',True)])
def test_final_entrypoint_continues_same_attempt_after_canary_reviews(tmp_path,monkeypatch,uncertain,phase,all_correct):
    _entrypoint_fixture(tmp_path,monkeypatch,uncertain,phase,all_correct)


def test_final_entrypoint_accounts_structural_recovery_without_old_single_draw_veto(tmp_path,monkeypatch):
    _entrypoint_fixture(tmp_path,monkeypatch,False,'canary',False,structural=True)


def test_null_gradient_and_cluster_are_durably_charged_local_draws(tmp_path,monkeypatch):
    result,requests,_,_=_entrypoint_fixture(tmp_path,monkeypatch,False,'canary',False,
        structural=True,structural_kind='null')
    assert result['ledger']['successes']==len(requests)
    assert result['accounting']['charged_total']==4*len(requests)
    assert len(list((tmp_path/'runs/execution/provider_response_receipts_private').glob('*.json')))==len(requests)
    assert result['accounting']['reserved_inflight']==0


def test_missing_owner_review_aborts_before_optimization_and_closes_accounting(tmp_path,monkeypatch):
    """A lost supervisor cannot turn paid initialization into Canary PASS."""
    from types import SimpleNamespace
    from multi_dataset_diverse_rl.search.schemas import SearchContractError
    original_write=canary_review.atomic_write_json
    def unavailable_owner(path,value):
        if not path.name.endswith('.owner_review.json'):
            original_write(path,value)
    monkeypatch.setattr(canary_review,'atomic_write_json',unavailable_owner)
    ticks=iter((0,canary_review.STRUCTURED_POLICY['maximum_wait_seconds']))
    monkeypatch.setattr(canary_review,'time',SimpleNamespace(monotonic=lambda:next(ticks),sleep=lambda _:None))
    with pytest.raises(SearchContractError,match='^CANARY_OWNER_REVIEW_TIMEOUT$'):
        _entrypoint_fixture(tmp_path,monkeypatch,False)
    run_root=tmp_path/'runs/execution'
    lifecycle=read_json(run_root/'lifecycle.json')
    assert lifecycle['status']=='EXECUTION_ABORTED'
    assert lifecycle['provider_usage']['solver']==300
    assert lifecycle['provider_usage']['pattern_gradient']==0
    assert lifecycle['provider_usage']['reflection']==0
    assert read_json(run_root/'accounting_end.json')['reserved_inflight']==0
    assert read_json(run_root/'consumed_authorization.json')['consumed'] is True
    assert (run_root/'INITIAL_SOLVER_PROFILE.review_pending.json').exists()
    assert not (run_root/'INITIAL_SOLVER_PROFILE.review_pass.json').exists()
    assert not (run_root/'execution_summary.json').exists()
    assert (run_root/'trajectory_private.jsonl').read_bytes()==b''
    assert (run_root/'raw_evidence_inventory.json').exists()
    assert (run_root/'initial_state_private.json').exists()


@pytest.mark.parametrize('wrong_field',['approved','review_identity_sha256','startup_identity_sha256'])
def test_owner_review_rejects_unapproved_or_stale_receipt(tmp_path,monkeypatch,wrong_field):
    from multi_dataset_diverse_rl.search.schemas import SearchContractError
    original=canary_review.atomic_write_json
    def stale_review(path,value):
        original(path,value)
        if path.name.endswith('.review_pending.json'):
            decision=dict(approved=True,scientific_method_changed=False,
                review_identity_sha256=value['review_identity_sha256'],
                startup_identity_sha256=value['startup_identity_sha256'])
            decision[wrong_field]=False if wrong_field=='approved' else 'b'*64
            original(path.parent/'INITIAL_SOLVER_PROFILE.owner_review.json',decision)
    monkeypatch.setattr(canary_review,'atomic_write_json',stale_review)
    with pytest.raises(SearchContractError,match='^CANARY_OWNER_REVIEW_NOT_APPROVED$'):
        canary_review.review('INITIAL_SOLVER_PROFILE',dict(logical_profiles=300),
            contract=dict(canary_review_policy=canary_review.STRUCTURED_POLICY,execution_attempt_id='synthetic'),
            payload=dict(startup_identity_sha256='a'*64),run_root=tmp_path)
    assert not (tmp_path/'INITIAL_SOLVER_PROFILE.review_pass.json').exists()


@pytest.mark.parametrize('artifact',['raw_evidence_inventory.json','SEARCH_COMPLETE_RECEIPT.json','lifecycle.json'])
def test_terminal_persistence_fault_cannot_publish_completion_or_reopen_scope(tmp_path,monkeypatch,artifact):
    original=autonomous_math.atomic_write_json;failed=[]
    def broken_once(path,value):
        terminal=(path.name!='lifecycle.json' or value.get('status')=='EXECUTION_COMPLETE')
        if path.name==artifact and terminal and not failed:
            failed.append(path.name)
            raise OSError('SYNTHETIC_TERMINAL_STORAGE_FAILURE')
        original(path,value)
    monkeypatch.setattr(autonomous_math,'atomic_write_json',broken_once)
    with pytest.raises(OSError,match='SYNTHETIC_TERMINAL_STORAGE_FAILURE'):
        _entrypoint_fixture(tmp_path,monkeypatch,False,'canary')
    run_root=tmp_path/'runs/execution'
    assert failed==[artifact]
    assert read_json(run_root/'lifecycle.json')['status']=='EXECUTION_ABORTED'
    assert read_json(run_root/'accounting_end.json')['reserved_inflight']==0
    assert read_json(run_root/'consumed_authorization.json')['consumed'] is True
    milestone=read_json(run_root/'scientific_compute_returned_private.json')
    assert milestone['resume_authorized'] is False
    assert (run_root/'final_state_private.json').exists()


def test_resources_and_hash_readback_precede_terminal_marker(tmp_path,monkeypatch):
    events=[];original_close=autonomous_math.TokenLedger.close
    def close(ledger):
        events.append('ledger_close');original_close(ledger)
    monkeypatch.setattr(autonomous_math.TokenLedger,'close',close)
    original_write=autonomous_math.atomic_write_json
    def write(path,value):
        if path.name=='lifecycle.json' and value['status']=='EXECUTION_COMPLETE':
            assert events==['ledger_close']
            receipt=read_json(path.parent/'SEARCH_COMPLETE_RECEIPT.json')
            assert receipt['raw_inventory_sha256']==autonomous_math.file_sha(path.parent/'raw_evidence_inventory.json')
            for record in read_json(path.parent/'raw_evidence_inventory.json')['files']:
                assert autonomous_math.file_sha(path.parent/record['path'])==record['sha256']
            events.append('terminal_marker')
        original_write(path,value)
    monkeypatch.setattr(autonomous_math,'atomic_write_json',write)
    _entrypoint_fixture(tmp_path,monkeypatch,True,'canary',all_correct=True)
    # Fixture also attempts forbidden reuse; its fresh ledger is closed on abort.
    assert events==['ledger_close','terminal_marker','ledger_close']


@pytest.mark.parametrize('phase',['canary','pilot'])
def test_seed84_arm_b_frozen_panels_complete_with_fake_provider_only(tmp_path,monkeypatch,phase):
    from multi_dataset_diverse_rl.benchmarks.math_canary_inputs import ARM_B_TEAM_VERSION
    c=read_json(ROOT/('experiments/execution_bindings/a4_v25_arm_b_recovery_seed84_'+phase+'_attempt1.json'))
    development=dict(path=c['low_cost_subsets_path'],sha256=c['low_cost_subsets_sha256'],protocol=c['low_cost_protocol'])
    result,_,_,_= _entrypoint_fixture(tmp_path,monkeypatch,False,phase,
        seed=84,team_version=ARM_B_TEAM_VERSION,development_subsets=development)
    initial=read_json(tmp_path/'runs/execution/initial_state_private.json')
    assert len(set(json.dumps(p,sort_keys=True) for p in initial['member_prompts']))==1
    assert result['accounting']['reserved_inflight']==0 and result['memory_audit']['initial_memory_entries']==5


@pytest.mark.parametrize('artifact',['final_team_private.json','final_state_private.json'])
def test_corrupt_terminal_checkpoint_readback_remains_fatal(tmp_path,monkeypatch,artifact):
    from multi_dataset_diverse_rl.governance.token_accounting import OperationalAbort
    original=autonomous_math.atomic_write_json
    def corrupt(path,value):
        original(path,{**value,'state_id':'SYNTHETIC_CORRUPTION'} if path.name==artifact else value)
    monkeypatch.setattr(autonomous_math,'atomic_write_json',corrupt)
    with pytest.raises(OperationalAbort,match='EXECUTION_PERSISTENCE_MISMATCH'):
        _entrypoint_fixture(tmp_path,monkeypatch,False,'canary')
    run_root=tmp_path/'runs/execution'
    assert read_json(run_root/'lifecycle.json')['status']=='EXECUTION_ABORTED'
    assert read_json(run_root/'accounting_end.json')['reserved_inflight']==0
    assert read_json(run_root/'consumed_authorization.json')['consumed'] is True
    assert not (run_root/'SEARCH_COMPLETE_RECEIPT.json').exists()


def test_resource_close_failure_cannot_follow_a_false_completion(tmp_path,monkeypatch):
    original=autonomous_math.TokenLedger.close;failed=[]
    def close(ledger):
        original(ledger)
        if not failed:
            failed.append(True);raise OSError('SYNTHETIC_RESOURCE_CLOSE_FAILURE')
    monkeypatch.setattr(autonomous_math.TokenLedger,'close',close)
    with pytest.raises(OSError,match='SYNTHETIC_RESOURCE_CLOSE_FAILURE'):
        _entrypoint_fixture(tmp_path,monkeypatch,True,'canary',all_correct=True)
    run_root=tmp_path/'runs/execution'
    assert read_json(run_root/'lifecycle.json')['status']=='EXECUTION_ABORTED'
    assert (run_root/'scientific_compute_returned_private.json').exists()
    assert not (run_root/'SEARCH_COMPLETE_RECEIPT.json').exists()
