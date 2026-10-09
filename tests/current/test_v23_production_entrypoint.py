"""Final execution wrapper and frozen memberships with synthetic data/provider.

Only Optimize/Shadow are opened. Paid credentials and network are absent.
All accounting and execution writes use a separate temporary workspace.
"""
import asyncio,json
from pathlib import Path
import pytest
from tests.current.test_optimization_evidence_v23 import response,HYPOTHESIS
from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
from multi_dataset_diverse_rl.governance import autonomous_math,canary_review
from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json,read_json
from multi_dataset_diverse_rl.benchmarks.math_low_cost import read_subsets
from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
from multi_dataset_diverse_rl.search.binary_runtime import CorrectnessExample

ROOT=Path(__file__).resolve().parents[2]
FRESH='experiments/execution_bindings/a4_v23_only_seed81_20261008_attempt1.json'

@pytest.mark.parametrize('uncertain',[False,True])
def test_final_entrypoint_continues_same_attempt_after_both_canary_reviews(tmp_path,monkeypatch,uncertain):
    c=read_json(ROOT/FRESH);binding=execution_binding(ROOT,c);assert not binding.blockers()
    subsets=read_subsets(ROOT,c)['memberships'];adapter=binding.benchmark()
    def synthetic_examples(role):
        return tuple(CorrectnessExample(protocol_input('math',r['stable_example_id'],
            dict(problem=f'Synthetic {role} case {i}: find the result.'),adapter.output_contract,protocol=adapter.protocol),
            '2',f'Synthetic reference operation for case {i}.' if role=='optimize' else None)
            for i,r in enumerate(subsets['pilot_'+role]))
    monkeypatch.setattr(binding,'examples',synthetic_examples)
    optimize=binding.examples('optimize');shadow=binding.examples('shadow')
    answer_by_problem={binding.benchmark().format_input(e.item):(e.reference,i,split)
        for split,examples in [('optimize',optimize),('shadow',shadow)] for i,e in enumerate(examples)}
    for key in ('initial_team_path','pattern_prompt_path','gradient_prompt_path','validation_accounting_metadata_path'):
        p=tmp_path/c[key];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/c[key]).read_bytes())
    atomic_write_json(tmp_path/FRESH,c)
    requests=[]
    def transport(req):
        requests.append(req)
        if req['model']=='qwen3-8b':
            problem=req['messages'][1]['content'].split('\n\n',1)[1]
            answer,i,split=answer_by_problem[problem]
            return response('Synthetic visible calculation and constraint check.\nFINAL_ANSWER: '+(answer if i>=6 else '999999997'))
        if len(req['messages'])==2:
            p=json.loads(req['messages'][1]['content'])
            if 'example' in p:
                return response(json.dumps(dict(disposition='UNCERTAIN' if uncertain else 'ACTIONABLE',
                    observed_failure='The actual written operation differs from the labeled reference.',
                    diagnosis='The observed transformation may omit a constraint.',
                    reusable_correction=None if uncertain else HYPOTHESIS,
                    expected_effect=None if uncertain else 'Retain constraints during transformations.')))
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
    payload=dict(manifest=dict(source_sha='0'*40,execution_binding=dict(path=FRESH)),
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
    assert [r['stage'] for r in reviews]==['INITIAL_SOLVER_PROFILE','FIRST_COMPLETE_OPPORTUNITY']
    assert reviews[0]['audit']['logical_profiles']==300 and reviews[0]['audit']['initial_memory_entries']==5
    assert reviews[1]['audit']['gradient_diagnostics']==6
    assert reviews[1]['audit']['branch_observation']['full']=='NOT_OBSERVED'
    assert len(result['result']['trace'])==5
    assert result['result']['stop_reason']=='OPERATIONAL_OPPORTUNITY_CEILING'
    assert result['pilot_status']=='INCOMPLETE_OPERATIONAL_TRUNCATION'
    assert result['accounting']['authorized_total']==2_000_000 and not result['accounting']['reserved_inflight']
    assert result['validation_calls']==result['test_calls']==0 and result['memory_audit']['initial_memory_entries']==5
    assert len([r for r in requests if r['model']=='qwen3-8b'])==300
    assert result['pattern_gradient_calls']==30 and result['pattern_cluster_calls']==(0 if uncertain else 5)
    assert client.closed and (run_root/'SEARCH_CLOSED_RECEIPT.json').exists()
    with pytest.raises(Exception):asyncio.run(autonomous_math.execute_search(tmp_path,prep,run_root,payload))


def test_missing_owner_review_aborts_before_optimization_and_closes_accounting(tmp_path,monkeypatch):
    """A lost supervisor cannot turn paid initialization into Canary PASS."""
    from types import SimpleNamespace
    from multi_dataset_diverse_rl.search.schemas import SearchContractError
    original_write=canary_review.atomic_write_json
    def unavailable_owner(path,value):
        if not path.name.endswith('.owner_review.json'):
            original_write(path,value)
    monkeypatch.setattr(canary_review,'atomic_write_json',unavailable_owner)
    ticks=iter((0,canary_review.POLICY['maximum_wait_seconds']))
    monkeypatch.setattr(canary_review,'time',SimpleNamespace(monotonic=lambda:next(ticks),sleep=lambda _:None))
    with pytest.raises(SearchContractError,match='^CANARY_OWNER_REVIEW_TIMEOUT$'):
        test_final_entrypoint_continues_same_attempt_after_both_canary_reviews(tmp_path,monkeypatch,False)
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
            contract=dict(canary_review_policy=canary_review.POLICY,execution_attempt_id='synthetic'),
            payload=dict(startup_identity_sha256='a'*64),run_root=tmp_path)
    assert not (tmp_path/'INITIAL_SOLVER_PROFILE.review_pass.json').exists()
