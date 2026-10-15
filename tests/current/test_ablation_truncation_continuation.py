"""Per-item bounded output recovery and exact paid-prefix continuation (zero API)."""
from copy import deepcopy
import json
from pathlib import Path
from threading import Lock

import pytest

from multi_dataset_diverse_rl.research.ablations import governance as gov, continuation as cont
from multi_dataset_diverse_rl.research.ablations.interfaces import ModuleProvider, OptimizeContext
from multi_dataset_diverse_rl.research.ablations.formal_adapters import discover
from multi_dataset_diverse_rl.research.ablations.pattern import LargestPatternModule
from multi_dataset_diverse_rl.research.ablations.runtime import AblationBroker
from multi_dataset_diverse_rl.research.ablations.synthetic import SyntheticTransport, synthetic_rows
from multi_dataset_diverse_rl.research.ablations.config import build_plan, default_config
from multi_dataset_diverse_rl.research.ablations.offline_runtime import OfflineAblationBroker
from multi_dataset_diverse_rl.benchmarks.data_freeze import digest
from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger, POLICY_GATED_32_UNCAPPED, OperationalAbort
from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json, read_json
from multi_dataset_diverse_rl.research.single_prompt_governance import inventory
from multi_dataset_diverse_rl.search.generation_failures import GeneratedOutputFailure
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.search.system_prompt import SEED
from test_ablation_execution import phase_smoke

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def amended(tmp_path):
    plan = build_plan(default_config())
    scope = dict(scope_identity=digest('amended'), call_ceiling=190440, token_ceiling=None,
        network_recovery=True, generation_recovery=cont.recovery_policy())
    ledger = TokenLedger(tmp_path/'ledger',task_sha256=scope['scope_identity'],policy=POLICY_GATED_32_UNCAPPED)
    transport = SyntheticTransport()
    broker = OfflineAblationBroker(plan=plan,scope=scope,phase='train',transport=transport,ledger=ledger,
        receipts=ProviderResponseReceipts(tmp_path/'receipts',attempt_id='fake',startup_identity_sha256=scope['scope_identity']),
        dispatch_guard=lambda _:None)
    broker.bind_cell(next(c for c in plan['cells'] if c['arm_id']=='C'),plan)
    yield broker,transport
    assert not ledger.inflight
    ledger.close()


def truncated(value):
    value.update(finish_reason='length',provider_usage_details={},provider_reasoning_character_count=None)
    return value


@pytest.mark.parametrize('role',['pattern_gradient','pattern_cluster','reflection'])
def test_missing_witness_length_is_discarded_not_fatal(amended,role):
    b,t=amended; original=t.for_role
    t.for_role=lambda r,q:truncated(original(r,q))
    args=dict(role=role,split='optimize',stage='x',messages=[dict(role='system',content='fixture'),dict(role='user',content='{"gradients":[]}')])
    if role=='reflection':
        assert b.complete(**args)['finish_reason']=='length'
    else:
        with pytest.raises(GeneratedOutputFailure,match='GENERATED_OUTPUT_TRUNCATED'):b.complete(**args)
    assert not b.network_abort.is_set()
    assert b.usage_view()['provider_calls']==1


@pytest.mark.parametrize('evidence',[dict(provider_usage_details=dict(completion_tokens_details=dict(reasoning_tokens=1))),
    dict(provider_reasoning_character_count=1),dict(provider_thinking_indicators=['reasoning_content'])])
def test_length_with_positive_thinking_still_fatal(amended,evidence):
    b,t=amended; original=t.for_role
    def respond(r,q):
        value=truncated(original(r,q));value.update(evidence);return value
    t.for_role=respond
    with pytest.raises(OperationalAbort,match='NONTHINKING'):
        b.complete(role='pattern_gradient',split='optimize',stage='x',messages=[dict(role='user',content='{}')])
    assert b.network_abort.is_set()


def test_length_without_disabled_thinking_wire_is_fatal(amended):
    b,_=amended
    with pytest.raises(OperationalAbort,match='NONTHINKING'):
        b._check_nonthinking(dict(messages=[]),dict(text='x',finish_reason='length'), 'pattern_gradient','k')


@pytest.mark.parametrize('failure',['one','all','cluster','transient'])
def test_native_extraction_filters_exhausted_items_only(amended,failure):
    b,t=amended; original=t.for_role
    rows=synthetic_rows('optimize',3)
    observations=b.evaluate(SEED,rows,'optimize',42,0)
    b.round_index=14
    failed=rows[0]['stable_example_id']; gradient_calls={}
    def respond(r,q):
        value=original(r,q)
        if r=='pattern_gradient':
            ident=json.loads(q['messages'][-1]['content'])['example']['example_id']
            gradient_calls[ident]=gradient_calls.get(ident,0)+1
            if failure=='all' or ident==failed and failure=='one' or failure=='transient' and ident==failed and gradient_calls[ident]==1:
                return truncated(value)
        if r=='pattern_cluster' and failure=='cluster':return truncated(value)
        return value
    t.for_role=respond
    context=OptimizeContext(b.cell_id,42,1,14,SEED,rows,observations)
    provider=ModuleProvider(b,12)
    prepared=LargestPatternModule().prepare(context,None,provider)
    sf=prepared.audit['complete_diagnosis']['structured_feedback']
    clusters=[json.loads(q['messages'][-1]['content']) for r,q in t.calls if r=='pattern_cluster']
    if failure=='one':
        assert gradient_calls[failed]==3
        assert sf['diagnostics'][0]['exhausted']
        assert {g['example_id'] for g in sf['gradients']}=={r['stable_example_id'] for r in rows[1:]}
        assert {g['example_id'] for g in clusters[0]['gradients']}=={'e1','e2'}
        assert {e['example_id'] for e in prepared.errors}=={r['stable_example_id'] for r in rows[1:]}
    elif failure=='all':
        assert not clusters and all(d['exhausted'] for d in sf['diagnostics'])
        assert prepared.structured_feedback is None and len(prepared.errors)==3
    elif failure=='cluster':
        assert len(clusters)==3 and prepared.structured_feedback is None and len(prepared.errors)==3
    else:
        assert gradient_calls[failed]==2 and sf['diagnostics'][0]['recovery_succeeded']
        assert len(prepared.errors)==3
    assert not b.network_abort.is_set()


def test_inherited_exhausted_item_not_called_others_regenerated(amended):
    b,t=amended; rows=synthetic_rows('optimize',3)
    observations=b.evaluate(SEED,rows,'optimize',42,0);b.round_index=14
    failed=rows[0]['stable_example_id']
    counts={'pattern_gradient:'+failed:3,'pattern_gradient:'+rows[1]['stable_example_id']:1}
    b.scope['continuation']=dict(unfinished_module_draws={b.cell_id:{'14':counts}})
    provider=ModuleProvider(b,12)
    prepared=LargestPatternModule().prepare(OptimizeContext(b.cell_id,42,1,14,SEED,rows,observations),None,provider)
    calls=[json.loads(q['messages'][-1]['content'])['example']['example_id'] for r,q in t.calls if r=='pattern_gradient']
    assert failed not in calls and len(calls)==2
    assert provider.draw_counts()['pattern_gradient:'+failed]==3
    assert provider.draw_counts()['pattern_gradient:'+rows[1]['stable_example_id']]==2
    assert len(prepared.errors)==2


def test_continuation_prefix_hydrates_only_solver_and_preserves_costs_then_test(phase_smoke,monkeypatch):
    root,prep,parent,old_scope,transport=phase_smoke
    parent_results=read_json(parent/'results.json')
    scope=deepcopy(old_scope)
    usage=read_json(parent/'execution_summary.json')['usage']
    scope.update(generation_recovery=cont.recovery_policy(),continuation=dict(parent_run=parent.relative_to(root).as_posix(),
        parent_inventory=inventory(parent),request_identity_scope=old_scope['scope_identity'],inherited_usage=usage,
        completed_rounds={r['cell']['cell_id']:r['rounds_completed'] for r in parent_results},
        cell_usage={r['cell']['cell_id']:r['usage'] for r in parent_results},unfinished_module_draws={}))
    scope['execution']=dict(run_root='runs/continued/train_run')
    scope['scope_identity']=digest({k:v for k,v in scope.items() if k!='scope_identity'})
    newprep=root/'runs/continued/prep';train=newprep.parent/'train_run'
    auth=gov.authorization_template(scope);auth.update(approved=True,adaptive_validation_access=True,test_access=True,user_authorization_evidence_sha256=digest('new human'))
    atomic_write_json(newprep/'authorization.json',auth)
    for n in ('prep','handoff','execution_plan'):atomic_write_json(newprep/(n+'.json'),{})
    monkeypatch.setattr(gov,'validate_prep',lambda *a,**k:scope)
    before=len(transport.calls)
    gov._execute_phase(root,newprep,train,scope,train_scope=scope)
    assert len(transport.calls)==before # no Solver or Teacher paid-response replay
    results,seal=gov.verify_training(train,scope)
    assert [r['usage'] for r in results]==[r['usage'] for r in parent_results]
    assert read_json(train/'execution_summary.json')['usage']==usage
    assert read_json(train/'solver_cache_import.json')['teacher_responses_replayed']==0
    assert read_json(train/'solver_cache_import.json')['imported_count']>0
    gov._execute_phase(root,newprep,train.parent/'test_run',gov.final_test_scope(scope,seal),train_scope=scope,parent_results=results,terminal_sha=seal)
    assert gov.verify_training(train,scope)[1]==seal
    atomic_write_json(parent/'tampered.json',{})
    with pytest.raises(SearchContractError,match='SEALED_PARENT_CHANGED'):gov.verify_training(train,scope)


def test_recovery_manifest_is_exact_and_preserves_three_repeat_parent():
    manifest=read_json(ROOT/cont.MANIFEST)
    assert manifest==cont.manifest_template(ROOT)
    cont.validate_manifest(ROOT,manifest)
    assert manifest['generation_recovery']['diagnostic_truncation_exhaustion']=='skip_failed_gradient_item_cluster_successful_items_continue_optimizer'
    assert gov.manifest_template(ROOT)['config']['repeats_per_arm']==3
    changed=deepcopy(manifest);changed['generation_recovery']['gradient_draws_per_example']=4
    with pytest.raises(SearchContractError):cont.validate_manifest(ROOT,changed)


def test_one_failed_gradient_does_not_waste_optimizer_opportunity(amended):
    from multi_dataset_diverse_rl.research.ablations.engine import run_cell
    b,t=amended;original=t.for_role
    config=default_config();config['rounds']=config['runtime']['rounds']=1
    plan=build_plan(config);b.plan=plan
    cell=next(c for c in plan['cells'] if c['arm_id']=='C');rows=synthetic_rows('optimize',3)
    def respond(role,request):
        value=original(role,request)
        if role=='pattern_gradient' and json.loads(request['messages'][-1]['content'])['example']['example_id']==rows[0]['stable_example_id']:
            return truncated(value)
        return value
    t.for_role=respond;artifacts={}
    result=run_cell(plan,cell,SEED.to_dict(),rows,synthetic_rows('validation',3),b,lambda k,v:artifacts.update({k:v}))
    assert result['metrics'][0]['candidate_evaluated']
    assert result['metrics'][0]['decision']=='REJECTED_NO_STRICT_VALIDATION_GAIN'
    assert result['metrics'][0]['optimizer_example_count']==2
    assert result['usage']['reflection']==1
    assert max(artifacts['round_1_module']['audit']['diagnostic_draw_counts'].values())==3
