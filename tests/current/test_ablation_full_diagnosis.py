"""Full diagnostic universe, shared formal sampler and bounded Optimizer projection."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from multi_dataset_diverse_rl.research.ablations.config import default_config,batch_config,build_plan
from multi_dataset_diverse_rl.research.ablations import governance as gov,aligned
from multi_dataset_diverse_rl.research.ablations.engine import run_cell
from multi_dataset_diverse_rl.research.ablations.full_pattern import FullDiagnosisPatternModule,project_full
from multi_dataset_diverse_rl.research.ablations.offline_runtime import OfflineAblationBroker
from multi_dataset_diverse_rl.research.ablations.synthetic import SyntheticTransport,synthetic_rows
from multi_dataset_diverse_rl.search.schemas import EvidenceItem,SearchContractError
from multi_dataset_diverse_rl.search.system_prompt import SEED
from multi_dataset_diverse_rl.search import population_evidence as formal
from multi_dataset_diverse_rl.search.optimization_evidence import POLICY
from multi_dataset_diverse_rl.benchmarks.data_freeze import digest
from multi_dataset_diverse_rl.governance.token_accounting import TokenLedger,POLICY_GATED_32_UNCAPPED
from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
from multi_dataset_diverse_rl.persistence.durable_io import read_json
from multi_dataset_diverse_rl.research.single_prompt import batch_ids

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def full(tmp_path):
    config=default_config();config['rounds']=config['runtime']['rounds']=1
    plan=build_plan(config);scope=dict(scope_identity=digest('FULL_FIXTURE'),call_ceiling=216900,
        token_ceiling=None,network_recovery=True,generation_recovery=config['runtime']['diagnosis_recovery'])
    ledger=TokenLedger(tmp_path/'ledger',task_sha256=scope['scope_identity'],policy=POLICY_GATED_32_UNCAPPED)
    transport=SyntheticTransport()
    broker=OfflineAblationBroker(plan=plan,scope=scope,phase='train',transport=transport,ledger=ledger,
        receipts=ProviderResponseReceipts(tmp_path/'receipts',attempt_id='fake',startup_identity_sha256=scope['scope_identity']),
        dispatch_guard=lambda _:None)
    cell=next(c for c in plan['cells'] if c['arm_id']=='C')
    rows=synthetic_rows('optimize',8)
    for n,row in enumerate(rows):row['content'].update(type=['algebra','geometry','number'][n%3],level='Level 1')
    broker.bind_cell(cell,plan)
    yield broker,transport,plan,cell,rows
    assert not ledger.inflight
    ledger.close()


def run_full(full):
    b,t,plan,cell,rows=full;artifacts={}
    result=run_cell(plan,cell,SEED.to_dict(),rows,synthetic_rows('validation',4),b,
        lambda name,value:artifacts.update({name:deepcopy(value)}))
    return result,artifacts


def test_more_than_three_errors_all_diagnosed_before_cluster_then_only_three_examples(full):
    b,t,_,cell,rows=full;original=t.for_role
    def respond(role,request):
        value=original(role,request)
        if role=='pattern_cluster':
            ids=[g['example_id'] for g in json.loads(request['messages'][-1]['content'])['gradients']]
            assert len(ids)==8 # beyond the former diagnostic Batch
            value['text']=json.dumps(dict(patterns=[dict(generalized_gradient='Check boundary conditions before simplifying.',support_ids=ids[:2]),
                dict(generalized_gradient='Verify algebraic transformations preserve the original constraints.',support_ids=ids[2:])],unassigned_ids=[]))
        return value
    t.for_role=respond
    result,artifacts=run_full(full)
    gradient_packets=[json.loads(q['messages'][-1]['content']) for role,q in t.calls if role=='pattern_gradient']
    assert {q['example']['example_id'] for q in gradient_packets}=={r['stable_example_id'] for r in rows}
    optimizer=json.loads(next(q for role,q in t.calls if role=='reflection')['messages'][-1]['content'])
    chosen={e['example_id'] for e in optimizer['errors']}
    assert len(chosen)==3 and chosen<={r['stable_example_id'] for r in rows[2:]}
    sf=optimizer['structured_feedback']
    assert len(sf['patterns'])==1 and len(sf['patterns'][0]['support_ids'])==6
    assert {g['example_id'] for g in sf['gradients']}==chosen
    assert {d['example_id'] for d in sf['diagnostics']}==chosen
    assert sf['partition_audit']==sf['unassigned_ids']==sf['cluster_structural_errors']==[]
    audit=artifacts['round_1_module']['audit']
    assert len(audit['complete_diagnosis']['structured_feedback']['diagnostics'])==8
    projected,focus=project_full(audit['complete_diagnosis'],42,1,audit['sampling_context'])
    from multi_dataset_diverse_rl.research.ablations.generation import optimizer_error_view
    assert [optimizer_error_view(e) for e in projected['errors']]==optimizer['errors'] and focus==audit['focus']
    assert result['metrics'][0]['candidate_evaluated'] and result['usage']['pattern_gradient']==8
    assert result['usage']['reflection']==1


@pytest.mark.parametrize('failure',['one_gradient','all_gradients','cluster','all_uncertain'])
def test_full_diagnosis_generation_exhaustion_is_per_item_or_fallback_not_whole_opportunity(full,failure):
    b,t,_,_,rows=full;original=t.for_role;bad=rows[0]['stable_example_id']
    def respond(role,request):
        value=original(role,request)
        if role=='pattern_gradient':
            ident=json.loads(request['messages'][-1]['content'])['example']['example_id']
            if failure=='all_uncertain':
                value['text']=json.dumps(dict(disposition='UNCERTAIN',observed_failure='Reference and prediction differ.',diagnosis='Insufficient evidence to locate the failure.',reusable_correction=None,suggested_block=None,expected_effect=None))
            elif failure=='all_gradients' or failure=='one_gradient' and ident==bad:
                value.update(finish_reason='length',provider_usage_details={})
        elif role=='pattern_cluster' and failure=='cluster':value.update(finish_reason='length',provider_usage_details={})
        return value
    t.for_role=respond
    result,artifacts=run_full(full)
    audit=artifacts['round_1_module']['audit'];diagnosis=audit['complete_diagnosis']['structured_feedback']
    if failure=='one_gradient':
        assert len(diagnosis['gradients'])==7
        assert all(bad not in p['support_ids'] for p in diagnosis['patterns'])
        assert all(g['example_id']!=bad for g in diagnosis['gradients'])
        assert result['usage']['pattern_gradient']==10
    else:
        assert audit['focus']['fallback_to_B']
        ids=batch_ids(tuple(r['stable_example_id'] for r in rows),42,1,3)
        opportunity=artifacts['round_1_opportunity']
        assert tuple(e['example_id'] for e in opportunity['errors'])==ids
        assert opportunity['structured_feedback'] is None
        if failure in ('all_gradients','all_uncertain'):assert result['usage']['pattern_cluster']==0
        if failure=='all_uncertain':assert result['usage']['pattern_gradient']==8
        if failure=='cluster':assert result['usage']['pattern_cluster']==3
    assert result['metrics'][0]['candidate_evaluated'] and result['usage']['reflection']==1
    assert not b.network_abort.is_set()


def test_correct_items_never_enter_diagnosis(full):
    b,t,_,_,rows=full;original=t.for_role
    def respond(role,request):
        value=original(role,request)
        if role=='solver' and request['messages'][1]['content']==rows[0]['content']['problem']:
            value['text']='Final answer: 0'
        return value
    t.for_role=respond;_,artifacts=run_full(full)
    diagnoses=artifacts['round_1_module']['audit']['complete_diagnosis']['structured_feedback']['diagnostics']
    assert len(diagnoses)==7 and rows[0]['stable_example_id'] not in {d['example_id'] for d in diagnoses}


def test_actual_formal_controller_and_ablation_use_same_sampler(monkeypatch,full):
    from types import SimpleNamespace
    b,t,_,_,_=full
    actual=formal.sample_pattern_support;calls=[]
    def record(rows,supports,**kwargs):
        calls.append(kwargs);return actual(rows,supports,**kwargs)
    monkeypatch.setattr(formal,'sample_pattern_support',record)
    # Team call has the exact pre-refactor salt and evidence ordering.
    rows=tuple(EvidenceItem(str(n),'optimize',frozenset({'pattern'}),dict(target_member_correct=False,
        task_metadata=dict(type=str(n%3)))) for n in range(8))
    state=SimpleNamespace(team_state_id='measured-team-state',member_prompts=(SEED,))
    context=dict(patterns=[dict(pattern_id='p',support_ids=[r.example_id for r in rows])],focus_mechanism_id='p')
    view,_=formal.PatternSupportEvidence(42,POLICY).compose_opportunity(state,None,0,rows,context,ordinal=4)
    assert calls==[dict(seed=42,state_id='measured-team-state',member_id=0,ordinal=4)]
    assert view.repair_examples==actual(rows,[r.example_id for r in rows],**calls[0])
    from multi_dataset_diverse_rl.research.ablations import full_pattern
    assert full_pattern.sample_pattern_support is actual


def test_new_identity_budgets_defaults_and_historical_manifest_preservation():
    plan=build_plan(default_config());historical=read_json(ROOT/gov.MANIFEST)
    assert plan['training_semantic_call_ceiling']==216900
    assert plan['final_test_semantic_call_ceiling']==10800
    assert plan['module_bindings']['C']['semantic_calls_per_opportunity']==453
    assert len(plan['cells'])==9 and {c['seed'] for c in plan['cells']}=={42}
    assert default_config()['arms'][2]==dict(arm_id='C',error_source='all_wrong',module='full_gradient_pattern')
    assert build_plan(batch_config())['plan_identity']!=historical['plan']['plan_identity']
    assert historical['config']['arms'][2]['error_source']=='random_wrong_batch'
    assert plan['plan_identity']!=historical['plan']['plan_identity']
    assert read_json(ROOT/'experiments/ablations/pattern/default.json')==default_config()
    with pytest.raises(SearchContractError):aligned.validate_manifest(ROOT,read_json(ROOT/aligned.MANIFEST))
    assert not plan['ready_to_run']


def test_random_diagnostic_preselection_cannot_be_enabled_for_full_module():
    config=default_config();config['arms'][2]['error_source']='random_wrong_batch'
    with pytest.raises(SearchContractError,match='MODULE_ERROR_SOURCE'):build_plan(config)


def test_wrong_sampling_metadata_cannot_reproject(full):
    _,artifacts=run_full(full);audit=artifacts['round_1_module']['audit']
    bad=deepcopy(audit['sampling_context']);bad['task_metadata'].pop(next(iter(bad['task_metadata'])))
    with pytest.raises(SearchContractError,match='SAMPLING_SOURCE'):
        project_full(audit['complete_diagnosis'],42,1,bad)


def test_full_manifest_cannot_reuse_historical_scope_or_authorization():
    from scripts.run_experiment import preflight
    manifest=read_json(ROOT/aligned.MANIFEST)
    with pytest.raises(SearchContractError):preflight(manifest)
    old=dict(scope_identity=digest('old-scope'),plan=build_plan(batch_config()))
    scope=dict(scope_identity=digest(['new-plan',manifest['plan']['plan_identity']]),plan=manifest['plan'])
    old_auth=gov.authorization_template(old)
    with pytest.raises(SearchContractError,match='AUTHORIZATION_SCOPE'):gov.validate_authorization(scope,old_auth,required=True)


from test_ablation_execution import phase_smoke


@pytest.mark.parametrize('phase_smoke',[True],indirect=True)
def test_full_diagnosis_governed_training_closure_and_final_test(phase_smoke):
    root,prep,train,scope,transport=phase_smoke
    results,seal=gov.verify_training(train,scope)
    assert len(results)==3 and all(r['rounds_completed']==1 for r in results)
    c=next(r for r in results if r['cell']['arm_id']=='C')
    module=read_json(train/c['cell']['cell_id']/'round_1_module.json')
    assert len(module['audit']['complete_diagnosis']['structured_feedback']['diagnostics'])==150
    assert len(module['audit']['focus']['representative_example_ids'])==3
    assert c['metrics'][0]['optimizer_example_count']==3
    assert read_json(train/'access.json')['test_access'] is False
    test_root=train.parent/'test_run'
    gov._execute_phase(root,prep,test_root,gov.final_test_scope(scope,seal),train_scope=scope,
        parent_results=results,terminal_sha=seal)
    assert read_json(test_root/'lifecycle.json')['status']=='EXECUTION_COMPLETE'
    assert gov.verify_training(train,scope)[1]==seal



def test_fewer_than_three_supports_are_not_padded(full):
    _,artifacts=run_full(full);audit=artifacts['round_1_module']['audit']
    feedback=deepcopy(audit['complete_diagnosis']);sf=feedback['structured_feedback']
    ident=feedback['errors'][0]['example_id']
    sf['patterns'][0]['support_ids']=[ident]
    sf['unassigned_ids']=[e['example_id'] for e in feedback['errors'][1:]]
    projected,focus=project_full(feedback,42,1,audit['sampling_context'])
    assert [e['example_id'] for e in projected['errors']]==[ident]
    assert len(projected['structured_feedback']['gradients'])==1
    assert focus['representative_example_ids']==[ident]


def test_full_freeze_and_exact_fresh_grant_checks_without_real_authorization(tmp_path,monkeypatch):
    # The archived module-alignment manifest cannot freeze the new Guard identity.
    prep=ROOT/'runs/ablations/verification_full_diagnosis_v5'/('structural_reject_'+tmp_path.name)/'prep'
    from multi_dataset_diverse_rl.governance.frozen_documents import validate_published_single_prompt
    frozen=read_json(ROOT/aligned.MANIFEST)
    assert validate_published_single_prompt(ROOT,aligned.MANIFEST,frozen)
    with pytest.raises(SearchContractError):aligned._scope(ROOT,prep)
    assert not prep.exists()



def test_same_seed_prompt_round_and_support_do_not_add_a_repeat_salt(full):
    _,artifacts=run_full(full);audit=artifacts['round_1_module']['audit']
    same=deepcopy(audit['sampling_context']);same['cell_id']=same['cell_id'].replace('repeat01','repeat03')
    left,lf=project_full(audit['complete_diagnosis'],42,1,audit['sampling_context'])
    right,rf=project_full(audit['complete_diagnosis'],42,1,same)
    assert left==right and lf==rf


def test_formal_diagnostic_capacity_exceeds_batch_but_is_bounded():
    from multi_dataset_diverse_rl.research.ablations.interfaces import OptimizeContext
    module=FullDiagnosisPatternModule()
    assert module.binding()['semantic_calls_per_opportunity']==453
    context=OptimizeContext('C',42,1,1,SEED,synthetic_rows('optimize',151),{})
    with pytest.raises(SearchContractError,match='OPTIMIZE_BOUND'):module.prepare(context,None,object())
