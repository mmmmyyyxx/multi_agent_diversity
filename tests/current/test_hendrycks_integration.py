"""Current full-chain metric routing, cache separation and archival boundaries."""
from dataclasses import asdict
from types import SimpleNamespace
import json
import pytest
from multi_dataset_diverse_rl.math_scoring import CONTRACT_ID,contract_sha256,RetiredMathScorerError
from multi_dataset_diverse_rl.math_scoring.prediction import cache_identity
from multi_dataset_diverse_rl.benchmarks.math import MATHBenchmarkAdapter
from multi_dataset_diverse_rl.benchmarks.math_domain_v2 import MATHBenchmarkAdapterV2
from multi_dataset_diverse_rl.benchmarks.math_flexible_answer import classify_prediction,prediction_from_persisted
from multi_dataset_diverse_rl.benchmarks.math_prediction_validity import resolve_predictions
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.benchmarks.protocols import protocol_input
from multi_dataset_diverse_rl.research.single_prompt_runtime import StudyBroker
from multi_dataset_diverse_rl.persistence.exact_output_cache import DurableExactOutputCache,FIELDS

@pytest.mark.parametrize('raw,gold,correct',[(r'\boxed{0.5}',r'\frac{1}{2}',True),(r'\boxed{x+x}','2x',False),('Final answer: 2','2',False)])
def test_member_feedback_and_single_prompt_all_phases_share_core(monkeypatch,raw,gold,correct):
    import multi_dataset_diverse_rl.benchmarks.math_domain_v2 as domain
    monkeypatch.setattr(domain,'domain_matrix',lambda *args:(_ for _ in ()).throw(AssertionError('SYMBOLIC_CORRECTNESS_FORBIDDEN')))
    for adapter in (MATHBenchmarkAdapter(),MATHBenchmarkAdapterV2()):
        item=protocol_input('math','synthetic',{'problem':'Synthetic question'},adapter.output_contract,protocol=adapter.protocol)
        parsed=adapter.parse_member_output(raw,item)
        assert bool(adapter.score_member_output(parsed,gold)) is correct
        assert adapter.build_task_feedback(parsed,gold)==('correct' if correct else 'incorrect')
    for phase in ('train','validation','test'):
        broker=StudyBroker.__new__(StudyBroker);broker.phase=phase
        pred=classify_prediction(raw)
        assert broker.score_resolved(resolve_predictions((pred,)*(1 if pred.prediction_valid else 4)),{'reference_final_answer':gold}) is correct

def test_current_team_plurality_retains_aggregation_scoring_mismatch():
    from multi_dataset_diverse_rl.search.scientific_aggregation import EquivalencePluralityAggregation
    adapter=MATHBenchmarkAdapter()
    item=protocol_input('math','synthetic',{'problem':'Synthetic question'},adapter.output_contract,protocol=adapter.protocol)
    outputs=(r'\boxed{x+x}',r'\boxed{2x}',r'\boxed{2x}',r'\boxed{3}',r'\boxed{4}')
    aggregation=EquivalencePluralityAggregation()
    result=aggregation.aggregate_sync(item=item,member_outputs=outputs,benchmark=adapter)
    assert result.diagnostics['winner_member']==0
    assert adapter.score_member_output(result.parsed_output,'2x')==0
    swapped=(outputs[1],outputs[0],*outputs[2:])
    selected=aggregation.aggregate_sync(item=item,member_outputs=swapped,benchmark=adapter)
    assert adapter.score_member_output(selected.parsed_output,'2x')==1

def test_old_records_and_old_cache_scope_are_rejected_before_reuse(tmp_path):
    pred=asdict(classify_prediction(r'\boxed{2}'))
    assert prediction_from_persisted(pred).prediction_valid
    del pred['math_scoring_contract'];del pred['math_scoring_sha256']
    with pytest.raises(SearchContractError):prediction_from_persisted(pred)
    context={field:'frozen-'+field for field in FIELDS};cache=DurableExactOutputCache(tmp_path/'cache',context)
    assert cache.context['math_scoring_contract']==CONTRACT_ID
    assert cache.context['math_scoring_sha256']==contract_sha256()
    (tmp_path/'cache/scope.json').write_text(json.dumps(context))
    with pytest.raises(SearchContractError,match='SCIENTIFIC_ATTEMPT'):DurableExactOutputCache(tmp_path/'cache',context)

def test_old_single_prompt_scope_fails_before_transport_or_ledger():
    with pytest.raises(RetiredMathScorerError):StudyBroker(phase='test',scope={'scope_identity':'historical'},transport=None,ledger=None,receipts=None,dispatch_guard=None)

def test_old_current_team_binding_fails_before_provider():
    from tests.current.population_support import contract
    from multi_dataset_diverse_rl.search.provider_runtime import RequestBroker
    current=contract();assert current['math_scoring_contract']==CONTRACT_ID
    del current['math_scoring_contract']
    with pytest.raises(SearchContractError,match='FRESH_CONTRACT'):RequestBroker(contract=current,transport=None,arm='A4',seed=42)

@pytest.mark.parametrize('module,function',[('math_worker','evaluate'),('math_power_domain','power_matrix'),('math_certified_domain','certified_matrix')])
def test_retired_workers_cannot_supply_an_alternate_judge(module,function):
    from importlib import import_module
    target=import_module('multi_dataset_diverse_rl.benchmarks.'+module)
    with pytest.raises(RuntimeError,match='RETIRED'):getattr(target,function)(('2','2'))

def test_legacy_a2_test_execution_is_stopped_before_reading_or_consuming_authorization(tmp_path):
    from multi_dataset_diverse_rl.research.ablations.full_error_test_resume import execute
    with pytest.raises(RuntimeError,match='RETIRED'):execute(tmp_path,tmp_path/'missing',tmp_path/'run')
    assert not (tmp_path/'run').exists()

def test_historical_manifest_is_immutable_metadata_not_current_execution():
    from pathlib import Path
    from multi_dataset_diverse_rl.governance.frozen_documents import validate_published_single_prompt
    from multi_dataset_diverse_rl.research.ablations.full_error_control import build_plan
    root=Path(__file__).resolve().parents[2]
    path='experiments/manifests/single_prompt_full_error_control_v2.json'
    old=json.loads((root/path).read_bytes())
    assert validate_published_single_prompt(root,path,old)
    with pytest.raises(SearchContractError):build_plan(old)
    changed=json.loads(json.dumps(old));changed['runtime'].update(cache_identity())
    with pytest.raises(ValueError,match='HISTORICAL_MANIFEST_CHANGED'):validate_published_single_prompt(root,path,changed)

@pytest.mark.parametrize('name',['gradient_pattern_hendrycks_v1','all_errors_hendrycks_v1'])
def test_new_drafts_preflight_without_authorization_or_data_access(name):
    from pathlib import Path
    import importlib.util
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('scoring_migration_entrypoint',root/'scripts/run_experiment.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    manifest=json.loads((root/('configs/'+name+'.json')).read_bytes())
    result=module.preflight(manifest)
    assert result['status']=='HENDRYCKS_DRAFT_PREFLIGHT_NOT_AUTHORIZED'
    assert result['math_scoring_contract']==CONTRACT_ID and result['real_api_calls']==0
    assert result['ready_to_run'] is False

def test_complete_box_on_length_finish_is_not_rejected_by_an_extra_gate():
    p=classify_prediction(r'\boxed{2}','length')
    assert p.prediction_valid and resolve_predictions((p,)).semantic_attempt_count==1

def test_reference_control_cache_cannot_import_a_legacy_grade():
    from multi_dataset_diverse_rl.research.single_prompt_runtime import validate_scoring_observation
    p=resolve_predictions((classify_prediction(r'\boxed{2}'),))
    value=dict(correct=True,valid=True,answer='2',prediction=asdict(p))
    assert validate_scoring_observation(value)==value
    value['prediction'].pop('math_scoring_contract')
    value['prediction'].pop('math_scoring_sha256')
    with pytest.raises(SearchContractError,match='HISTORICAL_MATH_CACHE'):validate_scoring_observation(value)

def test_historical_data_parent_grants_no_heldout_or_old_execution_access():
    from pathlib import Path
    from multi_dataset_diverse_rl.research.single_prompt_governance import historical_data_inputs,validate_manifest
    root=Path(__file__).resolve().parents[2]
    old,reader=historical_data_inputs(root)
    with pytest.raises(SearchContractError):validate_manifest(root,old)
    with pytest.raises(SearchContractError):reader.heldout_rows(None)
    assert reader.manifest['counts']==old['data']['counts']
