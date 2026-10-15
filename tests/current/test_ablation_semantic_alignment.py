"""Every exposed adapter has explicit semantics; zero provider/network access."""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from multi_dataset_diverse_rl.research.ablations import aligned,governance as gov
from multi_dataset_diverse_rl.research.ablations.config import default_config,batch_config,build_plan
from multi_dataset_diverse_rl.research.ablations.modules import default_registry,ModuleRegistry,NoModule
from multi_dataset_diverse_rl.research.ablations.semantic_alignment import inventory
from multi_dataset_diverse_rl.research.ablations.generation import (
    candidate_checks,optimizer_error_view,optimizer_packet,generate_candidate,audit_optimizer_request)
from multi_dataset_diverse_rl.research.ablations.interfaces import PreparedFeedback,ModuleProvider
from multi_dataset_diverse_rl.research.ablations.formal_adapters import (
    GradientTransportAdapter,ClusterTransportAdapter,production_memory_binding,new_production_memory)
from multi_dataset_diverse_rl.search.reflection_steps import worked_reference,inference_dependency_checks
from multi_dataset_diverse_rl.search.textual_gradients import PerExampleGradientProvider,GradientClusterProvider
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.search.system_prompt import SEED
from multi_dataset_diverse_rl.persistence.durable_io import atomic_write_json,read_json
from multi_dataset_diverse_rl.benchmarks.data_freeze import digest
from test_ablation_full_diagnosis import full
from test_single_prompt_ablations import ScriptedPort

ROOT=Path(__file__).resolve().parents[2]


def error():
    return dict(example_id='fixture',problem='A synthetic problem.',reference='2',
        reference_solution='A private worked derivation requires checking the final substitution carefully.',
        solver_response='The observed response substitutes an unsupported boundary value into the constraint.',
        prediction='3',valid=True,correct=False,invalid_reason=None,semantic_attempt_count=1)


@pytest.mark.parametrize('name',['formal_gradient','memory','largest_pattern'])
def test_unenabled_or_historical_names_cannot_be_a_current_arm(name):
    registry=default_registry()
    with pytest.raises(SearchContractError,match='NOT_REGISTERED'):registry.create(name)
    config=default_config();config['arms'][2].update(module=name,error_source='random_wrong_batch')
    with pytest.raises(SearchContractError):build_plan(config)


def test_legacy_registry_cannot_bypass_current_flow_admission():
    config=default_config();config['arms'][2].update(module='largest_pattern',error_source='random_wrong_batch')
    with pytest.raises(SearchContractError,match='HISTORICAL_MODULE'):
        build_plan(config,default_registry(include_legacy=True))
    assert build_plan(batch_config())==read_json(ROOT/gov.MANIFEST)['plan']


def test_registry_and_complete_native_port_inventory_do_not_claim_memory_or_team_adapters():
    value=inventory()
    assert len(value['team_modules'])==20
    assert len(value['boundary_capabilities'])>=15
    assert value['boundary_capabilities']['memory_storage']['single_prompt_adaptation']=='measurement_adapter_unresolved'
    assert all(v['team_status']=='ACTUAL_FORMAL_COMPONENT' for v in value['team_modules'].values())
    assert value['independent']['formal_gradient']['activation'] is False
    assert value['independent']['memory']['activation'] is False
    assert value['synthetic_team_controls']['memory_delivery'].startswith('delivery_only')
    assert value['synthetic_team_controls']['reflection_generation'].startswith('pipeline_negative')
    plan=build_plan(default_config())
    assert plan['module_alignment']['C']['status']=='FORMAL_WITH_APPROVED_ADAPTATIONS'
    assert plan['module_alignment']['A']['status']==plan['module_alignment']['B']['status']=='APPROVED_CONTROL'
    assert not value['real_api_authorized']


def test_custom_module_is_explicitly_unreviewed_not_promoted_to_real_conformance():
    r=ModuleRegistry();r.register('custom',NoModule)
    config=default_config();config['arms']=[dict(arm_id='X',module='custom',error_source='all_wrong')]
    plan=build_plan(config,r)
    assert plan['module_alignment']['X']['status']=='UNREVIEWED'
    assert plan['execution_blocker']=='MODULE_ALIGNMENT_REVIEW_REQUIRED'


def test_memory_native_core_only_and_truthful_interface_boundaries():
    from multi_dataset_diverse_rl.search.population_memory import PopulationEditMemory,LIMITS
    from multi_dataset_diverse_rl.search.memory_components import MeasuredEditExperienceRecorder,ParentScopedMemoryRetrieval,WholeMeasuredCardDelivery
    memory=new_production_memory();binding=production_memory_binding()
    assert type(memory) is PopulationEditMemory and memory.limits==LIMITS
    assert type(memory.recorder) is MeasuredEditExperienceRecorder
    assert type(memory.retrieval) is ParentScopedMemoryRetrieval
    assert type(memory.delivery) is WholeMeasuredCardDelivery
    assert not binding['runnable'] and binding['user_decision']=='keep_unenabled_interface_audit_only'
    assert not memory.competence and not memory.revision
    assert all(not rows for rows in memory.entries.values())
    assert set(binding['unresolved'])=={'single_prompt_bootstrap_scope','absent_team_vote_and_shadow_fields',
        'which_candidate_outcomes_to_record','optimizer_delivery_contract'}


@pytest.mark.parametrize('source',['reference_solution','solver_response','diagnostic'])
def test_shared_formal_private_evidence_copy_guard(source):
    e=error();sf=None
    if source=='diagnostic':
        text='The observed calculation omits the original domain restriction during the substitution.'
        sf=dict(diagnostics=[dict(observed_failure=text,diagnosis='A domain check is needed.')])
    else:text=e[source]
    child=SEED.edit('strategy',text)
    assert 'private_solution_or_response_copy' in candidate_checks(child,parent=SEED,errors=[e],structured_feedback=sf)


@pytest.mark.parametrize('text',['Read peer responses before selecting your final answer.',
    'Use the root_team snapshot to determine the answer.','Consult other members’ answers before finalizing.'])
def test_shared_formal_inference_dependency_guard(text):
    assert inference_dependency_checks(text)==('inference_unavailable_group_dependency',)
    assert 'inference_unavailable_group_dependency' in candidate_checks(SEED.edit('strategy',text),parent=SEED,errors=[error()])


def test_abstract_repair_instruction_remains_legal():
    child=SEED.edit('strategy','Check arithmetic consistency before finalizing.')
    sf=dict(diagnostics=[dict(observed_failure='An observed operation is inconsistent.',diagnosis='Check arithmetic consistency before finalizing.',
        reusable_correction='Check arithmetic consistency before finalizing.')])
    assert not candidate_checks(child,parent=SEED,errors=[error()],structured_feedback=sf)


def test_native_worked_reference_projection_is_used_and_original_remains_private():
    e=error();e['reference_solution']='Unicode数学 '+('bounded source text '*400)
    original=deepcopy(e)
    view=optimizer_error_view(e)
    assert view['reference_solution']==worked_reference(e['reference_solution'])
    assert len(view['reference_solution']['text'])==4096
    assert view['reference_solution']['truncated'] and view['reference_solution']['split']=='optimize'
    assert view['reference_solution']['original_characters']==len(e['reference_solution'])
    assert e==original


def test_copied_private_text_retries_before_one_optimizer_opportunity():
    e=error();good='Check constraints and verify transformations before finalizing.'
    broker=ScriptedPort(outputs=[json.dumps(dict(block='strategy',content=e['reference_solution'])),
        json.dumps(dict(block='strategy',content=good))]);broker.round_index=1
    artifacts={}
    candidate,draws=generate_candidate(SEED,PreparedFeedback([e],None,None,{},None),broker,1,4,
        lambda n,v:artifacts.update({n:v}))
    assert candidate.strategy==good and draws==2
    assert artifacts['round_1_generation_1']['errors']==['private_solution_or_response_copy']
    assert broker.packets[0]['errors'][0]['reference_solution']==worked_reference(e['reference_solution'])


@pytest.mark.parametrize('adapter_type,native_type,packet',[
    (GradientTransportAdapter,PerExampleGradientProvider,dict(example=dict(example_id='synthetic'))),
    (ClusterTransportAdapter,GradientClusterProvider,dict(gradients=[dict(example_id='e1',gradient='Check constraints before simplification.')]))])
def test_actual_native_and_independent_diagnostic_requests_match(full,adapter_type,native_type,packet):
    from tests.current.population_support import contract
    b,t,_,_,_=full
    class NativePort:
        def __init__(self):self.contract=contract()
        def _request_identity(self,**kwargs):return b._request_identity(**kwargs)
        def complete(self,**kwargs):return b.complete(**kwargs)
    native=native_type(NativePort())
    independent=adapter_type(ModuleProvider(b,453))
    native._call(packet);independent._call(packet)
    assert t.calls[-2:][0]==t.calls[-2:][1]
    assert b.usage_view()['provider_calls']==2 and b.usage_view()['cache_hits']==0


def test_input_context_bound_stops_before_any_transport(full):
    b,t,_,_,_=full
    b.plan['config']['runtime']['optimizer_input_limit']=1
    with pytest.raises(SearchContractError,match='CONTEXT_LIMIT'):
        GradientTransportAdapter(ModuleProvider(b,453))._call(dict(example=dict(example_id='synthetic')))
    assert not t.calls and b.usage_view()['provider_calls']==0


def test_actual_optimizer_input_is_bound_to_opportunity_and_feedback(tmp_path):
    e=error();folder=tmp_path/'C';folder.mkdir()
    opportunity=dict(parent_prompt=SEED.to_dict(),errors=[e],structured_feedback=None,module_context=None)
    atomic_write_json(folder/'round_1_opportunity.json',opportunity)
    from multi_dataset_diverse_rl.research.ablations.generation import INSTRUCTION
    packet=optimizer_packet(SEED,[e],None,None)
    record=dict(logical_identity=dict(stage='round_1_legal_candidate_draw_1',ordinal=1),
        request=dict(messages=[dict(role='system',content=INSTRUCTION),dict(role='user',content=json.dumps(packet))]))
    audit_optimizer_request(tmp_path,'C',record)
    bad=deepcopy(record);wire=json.loads(bad['request']['messages'][1]['content']);wire['errors']=[]
    bad['request']['messages'][1]['content']=json.dumps(wire)
    with pytest.raises(SearchContractError,match='EVIDENCE_MISMATCH'):audit_optimizer_request(tmp_path,'C',bad)


def test_old_frozen_preparation_cannot_start_a_current_process(tmp_path,monkeypatch):
    prep=ROOT/'runs/ablations/verification_module_alignment_v6'/('historical_'+tmp_path.name)
    atomic_write_json(prep/'prep.json',dict(schema_version='single_prompt_full_diagnosis_prep_v5',scope={}))
    def forbidden():raise AssertionError('provider construction must not occur')
    monkeypatch.setattr(gov,'_transport',forbidden)
    with pytest.raises(SearchContractError,match='HISTORICAL_PREP'):
        gov.execute(ROOT,prep,prep/'never_created')
    assert not (prep/'never_created').exists()


def test_v5_frozen_manifest_is_metadata_only_and_not_silently_rewritten():
    from scripts.run_experiment import preflight
    old=read_json(ROOT/aligned.HISTORICAL_MANIFEST)
    aligned.validate_historical_manifest(ROOT,old)
    assert preflight(old)['status']=='ABLATION_FROZEN_V5_HISTORICAL_ONLY_NOT_EXECUTABLE'
    changed=deepcopy(old);changed['config']['rounds']=1
    with pytest.raises(SearchContractError,match='FROZEN_V5'):aligned.validate_historical_manifest(ROOT,changed)


def test_declared_response_policy_matches_actual_formal_codec():
    from multi_dataset_diverse_rl.benchmarks.math_response_evidence import trajectory_policy
    assert default_config()['runtime']['trajectory_policy']==trajectory_policy()
