"""Default equivalence surfaces and explicit synthetic mechanism interventions."""
import asyncio,json
from dataclasses import replace
import pytest
from tests.current.population_support import graph
from multi_dataset_diverse_rl.search.mechanisms import TeamModules,DEFAULT_TEAM_MODULES,register_mechanism
from multi_dataset_diverse_rl.search.reflection_steps import DisabledGroupChannel,DisabledReflection
from multi_dataset_diverse_rl.search.memory_components import DisabledMemoryDelivery
from multi_dataset_diverse_rl.search.schemas import SearchContractError

def test_catalog_default_dependencies_and_authority():
    binding=DEFAULT_TEAM_MODULES.binding()
    assert len(binding['modules'])==20 and not DEFAULT_TEAM_MODULES.changed()
    assert binding['modules']['who']['authority']=='select_member'
    assert binding['modules']['reflection_generation']['authority']=='propose_edit_only'
    assert binding['modules']['group_channel']['model_roles']==['group_diagnosis']
    assert 'memory_retrieval' in binding['modules']['memory_delivery']['dependencies']

@pytest.mark.parametrize('field,factory',[
    ('group_channel',DisabledGroupChannel),('reflection_generation',DisabledReflection),('memory_delivery',DisabledMemoryDelivery)])
def test_disabled_components_require_explicit_intervention(field,factory,tmp_path):
    modules=replace(DEFAULT_TEAM_MODULES,**{field:factory})
    with pytest.raises(SearchContractError,match='NEW_FROZEN_METHOD'):graph(tmp_path,modules=modules)

def test_group_disabled_is_not_no_supported_insight_and_has_no_call(tmp_path):
    modules=replace(DEFAULT_TEAM_MODULES,group_channel=DisabledGroupChannel)
    run,broker,requests,packets=graph(tmp_path,modules=modules,synthetic_intervention=True)
    result=asyncio.run(run.run(max_opportunities=1))
    assert result.method_identity != run.method.identity()
    assert not any(r['messages'][0]['content'].startswith('Diagnose only') for r in requests)
    assert len(packets)==3 and all(p['group_experience']['status']=='MODULE_DISABLED' for p in packets)
    saved=json.loads(next(tmp_path.glob('group_reflection.private/*/group.json')).read_bytes())
    assert saved['audit']['accounting']['successful_calls']==0
    assert run.population.scope['module_intervention']

def test_reflection_disabled_creates_no_candidate_or_fake_success(tmp_path):
    run,broker,requests,packets=graph(tmp_path,modules=replace(DEFAULT_TEAM_MODULES,
        reflection_generation=DisabledReflection),synthetic_intervention=True)
    events={};run.execution_observer=lambda stage,data:events.update({stage:data})
    result=asyncio.run(run.run(max_opportunities=1))
    search=events['EVALUATION']['search']
    assert not packets and not search.candidates and search.search_state['reflection_draws']==0
    assert all(r['disposition']=='MODULE_DISABLED' for r in search.search_state['generation_outcomes'])

def test_memory_delivery_disabled_uses_actual_store_without_fabricated_cards(tmp_path):
    run,broker,requests,packets=graph(tmp_path,modules=replace(DEFAULT_TEAM_MODULES,
        memory_delivery=DisabledMemoryDelivery),synthetic_intervention=True)
    asyncio.run(run.run(max_opportunities=1))
    assert all(p['scoped_experience']['status']=='MODULE_DISABLED' and not p['scoped_experience']['entries'] for p in packets)
    assert run.memory.revision>0 and run.memory.delivered==0
    assert type(run.memory.detached().delivery) is DisabledMemoryDelivery

def test_incompatible_who_rejected_before_provider():
    class WrongWho:
        def select(self):return 0
    with pytest.raises(SearchContractError,match='SIGNATURE_MISMATCH'):
        register_mechanism('who',WrongWho,identity='BROKEN_SYNTHETIC_WHO')

def test_single_prompt_normalization_has_no_team_scoring(monkeypatch):
    from multi_dataset_diverse_rl.search.textual_gradients import normalize_gradient_partition
    from multi_dataset_diverse_rl.search.pattern_selection import WeightedSupportScorer,LargestUniqueSupportSelector
    from tests.current.test_weighted_error_responsibility import row
    def forbidden(*args,**kwargs):raise AssertionError('team scoring must not normalize an independent partition')
    monkeypatch.setattr(WeightedSupportScorer,'score',forbidden)
    evidence=(row(0,None),)
    text='Check applicable constraints before transforming an expression.'
    value=normalize_gradient_partition(dict(patterns=[dict(generalized_gradient=text,support_ids=['0'])],unassigned_ids=[]),
        evidence,[dict(example_id='0',gradient=text)])
    selected,ties=LargestUniqueSupportSelector().select(value['patterns'],seed=42,ordinal=1)
    assert selected['support_ids']==('0',) and 'responsibility' not in selected


def test_who_is_replaced_without_editing_gradient_or_layer1(tmp_path):
    from multi_dataset_diverse_rl.search.policies import TargetPolicyV1
    class LastEligibleWho(TargetPolicyV1):
        identity='SYNTHETIC_LAST_ELIGIBLE_WHO_V1'
        def select(self,state,diagnosis,feasible_members,history):
            value=super().select(state,diagnosis,feasible_members,history)
            return replace(value,selected_member=max(value.eligible_members),reason='SYNTHETIC_CONTROL')
    register_mechanism('who',LastEligibleWho,identity=LastEligibleWho.identity)
    run,_,_,packets=graph(tmp_path,modules=replace(DEFAULT_TEAM_MODULES,who=LastEligibleWho),synthetic_intervention=True)
    events={};run.execution_observer=lambda stage,data:events.update({stage:data})
    asyncio.run(run.run(max_opportunities=1))
    assert events['OPPORTUNITY']['opportunity'].target_member==4 and len(packets)==3


def test_new_source_is_required_but_default_method_contract_is_unchanged():
    from tests.current.population_support import contract
    from multi_dataset_diverse_rl.search.current_policy import CURRENT_POLICY_BUNDLE
    from multi_dataset_diverse_rl import current_contract
    assert current_contract.CURRENT_METHOD_VERSION=='unified_team_prompt_search_v2_12_count_responsibility_v1'
    CURRENT_POLICY_BUNDLE.validate_contract(contract())


def test_gradient_only_is_not_enabled_by_an_adapter_name():
    from tests.current.test_single_prompt_ablations import small_plan
    from multi_dataset_diverse_rl.research.ablations.modules import default_registry
    with pytest.raises(SearchContractError,match='USER_DISABLED'):
        small_plan(rounds=1,module='formal_gradient')
    with pytest.raises(SearchContractError,match='NOT_REGISTERED'):
        default_registry().create('memory')


def test_sampler_replacement_replays_its_own_plans_and_detached_policies():
    from multi_dataset_diverse_rl.search.population_policies import FitnessWeightedSampling
    from multi_dataset_diverse_rl.search.parent_population import ParentPopulation
    from tests.current.test_vote_parent_population import add
    class UniformSampling(FitnessWeightedSampling):
        def probabilities(self,parents):return tuple(1/len(parents) for _ in parents)
    register_mechanism('population_sampling',UniformSampling,identity='SYNTHETIC_UNIFORM_PARENT_V1')
    pool=ParentPopulation(scope=dict(method_identity='mock',support_sha256='a'*64,support_count=60),sampling=UniformSampling())
    for i,n in enumerate([60,30,0]):add(pool,0,n,i)
    for member in range(1,5):add(pool,member,20,member)
    _,plan=pool.select_three(0,seed=81,opportunity_id='controlled')
    assert [d['probability'] for d in plan['distribution']]==pytest.approx([1/3]*3)
    restored=pool.detached();restored.restore(pool.export())
    assert isinstance(restored.sampling,UniformSampling) and restored.export()==pool.export()


def test_sampling_interface_requires_draw_as_well_as_probabilities():
    class IncompleteSampling:
        def probabilities(self,parents):return (1.,)
    with pytest.raises(SearchContractError,match='SIGNATURE_MISMATCH'):
        register_mechanism('population_sampling',IncompleteSampling,identity='INCOMPLETE')


def test_disabled_memory_delivery_survives_atomic_boundary_recovery(tmp_path):
    from multi_dataset_diverse_rl.persistence.durable_io import read_json,atomic_write_json
    modules=replace(DEFAULT_TEAM_MODULES,memory_delivery=DisabledMemoryDelivery)
    run,_,_,_=graph(tmp_path/'source',modules=modules,synthetic_intervention=True)
    asyncio.run(run.run(max_opportunities=1))
    body=read_json(run.checkpoint.path)
    fresh,_,requests,_=graph(tmp_path/'restore',modules=modules,synthetic_intervention=True)
    fresh.checkpoint.path=run.checkpoint.path
    atomic_write_json(run.checkpoint.path.with_suffix('.pending.json'),dict(scope=body['scope'],
        opportunity_ordinal=body['next_ordinal'],state='BOUNDARY_COMPLETE'))
    before=len(requests);fresh.checkpoint.restore()
    assert len(requests)==before and isinstance(fresh.memory.delivery,DisabledMemoryDelivery)
    assert fresh.memory.export()==body['memory']
    default,_,_,_=graph(tmp_path/'default');default.checkpoint.path=run.checkpoint.path
    with pytest.raises(SearchContractError,match='SCOPE_MISMATCH'):default.checkpoint.restore()


def test_inventory_keeps_team_and_independent_boundaries_honest():
    from multi_dataset_diverse_rl.search.mechanisms import capability_inventory
    ports=capability_inventory()['boundary_capabilities']
    assert ports['private_shadow']['data_scopes']==['adaptive_gate']
    assert ports['pattern_clustering']['single_prompt_adaptation']=='direct_adapter'
    assert ports['memory_storage']['single_prompt_adaptation']=='measurement_adapter_unresolved'


def test_incompatible_strategy_outputs_cannot_change_pool_or_slot_contract():
    from tests.current.test_vote_parent_population import pool,add
    from multi_dataset_diverse_rl.search.population_policies import FitnessWeightedSampling,BestThreeRetention
    p=pool();add(p,0,20);before=p.export()
    class EmptyRetention(BestThreeRetention):
        def keep(self,ordered):return ()
    p.retention=EmptyRetention()
    with pytest.raises(SearchContractError,match='RETENTION_OUTPUT_INVALID'):add(p,0,30,2)
    assert p.export()==before
    class ZeroSampling(FitnessWeightedSampling):
        def probabilities(self,parents):return (0.,)
    p.sampling=ZeroSampling()
    with pytest.raises(SearchContractError,match='SAMPLING_OUTPUT_INVALID'):p.select_three(0,seed=1,opportunity_id='bad_probability')
    class FourDraws(FitnessWeightedSampling):
        def draw(self,parents,probabilities,**kwargs):return (parents[0],)*4
    p.sampling=FourDraws()
    with pytest.raises(SearchContractError,match='DRAW_OUTPUT_INVALID'):p.select_three(0,seed=1,opportunity_id='bad_draws')
    assert not p.plans


def test_generator_replacement_does_not_bypass_candidate_safety(tmp_path):
    from multi_dataset_diverse_rl.search.reflection_steps import SingleStepReflection,SlotGeneration
    from multi_dataset_diverse_rl.search.schemas import SearchCandidate
    class UnsafeGenerator(SingleStepReflection):
        def generate(self,**kwargs):
            child=kwargs['parent'].prompt.edit('strategy','Apply a general repair.').edit('role','A changed second block.')
            return SlotGeneration(SearchCandidate('unsafe',child),[],0,[],'GENERATED')
    register_mechanism('reflection_generation',UnsafeGenerator,identity='SYNTHETIC_UNSAFE_CONTROL')
    run,_,_,_=graph(tmp_path,modules=replace(DEFAULT_TEAM_MODULES,reflection_generation=UnsafeGenerator),
        synthetic_intervention=True)
    with pytest.raises(SearchContractError,match='UNSAFE_OUTPUT'):asyncio.run(run.run(max_opportunities=1))
