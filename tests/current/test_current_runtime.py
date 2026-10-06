"""Current eligibility, explicit replay boundaries and dependency conformance."""
import ast
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from multi_dataset_diverse_rl.benchmarks.math_domain_binding import execution_binding
from multi_dataset_diverse_rl.governance.current_dependencies import current_dependency_graph
from multi_dataset_diverse_rl.search.current_composition import build_current_team_prompt_search
from multi_dataset_diverse_rl.search.current_layer1 import CurrentLayer1Config, CurrentOptimizer, CurrentEngine
from multi_dataset_diverse_rl.search.current_policy import CURRENT_POLICY_BUNDLE, CurrentPolicyBundle
from multi_dataset_diverse_rl.search.schemas import SearchContractError

ROOT=Path(__file__).resolve().parents[2]
PROFILE='experiments/execution_bindings/math_v2_1_gradient_pattern_offline_profile_v4.json'


def contract():return json.loads((ROOT/PROFILE).read_bytes())


@pytest.mark.parametrize('field,value',[
    ('pattern_policy',{'discovery':'set_level_wrong_pattern_discovery_v3'}),
    ('memory_policy_identity','structured_action_failure_memory_v3'),
    ('panel_evidence_policy','pattern_representatives_preservation_transition_v4'),
    ('optimizer_input_schema','PATTERN_CURRENT_EVIDENCE_MEMORY_INPUT_V3'),
    ('layer1_search_policy',{'identity_version':'gepa_team_candidate_exposure_v2'}),
])
def test_current_binding_rejects_legacy_before_any_provider(field,value):
    c=contract();c[field]=value
    with pytest.raises(SearchContractError,match='CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN'):
        execution_binding(ROOT,c)


@pytest.mark.parametrize('field,value',[
    ('pattern_policy','set_level_wrong_pattern_discovery_v3'),
    ('memory_policy','structured_action_failure_memory_v3'),
    ('evidence_policy','pattern_representatives_preservation_transition_v4'),
    ('search_engine','gepa_team_candidate_exposure_v2'),
])
def test_current_builder_rejects_legacy_method(field,value):
    method=execution_binding(ROOT,contract()).method('A4')
    with pytest.raises(SearchContractError,match='CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN'):
        CURRENT_POLICY_BUNDLE.validate_method(replace(method,**{field:value}))


def test_current_requires_whole_bundle():
    with pytest.raises(SearchContractError,match='CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN'):
        CurrentPolicyBundle(memory='structured_action_failure_memory_v3')


def test_missing_gradient_provider_has_no_silent_fallback():
    method=execution_binding(ROOT,contract()).method('A4')
    with pytest.raises(SearchContractError,match='HOLD_PRE_PROVIDER: CURRENT_GRADIENT_PROVIDER_NOT_BOUND'):
        build_current_team_prompt_search(benchmark=None,aggregation=None,examples=(),prompts=(),solver=None,
            optimizer=NS(config=CurrentLayer1Config()),method=method,seed=81,shadow_loader=None,
            shadow_count=40,runtime_readiness=lambda:(),pattern_provider=None)


def test_missing_memory_provider_has_no_silent_fallback(monkeypatch):
    from multi_dataset_diverse_rl.search import current_composition as composition
    monkeypatch.setattr(composition,'BinaryTeamStateStore',lambda **kw:object())
    monkeypatch.setattr(composition,'StructuredRollingRiskMemoryV4',lambda **kw:None)
    method=execution_binding(ROOT,contract()).method('A4')
    port=NS(gradient_provider=lambda _:None,support_id_transport=method.mechanism_config['pattern_support_id_transport'])
    with pytest.raises(SearchContractError,match='HOLD_PRE_PROVIDER: CURRENT_MEMORY_PROVIDER_NOT_BOUND'):
        build_current_team_prompt_search(benchmark=None,aggregation=NS(identity=method.aggregation_policy),examples=(),prompts=(),solver=None,
            optimizer=NS(config=CurrentLayer1Config()),method=method,seed=81,shadow_loader=None,
            shadow_count=40,runtime_readiness=lambda:(),pattern_provider=port)


def test_active_inheritance_has_no_old_pattern_or_memory_treatment():
    assert [c.__name__ for c in CurrentOptimizer.__mro__]==['GradientPatternMemoryOptimizer','BoundedMemoryOptimizer','object']
    assert [c.__name__ for c in CurrentEngine.__mro__]==['GradientPatternMemoryEngine','LocalTaskEngine','object']
    from multi_dataset_diverse_rl.search.rolling_risk_memory import StructuredRollingRiskMemoryV4
    assert [c.__name__ for c in StructuredRollingRiskMemoryV4.__mro__]==['StructuredRollingRiskMemoryV4','PrivateActionMemory','MemoryState','object']


def test_current_dependency_closure_has_no_legacy_runtime():
    graph=current_dependency_graph(ROOT)
    assert graph['legacy_namespace_dependencies']==[]
    assert graph['legacy_class_definitions']==[]
    definitions=[n.name for p in graph['modules'] for n in ast.walk(ast.parse((ROOT/p).read_text(encoding='utf-8-sig'))) if isinstance(n,ast.ClassDef)]
    for name in ('GradientPatternDiscovery','StructuredRollingRiskMemoryV4','GradientPatternMemoryEngine'):
        assert definitions.count(name)==1


def test_dependency_audit_detects_injected_legacy_import(monkeypatch):
    from multi_dataset_diverse_rl.governance import current_dependencies as dependencies
    original=dependencies.local_imports
    legacy=ROOT/'multi_dataset_diverse_rl/search/legacy/pattern_layer1.py'
    def injected(root,path,**kwargs):
        result=original(root,path,**kwargs)
        return result|{legacy} if path.name=='current_composition.py' else result
    monkeypatch.setattr(dependencies,'local_imports',injected)
    graph=dependencies.current_dependency_graph(ROOT)
    assert legacy.relative_to(ROOT).as_posix() in graph['legacy_namespace_dependencies']
    assert any(row['name']=='PatternMemoryOptimizer' for row in graph['legacy_class_definitions'])


def test_legacy_implementation_does_not_determine_current_source_identity():
    from multi_dataset_diverse_rl.governance.source_identity import build_unified_source_identity
    identity=build_unified_source_identity(ROOT)
    for scope in identity['scopes'].values():
        assert not any('/legacy/' in row['path'] for row in scope['files'])
    assert 'multi_dataset_diverse_rl/benchmarks/math_pattern_binding.py' not in {
        row['path'] for row in identity['scopes']['benchmark']['files']}


def test_current_manifest_rejects_old_engine_without_schema_migration():
    from multi_dataset_diverse_rl.governance.unified_execution import preexecution_manifest,bound_preflight
    m=preexecution_manifest(ROOT,source_sha='0'*40,frozen=False,binding_path=PROFILE,experiment_id='synthetic_current')
    m['search_engine_identity']='gepa_team_candidate_exposure_v2'
    result=bound_preflight(ROOT,m)
    assert result['gate']=='HOLD' and result['provider_attempts']==0
    assert 'MANIFEST_EXECUTION_BINDING_MISMATCH' in result['blockers']


def test_historical_binding_requires_explicit_replay_factory():
    c=json.loads((ROOT/'experiments/execution_bindings/math_v2_1_pattern_canary_v4.json').read_bytes())
    with pytest.raises(SearchContractError,match='CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN'):execution_binding(ROOT,c)
    from multi_dataset_diverse_rl.benchmarks.legacy.math_domain_binding import execution_binding as replay_binding
    # V1_1 replay needs its frozen source/artifact; the V1_2 worktree cannot
    # silently execute it with a changed initial treatment.
    assert replay_binding(ROOT,c).blockers()


def test_amended_treatment_identity_and_n_plus_one_ceiling_are_frozen():
    c=contract();b=execution_binding(ROOT,c)
    assert b.method('A4').identity()=='bb08c5f6c7228e0a6338017d2df5929ad95dd2fc1b016fdc871bc5923038af49'
    assert asdict(CurrentLayer1Config())==c['layer1_search_policy']
    assert c['provider_bounds']['pattern_calls']==c['provider_bounds']['pattern_gradient_calls']+c['provider_bounds']['pattern_cluster_calls']==13
    assert c['provider_bounds']['successful_provider_calls']==1651
    assert c['access']['test']=='sealed'


def test_flat_current_authorization_scope_matches_frozen_gradient_scope():
    from multi_dataset_diverse_rl.governance.unified_execution import execution_scope
    from multi_dataset_diverse_rl.governance.legacy.unified_execution import execution_scope as previous_scope
    manifest=dict(source_sha='0'*40,preregistration_identity='0'*64,execution_binding={'sha256':'0'*64})
    historical=json.loads((ROOT/'experiments/execution_bindings/math_v2_1_gradient_pattern_offline_profile_v2.json').read_bytes())
    with pytest.raises(SearchContractError,match='CURRENT_RUNTIME_LEGACY_POLICY_FORBIDDEN'):
        execution_scope(manifest,historical)
    assert previous_scope(manifest,historical)['pattern_abstraction_guard']=='PATTERN_ABSTRACTION_SPECIFIC_CONTENT_GUARD_V4'
    assert execution_scope(manifest,contract())!=previous_scope(manifest,historical)


@pytest.mark.parametrize('package',['openai','math-verify'])
def test_current_binding_checks_installed_pins_before_provider(monkeypatch,package):
    from multi_dataset_diverse_rl.benchmarks import current_math_dependencies as dependencies
    version=dependencies.importlib.metadata.version
    monkeypatch.setattr(dependencies.importlib.metadata,'version',lambda name:'changed' if name==package else version(name))
    assert execution_binding(ROOT,contract()).blockers()[0].startswith('CURRENT_DATA_')


def test_current_binding_checks_canonical_bytes_before_provider(monkeypatch):
    from multi_dataset_diverse_rl.benchmarks import current_math_dependencies as dependencies
    canonical=ROOT/contract()['canonical_root']/'manifests/math.json'
    original=dependencies.file_hash
    monkeypatch.setattr(dependencies,'file_hash',lambda path:'0'*64 if path==canonical else original(path))
    assert execution_binding(ROOT,contract()).blockers()==('CURRENT_DATA_CANONICAL_MANIFEST_IDENTITY_MISMATCH',)


def test_current_binding_checks_subset_metadata_before_provider(monkeypatch):
    from copy import deepcopy
    from multi_dataset_diverse_rl.benchmarks import current_math_dependencies as dependencies
    subsets=deepcopy(dependencies.read_subsets(ROOT,contract()))
    subsets['metadata_universe'][0]['source_index']+=1
    monkeypatch.setattr(dependencies,'read_subsets',lambda *_:subsets)
    assert execution_binding(ROOT,contract()).blockers()==('CURRENT_DATA_LOW_COST_SUPERSET_MEMBERSHIP_MISMATCH',)
