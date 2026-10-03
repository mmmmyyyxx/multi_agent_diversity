"""Offline metadata, archive and source-scope regression guards."""
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from multi_dataset_diverse_rl.governance.registries import load_yaml
from multi_dataset_diverse_rl.governance.repository import (
    audit_repository,validate_registry_v2,validate_lineage_v2,validate_manifest_v2,
    CURRENT_DOCS,ERAS,STATUSES,build_report_index,import_guard,
)
from multi_dataset_diverse_rl.governance.source_identity import build_unified_source_identity

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def registry():return load_yaml(ROOT/'experiments/registry.yaml')


@pytest.fixture(scope='module')
def lineage():return load_yaml(ROOT/'experiments/lineage.yaml')


@pytest.fixture
def template():return load_yaml(ROOT/'experiments/templates/unified_experiment_v2.yaml')


def test_current_docs_do_not_claim_two_layer_is_active():
    for path in CURRENT_DOCS:
        assert not re.search(r'active research (?:architecture|direction) is (?:backend-neutral )?Layer', (ROOT/path).read_text(encoding='utf-8'),re.I)


def test_current_docs_do_not_claim_v15_is_active():
    for path in CURRENT_DOCS:
        assert 'canonical runtime is `member_aware_peer_state_v15`' not in (ROOT/path).read_text(encoding='utf-8')


def test_current_docs_agree_on_unified_architecture():
    for path in CURRENT_DOCS:
        assert 'sole active research architecture is Unified Team Prompt Search' in (ROOT/path).read_text(encoding='utf-8').replace('\n',' ')


def test_reports_are_not_normative_authorities():
    assert 'Reports are evidence, never design authority.' in (ROOT/'AGENTS.md').read_text(encoding='utf-8')
    assert build_report_index(ROOT)['reports_are_normative'] is False


def test_registry_unique_ids(registry):
    ids=[r['experiment_id'] for r in registry['experiments']];assert len(ids)==len(set(ids))
    duplicate=deepcopy(registry);duplicate['experiments'].append(duplicate['experiments'][0])
    assert any('duplicate' in e for e in validate_registry_v2(ROOT,duplicate)[0])


@pytest.mark.parametrize('field',['manifest','report'])
def test_registry_manifest_paths(registry,field):
    for row in registry['experiments']:
        assert (ROOT/row[field]).exists() if row[field] else row['unavailable'][field]
    broken=deepcopy(registry);broken['experiments'][0][field]='missing_governance_artifact'
    assert any('missing '+field in e for e in validate_registry_v2(ROOT,broken)[0])


def test_registry_report_paths(registry):assert validate_registry_v2(ROOT,registry)[0]==[]


def test_registry_status_enum(registry):assert all(r['status'] in STATUSES for r in registry['experiments'])


def test_registry_era_enum(registry):assert all(r['era'] in ERAS for r in registry['experiments'])


def test_registry_no_stale_running_without_runtime_evidence(registry):
    broken=deepcopy(registry);broken['experiments'][0]['status']='RUNNING'
    assert any('stale RUNNING' in e for e in validate_registry_v2(ROOT,broken)[0])


def test_lineage_acyclic(registry,lineage):
    assert validate_lineage_v2(registry,lineage)==[]
    broken=deepcopy(lineage);e=broken['edges'][0]
    broken['edges'].append({'from':e['to'],'to':e['from'],'relation':'followup_of'})
    assert any('cycle' in e for e in validate_lineage_v2(registry,broken))


def test_lineage_nodes_registered(registry,lineage):
    assert {r['experiment_id'] for r in registry['experiments']}=={r['experiment_id'] for r in lineage['nodes']}


def test_registry_scientific_nodes_in_lineage(registry,lineage):test_lineage_nodes_registered(registry,lineage)


def test_current_frontier_exists(registry):
    frontier=load_yaml(ROOT/'experiments/current_frontier.yaml')
    assert frontier['last_governance_milestone'] in {r['experiment_id'] for r in registry['experiments']}


@pytest.mark.parametrize('poison,expected',[
    ('unregistered_experiment','current experiment is not registered'),
    ('math_v2_preexecution_freeze_v1_1','current experiment differs from canary manifest identity'),
])
def test_current_experiment_registration_fails_closed(monkeypatch,poison,expected):
    import multi_dataset_diverse_rl.governance.repository as repository
    original=repository.load_yaml
    def read(path):
        result=original(path)
        if Path(path)==ROOT/'experiments/current_frontier.yaml':
            result=deepcopy(result)
            result['current_experiment']=poison
            result['real_execution_ready']='true_for_canary_only'
        return result
    monkeypatch.setattr(repository,'load_yaml',read)
    assert expected in repository.audit_repository(ROOT,check_generated=False)['errors']


def test_current_frontier_has_no_api_authorization():
    frontier=load_yaml(ROOT/'experiments/current_frontier.yaml')
    assert frontier['real_api_authorized'] is False
    if frontier['real_execution_ready'] == 'true_for_canary_only':
        from multi_dataset_diverse_rl.governance.unified_execution import bound_preflight
        assert frontier['validation_access']=='not_authorized'
        manifest=load_yaml(ROOT/frontier['canary_manifest'])
        assert manifest['lifecycle']['status']=='PREEXECUTION_FROZEN'
        assert bound_preflight(ROOT,manifest)['gate']=='CANARY_READY_NOT_AUTHORIZED'
    else:
        assert frontier['real_execution_ready'] is False
    assert frontier['test_access']=='sealed'


def test_manifest_v2_schema(template):
    assert validate_manifest_v2(ROOT,template)==[]
    for field in ['search_engine_identity','split_identity','hash_closure','authorization','access']:
        broken=deepcopy(template);del broken[field]
        assert validate_manifest_v2(ROOT,broken)


def test_manifest_default_api_false(template):assert template['authorization']['real_api_authorized'] is False


def test_manifest_test_sealed(template):assert template['access']['test_access']=='sealed'


def test_manifest_pattern_null_default(template):assert template['pattern_identity']=='null_pattern_v1'


def test_manifest_memory_null_default(template):assert template['memory_identity']=='null_memory_v1'


def test_manifest_frozen_requires_bindings(template):
    template['lifecycle']['status']='FROZEN'
    assert any('requires source_sha' in e for e in validate_manifest_v2(ROOT,template))
    template['authorization']['real_api_authorized']=True
    assert any('authorization requires' in e for e in validate_manifest_v2(ROOT,template))


def test_archived_specs_marked_historical():
    for path in (ROOT/'docs/archive/specs').glob('*.md'):assert 'archive_status: HISTORICAL_REPLAY_ONLY' in path.read_text(encoding='utf-8')


def test_historical_invariant_index_complete():
    body=(ROOT/'docs/archive/specs/historical_current_spec.md').read_text(encoding='utf-8')
    index=load_yaml(ROOT/'docs/archive/specs/historical_invariant_index.json')
    assert set(re.findall(r'INV-[A-Z0-9-]+',body))=={r['invariant_id'] for r in index['invariants']}


@pytest.fixture
def identity_workspace(tmp_path):
    for path,text in {
        'multi_dataset_diverse_rl/versions.py':'POLICY = "one"\n',
        'multi_dataset_diverse_rl/search/core.py':'VALUE = 1\n',
        'docs/design/CURRENT_SPEC.md':'Current scientific contract\n',
        'AGENTS.md':'Governance\n','README.md':'Overview\n',
        'docs/archive/old.md':'History\n','reports/old/README.md':'Evidence\n',
        'multi_dataset_diverse_rl/benchmarks/contract.py':'BENCHMARK = 1\n',
    }.items():
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    return tmp_path


@pytest.mark.parametrize('path',['README.md','docs/archive/old.md','reports/old/README.md'])
def test_current_source_identity_excludes_archive(identity_workspace,path):
    root=identity_workspace;before=build_unified_source_identity(root)
    (root/path).write_text('Different prose\n');after=build_unified_source_identity(root)
    assert before['scientific_source_hash']==after['scientific_source_hash']


@pytest.mark.parametrize('path,text',[('docs/design/CURRENT_SPEC.md','Scientific change\n'),
    ('multi_dataset_diverse_rl/search/core.py','VALUE = 2\n'),
    ('multi_dataset_diverse_rl/versions.py','POLICY = "two"\n')])
def test_scientific_changes_affect_identity(identity_workspace,path,text):
    root=identity_workspace;before=build_unified_source_identity(root)
    (root/path).write_text(text);after=build_unified_source_identity(root)
    assert before['scientific_source_hash']!=after['scientific_source_hash']


@pytest.mark.parametrize('path,changed_hash',[
    ('AGENTS.md','governance_hash'),
    ('multi_dataset_diverse_rl/benchmarks/contract.py','benchmark_contract_hash')])
def test_identity_scopes_are_separate(identity_workspace,path,changed_hash):
    root=identity_workspace;before=build_unified_source_identity(root)
    (root/path).write_text('VALUE = 2\n');after=build_unified_source_identity(root)
    assert before['scientific_source_hash']==after['scientific_source_hash']
    assert before[changed_hash]!=after[changed_hash]


def test_current_search_does_not_import_historical_runner():assert import_guard(ROOT)==[]


def test_current_search_does_not_import_mars_controller(identity_workspace):
    root=identity_workspace;(root/'multi_dataset_diverse_rl/search/core.py').write_text('import multi_dataset_diverse_rl.local_optimizers.mars_native\n')
    assert import_guard(root)


def test_current_benchmark_contracts_do_not_import_reports(identity_workspace):
    root=identity_workspace;(root/'multi_dataset_diverse_rl/benchmarks/contract.py').write_text('from reports import metric\n')
    assert import_guard(root)


def test_operational_bootstrap_remains_governed(identity_workspace):
    root=identity_workspace
    (root/'multi_dataset_diverse_rl/__init__.py').write_text('from .system import VALUE\n')
    path=root/'multi_dataset_diverse_rl/system.py';path.write_text('VALUE = 1\n')
    before=build_unified_source_identity(root);path.write_text('VALUE = 2\n')
    after=build_unified_source_identity(root)
    assert before['scientific_source_hash']==after['scientific_source_hash']
    assert before['governance_hash']!=after['governance_hash']
    assert 'multi_dataset_diverse_rl/system.py' in after['operational_bootstrap_paths']


def test_scope_hash_is_portable_across_line_endings(identity_workspace):
    root=identity_workspace
    path=root/'multi_dataset_diverse_rl/search/core.py'
    path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
    before=build_unified_source_identity(root)
    path.write_bytes(path.read_bytes().replace(b'\n',b'\r\n'))
    after=build_unified_source_identity(root)
    assert before['scientific_source_hash']==after['scientific_source_hash']
    a=next(r for r in before['scopes']['scientific']['files'] if r['path'].endswith('core.py'))
    b=next(r for r in after['scopes']['scientific']['files'] if r['path'].endswith('core.py'))
    assert a['raw_sha256']!=b['raw_sha256']


def test_readme_cannot_be_bound_as_scientific_config(identity_workspace):
    with pytest.raises(ValueError):
        build_unified_source_identity(identity_workspace,config_paths=[Path('README.md')])


def test_complete_governance_audit():
    result=audit_repository(ROOT)
    assert result['ok'],result['errors']


def test_real_canary_is_execution_evidence_only(registry):
    row=next(r for r in registry['experiments'] if r['experiment_id']=='math_v2_a1_seed81_real_canary_v1')
    assert row['kind']=='REAL_CANARY'
    assert row['classifier']=='CANARY_OPERATIONAL_FAILURE'
    assert row['scientific_status']=='EXECUTION_ABORTED_EFFICACY_UNEVALUATED'
    assert row['authorization_consumed'] is True
    assert row['active_for_new_work'] is False


def _assert_canary_does_not_unlock_formal_or_heldout(frontier,registry):
    previous=next(r for r in registry['experiments'] if r['experiment_id']==frontier['last_canary_milestone'])
    assert previous['kind']=='REAL_CANARY' and previous['authorization_consumed'] is True
    # A later operational preflight can close Canary readiness. Historical
    # failure evidence must not require live readiness to remain open.
    assert frontier['real_execution_ready'] in (False, 'true_for_canary_only')
    next_attempt = frontier['next_canary_attempt_id']
    if next_attempt is None:
        assert frontier['real_execution_ready'] is False
        if frontier['current_method'] == 'unified_team_prompt_search_v2_1':
            if frontier['autonomous_authorization_status'] == 'PRIOR_SCOPES_DO_NOT_AUTHORIZE_V2_1':
                assert frontier['current_canary_status'] == 'METHOD_SEMANTIC_CONTRACT_REFREEZE_REQUIRED'
            else:
                assert frontier['autonomous_authorization_status'] in {
                    'V2_1_CONTINUATION_RECEIVED_EXACT_SCOPE_PENDING', 'V2_1_CONTINUATION_EXACT_SCOPES_CLOSED',
                    'LOW_COST_TASK_COMPLETE_SCOPES_CLOSED'}
                current=next(r for r in registry['experiments'] if r['experiment_id']==frontier['current_experiment'])
                manifest=load_yaml(ROOT/current['manifest'])
                assert manifest['method_identity']=='unified_team_prompt_search_v2_1'
                assert manifest['execution_binding']['identity'] in {'MATH_V2_1_EXECUTION_BINDING_V1','MATH_V2_1_EXECUTION_BINDING_V2','MATH_V2_1_EXECUTION_BINDING_V3','MATH_V2_1_LOW_COST_EXECUTION_BINDING_V1'}
                if manifest['execution_binding']['identity'] in {'MATH_V2_1_EXECUTION_BINDING_V2','MATH_V2_1_EXECUTION_BINDING_V3','MATH_V2_1_LOW_COST_EXECUTION_BINDING_V1'}:
                    from multi_dataset_diverse_rl.benchmarks.math_solver_decoding import solver_decoding_contract
                    assert manifest['solver_decoding_policy']==solver_decoding_contract()
                if manifest['execution_binding']['identity']=='MATH_V2_1_EXECUTION_BINDING_V3':
                    from multi_dataset_diverse_rl.benchmarks.math_prediction_validity import prediction_validity_contract
                    assert manifest['prediction_validity_policy']==prediction_validity_contract()
                if manifest['execution_binding']['identity']=='MATH_V2_1_LOW_COST_EXECUTION_BINDING_V1':
                    from multi_dataset_diverse_rl.benchmarks.math_prediction_validity import frozen_prediction_policy,invalid_recovery_contract
                    binding=load_yaml(ROOT/manifest['execution_binding']['path'])
                    assert manifest['prediction_validity_policy']==frozen_prediction_policy(binding)
                    assert manifest['invalid_recovery_policy']==invalid_recovery_contract()
                assert manifest['authorization']['real_api_authorized'] is False
        else:
            assert frontier['current_canary_status'] == 'STOP_SCIENTIFIC_CONTRACT_AMENDMENT_REQUIRED'
            assert frontier['autonomous_authorization_status'] == 'HALTED_BY_NON_OPERATIONAL_FAILURE'
    else:
        # Attempt identity belongs to the registered execution binding. A
        # historical naming prefix cannot govern a new preregistered protocol.
        next_node=next(r for r in registry['experiments'] if r['experiment_id']==frontier['next_canary_milestone'])
        assert next_node['kind']=='REAL_CANARY'
        manifest=load_yaml(ROOT/next_node['manifest'])
        binding=load_yaml(ROOT/manifest['execution_binding']['path'])
        assert manifest['method_identity']==frontier['current_method']
        assert binding['execution_phase']=='canary'
        assert next_attempt==binding['execution_attempt_id']==binding['canary_attempt_id']==binding['cache_namespace']
        assert manifest['authorization']['real_api_authorized'] is False
    assert frontier['formal_a1_ready'] is False
    assert frontier['formal_a1_authorized'] is False
    assert frontier['real_api_authorized'] is False
    assert frontier['validation_access']=='not_authorized'
    assert frontier['test_access']=='sealed'


def test_canary_abort_does_not_unlock_formal_or_heldout():
    _assert_canary_does_not_unlock_formal_or_heldout(
        load_yaml(ROOT/'experiments/current_frontier.yaml'),load_yaml(ROOT/'experiments/registry.yaml'))


@pytest.mark.parametrize('field,value',[('next_canary_attempt_id','unregistered_attempt'),
    ('next_canary_milestone','math_v2_1_a1_seed81_low_cost_pilot_v1')])
def test_pending_canary_identity_mismatch_fails_closed(field,value):
    frontier=load_yaml(ROOT/'experiments/current_frontier.yaml')
    frontier['next_canary_milestone']='math_v2_1_a1_seed81_low_cost_canary_v1'
    frontier['next_canary_attempt_id']='math_v2_1_low_cost_A1_seed81_canary_attempt1'
    frontier[field]=value
    with pytest.raises(AssertionError):
        _assert_canary_does_not_unlock_formal_or_heldout(frontier,load_yaml(ROOT/'experiments/registry.yaml'))
