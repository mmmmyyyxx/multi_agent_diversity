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
            # Model a concrete frozen Canary; the V2.2 frontier has none.
            result['canary_manifest']='experiments/manifests/math_v2_1_gradient_pattern_seed81_canary_v3.yaml'
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


def assert_closed_current_scope(frontier,registry):
    assert frontier['current_method']=='unified_team_prompt_search_v2_5_flexible_answer_v1'
    assert frontier['real_execution_ready'] is False
    assert frontier['real_api_authorized'] is False
    assert frontier['validation_access']=='not_authorized' and frontier['test_access']=='sealed'
    for field in ('next_canary_authorized','next_pilot_authorized','next_validation_authorized',
                  'pending_pilot_validation_authorized','pending_pilot_test_authorized',
                  'pilot_search_complete','pilot_validation_complete','old_evidence_reuse_allowed'):
        assert frontier[field] is False
    assert frontier['pending_pilot_operational_retry_limit']==0
    assert frontier['initial_accuracy'] in {
        'UNMEASURED_UNDER_FLEXIBLE_PARSER', 'MEASURED_IN_FROZEN_CANARY_NOT_REUSABLE'}
    if frontier['initial_accuracy']=='MEASURED_IN_FROZEN_CANARY_NOT_REUSABLE':
        assert frontier['authorization_scopes_closed'] is True
        assert frontier['unresolved_reservations']==0
        assert frontier['canary_manifest'] is None
        evidence=frontier['initial_accuracy_report']
        assert evidence==frontier['current_canary_report']+'/initial_metrics.json'
        row=next(r for r in registry['experiments'] if r['report']==frontier['current_canary_report'])
        assert row['active_for_new_work'] is False
        assert row['authorization_consumed'] is True and row['authorization_scope_closed'] is True
        metrics=json.loads((ROOT/evidence).read_text(encoding='utf-8'))
        assert metrics['validity']=='INDEPENDENTLY_VERIFIED_INITIAL_OBSERVATIONS'
        assert metrics['interpretation']=='DESCRIPTIVE_OPTIMIZE12_ONLY_NO_GENERALIZATION'
    active={r['experiment_id'] for r in registry['experiments'] if r['active_for_new_work']}
    if frontier['canary_manifest'] is None:
        assert frontier['current_experiment']=='NO_AUTHORIZED_REAL_EXPERIMENT'
        assert frontier['current_execution_binding'] is None
        assert active=={'math_flexible_answer_parser_v3'}
        m=load_yaml(ROOT/'experiments/manifests/math_flexible_answer_parser_v3.yaml')
        assert m['execution_binding']['path'] is None and m['execution_binding']['sha256'] is None
    else:
        eid=frontier['current_experiment']
        row=next(r for r in registry['experiments'] if r['experiment_id']==eid)
        assert active=={'math_flexible_answer_parser_v3',eid}
        assert row['manifest']==frontier['canary_manifest']
        m=load_yaml(ROOT/row['manifest'])
        assert m['execution_binding']['path']==frontier['current_execution_binding']
        assert isinstance(m['execution_binding']['sha256'],str) and len(m['execution_binding']['sha256'])==64
    assert m['authorization']['real_api_authorized'] is False
    assert m['lifecycle']['status'] in {'DRAFT','PREEXECUTION_FROZEN'}
    if m['lifecycle']['status']=='DRAFT':
        assert m['source_sha'] is None
    else:
        assert isinstance(m['source_sha'],str) and len(m['source_sha'])==40
    assert m['execution_binding']['identity']=='MATH_FLEXIBLE_ANSWER_EVIDENCE_BINDING_V3'
    assert m['method_identity']==frontier['current_method']


def test_current_scope_is_fresh_and_unapproved(registry):
    assert_closed_current_scope(load_yaml(ROOT/'experiments/current_frontier.yaml'),registry)


@pytest.mark.parametrize('field,value',[
    ('real_api_authorized',True),('next_canary_authorized',True),('next_pilot_authorized',True),
    ('next_validation_authorized',True),('pending_pilot_test_authorized',True),
    ('pending_pilot_operational_retry_limit',1),('old_evidence_reuse_allowed',True),
    ('current_method','unified_team_prompt_search_v2_3_optimization_evidence'),
    ('initial_accuracy',0.8),('pilot_search_complete',True)])
def test_preparation_cannot_authorize_or_import_results(registry,field,value):
    frontier=deepcopy(load_yaml(ROOT/'experiments/current_frontier.yaml'))
    assert_closed_current_scope(frontier,registry)
    frontier[field]=value
    with pytest.raises(AssertionError):assert_closed_current_scope(frontier,registry)
