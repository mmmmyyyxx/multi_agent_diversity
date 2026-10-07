"""Compliance completion never chooses semantic supports or changes same-F."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import pytest
from multi_dataset_diverse_rl.search.partition_completion import (
    IDENTITY, complete_known_alias_partition, completion_statistics)
from multi_dataset_diverse_rl.search.textual_gradients import (
    GradientClusterProvider, score_gradient_partition, validate_gradient)
from multi_dataset_diverse_rl.search.schemas import EvidenceItem, SearchContractError
from multi_dataset_diverse_rl.current_contract import TARGET_CORRECTNESS_SIGNAL_VERSION

TEXT='Check constraints before transforming intermediate expressions.'
def rows():
    return tuple(EvidenceItem(f'id{i}','optimize',frozenset({'REPAIR'}),dict(
        input_payload='A synthetic symbolic expression.',gold='12347',target_output='98765',
        target_member_valid=True,target_member_correct=False,
        correctness_signal_identity=TARGET_CORRECTNESS_SIGNAL_VERSION,
        responsibility_labels=('direct_flip',) if i==1 else ('coverage',),
        team_margin=1,team_disagreement=2)) for i in range(4))
def part(ids=('e1',),unassigned=()):
    return dict(patterns=[dict(generalized_gradient=TEXT,support_ids=list(ids))],unassigned_ids=list(unassigned))
def normalize(raw,aliases=('e1','e2','e3','e4')):
    return complete_known_alias_partition(raw,aliases,
        validate_generalized=lambda text:validate_gradient(text,rows(),generalized=True))

@pytest.mark.parametrize('ids,expected',[(('e1','e2','e3'),['e4']), (('e3',),['e1','e2','e4'])])
def test_missing_only_preserves_original_and_appends_in_input_order(ids,expected):
    raw=part(ids);saved=deepcopy(raw);normalized,audit=normalize(raw)
    assert raw==saved and normalized['patterns']==raw['patterns']
    assert normalized['unassigned_ids']==expected
    assert audit['completion_applied'] and audit['missing_aliases']==expected
    assert audit['duplicate_count']==audit['unknown_count']==0
    assert audit['normalized_membership_count']==4

def test_existing_unassigned_order_preserved_and_input_order_not_lexical():
    normalized,audit=normalize(part(('e1',),('e4',)),('e3','e1','e4','e2'))
    assert normalized['unassigned_ids']==['e4','e3','e2']
    assert audit['missing_aliases']==['e3','e2']

@pytest.mark.parametrize('mutation',['duplicate','overlap','unassigned_overlap','unassigned_duplicate',
    'unknown','unknown_unassigned','empty','extra','schema','bad_gradient','leak','nonstring'])
def test_omission_does_not_mask_any_other_defect(mutation):
    raw=part()
    if mutation=='duplicate':raw['patterns'][0]['support_ids'].append('e1')
    if mutation=='overlap':raw['patterns'].append(deepcopy(raw['patterns'][0]))
    if mutation=='unassigned_overlap':raw['unassigned_ids']=['e1']
    if mutation=='unassigned_duplicate':raw['unassigned_ids']=['e2','e2']
    if mutation=='unknown':raw['patterns'][0]['support_ids'].append('foreign')
    if mutation=='unknown_unassigned':raw['unassigned_ids']=['foreign']
    if mutation=='empty':raw['patterns'][0]['support_ids']=[]
    if mutation=='extra':raw['patterns'][0]['confidence']=1
    if mutation=='schema':raw['other']=[]
    if mutation=='bad_gradient':raw['patterns'][0]['generalized_gradient']='x'*601
    if mutation=='leak':raw['patterns'][0]['generalized_gradient']='Use the given value 12347.'
    if mutation=='nonstring':raw['patterns'][0]['support_ids']=[None]
    saved=deepcopy(raw)
    with pytest.raises(SearchContractError) as exc:normalize(raw)
    assert raw==saved and not exc.value.partition_completion_audit['completion_applied']

def test_complete_partition_is_identical_and_not_completed():
    raw=part(('e1','e3'),('e4','e2'));normalized,audit=normalize(raw)
    assert normalized==raw and normalized is not raw
    assert audit['raw_complete'] and not audit['completion_applied']
    assert audit['raw_partition_sha256']==audit['normalized_partition_sha256']

def decoded(value):
    out=deepcopy(value);mapping={f'e{i+1}':r.example_id for i,r in enumerate(rows())}
    for p in out['patterns']:p['support_ids']=[mapping[x] for x in p['support_ids']]
    out['unassigned_ids']=[mapping[x] for x in out['unassigned_ids']]
    return out
def score(value,evidence=None):
    evidence=evidence or rows()
    return score_gradient_partition(decoded(value),evidence,
        [dict(example_id=r.example_id,gradient=TEXT) for r in evidence])

def test_all_missing_empty_patterns_retains_original_not_actionable():
    normalized,audit=normalize(dict(patterns=[],unassigned_ids=[]))
    assert normalized==dict(patterns=[],unassigned_ids=['e1','e2','e3','e4'])
    assert audit['missing_count']==4
    with pytest.raises(SearchContractError,match='NOT_ACTIONABLE'):score(normalized)

def test_high_F_missing_sample_does_not_change_any_pattern_F_or_selection():
    raw=part(('e1','e3'));normalized,_=normalize(raw)
    explicit=part(('e1','e3'),('e2','e4'))
    a=score(normalized);b=score(explicit)
    assert a==b and a['Coverage']==0.5 and a['selected_pattern_responsibility']==2
    changed=tuple(replace(r,signals={**r.signals,'responsibility_labels':('direct_flip','near_margin','coverage')})
        if i==1 else r for i,r in enumerate(rows()))
    c=score(normalized,changed)
    assert c['patterns']==a['patterns'] and c['focus_mechanism_id']==a['focus_mechanism_id']

def test_provider_one_call_and_separate_raw_normalized_records_no_provenance_wire():
    events=[];wire=[]
    provider=GradientClusterProvider(None,partition_completion_policy=IDENTITY,partition_writer=events.append)
    def one(payload):
        wire.append(deepcopy(payload));provider.calls+=1;return part(('e1','e3'))
    provider._call=one
    result=provider.cluster({'gradients':[dict(example_id=r.example_id,gradient=TEXT) for r in rows()]},evidence_rows=rows())
    assert len(wire)==len(events)==provider.calls==1
    assert set(wire[0])=={'gradients'} and all(set(g)=={'example_id','gradient'} for g in wire[0]['gradients'])
    assert result==decoded(part(('e1','e3'),('e2','e4')))
    assert events[0]['raw_partition']==part(('e1','e3'))
    assert events[0]['normalized_controller_partition']==part(('e1','e3'),('e2','e4'))
    stats=completion_statistics(provider.partition_audit)
    assert stats['cluster_calls']==stats['partition_completion_events']==1
    assert stats['total_missing_aliases_completed']==2

def test_writer_failure_never_redraws_or_scores():
    provider=GradientClusterProvider(None,partition_completion_policy=IDENTITY,
        partition_writer=lambda _:(_ for _ in ()).throw(OSError('critical write')))
    provider._call=lambda _:part()
    with pytest.raises(OSError):provider.cluster({'gradients':[dict(example_id=r.example_id,gradient=TEXT) for r in rows()]},evidence_rows=rows())
    assert provider.partition_audit==[]

@pytest.mark.parametrize('bad_alias,field',[('e1','duplicate_partition_failures'),('foreign','unknown_alias_failures')])
def test_failure_telemetry_preserves_raw_and_has_no_normalized_partition(bad_alias,field):
    events=[];provider=GradientClusterProvider(None,partition_completion_policy=IDENTITY,partition_writer=events.append)
    raw=part();raw['patterns'][0]['support_ids'].append(bad_alias)
    provider._call=lambda _:deepcopy(raw)
    with pytest.raises(SearchContractError):
        provider.cluster({'gradients':[dict(example_id=r.example_id,gradient=TEXT) for r in rows()]},evidence_rows=rows())
    assert events[0]['raw_partition']==raw and events[0]['normalized_controller_partition'] is None
    assert completion_statistics(provider.partition_audit)[field]==1

def test_normalization_journal_1005_cycles_durable_and_raw_immutable(tmp_path):
    from multi_dataset_diverse_rl.persistence.durable_io import append_jsonl
    path=tmp_path/'normalization.jsonl';raw=part();saved=deepcopy(raw)
    for i in range(1005):
        normalized,audit=normalize(raw)
        append_jsonl(path,dict(index=i,raw_partition=raw,normalized_controller_partition=normalized,audit=audit))
    records=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
    assert len(records)==1005 and raw==saved
    assert [r['index'] for r in records]==list(range(1005))
    assert all(r['raw_partition']==saved and r['normalized_controller_partition']['unassigned_ids']==['e2','e3','e4'] for r in records)

def test_policy_unknown_rejected_before_provider():
    with pytest.raises(SearchContractError,match='POLICY_MISMATCH'):
        GradientClusterProvider(None,partition_completion_policy='relaxed')

def test_missing_provenance_rejected_before_provider():
    provider=GradientClusterProvider(None,partition_completion_policy=IDENTITY)
    provider._call=lambda _:pytest.fail('provider called before admission')
    with pytest.raises(SearchContractError,match='PROVENANCE_REQUIRED'):
        provider.cluster({'gradients':[dict(example_id=r.example_id,gradient=TEXT) for r in rows()]})

@pytest.mark.parametrize('mutation',['manifest_only','mechanism_only','unknown','mixed'])
def test_manifest_completion_identity_must_match_effective_method(mutation):
    from multi_dataset_diverse_rl.governance.repository import validate_manifest_v2
    from multi_dataset_diverse_rl.governance.registries import load_yaml
    root=Path(__file__).resolve().parents[2]
    manifest=load_yaml(root/'experiments/manifests/math_v2_1_gradient_pattern_seed81_pilot_v4.yaml')
    assert not validate_manifest_v2(root,manifest)
    if mutation=='manifest_only':manifest['mechanism_config'].pop('partition_completion_policy')
    if mutation=='mechanism_only':manifest.pop('partition_completion_policy')
    if mutation=='unknown':manifest['partition_completion_policy']='other'
    if mutation=='mixed':manifest['mechanism_config']['partition_completion_policy']='other'
    assert validate_manifest_v2(root,manifest)

@pytest.mark.parametrize('field,value', [('pattern_policy',{}),('memory_limits',{}),('provider_bounds',{}),
    ('gradient_prompt_sha256','0'*64),('partition_completion_policy','other')])
def test_fresh_binding_rejects_scientific_drift(field,value):
    from multi_dataset_diverse_rl.benchmarks.partition_completion_contract import validate_partition_completion
    from multi_dataset_diverse_rl.benchmarks.math_gradient_pattern_binding import MATHGradientPatternBinding
    root=Path(__file__).resolve().parents[2]
    c=json.loads((root/'experiments/execution_bindings/math_v2_1_gradient_pattern_seed81_pilot_v4.json').read_bytes())
    c[field]=value
    with pytest.raises(SearchContractError):validate_partition_completion(MATHGradientPatternBinding(root,c))
