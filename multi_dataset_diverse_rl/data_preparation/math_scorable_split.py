"""Whole-source V2 eligibility before unchanged deterministic stratification."""
from collections import Counter
import hashlib
import json
from ..benchmarks.math_domain_prep import CONTEXT,source_references
from ..benchmarks.data_freeze import canonical,digest,file_hash,immutable_write,jsonl_bytes,membership,read_jsonl
from ..benchmarks.experiment_splits import ACCESS,COUNTS,ROLES,SEED,distribution,overlap_audit,stratified_math_groups
from ..benchmarks.math_domain_v2 import SETTINGS

PROTOCOL='benchmark_experiment_split_v2'
POLICY='MATH_SCORABLE_REFERENCE_V2'


def freeze_scorable_split(canonical_root,destination,*,expected_canonical_sha256,census_path,expected_census_sha256,context):
    if context!=CONTEXT or file_hash(census_path)!=expected_census_sha256:
        raise ValueError('EVALUATOR_CENSUS_IDENTITY_MISMATCH')
    projected=list(source_references(canonical_root,expected_manifest_sha256=expected_canonical_sha256,context=context))
    audited={row['stable_example_id']:row for row in read_jsonl(census_path)}
    if len(audited)!=12500 or {r['stable_example_id'] for r in projected}!=set(audited):
        raise ValueError('FULL_REFERENCE_DOMAIN_AUDIT_REQUIRED')
    for row in projected:
        observed=audited[row['stable_example_id']]
        if observed['reference_sha256']!=hashlib.sha256((row['reference'] or '').encode()).hexdigest():
            raise ValueError('REFERENCE_CENSUS_HASH_MISMATCH')
    legacy=json.loads((canonical_root/'manifests/math.json').read_bytes())
    eligible={};summary={}
    for source in ['train','test']:
        rows=read_jsonl(canonical_root/'raw/math'/f'{source}.jsonl')
        eligible[source]=[r for r in rows if audited[r['stable_example_id']].get('scorable') is True]
        excluded=[r for r in rows if not audited[r['stable_example_id']].get('scorable')]
        summary[source]=dict(total=len(rows),eligible_count=len(eligible[source]),unsupported_count=len(excluded),
            eligible_universe_hash=digest([r['stable_example_id'] for r in eligible[source]]),
            excluded_identity_set_hash=digest(sorted(r['stable_example_id'] for r in excluded)),
            unsupported_by_subject=dict(Counter(r['content']['type'] for r in excluded)),
            unsupported_by_level=dict(Counter(r['content']['level'] for r in excluded)))
    old=json.loads((destination.parent/'benchmark_experiment_v1_1/math.json').read_bytes())
    old_members=read_jsonl(destination.parent/'benchmark_experiment_v1_1/math.ids.jsonl')
    keep=all(audited[r['stable_example_id']].get('scorable') is True for r in old_members)
    if keep:
        lookup={r['stable_example_id']:r for rows in eligible.values() for r in rows}
        groups={role:[lookup[r['stable_example_id']] for r in old_members if r['project_split']==role] for role in ROLES}
    else:
        groups=stratified_math_groups(eligible,COUNTS['math'],seed=SEED)
    overlap=overlap_audit(groups)
    if overlap['status']!='DISJOINT' or {r:len(v) for r,v in groups.items()}!=COUNTS['math']:
        raise ValueError('SCORABLE_SPLIT_CONFORMANCE_FAILURE')
    members=[]
    for role in ROLES:
        for row in groups[role]:
            entry=membership(row,role)
            entry.update(reference_sha256=audited[row['stable_example_id']]['reference_sha256'],reference_scorable=True)
            members.append(entry)
    ids=jsonl_bytes(members)
    manifest=dict(benchmark_id='math',status='FROZEN',protocol=PROTOCOL,
        canonical_dataset_manifest_sha256=expected_canonical_sha256,canonical_sources=legacy['source']['canonical_sources'],
        source_revision=legacy['source']['revision'],reference_extractor='MATH_REFERENCE_EXTRACTOR_V1',
        reference_validity_policy=POLICY,answer_domain='MATH_ANSWER_DOMAIN_V2',
        verify_settings_sha256=digest(SETTINGS),source_reference_census_sha256=expected_census_sha256,
        source_reference_audit=summary,split_seed=SEED,
        split_algorithm='whole-source scorable universe; unchanged subject proportional largest remainder; lexicographic ties; SHA256(str(seed)+stable_id); sequential disjoint roles',
        counts=COUNTS['math'],hashes={role:digest([r['stable_example_id'] for r in groups[role]]) for role in ROLES},
        membership_file='math.ids.jsonl',membership_sha256=hashlib.sha256(ids).hexdigest(),
        role_membership_sha256={role:hashlib.sha256(jsonl_bytes([r for r in members if r['project_split']==role])).hexdigest() for role in ROLES},
        pairwise_overlap_audit=overlap,access_policy=ACCESS,invalid_reference_counts_by_role=dict.fromkeys(ROLES,0),
        distributions={role:distribution('math',rows) for role,rows in groups.items()},
        existing_role_memberships_preserved=keep,canonical_dataset_changed=False,
        amendment_before_new_efficacy=True,prior_canary_results_not_used_for_eligibility=True)
    for path,raw in [(destination/'math.json',canonical(manifest)+b'\n'),(destination/'math.ids.jsonl',ids)]:
        immutable_write(path,raw)
    return manifest
