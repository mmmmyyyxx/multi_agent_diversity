"""Metadata-only proportional development subsets of the immutable MATH split."""
from collections import Counter,defaultdict
import hashlib,json
from .data_freeze import digest,file_hash
from .experiment_splits import quotas
from ..search.schemas import SearchContractError
from .. import versions

SELECTION=dict(identity='MATH_LOW_COST_SUBJECT_LEVEL_HASH_V1',strata=['subject','level'],
    apportionment='integer_largest_remainder_lexical_tie',minimum_per_stratum=0,
    ordering='sha256_identity_role_stable_id',output_order='source_superset_order',
    performance_used=False)
COUNTS=dict(canary_optimize=12,pilot_optimize=60,pilot_shadow=40,pilot_validation=100)


def choose(rows,count,role):
    pools=defaultdict(list)
    for row in rows:pools[(row['subject'],row['level'])].append(row)
    allocated=quotas({k:len(v) for k,v in pools.items()},count)
    ids=set()
    for stratum in sorted(pools):
        ordered=sorted(pools[stratum],key=lambda r:(hashlib.sha256(
            (SELECTION['identity']+':'+role+':'+r['stable_example_id']).encode()).hexdigest(),r['stable_example_id']))
        ids.update(r['stable_example_id'] for r in ordered[:allocated[stratum]])
    return [r for r in rows if r['stable_example_id'] in ids]


def build_subsets(metadata,split_sha):
    groups={role:[r for r in metadata if r['project_split']==role] for role in ('optimize','shadow','validation')}
    if {k:len(v) for k,v in groups.items()} != dict(optimize=150,shadow=300,validation=300):
        raise SearchContractError('LOW_COST_SUPERSET_COUNTS_MISMATCH')
    selected=dict(pilot_optimize=choose(groups['optimize'],60,'pilot_optimize'),
        pilot_shadow=choose(groups['shadow'],40,'pilot_shadow'),
        pilot_validation=choose(groups['validation'],100,'pilot_validation'))
    selected['canary_optimize']=choose(selected['pilot_optimize'],12,'canary_optimize')
    return dict(identity=versions.MATH_LOW_COST_PROTOCOL_VERSION,selection_policy=SELECTION,
        source_superset_sha256=split_sha,counts=COUNTS,metadata_universe=metadata,
        memberships=selected,membership_hashes={k:digest([r['stable_example_id'] for r in v]) for k,v in selected.items()},
        test_access='sealed')


def read_subsets(root,contract):
    p=root/contract['low_cost_subsets_path']
    if file_hash(p)!=contract['low_cost_subsets_sha256']:
        raise SearchContractError('LOW_COST_SUBSET_ARTIFACT_HASH_MISMATCH')
    data=json.loads(p.read_bytes())
    if data != build_subsets(data['metadata_universe'],contract['split_manifest_sha256']):
        raise SearchContractError('LOW_COST_SUBSET_SELECTION_MISMATCH')
    return data


def protocol_contract(subsets):
    return dict(identity=versions.MATH_LOW_COST_PROTOCOL_VERSION,counts=COUNTS,
        selection_policy=SELECTION,source_superset_sha256=subsets['source_superset_sha256'],
        membership_hashes=subsets['membership_hashes'],canary_stop='one_complete_production_opportunity_v1',
        pilot_stop='team_epoch_no_commit_v1',test_access='sealed',scientific_scope='development_only',
        analysis_policy=dict(identity='MATH_V2_1_PILOT_DEVELOPMENT_ANALYSIS_V1',
            classification='sign_of_final_minus_initial_VoteAcc',resamples=10000,seed=81,
            paired_resampling_unit='frozen_validation_example',scope='development_uncertainty_only_no_formal_significance_no_search_feedback'))
