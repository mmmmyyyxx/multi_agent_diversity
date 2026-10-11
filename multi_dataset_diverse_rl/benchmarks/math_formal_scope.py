"""Finite formal execution scope; no change to search or admission semantics."""
from copy import deepcopy
from .data_freeze import digest
from ..search.schemas import SearchContractError

BOUND = dict(identity='MATH_FORMAL_SEARCH_RESOURCE_BOUND_V1',
    max_opportunities=50, token_ceiling=30_000_000,
    scientific_stopper='team_epoch_no_commit_v1', guarantees_saturation=False)
ACCOUNTING_SCOPE = 'FRESH_V25_FORMAL_SINGLE_ARM_30M_V1'
SELECTION = dict(identity='MATH_FORMAL_FULL_OPTIMIZE_METADATA_MEMBERSHIP_V1',
    optimize='all_frozen_optimize_150', shadow='metadata_stratified_seed_hash_40',
    validation='all_frozen_validation_300_post_search_only',
    performance_used=False, output_order='source_superset_order')


def require_bound(value):
    if (value != BOUND or not isinstance(value, dict)
            or any(type(value[k]) is not type(v) for k,v in BOUND.items())):
        raise SearchContractError('FORMAL_SEARCH_RESOURCE_BOUND_NOT_FROZEN')


def build_subsets(metadata, split_sha, *, seed):
    from .math_low_cost import choose
    if type(seed) is not int or seed < 0:
        raise SearchContractError('FORMAL_SELECTION_SEED_INVALID')
    groups={role:[r for r in metadata if r['project_split']==role]
        for role in ('optimize','shadow','validation')}
    if ({k:len(v) for k,v in groups.items()} != dict(optimize=150,shadow=300,validation=300)
            or len({r['stable_example_id'] for rows in groups.values() for r in rows}) != 750):
        raise SearchContractError('FORMAL_SELECTION_UNIVERSE_MISMATCH')
    selected=dict(formal_optimize=groups['optimize'],
        formal_shadow=choose(groups['shadow'],40,SELECTION['identity']+':'+str(seed)+':shadow'))
    return dict(identity=SELECTION['identity'],selection_policy=deepcopy(SELECTION),
        seed=seed,phase='formal',source_superset_sha256=split_sha,
        counts=dict(formal_optimize=150,formal_shadow=40),metadata_universe=metadata,
        memberships=selected,membership_hashes={k:digest([r['stable_example_id'] for r in v])
            for k,v in selected.items()},
        validation_membership_sha256=digest([r['stable_example_id'] for r in groups['validation']]),
        validation_access='not_authorized',test_access='sealed')


def protocol(subsets):
    return {k:deepcopy(subsets[k]) for k in ('identity','selection_policy','seed','phase',
        'source_superset_sha256','counts','membership_hashes','validation_membership_sha256',
        'validation_access','test_access')}
