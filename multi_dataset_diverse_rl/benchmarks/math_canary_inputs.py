"""Frozen initial conditions and metadata-only fresh canary membership selection."""
from copy import deepcopy
import hashlib
import re

from ..search.schemas import SearchContractError
from ..search.system_prompt import SEED, SystemPrompt
from ..versions import MATH_ARM_B_INITIAL_TEAM_VERSION as ARM_B_TEAM_VERSION
from .data_freeze import digest

ARM_B_SEED = SystemPrompt(SEED.role, SEED.strategy,
    'End your response with a final answer on a separate line.\n'
    'The line must start with the exact text "Final answer: "\n'
    'followed by only the mathematical result.\n'
    'Do not use a Markdown heading, bullet, or bold formatting\n'
    'for this line. Do not write anything after it.')
FRESH_SELECTION = dict(identity='MATH_FRESH_CANARY_MEMBERSHIP_V2',
    strata=['subject', 'level'], apportionment='integer_largest_remainder_lexical_tie',
    ordering='sha256_policy_seed_role_stable_id', output_order='source_superset_order',
    ordering_namespace='MATH_LOW_COST_SUBJECT_LEVEL_HASH_V1',
    exclusion='prior_actual_use_example_hashes', performance_used=False)
FRESH_COUNTS = dict(canary_optimize=12, pilot_shadow=40)
RECOVERY_SELECTION = {**FRESH_SELECTION, 'identity':'MATH_FRESH_RECOVERY_DEVELOPMENT_MEMBERSHIP_V3',
    'phase_bound':True,'exclusion':'unseen_first_minimum_metadata_only_prior_use_if_insufficient',
    'cross_attempt_memberships':'frozen_before_outcomes_independent_realizations'}


def build_recovery_subsets(metadata, split_sha, *, seed, phase, excluded_example_hashes):
    """Fresh development memberships only; no new split or outcome selection."""
    from .math_low_cost import choose
    if phase not in {'canary','pilot'}:
        raise SearchContractError('RECOVERY_SELECTION_PHASE_INVALID')
    # Reuse the historical selector's strict input checks, not its memberships.
    build_fresh_canary_subsets(metadata,split_sha,seed=seed,excluded_example_hashes=excluded_example_hashes)
    counts={phase+'_optimize':12 if phase=='canary' else 60,'pilot_shadow':40}
    excluded=set(excluded_example_hashes);selected={};prior_counts={}
    for name,count in counts.items():
        role='shadow' if name=='pilot_shadow' else 'optimize'
        pool=[r for r in metadata if r['project_split']==role and
            hashlib.sha256(r['stable_example_id'].encode()).hexdigest() not in excluded]
        namespace=RECOVERY_SELECTION['identity']+':'+str(seed)+':'+phase+':'+name
        fresh=choose(pool,min(count,len(pool)),namespace+':unseen')
        deficit=count-len(fresh)
        used=[r for r in metadata if r['project_split']==role and
            hashlib.sha256(r['stable_example_id'].encode()).hexdigest() in excluded]
        if len(used)<deficit:
            raise SearchContractError('RECOVERY_DEVELOPMENT_POOL_INSUFFICIENT:'+role)
        reuse=choose(used,deficit,namespace+':prior_use') if deficit else []
        ids={r['stable_example_id'] for r in fresh+reuse}
        selected[name]=[r for r in metadata if r['stable_example_id'] in ids]
        prior_counts[name]=len(reuse)
    return dict(identity=RECOVERY_SELECTION['identity'],selection_policy=deepcopy(RECOVERY_SELECTION),
        seed=seed,phase=phase,excluded_example_hashes=excluded_example_hashes,source_superset_sha256=split_sha,
        counts=counts,prior_actual_use_counts=prior_counts,metadata_universe=metadata,memberships=selected,
        membership_hashes={k:digest([r['stable_example_id'] for r in v]) for k,v in selected.items()},
        validation_access='not_authorized',test_access='sealed')


def recovery_protocol(subsets):
    return {**fresh_canary_protocol(subsets),'phase':subsets['phase'],
        'prior_actual_use_counts':deepcopy(subsets['prior_actual_use_counts'])}


def initial_prompt(version):
    from ..current_contract import MATH_INITIAL_TEAM_VERSION
    if version == MATH_INITIAL_TEAM_VERSION:
        return SEED
    if version == ARM_B_TEAM_VERSION:
        return ARM_B_SEED
    raise SearchContractError('CURRENT_INITIAL_TEAM_VERSION_UNKNOWN')


def build_fresh_canary_subsets(metadata, split_sha, *, seed, excluded_example_hashes):
    from .math_low_cost import choose
    if type(seed) is not int or seed < 0 or not isinstance(excluded_example_hashes, list):
        raise SearchContractError('FRESH_CANARY_SELECTION_INPUT_INVALID')
    if (excluded_example_hashes != sorted(set(excluded_example_hashes)) or
            any(not isinstance(v, str) or re.fullmatch('[0-9a-f]{64}', v) is None
                for v in excluded_example_hashes)):
        raise SearchContractError('FRESH_CANARY_EXCLUSION_INVALID')
    excluded = set(excluded_example_hashes)
    selected = {}
    for name, role in [('canary_optimize', 'optimize'), ('pilot_shadow', 'shadow')]:
        pool = [r for r in metadata if r['project_split'] == role and
                hashlib.sha256(r['stable_example_id'].encode()).hexdigest() not in excluded]
        if len(pool) < FRESH_COUNTS[name]:
            raise SearchContractError('FRESH_CANARY_UNSEEN_POOL_INSUFFICIENT')
        selected[name] = choose(pool, FRESH_COUNTS[name],
            FRESH_SELECTION['identity'] + ':' + str(seed) + ':' + name)
    return dict(identity=FRESH_SELECTION['identity'], selection_policy=deepcopy(FRESH_SELECTION),
        seed=seed, excluded_example_hashes=excluded_example_hashes,
        source_superset_sha256=split_sha, counts=deepcopy(FRESH_COUNTS), metadata_universe=metadata,
        memberships=selected,
        membership_hashes={k: digest([r['stable_example_id'] for r in v]) for k, v in selected.items()},
        validation_access='not_authorized', test_access='sealed')


def fresh_canary_protocol(subsets):
    return {k: deepcopy(subsets[k]) for k in ('identity', 'selection_policy', 'seed',
        'source_superset_sha256', 'counts', 'membership_hashes', 'validation_access', 'test_access')}
