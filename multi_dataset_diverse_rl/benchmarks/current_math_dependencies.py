"""Flat MATH integrity checks; parent JSON is provenance, never an executor."""
import hashlib
import importlib.metadata
import json
from .data_freeze import digest, file_hash, SOURCE_PINS
from .experiment_splits import COUNTS, ROLES
from .math_low_cost import read_subsets
from .math_worker import PINS
from ..search.schemas import SearchContractError
from ..evaluation.mutable_prompt_contract import validate_mutable_decision_procedure


def require(condition, category):
    if not condition:raise SearchContractError('CURRENT_DATA_'+category)


def validate_effective_math_dependencies(binding):
    """Preserve frozen data, environment, team and accounting preconditions."""
    c=binding.contract
    try:
        require(c['evaluator_pins']==PINS and all(importlib.metadata.version(k)==v for k,v in PINS.items()),
            'EVALUATOR_DEPENDENCY_IDENTITY_MISMATCH')
        require(c['provider_sdk']=={'name':'openai','version':importlib.metadata.version('openai')},
            'PROVIDER_SDK_IDENTITY_MISMATCH')
    except importlib.metadata.PackageNotFoundError as error:
        raise SearchContractError('CURRENT_DATA_DEPENDENCY_UNAVAILABLE') from error
    canonical_path=binding.path(c['canonical_root'])/'manifests/math.json'
    require(file_hash(canonical_path)==c['canonical_manifest_sha256'],'CANONICAL_MANIFEST_IDENTITY_MISMATCH')
    canonical=json.loads(canonical_path.read_bytes())
    require(all(canonical['source'].get(k)==v for k,v in SOURCE_PINS['math'].items()),'CANONICAL_SOURCE_IDENTITY_MISMATCH')
    # Hashing sealed files checks integrity; no held-out records are parsed.
    for source in canonical['source']['canonical_sources']:
        require(file_hash(binding.path(c['canonical_root'])/'raw/math'/(source['name']+'.jsonl'))==source['canonical_sha256'],
            'CANONICAL_SOURCE_HASH_MISMATCH')
    reader=binding.reader()
    require(reader.manifest['counts']==COUNTS['math'] and reader.manifest['hashes']==c['membership_hashes']
        and reader.manifest['canonical_dataset_manifest_sha256']==c['canonical_manifest_sha256'],'SPLIT_MISMATCH')
    for role in ROLES:
        rows=[r for r in reader.members if r['project_split']==role]
        require(len(rows)==COUNTS['math'][role] and digest([r['stable_example_id'] for r in rows])==c['membership_hashes'][role]
            and all(r['source_split']==('test' if role=='test' else 'train') for r in rows),'MEMBERSHIP_MISMATCH')
        for other in ROLES:
            if other==role:continue
            peers=[r for r in reader.members if r['project_split']==other]
            require(not any({r[k] for r in rows}&{r[k] for r in peers} for k in ('stable_example_id','content_sha256','input_sha256')),
                'SPLIT_OVERLAP')
    subsets=read_subsets(binding.root,c)
    original={r['stable_example_id']:r for r in reader.members}
    fields=('stable_example_id','source_split','source_index','content_sha256','input_sha256','project_split')
    require(all(all(r[k]==original[r['stable_example_id']][k] for k in fields) for r in subsets['metadata_universe']),
        'LOW_COST_SUPERSET_MEMBERSHIP_MISMATCH')
    team=json.loads(binding.path(c['initial_team_path']).read_bytes())
    require(team['team_version']==c['initial_team_version'] and team['initial_team_data_dependency']=='NONE'
        and [m['member_id'] for m in team['members']]==list(range(5)),'INITIAL_TEAM_CONTRACT_MISMATCH')
    for member in team['members']:
        validate_mutable_decision_procedure(member['prompt'])
        require(hashlib.sha256(member['prompt'].encode()).hexdigest()==member['prompt_sha256'],'INITIAL_PROMPT_HASH_MISMATCH')
    require(len({m['prompt'] for m in team['members']})==5
        and digest([m['prompt_sha256'] for m in team['members']])==c['initial_team_sha256']
        and team['ordered_team_sha256']==c['initial_team_sha256'],'INITIAL_TEAM_HASH_MISMATCH')
    metadata=json.loads(binding.path(c['validation_accounting_metadata_path']).read_bytes())
    rows=subsets['memberships']['pilot_validation']
    require([(r['example_id'],r['input_sha256']) for r in metadata['examples']]==[(r['stable_example_id'],r['input_sha256']) for r in rows]
        and all(type(r['blank_prompt_serialized_request_bytes']) is int and r['blank_prompt_serialized_request_bytes']>0 for r in metadata['examples'])
        and all(metadata.get(k)==c[k] for k in ('invalid_recovery_policy','prediction_validity_policy','solver_decoding_policy',
            'decoding','solver_output_interface','low_cost_protocol')),'VALIDATION_ACCOUNTING_METADATA_MISMATCH')
    require(binding.path(c['token_ledger_directory']).is_relative_to(binding.root/'runs')
        and all(len(c[k])==64 for k in ('task_authorization_sha256','continuation_authorization_sha256')),
        'AUTHORIZATION_LEDGER_MISMATCH')
