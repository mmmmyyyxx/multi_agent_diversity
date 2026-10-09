"""Flat MATH integrity checks; parent JSON is provenance, never an executor."""
import hashlib
import importlib.metadata
import json
from .data_freeze import digest, file_hash, SOURCE_PINS
from .experiment_splits import COUNTS, ROLES
from .math_low_cost import read_subsets
from .math_worker import PINS
from ..search.schemas import SearchContractError
from ..current_contract import MATH_INITIAL_TEAM_VERSION, MATH_INITIAL_PROMPT


def require(condition, category):
    if not condition:raise SearchContractError('CURRENT_DATA_'+category)


def validate_current_initial_team(team, contract):
    require(team['team_version']==contract['initial_team_version']==MATH_INITIAL_TEAM_VERSION
        and team['initial_team_data_dependency']=='NONE'
        and len(team['members'])==5
        and team['ordered_member_ids']==[m['member_id'] for m in team['members']]==list(range(5)),
        'INITIAL_TEAM_CONTRACT_MISMATCH')
    from ..search.system_prompt import SystemPrompt
    require(team['semantic_contract']==['complete editable system prompt']
        and all(SystemPrompt.from_dict(m['prompt'])==MATH_INITIAL_PROMPT for m in team['members']),
        'INITIAL_TEAM_SYMMETRY_MISMATCH')
    for member in team['members']:
        prompt=SystemPrompt.from_dict(member['prompt'])
        require(prompt.prompt_hash==member['prompt_sha256']
            and member['block_hashes']=={k:prompt.block_hash(k) for k in ('role','strategy','answer')}
            and member['rendered_system_sha256']==prompt.system_hash,
            'INITIAL_PROMPT_HASH_MISMATCH')
    require(digest([m['prompt_sha256'] for m in team['members']])==contract['initial_team_sha256']
        and team['ordered_team_sha256']==contract['initial_team_sha256'],'INITIAL_TEAM_HASH_MISMATCH')


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
    team_path=binding.path(c['initial_team_path'])
    require(file_hash(team_path)==c['initial_team_artifact_sha256'],'INITIAL_TEAM_ARTIFACT_HASH_MISMATCH')
    validate_current_initial_team(json.loads(team_path.read_bytes()),c)
    metadata=json.loads(binding.path(c['validation_accounting_metadata_path']).read_bytes())
    require(metadata==dict(identity='V24_HELDOUT_SEAL_NO_ACCESS_METADATA_V1',
        validation_model_calls=0,test_model_calls=0,heldout_accounting_reserve=0,
        validation_access='not_authorized',test_access='sealed'),
        'VALIDATION_ACCOUNTING_METADATA_MISMATCH')
    require(binding.path(c['token_ledger_directory']).is_relative_to(binding.root/'runs')
        and all(len(c[k])==64 for k in ('task_authorization_sha256','continuation_authorization_sha256')),
        'AUTHORIZATION_LEDGER_MISMATCH')
