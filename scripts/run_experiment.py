"""Closed current experiment entrypoint; retired execution needs original source."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from multi_dataset_diverse_rl.search.schemas import SearchContractError
from multi_dataset_diverse_rl.governance.unified_execution import bound_preflight,validate_prep,execute_canary


def _load(path):
    value=yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(value,dict):raise ValueError('production manifest must be an object')
    return value


def preflight(manifest):
    if manifest.get('schema_version') == 'single_prompt_gated_network_continuation_manifest_v1':
        from multi_dataset_diverse_rl.research.gated_prompt_network import validate_operation_manifest,operation_contract
        validate_operation_manifest(manifest)
        return dict(gate='GATED_NETWORK_CONTINUATION_OFFLINE_ONLY',real_api_authorized=False,provider_attempts=0,operation_policy=operation_contract())
    if manifest.get('schema_version') == 'single_prompt_validation_gated_manifest_v2':
        from multi_dataset_diverse_rl.research.gated_prompt_governance import validate_manifest
        from multi_dataset_diverse_rl.research.gated_prompt_policy import gate_membership,call_ceiling
        reader=validate_manifest(ROOT,manifest)
        return dict(gate='VALIDATION100_GATED_OFFLINE_ONLY',real_api_authorized=False,
            provider_attempts=0,validation_content_reads=0,test_content_reads=0,
            gate_membership_sha256=gate_membership(reader)['membership_sha256'],call_ceiling=call_ceiling())
    if manifest.get('schema_version') == 'single_prompt_timeout_continuation_manifest_v1':
        from multi_dataset_diverse_rl.research.single_prompt_continuation import validate_operation_manifest, operation_contract
        from multi_dataset_diverse_rl.research.single_prompt_governance import validate_manifest, MANIFEST
        validate_operation_manifest(manifest)
        validate_manifest(ROOT, _load(ROOT/MANIFEST))
        return dict(gate='INDEPENDENT_TIMEOUT_CONTINUATION_OFFLINE_ONLY', real_api_authorized=False,
            provider_attempts=0, validation_calls=0, test_calls=0, operation_policy=operation_contract())
    if manifest.get('schema_version') == 'single_prompt_study_manifest_v1':
        from multi_dataset_diverse_rl.research.single_prompt_governance import validate_manifest
        from multi_dataset_diverse_rl.research.single_prompt_policy import call_ceiling
        validate_manifest(ROOT, manifest)
        return dict(gate='INDEPENDENT_STUDY_OFFLINE_PREPARATION_ONLY', real_api_authorized=False,
            provider_attempts=0, validation_calls=0, test_calls=0,
            call_ceilings={p: call_ceiling(p) for p in ('train', 'validation', 'test')})
    return bound_preflight(ROOT,manifest)


def governed_preflight(prep):
    from multi_dataset_diverse_rl.research import gated_prompt_governance as gated
    if _load(prep/'prep.json').get('schema_version') in (gated.PREP_SCHEMA,gated.network.PREP_SCHEMA):
        scope=gated.validate_prep(ROOT,prep)
        return dict(gate='VALIDATION100_GATED_FROZEN_NOT_AUTHORIZED',scope_identity=scope['scope_identity'],provider_attempts=0)
    from multi_dataset_diverse_rl.research import single_prompt_governance as study
    if _load(prep/'prep.json').get('schema_version') in (study.PREP_SCHEMA, study.CONTINUATION_PREP_SCHEMA):
        scope = study.validate_prep(ROOT, prep)
        return dict(gate='INDEPENDENT_STUDY_FROZEN_NOT_AUTHORIZED', phase=scope['phase'],
            token_ceiling=scope['token_ceiling'], scope_identity=scope['scope_identity'], provider_attempts=0)
    from multi_dataset_diverse_rl.governance import postfreeze_validation
    if _load(prep/'prep.json').get('schema_version')==postfreeze_validation.PREP_SCHEMA:
        value,_=postfreeze_validation.validate_prep(ROOT,prep)
        return dict(gate='VALIDATION_READY_NOT_AUTHORIZED',ready_for_authorization=True,
            attempt_id=value['scope']['attempt_id'],startup_identity_sha256=value['startup_identity_sha256'],
            provider_attempts=0,validation_calls=0,test_calls=0)
    value=validate_prep(ROOT,prep)
    phase=value['scope']['phase'].removesuffix('_search_only').upper()
    return dict(gate=phase+'_READY_NOT_AUTHORIZED',ready_for_authorization=True,
        attempt_id=value['scope']['attempt_id'],startup_identity_sha256=value['startup_identity_sha256'],
        provider_attempts=0,validation_calls=0,test_calls=0)


async def execute_frozen(prep,run_root):
    from multi_dataset_diverse_rl.research import gated_prompt_governance as gated
    if _load(prep/'prep.json').get('schema_version') in (gated.PREP_SCHEMA,gated.network.PREP_SCHEMA):
        return await asyncio.to_thread(gated.execute,ROOT,prep,run_root)
    from multi_dataset_diverse_rl.research import single_prompt_governance as study
    if _load(prep/'prep.json').get('schema_version') in (study.PREP_SCHEMA, study.CONTINUATION_PREP_SCHEMA):
        return await asyncio.to_thread(study.execute, ROOT, prep, run_root)
    from multi_dataset_diverse_rl.governance import postfreeze_validation
    if _load(prep/'prep.json').get('schema_version')==postfreeze_validation.PREP_SCHEMA:
        return await postfreeze_validation.execute(ROOT,prep,run_root)
    return await execute_canary(ROOT,prep,run_root)


async def _execute(manifest):
    raise SearchContractError('ABORT_PRE_PROVIDER: FROZEN_EXECUTION_GOVERNANCE_NOT_BOUND')


def main():
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest',type=Path);source.add_argument('--prep',type=Path)
    parser.add_argument('--run-root',type=Path)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    def resolve(path):return path if path is None or path.is_absolute() else ROOT/path
    if args.preflight:
        result=governed_preflight(resolve(args.prep)) if args.prep else preflight(_load(resolve(args.manifest)))
    else:
        if args.prep is None or args.run_root is None:raise SearchContractError('ABORT_PRE_PROVIDER: frozen prep and fresh run root are required')
        result=asyncio.run(execute_frozen(resolve(args.prep),resolve(args.run_root)))
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
