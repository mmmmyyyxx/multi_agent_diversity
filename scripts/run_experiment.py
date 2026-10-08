"""Current-only experiment entrypoint; historical execution uses replay_experiment.py."""
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
    return bound_preflight(ROOT,manifest)


def governed_preflight(prep):
    value=validate_prep(ROOT,prep)
    return dict(gate=('PILOT_READY_NOT_AUTHORIZED' if value['scope'].get('phase')=='pilot_search_only'
        else 'CANARY_READY_NOT_AUTHORIZED'),ready_for_authorization=True,
        attempt_id=value['scope']['attempt_id'],startup_identity_sha256=value['startup_identity_sha256'],
        provider_attempts=0,validation_calls=0,test_calls=0)


async def execute_frozen(prep,run_root):return await execute_canary(ROOT,prep,run_root)


async def _execute(manifest):
    raise SearchContractError('ABORT_PRE_PROVIDER: FROZEN_EXECUTION_GOVERNANCE_NOT_BOUND')


def main():
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest',type=Path);source.add_argument('--prep',type=Path)
    source.add_argument('--manual-probe-prep',type=Path)
    parser.add_argument('--run-root',type=Path)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    def resolve(path):return path if path is None or path.is_absolute() else ROOT/path
    if args.manual_probe_prep is not None:
        from multi_dataset_diverse_rl.diagnostics.manual_capacity import preflight as probe_preflight, execute as probe_execute
        if args.execute and args.run_root is None:
            raise SearchContractError('MANUAL_PROBE_FRESH_RUN_ROOT_REQUIRED')
        result=(probe_preflight(ROOT,resolve(args.manual_probe_prep)) if args.preflight
            else probe_execute(ROOT,resolve(args.manual_probe_prep),resolve(args.run_root)))
        print(json.dumps(result,indent=2,sort_keys=True));return
    if args.preflight:
        result=governed_preflight(resolve(args.prep)) if args.prep else preflight(_load(resolve(args.manifest)))
    else:
        if args.prep is None or args.run_root is None:raise SearchContractError('ABORT_PRE_PROVIDER: frozen prep and fresh run root are required')
        result=asyncio.run(execute_frozen(resolve(args.prep),resolve(args.run_root)))
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
