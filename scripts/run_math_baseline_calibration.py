"""Independent governed baseline Stage 1, never a Unified optimization entrypoint."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main():
    from multi_dataset_diverse_rl.diagnostics.math_calibration_execution import verify_bundle,execute_paid
    from multi_dataset_diverse_rl.diagnostics.math_calibration_accounting import CalibrationAbort
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prep',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--execute',action='store_true')
    args=parser.parse_args();prep=args.prep if args.prep.is_absolute() else ROOT/args.prep
    try:
        if args.preflight:
            scope,_,selected=verify_bundle(ROOT,prep)
            result=dict(status='FROZEN_STAGE1_READY',source_sha=scope['source_sha'],startup_identity_sha256=scope['startup_identity_sha256'],
                n=len(selected),arms=scope['arms'],cap=scope['authorized_total'],api_calls=0)
        else:result=execute_paid(ROOT,prep)
        print(json.dumps(result,sort_keys=True),flush=True)
        return 0 if result.get('status')!='EXECUTION_ABORTED' else 1
    except CalibrationAbort as error:
        print(json.dumps(dict(status='EXECUTION_BLOCKED',category=str(error))),flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
