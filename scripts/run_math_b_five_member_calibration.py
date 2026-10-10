"""Independent B-only diagnostic. Paid execution requires a new frozen scope."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main():
    from multi_dataset_diverse_rl.diagnostics.math_b5_execution import freeze_review,verify_review,verify_bundle,execute_paid
    from multi_dataset_diverse_rl.diagnostics.math_b5_accounting import CalibrationAbort
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prep',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-review',action='store_true')
    mode.add_argument('--review-preflight',action='store_true')
    mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--audit-results',action='store_true')
    args=parser.parse_args();prep=(ROOT/args.prep).resolve()
    try:
        if args.audit_results:
            from multi_dataset_diverse_rl.diagnostics.math_b5_results import audit_results
            result=audit_results(ROOT,prep)
        elif args.execute:
            summary=execute_paid(ROOT,prep)
            result=dict(status=summary['status'],completed_logicals=summary['completed_logicals'],
                charged_tokens=summary['accounting']['charged_total'],provider_attempts=summary['accounting']['physical_attempts'])
        else:
            if args.freeze_review:scope=freeze_review(ROOT,prep)
            elif args.review_preflight:scope,_,_=verify_review(ROOT,prep)
            else:scope,_,_=verify_bundle(ROOT,prep)
            result={k:scope[k] for k in ('source_sha','review_scope_sha256','protocol_sha256','membership_sha256',
                'n','members','seed','authorized_total','READY_TO_RUN','real_api_authorized')}
            result['status']='SOURCE_FROZEN_AUTHORIZATION_PENDING' if scope['READY_TO_RUN'] is False else 'AUTHORIZED_READY'
            result['api_calls']=0
        print(json.dumps(result,sort_keys=True),flush=True)
        return int(result.get('status')=='EXECUTION_ABORTED')
    except (CalibrationAbort,FileNotFoundError) as error:
        category=str(error) if isinstance(error,CalibrationAbort) else 'REQUIRED_FRESH_FREEZE_OR_AUTHORIZATION_MISSING'
        print(json.dumps(dict(status='EXECUTION_BLOCKED',category=category)),flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
