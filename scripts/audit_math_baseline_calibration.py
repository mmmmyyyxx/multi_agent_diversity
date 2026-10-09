"""Read-only historical audit, frozen-plan dry run or closed synthetic fake run.

Use tests/formal_zero_api_runner.py --offline-command to launch this script.
There is deliberately no API execution option or provider-client injection.
"""
import argparse
import json
import os
from pathlib import Path
import re
import socket


def require_zero_api_guard():
    if (os.environ.get('FORMAL_ZERO_API_GUARD_REQUIRED')!='1'
            or socket.socket.connect.__module__!='sitecustomize'
            or any(re.search(r'API.?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTHORIZATION',name,re.I)
                for name in os.environ)):
        raise RuntimeError('CREDENTIAL_FREE_PREIMPORT_NETWORK_GUARD_REQUIRED')


def main():
    require_zero_api_guard()
    from multi_dataset_diverse_rl.diagnostics.math_baseline_audit import audit_initial
    from multi_dataset_diverse_rl.diagnostics.math_calibration_plan import validate_plan,FakeProvider,fake_recovery
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--fake-synthetic',action='store_true')
    parser.add_argument('--private-output',default='runs/math_baseline_calibration_audit_20261009/reproduction')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    if args.dry_run:
        path=root/'experiments/protocols/math_baseline_calibration_v1/protocol.json'
        plan=json.loads(path.read_bytes());validate_plan(plan)
        print(json.dumps(dict(status='DESIGN_ONLY_NO_REQUESTS',protocol_sha256=plan['protocol_sha256'],
            membership_sha256=plan['data']['membership_sha256'],n=len(plan['data']['membership']),
            real_api_authorized=False,READY_TO_RUN=False,api_calls=0)))
        return
    if args.fake_synthetic:
        example=dict(example_id='SYNTHETIC_FIXTURE_NOT_DATASET',problem='SYNTHETIC',reference='2')
        provider=FakeProvider({(arm,example['example_id'],0):[dict(text='Final answer: 2' if arm!='C' else r'\boxed{2}')]
            for arm in ('A','B','C')})
        result={arm:fake_recovery(arm,example,0,provider) for arm in ('A','B','C')}
        assert all(row['first_correct'] for row in result.values())
        print(json.dumps(dict(status='SYNTHETIC_FAKE_ONLY',fake_calls=len(provider.calls),api_calls=0)))
        return
    destination=(root/args.private_output).resolve()
    runs=(root/'runs').resolve()
    if (not destination.is_relative_to(runs) or not destination.relative_to(runs).parts
            or not destination.relative_to(runs).parts[0].startswith('math_baseline_calibration_audit_')):
        raise ValueError('NEW_PRIVATE_AUDIT_DIRECTORY_REQUIRED')
    result=audit_initial(root);destination.mkdir(parents=True,exist_ok=True)
    for key in ('attempts','logicals','errors','optimize_rows','summary','deployment'):
        suffix='_private' if key in ('attempts','logicals','errors','optimize_rows') else ''
        (destination/(key+suffix+'.json')).write_bytes((json.dumps(result[key],indent=2)+'\n').encode())
    print(json.dumps(dict(status='READ_ONLY_AUDIT_COMPLETE',summary=result['summary'],api_calls=0)))


if __name__=='__main__':main()
