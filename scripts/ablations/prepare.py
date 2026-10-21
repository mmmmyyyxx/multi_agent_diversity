"""Thin zero-API freeze entrypoint for the registered isolated study."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from multi_dataset_diverse_rl.research.ablations.aligned import prepare
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--resume-full-error-test',action='store_true',help='Test-only exact paid-prefix A2 continuation, zero API preparation.')
    parser.add_argument('--full-error-control', action='store_true', help='Fresh all-error A control, three Seed42 repeats; zero API preparation.')
    parser.add_argument('--continue-parent', type=Path)
    parser.add_argument('--c-only',action='store_true',help='Fresh reviewed C-only three-repeat experiment; no old state/cache.')
    parser.add_argument('--pattern-directions',action='store_true',help='Fresh C0/C1 sequential/C2 parallel single-Parent freeze; zero API.')
    parser.add_argument('--pattern-canary',action='store_true',help='Independent one-round C0/C1/C2 canary freeze; preparation makes no API calls.')
    parser.add_argument('--resume-c-test',action='store_true',help='Test-only continuation of the pinned scorer-aborted C Test.')
    parser.add_argument('--extend-c-test-deadline',action='store_true',help='Test-only thirty-second grading continuation of the stopped V8 attempt.')
    parser.add_argument('--certify-c-test',action='store_true',help='Test-only exact far-inequality certificate continuation.')
    parser.add_argument('--completed-ab-test',action='store_true',help='Test only the six completed historical A/B cells.')
    parser.add_argument('--resume-ab-test',action='store_true',help='Continue paid A/B Tests after local scorer repair.')
    parser.add_argument('--resume-pattern-scheduling',action='store_true',help='Frozen paid-prefix continuation with Solver96/Gradient16/Test96.')
    args = parser.parse_args()
    if args.resume_full_error_test:
        if any(v for k,v in vars(args).items() if k not in ('resume_full_error_test','destination')):
            parser.error('A2 Test continuation cannot be combined with other modes.')
        from multi_dataset_diverse_rl.research.ablations.full_error_test_resume import prepare as prepare_test
        print(json.dumps(prepare_test(ROOT,args.destination),indent=2))
        return
    if args.full_error_control:
        if any(value for key,value in vars(args).items() if key not in ('full_error_control','destination')):
            parser.error('All-error control cannot be combined with other preparation modes.')
        from multi_dataset_diverse_rl.research.ablations.full_error_control import prepare as prepare_control
        print(json.dumps(prepare_control(ROOT,args.destination),indent=2))
        return
    if args.resume_pattern_scheduling:
        if any(value for key,value in vars(args).items() if key not in ('resume_pattern_scheduling','destination')):
            parser.error('Scheduling continuation cannot be combined with other modes.')
        from multi_dataset_diverse_rl.research.ablations.proposal_continuation import prepare as resume_prepare
        print(json.dumps(resume_prepare(ROOT,args.destination),indent=2))
        return
    if args.pattern_canary:
        if any(value for key,value in vars(args).items() if key not in ('pattern_canary','destination')):
            parser.error('Canary cannot be combined with other preparation modes.')
        from multi_dataset_diverse_rl.research.ablations.proposal_canary import prepare as prepare_canary
        print(json.dumps(prepare_canary(ROOT,args.destination),indent=2))
        return
    if args.pattern_directions:
        if any((args.continue_parent,args.c_only,args.resume_c_test,args.extend_c_test_deadline,args.certify_c_test,args.completed_ab_test,args.resume_ab_test)):
            parser.error('Top-2 Pattern is a separate single-Parent experiment.')
        from multi_dataset_diverse_rl.research.ablations.proposal_governance import prepare as prepare_proposals
        print(json.dumps(prepare_proposals(ROOT,args.destination),indent=2))
        return
    if args.continue_parent:
        parser.error('Historical Batch continuation is not a current module-aligned experiment; use its frozen Git source for historical replay.')
    if args.resume_ab_test:
        if args.completed_ab_test or args.c_only or args.resume_c_test or args.extend_c_test_deadline or args.certify_c_test:parser.error('Test-only scopes cannot be combined.')
        from multi_dataset_diverse_rl.research.ablations.ab_test_continuation import prepare as prepare_ab_continuation
        value=prepare_ab_continuation(ROOT,args.destination)
    elif args.completed_ab_test:
        if args.c_only or args.resume_c_test or args.extend_c_test_deadline or args.certify_c_test:parser.error('Test-only scopes cannot be combined.')
        from multi_dataset_diverse_rl.research.ablations.completed_ab_test import prepare as prepare_ab
        value=prepare_ab(ROOT,args.destination)
    elif args.certify_c_test:
        if args.c_only or args.resume_c_test or args.extend_c_test_deadline:parser.error('Each Test continuation has its own frozen scope.')
        from multi_dataset_diverse_rl.research.ablations.test_certified import prepare as prepare_certified
        value=prepare_certified(ROOT,args.destination)
    elif args.extend_c_test_deadline:
        if args.c_only or args.resume_c_test:parser.error('Each Test continuation has its own frozen scope.')
        from multi_dataset_diverse_rl.research.ablations.test_deadline import prepare as prepare_deadline
        value=prepare_deadline(ROOT,args.destination)
    elif args.resume_c_test:
        if args.c_only:parser.error('C-only training and Test continuation are separate scopes.')
        from multi_dataset_diverse_rl.research.ablations.test_resume import prepare as prepare_test
        value=prepare_test(ROOT,args.destination)
    elif args.c_only:value = prepare(ROOT,args.destination,c_only=True)
    else:
        from multi_dataset_diverse_rl.research.ablations.population_governance import prepare as population_prepare
        value=population_prepare(ROOT,args.destination)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
