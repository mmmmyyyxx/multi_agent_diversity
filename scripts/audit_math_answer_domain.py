"""Compose isolated offline reference census; no experiment/provider execution."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from multi_dataset_diverse_rl.data_preparation.math_domain_census import run_census


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical-root',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--native',action='store_true')
    args=parser.parse_args()
    if not args.destination.resolve().is_relative_to(ROOT/'runs'):
        raise ValueError('PRIVATE_PREPARATION_DESTINATION_REQUIRED')
    print(json.dumps(run_census(args.canonical_root,expected_manifest_sha256=args.manifest_sha256,
        destination=args.destination,native=args.native)))


if __name__=='__main__': main()
