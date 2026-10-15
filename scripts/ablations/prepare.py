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
    parser.add_argument('--continue-parent', type=Path)
    parser.add_argument('--c-only',action='store_true',help='Fresh reviewed C-only three-repeat experiment; no old state/cache.')
    args = parser.parse_args()
    if args.continue_parent:
        parser.error('Historical Batch continuation is not a current module-aligned experiment; use its frozen Git source for historical replay.')
    value = prepare(ROOT,args.destination,c_only=args.c_only)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
