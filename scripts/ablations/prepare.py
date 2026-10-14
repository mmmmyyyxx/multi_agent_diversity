"""Thin zero-API freeze entrypoint for the registered isolated study."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from multi_dataset_diverse_rl.research.ablations.governance import prepare
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(ROOT, args.destination), indent=2))


if __name__ == '__main__':
    main()
