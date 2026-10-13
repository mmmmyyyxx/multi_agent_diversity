"""Offline phase freeze; this command creates no API authorization."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from multi_dataset_diverse_rl.research.single_prompt_governance import prepare
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('canary', 'train', 'validation', 'test'), required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--token-ceiling', type=int)
    parser.add_argument('--search-root', type=Path)
    parser.add_argument('--continuation-root', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(ROOT, args.destination, phase=args.phase,
        token_ceiling=args.token_ceiling, search_root=args.search_root,
        continuation_root=args.continuation_root), indent=2))


if __name__ == '__main__':
    main()
