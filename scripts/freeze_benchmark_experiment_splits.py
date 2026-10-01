"""Composition for offline experiment split freeze/verification; no providers."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multi_dataset_diverse_rl.benchmarks.data_freeze import file_hash
from multi_dataset_diverse_rl.benchmarks.experiment_splits import freeze_experiment_split, verify_experiment_split
from multi_dataset_diverse_rl.versions import CURRENT_RESEARCH_BENCHMARK_SUITE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=CURRENT_RESEARCH_BENCHMARK_SUITE)
    parser.add_argument("--canonical-root", type=Path, default=Path("data/benchmark_suite_v1"))
    parser.add_argument("--destination", type=Path, default=Path("experiments/data_splits/benchmark_experiment_v1"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify_experiment_split(args.canonical_root, args.destination, args.benchmark,
            expected_manifest_sha256=file_hash(args.destination / (args.benchmark + ".json")))
    else:
        result = freeze_experiment_split(args.canonical_root, args.destination, args.benchmark,
            expected_canonical_sha256=file_hash(args.canonical_root / "manifests" / (args.benchmark + ".json")))
        result = {"counts": result["counts"], "overlap": result["pairwise_overlap_audit"]["status"]}
    print(json.dumps({"benchmark_id": args.benchmark, **result, "provider_attempts": 0}))

if __name__ == "__main__":
    main()
