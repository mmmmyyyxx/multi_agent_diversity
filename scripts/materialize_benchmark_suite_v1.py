"""Composition entrypoint for public data downloads and offline verification."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multi_dataset_diverse_rl.benchmarks.data_freeze import BENCHMARK_IDS, freeze_local, freeze_suite, verify_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--benchmark", choices=BENCHMARK_IDS)
    choice.add_argument("--all", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "data/benchmark_suite_v1")
    args = parser.parse_args()
    for benchmark in BENCHMARK_IDS if args.all else (args.benchmark,):
        try:
            if args.download:
                from multi_dataset_diverse_rl.benchmarks.data_sources import download_source
                sources, metadata = download_source(args.root, benchmark)
                result = freeze_local(args.root, benchmark, sources, metadata)
                print(json.dumps({"benchmark_id": benchmark, "counts": result["counts"], "isolation": result["isolation"]["status"]}), flush=True)
            else:
                print(json.dumps(verify_only(args.root, benchmark)), flush=True)
        except Exception as exc:
            # Upstream CSV/parser exceptions can include private row values.
            print(json.dumps({"benchmark_id": benchmark, "status": "FAILED_CLOSED", "error_type": type(exc).__name__,
                              "provider_attempts": 0, "raw_content_logged": False}), file=sys.stderr)
            return 1
    if args.all:
        if args.download:
            freeze_suite(args.root)
        else:
            from multi_dataset_diverse_rl.benchmarks.data_freeze import file_hash
            suite = json.loads((args.root / "manifests/suite.json").read_text(encoding="utf-8"))
            for benchmark, sha in suite["benchmark_manifest_sha256"].items():
                if file_hash(args.root / "manifests" / (benchmark + ".json")) != sha:
                    raise ValueError("SUITE_MANIFEST_HASH_MISMATCH")


if __name__ == "__main__":
    raise SystemExit(main())
