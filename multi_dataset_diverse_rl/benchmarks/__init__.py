"""Benchmark contracts for the opt-in unified search path."""

from .registry import BENCHMARKS, BenchmarkSpec, benchmark_spec, benchmark_preflight
from .hotpotqa import HotpotQAAnswerAdapter
from .blocked import (HoVerBenchmarkAdapter, IFBenchBenchmarkAdapter,
                      PUPABenchmarkAdapter, MATHBenchmarkAdapter)
from .access import SplitAccessPolicy, DataPurpose, FrozenSplitReader, AdaptiveGateFeedback

__all__ = ["BENCHMARKS", "BenchmarkSpec", "benchmark_spec", "benchmark_preflight",
           "HotpotQAAnswerAdapter", "HoVerBenchmarkAdapter", "IFBenchBenchmarkAdapter",
           "PUPABenchmarkAdapter", "MATHBenchmarkAdapter", "SplitAccessPolicy",
           "DataPurpose", "FrozenSplitReader", "AdaptiveGateFeedback"]
