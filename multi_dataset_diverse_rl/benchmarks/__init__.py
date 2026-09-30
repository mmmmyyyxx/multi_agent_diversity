"""Benchmark contracts for the opt-in unified search path."""

from .registry import BENCHMARKS, BenchmarkSpec, benchmark_spec, benchmark_preflight
from .hotpotqa import HotpotQAAnswerAdapter
from .blocked import (HoVerBenchmarkAdapter, IFBenchBenchmarkAdapter,
                      PUPABenchmarkAdapter, MATHBenchmarkAdapter)

__all__ = ["BENCHMARKS", "BenchmarkSpec", "benchmark_spec", "benchmark_preflight",
           "HotpotQAAnswerAdapter", "HoVerBenchmarkAdapter", "IFBenchBenchmarkAdapter",
           "PUPABenchmarkAdapter", "MATHBenchmarkAdapter"]
