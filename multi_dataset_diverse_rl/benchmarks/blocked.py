"""Compatibility import path; precise scientific holds live in protocol contracts."""
from .hover import HoVerBenchmarkAdapter
from .ifbench import IFBenchBenchmarkAdapter
from .pupa import PUPABenchmarkAdapter
from .math import MATHBenchmarkAdapter

__all__ = ["HoVerBenchmarkAdapter", "IFBenchBenchmarkAdapter", "PUPABenchmarkAdapter", "MATHBenchmarkAdapter"]
