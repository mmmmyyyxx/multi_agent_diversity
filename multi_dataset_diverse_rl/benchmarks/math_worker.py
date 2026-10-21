"""Offline math-verify worker; the parent owns the portable process deadline.

math-verify 0.6.0's Windows timeout closure is not spawn-pickleable. Replacing
only its timeout decorators inside this disposable process preserves parser
and grader code; the parent kills the whole worker on deadline on every OS.
"""
from __future__ import annotations

import importlib.metadata
import json
import sys
from .. import versions


PINS = versions.MATH_EVALUATOR_DEPENDENCY_PINS


def evaluate(expressions: list[str]) -> dict:
    raise RuntimeError('HISTORICAL_MATH_SCORER_RETIRED_USE_NEW_FROZEN_ATTEMPT')


def main() -> int:
    # This worker has no network role even outside the test harness.
    raise RuntimeError('HISTORICAL_MATH_SCORER_RETIRED_USE_NEW_FROZEN_ATTEMPT')


if __name__ == "__main__":
    raise SystemExit(main())
