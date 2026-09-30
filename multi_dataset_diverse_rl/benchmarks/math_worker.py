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
    for package, version in PINS.items():
        if importlib.metadata.version(package) != version:
            raise ValueError("MATH_EVALUATOR_DEPENDENCY_IDENTITY_MISMATCH")
    import math_verify.parser as parser
    import math_verify.grader as grader
    from math_verify import parse, verify, LatexExtractionConfig
    no_inner_timeout = lambda *args, **kwargs: lambda function: function
    parser.timeout = no_inner_timeout
    grader.timeout = no_inner_timeout
    extracted = [parse(text if "$" in text or "\\boxed" in text else "$" + text + "$",
        extraction_config=(LatexExtractionConfig(),),
        fallback_mode="no_fallback", extraction_mode="first_match",
        parsing_timeout=versions.MATH_EVALUATOR_INNER_TIMEOUT_SECONDS)
        for text in expressions]
    valid = [bool(row) and not any(isinstance(x, str) for x in row) for row in extracted]
    matrix = [[bool(valid[i] and valid[j] and verify(extracted[i], extracted[j],
        float_rounding=versions.MATH_EVALUATOR_FLOAT_ROUNDING,
        numeric_precision=versions.MATH_EVALUATOR_NUMERIC_PRECISION, strict=True,
        timeout_seconds=versions.MATH_EVALUATOR_INNER_TIMEOUT_SECONDS))
        for j in range(len(extracted))] for i in range(len(extracted))]
    return {"valid": valid, "equivalence": matrix}


def main() -> int:
    # This worker has no network role even outside the test harness.
    import socket
    def blocked(*args, **kwargs):
        raise RuntimeError("MATH_WORKER_NETWORK_FORBIDDEN")
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    try:
        request = json.load(sys.stdin)
        result = evaluate(request["expressions"])
        json.dump(result, sys.stdout)
        return 0
    except Exception:
        json.dump({"error": "MATH_EVALUATOR_FAILURE"}, sys.stdout)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
