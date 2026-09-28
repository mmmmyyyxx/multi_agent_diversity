"""Run an offline Python script under the pre-import network/credential guard."""

import json
import os
import runpy
import sys

import sitecustomize


if os.environ.get("FORMAL_ZERO_API_GUARD_REQUIRED") != "1":
    raise SystemExit("network guard was not required")
if sitecustomize.network_attempt_count():
    raise SystemExit("network attempt preceded offline command")
if any(marker in key.upper() for key in os.environ for marker in (
    "API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTHORIZATION",
)):
    raise SystemExit("credential-like environment name present")
if len(sys.argv) < 2:
    raise SystemExit("offline script path required")

script, *arguments = sys.argv[1:]
sys.argv = [script, *arguments]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    attempts = sitecustomize.network_attempt_count()
    print(json.dumps({"network_attempt_count": attempts, "blocked": True}, sort_keys=True))
    if attempts:
        raise SystemExit("offline command attempted network")
