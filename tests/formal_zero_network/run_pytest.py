"""Assert no network was attempted anywhere in a guarded pytest campaign."""

import json
import os
import sys

import sitecustomize


if os.environ.get("FORMAL_ZERO_API_GUARD_REQUIRED") != "1":
    raise SystemExit("network guard was not required")
if sitecustomize.network_attempt_count() != 0:
    raise SystemExit("network attempt preceded pytest")
if any(marker in key.upper() for key in os.environ for marker in (
    "API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTHORIZATION",
)):
    raise SystemExit("credential-like environment name present")

import pytest  # noqa: E402

status = int(pytest.main(sys.argv[1:]))
attempts = sitecustomize.network_attempt_count()
print(json.dumps({"network_attempt_count": attempts, "blocked": True}, sort_keys=True))
raise SystemExit(status or (1 if attempts else 0))
