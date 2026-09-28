"""Launch guarded fake-provider pytest in a fresh credential-free Python process.

This is test infrastructure, not an experiment runner. The child loads the
network guard via sitecustomize before importing provider/application code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
GUARD = Path(__file__).resolve().parent / "formal_zero_network"
_SECRET_NAME = re.compile(r"(?:API.?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTHORIZATION)", re.I)


def credential_free_environment() -> dict[str, str]:
    env = {name: value for name, value in os.environ.items() if not _SECRET_NAME.search(name)}
    env["PYTHONPATH"] = os.pathsep.join((str(GUARD), str(ROOT)))
    env["PYTHONNOUSERSITE"] = "1"
    env["FORMAL_ZERO_API_GUARD_REQUIRED"] = "1"
    return env


def main() -> int:
    env = credential_free_environment()
    assert not any(_SECRET_NAME.search(name) for name in env)
    args = sys.argv[1:]
    if args[:1] == ["--evidence-dir"]:
        if len(args) < 2:
            raise ValueError("--evidence-dir requires an ignored runs/ path")
        destination = Path(args[1]).resolve()
        if not destination.is_relative_to((ROOT / "runs").resolve()):
            raise ValueError("fake evidence must stay under ignored runs/")
        env["FORMAL_V3_EVIDENCE_CAPTURE_DIR"] = str(destination)
        args = args[2:]
    controls = subprocess.run(
        [sys.executable, "-c", (
            "import socket, sitecustomize; "
            "assert sitecustomize.network_attempt_count() == 0; "
            "assert socket.socket.connect.__module__ == 'sitecustomize'; "
            "print('NETWORK_GUARD_BOOTSTRAP_PASS')"
        )],
        cwd=ROOT, env=env, check=False,
    )
    if controls.returncode:
        return controls.returncode
    negative = subprocess.run(
        [sys.executable, str(GUARD / "negative_control.py")],
        cwd=ROOT, env=env, check=False,
    )
    if negative.returncode:
        return negative.returncode
    print(json.dumps({
        "real_credentials_present": False,
        "network_guard_active_before_application_import": True,
    }, sort_keys=True), flush=True)
    if args[:1] == ["--offline-command"]:
        if len(args) < 2:
            raise ValueError("--offline-command requires a Python script")
        command = [sys.executable, str(GUARD / "run_command.py"), *args[1:]]
    else:
        args = args or ["tests/test_formal_v3_full_fake.py"]
        command = [sys.executable, str(GUARD / "run_pytest.py"), "-q", *args]
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
