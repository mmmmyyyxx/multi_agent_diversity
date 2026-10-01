"""Credential-free launcher allowing pinned public data/dependencies only."""
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    log = ROOT / "runs/multibench_dataset_migration_v1/network_guard.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if not re.search(r"API.?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTHORIZATION", k, re.I)}
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "tests/public_data_network"), str(ROOT)))
    env["PYTHONNOUSERSITE"] = "1"
    env["PUBLIC_DATA_NETWORK_LOG"] = str(log)
    # Bootstrap is checked before the requested script imports project code.
    control = subprocess.run([sys.executable, "-c",
        "import socket,sitecustomize; assert socket.getaddrinfo.__module__=='sitecustomize'; print('PUBLIC_DATA_GUARD_BOOTSTRAP_PASS')"], env=env, cwd=ROOT)
    if control.returncode:
        return control.returncode
    return subprocess.run([sys.executable, *sys.argv[1:]], env=env, cwd=ROOT).returncode

if __name__ == "__main__":
    raise SystemExit(main())
