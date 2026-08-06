from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.protocol import (
    AUXILIARY_SEARCH_CONTROL_SETTINGS,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly one explicitly opted-in v13 search-budget control."
        )
    )
    parser.add_argument(
        "--setting",
        choices=AUXILIARY_SEARCH_CONTROL_SETTINGS,
        required=True,
    )
    parser.add_argument("--workspace", type=Path, default=Path("."))
    args, remaining = parser.parse_known_args()
    workspace = args.workspace.resolve()
    command = [
        sys.executable,
        str(workspace / "scripts" / "run_task_level_accuracy.py"),
        "--workspace",
        str(workspace),
        *remaining,
        "--settings",
        args.setting,
        "--allow_auxiliary_setting",
        "1",
        "--allow_legacy_setting",
        "0",
        "--optimized_only",
        "1",
    ]
    return subprocess.run(command, cwd=workspace, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
