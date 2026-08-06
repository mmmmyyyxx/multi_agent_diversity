from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.protocol import (
    AUXILIARY_SEARCH_CONTROL_SETTINGS,
    MAIN_ABLATION_SETTINGS,
)
from multi_dataset_diverse_rl.versions import (
    CHECKPOINT_VERSION,
    METHOD_VERSION,
)


EXPECTED_BUDGETS = {
    "aux_dual_target_budget_matched_2x1": (2, 1, 2),
    "aux_single_target_compute_matched_1x4": (1, 4, 4),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(run_dir: Path, expected_setting: str) -> dict[str, Any]:
    failures: list[str] = []
    if expected_setting not in AUXILIARY_SEARCH_CONTROL_SETTINGS:
        failures.append("expected_setting_is_not_auxiliary")
    if expected_setting in MAIN_ABLATION_SETTINGS:
        failures.append("auxiliary_setting_leaked_into_main_matrix")
    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        return {
            "gate": "FAIL",
            "failures": ["missing_run_meta"],
            "run_dir": str(run_dir),
        }
    meta = _read_json(meta_path)
    protocol = dict(meta.get("experiment_protocol", {}))
    budget = dict(protocol.get("candidate_budget_contract", {}))
    expected_branch, expected_per_branch, expected_total = EXPECTED_BUDGETS[
        expected_setting
    ]
    checks = {
        "method_version": meta.get("method_version") == METHOD_VERSION,
        "checkpoint_version": meta.get("checkpoint_version")
        == CHECKPOINT_VERSION,
        "canonical_setting": meta.get("canonical_experiment_setting")
        == expected_setting,
        "auxiliary_protocol": protocol.get("auxiliary_protocol") is True,
        "legacy_protocol": protocol.get("legacy_protocol") is False,
        "target_branch_count": int(
            budget.get("target_branch_count", -1)
        ) == expected_branch,
        "candidates_per_target_branch": int(
            budget.get("candidates_per_target_branch", -1)
        ) == expected_per_branch,
        "total_generated_candidates_per_update": int(
            budget.get("total_generated_candidates_per_update", -1)
        ) == expected_total,
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    return {
        "gate": "PASS" if not failures else "FAIL",
        "setting": expected_setting,
        "checks": checks,
        "failures": failures,
        "run_dir": str(run_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument(
        "--setting",
        choices=AUXILIARY_SEARCH_CONTROL_SETTINGS,
        required=True,
    )
    args = parser.parse_args()
    payload = audit(args.run_dir.resolve(), args.setting)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
