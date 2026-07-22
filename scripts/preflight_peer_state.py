from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.protocol import CandidateBudgetContract, experiment_protocol
from scripts.experiment_config import DEFAULT_EXPERIMENT_SETTING_NAMES, select_settings


EXPECTED_SETTINGS = [
    "shared_baseline", "shared_independent_accuracy_tcs", "shared_peer_state_credit_round_robin",
    "shared_peer_state_responsibility", "shared_peer_state_full",
]


def preflight(workspace: Path, allow_dirty: bool = False) -> dict:
    errors = []
    configs = [Config.from_flat(**setting.resolved_overrides()) for setting in select_settings("all")]
    if DEFAULT_EXPERIMENT_SETTING_NAMES != EXPECTED_SETTINGS:
        errors.append("experiment settings do not match the frozen five-setting protocol")
    for cfg in configs:
        if cfg.training.method_version != "peer_state_counterfactual_v1":
            errors.append(f"unexpected method version: {cfg.training.method_version}")
        if cfg.training.agents != 5 or cfg.peer_state.aggregation_mode != "plurality":
            errors.append("all settings must use five equal-weight plurality voters")
        if cfg.peer_state.vote_tie_break != "abstain":
            errors.append("all canonical settings must use tie-as-abstain")
    budget = CandidateBudgetContract(2, 2, 6, 12, 6, 6, 4)
    protocols = {
        name: experiment_protocol(
            name,
            initialization_mode="shared_identical",
            tie_policy="abstain",
            candidate_budget_contract=budget,
        )
        for name in EXPECTED_SETTINGS
    }
    b3 = protocols["shared_peer_state_responsibility"]
    b4 = protocols["shared_peer_state_full"]
    b3_payload = b3.__dict__ | {
        "name": b4.name,
        "tcs_context_policy": b4.tcs_context_policy,
    }
    if b3_payload != b4.__dict__:
        errors.append("B3 and B4 must differ only in responsibility-conditioned TCS")
    help_result = subprocess.run(
        [sys.executable, "scripts/run_task_level_accuracy.py", "--help"],
        cwd=workspace,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if help_result.returncode != 0:
        errors.append("task runner parser failed to build")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=workspace, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=workspace, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"ok": False, "errors": [f"git inspection failed: {exc}"]}
    if dirty and not allow_dirty:
        errors.append("git working tree is not clean")
    return {
        "ok": not errors, "git_commit": head, "git_dirty": dirty,
        "method_version": "peer_state_counterfactual_v1", "settings": EXPECTED_SETTINGS,
        "legacy_compatibility_enabled": False, "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--allow_dirty", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    report = preflight(args.workspace.resolve(), bool(args.allow_dirty))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
