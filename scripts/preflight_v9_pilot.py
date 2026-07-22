import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.cli import configure_runtime_probe_version, rollout_method_metadata
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.sequential_state import stage_a_accuracy_prefilter_key
from scripts.experiment_config import select_settings


MATCHED_V9_SETTINGS = (
    "shared_v9_sequential_accuracy",
    "shared_v9_sequential_accuracy_state",
    "shared_v9_sequential_accuracy_state_vote",
    "shared_v9_sequential_accuracy_state_vote_diversity",
)
INTENDED_INCREMENTAL_FLAGS = (
    "state_distribution_reward_enabled",
    "state_vote_reward_enabled",
    "state_diversity_constraints_enabled",
)


def _git(workspace: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=workspace, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _resolved_configs(setting_names: Iterable[str]) -> list[tuple[str, Config]]:
    names = list(setting_names)
    selected = select_settings(",".join(names))
    selected_by_name = {setting.name: setting for setting in selected}
    return [
        (name, Config(**selected_by_name[name].resolved_overrides()))
        for name in names
    ]


def static_preflight_errors(setting_names: Iterable[str] = MATCHED_V9_SETTINGS) -> list[str]:
    names = tuple(setting_names)
    errors: list[str] = []
    if names != MATCHED_V9_SETTINGS:
        errors.append(f"matched settings must be exactly {','.join(MATCHED_V9_SETTINGS)}")
        return errors

    resolved = _resolved_configs(names)
    for name, cfg in resolved:
        metadata = rollout_method_metadata(cfg)
        required = {
            "method_version": "v9_state_conditioned_error",
            "agents": 5,
            "aggregation_mode": "plurality",
            "candidate_eval_data_source": "optimization_train",
            "eval_test_each_epoch": False,
            "state_bottom2_reward_enabled": False,
            "state_c2_wrong_split_enabled": False,
            "state_rollout_exploration_enabled": False,
            "state_exploration_parent_enabled": False,
        }
        for field, expected in required.items():
            actual = getattr(cfg, field)
            if actual != expected:
                errors.append(f"{name}: {field}={actual!r}, expected {expected!r}")
        for field, expected in {
            "rollout_qd_method": False,
            "rollout_archive_enabled": False,
            "joint_team_enumeration_enabled": False,
            "equal_vote_weighting": True,
        }.items():
            if metadata.get(field) != expected:
                errors.append(f"{name}: metadata {field}={metadata.get(field)!r}, expected {expected!r}")
        holder = type("ProbeVersionHolder", (), {})()
        configure_runtime_probe_version(holder, cfg)
        if holder.prompt_probe_version != cfg.probe_stability_version:
            errors.append(f"{name}: runtime/config probe version mismatch")

    config_rows = [(name, cfg.to_flat_dict()) for name, cfg in resolved]
    for index, expected_flag in enumerate(INTENDED_INCREMENTAL_FLAGS):
        left_name, left = config_rows[index]
        right_name, right = config_rows[index + 1]
        differences = {key for key in set(left) | set(right) if left.get(key) != right.get(key)}
        if differences != {expected_flag}:
            errors.append(
                f"{left_name} -> {right_name}: unexpected resolved differences {sorted(differences)!r}"
            )

    low_vote = {
        "prompt_hash": "a",
        "metrics": {
            "candidate_target_accuracy": 0.8,
            "candidate_invalid_rate": 0.0,
            "candidate_team_accuracy": 0.0,
            "state_vote_reward": -100.0,
            "diversity_constraint_slack": -100.0,
        },
    }
    high_vote = {
        **low_vote,
        "prompt_hash": "b",
        "metrics": {
            **low_vote["metrics"],
            "candidate_team_accuracy": 1.0,
            "state_vote_reward": 100.0,
            "diversity_constraint_slack": 100.0,
        },
    }
    if stage_a_accuracy_prefilter_key(low_vote)[:-1] != stage_a_accuracy_prefilter_key(high_vote)[:-1]:
        errors.append("Stage A key leaks team, state, or diversity signals")
    return errors


def run_preflight(workspace: Path, *, allow_dirty: bool = False, run_root: Path | None = None) -> dict[str, Any]:
    workspace = workspace.resolve()
    errors = static_preflight_errors()
    try:
        head = _git(workspace, "rev-parse", "HEAD")
        dirty = bool(_git(workspace, "status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"ok": False, "errors": [*errors, f"git inspection failed: {exc}"], "workspace": str(workspace)}
    if dirty and not allow_dirty:
        errors.append("git working tree is not clean")

    checked_run_meta = 0
    if run_root is not None:
        root = run_root.resolve()
        if not root.exists():
            errors.append(f"run root does not exist: {root}")
        else:
            for path in root.rglob("run_meta.json"):
                checked_run_meta += 1
                try:
                    meta = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"invalid run metadata {path}: {exc}")
                    continue
                if str(meta.get("git_commit", "")) != head:
                    errors.append(f"{path}: git_commit does not match current HEAD")
                if str(meta.get("prompt_probe_version", "")) != str(meta.get("probe_stability_version", "")):
                    errors.append(f"{path}: prompt/config probe version mismatch")
            if checked_run_meta == 0:
                errors.append(f"no run_meta.json found under {root}")

    return {
        "ok": not errors,
        "workspace": str(workspace),
        "git_commit": head,
        "git_dirty": dirty,
        "checked_settings": list(MATCHED_V9_SETTINGS),
        "checked_run_meta": checked_run_meta,
        "stage_a_policy": "candidate_target_accuracy,-candidate_invalid_rate,stable_prompt_hash",
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-fast checks for the matched V9 A0-A3 pilot.")
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--run_root", type=Path)
    parser.add_argument("--allow_dirty", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    report = run_preflight(args.workspace, allow_dirty=bool(args.allow_dirty), run_root=args.run_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
