from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.tcs import TeacherRepairPlan
from scripts.analyze_v18_historical_teacher_safety import collect
from scripts.prepare_v18_safety_only_critic_pilot import prepare as prepare_base
from scripts.v18_safety_only_critic_pilot_support import canonical_hash, read_json, sha256_file, write_json
from scripts.v18_teacher_critic_pipeline_support import ARMS, deterministic_hard_gate


HISTORICAL_ROOT = ROOT / "runs" / "v18_hybrid_online_accumulation_pilot_20260822"
SOURCE_FILES = (
    "scripts/v18_teacher_critic_pipeline_support.py",
    "scripts/prepare_v18_teacher_critic_pipeline_ablation.py",
    "scripts/run_v18_teacher_critic_pipeline_ablation.py",
    "scripts/audit_v18_teacher_critic_pipeline_ablation.py",
    "scripts/analyze_v18_teacher_critic_pipeline_ablation.py",
    "scripts/prepare_v18_safety_only_critic_pilot.py",
    "scripts/generic_m20_probe_support.py",
    "tests/test_v18_teacher_critic_pipeline.py",
    "experiments/v18_teacher_critic_pipeline_ablation_20260902/DESIGN_SPEC.md",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def historical_regression() -> dict[str, Any]:
    plans, _ = collect(HISTORICAL_ROOT)
    passed = [row for row in plans if row["critic_approved"]]
    if not passed:
        raise ValueError("no historical canonical-passed Teacher plans")
    rejected = []
    categories: Counter[str] = Counter()
    fields: Counter[str] = Counter()
    for row in passed:
        source_path = HISTORICAL_ROOT / row["trajectory"] / "tcs_rounds.jsonl"
        source_rows = [
            json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        source = next(
            item for item in source_rows
            if item.get("role") == "teacher"
            and str(item.get("teacher_plan_hash", "")) == row["teacher_plan_hash"]
            and isinstance(item.get("repair_plan"), dict)
        )
        gate = deterministic_hard_gate(TeacherRepairPlan(**source["repair_plan"]))
        if not gate["pass"]:
            rejected.append(row["teacher_plan_hash"])
            categories[gate["category"]] += 1
            fields[gate["field_location"]] += 1
    rate = len(rejected) / len(passed)
    return {
        "regression_version": "v18_conservative_hard_gate_historical_v1",
        "canonical_passed_plan_count": len(passed),
        "deterministic_reject_count": len(rejected),
        "deterministic_reject_rate": rate,
        "maximum_allowed_reject_rate": 0.10,
        "gate": "PASS" if rate <= 0.10 else "FAIL",
        "category_counts": dict(sorted(categories.items())),
        "field_counts": dict(sorted(fields.items())),
        "rejected_plan_hashes": sorted(rejected),
        "api_calls": 0,
        "test_calls": 0,
    }


def prepare(out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("fresh preparation root required")
    if git("status", "--porcelain"):
        raise RuntimeError("tracked worktree must be clean")
    out.mkdir(parents=True)
    base = out / "base_reconstruction"
    base_gate = prepare_base(base)
    regression = historical_regression()
    write_json(out / "historical_hard_gate_regression.json", regression)
    if regression["gate"] != "PASS":
        write_json(out / "preflight.json", {
            "gate": "HOLD", "reason": "historical_hard_gate_false_positive_rate",
            "api_calls": 0, "test_calls": 0,
        })
        return read_json(out / "preflight.json")
    registry = read_json(base / "private_registry.json")
    registry.update({
        "registry_version": "v18_teacher_critic_pipeline_ablation_v1",
        "execution_commit": git("rev-parse", "HEAD"),
        "arms": list(ARMS),
        "source_candidates_per_branch": 2,
        "revision_per_valid_source": 1,
        "validation_after_all_train_decisions_frozen": True,
        "test_enabled": False,
        "trajectory_commit_enabled": False,
        "arm_selection_version": "v18_teacher_critic_pipeline_selection_v1",
        "historical_regression_sha256": sha256_file(out / "historical_hard_gate_regression.json"),
    })
    registry["registry_content_hash"] = canonical_hash({
        key: value for key, value in registry.items() if key != "registry_content_hash"
    })
    write_json(out / "private_registry.json", registry)
    freeze = {
        "execution_commit": registry["execution_commit"],
        "registry_sha256": sha256_file(out / "private_registry.json"),
        "historical_regression_sha256": registry["historical_regression_sha256"],
        "files": [
            {"path": path, "sha256": sha256_file(ROOT / path)} for path in SOURCE_FILES
        ],
    }
    write_json(out / "source_freeze.json", freeze)
    gate = {
        "gate": "PASS",
        "case_count": len(registry["cases"]),
        "arm_count": len(ARMS),
        "branch_count": len(registry["cases"]) * len(ARMS),
        "context_reconstruction_match_count": base_gate["context_reconstruction_match_count"],
        "historical_canonical_passed_plan_count": regression["canonical_passed_plan_count"],
        "historical_hard_gate_reject_rate": regression["deterministic_reject_rate"],
        "api_calls": 0,
        "test_calls": 0,
    }
    write_json(out / "preflight.json", gate)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if ROOT.resolve() not in args.out.resolve().parents:
        raise SystemExit("project-local output required")
    print(json.dumps(prepare(args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
