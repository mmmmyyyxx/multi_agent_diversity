from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from multi_dataset_diverse_rl.compatibility_repair import (
    COLLATERAL_REJECTION_REASONS,
    ONLINE_COMPATIBILITY_REPAIR_VERSION,
    REPAIR_INSTRUCTION,
    REPAIR_SYSTEM_PROMPT,
    repair_eligible,
)
from multi_dataset_diverse_rl.utils import normalize_prompt_text
from scripts.admit_v18_hybrid_online_scientific_analysis import artifact_tree_identity


ROOT = Path(__file__).resolve().parents[1]
CASES = ((59, 3), (61, 5))
ARM = "HYBRID_BASE"
SOURCE_FILES = (
    "multi_dataset_diverse_rl/compatibility_repair.py",
    "multi_dataset_diverse_rl/candidate_selection.py",
    "multi_dataset_diverse_rl/system.py",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def prompt_hash(value: str) -> str:
    return hashlib.sha256(normalize_prompt_text(value).encode("utf-8")).hexdigest()


def build(*, run_root: Path, admission: dict[str, Any], out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("fresh Phase A output root required")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean")
    head = git("rev-parse", "HEAD")
    if admission.get("scientific_analysis_admitted") is not True:
        raise ValueError("V18 scientific admission must pass")
    raw_identity = artifact_tree_identity(run_root)
    if raw_identity != admission["raw_artifact_identity"]:
        raise ValueError("historical V18 raw artifact identity mismatch")
    sources = []
    for seed, update_index in CASES:
        run = run_root / f"seed{seed}" / ARM
        decisions = read_jsonl(run / "candidate_decisions.jsonl")
        decision = next(row for row in decisions if int(row["update_index"]) == update_index)
        branch_by_target = {int(row["target_agent_id"]): row for row in decision["branches"]}
        recovered_count = 0
        for candidate in decision["candidates"]:
            constraint = candidate.get("constraint")
            evaluation = candidate.get("evaluation")
            if not constraint or not evaluation:
                continue
            if not bool(constraint["passed"]) or int(constraint["vote_loss_count"]) <= 0:
                continue
            recovered_count += 1
            source_prompt = str(evaluation["prompt"])
            source_hash = str(candidate["prompt_hash"])
            if prompt_hash(source_prompt) != source_hash:
                raise ValueError("source candidate prompt hash mismatch")
            target = int(candidate["target_agent_id"])
            branch = branch_by_target[target]
            responsibility_gain_count = int(
                evaluation["marginal"]["assigned_residual_repair_count"]
            )
            minimum_loss_evidence_count = int(
                constraint["pivotal_correct_loss_count"]
            ) + int(constraint["unique_correct_loss_count"])
            rejection_reasons = tuple(map(str, constraint["rejection_reasons"]))
            eligible = repair_eligible(
                responsibility_gain_count=responsibility_gain_count,
                rejection_reasons=rejection_reasons,
                loss_evidence_count=minimum_loss_evidence_count,
            )
            failure_reasons = []
            if responsibility_gain_count <= 0:
                failure_reasons.append("no_responsibility_gain")
            if minimum_loss_evidence_count <= 0:
                failure_reasons.append("no_minimum_loss_evidence")
            if not COLLATERAL_REJECTION_REASONS.intersection(rejection_reasons):
                failure_reasons.append("no_frozen_collateral_rejection_reason")
            sources.append({
                "case_id": f"seed{seed}_update{update_index}_{source_hash[:12]}",
                "seed": seed,
                "arm": ARM,
                "update_index": update_index,
                "parent_team_hash": str(decision["parent_team_hash"]),
                "target_agent_id": target,
                "target_selection_rank": int(candidate["target_selection_rank"]),
                "assigned_responsibility_hashes": sorted(map(str, branch["assigned_question_hashes"])),
                "source_candidate_hash": source_hash,
                "source_candidate_stage": str(candidate["candidate_stage"]),
                "source_common_safe": bool(constraint["passed"]),
                "source_target_gain": int(constraint["target_gain"]),
                "source_vote_gain_count": int(constraint["vote_gain_count"]),
                "source_vote_loss_count": int(constraint["vote_loss_count"]),
                "source_vote_net_gain": int(constraint["vote_net_gain"]),
                "source_rejection_reasons": list(rejection_reasons),
                "source_responsibility_gain_count": responsibility_gain_count,
                "minimum_loss_evidence_count": minimum_loss_evidence_count,
                "frozen_m2f_repair_eligible": eligible,
                "frozen_m2f_eligibility_failure_reasons": failure_reasons,
                "source_prompt_hash_verified": True,
            })
        expected = 4 if seed == 59 else 3
        if recovered_count != expected:
            raise ValueError("harmful candidate pool inventory mismatch")
    if len(sources) != 7:
        raise ValueError("expected seven feasible vote-loss sources")
    if any(not row["source_common_safe"] or row["source_vote_loss_count"] <= 0 for row in sources):
        raise ValueError("external source filter mismatch")
    eligible = [row for row in sources if row["frozen_m2f_repair_eligible"]]
    freeze = {
        "freeze_version": "v18_harmful_commit_m2f_source_freeze_v1",
        "execution_commit": head,
        "v18_raw_artifact_identity": raw_identity,
        "m2f_version": ONLINE_COMPATIBILITY_REPAIR_VERSION,
        "repair_system_prompt_sha256": hashlib.sha256(REPAIR_SYSTEM_PROMPT.encode()).hexdigest(),
        "repair_instruction_sha256": hashlib.sha256(REPAIR_INSTRUCTION.encode()).hexdigest(),
        "source_files": [
            {"path": relative, "sha256": sha256_file(ROOT / relative)}
            for relative in SOURCE_FILES
        ],
        "source_candidate_count": len(sources),
        "eligible_source_candidate_count": len(eligible),
    }
    registry = {
        "registry_version": "v18_harmful_commit_m2f_registry_v1",
        "execution_commit": head,
        "cases": sources,
        "source_count": len(sources),
        "eligible_count": len(eligible),
        "model": "qwen3-14b",
        "thinking": False,
        "repair_attempts_per_eligible_source": 1,
        "validation_after_train_decision_only": True,
        "test_enabled": False,
        "commit_enabled": False,
        "trajectory_mutation_enabled": False,
        "classifier": {
            "labels": [
                "M2F_WRITEBACK_RISK_REDUCTION_SUPPORTED",
                "M2F_TRAIN_COLLATERAL_ONLY",
                "M2F_NOT_SUPPORTED",
                "M2F_HARMFUL",
            ],
            "target_gain_retention_high_threshold": 0.8,
            "target_gain_retention_harmful_threshold": 0.5,
        },
    }
    registry["registry_content_hash"] = hashlib.sha256(
        json.dumps(registry, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    gate = {
        "phase_a_version": "v18_harmful_commit_m2f_phase_a_v1",
        "phase_a_gate": (
            "PASS" if eligible else "STOP_INELIGIBLE_UNDER_FROZEN_M2F"
        ),
        "source_reconstruction_gate": "PASS",
        "m2f_semantics_gate": "PASS",
        "no_validation_leakage_gate": "PASS",
        "historical_raw_hash_gate": "PASS",
        "source_candidate_count": len(sources),
        "external_pool_filter_count": len(sources),
        "frozen_m2f_eligible_count": len(eligible),
        "phase_b_authorized_by_gate": bool(eligible),
        "api_calls": 0,
        "model_calls": 0,
        "solver_calls": 0,
        "evaluator_calls": 0,
        "new_validation_calls": 0,
        "new_test_calls": 0,
        "method_modified": False,
        "eligibility_modified": False,
        "raw_artifacts_modified": False,
        "stop_reason": (
            "existing M2F requires a collateral rejection reason, but all seven sources are Common-Safe feasible"
            if not eligible else ""
        ),
    }
    out.mkdir(parents=True)
    write_json(out / "private_source_registry.json", registry)
    write_json(out / "source_freeze.json", freeze)
    write_json(out / "phase_a_gate.json", gate)
    print(json.dumps({
        "phase_a_gate": gate["phase_a_gate"],
        "source_candidate_count": len(sources),
        "frozen_m2f_eligible_count": len(eligible),
        "phase_b_authorized_by_gate": bool(eligible),
        "api_calls": 0,
    }, indent=2))
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(
        run_root=args.run_root.resolve(),
        admission=read_json(args.admission.resolve()),
        out=args.out.resolve(),
    )


if __name__ == "__main__":
    main()
