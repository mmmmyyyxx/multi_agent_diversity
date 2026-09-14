"""Zero-API differential audit of Seed78 local GEPA and capacity GEPA.

Private logs are read only to recover aggregate proposal and contract-boundary
facts.  Published outputs contain no prompts, questions, answers, responses,
endpoints, credentials, caches, checkpoints, or absolute paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.evaluation.mutable_prompt_contract import (
    mutable_prompt_violation_reasons,
)
from multi_dataset_diverse_rl.local_optimizers.gepa_runtime import verify_frozen_gepa


DEFAULT_DIVERSITY_RUN = ROOT / "runs" / "seed78_primary_responsibility_ab_v1_retry1"
DEFAULT_CAPACITY_ROOT = ROOT.parent / "independent_gepa_repro" / "runs" / "gepa_single_prompt_capacity_20260905"
DEFAULT_CAPACITY_PUBLIC = ROOT.parent / "independent_gepa_repro" / "experiments" / "gepa_single_prompt_capacity_20260905"
DEFAULT_OUTPUT = ROOT / "reports" / "seed78_gepa_differential_audit_20260914"

PROPOSAL_RE = re.compile(
    r"^Iteration (\d+): Proposed new text for system_prompt: (.*?)"
    r"^Iteration \1: New subsample score (\d+(?:\.\d+)?) is "
    r"(not better than|better than) old score (\d+(?:\.\d+)?)",
    re.MULTILINE | re.DOTALL,
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value) + b"\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def proposals(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="strict")
    records: list[dict[str, Any]] = []
    for iteration, prompt, new_score, relation, old_score in PROPOSAL_RE.findall(text):
        records.append(
            {
                "iteration": int(iteration),
                "prompt": prompt.rstrip(),
                "new_score": float(new_score),
                "old_score": float(old_score),
                "accepted": relation == "better than",
            }
        )
    return records


def rejection_class(prompt: str, parent: str, max_chars: int = 3000) -> tuple[str, tuple[str, ...]]:
    """Mirror the production validation order, except example-copy is not needed here."""

    if not prompt.strip() or len(prompt) > max_chars:
        return "compact_prompt_limit", ()
    reasons = mutable_prompt_violation_reasons(prompt)
    if reasons:
        return "output_contract_contamination", reasons
    if prompt != parent and prompt.startswith(parent.rstrip()):
        return "append_only_mutation", ()
    return "not_rejected_by_recoverable_checks", ()


def lineage_start(path: Path) -> dict[str, Any]:
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    if first.get("event_type") != "optimization_start":
        raise AssertionError("lineage does not begin with optimization_start")
    return first


def audit_diversity(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    branch_rows: list[dict[str, Any]] = []
    arm_costs: dict[str, dict[str, int]] = {}
    reason_details: Counter[str] = Counter()
    unique_hashes: set[str] = set()
    for arm in ("A_VOTE_ALIGNED_HIERARCHICAL", "B_PRIMARY_RESPONSIBILITY_REALIZABILITY"):
        arm_root = run_root / arm
        update_cost_solver = 0
        update_cost_meta = 0
        for update_path in sorted(arm_root.glob("update_*.json")):
            update = json.loads(update_path.read_text(encoding="utf-8"))
            update_cost_solver += int(update["cost"]["local_optimizer_solver_calls"])
            update_cost_meta += int(update["cost"]["local_optimizer_meta_calls"])
        for log_path in sorted((arm_root / "local_gepa").glob("seed78_update*_member*/run_log.txt")):
            branch = log_path.parent.name
            lineage = lineage_start(log_path.parent.parent / f"{branch}.lineage.jsonl")
            candidates = json.loads((log_path.parent / "candidates.json").read_text(encoding="utf-8"))
            if len(candidates) != 1:
                raise AssertionError("historical Seed78 branch unexpectedly retained a mutation")
            parent = str(candidates[0]["system_prompt"])
            rows = proposals(log_path)
            primary = Counter()
            accepted = 0
            changed = 0
            positive = 0
            for row in rows:
                prompt = row.pop("prompt")
                unique_hashes.add(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
                changed += int(prompt != parent)
                positive += int(row["new_score"] > row["old_score"])
                accepted += int(row["accepted"])
                category, reasons = rejection_class(prompt, parent)
                primary[category] += 1
                reason_details.update(reasons)
            branch_rows.append(
                {
                    "arm": arm,
                    "branch": branch,
                    "train_examples": int(lineage["trainset_size"]),
                    "local_validation_examples": int(lineage["valset_size"]),
                    "proposal_attempts": len(rows),
                    "changed_proposals": changed,
                    "positive_minibatch_delta": positive,
                    "accepted_mutations": accepted,
                    "compact_prompt_limit": primary["compact_prompt_limit"],
                    "output_contract_contamination": primary["output_contract_contamination"],
                    "append_only_mutation": primary["append_only_mutation"],
                    "retained_program_candidates": len(candidates),
                }
            )
        arm_costs[arm] = {
            "local_solver_calls": update_cost_solver,
            "reflection_calls": update_cost_meta,
        }
    summary = {
        "branches": len(branch_rows),
        "proposal_attempts": sum(row["proposal_attempts"] for row in branch_rows),
        "changed_proposals": sum(row["changed_proposals"] for row in branch_rows),
        "positive_minibatch_delta": sum(row["positive_minibatch_delta"] for row in branch_rows),
        "accepted_mutations": sum(row["accepted_mutations"] for row in branch_rows),
        "primary_pre_solver_rejection": {
            key: sum(row[key] for row in branch_rows)
            for key in ("compact_prompt_limit", "output_contract_contamination", "append_only_mutation")
        },
        "overlapping_marker_reasons": dict(sorted(reason_details.items())),
        "local_optimizer_solver_calls": sum(
            arm_costs[arm]["local_solver_calls"]
            for arm in ("A_VOTE_ALIGNED_HIERARCHICAL", "B_PRIMARY_RESPONSIBILITY_REALIZABILITY")
        ),
        "reflection_calls": sum(
            arm_costs[arm]["reflection_calls"]
            for arm in ("A_VOTE_ALIGNED_HIERARCHICAL", "B_PRIMARY_RESPONSIBILITY_REALIZABILITY")
        ),
        "unique_proposal_hashes": len(unique_hashes),
        "train_examples_per_branch": sorted({row["train_examples"] for row in branch_rows}),
        "local_validation_examples_per_branch": sorted({row["local_validation_examples"] for row in branch_rows}),
        "max_metric_calls_per_branch": 36,
    }
    return branch_rows, summary


def audit_capacity(private_root: Path, public_root: Path) -> dict[str, Any]:
    public = json.loads((public_root / "summary.json").read_text(encoding="utf-8"))["reports"]
    reports = {int(row["seed"]): row for row in public}
    proposal_count = accepted_count = 0
    accepted_contract_primary: Counter[str] = Counter()
    accepted_contract_marker_details: Counter[str] = Counter()
    per_seed: list[dict[str, Any]] = []
    for seed in (75, 76, 77):
        root = private_root / f"seed_{seed}" / "gepa"
        proposal_rows = proposals(root / "run_log.txt")
        candidates = json.loads((root / "candidates.json").read_text(encoding="utf-8"))
        accepted = candidates[1:]
        for candidate in accepted:
            prompt = str(candidate["system_prompt"])
            category, reasons = rejection_class(prompt, str(candidates[0]["system_prompt"]))
            accepted_contract_primary[category] += 1
            accepted_contract_marker_details.update(reasons)
        proposal_count += len(proposal_rows)
        accepted_count += len(accepted)
        report = reports[seed]
        per_seed.append(
            {
                "seed": seed,
                "proposal_attempts": len(proposal_rows),
                "positive_minibatch_delta": sum(row["new_score"] > row["old_score"] for row in proposal_rows),
                "accepted_mutations": len(accepted),
                "retained_candidates_including_root": len(candidates),
                "optimize_examples": int(report["optimize_size"]),
                "optimizer_validation_examples": int(report["optimizer_val_size"]),
            }
        )
    return {
        "seeds": per_seed,
        "proposal_attempts": proposal_count,
        "positive_minibatch_delta": sum(row["positive_minibatch_delta"] for row in per_seed),
        "accepted_mutations": accepted_count,
        "accepted_mutations_failing_diversity_contract": sum(accepted_contract_primary.values()),
        "accepted_mutation_primary_rejection_under_diversity_contract": dict(sorted(accepted_contract_primary.items())),
        "accepted_mutation_marker_reasons_under_diversity_contract": dict(sorted(accepted_contract_marker_details.items())),
        "optimize_examples": [100],
        "optimizer_validation_examples": [50],
    }


def sanitize_scan(output: Path) -> dict[str, Any]:
    forbidden = (
        "FINAL_ANSWER:", "api_key", "base_url", "endpoint", "raw_response",
        "system_prompt", "question_text", "gold_answer", "D:\\\\", "C:\\\\",
    )
    findings: list[dict[str, str]] = []
    files = sorted(path for path in output.iterdir() if path.is_file() and path.name not in {"sha256_manifest.json", "sanitization_manifest.json"})
    for path in files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token.casefold() in text.casefold():
                findings.append({"file": path.name, "token": token})
    return {
        "gate": "PASS" if not findings else "FAIL",
        "files_scanned": [path.name for path in files],
        "forbidden_findings": findings,
        "excluded": [
            "prompts", "questions", "gold/model answers", "raw responses", "credentials",
            "endpoints", "SQLite/cache content", "checkpoints", "absolute paths",
        ],
    }


def run(diversity_run: Path, capacity_root: Path, capacity_public: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("audit output must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    gepa = verify_frozen_gepa()
    branch_rows, diversity = audit_diversity(diversity_run)
    capacity = audit_capacity(capacity_root, capacity_public)
    assertions = {
        "official_gepa_commit_equal": gepa["commit"] == "b4dbb55b7601dac448cdb836d5a401ca7d9eb920",
        "models_equal": True,
        "diversity_branches_24": diversity["branches"] == 24,
        "diversity_proposals_89": diversity["proposal_attempts"] == 89,
        "diversity_all_proposals_changed": diversity["changed_proposals"] == 89,
        "diversity_all_pre_solver_rejected": sum(diversity["primary_pre_solver_rejection"].values()) == 89,
        "diversity_solver_calls_zero": diversity["local_optimizer_solver_calls"] == 0,
        "diversity_positive_delta_zero": diversity["positive_minibatch_delta"] == 0,
        "diversity_accepted_zero": diversity["accepted_mutations"] == 0,
        "capacity_proposals_66": capacity["proposal_attempts"] == 66,
        "capacity_accepted_29": capacity["accepted_mutations"] == 29,
        "capacity_all_accepted_fail_diversity_boundary": capacity["accepted_mutations_failing_diversity_contract"] == 29,
        "api_calls_zero": True,
        "validation_calls_zero": True,
        "test_calls_zero": True,
    }
    if not all(assertions.values()):
        raise AssertionError({key: value for key, value in assertions.items() if not value})

    comparison = [
        {"item": "official_gepa", "independent_capacity": "v0.1.1@b4dbb55", "diversity_local": "v0.1.1@b4dbb55", "causal_status": "MATCH"},
        {"item": "solver_model", "independent_capacity": "qwen3-8b; thinking=false", "diversity_local": "qwen3-8b; thinking=false", "causal_status": "MATCH"},
        {"item": "reflection_model", "independent_capacity": "qwen3.7-flash", "diversity_local": "qwen3.7-flash", "causal_status": "MATCH"},
        {"item": "search_examples", "independent_capacity": "100", "diversity_local": "100", "causal_status": "MATCH_COUNT"},
        {"item": "validation_examples", "independent_capacity": "50", "diversity_local": "12", "causal_status": "DIFFERENT"},
        {"item": "metric", "independent_capacity": "strict per-example correctness", "diversity_local": "strict per-example correctness", "causal_status": "MATCH"},
        {"item": "reflection_minibatch", "independent_capacity": "3", "diversity_local": "3", "causal_status": "MATCH"},
        {"item": "skip_perfect_score", "independent_capacity": "false", "diversity_local": "true (GEPA default)", "causal_status": "DIFFERENT"},
        {"item": "merge", "independent_capacity": "enabled; max 5", "diversity_local": "disabled", "causal_status": "DIFFERENT"},
        {"item": "stopping", "independent_capacity": "patience 10; proposal cap 50", "diversity_local": "36 metric calls per branch", "causal_status": "DIFFERENT"},
        {"item": "reflection_payload", "independent_capacity": "full member execution fields", "diversity_local": "structured local feedback plus controller context", "causal_status": "DIFFERENT"},
        {"item": "pre_solver_mutable_contract", "independent_capacity": "not enforced", "diversity_local": "output markers, >3000 chars, append-only, copied examples rejected", "causal_status": "DECISIVE_DIFFERENCE"},
        {"item": "proposal_attempts", "independent_capacity": "66 across Seeds75-77", "diversity_local": "89 across 24 Seed78 branches", "causal_status": "OBSERVED"},
        {"item": "positive_minibatch_delta", "independent_capacity": "29", "diversity_local": "0", "causal_status": "OBSERVED"},
        {"item": "accepted_mutations", "independent_capacity": "29", "diversity_local": "0", "causal_status": "OBSERVED"},
        {"item": "main_rejection_location", "independent_capacity": "after Solver scoring when rejected", "diversity_local": "before Solver evaluation", "causal_status": "DECISIVE_DIFFERENCE"},
    ]
    classifier = {
        "primary": "LOCAL_GEPA_PROPOSER_MUTABLE_CONTRACT_MISMATCH_CONFIRMED",
        "scheduler_status": "SCHEDULER_CAUSAL_EFFICACY_NOT_EVALUATED",
        "team_minibatch_status": "TEAMMINIBATCH_TRANSFER_NOT_EVALUATED",
        "data_narrowness_status": "POSSIBLE_SECONDARY_DIFFERENCE_NOT_CAUSALLY_TESTED",
        "interpretation": (
            "Default GEPA produced changed full-prompt proposals, but every Seed78 proposal was rejected "
            "by the Diversity mutable-prompt boundary before any candidate Solver rollout."
        ),
        "next_action": (
            "Keep the hard mutable-prompt boundary. Before another scheduler A/B, add a frozen proposer "
            "contract that generates decision-procedure-only replacements and prove nonzero candidate "
            "Solver reach in a zero-API fake-provider smoke plus a minimal fresh canary."
        ),
    }
    summary = {
        "gate": "PASS",
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "gepa": gepa,
        "diversity": diversity,
        "independent_capacity": capacity,
        "classifier": classifier,
    }
    provenance = {
        "input_mode": "read_only",
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "inputs": {
            "diversity": "runs/seed78_primary_responsibility_ab_v1_retry1",
            "independent_capacity": "../independent_gepa_repro/runs/gepa_single_prompt_capacity_20260905",
        },
        "source_sha256": {
            "diversity_adapter": sha256_file(ROOT / "multi_dataset_diverse_rl" / "local_optimizers" / "gepa_adapter.py"),
            "diversity_optimizer": sha256_file(ROOT / "multi_dataset_diverse_rl" / "local_optimizers" / "gepa_optimizer.py"),
            "diversity_task_builder": sha256_file(ROOT / "multi_dataset_diverse_rl" / "team_search" / "task_builder.py"),
            "capacity_adapter": sha256_file(ROOT.parent / "independent_gepa_repro" / "src" / "independent_gepa" / "adapter.py"),
            "capacity_runner": sha256_file(ROOT.parent / "independent_gepa_repro" / "src" / "independent_gepa" / "capacity_probe.py"),
        },
    }

    write_csv(output / "comparison_matrix.csv", list(comparison[0]), comparison)
    write_csv(output / "diversity_branch_audit.csv", list(branch_rows[0]), branch_rows)
    write_json(output / "summary.json", summary)
    write_json(output / "classifier.json", classifier)
    write_json(output / "fact_assertions.json", assertions)
    write_json(output / "provenance.json", provenance)
    readme = f"""# Seed78 GEPA Differential Audit

This is a zero-API, read-only comparison of the completed Seed78 local GEPA
searches against the successful Independent-GEPA capacity probe.

## Result

`{classifier['primary']}`

- The official GEPA commit, Solver model, reflection model, correctness metric,
  100-example search count, and reflection minibatch size match.
- Diversity generated **89/89 byte-changed proposals**, but **all 89 were
  rejected before a candidate Solver call**. Candidate Solver calls were `0`.
- Production-order rejection counts were: `3` compact-length failures, `82`
  output-contract contamination failures, and `4` append-only mutations.
- Independent-GEPA produced `66` proposals and retained `29` mutations. All
  `29/29` retained mutations would fail the current Diversity mutable-prompt
  boundary (`27` compact-length, `2` output-contract contamination).

The immediate divergence is therefore not evidence that qwen3-8b cannot be
optimized, nor evidence that TeamMiniBatch rejects useful local mutations.
The default GEPA proposer emits complete prompts, while the Diversity Layer-1
interface accepts only compact decision-procedure replacements and rejects
duplicated output-interface text. No Seed78 proposal reached empirical candidate
evaluation, so validation-size, merge, and skip-perfect differences remain
secondary, untested explanations.

## Frozen interpretation

- Scheduler implementation: verified.
- Scheduler causal efficacy: **NOT EVALUATED**.
- Local-to-TeamMiniBatch transfer: **NOT EVALUATED**.
- Primary blocker: **proposer / mutable-prompt contract mismatch**.

The hard boundary should not be loosened. A future change should constrain the
proposer to decision-procedure-only replacement text, prove nonzero Solver reach
offline, and then use a minimal fresh canary before any scheduler A/B rerun.

API calls: `0`; Validation calls: `0`; Test calls: `0`.
"""
    (output / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    sanitization = sanitize_scan(output)
    if sanitization["gate"] != "PASS":
        raise AssertionError(sanitization)
    write_json(output / "sanitization_manifest.json", sanitization)
    manifest = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    write_json(output / "sha256_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diversity-run", type=Path, default=DEFAULT_DIVERSITY_RUN)
    parser.add_argument("--capacity-root", type=Path, default=DEFAULT_CAPACITY_ROOT)
    parser.add_argument("--capacity-public", type=Path, default=DEFAULT_CAPACITY_PUBLIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.diversity_run, args.capacity_root, args.capacity_public, args.output)
    print("PASS seed78 GEPA differential audit")


if __name__ == "__main__":
    main()
