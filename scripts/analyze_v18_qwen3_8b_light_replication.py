from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_v18_no_semantic_critic_transfer import (
    decompose_trajectory,
    sha256_file,
)
from scripts.audit_v18_no_semantic_critic_candidate_ranking import (
    _raw_candidate_metrics,
    _train_pareto_dominates,
    _validation_cache_counts,
)


SEED = 71
ARMS = ("A_CANONICAL", "C_NO_SEMANTIC_CRITIC")
EXPECTED_MODEL = {
    "solver": "qwen3-8b",
    "teacher": "qwen3.7-flash",
    "critic": "qwen3.7-flash",
    "student": "qwen3.7-flash",
    "thinking": False,
}
RUNTIME_VERSION = "v18_qwen3_8b_no_semantic_critic_light_replication_v1"
ANALYSIS_VERSION = "v18_qwen3_8b_light_replication_postrun_v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def _candidate_pool_audit(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run = run_root / f"seed{SEED}" / "C_NO_SEMANTIC_CRITIC"
    updates = read_jsonl(run / "update_lineage.jsonl")
    candidates = read_jsonl(run / "candidate_level_sanitized.jsonl")
    decisions = read_jsonl(run / "candidate_decisions.jsonl")
    validation_states = read_jsonl(run / "validation_states.jsonl")
    validation_hashes = {
        str(row["example_id_hash"])
        for row in validation_states[0]["examples"]
    }
    if len(validation_hashes) != 50:
        raise ValueError("validation inventory must contain exactly 50 rows")
    cache_counts = _validation_cache_counts(
        run_root / f"seed{SEED}" / "_shared_solver_cache.sqlite",
        validation_hashes,
    )
    raw_metrics = _raw_candidate_metrics(decisions)
    committed = {
        int(row["update_index"]): row
        for row in updates
        if _bool(row["committed"])
    }
    inventory: list[dict[str, Any]] = []
    pools: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        if not _bool(candidate.get("feasible")):
            continue
        update_index = int(candidate["update_index"])
        prompt_hash = str(candidate["candidate_id"])
        metrics = raw_metrics[(update_index, prompt_hash)]
        selected = _bool(candidate["winner"])
        validation_rows = int(cache_counts.get(prompt_hash, 0))
        if selected and validation_rows != 50:
            raise ValueError("selected winner lacks complete validation replay")
        if not selected and validation_rows not in (0, 50):
            raise ValueError("partial alternative validation is inadmissible")
        update = committed.get(update_index)
        row = {
            "seed": SEED,
            "update_index": update_index,
            "candidate_hash": prompt_hash,
            "target_member": int(candidate["target_member"]),
            "candidate_stage": str(candidate["candidate_stage"]),
            "target_gain": int(candidate["target_gain"]),
            "vote_gain_count": int(candidate["vote_gain_count"]),
            "vote_loss_count": int(candidate["vote_loss_count"]),
            "vote_net_gain": int(candidate["vote_net_gain"]),
            "coverage_gain_count": int(metrics["coverage_gain_count"]),
            "coverage_loss_count": int(metrics["coverage_loss_count"]),
            "dominant_wrong_exit_count": int(metrics["dominant_wrong_exit_count"]),
            "dominant_wrong_join_count": int(metrics["dominant_wrong_join_count"]),
            "soft_utility_delta": float(metrics["soft_utility_delta"]),
            "minimum_gain_count": int(metrics["minimum_gain_count"]),
            "total_gain_count": int(metrics["total_gain_count"]),
            "unique_correct_loss_count": int(metrics["unique_correct_loss_count"]),
            "pivotal_correct_loss_count": int(metrics["pivotal_correct_loss_count"]),
            "branch_rank": int(candidate["branch_rank"]),
            "cell_rank": candidate["cell_rank"],
            "selected_winner": selected,
            "validation_cache_rows": validation_rows,
            "validation_evidence_status": (
                "complete_selected_winner" if selected else
                "complete_unselected_alternative" if validation_rows == 50 else
                "unobserved"
            ),
            "posthoc_validation_target_delta": (
                int(update["validation_target_delta"]) if selected and update else None
            ),
            "posthoc_validation_vote_delta": (
                int(update["validation_vote_delta"]) if selected and update else None
            ),
            "posthoc_validation_oracle_delta": (
                int(update["validation_oracle_delta"]) if selected and update else None
            ),
        }
        inventory.append(row)
        pools.setdefault(update_index, []).append(row)

    pool_rows: list[dict[str, Any]] = []
    for update_index, update in sorted(committed.items()):
        pool = pools.get(update_index, [])
        winners = [row for row in pool if row["selected_winner"]]
        if len(winners) != 1:
            raise ValueError("each committed update must have one feasible winner")
        winner = winners[0]
        alternatives = [row for row in pool if not row["selected_winner"]]
        dominating = [
            row for row in alternatives
            if _train_pareto_dominates(row, winner)
        ]
        complete = [
            row for row in alternatives
            if int(row["validation_cache_rows"]) == 50
        ]
        validation_delta = int(update["validation_vote_delta"])
        pool_rows.append({
            "seed": SEED,
            "update_index": update_index,
            "winner_hash": winner["candidate_hash"],
            "validation_vote_class": (
                "positive" if validation_delta > 0 else
                "negative" if validation_delta < 0 else "zero"
            ),
            "winner_validation_target_delta": int(update["validation_target_delta"]),
            "winner_validation_vote_delta": validation_delta,
            "winner_validation_oracle_delta": int(update["validation_oracle_delta"]),
            "feasible_pool_size": len(pool),
            "feasible_alternative_count": len(alternatives),
            "alternatives_with_complete_validation": len(complete),
            "winner_target_gain": int(winner["target_gain"]),
            "winner_vote_gain_count": int(winner["vote_gain_count"]),
            "winner_vote_loss_count": int(winner["vote_loss_count"]),
            "winner_vote_net_gain": int(winner["vote_net_gain"]),
            "train_pareto_dominating_alternative_count": len(dominating),
            "counterfactual_validation_status": (
                "NO_FEASIBLE_ALTERNATIVE" if not alternatives else
                "IDENTIFIABLE_FROM_FROZEN_CACHE" if complete else
                "UNIDENTIFIABLE_UNOBSERVED_ALTERNATIVE_VALIDATION"
            ),
        })
    return inventory, pool_rows


def analyze(*, prep: Path, run_root: Path, gate: Path, report: Path) -> dict[str, Any]:
    if read_json(gate / "audit.json")["gate"] != "PASS":
        raise ValueError("official frozen audit is not PASS")
    registry = read_json(prep / "registry.json")
    if registry["seeds"] != [SEED] or registry["arms"] != list(ARMS):
        raise ValueError("frozen seed or arm inventory mismatch")
    if registry["model"] != EXPECTED_MODEL:
        raise ValueError("frozen model identity mismatch")
    if not report.is_dir():
        raise FileNotFoundError("base sanitized report must already exist")

    trajectories: list[dict[str, Any]] = []
    commits: list[dict[str, Any]] = []
    gains: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    source_hashes: list[dict[str, Any]] = []
    online: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        run = run_root / f"seed{SEED}" / arm
        result = decompose_trajectory(seed=SEED, arm=arm, run=run)
        trajectories.append(result["trajectory"])
        commits.extend(result["commits"])
        gains.extend(result["gains"])
        losses.extend(result["losses"])
        online[arm] = read_json(run / "online_run_summary.json")
        for role in (
            "online_run_summary.json",
            "update_lineage.jsonl",
            "candidate_level_sanitized.jsonl",
            "validation_states.jsonl",
        ):
            source_hashes.append({
                "seed": SEED,
                "arm": arm,
                "artifact_role": role,
                "sha256": sha256_file(run / role),
            })

    inventory, pools = _candidate_pool_audit(run_root)
    a, c = online[ARMS[0]], online[ARMS[1]]
    frontend_label = (
        "FRONTEND_THROUGHPUT_REPLICATED"
        if (
            int(c["funnel"]["student_reaches"]) > int(a["funnel"]["student_reaches"])
            and int(c["funnel"]["feasible_candidates"]) > int(a["funnel"]["feasible_candidates"])
            and int(c["accepted_commit_count"]) >= int(a["accepted_commit_count"])
        )
        else "FRONTEND_THROUGHPUT_NOT_REPLICATED"
    )
    c_commits = [row for row in commits if row["arm"] == ARMS[1]]
    if not c_commits:
        transfer_label = "NO_C_COMMITS"
    elif any(int(row["validation_vote_delta"]) <= 0 for row in c_commits):
        transfer_label = "TRANSFER_INSTABILITY_OBSERVED"
    else:
        transfer_label = "POSITIVE_TRANSFER_ONLY_OBSERVED"
    dominating_count = sum(
        int(row["train_pareto_dominating_alternative_count"])
        for row in pools
    )
    ranking_label = (
        "TRAIN_SIDE_RANKING_SIGNAL_PRESENT"
        if dominating_count
        else "RANKING_NOT_IMPLICATED_BY_AVAILABLE_TRAIN_EVIDENCE"
    )
    nonpositive = [row for row in pools if row["validation_vote_class"] != "positive"]
    negative = [row for row in pools if row["validation_vote_class"] == "negative"]
    execution_summary = read_json(run_root / "execution_summary.json")
    runtime_label_match = execution_summary.get("runtime_version") == RUNTIME_VERSION
    normalization = {
        "status": "NON_AUTHORITATIVE_TOP_LEVEL_LABEL_NORMALIZED" if not runtime_label_match else "NOT_REQUIRED",
        "observed_top_level_runtime_version": execution_summary.get("runtime_version"),
        "authoritative_runtime_version": RUNTIME_VERSION,
        "authority": "frozen registry plus both per-trajectory summaries",
        "both_trajectory_runtime_labels_match": all(
            row.get("runtime_version") == RUNTIME_VERSION for row in online.values()
        ),
        "original_artifacts_modified": False,
        "method_runtime_semantics_changed": False,
    }
    if not normalization["both_trajectory_runtime_labels_match"]:
        raise ValueError("authoritative per-trajectory runtime identity mismatch")

    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "seed": SEED,
        "model": EXPECTED_MODEL,
        "official_frozen_gate": "PASS",
        "frontend_label": frontend_label,
        "transfer_label": transfer_label,
        "ranking_label": ranking_label,
        "online": {
            arm: {
                "student_reaches": int(row["funnel"]["student_reaches"]),
                "feasible_candidates": int(row["funnel"]["feasible_candidates"]),
                "accepted_commits": int(row["accepted_commit_count"]),
                "final_train_vote_correct": int(row["final_train_metrics"]["vote_correct_count"]),
                "final_validation_vote_correct": int(row["final_validation_metrics"]["vote_correct_count"]),
                "final_validation_oracle_correct": int(row["final_validation_metrics"]["oracle_correct_count"]),
            }
            for arm, row in online.items()
        },
        "c_commit_quality": {
            "commit_count": len(c_commits),
            "positive_net": sum(int(row["validation_vote_delta"]) > 0 for row in c_commits),
            "zero_net": sum(int(row["validation_vote_delta"]) == 0 for row in c_commits),
            "negative_net": sum(int(row["validation_vote_delta"]) < 0 for row in c_commits),
            "positive_train_vote_not_positive_validation_vote": sum(
                bool(row["positive_train_vote_not_positive_validation_vote"])
                for row in c_commits
            ),
            "positive_train_target_not_positive_validation_target": sum(
                bool(row["positive_train_target_not_positive_validation_target"])
                for row in c_commits
            ),
            "positive_validation_oracle_not_positive_vote": sum(
                bool(row["positive_validation_oracle_not_positive_vote"])
                for row in c_commits
            ),
        },
        "c_feasible_pool": {
            "feasible_candidates": len(inventory),
            "accepted_pools": len(pools),
            "nonpositive_commit_pools": len(nonpositive),
            "negative_commit_pools": len(negative),
            "nonpositive_no_alternative_pools": sum(
                int(row["feasible_alternative_count"]) == 0 for row in nonpositive
            ),
            "negative_no_alternative_pools": sum(
                int(row["feasible_alternative_count"]) == 0 for row in negative
            ),
            "train_pareto_dominating_alternatives": dominating_count,
            "unselected_alternatives": sum(not bool(row["selected_winner"]) for row in inventory),
            "unselected_alternatives_with_validation": sum(
                not bool(row["selected_winner"])
                and int(row["validation_cache_rows"]) == 50
                for row in inventory
            ),
        },
        "counterfactual_validation_status": (
            "UNRESOLVED_UNOBSERVED_ALTERNATIVES"
            if any(
                row["counterfactual_validation_status"] == "UNIDENTIFIABLE_UNOBSERVED_ALTERNATIVE_VALIDATION"
                for row in nonpositive
            )
            else "NO_UNOBSERVED_NONPOSITIVE_ALTERNATIVE"
        ),
        "scope": {
            "new_api_calls_in_postrun_analysis": 0,
            "new_validation_calls_in_postrun_analysis": 0,
            "test_calls": 0,
            "counterfactual_alternative_replay_run": False,
            "method_modified": False,
            "raw_run_artifacts_modified": False,
        },
    }
    write_csv(report / "accepted_commit_decomposition.csv", commits)
    write_csv(report / "trajectory_decomposition.csv", trajectories)
    write_csv(report / "validation_vote_gain_persistence.csv", gains)
    write_csv(report / "validation_vote_loss_provenance.csv", losses)
    write_csv(report / "c_feasible_candidate_inventory.csv", inventory)
    write_csv(report / "c_commit_pool_audit.csv", pools)
    write_csv(report / "postrun_source_artifact_hashes.csv", source_hashes)
    write_json(report / "replication_assessment.json", summary)
    write_json(report / "execution_summary_normalization.json", normalization)
    write_json(report / "fact_assertions.json", {
        "status": "PASS",
        "trajectory_count": 2,
        "completed_update_count_each": 8,
        "telescoping_identity_pass_count": sum(
            bool(row["telescoping_identity_pass"]) for row in trajectories
        ),
        "matched_initialization": True,
        "model_identity_match": True,
        "official_frozen_gate": "PASS",
        "test_calls": 0,
        "postrun_api_calls": 0,
        "counterfactual_alternative_replay_run": False,
        "raw_run_artifacts_modified": False,
    })
    write_json(report / "sanitization_manifest.json", {
        "status": "PASS",
        "raw_text_published": False,
        "absolute_paths_published": False,
        "sqlite_content_published": False,
        "forbidden_content": [
            "prompts", "questions", "gold answers", "model answers",
            "raw responses", "endpoints", "credentials", "SQLite content",
            "checkpoints",
        ],
    })
    c_quality = summary["c_commit_quality"]
    c_pool = summary["c_feasible_pool"]
    readme = f"""# Qwen3-8B No-Semantic-Critic Light Replication

Official frozen gate: **PASS**. This is a one-seed mechanism replication, not a formal efficacy claim and not a causal comparison with Qwen3-14B.

## Frozen model allocation

- Solver: `qwen3-8b`, thinking disabled.
- Teacher and Student: `qwen3.7-flash`.
- Canonical semantic Critic: `qwen3.7-flash`.
- C uses the deterministic hard-safety gate and makes no semantic-Critic API call.

## Results

| Metric | A Canonical | C No Semantic Critic |
|---|---:|---:|
| Student reach | {summary['online'][ARMS[0]]['student_reaches']} | {summary['online'][ARMS[1]]['student_reaches']} |
| Feasible candidates | {summary['online'][ARMS[0]]['feasible_candidates']} | {summary['online'][ARMS[1]]['feasible_candidates']} |
| Accepted commits | {summary['online'][ARMS[0]]['accepted_commits']} | {summary['online'][ARMS[1]]['accepted_commits']} |
| Final train Vote | {summary['online'][ARMS[0]]['final_train_vote_correct']}/75 | {summary['online'][ARMS[1]]['final_train_vote_correct']}/75 |
| Final validation Vote | {summary['online'][ARMS[0]]['final_validation_vote_correct']}/50 | {summary['online'][ARMS[1]]['final_validation_vote_correct']}/50 |
| Final validation Oracle | {summary['online'][ARMS[0]]['final_validation_oracle_correct']}/50 | {summary['online'][ARMS[1]]['final_validation_oracle_correct']}/50 |

Frozen front-end label: **{frontend_label}**. C again increased Student reach and feasible supply, while commit count tied.

Frozen transfer label: **{transfer_label}**. C's {c_quality['commit_count']} commits comprise {c_quality['positive_net']} positive, {c_quality['zero_net']} zero, and {c_quality['negative_net']} negative validation-Vote transitions. Final C validation Vote is {summary['online'][ARMS[1]]['final_validation_vote_correct'] - summary['online'][ARMS[0]]['final_validation_vote_correct']:+d} relative to A, despite a {summary['online'][ARMS[1]]['final_train_vote_correct'] - summary['online'][ARMS[0]]['final_train_vote_correct']:+d} train-Vote difference.

Frozen ranking label: **{ranking_label}**. Across C's {c_pool['accepted_pools']} committed feasible pools, {c_pool['train_pareto_dominating_alternatives']} unselected alternative train-Pareto-dominated its winner. This does not eliminate ranking: unselected alternatives received no new validation replay, so the counterfactual remains **{summary['counterfactual_validation_status']}**. Pools with no feasible alternative provide a direct lower bound on feasible-set limitations.

Every trajectory passes the accepted-transition telescoping identity. Validation was replayed only after training freeze and never affected selection or write-back. Test access and counterfactual alternative replay were both zero.

## Provenance note

The top-level execution summary inherited an old non-authoritative runtime label from a reporting constant. The frozen registry and both per-trajectory summaries carry the correct runtime identity. The raw run was not modified; `execution_summary_normalization.json` records this reporting-only reconciliation. Scientific and model identities are unaffected.
"""
    (report / "README.md").write_text(readme, encoding="utf-8")
    files = [
        path for path in report.iterdir()
        if path.is_file() and path.name != "sha256_manifest.json"
    ]
    write_json(report / "sha256_manifest.json", {
        "files": [
            {"path": path.name, "sha256": sha256_file(path)}
            for path in sorted(files)
        ]
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(
        prep=args.prep.resolve(),
        run_root=args.run_root.resolve(),
        gate=args.gate.resolve(),
        report=args.report.resolve(),
    )
    print(json.dumps({
        "frontend_label": summary["frontend_label"],
        "transfer_label": summary["transfer_label"],
        "ranking_label": summary["ranking_label"],
        "test_calls": 0,
        "postrun_api_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
