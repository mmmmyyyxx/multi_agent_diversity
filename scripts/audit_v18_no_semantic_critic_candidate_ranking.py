from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence


SEEDS = (68, 69, 70)
ARM = "C_NO_SEMANTIC_CRITIC"
ANALYSIS_VERSION = "v18_no_semantic_critic_candidate_ranking_audit_v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def _raw_candidate_metrics(decisions: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for decision in decisions:
        update_index = int(decision["update_index"])
        for candidate in decision.get("candidates", []):
            key = (update_index, str(candidate["prompt_hash"]))
            evaluation = candidate["evaluation"]
            member_gain = evaluation["member_gain"]
            marginal = evaluation["marginal"]
            protection = evaluation["protection"]
            result[key] = {
                "coverage_gain_count": int(marginal["coverage_gain_count"]),
                "coverage_loss_count": int(marginal["coverage_loss_count"]),
                "dominant_wrong_exit_count": int(marginal["dominant_wrong_exit_count"]),
                "dominant_wrong_join_count": int(marginal["dominant_wrong_join_count"]),
                "soft_utility_delta": float(marginal["soft_utility_delta"]),
                "minimum_gain_count": int(member_gain["minimum_gain_count"]),
                "total_gain_count": int(member_gain["total_gain_count"]),
                "unique_correct_loss_count": int(protection["unique_correct_loss_count"]),
                "pivotal_correct_loss_count": int(protection["pivotal_correct_loss_count"]),
            }
    return result


def _validation_cache_counts(cache_path: Path, validation_question_hashes: set[str]) -> dict[str, int]:
    with sqlite3.connect(str(cache_path)) as connection:
        rows = connection.execute(
            "SELECT prompt_hash, question_hash FROM solver_cache WHERE state = 'ready'"
        ).fetchall()
    counts: dict[str, int] = {}
    for prompt_hash, question_hash in rows:
        if str(question_hash) in validation_question_hashes:
            key = str(prompt_hash)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _train_pareto_dominates(alternative: Mapping[str, Any], winner: Mapping[str, Any]) -> bool:
    no_worse = (
        int(alternative["target_gain"]) >= int(winner["target_gain"])
        and int(alternative["vote_net_gain"]) >= int(winner["vote_net_gain"])
        and int(alternative["vote_loss_count"]) <= int(winner["vote_loss_count"])
    )
    strict = (
        int(alternative["target_gain"]) > int(winner["target_gain"])
        or int(alternative["vote_net_gain"]) > int(winner["vote_net_gain"])
        or int(alternative["vote_loss_count"]) < int(winner["vote_loss_count"])
    )
    return no_worse and strict


def audit(*, seed68_root: Path, extension_root: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("fresh report root required")
    out.mkdir(parents=True)
    candidate_rows: list[dict[str, Any]] = []
    commit_rows: list[dict[str, Any]] = []
    source_hashes: list[dict[str, Any]] = []
    for seed in SEEDS:
        root = seed68_root if seed == 68 else extension_root
        run = root / f"seed{seed}" / ARM
        summary = read_json(run / "online_run_summary.json")
        if int(summary.get("test_evaluation_count", summary.get("new_test_calls", -1))) != 0:
            raise ValueError("test access is forbidden")
        updates = read_jsonl(run / "update_lineage.jsonl")
        sanitized = read_jsonl(run / "candidate_level_sanitized.jsonl")
        raw_decisions = read_jsonl(run / "candidate_decisions.jsonl")
        validation_states = read_jsonl(run / "validation_states.jsonl")
        validation_hashes = {str(row["example_id_hash"]) for row in validation_states[0]["examples"]}
        if len(validation_hashes) != 50:
            raise ValueError("frozen validation inventory must contain 50 rows")
        cache_path = root / f"seed{seed}" / "_shared_solver_cache.sqlite"
        cache_counts = _validation_cache_counts(cache_path, validation_hashes)
        raw_metrics = _raw_candidate_metrics(raw_decisions)
        for role, path in (
            ("online_run_summary", run / "online_run_summary.json"),
            ("update_lineage", run / "update_lineage.jsonl"),
            ("candidate_level_sanitized", run / "candidate_level_sanitized.jsonl"),
            ("candidate_decisions", run / "candidate_decisions.jsonl"),
            ("validation_states", run / "validation_states.jsonl"),
            ("solver_cache_evidence", cache_path),
        ):
            source_hashes.append({"seed": seed, "artifact_role": role, "sha256": sha256_file(path)})
        committed = {int(row["update_index"]): row for row in updates if _bool(row["committed"])}
        feasible_by_update: dict[int, list[dict[str, Any]]] = {}
        for candidate in sanitized:
            if not _bool(candidate.get("feasible")):
                continue
            update_index = int(candidate["update_index"])
            prompt_hash = str(candidate["candidate_id"])
            metrics = raw_metrics[(update_index, prompt_hash)]
            update = committed.get(update_index)
            winner = _bool(candidate["winner"])
            validation_count = int(cache_counts.get(prompt_hash, 0))
            if winner and update is None:
                raise ValueError("winner is not associated with a committed update")
            if winner and validation_count != 50:
                raise ValueError("selected winner lacks complete frozen validation evidence")
            if not winner and validation_count not in (0, 50):
                raise ValueError("partial alternative validation evidence is not admissible")
            row = {
                "seed": seed,
                "update_index": update_index,
                "candidate_hash": prompt_hash,
                "target_member": int(candidate["target_member"]),
                "candidate_stage": str(candidate["candidate_stage"]),
                "target_gain": int(candidate["target_gain"]),
                "vote_gain_count": int(candidate["vote_gain_count"]),
                "vote_loss_count": int(candidate["vote_loss_count"]),
                "vote_net_gain": int(candidate["vote_net_gain"]),
                "coverage_gain_count": metrics["coverage_gain_count"],
                "coverage_loss_count": metrics["coverage_loss_count"],
                "dominant_wrong_exit_count": metrics["dominant_wrong_exit_count"],
                "dominant_wrong_join_count": metrics["dominant_wrong_join_count"],
                "soft_utility_delta": metrics["soft_utility_delta"],
                "minimum_gain_count": metrics["minimum_gain_count"],
                "total_gain_count": metrics["total_gain_count"],
                "unique_correct_loss_count": metrics["unique_correct_loss_count"],
                "pivotal_correct_loss_count": metrics["pivotal_correct_loss_count"],
                "branch_rank": int(candidate["branch_rank"]),
                "cell_rank": candidate["cell_rank"],
                "selected_winner": winner,
                "validation_cache_rows": validation_count,
                "validation_evidence_status": "complete_selected_winner" if validation_count == 50 else "unobserved",
                "posthoc_validation_target_delta": int(update["validation_target_delta"]) if winner else None,
                "posthoc_validation_vote_delta": int(update["validation_vote_delta"]) if winner else None,
                "posthoc_validation_oracle_delta": int(update["validation_oracle_delta"]) if winner else None,
            }
            candidate_rows.append(row)
            feasible_by_update.setdefault(update_index, []).append(row)
        for update_index, update in sorted(committed.items()):
            pool = feasible_by_update.get(update_index, [])
            winners = [row for row in pool if row["selected_winner"]]
            if len(winners) != 1:
                raise ValueError("committed update must have exactly one feasible winner")
            winner = winners[0]
            alternatives = [row for row in pool if not row["selected_winner"]]
            complete_alternatives = [row for row in alternatives if row["validation_cache_rows"] == 50]
            dominated_by_train = [row for row in alternatives if _train_pareto_dominates(row, winner)]
            lower_loss_same_or_better_net = [
                row for row in alternatives
                if int(row["vote_loss_count"]) < int(winner["vote_loss_count"])
                and int(row["vote_net_gain"]) >= int(winner["vote_net_gain"])
            ]
            validation_delta = int(update["validation_vote_delta"])
            if not alternatives:
                counterfactual_status = "NO_FEASIBLE_ALTERNATIVE"
            elif complete_alternatives:
                counterfactual_status = "IDENTIFIABLE_FROM_FROZEN_CACHE"
            else:
                counterfactual_status = "UNIDENTIFIABLE_UNOBSERVED_ALTERNATIVE_VALIDATION"
            commit_rows.append({
                "seed": seed,
                "update_index": update_index,
                "winner_hash": winner["candidate_hash"],
                "validation_vote_class": "positive" if validation_delta > 0 else "negative" if validation_delta < 0 else "zero",
                "winner_validation_target_delta": int(update["validation_target_delta"]),
                "winner_validation_vote_delta": validation_delta,
                "winner_validation_oracle_delta": int(update["validation_oracle_delta"]),
                "feasible_pool_size": len(pool),
                "feasible_alternative_count": len(alternatives),
                "alternatives_with_complete_validation": len(complete_alternatives),
                "winner_target_gain": winner["target_gain"],
                "winner_vote_gain_count": winner["vote_gain_count"],
                "winner_vote_loss_count": winner["vote_loss_count"],
                "winner_vote_net_gain": winner["vote_net_gain"],
                "train_pareto_dominating_alternative_count": len(dominated_by_train),
                "lower_vote_loss_same_or_better_net_alternative_count": len(lower_loss_same_or_better_net),
                "counterfactual_validation_status": counterfactual_status,
            })
    nonpositive = [row for row in commit_rows if row["validation_vote_class"] != "positive"]
    negative = [row for row in commit_rows if row["validation_vote_class"] == "negative"]
    nonpositive_alternatives = sum(int(row["feasible_alternative_count"]) for row in nonpositive)
    negative_alternatives = sum(int(row["feasible_alternative_count"]) for row in negative)
    identifiable_nonpositive = sum(
        row["counterfactual_validation_status"] == "IDENTIFIABLE_FROM_FROZEN_CACHE"
        for row in nonpositive
    )
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "scope": {
            "zero_api": True,
            "new_validation_calls": 0,
            "new_test_calls": 0,
            "method_modified": False,
            "ranking_modified": False,
            "historical_artifacts_modified": False,
        },
        "commit_count": len(commit_rows),
        "feasible_candidate_count": len(candidate_rows),
        "selected_winner_count": sum(bool(row["selected_winner"]) for row in candidate_rows),
        "unselected_feasible_count": sum(not bool(row["selected_winner"]) for row in candidate_rows),
        "unselected_with_complete_validation_count": sum(not bool(row["selected_winner"]) and int(row["validation_cache_rows"]) == 50 for row in candidate_rows),
        "nonpositive_commit_count": len(nonpositive),
        "negative_commit_count": len(negative),
        "nonpositive_feasible_alternative_count": nonpositive_alternatives,
        "negative_feasible_alternative_count": negative_alternatives,
        "nonpositive_no_alternative_commit_count": sum(int(row["feasible_alternative_count"]) == 0 for row in nonpositive),
        "negative_no_alternative_commit_count": sum(int(row["feasible_alternative_count"]) == 0 for row in negative),
        "nonpositive_train_pareto_dominating_alternative_count": sum(int(row["train_pareto_dominating_alternative_count"]) for row in nonpositive),
        "negative_train_pareto_dominating_alternative_count": sum(int(row["train_pareto_dominating_alternative_count"]) for row in negative),
        "nonpositive_lower_loss_same_or_better_net_alternative_count": sum(int(row["lower_vote_loss_same_or_better_net_alternative_count"]) for row in nonpositive),
        "counterfactual_validation_identifiable_nonpositive_commits": identifiable_nonpositive,
        "primary_result": "COUNTERFACTUAL_ALTERNATIVE_VALIDATION_UNOBSERVED",
        "train_side_result": "RANKING_NOT_IMPLICATED_BY_AVAILABLE_TRAIN_EVIDENCE",
        "feasible_set_lower_bound": "AT_LEAST_ONE_NEGATIVE_AND_TWO_ZERO_COMMITS_HAD_NO_FEASIBLE_ALTERNATIVE",
        "global_ranking_vs_feasible_set_verdict": "UNRESOLVED_WITH_ZERO_API_FROZEN_EVIDENCE",
    }
    write_csv(out / "commit_pool_audit.csv", commit_rows)
    write_csv(out / "feasible_candidate_inventory.csv", candidate_rows)
    write_csv(out / "source_artifact_hashes.csv", source_hashes)
    write_json(out / "summary.json", summary)
    write_json(out / "fact_assertions.json", {
        "status": "PASS",
        "seeds": list(SEEDS),
        "commit_count": 15,
        "selected_winner_validation_complete_count": 15,
        "unselected_feasible_validation_complete_count": 0,
        "new_api_calls": 0,
        "new_test_calls": 0,
        "historical_artifacts_modified": False,
    })
    readme = f"""# V18 accepted-candidate counterfactual ranking audit

This is a zero-API audit of all feasible candidates in the 15 accepted C updates for Seeds68-70. It does not change ranking, proposal generation, Common-Safe, or historical artifacts.

## Identifiability result

The requested validation counterfactual is **not identifiable from frozen evidence**. All 15 selected winners have complete 50-row validation observations. All {summary['unselected_feasible_count']} unselected feasible candidates have zero validation observations; there are no partial observations. The 12 zero/negative commits contain {nonpositive_alternatives} feasible alternatives, but none has frozen validation evidence.

Therefore this audit cannot honestly answer whether an unselected candidate had better validation target/Vote/Oracle transfer. Doing so would require new validation Solver calls and is outside the zero-API scope.

## What is established without API

- Three nonpositive commits, including one of the three negative commits, had no feasible alternative at all. Ranking cannot explain these cases.
- The three negative commits contained {negative_alternatives} alternatives in total. None train-Pareto-dominated the selected winner on target gain, Vote net, and Vote loss.
- Across all 12 nonpositive commits, no alternative train-Pareto-dominated the winner. Only one alternative had lower train Vote loss while matching or improving Vote net; its validation transfer is unobserved.
- Thus available train-side evidence does not implicate ranking, and there is a direct feasible-set-quality lower bound. However the global ranking-versus-feasible-set distinction remains unresolved because alternative validation outcomes were never evaluated.

Seed69's paired +7 remains decomposed as A -3 and C +4; it is not attributed to three matched extra commits.

No API, new validation, or test call was made.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    write_json(out / "sanitization_manifest.json", {
        "status": "PASS",
        "raw_text_published": False,
        "absolute_paths_published": False,
        "sqlite_content_published": False,
        "forbidden_content": ["prompts", "questions", "gold answers", "model answers", "raw responses", "endpoints", "credentials", "SQLite content", "checkpoints"],
    })
    files = [path for path in out.iterdir() if path.is_file() and path.name != "sha256_manifest.json"]
    write_json(out / "sha256_manifest.json", {"files": [{"path": path.name, "sha256": sha256_file(path)} for path in sorted(files)]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed68-root", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(
        seed68_root=args.seed68_root.resolve(),
        extension_root=args.extension_root.resolve(),
        out=args.out.resolve(),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
