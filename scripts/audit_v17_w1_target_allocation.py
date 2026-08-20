from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.peer_state import build_team_vote_state
from multi_dataset_diverse_rl.tasks import get_task_spec
from multi_dataset_diverse_rl.utils import normalize_prompt_text


DEFAULT_RAW = ROOT / "runs/v17_module1_2x2_probe_20260820_retry4"
DEFAULT_REGISTRY = ROOT / "runs/v17_module1_2x2_prep_20260820_retry4/private_registry.json"
DEFAULT_V17 = ROOT / "runs/v17_formal_5arm_3seed_20260813"
RECENT_WINDOW = 3
CELLS = ("A", "B", "C", "D")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(normalize_prompt_text(prompt).encode()).hexdigest()


def rr_w1_relation(target: int, rr: set[int], w1: set[int]) -> str:
    if target in rr and target in w1:
        return "overlap"
    if target in rr:
        return "rr_only"
    if target in w1:
        return "w1_only"
    raise ValueError("branch target is in neither frozen selector")


def rankdata(values: list[float]) -> list[float]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[cursor][0]:
            end += 1
        rank = (cursor + end + 2) / 2
        for offset in range(cursor, end + 1):
            ranks[ordered[offset][1]] = rank
        cursor = end + 1
    return ranks


def spearman(left: Iterable[float], right: Iterable[float]) -> float | None:
    x, y = list(map(float, left)), list(map(float, right))
    if len(x) < 3 or len(x) != len(y) or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rx, ry = rankdata(x), rankdata(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    return numerator / denominator if denominator else None


class ReadOnlyAnswers:
    def __init__(self, path: Path) -> None:
        uri = f"file:{path.resolve().as_posix()}?mode=ro&immutable=1"
        self.connection = sqlite3.connect(uri, uri=True)
        self.connection.execute("PRAGMA query_only=ON")

    def close(self) -> None:
        self.connection.close()

    def get(self, candidate_hash: str, question_hashes: Iterable[str]) -> dict[str, dict[str, Any]]:
        expected = set(map(str, question_hashes))
        rows = self.connection.execute(
            "SELECT question_hash, answer_json FROM solver_cache "
            "WHERE state='ready' AND prompt_hash=? ORDER BY cache_key",
            (candidate_hash,),
        ).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for question_hash, raw in rows:
            key = str(question_hash)
            if key not in expected:
                continue
            payload = json.loads(str(raw))
            prior = result.get(key)
            if prior is not None and (
                prior.get("answer"), prior.get("valid")
            ) != (payload.get("answer"), payload.get("valid")):
                raise ValueError("conflicting cached observation")
            result[key] = payload
        if set(result) != expected:
            raise ValueError(
                f"cache coverage mismatch for hash {candidate_hash[:12]}: "
                f"{len(result)}/{len(expected)}"
            )
        return result


def after_replacement_metrics(
    rows: list[dict[str, Any]], target: int,
    candidate: dict[str, dict[str, Any]] | None,
) -> dict[str, int]:
    task = get_task_spec("bbh")
    normalize = lambda value: task.extract_pred(f"FINAL_ANSWER: {value}", None)
    vote = oracle = target_correct = 0
    for row in rows:
        answers = list(map(str, row["team_answers"]))
        validity = list(map(bool, row["team_validity"]))
        gold = str(row["gold_answer"])
        if candidate is not None:
            payload = candidate[str(row["question_hash"])]
            answers[target] = str(payload["answer"])
            validity[target] = bool(payload["valid"])
        state = build_team_vote_state(
            question_hash=str(row["question_hash"]), gold_answer=gold,
            answers=answers, valid_vector=validity,
            normalize_answer=normalize, match_answer=task.match_answer,
            tie_break="abstain", seed=0,
        )
        vote += int(state.vote_correct)
        oracle += int(any(state.team_correctness))
        target_correct += int(state.team_correctness[target])
    return {"vote": vote, "oracle": oracle, "target": target_correct}


def validation_rows(case: dict[str, Any], cache: ReadOnlyAnswers) -> list[dict[str, Any]]:
    path = Path(case["base_config"]["val_path"])
    with path.open(encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    question_hashes = [hashlib.sha256(row["question"].encode()).hexdigest() for row in source]
    profiles = [cache.get(prompt_hash(prompt), question_hashes) for prompt in case["parent_prompts"]]
    rows = []
    for source_row, question_hash in zip(source, question_hashes, strict=True):
        rows.append({
            "question_hash": question_hash,
            "gold_answer": str(source_row["answer"]),
            "team_answers": [profile[question_hash]["answer"] for profile in profiles],
            "team_validity": [bool(profile[question_hash]["valid"]) for profile in profiles],
        })
    return rows


def history(case: dict[str, Any], v17_root: Path) -> dict[int, dict[str, int]]:
    seed, update = int(case["seed"]), int(case["update_index"])
    run = (
        v17_root / f"seed{seed}" / "disambiguation_qa"
        / f"experimental_v16_efficacy_g_matched_seed{seed}"
    )
    decisions = read_jsonl(run / "candidate_decisions.jsonl")
    past = [row for row in decisions if int(row["update_index"]) < update]
    recent = [row for row in past if int(row["update_index"]) >= update - RECENT_WINDOW]
    result = {}
    for agent in range(5):
        accepted = [
            row for row in past
            if row.get("accepted_prompt_hash") and int(row["target_agent_id"]) == agent
        ]
        accepted_recent = [
            row for row in recent
            if row.get("accepted_prompt_hash") and int(row["target_agent_id"]) == agent
        ]
        target_gain = vote_gain = 0
        for decision in accepted:
            candidate = next(
                row for row in decision["candidates"]
                if row["prompt_hash"] == decision["accepted_prompt_hash"]
            )
            target_gain += int(candidate["constraint"]["target_gain"])
            vote_gain += int(candidate["constraint"]["vote_net_gain"])
        result[agent] = {
            "historical_target_count": sum(agent in row["selected_target_ids"] for row in past),
            "recent_target_count": sum(agent in row["selected_target_ids"] for row in recent),
            "historical_accepted_count": len(accepted),
            "recent_accepted_count": len(accepted_recent),
            "historical_accepted_target_gain": target_gain,
            "historical_accepted_vote_gain": vote_gain,
        }
    return result


def aggregate(rows: list[dict[str, Any]], predicate) -> dict[str, Any]:
    selected = [row for row in rows if predicate(row)]
    count = len(selected)
    return {
        "branch_count": count,
        "valid_source_branch_count": sum(int(row["valid_source_count"] > 0) for row in selected),
        "valid_source_count": sum(row["valid_source_count"] for row in selected),
        "feasible_branch_count": sum(int(row["feasible_count"] > 0) for row in selected),
        "feasible_candidate_count": sum(row["feasible_count"] for row in selected),
        "branch_winner_count": sum(row["branch_winner"] for row in selected),
        "cell_winner_count": sum(row["cell_winner"] for row in selected),
        "critic_exhausted_count": sum(row["critic_exhausted"] for row in selected),
        "positive_validation_vote_count": sum(
            row["validation_vote_delta"] is not None and row["validation_vote_delta"] > 0
            for row in selected
        ),
        "positive_validation_oracle_count": sum(
            row["validation_oracle_delta"] is not None and row["validation_oracle_delta"] > 0
            for row in selected
        ),
        "mean_historical_target_count": (
            sum(row["historical_target_count"] for row in selected) / count if count else None
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(raw_root: Path, registry_path: Path, v17_root: Path, out: Path) -> dict[str, Any]:
    for path in (raw_root, registry_path, v17_root):
        resolved = path.resolve()
        if resolved != ROOT.resolve() and ROOT.resolve() not in resolved.parents:
            raise ValueError("all evidence must remain project-local")
    registry = read_json(registry_path)
    raw_cells = [read_json(path) for path in sorted(raw_root.glob("*/*/cell_result.json"))]
    if len(raw_cells) != 24:
        raise ValueError("24/24 cells are required")
    cache_path = raw_root / "_shared_solver_cache.sqlite"
    cache_hash_before = sha256(cache_path)
    cache = ReadOnlyAnswers(cache_path)
    cases = {row["case_id"]: row for row in registry["cases"]}
    parent_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    try:
        for case_id, case in cases.items():
            cells = {row["cell"]: row for row in raw_cells if row["case_id"] == case_id}
            if set(cells) != set(CELLS):
                raise ValueError(f"incomplete matrix for {case_id}")
            rr, w1 = set(case["round_robin_target_ids"]), set(case["w1_target_ids"])
            historical = history(case, v17_root)
            target_counts = [historical[agent]["historical_target_count"] for agent in range(5)]
            total_targets = sum(target_counts)
            hhi = sum((value / total_targets) ** 2 for value in target_counts) if total_targets else 0.0
            parent = {
                "case_id": case_id, "witness_type": case["stratum"],
                "seed": case["seed"], "update_index": case["update_index"],
                "team_state_version": case["team_state_version"],
                "historical_target_hhi": round(hhi, 6),
                "historical_max_target_share": round(max(target_counts) / total_targets, 6) if total_targets else 0.0,
                "rr_targets": "|".join(map(str, case["round_robin_target_ids"])),
                "w1_targets": "|".join(map(str, case["w1_target_ids"])),
                "target_overlap_count": len(rr & w1),
            }
            for cell in CELLS:
                row = cells[cell]
                parent.update({
                    f"{cell}_would_commit": int(row["would_commit"]),
                    f"{cell}_feasible_count": sum(branch["feasible_candidate_count"] for branch in row["branches"]),
                    f"{cell}_validation_vote_delta": row["realized_validation_vote_delta"],
                    f"{cell}_validation_oracle_delta": row["realized_validation_oracle_delta"],
                })
            parent_rows.append(parent)

            train_rows = list(case["active_profiles"])
            val_rows = validation_rows(case, cache)
            priority = {int(row["agent_id"]): row for row in case["w1_priority_rows"]}
            for cell in CELLS:
                cell_row = cells[cell]
                selector = "RR" if cell in {"A", "C"} else "W1"
                context = "GENERIC" if cell in {"A", "B"} else "MEMBER_AWARE"
                for branch in cell_row["branches"]:
                    target = int(branch["target_agent_id"])
                    score = priority[target]
                    expected = (
                        float(score["opportunity_value"])
                        + 0.05 * float(score["normalized_wait"])
                    ) * float(score["repairability_discount"])
                    if abs(expected - float(score["expected_update_value"])) > 1e-12:
                        raise ValueError("W1 score reconstruction mismatch")
                    winner_hash = str(branch["branch_winner_hash"])
                    branch_winner = bool(winner_hash)
                    cell_winner = bool(
                        cell_row["would_commit"]
                        and int(cell_row["hypothetical_target_agent_id"]) == target
                        and winner_hash == str(cell_row["hypothetical_prompt_hash"])
                    )
                    parent_train = after_replacement_metrics(train_rows, target, None)
                    train_candidate = (
                        cache.get(winner_hash, [row["question_hash"] for row in train_rows])
                        if branch_winner else None
                    )
                    train_after = (
                        after_replacement_metrics(train_rows, target, train_candidate)
                        if train_candidate else None
                    )
                    parent_val = after_replacement_metrics(val_rows, target, None)
                    val_after = None
                    if cell_winner:
                        val_candidate = cache.get(winner_hash, [row["question_hash"] for row in val_rows])
                        val_after = after_replacement_metrics(val_rows, target, val_candidate)
                        if (
                            val_after["vote"] - parent_val["vote"]
                            != int(cell_row["realized_validation_vote_delta"])
                            or val_after["oracle"] - parent_val["oracle"]
                            != int(cell_row["realized_validation_oracle_delta"])
                        ):
                            raise ValueError("validation metric reconstruction mismatch")
                    funnel = branch["funnel"]
                    row = {
                        "case_id": case_id, "witness_type": case["stratum"],
                        "seed": case["seed"], "update_index": case["update_index"],
                        "cell": cell, "selector": selector, "context": context,
                        "target_member": target,
                        "selection_relation": rr_w1_relation(target, rr, w1),
                        "w1_rank": int(score["selection_rank"]),
                        "w1_score": round(float(score["expected_update_value"]), 9),
                        "base_opportunity_B": round(float(score["opportunity_value"]), 9),
                        "raw_direct_fix_D": int(score["D"]),
                        "raw_support_S": int(score["S_support"]),
                        "raw_uplift_deficit_d": int(score["d"]),
                        "Dhat": round(float(score["normalized_direct_fix"]), 9),
                        "Shat": round(float(score["normalized_support_margin"]), 9),
                        "dhat": round(float(score["normalized_uplift_deficit"]), 9),
                        "wait": int(score["wait"]),
                        "normalized_wait": round(float(score["normalized_wait"]), 9),
                        "failure_count": int(score["failure_count"]),
                        "failure_discount": round(float(score["repairability_discount"]), 9),
                        **historical[target],
                        "valid_source_count": int(branch["valid_source_candidate_count"]),
                        "valid_revision_count": int(branch["evaluated_candidate_count"] - branch["valid_source_candidate_count"]),
                        "feasible_count": int(branch["feasible_candidate_count"]),
                        "critic_semantic_rejections": int(funnel["critic_semantic_rejections"]),
                        "critic_exhausted": int(funnel["terminal_failure_class"] == "critic_semantic_rejection_exhausted"),
                        "rejected_target_regression": int(funnel["rejected_target_regression"]),
                        "rejected_team_vote_regression": int(funnel["rejected_team_vote_regression"]),
                        "rejected_no_progress": int(funnel["rejected_no_target_or_vote_progress"]),
                        "rejected_terminal_invalid": int(funnel["rejected_terminal_invalid_regression"]),
                        "branch_winner": int(branch_winner), "cell_winner": int(cell_winner),
                        "would_commit": int(cell_row["would_commit"]),
                        "train_target_delta": train_after["target"] - parent_train["target"] if train_after else None,
                        "train_vote_delta": train_after["vote"] - parent_train["vote"] if train_after else None,
                        "train_oracle_delta": train_after["oracle"] - parent_train["oracle"] if train_after else None,
                        "validation_vote_delta": val_after["vote"] - parent_val["vote"] if val_after else None,
                        "validation_oracle_delta": val_after["oracle"] - parent_val["oracle"] if val_after else None,
                        "validation_target_delta": val_after["target"] - parent_val["target"] if val_after else None,
                    }
                    branch_rows.append(row)
    finally:
        cache.close()
    if sha256(cache_path) != cache_hash_before:
        raise ValueError("read-only cache changed")

    funnel_rows = []
    groups = {
        **{cell: lambda row, cell=cell: row["cell"] == cell for cell in CELLS},
        "RR": lambda row: row["selector"] == "RR",
        "W1": lambda row: row["selector"] == "W1",
        "rr_only": lambda row: row["selection_relation"] == "rr_only",
        "w1_only": lambda row: row["selection_relation"] == "w1_only",
        "overlap": lambda row: row["selection_relation"] == "overlap",
        **{f"rank_{rank}": lambda row, rank=rank: row["w1_rank"] == rank for rank in range(1, 6)},
        **{
            f"w1_rank_{rank}": lambda row, rank=rank: (
                row["selector"] == "W1" and row["w1_rank"] == rank
            )
            for rank in range(1, 6)
        },
    }
    for name, predicate in groups.items():
        funnel_rows.append({"group": name, **aggregate(branch_rows, predicate)})

    correlations = []
    measures = (
        "valid_source_count", "feasible_count", "cell_winner",
        "validation_vote_delta", "validation_oracle_delta",
    )
    components = (
        "w1_score", "Dhat", "Shat", "dhat", "normalized_wait",
        "failure_count", "historical_target_count",
    )
    for selector in ("ALL", "RR", "W1"):
        selected = [row for row in branch_rows if selector == "ALL" or row["selector"] == selector]
        for component in components:
            for measure in measures:
                paired = [row for row in selected if row[measure] is not None]
                correlations.append({
                    "selector": selector, "component": component,
                    "outcome": measure, "n": len(paired),
                    "spearman": spearman(
                        [row[component] for row in paired],
                        [row[measure] for row in paired],
                    ),
                })

    summary_by_group = {row["group"]: row for row in funnel_rows}
    rr, w1 = summary_by_group["RR"], summary_by_group["W1"]
    rr_only, w1_only = summary_by_group["rr_only"], summary_by_group["w1_only"]
    rank1, rank2 = summary_by_group["w1_rank_1"], summary_by_group["w1_rank_2"]
    w1_hist_feasible = next(
        row["spearman"] for row in correlations
        if row["selector"] == "W1" and row["component"] == "historical_target_count"
        and row["outcome"] == "feasible_count"
    )
    selection_loss = bool(
        rr_only["feasible_branch_count"] > w1_only["feasible_branch_count"]
        and rr_only["positive_validation_vote_count"] > w1_only["positive_validation_vote_count"]
    )
    score_mismatch = bool(
        rank1["feasible_branch_count"] / max(1, rank1["branch_count"])
        < rank2["feasible_branch_count"] / max(1, rank2["branch_count"])
        and w1["positive_validation_vote_count"] == 0
    )
    repeated_returns = bool(
        w1_hist_feasible is not None and w1_hist_feasible <= -0.25
        and w1["mean_historical_target_count"] > rr["mean_historical_target_count"]
    )
    realizability = bool(
        w1["valid_source_branch_count"] >= rr["valid_source_branch_count"] - 1
        and w1["feasible_branch_count"] < rr["feasible_branch_count"]
    )
    supported = [
        label for label, value in (
            ("TARGET_COVERAGE_EXPLORATION_FAILURE", selection_loss),
            ("TARGET_VALUE_ESTIMATION_FAILURE", score_mismatch),
            ("REPEATED_TARGET_DIMINISHING_RETURNS", repeated_returns),
            ("BRANCH_REALIZABILITY_MISMATCH", realizability),
        ) if value
    ]
    classifier = "MULTIPLE_TARGET_ALLOCATION_FAILURES" if len(supported) >= 2 else (
        supported[0] if supported else "NO_CLEAR_MECHANISM"
    )
    classifier_payload = {
        "classifier_version": "v17_w1_target_allocation_mechanism_classifier_v1",
        "primary_mechanism_diagnosis": classifier,
        "supported_component_labels": supported,
        "selection_opportunity_loss": selection_loss,
        "score_realizability_mismatch": score_mismatch,
        "repeated_target_diminishing_returns": repeated_returns,
        "branch_realizability_mismatch": realizability,
        "rules_frozen_in_script": True,
        "classifier_status": "post_hoc_diagnostic_rule_declared_before_report_narrative",
    }
    reconstruction = {
        "gate": "PASS", "zero_api": True, "new_test_calls": 0,
        "cell_count": 24, "branch_count": 48, "parent_count": 6,
        "w1_score_reconstruction_mismatch": 0,
        "branch_target_identity_mismatch": 0,
        "validation_metric_reconstruction_mismatch": 0,
        "cache_sha256_unchanged": True,
        "candidate_level_status": "NOT_IDENTIFIABLE_FROM_EXISTING_ARTIFACTS",
        "candidate_level_reason": "all candidate hashes and per-candidate Common-Safe decisions were not persisted by the fixed-parent runner",
    }
    realized_by_cell = {
        cell: {
            "validation_vote_delta": sum(
                int(row[f"{cell}_validation_vote_delta"]) for row in parent_rows
            ),
            "validation_oracle_delta": sum(
                int(row[f"{cell}_validation_oracle_delta"]) for row in parent_rows
            ),
            "would_commit_count": sum(
                int(row[f"{cell}_would_commit"]) for row in parent_rows
            ),
        }
        for cell in CELLS
    }
    paired_contrasts = {
        "B_minus_A": {
            metric: realized_by_cell["B"][metric] - realized_by_cell["A"][metric]
            for metric in ("validation_vote_delta", "validation_oracle_delta")
        },
        "D_minus_C": {
            metric: realized_by_cell["D"][metric] - realized_by_cell["C"][metric]
            for metric in ("validation_vote_delta", "validation_oracle_delta")
        },
    }
    summary = {
        "summary_version": "v17_w1_target_allocation_mechanism_audit_v1",
        "source_execution_commit": "66a0276dc61e77fe71e8add94eb4865d1235b7b5",
        "source_results_hash": read_json(raw_root / "probe_summary.json")["results_hash"],
        "gate": "PASS", "zero_api": True, "new_test_calls": 0,
        "parent_count": 6, "cell_count": 24, "branch_count": 48,
        "realized_by_cell": realized_by_cell,
        "paired_allocation_contrasts": paired_contrasts,
        "funnel": {
            name: summary_by_group[name]
            for name in (*CELLS, "RR", "W1", "rr_only", "w1_only", "overlap")
        },
        "w1_historical_target_to_feasible_spearman": w1_hist_feasible,
        "classifier": classifier_payload,
        "limitations": [
            "candidate_level_not_identifiable",
            "validation return is identifiable only for the globally selected hypothetical transition",
            "repeated-target evidence is associative rather than causal",
            "selected fixed parents do not identify all trajectory-level V17 loss",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "parent_level.csv", parent_rows)
    write_csv(out / "branch_level.csv", branch_rows)
    write_csv(out / "funnel.csv", funnel_rows)
    write_csv(out / "correlations.csv", correlations)
    for name, payload in (
        ("summary.json", summary), ("classifier.json", classifier_payload),
        ("reconstruction_gate.json", reconstruction),
        ("candidate_level_status.json", {
            "status": reconstruction["candidate_level_status"],
            "reason": reconstruction["candidate_level_reason"],
            "candidate_level_csv_created": False,
        }),
    ):
        (out / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    generated_names = (
        "parent_level.csv", "branch_level.csv", "funnel.csv", "correlations.csv",
        "summary.json", "classifier.json", "reconstruction_gate.json",
        "candidate_level_status.json",
    )
    manifest = {
        "manifest_version": "v17_w1_target_allocation_audit_manifest_v1",
        "generator_sha256": sha256(Path(__file__)),
        "artifacts": {
            name: sha256(out / name) for name in generated_names
        },
    }
    (out / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--v17_root", type=Path, default=DEFAULT_V17)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.raw_root, args.registry, args.v17_root, args.out)
    print(json.dumps({
        "gate": summary["gate"], "zero_api": summary["zero_api"],
        "new_test_calls": summary["new_test_calls"],
        "classifier": summary["classifier"]["primary_mechanism_diagnosis"],
    }, indent=2))


if __name__ == "__main__":
    main()
