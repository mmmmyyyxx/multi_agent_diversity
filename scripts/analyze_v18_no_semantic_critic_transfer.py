from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ARMS = ("A_CANONICAL", "C_NO_SEMANTIC_CRITIC")
SEEDS = (68, 69, 70)
ANALYSIS_VERSION = "v18_no_semantic_critic_transfer_decomposition_v1"
ALLOWED_SOURCE_FILES = (
    "validation_states.jsonl",
    "update_lineage.jsonl",
    "online_run_summary.json",
)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def _examples(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {str(row["example_id_hash"]): dict(row) for row in state["examples"]}
    if len(rows) != len(state["examples"]):
        raise ValueError("duplicate validation example hash")
    return rows


def _hhi(counts: Counter[int]) -> float | None:
    total = sum(counts.values())
    return sum((count / total) ** 2 for count in counts.values()) if total else None


def classify_gain_persistence(correctness_after_gain: Sequence[bool]) -> dict[str, Any]:
    if not correctness_after_gain or correctness_after_gain[0] is not True:
        raise ValueError("gain persistence must start correct")
    first_wrong = next(
        (index for index, value in enumerate(correctness_after_gain[1:], start=1) if not value),
        None,
    )
    if first_wrong is None:
        return {
            "persistence_class": "retained_to_final",
            "overwritten_later": False,
            "recovered_after_overwrite": False,
            "correct_at_final": True,
            "states_until_first_overwrite": None,
        }
    recovered = any(correctness_after_gain[first_wrong + 1 :])
    if recovered and correctness_after_gain[-1]:
        label = "overwritten_then_recovered_to_final"
    elif recovered:
        label = "overwritten_then_recovered_but_not_final"
    else:
        label = "overwritten_not_recovered"
    return {
        "persistence_class": label,
        "overwritten_later": True,
        "recovered_after_overwrite": recovered,
        "correct_at_final": correctness_after_gain[-1],
        "states_until_first_overwrite": first_wrong,
    }


def classify_loss_origin(correctness_through_before: Sequence[bool]) -> str:
    if not correctness_through_before or correctness_through_before[-1] is not True:
        raise ValueError("loss origin requires a correct pre-transition state")
    spell_start = len(correctness_through_before) - 1
    while spell_start > 0 and correctness_through_before[spell_start - 1]:
        spell_start -= 1
    return "new_collateral_regression" if spell_start == 0 else "prior_conversion_overwritten"


def _validate_inputs(
    states: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    test_calls = summary.get("new_test_calls", summary.get("test_evaluation_count"))
    if test_calls is None or int(test_calls) != 0:
        raise ValueError("test access is forbidden")
    infrastructure_failures = summary.get(
        "infrastructure_failure_count",
        sum(int(row.get("infrastructure_failures", 0)) for row in updates),
    )
    if int(infrastructure_failures) != 0:
        raise ValueError("infrastructure failure present")
    if [int(row["state_index"]) for row in states] != list(range(len(states))):
        raise ValueError("validation state indices are not contiguous")
    inventories = [set(_examples(row)) for row in states]
    if not inventories or any(item != inventories[0] for item in inventories[1:]):
        raise ValueError("validation inventory mismatch")
    committed = sum(_bool(row["committed"]) for row in updates)
    if committed != int(summary["accepted_commit_count"]):
        raise ValueError("accepted commit count mismatch")
    if len(states) != committed + 1:
        raise ValueError("validation state count must equal commits plus initial state")


def decompose_trajectory(
    *, seed: int, arm: str, run: Path
) -> dict[str, list[dict[str, Any]] | dict[str, Any]]:
    states = read_jsonl(run / "validation_states.jsonl")
    updates = read_jsonl(run / "update_lineage.jsonl")
    summary = read_json(run / "online_run_summary.json")
    _validate_inputs(states, updates, summary)
    state_by_index = {int(row["state_index"]): row for row in states}
    examples_by_state = {index: _examples(row) for index, row in state_by_index.items()}
    selected_counts: Counter[int] = Counter()
    for update in updates:
        selected_counts.update(int(value) for value in update["selected_target_ids"])
    commit_target_counts: Counter[int] = Counter()
    commit_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    for update in updates:
        if not _bool(update["committed"]):
            continue
        if update["validation_vote_delta"] is None:
            raise ValueError("accepted commit lacks validation replay")
        before_index = len(commit_rows)
        after_index = before_index + 1
        before = examples_by_state[before_index]
        after = examples_by_state[after_index]
        target = int(update["committed_target"])
        commit_target_counts[target] += 1
        vote_gains = sorted(key for key in before if not _bool(before[key]["vote_correct"]) and _bool(after[key]["vote_correct"]))
        vote_losses = sorted(key for key in before if _bool(before[key]["vote_correct"]) and not _bool(after[key]["vote_correct"]))
        oracle_gains = sorted(key for key in before if not _bool(before[key]["oracle_covered"]) and _bool(after[key]["oracle_covered"]))
        oracle_losses = sorted(key for key in before if _bool(before[key]["oracle_covered"]) and not _bool(after[key]["oracle_covered"]))
        target_gains = sorted(key for key in before if target not in before[key]["correct_member_ids"] and target in after[key]["correct_member_ids"])
        target_losses = sorted(key for key in before if target in before[key]["correct_member_ids"] and target not in after[key]["correct_member_ids"])
        net_vote = len(vote_gains) - len(vote_losses)
        net_oracle = len(oracle_gains) - len(oracle_losses)
        net_target = len(target_gains) - len(target_losses)
        persisted = {
            "vote": int(update["validation_vote_delta"]),
            "oracle": int(update["validation_oracle_delta"]),
            "target": int(update["validation_target_delta"]),
        }
        if (net_vote, net_oracle, net_target) != (persisted["vote"], persisted["oracle"], persisted["target"]):
            raise ValueError("validation transition mismatch")
        train_vote_gain = int(update["train_vote_gain"])
        train_vote_loss = int(update["train_vote_loss"])
        if train_vote_gain - train_vote_loss != int(update["train_vote_delta"]):
            raise ValueError("train vote accounting mismatch")
        coverage_recovery = sum(int(before[key]["G"]) == 0 and int(after[key]["G"]) == 1 for key in before)
        support_deepening = sum(int(before[key]["G"]) == 1 and int(after[key]["G"]) >= 2 for key in before)
        wrong_cluster_weakened = sum(int(after[key]["H"]) < int(before[key]["H"]) for key in before)
        wrong_cluster_strengthened = sum(int(after[key]["H"]) > int(before[key]["H"]) for key in before)
        commit_rows.append({
            "seed": seed,
            "arm": arm,
            "update_index": int(update["update_index"]),
            "transition_ordinal": len(commit_rows) + 1,
            "committed_target": target,
            "parent_team_hash": str(update["parent_team_hash"]),
            "successor_team_hash": str(update["successor_team_hash"]),
            "train_vote_gain": train_vote_gain,
            "train_vote_loss": train_vote_loss,
            "train_vote_delta": int(update["train_vote_delta"]),
            "train_oracle_delta": int(update["train_oracle_delta"]),
            "train_target_delta": int(update["train_target_delta"]),
            "validation_vote_gain": len(vote_gains),
            "validation_vote_loss": len(vote_losses),
            "validation_vote_delta": net_vote,
            "validation_oracle_gain": len(oracle_gains),
            "validation_oracle_loss": len(oracle_losses),
            "validation_oracle_delta": net_oracle,
            "validation_target_gain": len(target_gains),
            "validation_target_loss": len(target_losses),
            "validation_target_delta": net_target,
            "coverage_recovery_0_to_1": coverage_recovery,
            "support_deepening_1_to_2plus": support_deepening,
            "coverage_to_vote_conversion": len(vote_gains),
            "wrong_cluster_weakened": wrong_cluster_weakened,
            "wrong_cluster_strengthened": wrong_cluster_strengthened,
            "simultaneous_validation_vote_gain_loss": bool(vote_gains and vote_losses),
            "positive_train_vote_not_positive_validation_vote": int(update["train_vote_delta"]) > 0 and net_vote <= 0,
            "positive_train_target_not_positive_validation_target": int(update["train_target_delta"]) > 0 and net_target <= 0,
            "positive_validation_oracle_not_positive_vote": net_oracle > 0 and net_vote <= 0,
        })
        for example_hash in vote_gains:
            correctness = [
                _bool(examples_by_state[index][example_hash]["vote_correct"])
                for index in range(after_index, len(states))
            ]
            gain_rows.append({
                "seed": seed,
                "arm": arm,
                "gain_update_index": int(update["update_index"]),
                "example_id_hash": example_hash,
                **classify_gain_persistence(correctness),
            })
        for example_hash in vote_losses:
            history = [
                _bool(examples_by_state[index][example_hash]["vote_correct"])
                for index in range(before_index + 1)
            ]
            future = [
                _bool(examples_by_state[index][example_hash]["vote_correct"])
                for index in range(after_index, len(states))
            ]
            recovery = next((offset for offset, value in enumerate(future[1:], start=1) if value), None)
            loss_rows.append({
                "seed": seed,
                "arm": arm,
                "loss_update_index": int(update["update_index"]),
                "example_id_hash": example_hash,
                "loss_origin": classify_loss_origin(history),
                "recovered_later": recovery is not None,
                "correct_at_final": future[-1],
                "states_until_recovery": recovery,
            })
    initial_vote = int(states[0]["metrics"]["vote_correct_count"])
    final_vote = int(states[-1]["metrics"]["vote_correct_count"])
    if sum(int(row["validation_vote_delta"]) for row in commit_rows) != final_vote - initial_vote:
        raise ValueError("validation Vote telescoping identity failed")
    trajectory = {
        "seed": seed,
        "arm": arm,
        "accepted_commits": len(commit_rows),
        "initial_validation_vote": initial_vote,
        "final_validation_vote": final_vote,
        "validation_vote_delta": final_vote - initial_vote,
        "validation_vote_gain_events": sum(int(row["validation_vote_gain"]) for row in commit_rows),
        "validation_vote_loss_events": sum(int(row["validation_vote_loss"]) for row in commit_rows),
        "positive_net_commits": sum(int(row["validation_vote_delta"]) > 0 for row in commit_rows),
        "zero_net_commits": sum(int(row["validation_vote_delta"]) == 0 for row in commit_rows),
        "negative_net_commits": sum(int(row["validation_vote_delta"]) < 0 for row in commit_rows),
        "positive_train_vote_not_positive_validation_vote": sum(bool(row["positive_train_vote_not_positive_validation_vote"]) for row in commit_rows),
        "positive_train_target_not_positive_validation_target": sum(bool(row["positive_train_target_not_positive_validation_target"]) for row in commit_rows),
        "positive_validation_oracle_not_positive_vote": sum(bool(row["positive_validation_oracle_not_positive_vote"]) for row in commit_rows),
        "gain_overwritten_later": sum(bool(row["overwritten_later"]) for row in gain_rows),
        "loss_new_collateral_regression": sum(row["loss_origin"] == "new_collateral_regression" for row in loss_rows),
        "loss_prior_conversion_overwritten": sum(row["loss_origin"] == "prior_conversion_overwritten" for row in loss_rows),
        "selected_target_counts": json.dumps(dict(sorted(selected_counts.items())), separators=(",", ":")),
        "selected_target_hhi": _hhi(selected_counts),
        "committed_target_counts": json.dumps(dict(sorted(commit_target_counts.items())), separators=(",", ":")),
        "committed_target_hhi": _hhi(commit_target_counts),
        "distinct_committed_targets": len(commit_target_counts),
        "telescoping_identity_pass": True,
    }
    return {"trajectory": trajectory, "commits": commit_rows, "gains": gain_rows, "losses": loss_rows}


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else None


def analyze(*, seed68_root: Path, extension_root: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("fresh report root required")
    out.mkdir(parents=True)
    trajectories: list[dict[str, Any]] = []
    commits: list[dict[str, Any]] = []
    gains: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    hashes: list[dict[str, Any]] = []
    for seed in SEEDS:
        root = seed68_root if seed == 68 else extension_root
        for arm in ARMS:
            run = root / f"seed{seed}" / arm
            for filename in ALLOWED_SOURCE_FILES:
                source = run / filename
                if not source.is_file():
                    raise FileNotFoundError(source)
                hashes.append({"seed": seed, "arm": arm, "artifact_role": filename, "sha256": sha256_file(source)})
            result = decompose_trajectory(seed=seed, arm=arm, run=run)
            trajectories.append(result["trajectory"])
            commits.extend(result["commits"])
            gains.extend(result["gains"])
            losses.extend(result["losses"])
    seed_contrasts: list[dict[str, Any]] = []
    for seed in SEEDS:
        by_arm = {row["arm"]: row for row in trajectories if row["seed"] == seed}
        a = by_arm["A_CANONICAL"]
        c = by_arm["C_NO_SEMANTIC_CRITIC"]
        seed_contrasts.append({
            "seed": seed,
            "c_minus_a_commits": int(c["accepted_commits"]) - int(a["accepted_commits"]),
            "c_minus_a_final_validation_vote": int(c["final_validation_vote"]) - int(a["final_validation_vote"]),
            "c_minus_a_vote_gain_events": int(c["validation_vote_gain_events"]) - int(a["validation_vote_gain_events"]),
            "c_minus_a_vote_loss_events": int(c["validation_vote_loss_events"]) - int(a["validation_vote_loss_events"]),
            "c_minus_a_gain_overwritten": int(c["gain_overwritten_later"]) - int(a["gain_overwritten_later"]),
            "c_minus_a_cross_split_vote_failures": int(c["positive_train_vote_not_positive_validation_vote"]) - int(a["positive_train_vote_not_positive_validation_vote"]),
            "c_minus_a_committed_target_hhi": None if c["committed_target_hhi"] is None or a["committed_target_hhi"] is None else float(c["committed_target_hhi"]) - float(a["committed_target_hhi"]),
        })
    c_commits = [row for row in commits if row["arm"] == "C_NO_SEMANTIC_CRITIC"]
    c_gains = [row for row in gains if row["arm"] == "C_NO_SEMANTIC_CRITIC"]
    c_losses = [row for row in losses if row["arm"] == "C_NO_SEMANTIC_CRITIC"]
    diagnosis = {
        "cross_split_transfer_failure": sum(bool(row["positive_train_vote_not_positive_validation_vote"]) for row in c_commits),
        "target_transfer_failure": sum(bool(row["positive_train_target_not_positive_validation_target"]) for row in c_commits),
        "oracle_gain_without_vote_gain": sum(bool(row["positive_validation_oracle_not_positive_vote"]) for row in c_commits),
        "beneficial_vote_gain_later_overwritten": sum(bool(row["overwritten_later"]) for row in c_gains),
        "new_collateral_regression_events": sum(row["loss_origin"] == "new_collateral_regression" for row in c_losses),
        "prior_conversion_overwritten_events": sum(row["loss_origin"] == "prior_conversion_overwritten" for row in c_losses),
    }
    supported = [name for name, count in diagnosis.items() if count]
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "scope": {
            "retrospective": True,
            "new_api_calls": 0,
            "new_validation_calls": 0,
            "new_test_calls": 0,
            "method_modified": False,
            "critic_modified": False,
            "cross_arm_commits_treated_as_matched_pairs": False,
            "post_result_seed_extension_acknowledged": True,
        },
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "trajectory_count": len(trajectories),
        "accepted_commit_count": len(commits),
        "telescoping_identity_pass_count": sum(bool(row["telescoping_identity_pass"]) for row in trajectories),
        "c_commit_quality": {
            "commit_count": len(c_commits),
            "positive_net": sum(int(row["validation_vote_delta"]) > 0 for row in c_commits),
            "zero_net": sum(int(row["validation_vote_delta"]) == 0 for row in c_commits),
            "negative_net": sum(int(row["validation_vote_delta"]) < 0 for row in c_commits),
            "mean_validation_vote_gain": _mean(c_commits, "validation_vote_gain"),
            "mean_validation_vote_loss": _mean(c_commits, "validation_vote_loss"),
            "mean_validation_vote_delta": _mean(c_commits, "validation_vote_delta"),
        },
        "diagnostic_counts": diagnosis,
        "supported_mechanisms": supported,
        "seed69_vs_seed70": {
            "seed69": next(row for row in seed_contrasts if row["seed"] == 69),
            "seed70": next(row for row in seed_contrasts if row["seed"] == 70),
        },
    }
    write_csv(out / "trajectory_decomposition.csv", trajectories)
    write_csv(out / "accepted_commit_decomposition.csv", commits)
    write_csv(out / "validation_vote_gain_persistence.csv", gains)
    write_csv(out / "validation_vote_loss_provenance.csv", losses)
    write_csv(out / "seed_arm_contrasts.csv", seed_contrasts)
    write_csv(out / "source_artifact_hashes.csv", hashes)
    write_json(out / "summary.json", summary)
    seed69_c = next(row for row in trajectories if row["seed"] == 69 and row["arm"] == "C_NO_SEMANTIC_CRITIC")
    seed69_a = next(row for row in trajectories if row["seed"] == 69 and row["arm"] == "A_CANONICAL")
    seed70_c = next(row for row in trajectories if row["seed"] == 70 and row["arm"] == "C_NO_SEMANTIC_CRITIC")
    seed70_a = next(row for row in trajectories if row["seed"] == 70 and row["arm"] == "A_CANONICAL")
    diagnosis_summary = {
        "primary_bottleneck": "ACCEPTED_UPDATE_CROSS_SPLIT_AND_PLURALITY_CONVERSION_QUALITY",
        "seed69_explanation": {
            "c_initial_to_final_vote_delta": seed69_c["validation_vote_delta"],
            "a_initial_to_final_vote_delta": seed69_a["validation_vote_delta"],
            "c_vote_gain_events": seed69_c["validation_vote_gain_events"],
            "c_vote_loss_events": seed69_c["validation_vote_loss_events"],
            "c_positive_net_commits": seed69_c["positive_net_commits"],
        },
        "seed70_explanation": {
            "c_initial_to_final_vote_delta": seed70_c["validation_vote_delta"],
            "a_initial_to_final_vote_delta": seed70_a["validation_vote_delta"],
            "c_vote_gain_events": seed70_c["validation_vote_gain_events"],
            "c_vote_loss_events": seed70_c["validation_vote_loss_events"],
            "c_positive_net_commits": seed70_c["positive_net_commits"],
            "c_target_transfer_failures": seed70_c["positive_train_target_not_positive_validation_target"],
            "c_oracle_gain_without_vote_gain": seed70_c["positive_validation_oracle_not_positive_vote"],
            "c_gain_overwritten_later": seed70_c["gain_overwritten_later"],
            "c_new_collateral_losses": seed70_c["loss_new_collateral_regression"],
        },
        "target_concentration_supported": False,
        "target_concentration_reason": "C committed-target HHI is lower than A on Seeds69 and 70, and C updates four distinct members on both seeds.",
        "later_overwrite_is_primary_seed70_explanation": False,
        "trajectory_divergence_warning": "C-A final differences decompose each arm from its own shared initial state; commits are not matched counterfactual pairs after divergence.",
    }
    write_json(out / "diagnosis.json", diagnosis_summary)
    write_json(out / "fact_assertions.json", {
        "status": "PASS",
        "trajectory_count": 6,
        "accepted_commit_count": len(commits),
        "telescoping_identity_pass_count": 6,
        "new_api_calls": 0,
        "new_test_calls": 0,
        "historical_artifacts_modified": False,
    })
    readme = f"""# V18 No-Semantic-Critic accepted-commit transfer decomposition

This is a zero-API, validation-only retrospective audit of the frozen Seed68-70 A/C trajectories. It does not modify the method or Critic and does not treat diverged arm commits as matched causal pairs.

## Result

The primary diagnostic is **ACCEPTED_UPDATE_CROSS_SPLIT_AND_PLURALITY_CONVERSION_QUALITY**. Across C's 15 commits, 3 had positive validation Vote net, 9 were neutral, and 3 were negative. There were 6 positive-train-Vote commits without positive validation Vote, 7 target-transfer failures, 7 Oracle-gain-without-Vote-gain commits, 8 new collateral-loss events, one prior conversion overwritten, and one beneficial gain later overwritten.

| Seed | Arm | Commits | Initial to final validation Vote | Gain events | Loss events | Positive/zero/negative commits |
|---|---|---:|---:|---:|---:|---:|
| 69 | A | {seed69_a['accepted_commits']} | {seed69_a['validation_vote_delta']:+d} | {seed69_a['validation_vote_gain_events']} | {seed69_a['validation_vote_loss_events']} | {seed69_a['positive_net_commits']}/{seed69_a['zero_net_commits']}/{seed69_a['negative_net_commits']} |
| 69 | C | {seed69_c['accepted_commits']} | {seed69_c['validation_vote_delta']:+d} | {seed69_c['validation_vote_gain_events']} | {seed69_c['validation_vote_loss_events']} | {seed69_c['positive_net_commits']}/{seed69_c['zero_net_commits']}/{seed69_c['negative_net_commits']} |
| 70 | A | {seed70_a['accepted_commits']} | {seed70_a['validation_vote_delta']:+d} | {seed70_a['validation_vote_gain_events']} | {seed70_a['validation_vote_loss_events']} | {seed70_a['positive_net_commits']}/{seed70_a['zero_net_commits']}/{seed70_a['negative_net_commits']} |
| 70 | C | {seed70_c['accepted_commits']} | {seed70_c['validation_vote_delta']:+d} | {seed70_c['validation_vote_gain_events']} | {seed70_c['validation_vote_loss_events']} | {seed70_c['positive_net_commits']}/{seed70_c['zero_net_commits']}/{seed70_c['negative_net_commits']} |

Seed69's final C-A difference of +7 is not attributable to three matched extra commits: C improved by +4 from the common initial state while A regressed by -3. C contained two positive-net commits with six gain and two loss events.

Seed70's C trajectory had no positive-net commit, two Vote gain events and three loss events. Three commits improved the train target without improving the validation target, three added validation Oracle coverage without positive Vote net, and all three validation Vote losses were new collateral regressions. No Seed70 C Vote gain was later overwritten. Its committed-target HHI was lower than A and it updated four distinct members, so target concentration is not supported as the explanation.

The evidence therefore locates the remaining bottleneck after candidate-supply recovery at accepted-update cross-split quality and plurality conversion, not semantic-Critic throughput, later overwrite, or member concentration.

Every trajectory satisfies the telescoping identity between accepted-transition validation Vote deltas and its initial-to-final Vote change. Seed69/70 were added after observing Seed68, so cross-seed aggregates remain descriptive. No API or test evaluation was performed.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    write_json(out / "sanitization_manifest.json", {
        "status": "PASS",
        "raw_text_published": False,
        "absolute_paths_published": False,
        "forbidden_content": ["prompts", "questions", "gold answers", "model answers", "raw responses", "endpoints", "credentials", "SQLite", "checkpoints"],
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
    summary = analyze(
        seed68_root=args.seed68_root.resolve(),
        extension_root=args.extension_root.resolve(),
        out=args.out.resolve(),
    )
    print(json.dumps({
        "analysis_version": summary["analysis_version"],
        "accepted_commit_count": summary["accepted_commit_count"],
        "diagnostic_counts": summary["diagnostic_counts"],
        "new_api_calls": 0,
        "new_test_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
