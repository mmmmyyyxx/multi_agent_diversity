from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARMS = ("W1_TOP2", "HYBRID_BASE")
SEEDS = (59, 60, 61)
ALLOWED_INPUT_FILES = (
    "validation_states.jsonl",
    "update_lineage.jsonl",
    "online_run_summary.json",
)
GAIN_PERSISTENCE_CLASSES = (
    "retained_to_final",
    "overwritten_then_recovered_to_final",
    "overwritten_then_recovered_but_not_final",
    "overwritten_not_recovered",
)
LOSS_ORIGIN_CLASSES = (
    "new_collateral_regression",
    "prior_conversion_overwritten",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
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
        for row in rows:
            writer.writerow({key: "NA" if value is None else value for key, value in row.items()})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _state_examples(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {str(row["example_id_hash"]): dict(row) for row in state["examples"]}
    if len(rows) != len(state["examples"]):
        raise ValueError("duplicate validation example hash")
    return rows


def validate_trajectory_inputs(
    states: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    if not states:
        raise ValueError("validation state sequence is empty")
    indices = [int(row["state_index"]) for row in states]
    if indices != list(range(len(states))):
        raise ValueError("validation state indices are not contiguous")
    hashes = [set(_state_examples(row)) for row in states]
    if any(item != hashes[0] for item in hashes[1:]):
        raise ValueError("validation example inventory changed across states")
    if int(summary["new_test_calls"]) != 0:
        raise ValueError("test calls are forbidden")
    if int(summary["infrastructure_failure_count"]) != 0:
        raise ValueError("infrastructure failure present")
    if int(summary["accepted_commit_count"]) != sum(_bool(row["committed"]) for row in updates):
        raise ValueError("accepted commit count mismatch")


def classify_gain_persistence(correctness_after_gain: list[bool]) -> dict[str, Any]:
    if not correctness_after_gain or correctness_after_gain[0] is not True:
        raise ValueError("gain persistence must begin from a correct state")
    later = correctness_after_gain[1:]
    first_wrong_offset = next((index for index, value in enumerate(later, start=1) if not value), None)
    if first_wrong_offset is None:
        label = "retained_to_final"
        recovered = False
    else:
        recovered = any(correctness_after_gain[first_wrong_offset + 1 :])
        if recovered and correctness_after_gain[-1]:
            label = "overwritten_then_recovered_to_final"
        elif recovered:
            label = "overwritten_then_recovered_but_not_final"
        else:
            label = "overwritten_not_recovered"
    return {
        "persistence_class": label,
        "overwritten_later": first_wrong_offset is not None,
        "recovered_after_overwrite": recovered,
        "correct_at_final": correctness_after_gain[-1],
        "states_until_first_overwrite": first_wrong_offset,
    }


def loss_origin(correctness_through_before: list[bool]) -> str:
    if not correctness_through_before or correctness_through_before[-1] is not True:
        raise ValueError("loss origin requires a correct pre-transition state")
    spell_start = len(correctness_through_before) - 1
    while spell_start > 0 and correctness_through_before[spell_start - 1]:
        spell_start -= 1
    return "new_collateral_regression" if spell_start == 0 else "prior_conversion_overwritten"


def decompose_trajectory(
    *, seed: int, arm: str, states: list[dict[str, Any]],
    updates: list[dict[str, Any]], summary: dict[str, Any],
) -> dict[str, Any]:
    validate_trajectory_inputs(states, updates, summary)
    state_by_index = {int(row["state_index"]): row for row in states}
    examples_by_state = {index: _state_examples(row) for index, row in state_by_index.items()}
    accepted = [row for row in updates if _bool(row["committed"])]
    commit_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    for update in accepted:
        if not _bool(update["validation_evaluated"]):
            raise ValueError("accepted commit lacks validation evaluation")
        before_index = int(update["validation_state_index_before"])
        after_index = int(update["validation_state_index_after"])
        if after_index != before_index + 1:
            raise ValueError("accepted validation transition is not adjacent")
        before = examples_by_state[before_index]
        after = examples_by_state[after_index]
        gains = sorted(key for key in before if not _bool(before[key]["vote_correct"]) and _bool(after[key]["vote_correct"]))
        losses = sorted(key for key in before if _bool(before[key]["vote_correct"]) and not _bool(after[key]["vote_correct"]))
        net = len(gains) - len(losses)
        metric_delta = int(state_by_index[after_index]["metrics"]["vote_correct_count"]) - int(
            state_by_index[before_index]["metrics"]["vote_correct_count"]
        )
        persisted_delta = int(update["validation_vote_delta"])
        if net != metric_delta or net != persisted_delta:
            raise ValueError("accepted transition vote delta mismatch")
        train_vote_delta = int(update["train_vote_delta"])
        transfer_class = "positive" if net > 0 else "negative" if net < 0 else "neutral"
        commit_rows.append({
            "seed": seed,
            "arm": arm,
            "update_index": int(update["update_index"]),
            "transition_ordinal": len(commit_rows) + 1,
            "parent_team_hash": str(update["parent_team_hash"]),
            "successor_team_hash": str(update["successor_team_hash"]),
            "committed_target": int(update["committed_target"]),
            "validation_state_before": before_index,
            "validation_state_after": after_index,
            "train_vote_delta": train_vote_delta,
            "train_target_delta": int(update["train_target_delta"]),
            "validation_gain_count": len(gains),
            "validation_loss_count": len(losses),
            "validation_net_delta": net,
            "validation_transfer_class": transfer_class,
            "train_vote_progress_not_transferred": train_vote_delta > 0 and net <= 0,
            "simultaneous_gain_and_loss": bool(gains and losses),
        })
        for example_hash in gains:
            correctness = [
                _bool(examples_by_state[index][example_hash]["vote_correct"])
                for index in range(after_index, len(states))
            ]
            persistence = classify_gain_persistence(correctness)
            first_overwrite_state = None
            if persistence["states_until_first_overwrite"] is not None:
                first_overwrite_state = after_index + int(persistence["states_until_first_overwrite"])
            gain_rows.append({
                "seed": seed,
                "arm": arm,
                "gain_update_index": int(update["update_index"]),
                "gain_validation_state": after_index,
                "example_id_hash": example_hash,
                **persistence,
                "first_overwrite_validation_state": first_overwrite_state,
            })
        for example_hash in losses:
            history = [
                _bool(examples_by_state[index][example_hash]["vote_correct"])
                for index in range(before_index + 1)
            ]
            later = [
                _bool(examples_by_state[index][example_hash]["vote_correct"])
                for index in range(after_index, len(states))
            ]
            origin = loss_origin(history)
            later_recovery_offset = next((idx for idx, value in enumerate(later[1:], start=1) if value), None)
            loss_rows.append({
                "seed": seed,
                "arm": arm,
                "loss_update_index": int(update["update_index"]),
                "loss_validation_state": after_index,
                "example_id_hash": example_hash,
                "loss_origin": origin,
                "recovered_later": later_recovery_offset is not None,
                "correct_at_final": later[-1],
                "states_until_recovery": later_recovery_offset,
            })
    initial_vote = int(states[0]["metrics"]["vote_correct_count"])
    final_vote = int(states[-1]["metrics"]["vote_correct_count"])
    transition_net_sum = sum(int(row["validation_net_delta"]) for row in commit_rows)
    if transition_net_sum != final_vote - initial_vote:
        raise ValueError("validation Vote telescoping identity failed")
    if len(commit_rows) != int(summary["accepted_commit_count"]):
        raise ValueError("accepted transition reconstruction incomplete")
    return {
        "commit_rows": commit_rows,
        "gain_rows": gain_rows,
        "loss_rows": loss_rows,
        "trajectory_row": {
            "seed": seed,
            "arm": arm,
            "accepted_commit_count": len(commit_rows),
            "initial_validation_vote_count": initial_vote,
            "final_validation_vote_count": final_vote,
            "initial_to_final_vote_delta": final_vote - initial_vote,
            "transition_net_sum": transition_net_sum,
            "telescoping_identity_pass": True,
            "validation_gain_count": len(gain_rows),
            "validation_loss_count": len(loss_rows),
            "gain_retained_to_final": sum(row["persistence_class"] == "retained_to_final" for row in gain_rows),
            "gain_overwritten_later": sum(bool(row["overwritten_later"]) for row in gain_rows),
            "gain_overwritten_then_recovered": sum(bool(row["recovered_after_overwrite"]) for row in gain_rows),
            "loss_new_collateral_regression": sum(row["loss_origin"] == "new_collateral_regression" for row in loss_rows),
            "loss_prior_conversion_overwritten": sum(row["loss_origin"] == "prior_conversion_overwritten" for row in loss_rows),
            "positive_net_commits": sum(int(row["validation_net_delta"]) > 0 for row in commit_rows),
            "zero_net_commits": sum(int(row["validation_net_delta"]) == 0 for row in commit_rows),
            "negative_net_commits": sum(int(row["validation_net_delta"]) < 0 for row in commit_rows),
            "simultaneous_gain_loss_commits": sum(bool(row["simultaneous_gain_and_loss"]) for row in commit_rows),
            "train_vote_progress_not_transferred_commits": sum(bool(row["train_vote_progress_not_transferred"]) for row in commit_rows),
        },
    }


def _quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    commits = len(rows)
    positive = sum(int(row["validation_net_delta"]) > 0 for row in rows)
    zero = sum(int(row["validation_net_delta"]) == 0 for row in rows)
    negative = sum(int(row["validation_net_delta"]) < 0 for row in rows)
    return {
        "commit_count": commits,
        "positive_net_count": positive,
        "zero_net_count": zero,
        "negative_net_count": negative,
        "positive_net_proportion": positive / commits if commits else None,
        "zero_net_proportion": zero / commits if commits else None,
        "negative_net_proportion": negative / commits if commits else None,
        "mean_gain_count_per_commit": mean(float(row["validation_gain_count"]) for row in rows),
        "mean_loss_count_per_commit": mean(float(row["validation_loss_count"]) for row in rows),
        "mean_net_delta_per_commit": mean(float(row["validation_net_delta"]) for row in rows),
        "simultaneous_gain_loss_count": sum(bool(row["simultaneous_gain_and_loss"]) for row in rows),
        "train_vote_progress_not_transferred_count": sum(bool(row["train_vote_progress_not_transferred"]) for row in rows),
    }


def classify_bottlenecks(
    commit_rows: list[dict[str, Any]], gain_rows: list[dict[str, Any]], loss_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_arm = {arm: [row for row in commit_rows if row["arm"] == arm] for arm in ARMS}
    quality = {arm: _quality(by_arm[arm]) for arm in ARMS}
    hybrid_losses = [row for row in loss_rows if row["arm"] == "HYBRID_BASE"]
    hybrid_gains = [row for row in gain_rows if row["arm"] == "HYBRID_BASE"]
    w1 = quality["W1_TOP2"]
    hybrid = quality["HYBRID_BASE"]
    flags = {
        "collateral_regression": bool(hybrid_losses) and (
            float(hybrid["mean_loss_count_per_commit"] or 0) > float(w1["mean_loss_count_per_commit"] or 0)
            or int(hybrid["negative_net_count"]) > int(w1["negative_net_count"])
        ),
        "transfer_failure": int(hybrid["train_vote_progress_not_transferred_count"]) > 0,
        "beneficial_conversion_later_overwritten": any(bool(row["overwritten_later"]) for row in hybrid_gains),
        "higher_throughput_lower_average_quality": (
            int(hybrid["commit_count"]) > int(w1["commit_count"])
            and float(hybrid["mean_net_delta_per_commit"] or 0) < float(w1["mean_net_delta_per_commit"] or 0)
        ),
    }
    supported = [name for name, value in flags.items() if value]
    return {
        "classifier_version": "v18_trajectory_gain_loss_classifier_v1",
        "rules_frozen_before_result_readout": True,
        "flags": flags,
        "supported_bottlenecks": supported,
        "final_diagnosis": "+".join(name.upper() for name in supported) if supported else "NO_CLEAR_TRAJECTORY_BOTTLENECK",
        "commit_quality": quality,
    }


def analyze(root: Path, admission: dict[str, Any], out: Path) -> dict[str, Any]:
    if admission.get("scientific_analysis_admitted") is not True:
        raise ValueError("V18 scientific analysis admission is required")
    if admission.get("post_hoc_corrected_gate_status") != "PASS":
        raise ValueError("post-hoc corrected gate must pass")
    if int(admission.get("new_test_calls", -1)) != 0:
        raise ValueError("source experiment contains test calls")
    if out.exists():
        raise FileExistsError("fresh output directory required")
    out.mkdir(parents=True)
    trajectory_rows: list[dict[str, Any]] = []
    commit_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    source_hashes: list[dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            run = root / f"seed{seed}" / arm
            for filename in ALLOWED_INPUT_FILES:
                path = run / filename
                if not path.is_file():
                    raise FileNotFoundError(path)
                source_hashes.append({
                    "seed": seed,
                    "arm": arm,
                    "artifact_role": filename,
                    "sha256": sha256_file(path),
                })
            result = decompose_trajectory(
                seed=seed,
                arm=arm,
                states=read_jsonl(run / "validation_states.jsonl"),
                updates=read_jsonl(run / "update_lineage.jsonl"),
                summary=read_json(run / "online_run_summary.json"),
            )
            trajectory_rows.append(result["trajectory_row"])
            commit_rows.extend(result["commit_rows"])
            gain_rows.extend(result["gain_rows"])
            loss_rows.extend(result["loss_rows"])
    classifier = classify_bottlenecks(commit_rows, gain_rows, loss_rows)
    aggregate = {}
    for arm in ARMS:
        trajectories = [row for row in trajectory_rows if row["arm"] == arm]
        gains = [row for row in gain_rows if row["arm"] == arm]
        losses = [row for row in loss_rows if row["arm"] == arm]
        aggregate[arm] = {
            **classifier["commit_quality"][arm],
            "initial_to_final_vote_delta_sum": sum(int(row["initial_to_final_vote_delta"]) for row in trajectories),
            "validation_gain_count": len(gains),
            "validation_loss_count": len(losses),
            "gain_retained_to_final_count": sum(row["persistence_class"] == "retained_to_final" for row in gains),
            "gain_overwritten_later_count": sum(bool(row["overwritten_later"]) for row in gains),
            "gain_overwritten_then_recovered_count": sum(bool(row["recovered_after_overwrite"]) for row in gains),
            "loss_new_collateral_regression_count": sum(row["loss_origin"] == "new_collateral_regression" for row in losses),
            "loss_prior_conversion_overwritten_count": sum(row["loss_origin"] == "prior_conversion_overwritten" for row in losses),
        }
    seed61 = {
        arm: next(row for row in trajectory_rows if row["seed"] == 61 and row["arm"] == arm)
        for arm in ARMS
    }
    summary = {
        "analysis_version": "v18_trajectory_gain_loss_decomposition_v1",
        "scope": {
            "validation_only": True,
            "new_api_calls": 0,
            "new_model_calls": 0,
            "new_test_calls": 0,
            "method_modified": False,
            "selector_modified": False,
            "cross_arm_commit_matching": False,
            "diverged_trajectories_treated_as_matched": False,
        },
        "source_gate": {
            "original_frozen_audit_status": admission["original_frozen_audit_status"],
            "post_hoc_corrected_gate_status": admission["post_hoc_corrected_gate_status"],
            "raw_artifact_identity": admission["raw_artifact_identity"],
        },
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "trajectory_count": len(trajectory_rows),
        "accepted_transition_count": len(commit_rows),
        "telescoping_identity_pass_count": sum(bool(row["telescoping_identity_pass"]) for row in trajectory_rows),
        "aggregate": aggregate,
        "seed61_focus": seed61,
        "classifier": classifier,
    }
    write_csv(out / "trajectory_decomposition.csv", trajectory_rows)
    write_csv(out / "accepted_commit_quality.csv", commit_rows)
    write_csv(out / "validation_gain_persistence.csv", gain_rows)
    write_csv(out / "validation_loss_provenance.csv", loss_rows)
    write_csv(out / "source_artifact_hashes.csv", source_hashes)
    write_json(out / "classifier.json", classifier)
    write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(
        args.root.resolve(),
        read_json(args.admission.resolve()),
        args.out.resolve(),
    )
    print(json.dumps({
        "analysis_version": summary["analysis_version"],
        "accepted_transition_count": summary["accepted_transition_count"],
        "telescoping_identity_pass_count": summary["telescoping_identity_pass_count"],
        "final_diagnosis": summary["classifier"]["final_diagnosis"],
        "new_api_calls": 0,
        "new_test_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
