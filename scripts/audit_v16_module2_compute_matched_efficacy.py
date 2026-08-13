from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ARMS = {
    "G-Matched": "experimental_v16_efficacy_g_matched",
    "R-M20": "experimental_v16_efficacy_r_m20",
    "R-M2F": "experimental_v16_efficacy_r_m2f",
}
SEEDS = (53, 54, 55)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit(root: Path, freeze: Path) -> dict[str, Any]:
    frozen = read_json(freeze)
    blockers: list[str] = []
    rows: list[dict[str, Any]] = []
    for arm, setting in ARMS.items():
        for seed in SEEDS:
            run = root / "disambiguation_qa" / f"{setting}_seed{seed}"
            required = (
                "run_meta.json", "final_summary.json", "cost_summary.json",
                "candidate_decisions.jsonl", "training_dynamics.jsonl",
                "frozen_initialization_match.json",
            )
            if any(not (run / name).is_file() for name in required):
                blockers.append(f"missing_run:{arm}:{seed}")
                continue
            meta = read_json(run / "run_meta.json")
            final = read_json(run / "final_summary.json")
            cost = read_json(run / "cost_summary.json")
            decisions = read_jsonl(run / "candidate_decisions.jsonl")
            dynamics = read_jsonl(run / "training_dynamics.jsonl")
            init = read_json(run / "frozen_initialization_match.json")
            identity = meta.get("run_identity", {})
            selection = final.get("selection_summary", {})
            if identity.get("git_commit") != frozen.get("git_head"):
                blockers.append(f"source_identity:{arm}:{seed}")
            if identity.get("experiment_setting") != setting:
                blockers.append(f"setting_identity:{arm}:{seed}")
            if not init.get("matched"):
                blockers.append(f"initialization:{arm}:{seed}")
            if int(meta.get("completed_update_count", -1)) != 8 or len(decisions) != 8:
                blockers.append(f"update_count:{arm}:{seed}")
            if (
                bool(selection.get("validation_used"))
                or int(selection.get("validation_evaluation_count", -1)) != 0
                or int(selection.get("test_evaluation_count", -1)) != 0
                or bool(meta.get("final_test_enabled"))
            ):
                blockers.append(f"train_only:{arm}:{seed}")
            if (
                int(cost.get("provider_call_budget", -1)) != 8000
                or int(cost.get("total_token_budget", -1)) != 3000000
                or bool(cost.get("provider_call_budget_exhausted"))
                or bool(cost.get("total_token_budget_exhausted"))
            ):
                blockers.append(f"budget:{arm}:{seed}")
            generic = read_jsonl(run / "loss_blind_generic_revision_events.jsonl")
            repair = read_jsonl(run / "online_compatibility_repair_events.jsonl")
            if arm == "G-Matched":
                if repair or any(
                    row.get("responsibility_evidence_exposed")
                    or row.get("candidate_specific_loss_evidence_exposed")
                    for row in generic
                ):
                    blockers.append(f"generic_leakage:{seed}")
            elif generic:
                blockers.append(f"unexpected_generic_revision:{arm}:{seed}")
            if arm == "R-M20" and repair:
                blockers.append(f"unexpected_m2f:{seed}")
            initial = dynamics[0]
            ending = dynamics[-1]
            initial_counts = [int(x) for x in initial["per_agent_correct_counts"]]
            final_counts = [int(x) for x in ending["per_agent_correct_counts"]]
            gains = [right - left for left, right in zip(initial_counts, final_counts, strict=True)]
            rows.append({
                "arm": arm,
                "setting": setting,
                "seed": seed,
                "final_train_vote_accuracy": float(ending["team_vote_accuracy"]),
                "final_train_vote_correct_count": int(ending["team_vote_correct_count"]),
                "g_min": min(gains),
                "g_sum": sum(gains),
                "accepted_updates": int(ending["accepted_update_count_so_far"]),
                "provider_calls": int(cost["successful_llm_calls"]),
                "prompt_tokens": int(cost["prompt_tokens"]),
                "completion_tokens": int(cost["completion_tokens"]),
                "total_tokens": int(cost["total_tokens"]),
                "generic_revision_attempted": sum(bool(x.get("revision_attempted")) for x in generic),
                "generic_revision_feasible": sum(bool(x.get("revision_feasible")) for x in generic),
                "generic_revision_committed": sum(bool(x.get("revision_committed")) for x in generic),
                "repair_eligible": sum(bool(x.get("repair_eligible")) for x in repair),
                "repair_attempted": sum(bool(x.get("repair_attempted")) for x in repair),
                "repair_feasible": sum(bool(x.get("repair_feasible")) for x in repair),
                "repair_committed": sum(bool(x.get("repair_committed")) for x in repair),
            })
    if len(rows) != 9:
        blockers.append("run_inventory")
    by_key = {(row["arm"], row["seed"]): row for row in rows}
    contrasts = []
    for seed in SEEDS:
        for left, right in (("G-Matched", "R-M20"), ("R-M20", "R-M2F"), ("G-Matched", "R-M2F")):
            if (left, seed) not in by_key or (right, seed) not in by_key:
                continue
            lrow, rrow = by_key[(left, seed)], by_key[(right, seed)]
            contrasts.append({
                "seed": seed, "left": left, "right": right,
                "vote_accuracy_delta_right_minus_left": rrow["final_train_vote_accuracy"] - lrow["final_train_vote_accuracy"],
                "g_min_delta_right_minus_left": rrow["g_min"] - lrow["g_min"],
                "g_sum_delta_right_minus_left": rrow["g_sum"] - lrow["g_sum"],
            })
    return {
        "gate": "PASS" if not blockers else "FAIL",
        "blockers": sorted(set(blockers)),
        "execution_commit": frozen.get("git_head", ""),
        "experiment_version": "v16_module2_compute_matched_efficacy_v1",
        "rows": rows,
        "paired_contrasts": contrasts,
        "validation_evaluations": 0 if not blockers else None,
        "test_evaluations": 0 if not blockers else None,
        "formal_efficacy_classifier": "DESCRIPTIVE_THREE_SEED_COMPARISON",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh audit output required")
    value = audit(args.run_root, args.freeze)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value[key] for key in ("gate", "blockers", "execution_commit")}, indent=2))
    raise SystemExit(0 if value["gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
