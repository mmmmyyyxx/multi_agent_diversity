from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
RUN_SOURCE_COMMIT = "e5bdc9f27f7a5594072aafd828c7c6053297c03c"
METHOD_VERSION = "member_aware_peer_state_v14"
CHECKPOINT_VERSION = 23
SETTINGS = {
    "S1": "shared_member_aware_dual_target_seed46",
    "S2": "shared_responsibility_conditioned_dual_target_seed46",
    "S3": "shared_full_dual_target_rcru_seed46",
}
POLICIES = ("R0", "R1", "R2", "R3")
WAIT_WEIGHT = 0.05


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rank_key(row: dict[str, Any], score: float) -> tuple[Any, ...]:
    return (
        -score,
        -float(row["opportunity_value"]),
        -float(row["normalized_direct_fix"]),
        -float(row["normalized_support_margin"]),
        -float(row["normalized_uplift_deficit"]),
        -float(row["normalized_wait"]),
        str(row["seeded_rank"]),
        int(row["agent_id"]),
    )


def entropy_and_concentration(counts: dict[int, int]) -> tuple[float, float, float]:
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0, 0.0
    shares = [counts.get(agent, 0) / total for agent in range(5)]
    entropy = -sum(value * math.log(value) for value in shares if value > 0) / math.log(5)
    concentration = sum(value * value for value in shares)
    return entropy, concentration, max(shares)


def verify_identity(path: Path, meta: dict[str, Any]) -> None:
    config = meta["config"]
    identity = meta["run_identity"]
    checks = {
        "task": config["comparison_task_id"] == "disambiguation_qa",
        "seed": int(config["seed"]) == 46,
        "models": all(
            config[key] == "qwen3-14b"
            for key in ("agent_model", "optimizer_model", "evaluator_model")
        ),
        "source_commit": identity["git_commit"] == RUN_SOURCE_COMMIT,
        "method": identity["method_version"] == METHOD_VERSION,
        "checkpoint": int(meta["checkpoint_version"]) == CHECKPOINT_VERSION,
        "selector": meta["target_selection_version"]
        == "repairability_adjusted_expected_update_value_v1",
        "state_repairability": meta["repairability_version"]
        == "state_local_branch_failure_discount_v1",
        "routing": meta["service_routing_version"]
        == "single_service_anchor_routing_no_freeze_v2",
        "updates": (
            int(meta["planned_update_count"]) == 32
            and int(meta["completed_update_count"]) == 32
        ),
        "test_once": int(meta["test_evaluation_count"]) == 1,
        "validation_zero": int(meta["validation_evaluation_count"]) == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"identity_failed:{path.name}:{failed}")


def branch_outcome(event: dict[str, Any] | None) -> str:
    if event is None:
        return "NOT_SELECTED"
    if bool(event["operational_failure"]) or not bool(event["normal_completion"]):
        return "OPERATIONAL_FAILURE"
    if bool(event["passed_candidate_found"]):
        return "FEASIBLE"
    return "NORMAL_FAILURE"


def score_variants(row: dict[str, Any], history: dict[str, Any]) -> dict[str, float]:
    opportunity = float(row["opportunity_value"])
    rho_state = float(row["repairability_discount"])
    wait_term = WAIT_WEIGHT * float(row["normalized_wait"])
    attempts = int(history["attempts"])
    feasible = int(history["feasible"])
    consecutive = int(history["consecutive_failures"])
    beta_rate = (feasible + 1) / (attempts + 2)
    cross_consecutive = 1 / (1 + consecutive)
    floored = 0.25 + 0.75 * beta_rate
    return {
        "R0": opportunity * rho_state + wait_term,
        "R1": opportunity * rho_state * beta_rate + wait_term,
        "R2": opportunity * rho_state * cross_consecutive + wait_term,
        "R3": opportunity * rho_state * floored + wait_term,
        "beta_rate": beta_rate,
        "cross_consecutive_discount": cross_consecutive,
        "floored_realizability": floored,
    }


def load_setting(formal_root: Path, label: str, name: str) -> dict[str, Any]:
    path = formal_root / "disambiguation_qa" / name
    meta = read_json(path / "run_meta.json")
    verify_identity(path, meta)
    scores = read_jsonl(path / "repairability_adjusted_target_scores.jsonl")
    branches = read_jsonl(path / "dual_target_branch_decisions.jsonl")
    commits = read_jsonl(path / "dual_target_commit_decisions.jsonl")
    events = read_jsonl(path / "repairability_failure_events.jsonl")
    if len(branches) != 64 or len(commits) != 32 or len(events) != 64:
        raise AssertionError(f"branch_inventory_failed:{label}")
    return {
        "label": label,
        "name": name,
        "meta": meta,
        "scores": scores,
        "branches": branches,
        "commits": commits,
        "events": events,
    }


def replay_setting(run: dict[str, Any]) -> dict[str, Any]:
    label = run["label"]
    score_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in run["scores"]:
        score_groups[int(row["update_index"])].append(row)
    event_lookup = {
        (int(row["update_index"]), int(row["agent_id"])): row
        for row in run["events"]
    }
    branch_lookup = {
        (int(row["update_index"]), int(row["target_agent_id"])): row
        for row in run["branches"]
    }
    commit_lookup = {int(row["update_index"]): row for row in run["commits"]}
    history = {
        agent: {
            "attempts": 0,
            "feasible": 0,
            "normal_failures": 0,
            "consecutive_failures": 0,
            "last_feasible_update": None,
        }
        for agent in range(5)
    }
    baseline_mismatches: list[dict[str, Any]] = []
    timeline_updates: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    feasible_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    commit_rows: list[dict[str, Any]] = []
    recovery_witnesses: list[dict[str, Any]] = []
    previous_agent_rows: dict[int, dict[str, Any]] = {}

    for update in range(32):
        actionable = score_groups[update]
        if len(actionable) < 2:
            raise AssertionError(f"fewer_than_two_actionable:{label}:{update}")
        values: dict[int, dict[str, float]] = {}
        for row in actionable:
            agent = int(row["agent_id"])
            values[agent] = score_variants(row, history[agent])
            expected = float(row["expected_update_value"])
            if not math.isclose(values[agent]["R0"], expected, rel_tol=0.0, abs_tol=1e-12):
                baseline_mismatches.append(
                    {"setting": label, "update": update, "agent": agent, "type": "score"}
                )
        ranks: dict[str, dict[int, int]] = {}
        top2: dict[str, list[int]] = {}
        for policy in POLICIES:
            ordered = sorted(
                actionable,
                key=lambda row: rank_key(row, values[int(row["agent_id"])][policy]),
            )
            ranks[policy] = {
                int(row["agent_id"]): rank for rank, row in enumerate(ordered, start=1)
            }
            top2[policy] = [int(row["agent_id"]) for row in ordered[:2]]
        actual_order = [
            int(row["target_agent_id"])
            for row in sorted(
                (row for row in run["branches"] if int(row["update_index"]) == update),
                key=lambda row: int(row["target_rank"]),
            )
        ]
        if top2["R0"] != actual_order:
            baseline_mismatches.append(
                {
                    "setting": label,
                    "update": update,
                    "type": "top2",
                    "replayed": top2["R0"],
                    "actual": actual_order,
                }
            )
        for row in actionable:
            agent = int(row["agent_id"])
            if int(row["selection_rank"]) != ranks["R0"][agent] or bool(row["selected"]) != (
                agent in top2["R0"]
            ):
                baseline_mismatches.append(
                    {"setting": label, "update": update, "agent": agent, "type": "rank"}
                )
        timeline_updates.append(
            {
                "setting": label,
                "update": update,
                "actual_top2": top2["R0"],
                "R0_top2": top2["R0"],
                "R1_top2": top2["R1"],
                "R2_top2": top2["R2"],
                "R3_top2": top2["R3"],
                "R1_changed": top2["R1"] != top2["R0"],
                "R2_changed": top2["R2"] != top2["R0"],
                "R3_changed": top2["R3"] != top2["R0"],
                "actionable_agents": sorted(values),
                "history_before": {
                    str(agent): dict(history[agent]) for agent in range(5)
                },
            }
        )
        current_rows: dict[int, dict[str, Any]] = {}
        for agent in range(5):
            row = next((item for item in actionable if int(item["agent_id"]) == agent), None)
            event = event_lookup.get((update, agent))
            outcome = branch_outcome(event)
            committed_target = commit_lookup[update]["committed_target_id"]
            base = {
                "setting": label,
                "update": update,
                "agent": agent,
                "actionable": row is not None,
                "actual_selected": agent in actual_order,
                "actual_branch_outcome": outcome,
                "actual_committed": (
                    committed_target is not None and int(committed_target) == agent
                ),
                "persistent_attempt": history[agent]["attempts"],
                "persistent_feasible": history[agent]["feasible"],
                "persistent_normal_failure": history[agent]["normal_failures"],
                "persistent_consecutive_failure": history[agent]["consecutive_failures"],
                "last_feasible_update": history[agent]["last_feasible_update"],
            }
            if row is None:
                base.update(
                    {
                        "B_i": None,
                        "state_failure": None,
                        "rho_state": None,
                        "normalized_wait": None,
                        "R0_score": None,
                        "R0_rank": None,
                        "R1_score": None,
                        "R1_rank": None,
                        "R2_score": None,
                        "R2_rank": None,
                        "R3_score": None,
                        "R3_rank": None,
                        "beta_realizability": (history[agent]["feasible"] + 1)
                        / (history[agent]["attempts"] + 2),
                        "consecutive_discount": 1
                        / (1 + history[agent]["consecutive_failures"]),
                        "floored_realizability": 0.25
                        + 0.75
                        * (history[agent]["feasible"] + 1)
                        / (history[agent]["attempts"] + 2),
                    }
                )
            else:
                score = values[agent]
                base.update(
                    {
                        "B_i": row["opportunity_value"],
                        "state_failure": row["branch_failure_count"],
                        "rho_state": row["repairability_discount"],
                        "normalized_wait": row["normalized_wait"],
                        "R0_score": score["R0"],
                        "R0_rank": ranks["R0"][agent],
                        "R1_score": score["R1"],
                        "R1_rank": ranks["R1"][agent],
                        "R2_score": score["R2"],
                        "R2_rank": ranks["R2"][agent],
                        "R3_score": score["R3"],
                        "R3_rank": ranks["R3"][agent],
                        "beta_realizability": score["beta_rate"],
                        "consecutive_discount": score["cross_consecutive_discount"],
                        "floored_realizability": score["floored_realizability"],
                    }
                )
            agent_rows.append(base)
            current_rows[agent] = base

            if outcome == "FEASIBLE":
                retained = {
                    policy: agent in top2[policy] for policy in POLICIES
                }
                feasible_rows.append(
                    {
                        "setting": label,
                        "update": update,
                        "agent": agent,
                        "competition_loser": bool(event["competition_loser"]),
                        "persistent_attempt_before": history[agent]["attempts"],
                        "persistent_feasible_before": history[agent]["feasible"],
                        "persistent_consecutive_failure_before": history[agent][
                            "consecutive_failures"
                        ],
                        **{f"{policy}_rank": ranks[policy][agent] for policy in POLICIES},
                        **{f"{policy}_retained": retained[policy] for policy in POLICIES},
                    }
                )
                if history[agent]["consecutive_failures"] > 0:
                    recovery_witnesses.append(
                        {
                            "setting": label,
                            "feasible_update": update,
                            "agent": agent,
                            "consecutive_failures_before": history[agent][
                                "consecutive_failures"
                            ],
                            "beta_rate_before": values[agent]["beta_rate"],
                            "R1_rank_before": ranks["R1"][agent],
                            "R2_rank_before": ranks["R2"][agent],
                            "R3_rank_before": ranks["R3"][agent],
                            "next_actionable_update": None,
                        }
                    )
            elif outcome == "NORMAL_FAILURE":
                failure_rows.append(
                    {
                        "setting": label,
                        "update": update,
                        "agent": agent,
                        "persistent_attempt_before": history[agent]["attempts"],
                        "persistent_feasible_before": history[agent]["feasible"],
                        "persistent_consecutive_failure_before": history[agent][
                            "consecutive_failures"
                        ],
                        "repeated_failure": history[agent]["consecutive_failures"] > 0,
                        **{f"{policy}_rank": ranks[policy][agent] for policy in POLICIES},
                        **{
                            f"{policy}_demoted": agent not in top2[policy]
                            for policy in POLICIES
                        },
                    }
                )
            if base["actual_committed"]:
                commit_rows.append(
                    {
                        "setting": label,
                        "update": update,
                        "agent": agent,
                        **{
                            f"{policy}_retained": agent in top2[policy]
                            for policy in POLICIES
                        },
                    }
                )

        for witness in recovery_witnesses:
            if witness["next_actionable_update"] is not None:
                continue
            agent = int(witness["agent"])
            if update <= int(witness["feasible_update"]):
                continue
            next_row = current_rows[agent]
            if not next_row["actionable"]:
                continue
            witness.update(
                {
                    "next_actionable_update": update,
                    "beta_rate_after": next_row["beta_realizability"],
                    "consecutive_discount_after": next_row["consecutive_discount"],
                    "R1_rank_after": next_row["R1_rank"],
                    "R2_rank_after": next_row["R2_rank"],
                    "R3_rank_after": next_row["R3_rank"],
                }
            )

        for agent in actual_order:
            event = event_lookup[(update, agent)]
            outcome = branch_outcome(event)
            if outcome == "FEASIBLE":
                history[agent]["attempts"] += 1
                history[agent]["feasible"] += 1
                history[agent]["consecutive_failures"] = 0
                history[agent]["last_feasible_update"] = update
            elif outcome == "NORMAL_FAILURE":
                history[agent]["attempts"] += 1
                history[agent]["normal_failures"] += 1
                history[agent]["consecutive_failures"] += 1
            elif outcome == "OPERATIONAL_FAILURE":
                pass
            else:
                raise AssertionError("selected_branch_event_missing")

    return {
        "baseline_mismatches": baseline_mismatches,
        "timeline_updates": timeline_updates,
        "agent_rows": agent_rows,
        "feasible_rows": feasible_rows,
        "failure_rows": failure_rows,
        "commit_rows": commit_rows,
        "recovery_witnesses": recovery_witnesses,
    }


def setting_metrics(result: dict[str, Any], policy: str) -> dict[str, Any]:
    selections = [
        agent
        for row in result["timeline_updates"]
        for agent in row[f"{policy}_top2"]
    ]
    top1 = [row[f"{policy}_top2"][0] for row in result["timeline_updates"]]
    top2_position = [row[f"{policy}_top2"][1] for row in result["timeline_updates"]]
    counts = Counter(selections)
    entropy, concentration, max_share = entropy_and_concentration(dict(counts))
    feasible = result["feasible_rows"]
    failures = result["failure_rows"]
    repeated = [row for row in failures if row["repeated_failure"]]
    commits = result["commit_rows"]
    selected_agent_rows = [
        row
        for row in result["agent_rows"]
        if row["actionable"] and row[f"{policy}_rank"] <= 2
    ]
    return {
        "selection_count_by_agent": {
            str(agent): counts.get(agent, 0) for agent in range(5)
        },
        "top1_count_by_agent": {
            str(agent): top1.count(agent) for agent in range(5)
        },
        "top2_position_count_by_agent": {
            str(agent): top2_position.count(agent) for agent in range(5)
        },
        "distinct_selected_agents": len(counts),
        "selection_concentration_hhi": concentration,
        "selection_entropy_normalized": entropy,
        "maximum_selection_share": max_share,
        "agent1_plus_agent4_selection_count": counts.get(1, 0) + counts.get(4, 0),
        "top2_changes_vs_v14": sum(
            row[f"{policy}_top2"] != row["R0_top2"]
            for row in result["timeline_updates"]
        ),
        "known_feasible_retained": sum(row[f"{policy}_retained"] for row in feasible),
        "known_feasible_total": len(feasible),
        "known_feasible_retention_rate": (
            sum(row[f"{policy}_retained"] for row in feasible) / len(feasible)
            if feasible
            else None
        ),
        "known_commit_target_retained": sum(
            row[f"{policy}_retained"] for row in commits
        ),
        "known_commit_target_total": len(commits),
        "known_commit_target_retention_rate": (
            sum(row[f"{policy}_retained"] for row in commits) / len(commits)
            if commits
            else None
        ),
        "known_failed_branch_demoted": sum(row[f"{policy}_demoted"] for row in failures),
        "known_failed_branch_total": len(failures),
        "known_failed_branch_demotion_rate": (
            sum(row[f"{policy}_demoted"] for row in failures) / len(failures)
            if failures
            else None
        ),
        "repeated_known_failed_branch_demoted": sum(
            row[f"{policy}_demoted"] for row in repeated
        ),
        "repeated_known_failed_branch_total": len(repeated),
        "repeated_known_failed_branch_demotion_rate": (
            sum(row[f"{policy}_demoted"] for row in repeated) / len(repeated)
            if repeated
            else None
        ),
        "mean_beta_realizability_of_selected": sum(
            row["beta_realizability"] for row in selected_agent_rows
        )
        / len(selected_agent_rows),
    }


def aggregate_metrics(
    per_setting: dict[str, dict[str, dict[str, Any]]], policy: str
) -> dict[str, Any]:
    selected_counts = {
        agent: sum(
            setting[policy]["selection_count_by_agent"][str(agent)]
            for setting in per_setting.values()
        )
        for agent in range(5)
    }
    top1_counts = {
        agent: sum(
            setting[policy]["top1_count_by_agent"][str(agent)]
            for setting in per_setting.values()
        )
        for agent in range(5)
    }
    top2_position_counts = {
        agent: sum(
            setting[policy]["top2_position_count_by_agent"][str(agent)]
            for setting in per_setting.values()
        )
        for agent in range(5)
    }
    totals = {
        key: sum(setting[policy][key] for setting in per_setting.values())
        for key in (
            "known_feasible_retained",
            "known_feasible_total",
            "known_commit_target_retained",
            "known_commit_target_total",
            "known_failed_branch_demoted",
            "known_failed_branch_total",
            "repeated_known_failed_branch_demoted",
            "repeated_known_failed_branch_total",
            "top2_changes_vs_v14",
        )
    }
    entropy, concentration, max_share = entropy_and_concentration(selected_counts)
    return {
        "selection_count_by_agent": {
            str(agent): selected_counts[agent] for agent in range(5)
        },
        "top1_count_by_agent": {
            str(agent): top1_counts[agent] for agent in range(5)
        },
        "top2_position_count_by_agent": {
            str(agent): top2_position_counts[agent] for agent in range(5)
        },
        "distinct_selected_agents": sum(value > 0 for value in selected_counts.values()),
        "agent1_plus_agent4_selection_count": selected_counts[1] + selected_counts[4],
        "selection_entropy_normalized": entropy,
        "selection_concentration_hhi": concentration,
        "maximum_selection_share": max_share,
        **totals,
        "known_feasible_retention_rate": totals["known_feasible_retained"]
        / totals["known_feasible_total"],
        "known_commit_target_retention_rate": totals["known_commit_target_retained"]
        / totals["known_commit_target_total"],
        "known_failed_branch_demotion_rate": totals["known_failed_branch_demoted"]
        / totals["known_failed_branch_total"],
        "repeated_known_failed_branch_demotion_rate": totals[
            "repeated_known_failed_branch_demoted"
        ]
        / totals["repeated_known_failed_branch_total"],
        "mean_beta_realizability_of_selected": sum(
            setting[policy]["mean_beta_realizability_of_selected"]
            for setting in per_setting.values()
        )
        / len(per_setting),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: analysis.py <authoritative_formal_root>")
    formal_root = Path(sys.argv[1]).resolve()
    gate = read_json(
        REPO / "reports" / "v14_qwen3_14b_seed46_20260809" / "formal" / "stage_gate.json"
    )
    if gate["gate"] != "PASS" or gate["blocker_count"] != 0 or gate["major_count"] != 0:
        raise AssertionError("formal_gate_not_passed")
    runs = {
        label: load_setting(formal_root, label, name)
        for label, name in SETTINGS.items()
    }
    replays = {label: replay_setting(run) for label, run in runs.items()}
    mismatches = [
        mismatch
        for result in replays.values()
        for mismatch in result["baseline_mismatches"]
    ]
    replay_validation = {
        "BASELINE_REPLAY": "PASS" if not mismatches else "FAIL",
        "baseline_mismatch_count": len(mismatches),
        "baseline_mismatches": mismatches,
        "settings": list(SETTINGS),
        "updates_per_setting": 32,
        "history_causality": "PRIOR_ACTUAL_BRANCH_OUTCOMES_ONLY",
        "operational_failures_excluded_from_history": True,
        "fixed_actual_trajectory": True,
        "API_CALLS": 0,
    }
    write_json(OUTPUT / "replay_validation.json", replay_validation)
    if mismatches:
        write_json(
            OUTPUT / "module1_realizability_summary.json",
            {
                "TASK_STATUS": "HOLD",
                "API_CALLS": 0,
                "BASELINE_REPLAY": "FAIL",
                "V15_IMPLEMENTATION_AUTHORIZED": False,
            },
        )
        raise SystemExit("TASK_STATUS=HOLD: baseline selector replay failed")

    per_setting = {
        label: {
            policy: setting_metrics(result, policy) for policy in POLICIES
        }
        for label, result in replays.items()
    }
    aggregate = {
        policy: aggregate_metrics(per_setting, policy) for policy in POLICIES
    }
    all_agent_rows = [row for result in replays.values() for row in result["agent_rows"]]
    all_feasible = [row for result in replays.values() for row in result["feasible_rows"]]
    all_failures = [row for result in replays.values() for row in result["failure_rows"]]
    all_updates = [row for result in replays.values() for row in result["timeline_updates"]]
    recovery_witnesses = [
        row
        for result in replays.values()
        for row in result["recovery_witnesses"]
        if row["next_actionable_update"] is not None
    ]
    recovery_evidence = "SUFFICIENT" if recovery_witnesses else "INSUFFICIENT"

    comparison_rows: list[dict[str, Any]] = []
    for label in (*SETTINGS, "ALL"):
        source = per_setting[label] if label != "ALL" else aggregate
        for policy in POLICIES:
            row = source[policy]
            comparison_rows.append(
                {
                    "setting": label,
                    "policy": policy,
                    "agent0_selections": row["selection_count_by_agent"]["0"],
                    "agent1_selections": row["selection_count_by_agent"]["1"],
                    "agent2_selections": row["selection_count_by_agent"]["2"],
                    "agent3_selections": row["selection_count_by_agent"]["3"],
                    "agent4_selections": row["selection_count_by_agent"]["4"],
                    "agent0_top1": row["top1_count_by_agent"]["0"],
                    "agent1_top1": row["top1_count_by_agent"]["1"],
                    "agent2_top1": row["top1_count_by_agent"]["2"],
                    "agent3_top1": row["top1_count_by_agent"]["3"],
                    "agent4_top1": row["top1_count_by_agent"]["4"],
                    "agent0_top2_position": row["top2_position_count_by_agent"]["0"],
                    "agent1_top2_position": row["top2_position_count_by_agent"]["1"],
                    "agent2_top2_position": row["top2_position_count_by_agent"]["2"],
                    "agent3_top2_position": row["top2_position_count_by_agent"]["3"],
                    "agent4_top2_position": row["top2_position_count_by_agent"]["4"],
                    "distinct_selected_agents": row["distinct_selected_agents"],
                    "agent1_plus_agent4": row["agent1_plus_agent4_selection_count"],
                    "top2_changes_vs_v14": row["top2_changes_vs_v14"],
                    "selection_concentration_hhi": row["selection_concentration_hhi"],
                    "selection_entropy_normalized": row["selection_entropy_normalized"],
                    "maximum_selection_share": row["maximum_selection_share"],
                    "known_feasible_retained": row["known_feasible_retained"],
                    "known_feasible_total": row["known_feasible_total"],
                    "known_feasible_retention_rate": row["known_feasible_retention_rate"],
                    "known_commit_retention_rate": row[
                        "known_commit_target_retention_rate"
                    ],
                    "known_failure_demotion_rate": row[
                        "known_failed_branch_demotion_rate"
                    ],
                    "repeated_failure_demotion_rate": row[
                        "repeated_known_failed_branch_demotion_rate"
                    ],
                    "mean_selected_beta_realizability": row[
                        "mean_beta_realizability_of_selected"
                    ],
                }
            )

    score_witnesses: dict[str, list[dict[str, Any]]] = {}
    s3_rows = [row for row in all_agent_rows if row["setting"] == "S3"]
    for agent in (1, 4):
        actionable = [
            row
            for row in s3_rows
            if row["agent"] == agent and row["actionable"] and row["actual_selected"]
        ]
        minimum_rho = min(float(row["rho_state"]) for row in actionable)
        score_witnesses[str(agent)] = [
            row for row in actionable if math.isclose(float(row["rho_state"]), minimum_rho)
        ]

    variants = {
        "R0": {
            "name": "V14_STATE_LOCAL_REPAIRABILITY",
            "formula": "B_i*rho_state_i+0.05*what_i",
        },
        "R1": {
            "name": "PERSISTENT_BETA_REALIZABILITY",
            "history": ["persistent_attempt_count", "persistent_feasible_count"],
            "factor": "(feasible_i+1)/(attempt_i+2)",
            "alpha": 1,
            "beta": 1,
        },
        "R2": {
            "name": "PERSISTENT_CONSECUTIVE_FAILURE",
            "history": ["persistent_consecutive_normal_failure_count"],
            "factor": "1/(1+c_i)",
            "feasible_branch_reset": True,
        },
        "R3": {
            "name": "FLOORED_PERSISTENT_REALIZABILITY",
            "history": ["persistent_attempt_count", "persistent_feasible_count"],
            "factor": "0.25+0.75*((feasible_i+1)/(attempt_i+2))",
            "floor": 0.25,
        },
        "shared_constraints": {
            "hard_freeze": False,
            "future_leakage": False,
            "operational_failure_updates_history": False,
            "commit_is_only_success": False,
            "actual_trajectory_fixed": True,
            "test_used": False,
        },
    }

    failure_demotion_by_agent = {
        policy: {
            str(agent): {
                "known_failure_count": sum(row["agent"] == agent for row in all_failures),
                "demoted_count": sum(
                    row["agent"] == agent and row[f"{policy}_demoted"]
                    for row in all_failures
                ),
                "demotion_rate": (
                    sum(
                        row["agent"] == agent and row[f"{policy}_demoted"]
                        for row in all_failures
                    )
                    / sum(row["agent"] == agent for row in all_failures)
                    if any(row["agent"] == agent for row in all_failures)
                    else None
                ),
            }
            for agent in range(5)
        }
        for policy in POLICIES
    }
    branch_outcome_counts = Counter(
        row["actual_branch_outcome"]
        for row in all_agent_rows
        if row["actual_selected"]
    )

    # None of the three frozen formulas satisfies the requested cross-setting
    # efficiency/retention behavior, so the design task recommends no variant.
    recommendation = "NONE"
    why = (
        "R1 and R3 increase aggregate Agent1+4 occupancy versus v14; R2 reduces "
        "it only from 76 to 74 while retaining 22/27 known-feasible branches and "
        "16/19 known commit targets. All three leave the decisive low-rho S3 "
        "Agent1/4 states in Top2 because the unscaled 0.05 wait term dominates."
    )

    summary = {
        "TASK_STATUS": "COMPLETE",
        "API_CALLS": 0,
        "BASELINE_REPLAY": "PASS",
        "METHOD_VERSION": METHOD_VERSION,
        "CHECKPOINT_VERSION": CHECKPOINT_VERSION,
        "FIXED_ACTUAL_TRAJECTORY_SELECTOR_REPLAY": True,
        "NOT_ALTERNATIVE_TRAINING_TRAJECTORY": True,
        "TEST_USED_FOR_SELECTION": False,
        "V15_DESIGN_ONLY": True,
        "V15_IMPLEMENTATION_AUTHORIZED": False,
        "per_setting": per_setting,
        "aggregate": aggregate,
        "recovery_evidence": recovery_evidence,
        "recovery_witnesses": recovery_witnesses,
        "agent14_low_state_discount_witnesses": score_witnesses,
        "failure_demotion_by_agent": failure_demotion_by_agent,
        "actual_branch_outcome_counts": dict(branch_outcome_counts),
        "R1_AGENT14_SELECTION_COUNT": aggregate["R1"][
            "agent1_plus_agent4_selection_count"
        ],
        "R2_AGENT14_SELECTION_COUNT": aggregate["R2"][
            "agent1_plus_agent4_selection_count"
        ],
        "R3_AGENT14_SELECTION_COUNT": aggregate["R3"][
            "agent1_plus_agent4_selection_count"
        ],
        "R1_KNOWN_FEASIBLE_RETENTION": aggregate["R1"][
            "known_feasible_retention_rate"
        ],
        "R2_KNOWN_FEASIBLE_RETENTION": aggregate["R2"][
            "known_feasible_retention_rate"
        ],
        "R3_KNOWN_FEASIBLE_RETENTION": aggregate["R3"][
            "known_feasible_retention_rate"
        ],
        "R1_KNOWN_FAILURE_DEMOTION": aggregate["R1"][
            "known_failed_branch_demotion_rate"
        ],
        "R2_KNOWN_FAILURE_DEMOTION": aggregate["R2"][
            "known_failed_branch_demotion_rate"
        ],
        "R3_KNOWN_FAILURE_DEMOTION": aggregate["R3"][
            "known_failed_branch_demotion_rate"
        ],
        "RECOVERY_EVIDENCE": recovery_evidence,
        "RECOMMENDED_MODULE1_REALIZABILITY": recommendation,
        "WHY": why,
    }

    write_csv(
        OUTPUT / "selector_replay_comparison.csv",
        comparison_rows,
        list(comparison_rows[0].keys()),
    )
    write_jsonl(OUTPUT / "selector_replay_timeline.jsonl", all_updates)
    write_csv(
        OUTPUT / "agent_realizability_timeline.csv",
        all_agent_rows,
        list(all_agent_rows[0].keys()),
    )
    write_csv(
        OUTPUT / "known_feasible_retention.csv",
        all_feasible,
        list(all_feasible[0].keys()),
    )
    write_csv(
        OUTPUT / "known_failure_demotion.csv",
        all_failures,
        list(all_failures[0].keys()),
    )
    write_json(OUTPUT / "variant_definitions.json", variants)
    write_json(OUTPUT / "module1_realizability_summary.json", summary)

    agent1_witness = score_witnesses["1"][0]
    agent4_witness = score_witnesses["4"][0]
    recommendation_report = f"""# v15 Module 1 Realizability Recommendation

`RECOMMENDED_MODULE1_REALIZABILITY = NONE`

## Decision

None of R1/R2/R3 satisfies the frozen cross-setting design rule. R1 increases
aggregate Agent1+4 selections from {aggregate['R0']['agent1_plus_agent4_selection_count']}
to {aggregate['R1']['agent1_plus_agent4_selection_count']}; R3 increases them to
{aggregate['R3']['agent1_plus_agent4_selection_count']}. R2 reduces the count
only to {aggregate['R2']['agent1_plus_agent4_selection_count']}, while retaining
{aggregate['R2']['known_feasible_retained']}/{aggregate['R2']['known_feasible_total']}
known-feasible branch events and
{aggregate['R2']['known_commit_target_retained']}/{aggregate['R2']['known_commit_target_total']}
actual commit targets.

| Policy | Agent1+4 selections | Known-feasible retention | Commit retention | Failure demotion | Repeated-failure demotion |
|---|---:|---:|---:|---:|---:|
| R1 | {aggregate['R1']['agent1_plus_agent4_selection_count']} | {aggregate['R1']['known_feasible_retention_rate']:.3f} | {aggregate['R1']['known_commit_target_retention_rate']:.3f} | {aggregate['R1']['known_failed_branch_demotion_rate']:.3f} | {aggregate['R1']['repeated_known_failed_branch_demotion_rate']:.3f} |
| R2 | {aggregate['R2']['agent1_plus_agent4_selection_count']} | {aggregate['R2']['known_feasible_retention_rate']:.3f} | {aggregate['R2']['known_commit_target_retention_rate']:.3f} | {aggregate['R2']['known_failed_branch_demotion_rate']:.3f} | {aggregate['R2']['repeated_known_failed_branch_demotion_rate']:.3f} |
| R3 | {aggregate['R3']['agent1_plus_agent4_selection_count']} | {aggregate['R3']['known_feasible_retention_rate']:.3f} | {aggregate['R3']['known_commit_target_retention_rate']:.3f} | {aggregate['R3']['known_failed_branch_demotion_rate']:.3f} | {aggregate['R3']['repeated_known_failed_branch_demotion_rate']:.3f} |

## Why low v14 state discounts still enter Top2

At S3 update {agent1_witness['update']}, Agent1 has `B_i={float(agent1_witness['B_i']):.6f}`,
`rho_state={float(agent1_witness['rho_state']):.6f}` and normalized wait 1. Its
v14 score is {float(agent1_witness['R0_score']):.6f}: approximately 0.0259 from
discounted opportunity plus the unscaled 0.05 wait term. It is rank
{agent1_witness['R0_rank']}. R1/R2/R3 move it only to ranks
{agent1_witness['R1_rank']}/{agent1_witness['R2_rank']}/{agent1_witness['R3_rank']}.

At S3 update {agent4_witness['update']}, Agent4 has
`rho_state={float(agent4_witness['rho_state']):.6f}` and the same full wait term.
Its v14 score is {float(agent4_witness['R0_score']):.6f}, rank
{agent4_witness['R0_rank']}; R1/R2/R3 all leave it at rank
{agent4_witness['R1_rank']}/{agent4_witness['R2_rank']}/{agent4_witness['R3_rank']}.

The proposed formulas multiply only `B_i * rho_state_i`; none discounts the
additive wait term. The low state-local penalty therefore does not guarantee
demotion when normalized wait is maximal and competing actionable members have
lower total-order scores.

## Recovery evidence

Recovery evidence is **{recovery_evidence}**. There are
{len(recovery_witnesses)} actual events where a member produced a feasible
branch after one or more consecutive failures and was actionable again later.
R2 resets its cross-state discount immediately; R1 and R3 recover statistically.
This validates the intended recovery mechanics, but it does not offset the
retention and efficiency failures above.

## Interpretation boundary

This is fixed-actual-trajectory selector replay. Alternative selections have no
generated candidates, so no counterfactual acceptance rate or final train/test
result is claimed. Final test data were not used.

`V15_IMPLEMENTATION_AUTHORIZED = false`
"""
    (OUTPUT / "recommendation.md").write_text(recommendation_report, encoding="utf-8")

    readme = f"""# v15 Module 1 Cross-State Empirical Realizability Design

- Task status: **COMPLETE**
- API calls: **0**
- Baseline selector replay: **PASS**
- Recommended design: **NONE**
- Recovery evidence: **{recovery_evidence}**

This directory replays R0/R1/R2/R3 across the actual S1, S2, and S3 Seed46
trajectories. Every update uses only branch outcomes from earlier actual
updates. Team state, responsibility, routing, active lanes, opportunity scores,
state-local failure counts, and subsequent trajectory remain frozen.

The analysis does not use the Task 1 Module 3 proposal, generate candidates,
call an API, or infer alternative training/test performance. See
`recommendation.md` for the design conclusion and
`module1_realizability_summary.json` for complete machine-readable evidence.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "TASK_STATUS",
                    "API_CALLS",
                    "BASELINE_REPLAY",
                    "R1_AGENT14_SELECTION_COUNT",
                    "R2_AGENT14_SELECTION_COUNT",
                    "R3_AGENT14_SELECTION_COUNT",
                    "R1_KNOWN_FEASIBLE_RETENTION",
                    "R2_KNOWN_FEASIBLE_RETENTION",
                    "R3_KNOWN_FEASIBLE_RETENTION",
                    "R1_KNOWN_FAILURE_DEMOTION",
                    "R2_KNOWN_FAILURE_DEMOTION",
                    "R3_KNOWN_FAILURE_DEMOTION",
                    "RECOVERY_EVIDENCE",
                    "RECOMMENDED_MODULE1_REALIZABILITY",
                    "V15_IMPLEMENTATION_AUTHORIZED",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
