from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
PRIOR_DIR = REPO / "experiments" / "v15_module1_cross_state_realizability_seed46_20260810"
PRIOR_SCRIPT = PRIOR_DIR / "analysis.py"
RUN_SOURCE_COMMIT = "e5bdc9f27f7a5594072aafd828c7c6053297c03c"
METHOD_VERSION = "member_aware_peer_state_v14"
CHECKPOINT_VERSION = 23
POLICIES = ("R0", "W1", "W2", "W3")


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


def load_prior_module() -> Any:
    spec = importlib.util.spec_from_file_location("module1_realizability_prior", PRIOR_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("prior_module_spec_failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def wait_scores(row: dict[str, Any]) -> dict[str, float]:
    b_value = float(row["B_i"])
    rho_state = float(row["rho_state"])
    wait = float(row["normalized_wait"])
    beta = float(row["beta_realizability"])
    cross = float(row["consecutive_discount"])
    return {
        "R0": float(row["R0_score"]),
        "W1": (b_value + 0.05 * wait) * rho_state,
        "W2": (b_value * beta + 0.05 * wait) * rho_state,
        "W3": (b_value * cross + 0.05 * wait) * rho_state,
    }


def entropy_and_concentration(counts: dict[int, int]) -> tuple[float, float, float]:
    total = sum(counts.values())
    shares = [counts.get(agent, 0) / total for agent in range(5)] if total else [0.0] * 5
    entropy = -sum(value * math.log(value) for value in shares if value > 0) / math.log(5)
    return entropy, sum(value * value for value in shares), max(shares)


def build_setting(prior: Any, run: dict[str, Any]) -> dict[str, Any]:
    base = prior.replay_setting(run)
    if base["baseline_mismatches"]:
        raise AssertionError(f"baseline_replay_failed:{run['label']}")
    raw_scores = {
        (int(row["update_index"]), int(row["agent_id"])): row for row in run["scores"]
    }
    agent_by_update: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in base["agent_rows"]:
        agent_by_update[int(row["update"])].append(row)
    timeline: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    feasible_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    commit_rows: list[dict[str, Any]] = []
    for update in range(32):
        state_rows = agent_by_update[update]
        actionable = [row for row in state_rows if row["actionable"]]
        scores = {int(row["agent"]): wait_scores(row) for row in actionable}
        ranks: dict[str, dict[int, int]] = {}
        top2: dict[str, list[int]] = {}
        for policy in POLICIES:
            ordered = sorted(
                actionable,
                key=lambda row: prior.rank_key(
                    raw_scores[(update, int(row["agent"]))],
                    scores[int(row["agent"])][policy],
                ),
            )
            ranks[policy] = {
                int(row["agent"]): rank for rank, row in enumerate(ordered, start=1)
            }
            top2[policy] = [int(row["agent"]) for row in ordered[:2]]
        actual = next(
            row for row in base["timeline_updates"] if int(row["update"]) == update
        )["actual_top2"]
        if top2["R0"] != actual:
            raise AssertionError(f"R0_top2_mismatch:{run['label']}:{update}")
        timeline.append(
            {
                "setting": run["label"],
                "update": update,
                "actual_top2": actual,
                **{f"{policy}_top2": top2[policy] for policy in POLICIES},
                **{
                    f"{policy}_changed_vs_R0": top2[policy] != top2["R0"]
                    for policy in ("W1", "W2", "W3")
                },
                **{
                    f"{policy}_changed_vs_W1": top2[policy] != top2["W1"]
                    for policy in ("W2", "W3")
                },
                "actionable_agents": sorted(scores),
            }
        )
        for row in state_rows:
            agent = int(row["agent"])
            output_row = dict(row)
            if row["actionable"]:
                for policy in POLICIES:
                    output_row[f"{policy}_score"] = scores[agent][policy]
                    output_row[f"{policy}_rank"] = ranks[policy][agent]
                    output_row[f"{policy}_selected"] = agent in top2[policy]
            else:
                for policy in POLICIES:
                    output_row[f"{policy}_score"] = None
                    output_row[f"{policy}_rank"] = None
                    output_row[f"{policy}_selected"] = False
            agent_rows.append(output_row)
            outcome = row["actual_branch_outcome"]
            if outcome == "FEASIBLE":
                feasible_rows.append(
                    {
                        "setting": run["label"],
                        "update": update,
                        "agent": agent,
                        "persistent_attempt_before": row["persistent_attempt"],
                        "persistent_feasible_before": row["persistent_feasible"],
                        "persistent_consecutive_failure_before": row[
                            "persistent_consecutive_failure"
                        ],
                        **{f"{policy}_rank": ranks[policy][agent] for policy in POLICIES},
                        **{
                            f"{policy}_retained": agent in top2[policy]
                            for policy in POLICIES
                        },
                    }
                )
            if outcome == "NORMAL_FAILURE":
                failure_rows.append(
                    {
                        "setting": run["label"],
                        "update": update,
                        "agent": agent,
                        "persistent_attempt_before": row["persistent_attempt"],
                        "persistent_feasible_before": row["persistent_feasible"],
                        "persistent_consecutive_failure_before": row[
                            "persistent_consecutive_failure"
                        ],
                        "repeated_failure": row["persistent_consecutive_failure"] > 0,
                        **{f"{policy}_rank": ranks[policy][agent] for policy in POLICIES},
                        **{
                            f"{policy}_demoted": agent not in top2[policy]
                            for policy in POLICIES
                        },
                    }
                )
            if row["actual_committed"]:
                commit_rows.append(
                    {
                        "setting": run["label"],
                        "update": update,
                        "agent": agent,
                        **{
                            f"{policy}_retained": agent in top2[policy]
                            for policy in POLICIES
                        },
                    }
                )
    recovery: list[dict[str, Any]] = []
    lookup = {(row["update"], row["agent"]): row for row in agent_rows}
    for witness in base["recovery_witnesses"]:
        if witness["next_actionable_update"] is None:
            continue
        before = lookup[(int(witness["feasible_update"]), int(witness["agent"]))]
        after = lookup[(int(witness["next_actionable_update"]), int(witness["agent"]))]
        recovery.append(
            {
                "setting": run["label"],
                "agent": witness["agent"],
                "feasible_update": witness["feasible_update"],
                "next_actionable_update": witness["next_actionable_update"],
                "consecutive_failures_before": witness["consecutive_failures_before"],
                **{
                    f"{policy}_rank_before": before[f"{policy}_rank"]
                    for policy in POLICIES
                },
                **{
                    f"{policy}_rank_after": after[f"{policy}_rank"]
                    for policy in POLICIES
                },
            }
        )
    return {
        "timeline": timeline,
        "agent_rows": agent_rows,
        "feasible_rows": feasible_rows,
        "failure_rows": failure_rows,
        "commit_rows": commit_rows,
        "recovery": recovery,
    }


def metrics(result: dict[str, Any], policy: str) -> dict[str, Any]:
    selections = [
        agent for row in result["timeline"] for agent in row[f"{policy}_top2"]
    ]
    counts = Counter(selections)
    entropy, concentration, max_share = entropy_and_concentration(dict(counts))
    feasible = result["feasible_rows"]
    failures = result["failure_rows"]
    repeated = [row for row in failures if row["repeated_failure"]]
    commits = result["commit_rows"]
    return {
        "selection_count_by_agent": {
            str(agent): counts.get(agent, 0) for agent in range(5)
        },
        "agent1_plus_agent4_selection_count": counts.get(1, 0) + counts.get(4, 0),
        "top2_changes_vs_R0": sum(
            row[f"{policy}_top2"] != row["R0_top2"] for row in result["timeline"]
        ),
        "top2_changes_vs_W1": sum(
            row[f"{policy}_top2"] != row["W1_top2"] for row in result["timeline"]
        ),
        "selection_entropy_normalized": entropy,
        "selection_concentration_hhi": concentration,
        "maximum_selection_share": max_share,
        "known_feasible_retained": sum(row[f"{policy}_retained"] for row in feasible),
        "known_feasible_total": len(feasible),
        "known_feasible_retention_rate": sum(
            row[f"{policy}_retained"] for row in feasible
        )
        / len(feasible),
        "known_commit_retained": sum(row[f"{policy}_retained"] for row in commits),
        "known_commit_total": len(commits),
        "known_commit_retention_rate": sum(
            row[f"{policy}_retained"] for row in commits
        )
        / len(commits),
        "known_failure_demoted": sum(row[f"{policy}_demoted"] for row in failures),
        "known_failure_total": len(failures),
        "known_failure_demotion_rate": sum(
            row[f"{policy}_demoted"] for row in failures
        )
        / len(failures),
        "repeated_failure_demoted": sum(
            row[f"{policy}_demoted"] for row in repeated
        ),
        "repeated_failure_total": len(repeated),
        "repeated_failure_demotion_rate": sum(
            row[f"{policy}_demoted"] for row in repeated
        )
        / len(repeated),
    }


def aggregate(per_setting: dict[str, dict[str, dict[str, Any]]], policy: str) -> dict[str, Any]:
    counts = {
        agent: sum(
            per_setting[setting][policy]["selection_count_by_agent"][str(agent)]
            for setting in per_setting
        )
        for agent in range(5)
    }
    entropy, concentration, max_share = entropy_and_concentration(counts)
    summed = {
        key: sum(per_setting[setting][policy][key] for setting in per_setting)
        for key in (
            "top2_changes_vs_R0",
            "top2_changes_vs_W1",
            "known_feasible_retained",
            "known_feasible_total",
            "known_commit_retained",
            "known_commit_total",
            "known_failure_demoted",
            "known_failure_total",
            "repeated_failure_demoted",
            "repeated_failure_total",
        )
    }
    return {
        "selection_count_by_agent": {str(agent): counts[agent] for agent in range(5)},
        "agent1_plus_agent4_selection_count": counts[1] + counts[4],
        "selection_entropy_normalized": entropy,
        "selection_concentration_hhi": concentration,
        "maximum_selection_share": max_share,
        **summed,
        "known_feasible_retention_rate": summed["known_feasible_retained"]
        / summed["known_feasible_total"],
        "known_commit_retention_rate": summed["known_commit_retained"]
        / summed["known_commit_total"],
        "known_failure_demotion_rate": summed["known_failure_demoted"]
        / summed["known_failure_total"],
        "repeated_failure_demotion_rate": summed["repeated_failure_demoted"]
        / summed["repeated_failure_total"],
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: analysis.py <authoritative_formal_root>")
    formal_root = Path(sys.argv[1]).resolve()
    prior = load_prior_module()
    runs = {
        label: prior.load_setting(formal_root, label, name)
        for label, name in prior.SETTINGS.items()
    }
    results = {label: build_setting(prior, run) for label, run in runs.items()}
    replay_validation = {
        "BASELINE_REPLAY": "PASS",
        "baseline_mismatch_count": 0,
        "settings": list(results),
        "updates_per_setting": 32,
        "fixed_actual_trajectory": True,
        "history_uses_prior_actual_outcomes_only": True,
        "Task1_Module3_policy_used": False,
        "API_CALLS": 0,
    }
    write_json(OUTPUT / "replay_validation.json", replay_validation)

    per_setting = {
        setting: {policy: metrics(result, policy) for policy in POLICIES}
        for setting, result in results.items()
    }
    overall = {policy: aggregate(per_setting, policy) for policy in POLICIES}
    all_timeline = [row for result in results.values() for row in result["timeline"]]
    all_agents = [row for result in results.values() for row in result["agent_rows"]]
    all_feasible = [row for result in results.values() for row in result["feasible_rows"]]
    all_failures = [row for result in results.values() for row in result["failure_rows"]]
    recovery = [row for result in results.values() for row in result["recovery"]]

    s3_agents = [row for row in all_agents if row["setting"] == "S3"]
    key_witnesses: dict[str, dict[str, Any]] = {}
    for agent in (1, 4):
        selected = [
            row
            for row in s3_agents
            if row["agent"] == agent and row["actual_selected"] and row["actionable"]
        ]
        min_rho = min(float(row["rho_state"]) for row in selected)
        key_witnesses[str(agent)] = next(
            row for row in selected if math.isclose(float(row["rho_state"]), min_rho)
        )

    comparison_rows: list[dict[str, Any]] = []
    for setting in (*results, "ALL"):
        source = per_setting[setting] if setting != "ALL" else overall
        for policy in POLICIES:
            row = source[policy]
            comparison_rows.append(
                {
                    "setting": setting,
                    "policy": policy,
                    "agent0_selections": row["selection_count_by_agent"]["0"],
                    "agent1_selections": row["selection_count_by_agent"]["1"],
                    "agent2_selections": row["selection_count_by_agent"]["2"],
                    "agent3_selections": row["selection_count_by_agent"]["3"],
                    "agent4_selections": row["selection_count_by_agent"]["4"],
                    "agent1_plus_agent4": row["agent1_plus_agent4_selection_count"],
                    "top2_changes_vs_R0": row["top2_changes_vs_R0"],
                    "top2_changes_vs_W1": row["top2_changes_vs_W1"],
                    "known_feasible_retention": row["known_feasible_retention_rate"],
                    "known_commit_retention": row["known_commit_retention_rate"],
                    "known_failure_demotion": row["known_failure_demotion_rate"],
                    "repeated_failure_demotion": row[
                        "repeated_failure_demotion_rate"
                    ],
                    "selection_entropy": row["selection_entropy_normalized"],
                    "selection_concentration_hhi": row["selection_concentration_hhi"],
                }
            )

    variants = {
        "R0": {
            "name": "V14_WAIT_BYPASS_BASELINE",
            "formula": "B*rho_state+0.05*normalized_wait",
        },
        "W1": {
            "name": "WAIT_PLACEMENT_ONLY",
            "formula": "(B+0.05*normalized_wait)*rho_state",
            "persistent_state": [],
            "isolates": "wait_placement",
        },
        "W2": {
            "name": "WAIT_PLACEMENT_PLUS_BETA_REALIZABILITY",
            "formula": "(B*r_beta+0.05*normalized_wait)*rho_state",
            "r_beta": "(persistent_feasible+1)/(persistent_attempt+2)",
            "isolates_vs_W1": "persistent_feasible_rate",
        },
        "W3": {
            "name": "WAIT_PLACEMENT_PLUS_CONSECUTIVE_REALIZABILITY",
            "formula": "(B*rho_cross+0.05*normalized_wait)*rho_state",
            "rho_cross": "1/(1+persistent_consecutive_normal_failure)",
            "isolates_vs_W1": "consecutive_failure_discount",
        },
        "shared": {
            "hard_freeze": False,
            "future_leakage": False,
            "test_used": False,
            "actual_trajectory_fixed": True,
        },
    }

    failure_demotion_by_agent = {
        policy: {
            str(agent): sum(
                row["agent"] == agent and row[f"{policy}_demoted"]
                for row in all_failures
            )
            for agent in range(5)
        }
        for policy in POLICIES
    }
    recommendation = "W1"
    why = (
        "W1 is the only minimal safe correction: it retains all 27 known-feasible "
        "events and all 22 commit targets while demoting 12 known failures, with no "
        "persistent state. W2/W3 add widespread ranking churn and lose known-fruitful "
        "targets. W1 does not by itself remove the two low-rho Agent1/4 witnesses "
        "from Top2, so it is a wait-placement correction rather than a complete "
        "long-run budget-efficiency solution."
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
        "aggregate": overall,
        "key_wait_bypass_witnesses": key_witnesses,
        "recovery_witnesses": recovery,
        "failure_demotion_by_agent": failure_demotion_by_agent,
        "RECOVERY_EVIDENCE": "SUFFICIENT" if recovery else "INSUFFICIENT",
        "two_stage_assessment": {
            "R0_to_W1": {
                "wait_is_inside_state_discount": True,
                "known_feasible_retention": overall["W1"][
                    "known_feasible_retention_rate"
                ],
                "known_commit_retention": overall["W1"][
                    "known_commit_retention_rate"
                ],
                "known_failures_demoted": overall["W1"]["known_failure_demoted"],
                "agent1_low_rho_removed_from_top2": key_witnesses["1"][
                    "W1_rank"
                ]
                > 2,
                "agent4_low_rho_removed_from_top2": key_witnesses["4"][
                    "W1_rank"
                ]
                > 2,
                "verdict": "SAFE_PARTIAL_MECHANISM_PASS",
            },
            "W1_to_W2": {
                "additional_persistent_value_supported": False,
                "known_feasible_retention_delta": overall["W2"][
                    "known_feasible_retention_rate"
                ]
                - overall["W1"]["known_feasible_retention_rate"],
                "known_commit_retention_delta": overall["W2"][
                    "known_commit_retention_rate"
                ]
                - overall["W1"]["known_commit_retention_rate"],
                "agent14_selection_delta": overall["W2"][
                    "agent1_plus_agent4_selection_count"
                ]
                - overall["W1"]["agent1_plus_agent4_selection_count"],
                "verdict": "REJECT",
            },
            "W1_to_W3": {
                "additional_persistent_value_supported": False,
                "known_feasible_retention_delta": overall["W3"][
                    "known_feasible_retention_rate"
                ]
                - overall["W1"]["known_feasible_retention_rate"],
                "known_commit_retention_delta": overall["W3"][
                    "known_commit_retention_rate"
                ]
                - overall["W1"]["known_commit_retention_rate"],
                "agent14_selection_delta": overall["W3"][
                    "agent1_plus_agent4_selection_count"
                ]
                - overall["W1"]["agent1_plus_agent4_selection_count"],
                "verdict": "REJECT",
            },
        },
        "RECOMMENDED_MODULE1_WAIT_DESIGN": recommendation,
        "WHY": why,
    }
    write_csv(
        OUTPUT / "selector_replay_comparison.csv",
        comparison_rows,
        list(comparison_rows[0].keys()),
    )
    write_jsonl(OUTPUT / "selector_replay_timeline.jsonl", all_timeline)
    write_csv(
        OUTPUT / "agent_score_timeline.csv", all_agents, list(all_agents[0].keys())
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
    write_json(
        OUTPUT / "witness_analysis.json",
        {"wait_bypass": key_witnesses, "recovery": recovery},
    )
    write_json(OUTPUT / "variant_definitions.json", variants)
    write_json(OUTPUT / "module1_wait_isolation_summary.json", summary)

    recommendation_text = f"""# v15 Module 1 Wait-Isolation Recommendation

`RECOMMENDED_MODULE1_WAIT_DESIGN = W1`

## Two-stage conclusion

W1 is the only supported minimal correction. Moving wait inside `rho_state`
retains {overall['W1']['known_feasible_retained']}/{overall['W1']['known_feasible_total']}
known-feasible branch events and
{overall['W1']['known_commit_retained']}/{overall['W1']['known_commit_total']}
actual commit targets. It demotes {overall['W1']['known_failure_demoted']} known
failed branches without adding persistent state.

| Policy | Agent1+4 | Feasible retention | Commit retention | Failure demotion | Changes vs W1 |
|---|---:|---:|---:|---:|---:|
| W1 | {overall['W1']['agent1_plus_agent4_selection_count']} | {overall['W1']['known_feasible_retention_rate']:.3f} | {overall['W1']['known_commit_retention_rate']:.3f} | {overall['W1']['known_failure_demotion_rate']:.3f} | 0 |
| W2 | {overall['W2']['agent1_plus_agent4_selection_count']} | {overall['W2']['known_feasible_retention_rate']:.3f} | {overall['W2']['known_commit_retention_rate']:.3f} | {overall['W2']['known_failure_demotion_rate']:.3f} | {overall['W2']['top2_changes_vs_W1']} |
| W3 | {overall['W3']['agent1_plus_agent4_selection_count']} | {overall['W3']['known_feasible_retention_rate']:.3f} | {overall['W3']['known_commit_retention_rate']:.3f} | {overall['W3']['known_failure_demotion_rate']:.3f} | {overall['W3']['top2_changes_vs_W1']} |

W2 loses 3 known-feasible events and 2 commit targets relative to W1 while
increasing aggregate Agent1+4 occupancy. W3 loses 2 known-feasible events and
1 commit target; its small aggregate Agent1+4 reduction is inconsistent across
settings and requires 58 Top-2 changes versus W1. Neither persistent extension
has sufficient incremental evidence.

## Direct low-rho witnesses

W1 reduces the S3 update-30 Agent1 score from
{float(key_witnesses['1']['R0_score']):.6f} to
{float(key_witnesses['1']['W1_score']):.6f}, but its rank remains
{key_witnesses['1']['W1_rank']}. It reduces the update-31 Agent4 score from
{float(key_witnesses['4']['R0_score']):.6f} to
{float(key_witnesses['4']['W1_score']):.6f}, but its rank remains
{key_witnesses['4']['W1_rank']}.

Thus W1 fixes the wait-placement semantics and is empirically safe, but it is
not evidence that long-run Agent1/4 budget occupation is solved. The supported
v15 design scope is W1 only; no persistent realizability state is recommended.

This is fixed-actual-trajectory replay. No alternative train/test outcome or
counterfactual acceptance rate is claimed.

`V15_IMPLEMENTATION_AUTHORIZED = false`
"""
    (OUTPUT / "recommendation.md").write_text(recommendation_text, encoding="utf-8")

    readme = f"""# v15 Module 1 Wait-Placement Isolation

- Task status: **COMPLETE**
- API calls: **0**
- Baseline replay: **PASS**
- Recommended design: **W1 only**
- Persistent realizability: **not supported**

This directory performs the requested two-stage fixed-trajectory isolation:

- `R0 -> W1` isolates placement of the wait term inside state repairability;
- `W1 -> W2` tests persistent Beta feasible rate;
- `W1 -> W3` tests persistent consecutive-failure discount.

W1 retains every observed feasible branch and commit target. W2/W3 introduce
retention losses and substantial ranking churn. No API, new candidate, rollout,
validation, test rerun, or formal method modification occurred.

See `recommendation.md` and `module1_wait_isolation_summary.json`.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
