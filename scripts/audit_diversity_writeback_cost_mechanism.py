"""Zero-API audit of Diversity write-back cost and coverage-to-vote mechanics.

The script intentionally reads historical artifacts only.  Its most important
accounting rule is conservative: `llm_calls.jsonl` has no update/candidate
identifier, so Solver usage is reported as an empirical-rollout total and is
never silently assigned to individual candidates, winners, or stages.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "vote_aligned_confirmatory_seed76_77_v1"
COST_REPORT = ROOT / "reports" / "cross_method_cost_accounting_20260907"
OUT_DEFAULT = ROOT / "reports" / "diversity_writeback_cost_mechanism_audit_20260908"


def rows_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    values = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_num(value: Any) -> int:
    return int(value or 0)


def percentage(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def llm_totals(path: Path) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows_jsonl(path):
        role = str(row.get("client_role") or row.get("role") or "unknown")
        bucket = totals[role]
        bucket["logical_calls"] += 1
        bucket["provider_calls"] += 1
        bucket["successful_calls"] += int(bool(row.get("success")))
        bucket["input_tokens"] += safe_num(row.get("prompt_tokens"))
        bucket["output_tokens"] += safe_num(row.get("completion_tokens"))
        bucket["total_tokens"] += safe_num(row.get("total_tokens"))
    return {key: dict(value) for key, value in totals.items()}


def commit_class(metric: dict[str, Any]) -> str:
    """Frozen mutually-exclusive order: C4 precedes C3 per the task text."""
    mean_delta = metric["mean_member_delta"]
    vote_delta = metric["vote_delta"]
    oracle_delta = metric["oracle_delta"]
    net_member = metric["net_correct_member_votes"]
    if mean_delta > 0 and vote_delta >= 0 and net_member > 0:
        return "C1_BROAD_GAIN"
    if oracle_delta > 0 and vote_delta > 0:
        return "C2_USEFUL_SPECIALIZATION"
    if oracle_delta > 0 and mean_delta < 0:
        return "C4_COVERAGE_WITH_COLLATERAL"
    if oracle_delta > 0 and vote_delta == 0:
        return "C3_COVERAGE_ONLY"
    if vote_delta > 0 and oracle_delta <= 0:
        return "C5_VOTE_ONLY"
    if vote_delta == 0 and oracle_delta == 0 and mean_delta == 0:
        return "C6_NEUTRAL"
    return "C7_REGRESSIVE_OR_OTHER"


def depth_bucket(g: int) -> str:
    return "G>=3" if g >= 3 else f"G={g}"


def analyze_seed(seed: int) -> dict[str, Any]:
    directory = RUN_ROOT / f"seed{seed}" / "P1_SHADOW_VOTE_ALIGNED_GENERIC"
    candidates = rows_jsonl(directory / "candidate_decisions.jsonl")
    transitions = rows_jsonl(directory / "g_transition_audit.jsonl")
    dynamics = rows_jsonl(directory / "training_dynamics.jsonl")
    priorities = rows_jsonl(directory / "target_priority_audit.jsonl")
    calls = llm_totals(directory / "llm_calls.jsonl")
    transition_by_update: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in transitions:
        transition_by_update[safe_num(row["update_index"])].append(row)

    opportunities: list[dict[str, Any]] = []
    funnel = Counter()
    for event in candidates:
        f = event["funnel"]
        update = safe_num(event["update_index"])
        generated = safe_num(f.get("raw_candidate_count"))
        valid = safe_num(f.get("valid_candidate_count"))
        feasible = safe_num(f.get("constraint_feasible"))
        committed = int(bool(event.get("writeback_approved")))
        funnel.update({"opportunities": 1, "generated": generated, "valid": valid,
                       "feasible": feasible, "candidate_winners": int(feasible > 0),
                       "committed": committed})
        opportunities.append({
            "seed": seed, "update_index": update, "target_member": event.get("target_agent_id"),
            "parent_state_id": event.get("parent_team_hash"), "num_generated_candidates": generated,
            "num_valid_candidates": valid, "num_feasible_candidates": feasible,
            "num_full_rollout_candidates": safe_num(f.get("stage_b_evaluated")), "committed": committed,
            "committed_candidate_id": event.get("accepted_prompt_hash"),
            "diagnostic_solver_calls": None, "proposal_optimizer_calls": None,
            "candidate_generation_optimizer_calls": None, "member_rollout_solver_calls": None,
            "team_rollout_solver_calls": None, "safety_eval_solver_calls": None,
            "shadow_eval_solver_calls": None, "post_commit_eval_solver_calls": None,
            "unknown_solver_calls": None, "unknown_optimizer_calls": None,
            "total_solver_calls": None, "total_optimizer_calls": None,
            "solver_input_tokens": None, "solver_output_tokens": None,
            "optimizer_input_tokens": None, "optimizer_output_tokens": None, "total_tokens": None,
            "attribution_status": "NOT_RECOVERABLE_PER_OPPORTUNITY_ZERO_API",
        })

    commit_metrics: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    for update, rows in sorted(transition_by_update.items()):
        n = len(rows)
        assert n == 100, f"seed {seed} update {update} expected 100 rows, got {n}"
        new_coverage = sum(r["G_before"] == 0 and r["G_after"] > 0 for r in rows)
        lost_coverage = sum(r["G_before"] > 0 and r["G_after"] == 0 for r in rows)
        new_member = sum((not r["target_correct_before"]) and r["target_correct_after"] for r in rows)
        lost_member = sum(r["target_correct_before"] and (not r["target_correct_after"]) for r in rows)
        new_vote = sum((not r["vote_correct_before"]) and r["vote_correct_after"] for r in rows)
        lost_vote = sum(r["vote_correct_before"] and (not r["vote_correct_after"]) for r in rows)
        target_delta_count = sum(int(r["target_correct_after"]) - int(r["target_correct_before"]) for r in rows)
        g_delta = sum(safe_num(r["G_after"]) - safe_num(r["G_before"]) for r in rows)
        metric = {
            "seed": seed, "update_index": update, "target_member": rows[0]["target_agent_id"],
            "row_count": n, "target_member_delta": target_delta_count / n,
            "mean_member_delta": g_delta / (n * 5), "vote_delta": (new_vote - lost_vote) / n,
            "oracle_delta": (new_coverage - lost_coverage) / n,
            "new_oracle_covered_cases": new_coverage, "lost_oracle_covered_cases": lost_coverage,
            "new_correct_member_votes": new_member, "lost_correct_member_votes": lost_member,
            "net_correct_member_votes": new_member - lost_member,
            "new_majority_correct_cases": new_vote, "lost_majority_correct_cases": lost_vote,
            "net_majority_cases": new_vote - lost_vote,
            "transition_source": "optimization_time_train100",
        }
        metric["commit_class"] = commit_class(metric)
        commit_metrics.append(metric)
        for r in rows:
            before, after = safe_num(r["G_before"]), safe_num(r["G_after"])
            if before == 0 and after > 0:
                kind = "newly_covered"
            elif before >= 3 and after <= 2:
                kind = "lost_deep_coverage"
            elif before <= 2 and after >= 3:
                kind = "deepened_to_coalition"
            elif before > 0 and after == 0:
                kind = "lost_coverage"
            else:
                kind = "other"
            depth_rows.append({"seed": seed, "update_index": update, "case_hash": r["question_hash"],
                               "G_before": before, "G_after": after, "before_bucket": depth_bucket(before),
                               "after_bucket": depth_bucket(after), "transition_kind": kind,
                               "vote_before": int(bool(r["vote_correct_before"])),
                               "vote_after": int(bool(r["vote_correct_after"]))})

    long_rows: list[dict[str, Any]] = []
    for row in dynamics:
        if row["update_index"] == -1 or row.get("state_changed"):
            counts = row.get("per_agent_correct_counts") or []
            long_rows.append({"seed": seed, "update_index": row["update_index"],
                              "commit_index": sum(int(x.get("state_changed", False)) for x in dynamics if x["update_index"] <= row["update_index"]),
                              "team_vote_accuracy": row.get("team_vote_accuracy"),
                              "mean_member_accuracy": row.get("mean_member_accuracy"),
                              "oracle_accuracy": row.get("oracle_accuracy"),
                              "correct_member_votes": sum(counts), "state_changed": int(bool(row.get("state_changed")))})

    # Responsibility signal is intentionally descriptive, using frozen lane counts.
    pri_by_update = {safe_num(row["update_index"]): row for row in priorities}
    assoc: list[dict[str, Any]] = []
    metric_by_update = {row["update_index"]: row for row in commit_metrics}
    for update, priority in pri_by_update.items():
        for item in priority.get("priorities", []):
            assoc.append({"seed": seed, "update_index": update, "agent_id": item.get("agent_id"),
                          "selected": int(bool(item.get("selected"))), "direct_flip": safe_num(item.get("direct_flip")),
                          "near_margin": safe_num(item.get("near_margin")), "pure_coverage": safe_num(item.get("pure_coverage")),
                          "committed": int(update in metric_by_update),
                          "target_member_delta": (metric_by_update.get(update) or {}).get("target_member_delta"),
                          "mean_member_delta": (metric_by_update.get(update) or {}).get("mean_member_delta"),
                          "vote_delta": (metric_by_update.get(update) or {}).get("vote_delta"),
                          "association_scope": "ASSOCIATIONAL"})
    return {"opportunities": opportunities, "funnel": funnel, "calls": calls, "commits": commit_metrics,
            "depth": depth_rows, "longitudinal": long_rows, "associations": assoc, "directory": directory}


def simple_svg(path: Path, title: str, series: dict[str, list[float]]) -> None:
    """Minimal dependency-free plot; values are aggregate, no private text."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    width, height, left, bottom = 720, 360, 55, 300
    values = [x for points in series.values() for x in points]
    low, high = (min(values), max(values)) if values else (0, 1)
    if low == high: high = low + 1
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>', f'<text x="20" y="25" font-size="16">{title}</text>',
             f'<line x1="{left}" y1="40" x2="{left}" y2="{bottom}" stroke="black"/>',
             f'<line x1="{left}" y1="{bottom}" x2="690" y2="{bottom}" stroke="black"/>']
    for number, (name, points) in enumerate(series.items()):
        if not points: continue
        coords=[]
        for index, value in enumerate(points):
            x=left + index * (635 / max(1, len(points)-1)); y=bottom - (value-low)/(high-low)*240
            coords.append(f"{x:.1f},{y:.1f}")
        color=colors[number % len(colors)]
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(coords)}"/>')
        lines.append(f'<text x="{450}" y="{55+number*18}" fill="{color}" font-size="12">{name}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def cost_rows() -> list[dict[str, str]]:
    with (COST_REPORT / "cross_method_optimization_cost.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    seed_data = [analyze_seed(76), analyze_seed(77)]
    all_opps = [x for data in seed_data for x in data["opportunities"]]
    all_commits = [x for data in seed_data for x in data["commits"]]
    all_depth = [x for data in seed_data for x in data["depth"]]
    all_long = [x for data in seed_data for x in data["longitudinal"]]
    all_assoc = [x for data in seed_data for x in data["associations"]]
    cost = cost_rows()
    diversity_cost = [x for x in cost if x["method"] == "Diversity" and x["seed"] in {"76", "77"}]
    total_solver_calls = sum(int(x["solver_calls"]) for x in diversity_cost)
    total_optimizer_calls = sum(int(x["optimizer_calls"]) for x in diversity_cost)
    total_tokens = sum(int(x["total_tokens"]) for x in diversity_cost)
    generated = sum(data["funnel"]["generated"] for data in seed_data)
    valid = sum(data["funnel"]["valid"] for data in seed_data)
    feasible = sum(data["funnel"]["feasible"] for data in seed_data)
    committed = len(all_commits)
    assert committed == 13 and total_solver_calls == 15209 and total_optimizer_calls == 429 and total_tokens == 10666948

    write_csv(out / "diversity_opportunity_costs.csv", all_opps, list(all_opps[0]))
    funnel_rows=[]
    for data in seed_data + [{"funnel": Counter({"opportunities":33,"generated":generated,"valid":valid,"feasible":feasible,"candidate_winners":sum(d["funnel"]["candidate_winners"] for d in seed_data),"committed":committed})}]:
        label = "combined" if "directory" not in data else data["directory"].parts[-3]
        base = data["funnel"]["opportunities"]
        for stage in ["opportunities","generated","valid","feasible","candidate_winners","committed"]:
            count=data["funnel"][stage]
            funnel_rows.append({"seed": label, "stage":stage,"count":count,"survival_from_opportunities":percentage(count,base),
                                "solver_calls_cumulative":None,"tokens_cumulative":None,
                                "attribution_status":"UNATTRIBUTABLE_BY_STAGE_FROM_UNINDEXED_SOLVER_LEDGER"})
    write_csv(out / "diversity_candidate_funnel.csv", funnel_rows, list(funnel_rows[0]))
    role_totals=Counter()
    for data in seed_data:
        for role, values in data["calls"].items():
            for key, value in values.items(): role_totals[(role,key)] += value
    stages=[
        {"stage":"A_responsibility_target_diagnostics","provider_calls":0,"total_tokens":0,"status":"DETERMINISTIC_LOCAL"},
        {"stage":"B_teacher_critic_planning","provider_calls":role_totals[("optimizer","successful_calls")],"total_tokens":role_totals[("optimizer","total_tokens")],"status":"MIXED_WITH_CANDIDATE_GENERATION_UNINDEXED_BY_ROLE"},
        {"stage":"C_student_candidate_generation","provider_calls":None,"total_tokens":None,"status":"NOT_SEPARABLE_FROM_OPTIMIZER_CLIENT_LEDGER"},
        {"stage":"D_parsing_validity","provider_calls":0,"total_tokens":0,"status":"DETERMINISTIC_LOCAL"},
        {"stage":"E_to_H_empirical_member_team_safety_shadow","provider_calls":total_solver_calls,"total_tokens":sum(int(x["input_tokens"])+int(x["output_tokens"]) for x in diversity_cost)-role_totals[("optimizer","total_tokens")],"status":"SOLVER_LEDGER_UNINDEXED_BY_EVALUATION_STAGE"},
        {"stage":"I_to_K_ranking_commit_refresh","provider_calls":0,"total_tokens":0,"status":"DETERMINISTIC_LOCAL"},]
    write_csv(out / "diversity_cost_by_stage.csv", stages, list(stages[0]))
    wasted=[
        {"bucket":"A_committed_winner_compute","solver_calls":None,"tokens":None,"status":"NOT_RECOVERABLE_UNINDEXED_LEDGER"},
        {"bucket":"B_nonwinning_feasible_candidate_compute","solver_calls":None,"tokens":None,"status":"NOT_RECOVERABLE_UNINDEXED_LEDGER"},
        {"bucket":"C_valid_infeasible_candidate_compute","solver_calls":None,"tokens":None,"status":"NOT_RECOVERABLE_UNINDEXED_LEDGER"},
        {"bucket":"D_invalid_candidate_compute","solver_calls":None,"tokens":None,"status":"NOT_RECOVERABLE_UNINDEXED_LEDGER"},
        {"bucket":"E_no_commit_opportunity_compute","solver_calls":None,"tokens":None,"status":"NOT_RECOVERABLE_UNINDEXED_LEDGER"},
        {"bucket":"F_duplicated_repeated_evaluation_compute","solver_calls":None,"tokens":None,"status":"NOT_RECOVERABLE_UNINDEXED_LEDGER"},
        {"bucket":"G_refresh_bookkeeping_compute","solver_calls":0,"tokens":0,"status":"DETERMINISTIC_LOCAL"},]
    write_csv(out / "diversity_wasted_compute.csv", wasted, list(wasted[0]))
    write_csv(out / "diversity_commit_utility.csv", all_commits, list(all_commits[0]))
    classes=[{"commit_class": key,"count":value,"fraction":percentage(value,committed)} for key,value in sorted(Counter(x["commit_class"] for x in all_commits).items())]
    write_csv(out / "diversity_commit_classes.csv", classes, list(classes[0]))
    write_csv(out / "diversity_coverage_depth_transitions.csv", all_depth, list(all_depth[0]))
    write_csv(out / "diversity_longitudinal_state_metrics.csv", all_long, list(all_long[0]))
    write_csv(out / "responsibility_associations.csv", all_assoc, list(all_assoc[0]))
    pressure=[
      {"quantity":"target gain","generation":"yes_responsibility_goal","feasibility":"yes","ranking":"yes","hard_constraint":"yes","soft_signal":"yes","code_evidence":"candidate_selection.py:61-127"},
      {"quantity":"Vote","generation":"indirect_lane_context","feasibility":"yes_nonregression","ranking":"yes","hard_constraint":"yes","soft_signal":"yes","code_evidence":"candidate_selection.py:61-127"},
      {"quantity":"MeanMember","generation":"no","feasibility":"derived_only","ranking":"yes_stage_A_total_gain","hard_constraint":"no","soft_signal":"yes","code_evidence":"candidate_selection.py:61-127"},
      {"quantity":"Oracle","generation":"residual_context","feasibility":"no","ranking":"no_direct_key","hard_constraint":"no","soft_signal":"indirect" ,"code_evidence":"candidate_selection.py:61-127"},
      {"quantity":"new coverage","generation":"yes","feasibility":"no","ranking":"assigned_residual_proxy","hard_constraint":"no","soft_signal":"yes","code_evidence":"candidate_selection.py:61-127"},
      {"quantity":"broad preservation","generation":"limited_case_context","feasibility":"terminal_invalid_only","ranking":"no_direct_key","hard_constraint":"no","soft_signal":"indirect","code_evidence":"candidate_selection.py:61-127"},
      {"quantity":"coalition depth","generation":"diagnostic_lane","feasibility":"no","ranking":"no_direct_key","hard_constraint":"no","soft_signal":"indirect","code_evidence":"candidate_selection.py:61-127"},
      {"quantity":"member transfer","generation":"no","feasibility":"no","ranking":"no","hard_constraint":"no","soft_signal":"no","code_evidence":"candidate_selection.py:61-127"},]
    write_csv(out / "selection_pressure_matrix.csv", pressure, list(pressure[0]))

    # Cross-method search structures use exact preceding accounting report; no sums are recomputed from prose.
    ge=[x for x in cost if x["method"]=="GEPA"]
    ma=[x for x in cost if x["method"]=="MARS"]
    gepa=[]
    mars=[]
    for row in ge:
        proposals=int(row["units"]); calls=int(row["solver_calls"]); tokens=int(row["total_tokens"])
        unique={"75":14,"76":4,"77":14}[row["seed"]]
        gepa.append({"seed":row["seed"],"proposals":proposals,"unique_candidates":unique,"solver_calls":calls,"optimizer_calls":int(row["optimizer_calls"]),"total_tokens":tokens,"solver_calls_per_proposal":calls/proposals,"solver_calls_per_unique_candidate":calls/unique,"tokens_per_proposal":tokens/proposals,"tokens_per_unique_candidate":tokens/unique,"topology":"candidate_to_broad_task_evaluation"})
    for row in ma:
        units=int(row["units"]); mars.append({"seed":row["seed"],"iterations":units,"solver_calls":int(row["solver_calls"]),"optimizer_calls":int(row["optimizer_calls"]),"total_tokens":int(row["total_tokens"]),"solver_calls_per_iteration":int(row["solver_calls"])/units,"optimizer_calls_per_iteration":int(row["optimizer_calls"])/units,"tokens_per_iteration":int(row["total_tokens"])/units,"topology":"planner_teacher_critic_student_to_task_evaluation"})
    write_csv(out / "gepa_search_funnel.csv", gepa, list(gepa[0]))
    write_csv(out / "mars_search_funnel.csv", mars, list(mars[0]))
    architecture=[
      {"method":"MARS","optimization_object":"single_prompt","proposal_mechanism":"multi_agent_meta_pipeline","member_specific_objective":"no","full_team_reevaluation":"no","solver_token_share":"0.6996","optimizer_token_share":"0.3004","optimization_tokens":"2951027/3_seeds","solver_calls":"2000","main_compute_bottleneck":"meta_reasoning"},
      {"method":"GEPA","optimization_object":"single_prompt","proposal_mechanism":"reflection","member_specific_objective":"no","full_team_reevaluation":"no","solver_token_share":"0.8815","optimizer_token_share":"0.1185","optimization_tokens":"3754403/3_seeds","solver_calls":"1996","main_compute_bottleneck":"task_evaluation"},
      {"method":"Diversity","optimization_object":"five_member_team","proposal_mechanism":"responsibility_conditioned","member_specific_objective":"yes","full_team_reevaluation":"yes","solver_token_share":"0.9355","optimizer_token_share":"0.0645","optimization_tokens":"10666948/Seeds76_77_exact","solver_calls":"15209","main_compute_bottleneck":"empirical_team_rollout_unindexed"}]
    write_csv(out / "cross_method_architecture.csv", architecture, list(architecture[0]))
    metrics=[]
    for method, rows in (("MARS",ma),("GEPA",ge),("Diversity",diversity_cost)):
        metrics.append({"method":method,"seeds":"76-77" if method=="Diversity" else "75-77","solver_calls":sum(int(x["solver_calls"]) for x in rows),"optimizer_calls":sum(int(x["optimizer_calls"]) for x in rows),"total_tokens":sum(int(x["total_tokens"]) for x in rows),"nominal_unit_warning":"NOT_DIRECTLY_COMPARABLE"})
    write_csv(out / "cross_method_cost_metrics.csv", metrics, list(metrics[0]))
    retrospective=[]
    for policy in ["P1_valid_only_pruning","P2_feasibility_early_stop","P3_one_stage_team_evaluation","P4_top_k_expensive_rollout"]:
        retrospective.append({"policy":policy,"solver_calls_saved":None,"tokens_saved":None,"historical_commits_retained":None,"historical_winning_candidates_lost":None,"status":"NOT_EVALUABLE_ZERO_API_UNINDEXED_SOLVER_LEDGER","interpretation":"RETROSPECTIVE_COST_LOWER_BOUND_ONLY"})
    write_csv(out / "retrospective_cost_counterfactuals.csv", retrospective, list(retrospective[0]))
    # Final common-contract replay is a separate frozen evaluation provenance;
    # never merge it with Train100 transition utility.
    common_inputs = [
        ROOT / "runs" / "common_solver_contract_v1_replay_20260906" / "summary_private.json",
        ROOT / "runs" / "common_solver_contract_v1_seed77_replay_20260907" / "summary_private.json",
    ]
    common_rows = []
    for path in common_inputs:
        summary = json.loads(path.read_text(encoding="utf-8"))
        fallback_seed = 76 if "20260906" in str(path) else 77
        for state in summary["states"]:
            if state["source_repo"] in {"COMMON", "Diversity"}:
                common_rows.append({"contract_id": summary["contract_id"], "split": summary["split"],
                                    "source_seed": state["source_seed"] or fallback_seed, "state_id": state["state_id"],
                                    "source_repo": state["source_repo"], "vote_accuracy": state["vote_accuracy"],
                                    "mean_member_accuracy": sum(state["individual_member_accuracy"]) / state["member_count"],
                                    "oracle_accuracy": state["oracle_accuracy"],
                                    "evaluation_role": "FINAL_STATE_COMMON_REPLAY"})
    write_csv(out / "common_contract_final_state_results.csv", common_rows, list(common_rows[0]))
    stage_mapping={"A":"deterministic responsibility and target diagnostics","B":"teacher/critic optimizer calls; role-level but unindexed","C":"student candidate generation mixed in optimizer ledger","D":"local parsing/validity","E-H":"empirical Solver evaluations, unindexed by candidate/update/stage","I-K":"local ranking/commit/refresh"}
    write_json(out / "stage_mapping.json", stage_mapping)
    hypotheses={"H_COST":"SUPPORTED","H_PROPOSAL":"PARTIALLY_SUPPORTED","H_SELECTION":"NOT_EVALUABLE","H_ACCUMULATION":"PARTIALLY_SUPPORTED","H_OBJECTIVE":"PARTIALLY_SUPPORTED","classifier_version":"writeback_cost_utility_v1","api_calls":0,"test_calls":0}
    write_json(out / "frozen_hypotheses.json", hypotheses)
    classification={"primary":"H_COST","secondary":"H_OBJECTIVE","reason":"93.55% of exact Seed76-77 optimization tokens are Solver-side while code gives no direct hard/ranking key to coalition depth or transfer; causal selection attribution is not available from this observational audit.","limits":["Solver ledger lacks update/candidate/stage identifiers.","MARS/GEPA use different historical optimization contracts.","No counterfactual candidates were evaluated."]}
    write_json(out / "classification_results.json", classification)
    inventory=[]
    for data in seed_data:
        for path in [data["directory"] / x for x in ["candidate_decisions.jsonl","g_transition_audit.jsonl","training_dynamics.jsonl","llm_calls.jsonl","target_priority_audit.jsonl"]]:
            inventory.append({"repository":"multi_agent_diversity","relative_path":str(path.relative_to(ROOT)).replace("\\","/"),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"role":"read_only_primary_source"})
    inventory.append({"repository":"multi_agent_diversity","relative_path":"reports/cross_method_cost_accounting_20260907/cross_method_optimization_cost.csv","sha256":hashlib.sha256((COST_REPORT/"cross_method_optimization_cost.csv").read_bytes()).hexdigest(),"role":"previous_raw-ledger_reconciled_source"})
    for path in common_inputs:
        inventory.append({"repository":"multi_agent_diversity","relative_path":str(path.relative_to(ROOT)).replace("\\\\","/"),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"role":"frozen_final_state_common_contract_replay"})
    write_json(out / "source_inventory.json", inventory)
    fact={"api_calls":0,"test_calls":0,"historical_artifacts_modified":0,"seeds":[76,77],"opportunities":33,"commits":13,"solver_calls":total_solver_calls,"optimizer_calls":total_optimizer_calls,"optimization_tokens":total_tokens,"tokens_per_commit":total_tokens/committed,"generated_candidates":generated,"valid_candidates":valid,"feasible_candidates":feasible,"assertions":"PASS"}
    write_json(out / "fact_assertions.json", fact)
    write_json(out / "provenance.json", {"method":"read_only historical audit","source_priority":"raw trajectory artifacts then preceding raw-ledger-reconciled accounting report","test_accessed":False,"api_called":False,"seed75_usage":"excluded_not_reconstructed"})
    # Simple figures with aggregated values only.
    for seed in (76,77):
        values=[r for r in all_long if r["seed"]==seed]
        simple_svg(out / f"figure_a_trajectory_seed{seed}.svg", f"Seed {seed}: trajectory", {"Vote":[r["team_vote_accuracy"] for r in values],"MeanMember":[r["mean_member_accuracy"] for r in values],"Oracle":[r["oracle_accuracy"] for r in values]})
    simple_svg(out / "figure_b_compute_funnel.svg", "Diversity Seed76-77 funnel (counts)", {"count":[33,generated,valid,feasible,committed]})
    depth_counts=Counter((r["transition_kind"],r["after_bucket"]) for r in all_depth)
    simple_svg(out / "figure_c_coverage_depth.svg", "Coverage depth transitions", {"new_G1_G2_G3+": [depth_counts[("newly_covered",x)] for x in ["G=1","G=2","G>=3"]],"lost_deep": [depth_counts[("lost_deep_coverage",x)] for x in ["G=1","G=2","G>=3"]]})
    readme=f"""# Diversity Write-back Cost & Coverage-to-Vote Failure Audit\n\nZero-API, zero-Test50, read-only audit of formal Seed76–77 P1 trajectories. Seed75 raw usage remains excluded because it is not fully recoverable.\n\n## Table A — Cost funnel\n\n| Opportunities | Generated | Valid | Feasible | Commits | Solver calls | Optimizer calls | Tokens | Tokens / commit |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n| 33 | {generated} | {valid} | {feasible} | 13 | 15,209 | 429 | 10,666,948 | {total_tokens/13:,.0f} |\n\n## Table B — Where compute goes\n\nThe exact role ledger assigns 93.55% of tokens to Solver-side empirical evaluation and 6.45% to optimizer calls. The Solver ledger has no update/candidate/stage ID, so winner/discarded and member/team/shadow sub-stages are deliberately `NOT_RECOVERABLE` rather than estimated.\n\n## Table C — What committed updates buy\n\n| Class | Count |\n|---|---:|\n""" + "\n".join(f"| {r['commit_class']} | {r['count']} |" for r in classes) + f"""\n\nCommit-level outcomes are in `diversity_commit_utility.csv`; they are optimization-time Train100 transitions and are not combined with common-contract final replay metrics.\n\n## Table D — Architecture and cost\n\n| Method | Optimization-only tokens | Solver calls | Main topology |\n|---|---:|---:|---|\n| MARS | 2,951,027 (3 seeds) | 2,000 | meta pipeline then task evaluation |\n| GEPA | 3,754,403 (3 seeds) | 1,996 | candidate then broad task evaluation |\n| Diversity | 10,666,948 (Seeds76–77) | 15,209 | member/team empirical rollout |\n\n![Figure A Seed76](figure_a_trajectory_seed76.svg)\n![Figure A Seed77](figure_a_trajectory_seed77.svg)\n![Figure B](figure_b_compute_funnel.svg)\n![Figure C](figure_c_coverage_depth.svg)\n\n## Answers\n\n**Q1 — Cost.** The exact 820,535 tokens per commit is rollout-heavy: 93.55% of Seed76–77 tokens are Solver-side. Candidate multiplicity is {generated/33:.2f} generated candidates/opportunity, with {total_solver_calls/generated:.1f} Solver calls/generated candidate and {total_solver_calls/13:.1f} Solver calls/commit. The ledger cannot identify the split among member, team, safety, and shadow passes, so it does not support a numerical winner-versus-discarded decomposition.\n\n**Q2 — Coverage versus plurality.** Responsibility and assigned residual evidence enter generation, while Common-Safe gives hard target/Vote non-regression. Mean-member total gain appears in Stage-A ranking, but Oracle, coalition depth, transfer, and broad preservation lack an equivalent direct hard/ranking key. This is code-grounded pressure asymmetry, not a causal proof of the observed final-state pattern.\n\n**Q3/Q4 — structural comparison.** GEPA has broad candidate-to-task evaluation and 88.15% Solver token share; MARS is meta-heavier (30.04% optimizer-token share). Diversity is more rollout-heavy (93.55% Solver token share) because five-member fixed-peer safety must be empirically checked. Historical contracts differ, so this is not a formal efficacy ranking.\n\n**Primary diagnosis:** H_COST supported: empirical Solver validation dominates observed cost. **Secondary diagnosis:** H_OBJECTIVE partially supported: selection pressure has no direct coalition-depth/transfer key. H_SELECTION is not evaluable without frozen alternative validation.\n\n**Smallest justified next experiment:** a frozen fixed-parent comparison that holds candidate generation constant and compares the historical Common-Safe winner with a coalition-depth-aware *analysis-only* ranking, with no trajectory write-back.\n"""
    common_table = """\n## Frozen final-state common-contract replay (separate provenance)\n\n| Seed | State | Vote | MeanMember | Oracle |\n|---:|---|---:|---:|---:|\n| 76 | P0 | .60 | .600 | .60 |\n| 76 | Diversity P1 final | .66 | .588 | .92 |\n| 77 | P0 | .62 | .620 | .62 |\n| 77 | Diversity P1 final | .58 | .580 | .86 |\n\nThe row-level aggregates are in `common_contract_final_state_results.csv`; this is separate provenance from Train100 transition utility.\n"""
    readme = readme.replace("\n## Table D — Architecture and cost", common_table + "\n## Table D — Architecture and cost")
    (out / "README.md").write_text(readme, encoding="utf-8")
    # Sanitation and manifest after all text artifacts exist.
    # SVG namespace URLs are structural markup, not provider endpoints.  Scan
    # textual report content for endpoint/credential material without treating
    # that standard namespace as an endpoint leak.
    prohibited=("D:\\\\", "https://", "FINAL_ANSWER:", "api_key", "dasHSCOPE".lower())
    files=[]
    for path in sorted(out.iterdir()):
        if path.name in {"sha256_manifest.json", "sanitization_manifest.json"}: continue
        raw=path.read_bytes(); text=raw.decode("utf-8", errors="ignore").lower()
        assert not any(term.lower() in text for term in prohibited), f"sanitization term in {path.name}"
        files.append({"path":path.name,"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)})
    write_json(out / "sanitization_manifest.json", {"status":"PASS","prohibited_patterns":list(prohibited),"files_checked":len(files)})
    write_json(out / "sha256_manifest.json", {"files":files})
    print(json.dumps(fact, sort_keys=True))


if __name__ == "__main__":
    main()
