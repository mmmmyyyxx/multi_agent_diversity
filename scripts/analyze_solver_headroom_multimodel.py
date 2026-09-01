from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from scripts.solver_headroom_multimodel_support import (
    CANDIDATES, REPORT_ROOT, ROLE_MODEL, RUN_ROOT, entrants, phase_a_rows,
    read_json, run_dir, selected_generic, validation_dir, write_json,
)


def accepted(run: Path) -> int:
    path=run/"candidate_decisions.jsonl"
    if not path.exists():return 0
    return sum(bool(json.loads(line).get("accepted_prompt_hash")) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    if REPORT_ROOT.exists():raise SystemExit("fresh report root required")
    smoke=read_json(RUN_ROOT/"phase_a/availability_smoke_private.json")
    selection=read_json(RUN_ROOT/"static_selection_private.json")
    if smoke["gate"] != "PASS" or len(smoke["candidates"]) != 8:raise RuntimeError("Phase A gate")
    selected_keys={r["key"] for r in selected_generic()}
    rows=[]
    total_validation_calls=0
    for entry in entrants():
        key,model=entry["key"],entry["model"]
        static=read_json(validation_dir(key,"STATIC")/"validation_summary_private.json")
        total_validation_calls += static["provider_calls"]
        row={
            "key":key,"solver_model":model,"seed":65,
            "static_validation_vote_acc":static["vote_accuracy"],
            "static_validation_oracle_acc":static["oracle_accuracy"],
            "static_oracle_vote_gap":static["oracle_accuracy"]-static["vote_accuracy"],
            "static_member_accuracies":static["per_agent_accuracies"],
            "static_terminal_invalid_rate":static["terminal_invalid_count"]/static["resolved_request_count"] if static["resolved_request_count"] else 0.0,
            "advanced_to_generic":key in selected_keys,
            "generic_validation_vote_acc":None,"generic_validation_oracle_acc":None,
            "generic_minus_static_vote_delta":None,"generic_oracle_vote_gap":None,
            "generic_member_accuracies":None,"accepted_commits":None,
            "local_headroom_label":"NOT_EVALUATED",
        }
        generic_path=validation_dir(key,"GENERIC")/"validation_summary_private.json"
        if generic_path.exists():
            generic=read_json(generic_path);total_validation_calls += generic["provider_calls"]
            delta=generic["vote_accuracy"]-static["vote_accuracy"]
            gap=generic["oracle_accuracy"]-generic["vote_accuracy"]
            row.update({"generic_validation_vote_acc":generic["vote_accuracy"],"generic_validation_oracle_acc":generic["oracle_accuracy"],"generic_minus_static_vote_delta":delta,"generic_oracle_vote_gap":gap,"generic_member_accuracies":generic["per_agent_accuracies"],"accepted_commits":accepted(run_dir(key,"GENERIC")),"local_headroom_label":"SUPPORTED_LOCAL" if delta >= 0.04 and gap >= 0.08 else "NO_LOCAL_HEADROOM_SIGNAL"})
        rows.append(row)
    supported=[r for r in rows if r["local_headroom_label"]=="SUPPORTED_LOCAL"]
    supported.sort(key=lambda r:(-r["generic_minus_static_vote_delta"],-r["generic_oracle_vote_gap"],r["static_validation_vote_acc"],next(i for i,(k,_) in enumerate(CANDIDATES) if k==r["key"])))
    chosen=supported[0] if supported else None
    availability={"gate":smoke["gate"],"candidates":[{"key":r["key"],"solver_model":r["model"],"listed":r["listed"],"smoke_attempted":r["smoke"]["attempted"],"smoke_success":r["smoke"]["success"],"status_code":r["smoke"]["status_code"],"error_type":r["smoke"]["error_type"],"static_eligible":r["static_eligible"]} for r in phase_a_rows()],"role_model":ROLE_MODEL,"test_calls":0}
    summary={"screening_version":"solver_headroom_multimodel_seed65_v1","seed":65,"role_model":ROLE_MODEL,"static_entrant_count":len(rows),"generic_selected_models":[r["model"] for r in selected_generic()],"qwen3_8b_anchor_generic_reported":any(r["key"]=="Q8" and r["generic_validation_vote_acc"] is not None for r in rows),"rows":rows,"provisional_decision":"SELECT_LOCAL" if chosen else "HOLD","provisional_solver":chosen["solver_model"] if chosen else "","validation_provider_calls":total_validation_calls,"smoke_successful_new_calls":smoke["successful_new_smoke_calls"],"test_evaluation_count":0,"full_method_run":False,"test_accessed":False,"seed66_status":"INTERRUPTED_EXCLUDED"}
    selection_out={"decision":summary["provisional_decision"],"selected_solver":summary["provisional_solver"],"single_seed_diagnostic":True,"criteria":{"vote_delta_at_least_0_04":True if chosen else False,"generic_oracle_vote_gap_at_least_0_08":True if chosen else False},"full_method_run":False,"test_accessed":False}
    REPORT_ROOT.mkdir(parents=True)
    write_json(REPORT_ROOT/"availability_smoke.json",availability);write_json(REPORT_ROOT/"summary.json",summary);write_json(REPORT_ROOT/"solver_selection.json",selection_out)
    cols=list(rows[0].keys())
    with (REPORT_ROOT/"per_seed_results.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        for row in rows:
            out=dict(row)
            for name in ("static_member_accuracies","generic_member_accuracies"):
                if out[name] is not None:out[name]=json.dumps(out[name],separators=(",",":"))
            writer.writerow(out)
    (REPORT_ROOT/"README.md").write_text("# Multi-model Solver Headroom Screening\n\n```text\nFULL_METHOD_NOT_RUN=true\nTEST_ACCESSED=false\n```\n\nThis amended screening used Seed65 only. All accessible models received a Static validation probe; at most three models advanced to Generic under the frozen Static-only gate. The completed qwen3-8b Seed65 pair was reused and not rerun. The interrupted Seed66 root was excluded.\n\nDecision: **"+summary["provisional_decision"]+"**\n\nProvisional Solver: `"+(summary["provisional_solver"] or "none")+"`\n\nThis is a one-seed screening diagnostic, not a formal multi-seed selection claim.\n",encoding="utf-8")
    (REPORT_ROOT/"validation_report.txt").write_text("MULTIMODEL_SOLVER_SCREENING\nphase_a=PASS\nstatic_cells="+str(len(rows))+"\ngeneric_selected="+str(len(selected_generic()))+"\ntest_calls=0\nfull_method_run=false\nfact_assertions=PENDING\ntests=PENDING\nsanitization=PENDING\n",encoding="utf-8")
    print({"decision":summary["provisional_decision"],"solver":summary["provisional_solver"],"rows":len(rows)})


if __name__=="__main__":main()
