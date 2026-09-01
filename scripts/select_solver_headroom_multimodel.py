from __future__ import annotations

import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from scripts.solver_headroom_multimodel_support import RUN_ROOT, entrants, read_json, validation_dir, write_json


def main() -> None:
    out=RUN_ROOT/"static_selection_retry1_private.json"
    if out.exists():raise SystemExit("selection already frozen")
    rows=[]
    for entry in entrants():
        val=read_json(validation_dir(entry["key"],"STATIC")/"validation_summary_private.json")
        gap=val["oracle_accuracy"]-val["vote_accuracy"]
        invalid_rate=val["terminal_invalid_count"]/val["resolved_request_count"] if val["resolved_request_count"] else 0.0
        qualified=0.50 <= val["vote_accuracy"] <= 0.64 and gap >= 0.08 and invalid_rate <= 0.01
        rows.append({"key":entry["key"],"model":entry["model"],"priority":entry["priority"],"static_vote_accuracy":val["vote_accuracy"],"static_oracle_accuracy":val["oracle_accuracy"],"oracle_vote_gap":gap,"terminal_invalid_rate":invalid_rate,"qualified":qualified})
    selected=sorted((r for r in rows if r["qualified"]),key=lambda r:(-r["oracle_vote_gap"],r["static_vote_accuracy"],r["priority"]))[:3]
    write_json(out,{"selection_version":"static_headroom_gate_v1_retry1","rows":rows,"selected":[{"key":r["key"],"model":r["model"],"priority":r["priority"]} for r in selected],"selected_count":len(selected),"test_calls":0,"validation_root":"validation_retry1"})
    print({"selected":[r["model"] for r in selected]})


if __name__ == "__main__":main()
