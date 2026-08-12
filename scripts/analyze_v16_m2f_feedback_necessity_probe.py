from __future__ import annotations
import argparse,json
from pathlib import Path
F1="F1_LOSS_BLIND";F2="F2_CANDIDATE_FEEDBACK"
def summarize(rows):
 return {"targeting_retention":sum(x["retained_source_repairs"] for x in rows)/sum(x["source_responsibility_repairs"] for x in rows),"collateral_loss":sum(x["nonresponsibility_loss"] for x in rows),"recovered_collateral":sum(x["loss_decomposition"]["recovered"] for x in rows),"new_collateral":sum(x["loss_decomposition"]["new"] for x in rows),"compatibility_rescues":sum(x["common_safe"] for x in rows),"critical_gain":sum(x["critical_gain"] for x in rows),"critical_loss":sum(x["critical_loss"] for x in rows),"critical_net":sum(x["critical_net"] for x in rows),"oracle_delta":sum(x["oracle_delta"] for x in rows),"vote_net":sum(x["vote_net"] for x in rows),"target_gain":sum(x["target_gain"] for x in rows)}
def main():
 p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();s=json.loads((a.run/"probe_summary.json").read_text(encoding="utf8"));groups={v:[x for x in s["cells"] if x["variant"]==v] for v in (F1,F2)};m={v:summarize(x) for v,x in groups.items()}
 # Descriptive causal classifier: F2 must strictly beat F1 on at least three of
 # five preregistered directions, with no targeting-retention deficit.
 directions=[m[F2]["targeting_retention"]>=m[F1]["targeting_retention"],m[F2]["collateral_loss"]<m[F1]["collateral_loss"],m[F2]["compatibility_rescues"]>m[F1]["compatibility_rescues"],m[F2]["critical_net"]>m[F1]["critical_net"],m[F2]["vote_net"]>m[F1]["vote_net"],m[F2]["recovered_collateral"]>m[F1]["recovered_collateral"]]
 label="FEEDBACK_NECESSARY" if directions[0] and sum(directions[1:])>=3 else "EXTRA_REVISION_SUFFICIENT" if m[F1]["targeting_retention"]>=m[F2]["targeting_retention"] and sum(not x for x in directions[1:])>=3 else "MIXED"
 d={"analysis_version":"v16_m2f_feedback_necessity_v1","classifier":label,"variants":m,"direction_checks":{"targeting_noninferior":directions[0],"collateral_better":directions[1],"more_rescues":directions[2],"critical_net_better":directions[3],"vote_net_better":directions[4],"more_recovery":directions[5]},"critical_net_analysis_metric_only":True,"oracle_delta_analysis_metric_only":True,"vote_net_analysis_metric_only":True};a.out.write_text(json.dumps(d,indent=2)+"\n",encoding="utf8");print(json.dumps(d,indent=2))
if __name__=="__main__":main()
