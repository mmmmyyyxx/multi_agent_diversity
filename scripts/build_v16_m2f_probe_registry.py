from __future__ import annotations
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
from m2f_probe_support import *

def build(summary_path:Path,base_registry_path:Path,cache_path:Path,collateral_path:Path,execution_commit:str)->tuple[dict,dict]:
 summary=json.loads(summary_path.read_text(encoding="utf-8"))
 if summary.get("execution_commit")!=SOURCE_EXECUTION_COMMIT: raise ValueError("wrong historical execution provenance")
 base=json.loads(base_registry_path.read_text(encoding="utf-8"))
 cases_by_id={c["case_id"]:c for c in base["cases"]};coll={r["candidate_hash"]:r for r in csv.DictReader(collateral_path.open(encoding="utf-8"))}
 source_cells={c["case_id"]:c for c in summary["cells"] if c["variant"]=="m20_current_v15"}
 out=[]
 for source_hash in SOURCE_HASHES:
  if source_hash not in coll: raise ValueError("missing frozen collateral candidate")
  frozen=coll[source_hash]
  if int(frozen["target_gain"])>=0 or int(frozen["responsibility_gain_count"])<=0 or int(frozen["nonresponsibility_loss_count"])<=0: raise ValueError("source candidate does not satisfy frozen selection")
  case=cases_by_id[frozen["case_id"]];cell=source_cells[case["case_id"]];candidate=next((x for x in cell["candidates"] if x["prompt_hash"]==source_hash),None)
  if candidate is None: raise ValueError("missing raw source candidate")
  prompt=candidate["evaluation"].get("prompt","")
  if sha_text(prompt)!=source_hash: raise ValueError("candidate hash mismatch")
  target=int(case["target_agent_id"]);system=evaluation_system(case,out_dir=ROOT/"runs/m2f_prep_read_only",cache_path=cache_path)
  if team_prompt_hash(system)!=case["parent_team_hash"]: raise ValueError("source parent-team hash mismatch")
  states,_,_=system.current_states_and_opportunities();by_hash={s.question_hash:s for s in states};answers=read_cached_answers(cache_path,source_hash,system)
  responsibility=set(case["assigned_question_hashes"])
  if responsibility_evidence_hash(responsibility)!=case["frozen_responsibility_evidence_hash"]: raise ValueError("responsibility-set mismatch")
  stable=set(case["stable_correct_question_hashes_by_agent"][str(target)]);gains=[];losses=[];all_loss={"stable":[],"pivotal":[],"unique":[],"fragile":[]}
  for qh,state in by_hash.items():
   obs=answers[qh];before=bool(state.team_correctness[target]);after=bool(obs.get("valid")) and system.match_answer(str(obs.get("answer","")),state.gold_answer)
   evidence={"question_hash":qh,"question":next(q["question"] for q in case["questions"] if q["question_hash"]==qh),"gold_answer":state.gold_answer,"parent_target_output":case["active_profiles"][next(i for i,q in enumerate(case["active_profiles"]) if q["question_hash"]==qh)]["team_answers"][target] if False else system.active_profiles[target][next(i for i,e in enumerate(system.fixed_probe.examples) if e.question_hash==qh)].answer,"source_candidate_target_output":str(obs.get("answer",""))}
   if qh in responsibility and not before and after: gains.append({**evidence,"responsibility_status":"assigned"})
   if qh not in responsibility and before and not after:
    role=competence_role(system,target,state,stable);all_loss[role].append({**evidence,"parent_competence_role":role,"parent_plurality_margin":int(state.plurality_margin)})
  pivotal=all_loss["pivotal"]+all_loss["unique"]
  stable_selected=sorted(all_loss["stable"],key=lambda x:(-x["parent_plurality_margin"],x["question_hash"]))[:STABLE_CAP]
  losses=sorted(pivotal,key=lambda x:x["question_hash"])+stable_selected
  numeric={"responsibility_gain_count":len(gains),"nonresponsibility_loss_count":sum(map(len,all_loss.values())),"pivotal_loss_count":len(pivotal),"stable_loss_count":len(all_loss["stable"]),"source_target_gain":int(frozen["target_gain"]),"source_vote_gain_count":int(candidate["vote_gain_count"]),"source_vote_loss_count":int(candidate["vote_loss_count"])}
  row={**case,"source_candidate_hash":source_hash,"source_candidate_prompt":prompt,"parent_prompt":case["parent_prompts"][target],"source_execution_commit":SOURCE_EXECUTION_COMMIT,"repair_evidence":sorted(gains,key=lambda x:x["question_hash"]),"loss_evidence":losses,"numeric_summary":numeric,"source_metrics":candidate,"all_source_loss_hashes":{k:sorted(x["question_hash"] for x in v) for k,v in all_loss.items()},"source_responsibility_gain_hashes":sorted(x["question_hash"] for x in gains)}
  row["repair_input_hash"]=hashlib.sha256(build_repair_request(row).encode()).hexdigest();out.append(row)
 if len(out)!=7: raise ValueError("source candidate count mismatch")
 registry={"registry_version":"v16_m2f_candidate_registry_v1","execution_commit":execution_commit,"source_execution_commit":SOURCE_EXECUTION_COMMIT,"source_audit_commit":SOURCE_AUDIT_COMMIT,"model":MODEL,"thinking":THINKING,"repair_attempts_per_source":1,"validation_enabled":False,"test_enabled":False,"commit_enabled":False,"optimizer_update_enabled":False,"repair_prompt_hash":hashlib.sha256((REPAIR_SYSTEM_PROMPT+REPAIR_INSTRUCTION).encode()).hexdigest(),"evidence_selection_rule":"all responsibility gains; all pivotal/unique losses; stable losses top-2 by parent margin descending then hash ascending","cases":out}
 registry["registry_content_hash"]=canonical_hash(registry)
 recovery={"status":"PASS","source_candidates_recovered":7,"source_hash_mismatch":0,"api_calls":0,"model_calls":0,"validation_calls":0,"test_calls":0,"source_candidate_hashes":list(SOURCE_HASHES)}
 return registry,recovery

def main():
 p=argparse.ArgumentParser();p.add_argument("--summary",type=Path,required=True);p.add_argument("--base_registry",type=Path,required=True);p.add_argument("--cache",type=Path,required=True);p.add_argument("--collateral",type=Path,required=True);p.add_argument("--execution_commit",required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 if a.out.exists() or ROOT.resolve() not in a.out.resolve().parents: raise SystemExit("fresh repo-local out required")
 reg,audit=build(a.summary,a.base_registry,a.cache,a.collateral,a.execution_commit);a.out.mkdir(parents=True);(a.out/"source_candidate_registry.json").write_text(json.dumps(reg,indent=2)+"\n",encoding="utf-8");(a.out/"source_candidate_recovery_audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8");print(json.dumps(audit,indent=2))
if __name__=="__main__":main()
