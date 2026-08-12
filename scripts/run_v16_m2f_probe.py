from __future__ import annotations
import argparse,asyncio,hashlib,json,os,subprocess,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
from m2f_probe_support import *
from generic_m20_probe_support import state_hash
from multi_dataset_diverse_rl.system import CandidateFunnel,CandidateRuntime
from multi_dataset_diverse_rl.tcs import StudentPromptCandidate

def write(path:Path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8");os.replace(tmp,path)
def head():return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
def dirty():return subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True).strip()
def freeze_gate(reg,freeze):
 errors=[]
 if head()!=reg["execution_commit"] or head()!=freeze.get("execution_commit"):errors.append("execution_commit")
 if dirty():errors.append("tracked_tree_dirty")
 if freeze.get("source_freeze_status")!="PASS":errors.append("source_freeze")
 if reg.get("registry_content_hash")!=canonical_hash({k:v for k,v in reg.items() if k!="registry_content_hash"}):errors.append("registry_hash")
 if reg.get("registry_content_hash")!=freeze.get("registry_content_hash"):errors.append("frozen_registry_hash")
 if len(reg.get("cases",()))!=7 or reg.get("model")!=MODEL or reg.get("thinking") is not False:errors.append("contract")
 return errors

async def run_case(case,out,cache):
 system=evaluation_system(case,out_dir=out/"evaluation",cache_path=cache);target=int(case["target_agent_id"]);before=state_hash(system);calls_before=len(system.llm.calls)
 result=await system.llm.chat_result(MODEL,REPAIR_SYSTEM_PROMPT,build_repair_request(case),system.cfg.tcs.student_temperature,REPAIR_MAX_TOKENS,"optimizer","m2f_repair")
 repaired=parse_repair_output(result.text,case);prompt_hash=system.prompt_hash(repaired);runtime=CandidateRuntime(student_candidate=StudentPromptCandidate(candidate_prompt=repaired),prompt=repaired,prompt_hash=prompt_hash,generation=1,parent_prompt_hash=system.prompt_hash(case["parent_prompt"]))
 funnel=CandidateFunnel(requested_candidate_count=1,raw_candidate_count=1,schema_valid_count=1,non_parent_count=1,deduplicated_count=1)
 _,incumbent,rows=await system.evaluate_candidates(target,[runtime],set(case["assigned_question_hashes"]),funnel,int(case["source_update_index"]));row=rows[0]
 if state_hash(system)!=before:raise RuntimeError("parent state mutated")
 states,_,_=system.current_states_and_opportunities();by_hash={s.question_hash:s for s in states};repair_gain=set();repair_loss={"stable":set(),"pivotal":set(),"unique":set(),"fragile":set()};stable=set(case["stable_correct_question_hashes_by_agent"][str(target)])
 for state,obs in zip(states,row.profile):
  before_correct=bool(state.team_correctness[target]);after=bool(obs.valid and system.match_answer(obs.answer,state.gold_answer));qh=state.question_hash
  if qh in set(case["assigned_question_hashes"]) and not before_correct and after:repair_gain.add(qh)
  if qh not in set(case["assigned_question_hashes"]) and before_correct and not after:repair_loss[competence_role(system,target,state,stable)].add(qh)
 source_gain=set(case["source_responsibility_gain_hashes"]);source_loss={k:set(v) for k,v in case["all_source_loss_hashes"].items()};source_all=set().union(*source_loss.values());repair_all=set().union(*repair_loss.values());source_piv=source_loss["pivotal"]|source_loss["unique"];repair_piv=repair_loss["pivotal"]|repair_loss["unique"]
 payload={"case_id":case["case_id"],"seed":case["source_seed"],"target_agent_id":target,"source_candidate_hash":case["source_candidate_hash"],"repair_candidate_hash":prompt_hash,"source_execution_commit":SOURCE_EXECUTION_COMMIT,"parent_team_hash":case["parent_team_hash"],"responsibility_evidence_hash":case["frozen_responsibility_evidence_hash"],"repair_input_hash":case["repair_input_hash"],"repair_output_hash":prompt_hash,"repair_attempt_count":1,"api_repair_calls":len(system.llm.calls)-calls_before,"source_responsibility_gain_hashes":sorted(source_gain),"repair_responsibility_gain_hashes":sorted(repair_gain),"source_loss_hashes":{k:sorted(v) for k,v in source_loss.items()},"repair_loss_hashes":{k:sorted(v) for k,v in repair_loss.items()},"retained_source_repairs":len(source_gain&repair_gain),"lost_source_repairs":len(source_gain-repair_gain),"new_responsibility_repairs":len(repair_gain-source_gain),"targeting_retention":len(source_gain&repair_gain)/len(source_gain),"loss_decomposition":loss_decomposition(source_all,repair_all),"pivotal_decomposition":loss_decomposition(source_piv,repair_piv),"stable_decomposition":loss_decomposition(source_loss["stable"],repair_loss["stable"]),"source_metrics":case["source_metrics"],"repair_metrics":{"target_gain":row.module2_diagnostics["target_gain"],"vote_gain_count":row.module2_diagnostics["vote_gain_count"],"vote_loss_count":row.module2_diagnostics["vote_loss_count"],"vote_net_gain":row.module2_diagnostics["vote_net_gain"],"candidate_geometry":row.module2_diagnostics["candidate_geometry"],"terminal_invalid_delta":row.final_evaluation.terminal_invalid_count-incumbent.terminal_invalid_count,"common_safe_feasible":bool(row.constraint.passed),"nonresponsibility_loss_count":len(repair_all),"stable_loss_count":len(repair_loss["stable"]),"pivotal_loss_count":len(repair_piv),"responsibility_gain_count":len(repair_gain)},"source_common_safe":bool(case["source_metrics"]["constraint"]["passed"]),"repair_common_safe":bool(row.constraint.passed),"compatibility_rescue":not bool(case["source_metrics"]["constraint"]["passed"]) and bool(row.constraint.passed),"team_prompt_commits":0,"parent_mutations":0,"optimizer_updates":0,"validation_calls":0,"test_calls":0}
 write(out/"cell_result.json",payload);write(out/"call_metadata.json",system.llm.calls);return payload

async def main_async(a):
 reg=json.loads(a.registry.read_text(encoding="utf-8"));freeze=json.loads(a.source_freeze.read_text(encoding="utf-8"));errors=freeze_gate(reg,freeze)
 if os.environ.get(AUTH_ENV)!="1":raise SystemExit(f"API execution blocked: {AUTH_ENV}=1 required")
 if errors:raise SystemExit("source freeze gate failed:"+','.join(errors))
 if a.out.exists() or ROOT.resolve() not in a.out.resolve().parents:raise SystemExit("fresh repo-local out required")
 a.out.mkdir(parents=True);results=[];first=False
 for index,case in enumerate(reg["cases"]):
  if first and freeze_gate(reg,freeze):raise RuntimeError("source changed after first call")
  row=await run_case(case,a.out/f"cell_{index+1}_{case['source_candidate_hash'][:12]}",a.out/"shared_solver_cache.sqlite");results.append(row)
  if not first:write(a.out/"first_success_source_freeze.json",{"execution_commit":head(),"registry_content_hash":reg["registry_content_hash"],"working_tree_source_hash":freeze["working_tree_source_hash"]});first=True
 if freeze_gate(reg,freeze):raise RuntimeError("source changed before finalization")
 write(a.out/"probe_summary.json",{"execution_commit":head(),"cell_count":7,"api_repair_calls":sum(x["api_repair_calls"] for x in results),"team_prompt_commits":0,"parent_mutations":0,"optimizer_updates":0,"validation_calls":0,"test_calls":0,"cells":results});print(json.dumps({"status":"complete","cells":7}))
def main():
 p=argparse.ArgumentParser();p.add_argument("--registry",type=Path,required=True);p.add_argument("--source_freeze",type=Path,required=True);p.add_argument("--out",type=Path,required=True);asyncio.run(main_async(p.parse_args()))
if __name__=="__main__":main()
