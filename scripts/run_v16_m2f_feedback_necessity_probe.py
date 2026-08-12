from __future__ import annotations
import argparse,asyncio,json,os,subprocess,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
from m2f_probe_support import *
from generic_m20_probe_support import source_manifest,state_hash
from audit_v16_m2f_critical_competence_exchange import exchange_for_candidate
from multi_dataset_diverse_rl.peer_state import build_team_vote_state
from multi_dataset_diverse_rl.system import CandidateFunnel,CandidateRuntime
from multi_dataset_diverse_rl.tcs import StudentPromptCandidate
VARIANTS=("F1_LOSS_BLIND","F2_CANDIDATE_FEEDBACK")
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp');t.write_text(json.dumps(v,indent=2)+"\n",encoding="utf8");os.replace(t,p)
def head():return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
def dirty():return subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True).strip()
def gate(reg,freeze,registry_path):
 e=[]
 if head()!=reg["execution_commit"] or head()!=freeze.get("execution_commit"):e.append("commit")
 if dirty():e.append("dirty")
 if freeze.get("source_freeze_status")!="PASS":e.append("freeze")
 if source_manifest().get("working_tree_source_hash")!=freeze.get("working_tree_source_hash"):e.append("source_hash")
 for relative,expected in freeze.get("definition_sha256",{}).items():
  path=ROOT/relative
  if not path.is_file() or __import__("hashlib").sha256(path.read_bytes()).hexdigest()!=expected:e.append("definition_hash")
 raw=json.loads(registry_path.read_text(encoding="utf8"));actual=canonical_hash({k:v for k,v in raw.items() if k!="registry_content_hash"})
 if actual!=freeze.get("registry_content_hash") or raw.get("registry_content_hash")!=actual:e.append("registry_hash")
 if len(reg.get("cases",()))!=7 or tuple(reg.get("variants",()))!=VARIANTS:e.append("contract")
 return e
async def cell(case,variant,out,cache,on_first_success):
 system=evaluation_system(case,out_dir=out/"evaluation",cache_path=cache);target=int(case["target_agent_id"]);before=state_hash(system);request=build_causal_control_request(case,variant)
 result=await system.llm.chat_result(MODEL,REPAIR_SYSTEM_PROMPT,request,system.cfg.tcs.student_temperature,REPAIR_MAX_TOKENS,"optimizer","m2f_feedback_control")
 on_first_success()
 parse_case=case if variant=="F2_CANDIDATE_FEEDBACK" else {**case,"loss_evidence":[]}
 prompt=parse_repair_output(result.text,parse_case);ph=system.prompt_hash(prompt);runtime=CandidateRuntime(StudentPromptCandidate(prompt),prompt,ph,1,system.prompt_hash(case["parent_prompt"]));f=CandidateFunnel(requested_candidate_count=1,raw_candidate_count=1,schema_valid_count=1,non_parent_count=1,deduplicated_count=1)
 _,inc,rows=await system.evaluate_candidates(target,[runtime],set(case["assigned_question_hashes"]),f,int(case["source_update_index"]));row=rows[0]
 if state_hash(system)!=before:raise RuntimeError("parent mutation")
 states,_,_=system.current_states_and_opportunities();gain=set();loss={"stable":set(),"pivotal":set(),"unique":set(),"fragile":set()};stable=set(case["stable_correct_question_hashes_by_agent"][str(target)])
 for s,o in zip(states,row.profile,strict=True):
  after=bool(o.valid and system.match_answer(o.answer,s.gold_answer));qh=s.question_hash
  if qh in set(case["assigned_question_hashes"]) and not s.team_correctness[target] and after:gain.add(qh)
  if qh not in set(case["assigned_question_hashes"]) and s.team_correctness[target] and not after:loss[competence_role(system,target,s,stable)].add(qh)
 source_gain=set(case["source_responsibility_gain_hashes"]);source_loss={k:set(v) for k,v in case["all_source_loss_hashes"].items()};sa=set().union(*source_loss.values());ra=set().union(*loss.values())
 parent_by_hash={s.question_hash:s for s in states};candidate_by_hash={}
 for s,o in zip(states,row.profile,strict=True):
  answers=list(s.team_answers);validity=list(s.team_validity);answers[target]=str(o.answer);validity[target]=bool(o.valid)
  candidate_by_hash[s.question_hash]=build_team_vote_state(question_hash=s.question_hash,gold_answer=s.gold_answer,answers=answers,valid_vector=validity,normalize_answer=system.normalize_answer,match_answer=system.match_answer,tie_break=system.protocol.tie_policy,seed=system.cfg.training.seed)
 exchange,_=exchange_for_candidate(parent_states=parent_by_hash,candidate_states_by_hash=candidate_by_hash,target=target,responsibility=set(case["assigned_question_hashes"]))
 payload={"case_id":case["case_id"],"seed":case["source_seed"],"target_agent_id":target,"variant":variant,"source_candidate_hash":case["source_candidate_hash"],"candidate_hash":ph,"parent_team_hash":case["parent_team_hash"],"responsibility_evidence_hash":case["frozen_responsibility_evidence_hash"],"request_hash":canonical_hash(request),"loss_evidence_count_exposed":len(case["loss_evidence"]) if variant==VARIANTS[1] else 0,"repair_attempt_count":1,"retained_source_repairs":len(source_gain&gain),"source_responsibility_repairs":len(source_gain),"total_responsibility_repairs":len(gain),"targeting_retention":len(source_gain&gain)/len(source_gain),"nonresponsibility_loss":len(ra),"loss_decomposition":loss_decomposition(sa,ra),"pivotal_loss":len(loss["pivotal"]|loss["unique"]),"critical_gain":exchange["critical_gain"],"critical_loss":exchange["critical_loss"],"critical_net":exchange["critical_net"],"oracle_delta":exchange["oracle_delta"],"target_gain":row.module2_diagnostics["target_gain"],"vote_gain":row.module2_diagnostics["vote_gain_count"],"vote_loss":row.module2_diagnostics["vote_loss_count"],"vote_net":row.module2_diagnostics["vote_net_gain"],"common_safe":bool(row.constraint.passed),"team_commits":0,"parent_mutations":0,"optimizer_updates":0,"validation_calls":0,"test_calls":0}
 write(out/"cell_result.json",payload);write(out/"call_metadata.json",system.llm.calls);return payload
async def run(a):
 reg=json.loads(a.registry.read_text(encoding="utf8"));freeze=json.loads(a.freeze.read_text(encoding="utf8"));e=gate(reg,freeze,a.registry)
 if os.environ.get(CAUSAL_AUTH_ENV)!="1":raise SystemExit("API authorization required")
 if e:raise SystemExit("freeze gate:"+','.join(e))
 if a.out.exists():raise SystemExit("fresh out required")
 a.out.mkdir(parents=True);rows=[];first=False
 def latch():
  nonlocal first
  if first:return
  problems=gate(reg,freeze,a.registry)
  if problems:raise RuntimeError("first-success freeze:"+','.join(problems))
  write(a.out/"first_success_source_freeze.json",{"execution_commit":head(),"source_hash":source_manifest()["working_tree_source_hash"],"registry_content_hash":reg["registry_content_hash"]});first=True
 for i,case in enumerate(reg["cases"]):
  order=VARIANTS if i%2==0 else tuple(reversed(VARIANTS))
  for variant in order:
   if first and gate(reg,freeze,a.registry):raise RuntimeError("source changed after first call")
   rows.append(await cell(case,variant,a.out/f"{i+1}_{variant}_{case['source_candidate_hash'][:10]}",a.out/"shared_solver_cache.sqlite",latch))
 if gate(reg,freeze,a.registry):raise RuntimeError("final freeze")
 write(a.out/"probe_summary.json",{"execution_commit":head(),"cell_count":14,"repair_api_calls":14,"validation_calls":0,"test_calls":0,"team_commits":0,"parent_mutations":0,"optimizer_updates":0,"cells":rows});print(json.dumps({"status":"complete","cells":14}))
def main():p=argparse.ArgumentParser();p.add_argument("--registry",type=Path,required=True);p.add_argument("--freeze",type=Path,required=True);p.add_argument("--out",type=Path,required=True);asyncio.run(run(p.parse_args()))
if __name__=="__main__":main()
