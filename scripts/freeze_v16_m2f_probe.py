from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
from m2f_probe_support import *
from generic_m20_probe_support import source_manifest,tracked_source_dirty
def main():
 p=argparse.ArgumentParser();p.add_argument("--registry",type=Path,required=True);p.add_argument("--preflight",type=Path,required=True);p.add_argument("--full_tests",required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();reg=json.loads(a.registry.read_text(encoding="utf8"));pre=json.loads(a.preflight.read_text(encoding="utf8"));head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip();dirty=tracked_source_dirty();errors=[]
 if reg["execution_commit"]!=head:errors.append("execution_commit")
 if dirty:errors.append("tracked_source_dirty")
 if pre["status"]!="PASS":errors.append("preflight")
 manifest=source_manifest();d={"freeze_version":"v16_m2f_source_freeze_v1","source_freeze_status":"PASS" if not errors else "FAIL","errors":errors,"execution_commit":head,"source_execution_commit":SOURCE_EXECUTION_COMMIT,"source_candidate_hashes":list(SOURCE_HASHES),"repair_prompt_hash":reg["repair_prompt_hash"],"evidence_selection_rule_hash":hashlib.sha256(reg["evidence_selection_rule"].encode()).hexdigest(),"model":MODEL,"thinking":False,"method_version":"member_aware_peer_state_v15","checkpoint_version":25,"experimental_repair_version":VERSION,"full_test_result":a.full_tests,"repo_dirty":bool(dirty),"registry_content_hash":reg["registry_content_hash"],**manifest};a.out.write_text(json.dumps(d,indent=2)+"\n",encoding="utf8");print(json.dumps({"status":d["source_freeze_status"],"errors":errors},indent=2));raise SystemExit(0 if not errors else 1)
if __name__=="__main__":main()
