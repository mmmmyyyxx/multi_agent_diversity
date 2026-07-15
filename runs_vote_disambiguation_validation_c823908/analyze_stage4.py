import csv, json, math, random, statistics
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "runs_vote_stage2_selector_pilot_v4_dcc9492" / "disambiguation_qa"
S3 = ROOT / "stage3_new_runs" / "disambiguation_qa"
S4 = ROOT / "stage4_new_runs" / "disambiguation_qa"
SETTINGS = ["shared_baseline", "shared_scalar_tcs_vote_first", "shared_guarded_diversity_tcs_vote_first"]

def path(setting, seed):
    if seed == 42:
        if setting == "shared_guarded_diversity_tcs_vote_first": return S3 / f"{setting}_seed42"
        return OLD / f"{setting}_seed42"
    return S4 / f"{setting}_seed{seed}"

def read_json(p): return json.loads(p.read_text(encoding="utf-8"))
def read_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def preds(p):
    q = p / "test_final_predictions.jsonl"
    if not q.exists(): q = p / "test_epoch1_predictions.jsonl"
    return read_jsonl(q)
def final(p):
    h = read_json(p / "history.json")
    return next((x["test"] for x in reversed(h) if "test" in x), {})
def write(name, rows):
    keys = list(rows[0]) if rows else []
    with (ROOT / name).open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

result=[]
for s in SETTINGS:
  for seed in (42,43,44):
    m=final(path(s,seed)); result.append({"setting":s,"seed":seed,**{k:m.get(k,0) for k in ["vote_acc","mean_individual_acc","best_individual_acc","mean_vote_margin","vote_tie_rate","oracle_acc","aggregation_gap","mean_invalid_rate","mean_boundary_useful_diversity","mean_embedding_diversity"]}})
write("stage4_results_by_seed.csv",result)

summary=[]
for s in SETTINGS:
  rs=[x for x in result if x["setting"]==s]; row={"setting":s}
  for k in ["vote_acc","mean_individual_acc","best_individual_acc","mean_vote_margin","vote_tie_rate","oracle_acc","aggregation_gap","mean_invalid_rate","mean_boundary_useful_diversity","mean_embedding_diversity"]:
    v=[float(x[k]) for x in rs]; row[k+"_mean"]=statistics.mean(v); row[k+"_std"]=statistics.stdev(v)
  summary.append(row)
write("stage4_summary.csv",summary)

boot=[]; mc=[]; trans=[]; strata=[]; decomp=[]
cats=lambda m: "stable correct" if m>0.2 else "fragile correct" if m>0 else "tie/boundary" if m==0 else "fragile wrong" if m>=-0.2 else "stable wrong"
for seed in (42,43,44):
  b=preds(path("shared_baseline",seed)); bm={r["question_hash"]:r for r in b}
  for s in SETTINGS[1:]:
    mm={r["question_hash"]:r for r in preds(path(s,seed))}; ids=sorted(set(bm)&set(mm)); d=[int(mm[i]["vote_correct"])-int(bm[i]["vote_correct"]) for i in ids]
    rng=random.Random(1000+seed+SETTINGS.index(s)); vals=[sum(d[rng.randrange(len(d))] for _ in d)/len(d) for _ in range(10000)]; vals.sort()
    boot.append({"setting":s,"seed":seed,"n":len(d),"difference":sum(d)/len(d),"ci_low":vals[249],"ci_high":vals[9749],"probability_positive":sum(x>0 for x in vals)/10000})
    b01=sum(not bm[i]["vote_correct"] and mm[i]["vote_correct"] for i in ids); b10=sum(bm[i]["vote_correct"] and not mm[i]["vote_correct"] for i in ids); n=b01+b10
    p=min(1.0,2*sum(math.comb(n,k) for k in range(min(b01,b10)+1))/(2**n)) if n else 1.0
    mc.append({"setting":s,"seed":seed,"wrong_to_correct":b01,"correct_to_wrong":b10,"exact_p":p})
    counts={k:0 for k in ["Wrong -> Correct","Correct -> Wrong","Correct -> Correct","Wrong -> Wrong"]}; dg=[];dh=[];dm=[]; fix={k:0 for k in ["G only","H only","G and H","other"]}
    for i in ids:
      x,y=bm[i],mm[i]; key=("Correct" if x["vote_correct"] else "Wrong")+" -> "+("Correct" if y["vote_correct"] else "Wrong"); counts[key]+=1
      a=y["gold_vote_count"]-x["gold_vote_count"]; c=y["largest_wrong_vote_count"]-x["largest_wrong_vote_count"]; dg.append(a);dh.append(c);dm.append(y["normalized_vote_margin"]-x["normalized_vote_margin"])
      if key=="Wrong -> Correct": fix["G and H" if a>0 and c<0 else "G only" if a>0 else "H only" if c<0 else "other"]+=1
    for k,v in counts.items(): trans.append({"setting":s,"seed":seed,"transition":k,"count":v,"rate":v/len(ids)})
    for c in ["stable correct","fragile correct","tie/boundary","fragile wrong","stable wrong"]:
      ix=[i for i in ids if cats(float(bm[i]["normalized_vote_margin"]))==c]; strata.append({"setting":s,"seed":seed,"baseline_stratum":c,"count":len(ix),"baseline_vote_acc":sum(bm[i]["vote_correct"] for i in ix)/len(ix) if ix else 0,"method_vote_acc":sum(mm[i]["vote_correct"] for i in ix)/len(ix) if ix else 0})
    decomp.append({"setting":s,"seed":seed,"mean_delta_G":statistics.mean(dg),"mean_delta_H":statistics.mean(dh),"mean_delta_margin":statistics.mean(dm),**{"wrong_to_correct_"+k.replace(" ","_").lower():v for k,v in fix.items()}})
write("paired_bootstrap.csv",boot);write("mcnemar_results.csv",mc);write("vote_transition.csv",trans);write("margin_stratified_analysis.csv",strata);write("accuracy_diversity_decomposition.csv",decomp)

pair=[]
for s in SETTINGS:
 for seed in (42,43,44):
  rr=preds(path(s,seed)); dfs=[]; cors=[]; agrees=[]
  for a,b in combinations(range(5),2):
   xa=[int(r["agent_correct"][a]) for r in rr]; xb=[int(r["agent_correct"][b]) for r in rr]; dfs.append(sum(not x and not y for x,y in zip(xa,xb))/len(rr)); agrees.append(sum(r["agent_answers"][a]==r["agent_answers"][b] for r in rr)/len(rr));
   ma=statistics.mean(xa);mb=statistics.mean(xb); den=(sum((x-ma)**2 for x in xa)*sum((y-mb)**2 for y in xb))**.5; cors.append(sum((x-ma)*(y-mb) for x,y in zip(xa,xb))/den if den else 0)
  dom=sum((not r["vote_correct"]) and r["largest_wrong_vote_count"]>=3 for r in rr)/len(rr)
  pair.append({"setting":s,"seed":seed,"pairwise_double_fault_rate":statistics.mean(dfs),"pairwise_correctness_correlation":statistics.mean(cors),"pairwise_answer_agreement":statistics.mean(agrees),"dominant_wrong_cluster_rate":dom})
write("pairwise_error_analysis.csv",pair)

print(json.dumps({"results":result,"summary":summary,"bootstrap":boot,"mcnemar":mc},indent=2))
