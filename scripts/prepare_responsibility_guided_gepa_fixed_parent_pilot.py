"""Freeze the six analytical parents and minibatches for RG-GEPA v1.

This preparation is deliberately zero-API.  The private registry holds frozen
historical prompt/team bytes for a future authorized run; the report copy holds
only hashes, counts, and opaque example IDs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.experimental_rg_gepa import (
    EvidenceItem, RGGEPAProtocol, deterministic_minibatch,
)


SOURCE_ROOT = ROOT / "runs" / "vote_aligned_confirmatory_seed76_77_v1"
DEFAULT_PREP = ROOT / "runs" / "responsibility_guided_gepa_fixed_parent_pilot_v1_prep_20260909"
DEFAULT_REPORT = ROOT / "reports" / "responsibility_guided_gepa_fixed_parent_pilot_v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def prompt_hash(prompt: str) -> str:
    return sha256_bytes(prompt.encode("utf-8"))


def team_hash(prompts: list[str]) -> str:
    return sha256_bytes(json.dumps([prompt_hash(prompt) for prompt in prompts], ensure_ascii=False, separators=(",", ":")).encode())


def profile_block(history: list[dict[str, Any]], state_index: int) -> list[dict[str, Any]]:
    begin = state_index * 100
    block = history[begin:begin + 100]
    if len(block) != 100:
        raise ValueError("historical parent profile block is incomplete")
    return block


def reconstruct_prompt_states(events: list[dict[str, Any]], initial_prompt: str) -> dict[int, list[str]]:
    prompts = [initial_prompt] * 5
    states: dict[int, list[str]] = {}
    for event in sorted(events, key=lambda row: int(row["update_index"])):
        update = int(event["update_index"])
        states[update] = list(prompts)
        expected = str(event["parent_team_hash"])
        if team_hash(prompts) != expected:
            raise ValueError(f"prompt reconstruction mismatch at update {update}")
        if event.get("writeback_approved"):
            candidate = next(
                (row for row in event["candidates"] if row["prompt_hash"] == event["accepted_prompt_hash"]),
                None,
            )
            if candidate is None:
                raise ValueError("accepted prompt is absent from candidate history")
            target = int(candidate["target_agent_id"])
            prompts[target] = str(candidate["evaluation"]["prompt"])
    return states


def source_rows(seed: int) -> dict[str, Any]:
    run = SOURCE_ROOT / f"seed{seed}" / "P1_SHADOW_VOTE_ALIGNED_GENERIC"
    meta = read_json(run / "run_meta.json")
    events = read_jsonl(run / "candidate_decisions.jsonl")
    checkpoint = read_json(run / "training_checkpoint.json")
    history = checkpoint["peer_state_history"]
    train_path = Path(meta["config"]["train_path"])
    if not train_path.is_file():
        raise FileNotFoundError("frozen Optimize100 split unavailable")
    with train_path.open(encoding="utf-8-sig", newline="") as handle:
        dataset = list(csv.DictReader(handle))
    if len(dataset) != 100:
        raise ValueError("RG-GEPA requires the frozen Optimize100 source split")
    initial = str(meta["config"]["shared_prompt"])
    states = reconstruct_prompt_states(events, initial)
    questions = [{"question": str(row["question"]), "answer": str(row["answer"]),
                  "example_id": sha256_bytes(str(row["question"]).encode())} for row in dataset]
    if len({row["example_id"] for row in questions}) != 100:
        raise ValueError("Optimize100 canonical IDs must be unique")
    return {"run": run, "meta": meta, "events": events, "history": history, "states": states, "questions": questions}


def select_cases(seed: int, source: dict[str, Any]) -> list[dict[str, Any]]:
    """First target by update/agent in each lane: coverage, margin, direct.

    This only uses pre-generation routing data.  It never branches on candidate
    quality, feasibility, validation, or final result.
    """
    lane_order = ("coverage", "margin_support", "direct_flip")
    selected: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for event in sorted(source["events"], key=lambda row: int(row["update_index"])):
        for branch in sorted(event["branches"], key=lambda row: int(row["target_agent_id"])):
            lane = str(branch["active_lane"])
            candidate_count = sum(
                int(candidate["target_agent_id"]) == int(branch["target_agent_id"])
                and str(candidate.get("candidate_stage")) == "m20_source"
                for candidate in event["candidates"]
            )
            if lane in lane_order and lane not in selected and candidate_count == 2:
                selected[lane] = (event, branch)
    if set(selected) != set(lane_order):
        raise ValueError(f"seed {seed} cannot satisfy predeclared lane balance")
    cases=[]
    for lane in lane_order:
        event, branch = selected[lane]
        update = int(event["update_index"])
        commits_before = sum(bool(row.get("writeback_approved")) for row in source["events"] if int(row["update_index"]) < update)
        state = profile_block(source["history"], commits_before)
        target = int(branch["target_agent_id"])
        candidate_texts=[]
        for candidate in event["candidates"]:
            if int(candidate["target_agent_id"]) == target and str(candidate.get("candidate_stage")) == "m20_source":
                candidate_texts.append({"candidate_id": str(candidate["prompt_hash"]), "prompt": str(candidate["evaluation"]["prompt"])})
        if len(candidate_texts) != 2:
            raise ValueError("current pool does not preserve two target candidates")
        cases.append({"case_id": f"seed{seed}_u{update}_{lane}_target{target}", "source_seed":seed,
                      "source_update_index":update,"target_member":target,"responsibility_type":lane,
                      "parent_team_hash":str(event["parent_team_hash"]),"parent_prompts":source["states"][update],
                      "parent_profile":state,"initial_profile":profile_block(source["history"],0),
                      "assigned_example_ids":sorted(map(str,branch["assigned_question_hashes"])),
                      "current_pool":candidate_texts,"questions":source["questions"],
                      "source_run_identity":str(source["meta"]["run_identity"]["git_commit"]),
                      "selection_rule":"first_update_then_lowest_target_per_lane_with_two_historical_pool_members_v1"})
    return cases


def evidence_for_case(case: dict[str, Any]) -> list[EvidenceItem]:
    profile={str(row["question_hash"]):row for row in case["parent_profile"]}
    assigned=set(case["assigned_example_ids"])
    items=[]
    lane=str(case["responsibility_type"])
    for example in case["questions"]:
        ident=str(example["example_id"]); row=profile[ident]
        g=int(row["gold_vote_count"])
        if ident in assigned:
            source="coverage" if lane=="coverage" else "conversion"
            items.append(EvidenceItem(ident,source,lane,bool(row["team_correctness"][case["target_member"]]),bool(row["vote_correct"]),g))
        elif g >= 3:
            items.append(EvidenceItem(ident,"preservation","preservation",bool(row["team_correctness"][case["target_member"]]),bool(row["vote_correct"]),g))
        elif not bool(row["vote_correct"]):
            items.append(EvidenceItem(ident,"representative",lane,bool(row["team_correctness"][case["target_member"]]),False,g))
        else:
            items.append(EvidenceItem(ident,"filler","broad_preservation",bool(row["team_correctness"][case["target_member"]]),True,g,ident))
    return deterministic_minibatch(items)


def sanitized_case(case: dict[str, Any], minibatch: list[EvidenceItem]) -> dict[str, Any]:
    return {"case_id":case["case_id"],"source_seed":case["source_seed"],"source_update_index":case["source_update_index"],
            "target_member":case["target_member"],"responsibility_type":case["responsibility_type"],
            "parent_team_hash":case["parent_team_hash"],"parent_prompt_hashes":[prompt_hash(x) for x in case["parent_prompts"]],
            "current_candidate_hashes":[row["candidate_id"] for row in case["current_pool"]],
            "minibatch_hash":sha256_json([item.example_id for item in minibatch]),"minibatch_size":len(minibatch),
            "selection_rule":case["selection_rule"]}


def prepare(prep: Path, report: Path) -> dict[str, Any]:
    if prep.exists() or report.exists():
        raise FileExistsError("RG-GEPA prep/report roots must be fresh")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before RG-GEPA freeze")
    protocol=RGGEPAProtocol()
    sources=[source_rows(76),source_rows(77)]
    cases=[case for seed, source in zip((76,77),sources,strict=True) for case in select_cases(seed,source)]
    if len(cases)!=6 or [case["responsibility_type"] for case in cases].count("coverage")!=2:
        raise AssertionError("six-case balanced parent selection failed")
    minibatches={case["case_id"]:evidence_for_case(case) for case in cases}
    for case in cases:
        if team_hash(case["parent_prompts"]) != case["parent_team_hash"]:
            raise AssertionError("frozen parent prompt identity mismatch")
    registry={"registry_version":"responsibility_guided_gepa_fixed_parent_registry_v1","execution_commit":git("rev-parse","HEAD"),
              "protocol":asdict(protocol),"protocol_hash":protocol.identity(),"case_selection_uses_outcomes":False,
              "api_authorized":False,"commit_enabled":False,"test_enabled":False,"external_validation_enabled":False,
              "cases":cases,"minibatches":{key:[item.__dict__ for item in value] for key,value in minibatches.items()}}
    registry["registry_hash"]=sha256_json(registry)
    prep.mkdir(parents=True)
    write_json(prep/"private_registry.json",registry)
    manifest=[sanitized_case(case,minibatches[case["case_id"]]) for case in cases]
    write_json(report/"fixed_parent_manifest.json",manifest)
    with (report/"minibatch_manifest.jsonl").open("w",encoding="utf-8") as handle:
        for case in cases:
            for item in minibatches[case["case_id"]]:
                handle.write(json.dumps({"seed":case["source_seed"],"update_index":case["source_update_index"],"target_member":case["target_member"],"example_id":item.example_id,"source_type":item.source_type,"responsibility_type":item.responsibility_type,"is_filler":item.source_type=="filler","parent_member_correct":item.parent_member_correct,"parent_vote_correct":item.parent_vote_correct,"parent_G":item.parent_g},sort_keys=True)+"\n")
    source_files=[
        Path("multi_dataset_diverse_rl/experimental_rg_gepa.py"),
        Path("scripts/prepare_responsibility_guided_gepa_fixed_parent_pilot.py"),
        Path("scripts/run_responsibility_guided_gepa_fixed_parent_pilot.py"),
        Path("tests/test_experimental_rg_gepa.py"),
        Path("tests/test_rg_gepa_execution_ledger.py"),
    ]
    freeze={"PRE_API_FREEZE_version":"rg_gepa_v1","execution_commit":registry["execution_commit"],"protocol_hash":protocol.identity(),"registry_hash":registry["registry_hash"],"fixed_parent_manifest_hash":sha256_json(manifest),"minibatches_hash":sha256_json(registry["minibatches"]),"solver_contract_id":protocol.solver_contract_id,"api_calls":0,"test_calls":0,"source_files":[{"path":str(path).replace("\\","/"),"sha256":sha256_bytes((ROOT/path).read_bytes())} for path in source_files],"status":"PASS"}
    write_json(prep/"PRE_API_FREEZE.json",freeze)
    hypotheses={
        "H1_PROGRESSIVE_COST":"NOT_EVALUATED_PRE_API",
        "H2_REFLECTION_QUALITY":"NOT_EVALUATED_PRE_API",
        "H3_PARETO_SELECTION":"NOT_EVALUATED_PRE_API",
        "H4_MINIBATCH_SIGNAL":"NOT_EVALUATED_PRE_API",
        "classification_rule":"frozen_before_provider_calls",
    }
    report_protocol={**asdict(protocol),"protocol_version":"responsibility_guided_gepa_fixed_parent_v1",
                     "protocol_hash":protocol.identity(),"case_count":6,"arms":["A0","A1","B0","B1"],
                     "A0_A1_shared_current_pool":True,"B0_B1_shared_reflection_pool":True,
                     "common_solver_contract_required":True,"actual_commit_count":0,
                     "test50_accessed":False,"external_validation_optimization_calls":0}
    write_json(report/"protocol.json",report_protocol)
    write_json(report/"frozen_hypotheses.json",hypotheses)
    write_json(report/"fact_assertions.json",{
        "status":"PASS","api_calls":0,"test50_calls":0,"external_validation_optimization_calls":0,
        "case_count":len(cases),"minibatch_size":12,"candidate_budget_per_target":2,
        "full_evaluation_promotion_limit":2,"actual_commits":0,"historical_artifacts_modified":0,
        "solver_contract_id":protocol.solver_contract_id,
    })
    write_json(report/"provenance.json",{
        "phase":"PRE_API_FREEZE","source_runs":["seed76/P1_SHADOW_VOTE_ALIGNED_GENERIC","seed77/P1_SHADOW_VOTE_ALIGNED_GENERIC"],
        "selection_rule":"first usable target branch per coverage/margin_support/direct_flip lane, by update then target", 
        "source_outcomes_used_for_case_selection":False,"private_registry":"runs-only; excluded from report",
    })
    (report/"README.md").write_text(
        "# Responsibility-Guided GEPA fixed-parent pilot v1\n\n"
        "This is the zero-API preflight and freeze only. It defines six analytical, non-committing parent cases: coverage, margin-support, and direct-flip from each of Seeds76 and 77. No provider call, Test50 access, ExternalValidation optimization call, candidate generation, full evaluation, Shadow gate, or write-back occurred.\n\n"
        "The eventual paired comparison is frozen as A0 current/full/current selector; A1 same current pool/progressive/current selector; B0 reflection/progressive/current selector; B1 same B pool/team-Pareto selector. A0/A1 and B0/B1 share their candidate pools respectively.\n\n"
        "`PRE_API_FREEZE.json` is private execution evidence under `runs/`; this report contains only sanitized identifiers, hashes, and minibatch metadata. API execution requires a separate explicit authorization and must use COMMON_SOLVER_CONTRACT_V1.\n",
        encoding="utf-8",
    )
    prohibited=("FINAL_ANSWER:","api_key","dashscope","D:\\\\","https://")
    hashes=[]
    for path in sorted(report.iterdir()):
        raw=path.read_bytes(); text=raw.decode("utf-8",errors="ignore").lower()
        if any(term.lower() in text for term in prohibited):
            raise ValueError(f"sanitization failure: {path.name}")
        hashes.append({"path":path.name,"sha256":sha256_bytes(raw),"bytes":len(raw)})
    write_json(report/"sanitization_manifest.json",{"status":"PASS","files_checked":len(hashes)})
    write_json(report/"sha256_manifest.json",{"files":hashes})
    return {"registry":registry,"manifest":manifest,"freeze":freeze}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--prep",type=Path,default=DEFAULT_PREP)
    parser.add_argument("--report",type=Path,default=DEFAULT_REPORT)
    args=parser.parse_args()
    result=prepare(args.prep.resolve(),args.report.resolve())
    print(json.dumps({"status":"PASS","api_calls":0,"test_calls":0,"case_count":len(result["manifest"]),"registry_hash":result["registry"]["registry_hash"]},sort_keys=True))


if __name__=="__main__":
    main()
