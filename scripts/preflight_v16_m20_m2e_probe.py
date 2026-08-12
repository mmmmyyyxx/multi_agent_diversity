from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generic_m20_probe_support import M20, M2E, generation_system, evaluation_system, team_prompt_hash
from multi_dataset_diverse_rl.tcs import (
    M20_CURRENT_V15, M2E_SCOPED_BEHAVIORAL_PATCH, build_teacher_request,
    construct_scoped_prompt, parse_student_candidates, parse_teacher_repair_plan,
)


def preflight(registry: dict, *, scratch: Path) -> dict:
    errors = []
    if tuple(registry.get("variants", ())) != (M20, M2E): errors.append("variants")
    if len(registry.get("cases", ())) != 8: errors.append("cases")
    mechanisms = []
    for index, case in enumerate(registry["cases"]):
        expected = [M20, M2E] if index % 2 == 0 else [M2E, M20]
        if case["cell_order"] != expected: errors.append(f"order:{case['case_id']}")
        systems = {
            v: generation_system(case, v, out_dir=scratch / case["case_id"] / v, cache_path="")
            for v in (M20, M2E)
        }
        target = int(case["target_agent_id"]); frozen = set(case["assigned_question_hashes"])
        contexts = {
            v: systems[v]._proposal_context(
                target, systems[v].agents[target].current_prompt, frozen
            )[0]
            for v in (M20, M2E)
        }
        if type(contexts[M20]) is not type(contexts[M2E]): errors.append(f"context:{case['case_id']}")
        m20_request = build_teacher_request(contexts[M20], evolution_variant=M20_CURRENT_V15)
        m2e_request = build_teacher_request(contexts[M2E], evolution_variant=M2E_SCOPED_BEHAVIORAL_PATCH)
        if "trigger_condition" not in m2e_request or "localized_behavior" not in m2e_request: errors.append(f"teacher:{case['case_id']}")
        parent = systems[M2E].agents[target].current_prompt
        plan = parse_teacher_repair_plan({
            "trigger_condition": "two interpretations remain semantically plausible",
            "localized_behavior": "Compare their decisive evidence before committing.",
        }, evolution_variant=M2E_SCOPED_BEHAVIORAL_PATCH)
        parsed = parse_student_candidates({"scoped_patches": [{
            "trigger_condition": plan.trigger_condition, "localized_behavior": plan.localized_behavior,
        }]}, parent_prompt=parent, context=contexts[M2E], expected_count=1, evolution_variant=M2E_SCOPED_BEHAVIORAL_PATCH)
        candidate = parsed.candidates[0].candidate_prompt
        if candidate[:len(parent)] != parent or candidate != construct_scoped_prompt(parent, plan.trigger_condition, plan.localized_behavior): errors.append(f"append:{case['case_id']}")
        evaluator = evaluation_system(case, out_dir=scratch / case["case_id"] / "eval", cache_path="")
        if team_prompt_hash(evaluator) != case["parent_team_hash"]: errors.append(f"parent:{case['case_id']}")
        mechanisms.append({"case_id": case["case_id"], "m20_teacher_sha256": hashlib.sha256(m20_request.encode()).hexdigest(), "m2e_teacher_sha256": hashlib.sha256(m2e_request.encode()).hexdigest(), "parent_prefix_byte_identical": True})
    return {"preflight_version": "v16_m20_m2e_preflight_v1", "status": "PASS" if not errors else "FAIL", "errors": errors, "case_count": 8, "cell_count": 16, "candidate_count": 32, "mechanisms": mechanisms, "api_calls": 0, "model_calls": 0, "validation_calls": 0, "test_calls": 0}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--registry",type=Path,required=True);parser.add_argument("--out",type=Path,required=True);parser.add_argument("--scratch",type=Path,required=True);args=parser.parse_args()
    result=preflight(json.loads(args.registry.read_text(encoding="utf-8")),scratch=args.scratch)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,indent=2));raise SystemExit(0 if result["status"]=="PASS" else 1)


if __name__ == "__main__": main()
