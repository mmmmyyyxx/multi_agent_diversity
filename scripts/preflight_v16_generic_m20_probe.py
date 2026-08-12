from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generic_m20_probe_support import (
    G0,
    M20,
    generation_hashes,
    generation_system,
    evaluation_system,
    responsibility_evidence_hash,
    system_for,
    team_prompt_hash,
)
from multi_dataset_diverse_rl.tcs import (
    AccuracyDiagnosisContext,
    M20_CURRENT_V15,
    SingleLaneDiagnosisContext,
    build_teacher_request,
    context_payload,
    serialize_context,
)


def _request_hashes(context: Any) -> dict[str, str]:
    teacher = build_teacher_request(
        context, evolution_variant=M20_CURRENT_V15
    )
    # The semantic freeze is established before provider output exists. Teacher
    # and context hashes are sufficient to prove that M20 input is byte-current.
    return {
        "context_sha256": hashlib.sha256(
            serialize_context(context).encode()
        ).hexdigest(),
        "teacher_request_sha256": hashlib.sha256(
            teacher.encode()
        ).hexdigest(),
    }


def _leakage_violations(
    context: AccuracyDiagnosisContext,
    frozen_hashes: set[str],
) -> list[str]:
    serialized = serialize_context(context)
    payload = context_payload(context)
    paths = json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()
    forbidden_markers = (
        "responsibility",
        "assigned",
        "repair_lane",
        "coverage_failure_count",
        "conversion_failure_count",
        "repair_distance",
        "vote_flip_gain",
        "margin_gain",
    )
    violations = [
        f"field:{marker}" for marker in forbidden_markers if marker in paths
    ]
    violations.extend(
        f"question_hash:{question_hash}"
        for question_hash in sorted(frozen_hashes)
        if question_hash in serialized
    )
    return violations


def preflight(registry: dict[str, Any], *, scratch: Path) -> dict[str, Any]:
    errors: list[str] = []
    cases: list[dict[str, Any]] = []
    if registry.get("registry_version") != (
        "v16_generic_m20_fixed_parent_registry_v1"
    ):
        errors.append("registry_version")
    if len(registry.get("cases", [])) != 8:
        errors.append("case_count")
    if int(registry.get("cell_count", -1)) != 16:
        errors.append("cell_count")
    if int(registry.get("candidate_count_per_cell", -1)) != 2:
        errors.append("candidate_budget")

    for case in registry.get("cases", []):
        target = int(case["target_agent_id"])
        frozen = set(map(str, case["assigned_question_hashes"]))
        expected_evidence_hash = str(
            case["frozen_responsibility_evidence_hash"]
        )
        if responsibility_evidence_hash(frozen) != expected_evidence_hash:
            errors.append(f"responsibility_hash:{case['case_id']}")
        systems = {
            variant: generation_system(
                case,
                variant,
                out_dir=scratch / str(case["case_id"]) / variant,
                cache_path=scratch / "offline.sqlite",
            )
            for variant in (G0, M20)
        }
        evaluator = evaluation_system(
            case,
            out_dir=scratch / str(case["case_id"]) / "evaluation",
            cache_path=scratch / "offline.sqlite",
        )
        parent_hashes = {
            variant: team_prompt_hash(system)
            for variant, system in {**systems, "evaluation": evaluator}.items()
        }
        if set(parent_hashes.values()) != {str(case["parent_team_hash"])}:
            errors.append(f"parent_hash:{case['case_id']}")
        prompt_profile_state_hashes = {
            variant: hashlib.sha256(json.dumps({
                "prompts": [agent.current_prompt for agent in system.agents],
                "active_profiles": [
                    [asdict(answer) for answer in profile]
                    for profile in system.active_profiles
                ],
                "initial_profiles": [
                    [asdict(answer) for answer in profile]
                    for profile in system.initial_profiles
                ],
                "team_state_version": system.team_state_version,
                "accepted_state_count": system.accepted_state_count,
            }, sort_keys=True).encode()).hexdigest()
            for variant, system in {**systems, "evaluation": evaluator}.items()
        }
        if len(set(prompt_profile_state_hashes.values())) != 1:
            errors.append(f"parent_prompt_profile_state:{case['case_id']}")

        contexts = {}
        diagnostics = {}
        for variant, system in systems.items():
            context, diag = system._proposal_context(
                target,
                system.agents[target].current_prompt,
                generation_hashes(variant, frozen),
            )
            contexts[variant] = context
            diagnostics[variant] = diag
        canonical_m20 = system_for(
            case,
            setting="shared_responsibility_conditioned_dual_target",
            out_dir=scratch / str(case["case_id"]) / "canonical_m20",
            cache_path=scratch / "offline.sqlite",
        )
        canonical_context, _ = canonical_m20._proposal_context(
            target,
            canonical_m20.agents[target].current_prompt,
            frozen,
        )
        m20_context_byte_match = (
            serialize_context(contexts[M20])
            == serialize_context(canonical_context)
        )
        m20_teacher_byte_match = (
            build_teacher_request(
                contexts[M20], evolution_variant=M20_CURRENT_V15
            )
            == build_teacher_request(
                canonical_context, evolution_variant=M20_CURRENT_V15
            )
        )
        if not m20_context_byte_match or not m20_teacher_byte_match:
            errors.append(f"m20_byte_compatibility:{case['case_id']}")
        if not isinstance(contexts[G0], AccuracyDiagnosisContext):
            errors.append(f"g0_context:{case['case_id']}")
        if not isinstance(contexts[M20], SingleLaneDiagnosisContext):
            errors.append(f"m20_context:{case['case_id']}")
        leakage = (
            _leakage_violations(contexts[G0], frozen)
            if isinstance(contexts[G0], AccuracyDiagnosisContext)
            else ["wrong_context_type"]
        )
        if leakage:
            errors.append(f"g0_leakage:{case['case_id']}")

        budgets = {
            variant: {
                "candidate_count": system.protocol.candidates_per_target_branch,
                "teacher_critic_max_rounds": (
                    system.cfg.tcs.teacher_critic_max_rounds
                ),
                "teacher_json_max_retries": (
                    system.cfg.tcs.teacher_json_max_retries
                ),
                "critic_json_max_retries": (
                    system.cfg.tcs.critic_json_max_retries
                ),
                "student_invalid_max_retries": (
                    system.cfg.tcs.student_invalid_max_retries
                ),
                "student_upstream_regeneration_max_count": (
                    system.cfg.tcs.student_upstream_regeneration_max_count
                ),
            }
            for variant, system in systems.items()
        }
        if budgets[G0] != budgets[M20]:
            errors.append(f"budget_mismatch:{case['case_id']}")
        if evaluator.protocol.sample_pool_policy != "member_aware_residuals":
            errors.append(f"evaluation_pool:{case['case_id']}")
        if evaluator.protocol.candidate_acceptance_policy != (
            "fixed_peer_monotone_target_or_vote"
        ):
            errors.append(f"common_safe:{case['case_id']}")
        if evaluator.protocol.candidate_ranking_policy != "common_monotone_safe":
            errors.append(f"ranking:{case['case_id']}")
        cases.append({
            "case_id": case["case_id"],
            "target_agent_id": target,
            "parent_team_hash": case["parent_team_hash"],
            "responsibility_evidence_hash": expected_evidence_hash,
            "g0_generator_responsibility_hash_count": 0,
            "g0_responsibility_leakage": leakage,
            "g0_context_type": type(contexts[G0]).__name__,
            "m20_context_type": type(contexts[M20]).__name__,
            "m20_repair_case_hashes": sorted(
                row.question_hash for row in contexts[M20].repair_cases
            ) if isinstance(contexts[M20], SingleLaneDiagnosisContext) else [],
            "context_characters": {
                variant: int(diagnostics[variant].context_characters)
                for variant in (G0, M20)
            },
            "request_hashes": {
                variant: _request_hashes(contexts[variant])
                for variant in (G0, M20)
            },
            "m20_context_byte_match": m20_context_byte_match,
            "m20_teacher_request_byte_match": m20_teacher_byte_match,
            "budgets": budgets,
            "generation_policies": {
                variant: systems[variant].protocol.tcs_context_policy
                for variant in (G0, M20)
            },
            "evaluation_policy": {
                "sample_pool_policy": evaluator.protocol.sample_pool_policy,
                "stage_a_policy": evaluator.protocol.stage_a_policy,
                "candidate_acceptance_policy": (
                    evaluator.protocol.candidate_acceptance_policy
                ),
                "candidate_ranking_policy": (
                    evaluator.protocol.candidate_ranking_policy
                ),
            },
        })
    return {
        "preflight_version": "v16_generic_m20_preflight_v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "case_count": len(cases),
        "cell_count": len(cases) * 2,
        "candidate_count": len(cases) * 4,
        "g0_responsibility_leakage_count": sum(
            bool(row["g0_responsibility_leakage"]) for row in cases
        ),
        "m20_context_semantics_changed": False,
        "module1_semantics_changed": False,
        "common_safe_semantics_changed": False,
        "commit_enabled": False,
        "parent_mutation_enabled": False,
        "validation_enabled": False,
        "final_test_enabled": False,
        "api_calls": 0,
        "model_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--scratch",
        type=Path,
        default=ROOT / "runs/v16_generic_m20_preflight_only",
    )
    args = parser.parse_args()
    result = preflight(
        json.loads(args.registry.read_text(encoding="utf-8")),
        scratch=args.scratch.resolve(),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
