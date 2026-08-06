from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.llm_client import LLMCallResult
from multi_dataset_diverse_rl.system import (
    CandidateFunnel,
    CandidateRuntime,
    PromptEnsembleOptimizationSystem,
)
from multi_dataset_diverse_rl.tcs import StudentPromptCandidate


def answer(value: str) -> PromptAnswer:
    return PromptAnswer(value, f"deterministic FINAL_ANSWER: {value}", True)


async def trajectory_solver(question: str, _agent_id: int, prompt: str) -> PromptAnswer:
    if "-improved" in prompt:
        return answer("A")
    if "-mid" in prompt:
        member_id = int(prompt.split("agent-", 1)[1].split("-", 1)[0])
        return answer("B" if question == f"q{member_id}_1" else "A")
    if "-base" in prompt:
        member_id = int(prompt.split("agent-", 1)[1].split("-", 1)[0])
        residual_owner = int(question.split("q", 1)[1].split("_", 1)[0])
        wrong_members = {
            residual_owner,
            (residual_owner + 1) % 5,
            (residual_owner + 2) % 5,
        }
        return answer("B" if member_id in wrong_members else "A")
    raise AssertionError(f"unexpected trajectory prompt: {prompt}")


def approved_critic_payload(system_prompt: str) -> str:
    return json.dumps({"failed_checks": [], "risk_case_ids": [], "feedback": ""})


def optimizer():
    max_tokens_seen: list[int | None] = []

    async def fake_optimizer(
        system_prompt: str,
        user_prompt: str,
        _temperature: float,
        _max_tokens: int | None,
    ) -> str:
        max_tokens_seen.append(_max_tokens)
        if "Check only explicit hard blockers" in system_prompt:
            return approved_critic_payload(system_prompt)
        payload = {
            "failure_pattern": "one residual member error remains",
            "repair_rule": (
                "Apply the verified decision rule before finalizing; abstain only "
                "when the explicit evidence remains insufficient."
            ),
            "preservation_rule": "Preserve every conclusion that still passes the verified rule.",
        }
        if system_prompt.startswith("Return strict JSON only."):
            parent = user_prompt.split("ParentPrompt:\n", 1)[1].split(
                "\nApprovedRepairPlan:", 1
            )[0]
            target = int(parent.split("agent-", 1)[1].split("-", 1)[0])
            suffix = "mid" if parent.endswith("-base") else "improved"
            return json.dumps({"candidate_prompts": [f"agent-{target}-{suffix}"]})
        return json.dumps(payload)

    fake_optimizer.max_tokens_seen = max_tokens_seen
    return fake_optimizer


def student_candidate(prompt: str) -> StudentPromptCandidate:
    return StudentPromptCandidate(candidate_prompt=prompt)


async def gate_solver(question: str, _agent_id: int, prompt: str) -> PromptAnswer:
    if prompt == "gate-vote-positive-target-regressing":
        return answer("A" if question == "q_vote" else "B")
    if prompt == "gate-vote-only-improved":
        return answer("A" if question in {"q_vote", "q_member_2"} else "B")
    member_id = int(prompt.split("gate-", 1)[1].split("-", 1)[0])
    if question == "q_vote":
        return answer("A" if member_id in {1, 2} else "B")
    return answer("A")


def call_result(
    text: str,
    *,
    finish_reason: str = "stop",
) -> LLMCallResult:
    completion_tokens = 1
    return LLMCallResult(
        text=text,
        prompt_tokens=1,
        completion_tokens=completion_tokens,
        total_tokens=completion_tokens + 1,
        latency_seconds=0.0,
        finish_reason=finish_reason,
    )


async def fault_smokes(data, prompts) -> dict[str, bool]:
    def system_for(name: str, fake_optimizer=None):
        cfg = Config.from_flat(
            out_dir=f"experiments/runs_deterministic_system_smoke_{name}",
            answer_format="option_letter",
            initialization_mode="provided_prompt_set",
            provided_prompts_json=json.dumps(prompts),
            num_candidates_per_parent=2,
            stage_b_candidate_budget=2,
            experiment_setting="shared_generic_evolution",
        )
        return PromptEnsembleOptimizationSystem(
            cfg, solver=trajectory_solver, optimizer_chat=fake_optimizer,
        )

    truncated = system_for("critic_truncation")

    async def truncating_chat(
        _model, system_prompt, _user_prompt, _temperature, max_tokens, _role,
    ):
        if "Check only explicit hard blockers" in system_prompt:
            return call_result("{", finish_reason="length")
        return call_result(json.dumps({
            "failure_pattern": "a residual reasoning check is skipped",
            "repair_rule": "Apply the explicit check and abstain if evidence remains tied.",
            "preservation_rule": "Keep conclusions that still pass every explicit check.",
        }))

    truncated._chat = truncating_chat
    await truncated.initialize_fixed_probe(data)
    trunc_funnel = CandidateFunnel()
    trunc_candidates = await truncated.propose_candidates(0, set(), trunc_funnel)

    rejection_calls = 0

    async def rejecting_optimizer(system_prompt, user_prompt, _temperature, _max_tokens):
        nonlocal rejection_calls
        if "Check only explicit hard blockers" in system_prompt:
            rejection_calls += 1
            if rejection_calls == 1:
                return json.dumps({
                    "failed_checks": ["actionable_specificity"],
                    "risk_case_ids": [],
                    "feedback": "Specify the verification order.",
                })
            return approved_critic_payload(system_prompt)
        if system_prompt.startswith("Return strict JSON only."):
            return json.dumps({"candidate_prompts": ["agent-0-mid"]})
        return json.dumps({
            "failure_pattern": "a residual reasoning check is skipped",
            "repair_rule": (
                "Apply each explicit constraint in order and abstain when the "
                "remaining candidates cannot be distinguished."
            ),
            "preservation_rule": "Keep conclusions that pass every explicit check.",
        })

    rejected = system_for("critic_rejection", rejecting_optimizer)
    await rejected.initialize_fixed_probe(data)
    rejection_funnel = CandidateFunnel()
    rejection_candidates = await rejected.propose_candidates(
        0, set(), rejection_funnel,
    )

    async def partial_optimizer(system_prompt, _user_prompt, _temperature, _max_tokens):
        if "Check only explicit hard blockers" in system_prompt:
            return approved_critic_payload(system_prompt)
        if system_prompt.startswith("Return strict JSON only."):
            return json.dumps({
                "candidate_prompts": ["agent-0-base", "agent-0-mid"]
            })
        return json.dumps({
            "failure_pattern": "a residual reasoning check is skipped",
            "repair_rule": "Apply the explicit check and abstain if evidence remains tied.",
            "preservation_rule": "Keep conclusions that pass every explicit check.",
        })

    partial = system_for("student_partial", partial_optimizer)
    await partial.initialize_fixed_probe(data)
    partial_funnel = CandidateFunnel()
    partial_candidates = await partial.propose_candidates(0, set(), partial_funnel)

    return {
        "critic_truncation": bool(
            not trunc_candidates
            and trunc_funnel.teacher_calls == 1
            and trunc_funnel.critic_truncated_responses == 2
            and trunc_funnel.student_calls == 0
            and trunc_funnel.terminal_failure_class
            == "critic_provider_truncation"
            and trunc_funnel.terminal_failure_role == "critic"
        ),
        "critic_semantic_rejection": bool(
            len(rejection_candidates) == 1
            and rejection_funnel.teacher_calls == 2
            and rejection_funnel.critic_semantic_rejections == 1
            and rejection_funnel.terminal_failure_class == ""
        ),
        "student_partial_validity": bool(
            len(partial_candidates) == 1
            and partial_funnel.student_calls == 1
            and partial_funnel.student_partially_valid_responses == 1
            and partial_funnel.terminal_failure_class == ""
        ),
    }


async def run_smoke() -> dict[str, object]:
    data = [
        {"question": f"q{agent_id}_{case_id}", "answer": "A"}
        for agent_id in range(5)
        for case_id in range(2)
    ]
    prompts = [f"agent-{agent_id}-base" for agent_id in range(5)]
    cfg = Config.from_flat(
        out_dir="experiments/runs_deterministic_system_smoke",
        answer_format="option_letter",
        initialization_mode="provided_prompt_set",
        provided_prompts_json=json.dumps(prompts),
        num_candidates_per_parent=1,
        stage_a_channel_top_k=1,
        stage_b_candidate_budget=1,
        stage_a_representative_size=10,
        stage_a_coverage_size=0,
        stage_a_conversion_size=0,
        stage_a_preservation_size=0,
        experiment_setting="shared_responsibility_conditioned_evolution",
        allow_legacy_setting=True,
    )
    fake_optimizer = optimizer()
    system = PromptEnsembleOptimizationSystem(
        cfg, solver=trajectory_solver, optimizer_chat=fake_optimizer
    )
    system.validation_probe = system.build_validation_probe(data)
    await system.initialize_fixed_probe(data)
    initial_validation = await system.evaluate_dataset(data, validation=True)
    system.ensure_responsibility_current()

    targets: list[int] = []
    refresh_deltas: list[int] = []
    for update_index in range(8):
        before_refresh = system.responsibility_refresh_count
        changed = await system.update_once(update_index)
        targets.append(system.candidate_decisions[-1]["target_agent_id"])
        refresh_deltas.append(system.responsibility_refresh_count - before_refresh)
        if not changed:
            break

    selected_validation = await system.evaluate_dataset(data, validation=True)
    validation_key = system.validation_key(
        selected_validation, initial_validation, epoch=1
    )
    transition_count = sum(row["funnel"]["accepted_candidate"] for row in system.candidate_decisions)

    gate_data = [
        {"question": "q_vote", "answer": "A"},
        {"question": "q_member_1", "answer": "A"},
        {"question": "q_member_2", "answer": "A"},
    ]
    gate_prompts = [f"gate-{agent_id}-base" for agent_id in range(5)]
    gate_cfg = Config.from_flat(
        out_dir="experiments/runs_deterministic_system_smoke_gate",
        answer_format="option_letter",
        initialization_mode="provided_prompt_set",
        provided_prompts_json=json.dumps(gate_prompts),
        num_candidates_per_parent=1,
        stage_a_channel_top_k=1,
        stage_b_candidate_budget=1,
        stage_a_representative_size=3,
        stage_a_coverage_size=0,
        stage_a_conversion_size=0,
        stage_a_preservation_size=0,
        experiment_setting="shared_responsibility_conditioned_evolution",
        allow_legacy_setting=True,
    )
    gate = PromptEnsembleOptimizationSystem(gate_cfg, solver=gate_solver)
    await gate.initialize_fixed_probe(gate_data)
    prompt = "gate-vote-positive-target-regressing"
    candidate = CandidateRuntime(
        student_candidate=student_candidate(prompt),
        prompt=prompt,
        prompt_hash=gate.prompt_hash(prompt),
        generation=1,
        parent_prompt_hash=gate.prompt_hash(gate.agents[0].current_prompt),
    )
    accepted, incumbent, evaluated = await gate.evaluate_candidates(
        0, [candidate], set(), CandidateFunnel()
    )
    candidate_evaluation = evaluated[0].final_evaluation
    incumbent_counts = incumbent.member_gain.incumbent_correct_counts
    candidate_counts = candidate_evaluation.member_gain.candidate_correct_counts
    vote_only_prompt = "gate-vote-only-improved"
    vote_only_candidate = CandidateRuntime(
        student_candidate=student_candidate(vote_only_prompt),
        prompt=vote_only_prompt,
        prompt_hash=gate.prompt_hash(vote_only_prompt),
        generation=1,
        parent_prompt_hash=gate.prompt_hash(gate.agents[0].current_prompt),
    )
    vote_only_accepted, _vote_only_incumbent, vote_only_evaluated = (
        await gate.evaluate_candidates(0, [vote_only_candidate], set(), CandidateFunnel())
    )
    vote_only_constraint = vote_only_evaluated[0].constraint
    fault_results = await fault_smokes(data, prompts)

    report = {
        "target_sequence": targets,
        "all_selected_targets_have_responsibility_portfolios": all(
            decision.get("target_agent_id") is None
            or decision.get("target_assigned_residual_count", 0) > 0
            for decision in system.candidate_decisions
        ),
        "compensation_paths_removed": (
            not hasattr(
                cfg.responsibility, "responsibility_" + "max_wait_updates"
            )
            and not hasattr(
                cfg.responsibility, "member_" + "catchup_mode"
            )
        ),
        "default_proposal_memory_disabled": (
            cfg.tcs.proposal_memory_mode == "off"
            and not any(
                event.get("memory_hit")
                for event in system.proposal_memory_events
            )
        ),
        "no_actionable_responsibility_is_noop": any(
            decision.get("stop_reason") == "no_actionable_responsibility"
            for decision in system.candidate_decisions
        ),
        "team_transition_count": transition_count,
        "responsibility_refresh_deltas": refresh_deltas,
            "one_refresh_per_team_transition": (
                system.responsibility_refresh_count == transition_count + 1
            ),
        "target_neutral_vote_positive_accepted": (
            vote_only_accepted is not None
            and vote_only_constraint is not None
            and vote_only_constraint.target_gain == 0
            and vote_only_constraint.vote_correct_candidate
            > vote_only_constraint.vote_correct_incumbent
            and vote_only_constraint.derived_team_pareto_passed
            and vote_only_constraint.terminal_invalid_nonregression_passed
            and vote_only_constraint.candidate_objective[1:]
            == vote_only_constraint.incumbent_objective[1:]
        ),
        "vote_positive_target_regression_rejected": (
            accepted is None
            and candidate_evaluation.marginal.net_vote_delta == 1
            and candidate_evaluation.member_gain.target_gain_vs_incumbent < 0
        ),
        "single_agent_replacement_preserves_other_member_counts": (
            candidate_counts[1:] == incumbent_counts[1:]
        ),
        "validation_key": validation_key,
        "real_validation_key_is_feasible": validation_key is not None,
        "stage_a_channels_seen": sorted(
            {
                channel
                for decision in system.candidate_decisions
                for row in decision["candidates"]
                for channel in (row["stage_a_decision"] or {}).get(
                    "selected_by_channels", []
                )
            }
        ),
        "typical_role_call_count": 3,
        "max_selected_pattern_count": max(
            (row["selected_pattern_count"] for row in system.tcs_context_history), default=0
        ),
        "max_selected_case_count": max(
            (row["selected_case_count"] for row in system.tcs_context_history), default=0
        ),
        "single_lane_context_only": all(
            row["context_type"] == "SingleLaneDiagnosisContext"
            and row["selected_pattern_count"] == 1
            and row["selected_case_count"] <= 3
            and row["context_characters"] <= 6000
            for row in system.tcs_context_history
        ),
        "service_portfolios_are_disjoint": (
            len({
                item.question_hash
                for rows in system.cached_service_portfolios.values()
                for item in rows
            })
            == sum(
                len(rows) for rows in system.cached_service_portfolios.values()
            )
        ),
        "student_raw_context_fields_seen": 0,
            "tcs_requests_use_provider_default": bool(
                not fake_optimizer.max_tokens_seen
                or all(value is None for value in fake_optimizer.max_tokens_seen)
            ),
        "solver_limit_remains_1800": bool(
            cfg.models.solver_max_tokens == 1800
            and all(
                row.get("configured_solver_max_tokens") == 1800
                for row in system.llm.calls
                if row["role"] == "solver"
            )
        ),
        "normal_funnel_audit_passed": all(
            decision["funnel"]["teacher_calls"] == 1
            and decision["funnel"]["critic_calls"] == 1
            and decision["funnel"]["student_calls"] == 1
            and decision["funnel"]["terminal_failure_class"] == ""
            and decision["funnel"]["raw_candidate_count"]
            <= decision["funnel"]["requested_candidate_count"]
            and decision["funnel"]["stage_a_evaluated"] >= 1
            and decision["funnel"]["stage_b_evaluated"] >= 1
            for decision in system.candidate_decisions
            if decision.get("update_lane") == "responsibility_conditioned"
        ),
        "fault_smokes": fault_results,
    }
    required = (
        "all_selected_targets_have_responsibility_portfolios",
        "compensation_paths_removed",
        "default_proposal_memory_disabled",
        "no_actionable_responsibility_is_noop",
        "one_refresh_per_team_transition",
        "target_neutral_vote_positive_accepted",
        "vote_positive_target_regression_rejected",
        "single_agent_replacement_preserves_other_member_counts",
        "real_validation_key_is_feasible",
        "tcs_requests_use_provider_default",
        "solver_limit_remains_1800",
        "normal_funnel_audit_passed",
        "single_lane_context_only",
        "service_portfolios_are_disjoint",
        *fault_results,
    )
    if not all(
        fault_results[key] if key in fault_results else report[key]
        for key in required
    ):
        raise SystemExit(f"deterministic system smoke failed: {report}")
    return report


def main() -> None:
    print(json.dumps({"system_smoke": asyncio.run(run_smoke())}, indent=2))


if __name__ == "__main__":
    main()
