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
from multi_dataset_diverse_rl.system import (
    CandidateFunnel,
    CandidateRuntime,
    PromptEnsembleOptimizationSystem,
)
from multi_dataset_diverse_rl.tcs import StudentCandidate


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
        return answer(
            "B" if question in {f"q{member_id}_0", f"q{member_id}_1"} else "A"
        )
    raise AssertionError(f"unexpected trajectory prompt: {prompt}")


def approved_critic_payload(system_prompt: str) -> str:
    facts = json.loads(
        system_prompt.split("DERIVED_CASE_FACTS:\n", 1)[1].split(
            "\nProposalContext:", 1
        )[0]
    )
    return json.dumps(
        {
            "case_fact_restatements": facts,
            "context_consistent": True,
            "sample_memorization_free": True,
            "executable_change": True,
            "internally_consistent": True,
            "preservation_rule_present": True,
            "output_contract_safe": True,
            "peer_copying_free": True,
            "stereotype_forcing_free": True,
            "non_generic_change": True,
            "blocking_reasons": [],
            "soft_concerns": [],
            "score": 0.1,
            "feedback": "deterministic approval",
        }
    )


def optimizer():
    async def fake_optimizer(
        system_prompt: str,
        user_prompt: str,
        _temperature: float,
        _max_tokens: int,
    ) -> str:
        if "Audit the Teacher" in system_prompt:
            return approved_critic_payload(system_prompt)
        payload = {
            "observed_failure_pattern": "one residual member error remains",
            "generalizable_mechanism": "the current decision rule misses a valid case",
            "decision_rule": "apply the verified rule before finalizing",
            "uncertainty_or_abstention_rule": "abstain only when evidence is insufficient",
            "preservation_conditions": "preserve every previously correct decision",
            "evidence_summary": "the fixed probe exposes a deterministic residual error",
        }
        if system_prompt == "Return strict JSON only.":
            context = json.loads(
                user_prompt.split("ProposalContext:\n", 1)[1].split(
                    "\nApprovedTeacherProposal:", 1
                )[0]
            )
            target = int(context["target_agent_id"])
            suffix = "mid" if context["parent_prompt"].endswith("-base") else "improved"
            prompt = f"agent-{target}-{suffix}"
            return json.dumps(
                {"candidates": [{"candidate_prompt": prompt, **payload}]}
            )
        return json.dumps(payload)

    return fake_optimizer


def student_candidate(prompt: str) -> StudentCandidate:
    return StudentCandidate(
        candidate_prompt=prompt,
        observed_failure_pattern="vote repair can hide member regression",
        generalizable_mechanism="paired counterfactual replacement",
        decision_rule="repair the vote",
        uncertainty_or_abstention_rule="abstain if unsupported",
        preservation_conditions="preserve member competence",
        evidence_summary="deterministic gate case",
    )


async def gate_solver(question: str, _agent_id: int, prompt: str) -> PromptAnswer:
    if prompt == "gate-vote-positive-member-regressing":
        return answer("A" if question == "q_vote" else "B")
    member_id = int(prompt.split("gate-", 1)[1].split("-", 1)[0])
    if question == "q_vote":
        return answer("A" if member_id in {1, 2} else "B")
    return answer("A")


async def run_smoke() -> dict[str, object]:
    data = [
        {"question": f"q{agent_id}_{case_id}", "answer": "A"}
        for agent_id in range(5)
        for case_id in range(2)
    ]
    prompts = [f"agent-{agent_id}-base" for agent_id in range(5)]
    cfg = Config.from_flat(
        out_dir="runs_deterministic_system_smoke",
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
    )
    system = PromptEnsembleOptimizationSystem(
        cfg, solver=trajectory_solver, optimizer_chat=optimizer()
    )
    system.validation_probe = system.build_validation_probe(data)
    await system.initialize_fixed_probe(data)
    initial_validation = await system.evaluate_dataset(data, validation=True)
    system.ensure_responsibility_current()

    targets: list[int] = []
    refresh_deltas: list[int] = []
    minimum_gain_deltas: list[int] = []
    vote_count_deltas: list[int] = []
    for update_index in range(8):
        before_refresh = system.responsibility_refresh_count
        before_minimum = system.current_team_member_gain_state()["minimum_gain_count"]
        before_vote_count = system.active_probe_metrics().vote_correct_count
        changed = await system.update_once(update_index)
        after_minimum = system.current_team_member_gain_state()["minimum_gain_count"]
        after_vote_count = system.active_probe_metrics().vote_correct_count
        targets.append(system.candidate_decisions[-1]["target_agent_id"])
        refresh_deltas.append(system.responsibility_refresh_count - before_refresh)
        minimum_gain_deltas.append(after_minimum - before_minimum)
        vote_count_deltas.append(after_vote_count - before_vote_count)
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
        out_dir="runs_deterministic_system_smoke_gate",
        answer_format="option_letter",
        initialization_mode="provided_prompt_set",
        provided_prompts_json=json.dumps(gate_prompts),
        stage_a_channel_top_k=1,
        stage_b_candidate_budget=1,
        stage_a_representative_size=3,
        stage_a_coverage_size=0,
        stage_a_conversion_size=0,
        stage_a_preservation_size=0,
    )
    gate = PromptEnsembleOptimizationSystem(gate_cfg, solver=gate_solver)
    await gate.initialize_fixed_probe(gate_data)
    prompt = "gate-vote-positive-member-regressing"
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

    report = {
        "target_sequence": targets,
        "all_eligible_selected_within_8": set(targets) == set(range(5)),
        "team_transition_count": transition_count,
        "responsibility_refresh_deltas": refresh_deltas,
        "one_refresh_per_team_transition": (
            system.responsibility_refresh_count == transition_count + 1
            and all(delta == 1 for delta in refresh_deltas)
        ),
        "vote_neutral_worst_member_positive_accepted": (
            transition_count == 8
            and any(
                minimum_delta == 1 and vote_delta == 0
                for minimum_delta, vote_delta in zip(
                    minimum_gain_deltas, vote_count_deltas, strict=True
                )
            )
        ),
        "vote_positive_member_regressing_rejected": (
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
    }
    required = (
        "all_eligible_selected_within_8",
        "one_refresh_per_team_transition",
        "vote_neutral_worst_member_positive_accepted",
        "vote_positive_member_regressing_rejected",
        "single_agent_replacement_preserves_other_member_counts",
        "real_validation_key_is_feasible",
    )
    if not all(report[key] for key in required):
        raise SystemExit(f"deterministic system smoke failed: {report}")
    return report


def main() -> None:
    print(json.dumps({"system_smoke": asyncio.run(run_smoke())}, indent=2))


if __name__ == "__main__":
    main()
