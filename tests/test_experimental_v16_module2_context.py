from __future__ import annotations

from dataclasses import replace

from multi_dataset_diverse_rl.candidate_selection import evaluate_constraints
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.evaluation.fixed_probe import ProbeExample
from multi_dataset_diverse_rl.module2_context import (
    C0_CURRENT_V15,
    C2_BOUNDARY_PLUS_PRESERVATION,
    C3_COALITION_AWARE_PRESERVATION,
    REPAIR_SET_MAX,
    PRESERVATION_SET_MAX,
    build_module2_context_sets,
    exact_repair_distance,
)
from multi_dataset_diverse_rl.peer_state import build_team_vote_state
from multi_dataset_diverse_rl.protocol import (
    EXPERIMENTAL_V16_MODULE2_SETTINGS,
    candidate_budget_contract,
    experiment_protocol,
)
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.tcs import (
    ExperimentalModule2DiagnosisContext,
    CompactPreviousOutcome,
    context_payload,
)


def team(qid: str, answers: list[str], gold: str = "G"):
    return build_team_vote_state(
        question_hash=qid,
        gold_answer=gold,
        answers=answers,
        valid_vector=[True] * 5,
        tie_break="abstain",
        seed=7,
    )


def sets(rows, assigned, stable=(), accepted_states=1):
    examples = tuple(
        ProbeExample(row.question_hash, row.question_hash, row.gold_answer)
        for row in rows
    )
    return build_module2_context_sets(
        examples=examples,
        states=rows,
        target_agent_id=0,
        assigned_question_hashes=set(assigned),
        stable_correct_question_hashes=set(stable),
        accepted_state_count=accepted_states,
        normalize_answer=lambda value: str(value or "").strip(),
        match_answer=lambda prediction, gold: prediction == gold,
        tie_break="abstain",
        seed=7,
    )


def test_exact_repair_distance_uses_plurality_and_tie_as_abstain():
    state = team("q", ["W", "W", "X", "Y", "Z"])
    assert state.top_tie is False
    assert exact_repair_distance(
        state,
        normalize_answer=str.strip,
        match_answer=lambda prediction, gold: prediction == gold,
        tie_break="abstain",
        seed=7,
    ) == 2
    tie = team("tie", ["G", "G", "W", "W", "X"])
    assert tie.top_tie and not tie.vote_correct and tie.plurality_margin == 0


def test_repair_eligibility_order_budget_and_metadata():
    rows = [
        team("r4", ["W", "W", "W", "W", "W"]),
        team("r2", ["W", "G", "W", "W", "X"]),
        team("r1", ["W", "G", "G", "W", "W"]),
        team("r3", ["W", "W", "X", "Y", "Z"]),
        team("vote-correct", ["W", "G", "G", "G", "X"]),
        team("target-correct", ["G", "W", "W", "X", "Y"]),
    ]
    result = sets(rows, {row.question_hash for row in rows})
    assert [row.tier[:2] for row in result.repair] == ["R1", "R2", "R3", "R4"]
    assert [row.question_hash for row in result.repair] == ["r1", "r2", "r3", "r4"]
    assert "vote-correct" not in {row.question_hash for row in result.repair}
    assert "target-correct" not in {row.question_hash for row in result.repair}
    assert result.repair[0].boundary_class == "one_repair_away"
    assert result.repair[0].target_role == "boundary_closing_member"
    assert len(result.repair) <= REPAIR_SET_MAX


def test_hash_tie_break_and_repair_budget_six():
    rows = [team(f"q{index:02d}", ["W", "G", "G", "W", "W"]) for index in range(9)]
    result = sets(rows, {row.question_hash for row in rows})
    assert [row.question_hash for row in result.repair] == [f"q{index:02d}" for index in range(6)]


def test_preservation_p1_p2_p3_order_and_budget():
    rows = [
        team("p1", ["G", "G", "W", "X", "Y"]),
        team("p2", ["G", "G", "G", "W", "X"]),
        team("p3", ["G", "W", "W", "X", "Y"]),
        *[team(f"stable-{index}", ["G", "W", "X", "Y", "Z"]) for index in range(8)],
    ]
    stable = {row.question_hash for row in rows}
    result = sets(rows, set(), stable, accepted_states=4)
    assert result.preservation[0].tier == "P1_VOTE_CRITICAL"
    assert result.preservation[1].tier == "P2_COALITION_SUPPORT"
    assert result.preservation[2].tier == "P3_STABLE_COMPETENCE"
    assert result.preservation[2].observed_correct_state_count == 4
    assert len(result.preservation) == PRESERVATION_SET_MAX


def test_c2_c3_membership_identity_and_metadata_isolation():
    built = sets(
        [team("r1", ["W", "G", "G", "W", "W"]), team("p1", ["G", "G", "W", "X", "Y"])],
        {"r1"},
        {"p1"},
    )
    common = dict(
        parent_prompt="parent",
        parent_prompt_hash="hash",
        repair_cases=built.repair,
        preservation_cases=built.preservation,
        previous_outcome=CompactPreviousOutcome("none", None),
    )
    c2 = context_payload(ExperimentalModule2DiagnosisContext(context_variant=C2_BOUNDARY_PLUS_PRESERVATION, **common))
    c3 = context_payload(ExperimentalModule2DiagnosisContext(context_variant=C3_COALITION_AWARE_PRESERVATION, **common))
    assert [row["case_id"] for row in c2["repair_responsibilities"]] == [row["case_id"] for row in c3["repair_responsibilities"]]
    assert "repair_distance" not in c2["repair_responsibilities"][0]
    assert c3["repair_responsibilities"][0]["repair_distance"] == 1
    assert "preservation_tier" not in c2["preservation_responsibilities"][0]
    assert c3["preservation_responsibilities"][0]["preservation_tier"].startswith("P1")


def test_experimental_settings_are_identity_distinct_and_canonical_matrix_unchanged():
    variants = []
    for name in EXPERIMENTAL_V16_MODULE2_SETTINGS:
        budget = candidate_budget_contract(
            name,
            candidates_per_target_branch=2,
            stage_b_budget_per_branch=2,
            stage_a_channel_top_k=2,
            representative_size=12,
            coverage_size=6,
            conversion_size=6,
            preservation_size=4,
        )
        protocol = experiment_protocol(
            name,
            initialization_mode="shared_identical",
            tie_policy="abstain",
            candidate_budget_contract=budget,
        )
        variants.append(protocol.module2_context_variant)
        assert protocol.target_branch_count == 2
        assert protocol.candidates_per_target_branch == 2
        assert protocol.candidate_acceptance_policy == "fixed_peer_monotone_target_or_vote"
    assert variants == [C0_CURRENT_V15, C2_BOUNDARY_PLUS_PRESERVATION, C3_COALITION_AWARE_PRESERVATION]


def test_setting_variant_mismatch_is_rejected_and_memory_defaults_off(tmp_path):
    cfg = Config.from_flat(
        out_dir=str(tmp_path),
        experiment_setting="experimental_v16_c2_boundary_plus_preservation",
        module2_context_variant=C2_BOUNDARY_PLUS_PRESERVATION,
    )
    system = PromptEnsembleOptimizationSystem(cfg)
    assert system.cfg.tcs.proposal_memory_mode == "off"
    bad = replace(cfg, tcs=replace(cfg.tcs, module2_context_variant=C0_CURRENT_V15))
    try:
        PromptEnsembleOptimizationSystem(bad)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("variant mismatch must fail")


def test_preservation_is_diagnostic_not_an_acceptance_guard():
    assert "P1" not in evaluate_constraints.__code__.co_names
    assert "preservation" not in evaluate_constraints.__code__.co_names
