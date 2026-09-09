from multi_dataset_diverse_rl.experimental_rg_gepa import (
    CandidateScore, EvidenceItem, RGGEPAProtocol, RG_GEPA_LEDGER_VERSION, TeamVector, deterministic_minibatch,
    progressive_promotions, team_pareto_winner, validate_ledger_record,
)


def score(identifier: str, values: tuple[int, int, int, int, int]) -> CandidateScore:
    return CandidateScore(identifier, TeamVector(*values), 0)


def test_progressive_gate_requires_pareto_improvement() -> None:
    parent = TeamVector(5, 5, 20, 5, 0)
    rows = [score("loss", (5, 4, 20, 5, 1)), score("gain", (5, 5, 21, 5, 1))]
    assert [row.candidate_id for row in progressive_promotions(parent=parent, candidates=rows)] == ["gain"]


def test_team_pareto_tie_break_is_deterministic() -> None:
    winner, frontier = team_pareto_winner([score("a", (5, 6, 20, 5, 1)), score("b", (6, 5, 20, 5, 1))])
    assert {row.candidate_id for row in frontier} == {"a", "b"}
    assert winner and winner.candidate_id == "b"


def test_minibatch_prioritizes_evidence_and_correct_filler() -> None:
    rows = [EvidenceItem(str(i), "coverage" if i < 2 else "filler", "coverage", i % 2 == 0, True, 1, str(i)) for i in range(12)]
    batch = deterministic_minibatch(rows)
    assert len(batch) == 12
    assert [item.example_id for item in batch[:2]] == ["0", "1"]


def test_protocol_and_ledger_are_fail_closed() -> None:
    assert len(RGGEPAProtocol().identity()) == 64
    record = {
        "ledger_version": RG_GEPA_LEDGER_VERSION,
        "seed": 1, "parent_id": "p", "update_index": 0, "candidate_id": "c",
        "proposal_engine": "gepa_reflection", "evaluation_stage": "reflection",
        "input_tokens": 1, "output_tokens": 2, "total_tokens": 3,
        "provider_attempt_id": "a", "logical_call_id": "logical-a", "attempt_index": 1,
        "record_kind": "optimizer_provider_attempt", "provider_attempts": 1,
        "successful_provider_calls": 1, "cache_hit": False,
        "logical_role": "reflection", "client_role": "optimizer", "success": True,
    }
    validate_ledger_record(record)
