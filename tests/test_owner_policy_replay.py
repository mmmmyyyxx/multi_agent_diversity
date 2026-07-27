from scripts.analyze_owner_policy_alignment import owner_replay


def _row(agent_id, *, improvement_need, gain):
    return {
        "agent_id": agent_id,
        "question_hash": "q",
        "M": 0,
        "member_error": True,
        "direct_vote_fix": False,
        "oracle_soft_utility_gain": 0.0,
        "coverage_opportunity": False,
        "dominant_wrong_member": False,
        "improvement_need": improvement_need,
        "gain_count": gain,
    }


def test_repair_only_policy_does_not_read_improvement_need():
    states = {0: {"q": [_row(0, improvement_need=0, gain=0), _row(1, improvement_need=99, gain=0)]}}
    first, _, _ = owner_replay(
        policy="B_repair_only", states=states, waits={0: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}}, seed=42,
    )
    changed = {0: {"q": [_row(0, improvement_need=99, gain=0), _row(1, improvement_need=0, gain=0)]}}
    second, _, _ = owner_replay(
        policy="B_repair_only", states=changed, waits={0: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}}, seed=42,
    )
    assert first == second


def test_relative_rank_is_late_and_outside_repair_only_pareto_vector():
    states = {0: {"q": [_row(0, improvement_need=0, gain=0), _row(1, improvement_need=0, gain=10)]}}
    _, rows, _ = owner_replay(
        policy="C_repair_only_relative_rank_late", states=states,
        waits={0: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}}, seed=42,
    )
    assert all(len(vector) == 4 for vector in rows[0]["candidate_vectors"].values())
    assert rows[0]["chosen_owner"] == 0
