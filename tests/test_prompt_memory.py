from multi_dataset_diverse_rl.prompt_memory import (
    PROMPT_MEMORY_SLOTS, contribution_signature, rebuild_prompt_memory, select_generation_parents,
)


def item(name, feasible=True, accuracy=0, vote=0, responsibility=0.0):
    return {"prompt_hash": name, "prompt": name, "metrics": {
        "constraints_passed": feasible, "candidate_target_correct_count": accuracy,
        "net_vote_delta": vote, "assigned_residual_utility_delta": responsibility,
    }}


def test_five_slots_and_active_and_rollback_semantics():
    active = item("active", feasible=False)
    competence = item("competence", accuracy=9)
    ensemble = item("ensemble", vote=2)
    responsibility = item("responsibility", responsibility=3.0)
    rollback = item("rollback", feasible=False)
    memory = rebuild_prompt_memory(
        [active, competence, ensemble, responsibility], active_prompt_hash="active", previous_active_item=rollback,
    )
    assert set(PROMPT_MEMORY_SLOTS) == {
        "active", "competence_best", "ensemble_best", "responsibility_best", "rollback",
    }
    assert memory[0]["prompt_memory_slot"] == "active"
    assert next(row for row in memory if row["prompt_hash"] == "rollback")["prompt_memory_slot"] == "rollback"
    assert "rollback" not in {row["prompt_hash"] for row in select_generation_parents(memory)}


def test_rollback_becomes_parent_only_after_current_constraints_pass():
    active = item("active", feasible=False)
    rollback = item("rollback", feasible=True)
    memory = rebuild_prompt_memory(
        [active, rollback], active_prompt_hash="active", previous_active_item=rollback,
    )
    assert "rollback" in {row["prompt_hash"] for row in select_generation_parents(memory)}


def test_answer_aware_signature_changes_with_wrong_answer():
    common = dict(
        fixed_probe_hash="probe", question_hashes=["q"], correctness_vector=[False], invalid_vector=[False],
        vote_contribution_vector=[0], coverage_contribution_vector=[0], unique_correct_vector=[False],
        pivotal_correct_vector=[False], dominant_wrong_membership_vector=[True],
    )
    assert contribution_signature(answer_hashes=["B"], **common) != contribution_signature(answer_hashes=["C"], **common)
