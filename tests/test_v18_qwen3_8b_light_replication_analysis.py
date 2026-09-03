from scripts.analyze_v18_qwen3_8b_light_replication import (
    ARMS,
    EXPECTED_MODEL,
    RUNTIME_VERSION,
    SEED,
)


def test_postrun_analysis_scope_is_frozen():
    assert SEED == 71
    assert ARMS == ("A_CANONICAL", "C_NO_SEMANTIC_CRITIC")
    assert EXPECTED_MODEL == {
        "solver": "qwen3-8b",
        "teacher": "qwen3.7-flash",
        "critic": "qwen3.7-flash",
        "student": "qwen3.7-flash",
        "thinking": False,
    }
    assert RUNTIME_VERSION == "v18_qwen3_8b_no_semantic_critic_light_replication_v1"
