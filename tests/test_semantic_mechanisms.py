from types import SimpleNamespace

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.lineage import empty_lineage_state, update_lineage_state
from multi_dataset_diverse_rl.mechanisms import normalize_mechanism_representation
from multi_dataset_diverse_rl.search_archive import cheap_prescreen, mechanism_is_novel
from multi_dataset_diverse_rl.system import TraceBeamSearchSystem


def item(name, step, embedding=None):
    representation = normalize_mechanism_representation(name, [step])
    if embedding is not None:
        representation["mechanism_embedding"] = list(embedding)
    return {
        "prompt": f"{name}.", "prompt_hash": name,
        "proposal": {"candidate_type": "mechanism_alternative", "mechanism_steps": [step]},
        "metrics": {"candidate_type": "mechanism_alternative", "mechanism_representation": representation},
    }


def test_specific_residual_only_mechanism_can_enter_alternative_search():
    candidate = item("coordinate", "Transform each coordinate into a rotation-invariant shape relation")
    representation = candidate["metrics"]["mechanism_representation"]
    assert representation["canonical_operations"] == []
    assert representation["family_kind"] == "semantic"
    assert cheap_prescreen(candidate, "parent", set()) == []
    assert mechanism_is_novel(candidate)


def test_generic_residual_only_mechanism_is_still_rejected():
    candidate = item("generic", "Reason carefully step by step")
    assert candidate["metrics"]["mechanism_representation"]["family_kind"] == "unknown"
    assert "missing_substantive_mechanism_operation" in cheap_prescreen(candidate, "parent", set())


def test_semantic_family_registry_merges_synonyms_but_separates_distinct_mechanisms():
    system = object.__new__(TraceBeamSearchSystem)
    system.cfg = Config(method_version="v8_stable_qd_lineage", semantic_niche_merge_threshold=0.88)
    system.mechanism_embedding_cache = {}
    system.semantic_mechanism_families = {}
    system.mechanism_embedding_cache_hit_count = 0
    system.mechanism_embedding_cache_miss_count = 0

    vectors = {
        "transform each coordinate into a rotation-invariant shape relation": [1.0, 0.0],
        "map coordinates into a shape relation invariant under rotation": [0.99, 0.01],
        "simulate a counterexample event against each rule": [0.0, 1.0],
    }

    class Encoder:
        def encode(self, rows, normalize_embeddings=True):
            return [vectors[row] for row in rows]

    system._load_embedding_model = lambda: Encoder()
    system._normalize_vector = lambda value: list(value)
    first = item("first", "Transform each coordinate into a rotation-invariant shape relation")
    synonym = item("synonym", "Map coordinates into a shape relation invariant under rotation")
    distinct = item("distinct", "Simulate a counterexample event against each rule")
    for candidate in (first, synonym, distinct):
        candidate["metrics"]["mechanism_steps"] = candidate["proposal"]["mechanism_steps"]
        system._attach_stable_mechanism_representation(candidate)
    first_id = first["metrics"]["mechanism_representation"]["family_id"]
    assert synonym["metrics"]["mechanism_representation"]["family_id"] == first_id
    assert distinct["metrics"]["mechanism_representation"]["family_id"] != first_id
    assert len(system.semantic_mechanism_families) == 2


def selected(name, family_id, embedding, correctness=(1, 0)):
    return {
        "prompt": name, "prompt_hash": name, "fold_behavior_stable": True,
        "mechanism_representation": {
            "normalized_operation_sequence": [], "semantic_residual_text": name,
            "family_kind": "semantic", "family_id": family_id, "mechanism_embedding": list(embedding),
        },
        "behavior_profile": {
            "correctness_vector": list(correctness), "error_vector": [1 - value for value in correctness],
            "rescue_vector": [0, 1], "accuracy": sum(correctness) / len(correctness),
        },
    }


def test_empty_canonical_sequences_do_not_collapse_into_one_lineage():
    cfg = Config(lineage_switch_confirmation_snapshots=2)
    state = empty_lineage_state()
    state.update({
        "lineage_status": "committed", "lineage_anchor_mechanism_family_id": "semantic:shape",
        "lineage_anchor_mechanism_embedding": [1.0, 0.0],
    })
    first = update_lineage_state(
        state, selected("event simulation", "semantic:event", [0.0, 1.0]),
        epoch=2, quality_gate_passed=True, config=cfg,
    )
    assert first["reason"] == "lineage_switch_pending"


def test_same_canonical_family_with_large_mechanism_drift_switches():
    cfg = Config(lineage_switch_confirmation_snapshots=2)
    state = empty_lineage_state()
    state.update({
        "lineage_status": "committed", "lineage_anchor_mechanism_signature": ["hard_elimination"],
        "lineage_anchor_mechanism_family_id": "canonical:hard", "lineage_anchor_mechanism_embedding": [1.0, 0.0],
    })
    candidate = selected("different elimination", "canonical:hard", [0.0, 1.0])
    candidate["mechanism_representation"]["normalized_operation_sequence"] = ["hard_elimination"]
    first = update_lineage_state(state, candidate, epoch=2, quality_gate_passed=True, config=cfg)
    assert first["reason"] == "lineage_switch_pending"
