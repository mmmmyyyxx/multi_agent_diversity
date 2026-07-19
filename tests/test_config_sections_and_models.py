from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.config_sections import canonical_config_dict, canonical_field_registry
from multi_dataset_diverse_rl.core import CandidateRecord, MechanismRepresentation, QualityAnchor, QualityCounts
from multi_dataset_diverse_rl.strategy_registry import build_policy_bundle


def test_canonical_config_sections_have_one_owner_per_flat_field():
    cfg = Config(beam_size=2, joint_representative_beam_size=2)
    sections = canonical_config_dict(cfg)
    flattened = {key: value for section in sections.values() for key, value in section.items()}
    assert flattened == cfg.to_flat_dict()
    assert len(flattened) == len(canonical_field_registry(Config))
    assert sections["archive"]["beam_size"] == 2


def test_typed_stable_qd_records_round_trip():
    mechanism = MechanismRepresentation(
        canonical_operations=["hard_elimination"], embedding=[1.0, 0.0],
        family_kind="canonical", family_id="canonical:hard_elimination", specificity_score=1.0,
    )
    record = CandidateRecord.from_dict({
        "candidate_id": "c", "prompt": "p", "prompt_hash": "h", "archive_bucket": "safe",
        "metrics": {"candidate_type": "mechanism_alternative", "mechanism_representation": mechanism.to_dict()},
    })
    assert CandidateRecord.from_dict(record.to_dict()).to_dict() == record.to_dict()
    anchor = QualityAnchor("a", 1, ["h"], QualityCounts(vote=2, per_agent_correct=[1, 1]))
    assert QualityAnchor.from_dict(anchor.to_dict()) == anchor


def test_v8_policy_bundle_reuses_existing_public_identity():
    cfg = Config(
        method_version="v8_stable_qd_lineage", target_selector_version="hybrid_competence_boundary_v2",
        candidate_selection_mode="competence_depth_pareto", archive_policy_version="safe_probation_qd_archive_v1",
        active_team_selector_version="joint_quality_diversity_v1", lineage_policy_version="stable_lineage_anchor_v1",
    )
    bundle = build_policy_bundle(cfg)
    assert bundle.target_selector.name == "hybrid_competence_boundary_v2"
    assert bundle.archive_policy.name == "safe_probation_qd_archive_v1"
