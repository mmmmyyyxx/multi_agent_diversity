from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_v16_responsibility_coherence.py"
SPEC = importlib.util.spec_from_file_location("responsibility_coherence_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_single_item_portfolio_entropy_and_distance_are_zero():
    signature = {
        "coarse_signature": ("coverage", "1"),
        "fine_signature": ("coverage", "1", "dominant", "2"),
    }
    metrics = AUDIT.coherence_metrics([signature])
    assert metrics["coarse_entropy_norm"] == 0
    assert metrics["fine_entropy_norm"] == 0
    assert metrics["mean_pairwise_signature_distance"] == 0
    assert metrics["fine_largest_cluster_share"] == 1


def test_coherence_metrics_are_deterministic_and_use_four_field_hamming():
    rows = [
        {
            "coarse_signature": ("conversion", "1"),
            "fine_signature": ("conversion", "1", "dominant", "2"),
        },
        {
            "coarse_signature": ("conversion", "1"),
            "fine_signature": ("conversion", "1", "non_dominant", "1"),
        },
    ]
    assert AUDIT.coherence_metrics(rows) == AUDIT.coherence_metrics(list(reversed(rows)))
    assert AUDIT.coherence_metrics(rows)["mean_pairwise_signature_distance"] == 0.5


def test_tied_maximum_wrong_clusters_count_as_dominant():
    row = {
        "question_hash": "q",
        "gold_answer": "a",
        "team_answers": ["b", "b", "c", "c", "a"],
        "team_validity": [True] * 5,
        "team_correctness": [False, False, False, False, True],
        "gold_vote_count": 1,
        "largest_wrong_vote_count": 2,
        "plurality_margin": -1,
        "vote_correct": False,
    }
    for target in (0, 2):
        signature = AUDIT.residual_signature(row, target_agent_id=target, seed=48)
        assert signature["target_wrong_cluster_role"] == "dominant"
        assert signature["target_wrong_cluster_size_bin"] == "2"


def test_exact_repair_distance_uses_real_tie_as_abstain_aggregator():
    row = {
        "question_hash": "q",
        "gold_answer": "a",
        "team_answers": ["b", "b", "b", "c", "c"],
        "team_validity": [True] * 5,
        "team_correctness": [False] * 5,
        "gold_vote_count": 0,
        "largest_wrong_vote_count": 3,
        "plurality_margin": -3,
        "vote_correct": False,
    }
    signature = AUDIT.residual_signature(row, target_agent_id=0, seed=48)
    assert signature["repair_distance"] == 3
    assert signature["repair_distance_bin"] == "3+"


def test_frozen_coherence_and_entropy_boundaries():
    assert AUDIT.coherence_bin(0.75) == "HIGH"
    assert AUDIT.coherence_bin(0.50) == "MEDIUM"
    assert AUDIT.coherence_bin(0.499999) == "LOW"
    assert AUDIT.entropy_bin(0.33) == "LOW"
    assert AUDIT.entropy_bin(0.67) == "MEDIUM"
    assert AUDIT.entropy_bin(0.670001) == "HIGH"


def test_reader_prohibits_test_validation_and_nonallowlisted_artifacts(tmp_path):
    roots = {seed: tmp_path / f"seed{seed}" for seed in AUDIT.REQUIRED_SEEDS}
    for root in roots.values():
        root.mkdir()
    reader = AUDIT.TrainOnlyReader(roots)
    with pytest.raises(PermissionError, match="prohibited"):
        reader.json(48, "final_test_differentiation.json")
    with pytest.raises(PermissionError, match="prohibited"):
        reader.json(48, "validation_answers.json")
    with pytest.raises(PermissionError, match="not allowlisted"):
        reader.json(48, "history.json")
    assert reader.test_files_read == 1
    assert reader.validation_files_read == 1


def test_sanitization_rejects_sensitive_fields_and_absolute_paths():
    with pytest.raises(AssertionError, match="sensitive"):
        AUDIT._sanitize_value({"gold_answer": "a"})
    with pytest.raises(AssertionError, match="absolute path"):
        AUDIT._sanitize_value({"source": str(ROOT)})


def test_spearman_ties_are_deterministic():
    assert AUDIT.spearman([1, 2, 3], [0, 0, 1]) > 0
    assert AUDIT.spearman([1, 1], [0, 1]) is None


def test_active_repair_validation_is_separate_from_visible_focus():
    candidate = {
        "prompt_hash": "candidate",
        "evaluation": {"marginal": {"assigned_residual_repair_count": 2}},
    }
    parent = {
        "v1": {"team_correctness": [False] * 5, "gold_answer": "a"},
        "v2": {"team_correctness": [False] * 5, "gold_answer": "a"},
    }
    cache = {
        ("candidate", "v1"): {"valid": True, "answer": "a"},
        ("candidate", "v2"): {"valid": True, "answer": "a"},
    }
    visible = AUDIT._candidate_fixed_hashes(
        candidate=candidate,
        visible_hashes={"v1"},
        parent_by_hash=parent,
        answer_cache=cache,
        target_agent_id=0,
    )
    assert visible == {"v1"}
    assert AUDIT._validate_active_repair_count(
        candidate=candidate,
        active_hashes={"v1", "v2"},
        parent_by_hash=parent,
        answer_cache=cache,
        target_agent_id=0,
    ) == 2


def test_analysis_spec_guard_rejects_definition_drift():
    spec = {
        "spec_version": "v16_responsibility_coherence_analysis_v1",
        "frozen_before_candidate_outcome_analysis": True,
        "historical_seeds": list(AUDIT.REQUIRED_SEEDS),
        "trajectory_roots": AUDIT.EXPECTED_TRAJECTORY_ROOTS,
        "allowed_split": "train_optimization_probe_only",
        "primary_unit": "historical_m20_target_branch",
        "primary_portfolio": "visible_module2_evidence",
        "secondary_portfolio": "full_module1_service_portfolio_when_exactly_reconstructible",
        "coarse_signature_fields": ["failure_class", "repair_distance_bin"],
        "fine_signature_fields": [
            "failure_class",
            "repair_distance_bin",
            "target_wrong_cluster_role",
            "target_wrong_cluster_size_bin",
        ],
        "repair_distance": "exact_subset_enumeration_through_real_aggregator",
        "repair_distance_bins": {"1": [1], "2": [2], "3+": [3, 4, 5]},
        "coherence_bins": {
            "HIGH": {"min_inclusive": 0.75},
            "MEDIUM": {"min_inclusive": 0.5, "max_exclusive": 0.75},
            "LOW": {"max_exclusive": 0.5},
        },
        "entropy_bins": {
            "LOW": {"max_inclusive": 0.33},
            "MEDIUM": {"min_exclusive": 0.33, "max_inclusive": 0.67},
            "HIGH": {"min_exclusive": 0.67},
        },
        "branch_outcomes": [
            "any_valid_candidate", "any_repair_gain_candidate",
            "any_common_safe_feasible_candidate", "any_F_candidate",
            "any_target_regression_candidate", "best_feasible_candidate_count",
        ],
        "candidate_focus_strata": [
            "all_repair_gain", "common_safe_feasible_repair_gain",
            "committed_repair_gain",
        ],
        "coherence_classifier": {
            "outcome_association": "any_available_predeclared_directional_check_strictly_favorable",
            "directional_checks": [
                "spearman_cluster_share_feasible_gt_0",
                "spearman_entropy_F_gt_0",
                "spearman_entropy_target_regression_gt_0",
                "high_coherence_feasible_rate_gt_low",
                "high_entropy_F_rate_gt_low",
                "high_entropy_regression_rate_gt_low",
            ],
            "repair_concentration_min_candidates": 2,
            "repair_focus_mean_min": 0.75,
            "single_fine_signature_share_min": 0.5,
            "SUPPORTED": "outcome_association_and_repair_concentration",
            "PARTIAL": "exactly_one",
            "NOT_SUPPORTED": "neither",
        },
        "forbidden_path_tokens": list(AUDIT.FORBIDDEN_PATH_TOKENS),
        "no_p_values": True,
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
    }
    AUDIT.validate_analysis_spec(spec)
    spec["coherence_bins"]["HIGH"]["min_inclusive"] = 0.7
    with pytest.raises(ValueError, match="coherence_bins"):
        AUDIT.validate_analysis_spec(spec)


def test_active_slice_is_exact_lane_filtered_service_portfolio():
    snapshot = {
        "service_assignment_by_question": {
            "coverage": {"service_agent_id": 2, "repair_lane": "coverage"},
            "direct": {"service_agent_id": 2, "repair_lane": "direct_flip"},
            "peer": {"service_agent_id": 1, "repair_lane": "coverage"},
        }
    }
    assert AUDIT.active_service_hashes(
        snapshot, target_agent_id=2, active_lane="coverage"
    ) == {"coverage"}
    assert AUDIT._full_service_hashes(snapshot, 2) == {"coverage", "direct"}
    assert AUDIT.exact_branch_active_hashes(
        snapshot,
        {"active_lane": "coverage", "assigned_question_hashes": ["coverage"]},
        target_agent_id=2,
    ) == {"coverage"}
    with pytest.raises(AssertionError, match="exact lane-filtered"):
        AUDIT.exact_branch_active_hashes(
            snapshot,
            {"active_lane": "coverage", "assigned_question_hashes": ["direct"]},
            target_agent_id=2,
        )


def test_branch_audit_provenance_requires_exact_parent_state_and_identity():
    audit = {
        "update_index": 3,
        "target_agent_id": 2,
        "team_state_version": 1,
        "parent_team_hash": "parent",
        "active_lane": "coverage",
    }
    decision = {"parent_team_hash": "parent"}
    branch = {"active_lane": "coverage"}
    AUDIT.validate_branch_provenance(
        branch_audit=audit,
        decision=decision,
        branch=branch,
        update_index=3,
        target_agent_id=2,
        team_state_version=1,
    )
    audit["team_state_version"] = 2
    with pytest.raises(AssertionError, match="provenance mismatch"):
        AUDIT.validate_branch_provenance(
            branch_audit=audit,
            decision=decision,
            branch=branch,
            update_index=3,
            target_agent_id=2,
            team_state_version=1,
        )


def test_visible_hashes_require_current_single_lane_dominant_pattern():
    context = {
        "update_index": 3,
        "target_agent_id": 2,
        "context_type": "SingleLaneDiagnosisContext",
        "context_class": "SingleLaneDiagnosisContext",
        "context_mode": "member_aware_responsibility_conditioned",
        "diagnosis_aggregation_version": "single_lane_pattern_aggregation_v1",
        "selected_pattern_count": 1,
        "selected_pattern_ids": ["dominant"],
        "selected_context_pattern_question_hashes": {"dominant": ["q1", "q2"]},
    }
    assert AUDIT.visible_repair_hashes(
        context, update_index=3, target_agent_id=2
    ) == {"q1", "q2"}
    context["selected_context_pattern_question_hashes"]["other"] = ["q3"]
    with pytest.raises(ValueError, match="non-dominant"):
        AUDIT.visible_repair_hashes(context, update_index=3, target_agent_id=2)


def test_candidate_geometry_requires_fields_and_validates_persisted_value():
    candidate = {
        "constraint": {"target_gain": 1, "vote_net_gain": 0},
        "module2_diagnostics": {"candidate_geometry": "B"},
    }
    assert AUDIT._candidate_geometry(candidate) == "B"
    candidate["module2_diagnostics"]["candidate_geometry"] = "F"
    with pytest.raises(AssertionError, match="differs"):
        AUDIT._candidate_geometry(candidate)
    with pytest.raises(ValueError, match="canonical geometry fields"):
        AUDIT._candidate_geometry({"constraint": {"target_gain": 1}})
    candidate = {
        "constraint": {"target_gain": 1, "vote_net_gain": 0},
        "module2_diagnostics": {
            "candidate_geometry": "B",
            "target_gain": 0,
            "vote_net_gain": 0,
        },
    }
    with pytest.raises(AssertionError, match="inputs differ"):
        AUDIT._candidate_geometry(candidate)


def _cache_namespace(seed: int = 48):
    return {
        "model_request_identity": "request",
        "parser_version": "parser",
        "temperature": 0.0,
        "evaluation_replica_seed": seed,
        "solver_model": "model",
        "max_tokens": 1800,
        "output_contract_version": "contract",
    }


def _make_cache(path: Path, rows: list[tuple[str, str, str, str]]):
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE cache_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "INSERT INTO cache_metadata VALUES ('schema_version', 'shared_solver_cache_v1')"
    )
    connection.execute(
        "CREATE TABLE solver_cache (cache_key TEXT PRIMARY KEY, state TEXT, "
        "model_request_identity TEXT, solver_model TEXT, endpoint_identity TEXT, "
        "output_contract_version TEXT, parser_version TEXT, temperature REAL, "
        "max_tokens INTEGER, evaluation_replica_seed INTEGER, prompt_hash TEXT, "
        "question_hash TEXT, answer_json TEXT)"
    )
    for cache_key, prompt_hash, question_hash, request_identity in rows:
        connection.execute(
            "INSERT INTO solver_cache VALUES (?, 'ready', ?, 'model', 'endpoint', "
            "'contract', 'parser', 0.0, 1800, 48, ?, ?, ?)",
            (
                cache_key,
                request_identity,
                prompt_hash,
                question_hash,
                json.dumps(
                    {"answer": "a", "valid": True, "request_identity": request_identity}
                ),
            ),
        )
    connection.commit()
    connection.close()


def test_solver_cache_reader_uses_exact_namespace(tmp_path):
    roots = {seed: tmp_path / f"seed{seed}" for seed in AUDIT.REQUIRED_SEEDS}
    for root in roots.values():
        root.mkdir()
    _make_cache(
        roots[48] / "_solver_cache.sqlite",
        [("right", "prompt", "q", "request"), ("wrong", "prompt", "q", "other")],
    )
    answers = AUDIT.TrainOnlyReader(roots).candidate_answers(
        48, {"prompt"}, namespace=_cache_namespace()
    )
    assert set(answers) == {("prompt", "q")}
    assert answers[("prompt", "q")]["request_identity"] == "request"


def test_solver_cache_reader_fails_on_duplicate_observations(tmp_path):
    roots = {seed: tmp_path / f"seed{seed}" for seed in AUDIT.REQUIRED_SEEDS}
    for root in roots.values():
        root.mkdir()
    _make_cache(
        roots[48] / "_solver_cache.sqlite",
        [("one", "prompt", "q", "request"), ("two", "prompt", "q", "request")],
    )
    with pytest.raises(ValueError, match="duplicate solver observations"):
        AUDIT.TrainOnlyReader(roots).candidate_answers(
            48, {"prompt"}, namespace=_cache_namespace()
        )
