"""Generate the sanitized, zero-provider saturation refactor audit bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "saturation_mode_refactor_20260922"
PREP = ROOT / "reports" / "saturation_mode_prep" / "stopping_path_audit.json"


def write_json(name: str, value) -> None:
    (OUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PREP, OUT / "stopping_path_audit.json")
    write_json("saturation_contract.json", {
        "contract_version": "backend_neutral_saturation_stopping_v1",
        "run_modes": ["fixed_budget", "saturation"],
        "saturation": {
            "enabled": True,
            "scientific_budget_enabled": False,
            "normal_stop": "SATURATION_REACHED",
            "rule": "NO_ACCEPTED_UPDATE_FOR_K_COMPLETE_OPTIMIZATION_UNITS",
        },
        "accepted_update": {
            "native": "strict backend-native deployable state change",
            "layer2_local": "Layer1 strict local accepted update",
            "layer2_outer": "successful atomic TEAM_COMMIT",
            "equal_score_archive_only_resets": False,
            "vote_neutral_common_safe_commit_resets": True,
        },
        "canonical_stop_reasons": [
            "SATURATION_REACHED", "SCIENTIFIC_BUDGET_REACHED",
            "EMERGENCY_PROVIDER_CALL_CEILING", "EMERGENCY_OPTIMIZER_STEP_CEILING",
            "EMERGENCY_TEAM_EPOCH_CEILING", "EMERGENCY_WALL_TIME_CEILING",
            "PROVIDER_FAILURE", "OPERATIONAL_ABORT", "INVALID_STATE",
            "INSUFFICIENT_LAYER2_EVIDENCE",
        ],
    })
    write_json("four_mode_semantics.json", {
        "GEPA_NATIVE": {
            "unit": "complete pinned EpochShuffledBatchSampler traversal",
            "acceptance": "pinned strict subsample improvement and accepted candidate",
            "equal_score": "rejected",
            "saturation_ready": True,
        },
        "GEPA_LAYER2": {
            "unit": "complete frozen ordered evidence schedule pass",
            "epoch_policy": "replay_same_frozen_packet_v1",
            "native_sampler_fallback": False,
            "saturation_ready": True,
        },
        "MARS_NATIVE": {
            "unit": "complete Teacher/Critic/Student/Target round",
            "strict_deployable_update": "score exceeds best accepted deployable score",
            "native_stability_check_retained": True,
            "saturation_ready": True,
        },
        "MARS_LAYER2": {
            "unit": "packet-owned Planner/Teacher/Critic/Student/Target round",
            "global_dataset_access": False,
            "saturation_ready": True,
        },
    })
    write_json("team_epoch_semantics.json", {
        "version": "scheduler_eligible_coverage_v1",
        "eligible_members": "all unique members in scheduler summaries at epoch start",
        "zero_score_members": "remain eligible under existing always-two/fallback policy",
        "selection": "observe existing scheduler decisions; no re-ranking",
        "boundary": "every frozen eligible member has received >=1 opportunity",
        "member_may_repeat": True,
        "persistent_realizability_reset_at_epoch": False,
        "team_patience_reset": "successful atomic commit only",
    })
    write_json("hidden_budget_audit.json", {
        "scientific_budgets_disableable": True,
        "saturation_scientific_budget_enabled": False,
        "poison_values_tested": {"max_metric_calls": 1, "native_unit_limit": 1,
                                  "optimizer_call_limit": 1},
        "mars_native_ignored": True,
        "mars_layer2_ignored": True,
        "gepa_max_metric_calls_passed_to_pinned_engine": None,
        "fixed_budget_path_unchanged": True,
        "emergency_ceilings_active": True,
    })
    write_json("packet_exhaustion_audit.json", {
        "policy": "replay_same_frozen_packet_v1",
        "one_batch_test": "PASS",
        "schedule_exhaustion_semantics": "END_OF_LOCAL_EVIDENCE_EPOCH",
        "full_run_termination": False,
        "native_fallback": False,
        "packet_mutation": False,
    })
    write_json("persistent_realizability_audit.json", {
        "owner": "existing Layer2 scheduler",
        "selected_no_commit": "existing valid-outcome failure update remains unchanged",
        "teammate_commit": "does not reset another member",
        "team_epoch_boundary_reset": False,
        "saturation_controller_mutates_f_i_or_R_i": False,
    })
    write_json("focus_anchor_audit.json", {
        "owner": "existing atomic write-back/transition pipeline",
        "successful_team_commit": "latest committed transition advances exactly once",
        "local_accepted_without_commit": "no focus/anchor advance",
        "new_evidence_epoch_without_commit": "same frozen packet replay",
        "saturation_controller_mutates_transition": False,
    })
    write_json("emergency_ceiling_audit.json", {
        "mandatory": ["provider_calls", "optimizer_steps", "team_epochs", "wall_seconds"],
        "scientific_convergence": False,
        "execution_classification": "EXECUTION_ABORTED",
        "test": "emergency ceiling != SATURATION_REACHED: PASS",
        "oscillation": {
            "state_hashes_persisted": True, "accepted_update_hashes_persisted": True,
            "revisited_state_count": True,
            "diagnostic_only": "OSCILLATORY_SEARCH_SUSPECTED",
        },
    })
    write_json("cost_accounting_schema.json", {
        "fields": [
            "solver_calls", "optimizer_calls", "reflection_calls", "planner_calls",
            "teacher_calls", "critic_calls", "student_calls", "team_evaluation_calls",
            "provider_attempts", "provider_successes", "provider_failures",
            "prompt_tokens", "completion_tokens", "total_tokens", "wall_seconds",
        ],
        "scientific_budget_independent": True,
        "raw_prompts_or_responses": False,
    })
    write_json("patience_options.json", {
        "historical_inputs": [
            {"source": "seed78_gepa_differential_audit_20260914",
             "proposal_attempts": 66, "accepted_mutations": 29,
             "scope": "independent GEPA capacity; diagnostic only"},
            {"source": "diversity_writeback_cost_mechanism_audit_20260908",
             "opportunities": 33, "commits": 13,
             "scope": "historical Diversity write-back; diagnostic only"},
            {"source": "level_b_gepa_real_canary_v2_stagefix2_transportfix1_authorized1",
             "proposal_attempts": 3, "accepted_mutations": 0,
             "scope": "small real-provider canary; diagnostic only"},
        ],
        "local_options": [2, 3, 5],
        "team_options": [1, 2, 3],
        "recommended_initial_local": 3,
        "recommended_initial_team": 2,
        "rationale": "middle option limits one-unit noise without requiring five empty passes; freeze before formal API use",
        "selected_from_future_results": False,
    })
    write_json("provider_isolation_audit.json", {
        "real_provider_attempts": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
        "test_environment": "provider keys and endpoint variables explicitly empty",
        "provider_client_used_by_new_tests": False,
        "status": "PASS",
    })
    write_json("test_summary.json", {
        "focused_saturation_tests": "13 passed",
        "preexisting_four_mode_tests": "54 passed",
        "full_tests": "1174 passed, 15 failed, 13 errors",
        "full_test_nonpassing_class": "UNCHANGED_MISSING_HISTORICAL_PRIVATE_ARTIFACTS",
        "representative_missing_inputs": [
            "runs/v15f48/.../run_meta.json",
            "runs/v16p51/.../run_meta.json",
            "runs/accepted_local_mutation_team_transfer_v2_prep/private_bundle.json",
            "runs/common_solver_contract_v1_prep_20260906/private_replay_registry.json",
            "runs/local_gepa_acceptance_rate_pilot_phase_b_v2_freeze/selected_parent_tasks_private.json",
        ],
        "compileall": "PASS",
        "new_failure_class": None,
        "real_provider_attempts": 0,
    })
    readme = """# Saturation-mode stopping refactor

This zero-provider refactor adds two coexisting execution regimes to all four
unified backend modes: fixed-budget and saturation. Saturation disables normal
scientific caps and stops only after repeated complete units with no accepted
deployable update. High emergency ceilings remain operational aborts and never
mean convergence.

## Complete units

| Mode | Complete local unit | Patience reset |
|---|---|---|
| GEPA_NATIVE | pinned native shuffled-data epoch | strict accepted GEPA mutation |
| GEPA_LAYER2 | full frozen packet schedule pass | strict accepted local mutation |
| MARS_NATIVE | T/C/S plus Target evaluation round | new best deployable prompt |
| MARS_LAYER2 | packet-owned T/C/S plus Target round | strict accepted local mutation |

Layer-2 outer patience is separate and resets only after atomic write-back. A
Vote-neutral Common-Safe commit counts; a local candidate rejected by team
gates does not. Packet exhaustion starts the next deterministic replay epoch
without native sampling fallback.

Recommended pre-freeze options are local patience 2/3/5 and team patience
1/2/3; the initial middle recommendation is local=3 and team=2. These values
must be preregistered before any real saturation run.

Saturation results estimate approximate within-method performance ceilings.
They do not answer equal-budget efficiency and must not be used for unqualified
GEPA-vs-MARS superiority claims.

API calls = 0; Validation50 calls = 0; Test50 calls = 0.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8", newline="\n")

    forbidden = ("api_key", "compatible-mode", "question_text", "raw_response")
    files = sorted(path for path in OUT.iterdir() if path.is_file())
    scan = {}
    for path in files:
        text = path.read_text(encoding="utf-8").lower()
        scan[path.name] = (
            not any(token in text for token in forbidden)
            and re.search(r"sk-[a-z0-9_.-]{12,}", text) is None
        )
    write_json("sanitization_manifest.json", {
        "status": "PASS" if all(scan.values()) else "FAIL",
        "checks": scan,
        "excluded": ["prompts", "questions", "answers", "raw responses", "credentials",
                     "endpoints", "caches", "checkpoints", "absolute paths"],
    })
    manifest = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "sha256_manifest.json":
            manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json("sha256_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
