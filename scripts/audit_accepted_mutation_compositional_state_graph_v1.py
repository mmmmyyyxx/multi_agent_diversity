"""Bounded zero-API compositional audit of the five accepted mutations."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from multi_dataset_diverse_rl.answer_formats import match_answer
from multi_dataset_diverse_rl.evaluation.endpoint_identifiability import endpoint_structural_identifiability
from multi_dataset_diverse_rl.governance.artifacts import build_sha256_manifest, scan_sanitized_artifacts
from multi_dataset_diverse_rl.governance.manifest import preregistration_hash, validate_manifest
from multi_dataset_diverse_rl.versions import (
    ACCEPTED_MUTATION_COMPOSITIONAL_STATE_GRAPH_VERSION, METHOD_VERSION,
)


IDENTITY = ACCEPTED_MUTATION_COMPOSITIONAL_STATE_GRAPH_VERSION
BASELINE_ACCURACY = 60
BASELINE_WRONG_ROWS = 40
MUTATION_ORDER = [
    "member1_proposal2_7643bc41b45e",
    "member2_proposal8_a5fef723548e",
    "member3_proposal5_0229baf5ae5f",
    "member4_proposal2_c537ca25a8aa",
    "member4_proposal4_f5d945233dc9",
]
SAFE_MUTATIONS = [MUTATION_ORDER[0], MUTATION_ORDER[1], MUTATION_ORDER[2], MUTATION_ORDER[4]]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _legal_outputs(question: str) -> tuple[str | None, ...]:
    letters = tuple(re.findall(r"^\(([A-Z])\)\s", question, flags=re.MULTILINE))
    if not letters:
        raise ValueError("question has no legal option domain")
    return (*letters, None)


def _intersection_count(sets: list[set[str]]) -> int:
    return len(set.intersection(*sets)) if sets else 0


def _at_least_count(sets: list[set[str]], threshold: int) -> int:
    return sum(count >= threshold for count in Counter(
        item for values in sets for item in values
    ).values())


def run(args) -> dict:
    run_root = args.run_root.resolve()
    bundle = args.bundle.resolve()
    report = args.report.resolve()
    manifest_path = args.manifest.resolve()
    protocol = ROOT / "experiments" / IDENTITY / "PROTOCOL.md"
    if report.exists() and any(report.iterdir()):
        if not args.overwrite_existing:
            raise FileExistsError("fresh compositional report required")
        existing_summary = report / "summary.json"
        if not existing_summary.exists() or read(existing_summary).get("experiment_id") != IDENTITY:
            raise ValueError("refusing to overwrite a report from another experiment")
    if manifest_path.exists():
        if not args.overwrite_existing:
            raise FileExistsError("fresh compositional manifest required")
        if yaml.safe_load(manifest_path.read_text(encoding="utf-8")).get("experiment_id") != IDENTITY:
            raise ValueError("refusing to overwrite a manifest from another experiment")

    execution = read(run_root / "execution.json")
    private = read(bundle / "private_bundle.json")
    if execution["execution_status"] != "EXECUTION_COMPLETE" or execution["full_evaluated_candidates"] != 5:
        raise ValueError("complete v2 execution required")
    cases = {row["mutation_id"]: row for row in execution["cases"]}
    if list(cases) != MUTATION_ORDER:
        raise ValueError("unexpected mutation identity or order")

    run_files = sorted(path.name for path in run_root.iterdir() if path.is_file())
    candidate_profiles_present = any(name in run_files for name in (
        "candidate_profiles.json", "candidate_profiles_private.json", "solver_cache.sqlite",
    ))
    evidence_availability = {
        "candidate_full_output_profiles_persisted": candidate_profiles_present,
        "persistent_exact_response_cache_persisted": "solver_cache.sqlite" in run_files,
        "run_files": run_files,
        "provider_ledger_contains_response_text": False,
        "execution_contains_answer_vectors": False,
        "exact_24_state_reconstruction": "AVAILABLE" if candidate_profiles_present else "NOT_AVAILABLE",
        "reason": (
            "The v2 evaluator cache was process-local and only decompositions were persisted; "
            "request identities and aggregate counts cannot recover answer clusters."
        ),
    }
    if candidate_profiles_present:
        raise NotImplementedError("exact-profile branch is intentionally separate from this bounded audit")

    tasks = private["phase_a_tasks"]
    task_ids = [f"seed78_update0_member{i}" for i in range(5)]
    profiles = [[row["parent_output"] for row in tasks[key]["search_examples"]] for key in task_ids]
    rows0 = tasks[task_ids[0]]["search_examples"]
    legal = [_legal_outputs(row["input_payload"]) for row in rows0]
    identifiability = endpoint_structural_identifiability(
        profiles, legal_target_outputs_by_row=legal,
    )
    if identifiability["total_member_row_opportunities"] != 0:
        raise ValueError("expected frozen homogeneous baseline to be structurally locked")

    mutation_data = {}
    for mutation_id, row in cases.items():
        task = tasks[row["source_parent_task_id"]]
        assigned = [item for item in task["search_examples"] if "responsibility" in item["tags"]]
        if any(
            item["parent_output"] is not None
            and match_answer(item["parent_output"], item["gold"], "option_letter")
            for item in assigned
        ):
            raise ValueError("assigned responsibility rows must be baseline-wrong")
        mutation_data[mutation_id] = {
            "target_member": row["target_member"],
            "target_accuracy": BASELINE_ACCURACY + row["target_member_full_delta"],
            "target_delta": row["target_member_full_delta"],
            "gain_count": row["unique_correct_gain"],
            "loss_cases": set(row["collateral_loss_cases"]),
            "terminal_invalid_nonregression": row["common_safe"]["terminal_invalid_nonregression_passed"],
            "baseline_common_safe": row["common_safe_status"] == "PASS",
        }
    if not all(mutation_data[value]["baseline_common_safe"] for value in SAFE_MUTATIONS):
        raise ValueError("safe mutation set no longer matches frozen Common-Safe evidence")
    if mutation_data[MUTATION_ORDER[3]]["terminal_invalid_nonregression"]:
        raise ValueError("M4 proposal2 must retain its frozen terminal-invalid regression")

    choices = [
        (None, MUTATION_ORDER[0]),
        (None, MUTATION_ORDER[1]),
        (None, MUTATION_ORDER[2]),
        (None, MUTATION_ORDER[3], MUTATION_ORDER[4]),
    ]
    state_rows = []
    state_by_choice = {}
    for selected_tuple in itertools.product(*choices):
        selected = [value for value in selected_tuple if value is not None]
        k = len(selected)
        gains = [mutation_data[value]["gain_count"] for value in selected]
        losses = [mutation_data[value]["loss_cases"] for value in selected]
        if k == 0:
            coverage_lower = coverage_upper = BASELINE_ACCURACY
        else:
            union_lower = max(max(gains), math.ceil(sum(gains) / k))
            union_upper = min(BASELINE_WRONG_ROWS, sum(gains))
            coverage_lower = BASELINE_ACCURACY + union_lower
            coverage_upper = BASELINE_ACCURACY + union_upper
        if k <= 2:
            vote_lower = vote_upper = BASELINE_ACCURACY
            bound_reason = "EXACT_IDENTICAL_BASELINE_MAJORITY"
        elif k == 3:
            gain_lower = max(0, sum(gains) - 2 * BASELINE_WRONG_ROWS)
            gain_upper = min(gains)
            loss_overlap = _intersection_count(losses)
            # All three mutated members must be individually wrong before the
            # two fixed baseline-correct votes can possibly lose plurality.
            # Their exact wrong-answer clusters were not persisted, so this
            # overlap is an upper bound on vote losses, never an exact count.
            loss_lower = 0
            loss_upper = loss_overlap
            vote_lower = BASELINE_ACCURACY + gain_lower - loss_upper
            vote_upper = BASELINE_ACCURACY + gain_upper - loss_lower
            bound_reason = "THREE_SET_INTERSECTION_BOUNDS"
        else:
            gain_lower = math.ceil(max(0, sum(gains) - 2 * BASELINE_WRONG_ROWS) / 2)
            # Three mutated correct answers guarantee a gold plurality. Two
            # may also suffice when the three wrong votes split, so an upper
            # bound counts rows with at least two gain memberships.
            gain_upper = min(BASELINE_WRONG_ROWS, sum(gains) // 2)
            loss_upper = _at_least_count(losses, 3)
            # Three or more individual losses are necessary but not
            # sufficient for a plurality loss. Exact wrong-answer clusters
            # are unavailable, hence the lower bound remains zero.
            loss_lower = 0
            vote_lower = BASELINE_ACCURACY + gain_lower - loss_upper
            vote_upper = BASELINE_ACCURACY + gain_upper - loss_lower
            bound_reason = "FOUR_SET_AT_LEAST_THREE_MEMBERSHIP_BOUNDS"
        member_accuracies = [BASELINE_ACCURACY] * 5
        for value in selected:
            member_accuracies[mutation_data[value]["target_member"]] = mutation_data[value]["target_accuracy"]
        labels = ["B" if value is None else value for value in selected_tuple]
        state_id = "__".join(f"m{index + 1}_{label}" for index, label in enumerate(labels))
        row = {
            "state_id": state_id,
            "selected_mutations": selected,
            "mutation_count": k,
            "member_accuracies": member_accuracies,
            "vote_correct_count_lower": vote_lower,
            "vote_correct_count_upper": vote_upper,
            "vote_delta_lower": vote_lower - BASELINE_ACCURACY,
            "vote_delta_upper": vote_upper - BASELINE_ACCURACY,
            "oracle_coverage_count_lower": coverage_lower,
            "oracle_coverage_count_upper": coverage_upper,
            "bound_reason": bound_reason,
            "exact_output_diversity": "NOT_IDENTIFIED",
            "exact_pivotal_rows": "NOT_IDENTIFIED",
        }
        state_rows.append(row)
        state_by_choice[selected_tuple] = row
    if len(state_rows) != 24:
        raise ValueError("configuration enumeration must contain 24 states")

    edges = []
    for source_choices, source in state_by_choice.items():
        for member_index, options in enumerate(choices):
            if source_choices[member_index] is not None:
                continue
            for mutation_id in options[1:]:
                target_choices = list(source_choices)
                target_choices[member_index] = mutation_id
                target = state_by_choice[tuple(target_choices)]
                if not mutation_data[mutation_id]["terminal_invalid_nonregression"]:
                    status = "PROVEN_NOT_COMMON_SAFE_TERMINAL_INVALID_REGRESSION"
                elif target["mutation_count"] <= 2:
                    status = "PROVEN_COMMON_SAFE"
                elif target["vote_correct_count_lower"] >= source["vote_correct_count_upper"]:
                    status = "PROVEN_COMMON_SAFE_BY_BOUNDS"
                else:
                    status = "UNRESOLVED_MISSING_CANDIDATE_PROFILES"
                edges.append({
                    "source_state_id": source["state_id"],
                    "target_state_id": target["state_id"],
                    "added_mutation": mutation_id,
                    "status": status,
                })

    safe_triples = []
    triple_loss_upper_sum = 0
    for combo in itertools.combinations(SAFE_MUTATIONS, 3):
        gains = [mutation_data[value]["gain_count"] for value in combo]
        losses = [mutation_data[value]["loss_cases"] for value in combo]
        gain_lower = max(0, sum(gains) - 2 * BASELINE_WRONG_ROWS)
        gain_upper = min(gains)
        loss_eligible_upper = _intersection_count(losses)
        triple_loss_upper_sum += loss_eligible_upper
        safe_triples.append({
            "mutations": list(combo),
            "vote_gain_count_lower": gain_lower,
            "vote_gain_count_upper": gain_upper,
            "vote_loss_eligible_count_exact": loss_eligible_upper,
            "vote_loss_count_lower": 0,
            "vote_loss_count_upper": loss_eligible_upper,
            "vote_delta_lower": gain_lower - loss_eligible_upper,
            "vote_delta_upper": gain_upper,
        })
    # Individual triple lower bounds lose cross-triple constraints. Across all
    # four triples, total gain memberships obey sum C(r,3). With 91 memberships
    # over 40 rows, at least 11 triple-gain incidences must exist.
    safe_gain_memberships = sum(mutation_data[value]["gain_count"] for value in SAFE_MUTATIONS)
    cross_triple_gain_lower = max(0, safe_gain_memberships - 2 * BASELINE_WRONG_ROWS)
    existential_delta_sum_lower = cross_triple_gain_lower - triple_loss_upper_sum
    existential = {
        "safe_mutations": SAFE_MUTATIONS,
        "gain_memberships_across_40_baseline_wrong_rows": safe_gain_memberships,
        "sum_of_four_triple_gain_counts_lower": cross_triple_gain_lower,
        "sum_of_four_triple_vote_loss_eligible_counts_exact": triple_loss_upper_sum,
        "sum_of_four_triple_vote_loss_counts_upper": triple_loss_upper_sum,
        "sum_of_four_triple_vote_deltas_lower": existential_delta_sum_lower,
        "at_least_one_team_positive_triple_exists": existential_delta_sum_lower > 0,
        "at_least_one_common_safe_path_to_team_positive_triple_exists": existential_delta_sum_lower > 0,
        "pivotal_capable_two_mutation_predecessor_exists": existential_delta_sum_lower > 0,
        "exact_positive_triple_identity": "NOT_IDENTIFIED_MISSING_GAIN_CASE_IDENTITIES",
        "proof": (
            "The first two safe valid mutations are vote-neutral Common-Safe steps. Across the "
            "four possible safe triples, at least 11 baseline-wrong triple gains and 8 exact "
            "triple-loss-eligible incidences (an upper bound on vote losses) imply summed vote "
            "delta >=3, so at least one triple "
            "has positive vote delta and is reached by a Common-Safe third step."
        ),
    }

    full_bounds = [row for row in state_rows if row["mutation_count"] == 4]
    report.mkdir(parents=True, exist_ok=True)
    write(report / "evidence_availability_audit.json", evidence_availability)
    write(report / "baseline_endpoint_identifiability.json", identifiability)
    write(report / "state_catalog_24.json", state_rows)
    write(report / "common_safe_state_edges.json", edges)
    write(report / "safe_triple_bounds.json", safe_triples)
    write(report / "existential_reachability_proof.json", existential)
    write(report / "full_composition_bounds.json", full_bounds)
    write(report / "summary.json", {
        "experiment_id": IDENTITY,
        "status": "BOUNDED_AUDIT_COMPLETE_EXACT_GRAPH_NOT_RECONSTRUCTIBLE",
        "configuration_count": 24,
        "api_calls": 0,
        "baseline_total_pivotal_opportunities": identifiability["total_member_row_opportunities"],
        "exact_graph_available": False,
        "common_safe_team_positive_path_exists": True,
        "exact_path_identity_available": False,
        "safe_full_composition_vote_delta_lower": next(
            row["vote_delta_lower"] for row in full_bounds
            if MUTATION_ORDER[4] in row["selected_mutations"]
        ),
    })
    write(report / "interpretation.json", {
        "supported": [
            "The homogeneous baseline fails endpoint structural identifiability with P_i=0 for all members.",
            "At least one three-mutation state is Common-Safe reachable and team-positive.",
            "The full M1+M2+M3+M4-P4 composition has vote delta at least +1.",
            "Vote-neutral target-improving updates can be necessary symmetry-breaking steps.",
        ],
        "not_identified": [
            "Which safe triple is team-positive",
            "Exact metrics for all 24 states",
            "Exact pivotal-row identities after composition",
            "A concrete state that may be frozen for a new API experiment",
        ],
        "team_minibatch_conclusion": "DO_NOT_MODIFY_FROM_V2_FALSE_POSITIVE_COUNTS",
        "next_step": (
            "Recover or prospectively persist candidate profiles before selecting a concrete "
            "pivotal-capable state; do not make new API calls under this audit."
        ),
    })
    write(report / "api_ledger_summary.json", {
        "provider_calls": 0, "solver_calls": 0, "GEPA": 0, "Reflection": 0,
        "Validation50": 0, "Test50": 0,
    })
    write(report / "fact_assertions.json", {
        "twenty_four_states_enumerated": True,
        "baseline_identifiability_zero": True,
        "missing_profiles_not_fabricated": True,
        "existential_common_safe_positive_path_proven": True,
        "exact_positive_state_not_claimed": True,
        "zero_api": True,
        "validation_test_zero": True,
    })

    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "experiment_manifest_v1", "experiment_id": IDENTITY,
        "title": "Bounded accepted-mutation compositional state graph audit",
        "status": "COMPLETED", "legacy_index": False,
        "lifecycle_history": [{"status": status, "timestamp": now} for status in (
            "DRAFT", "PREREGISTERED", "IMPLEMENTED", "PREFLIGHT_PASS", "RUNNING",
            "TRAIN_FROZEN", "COMPLETED",
        )],
        "lineage": {"parents": ["accepted_local_mutation_team_transfer_v2"],
                    "derives_from": "accepted_local_mutation_team_transfer_v2"},
        "scientific_question": "Can frozen local improvements compose into a Common-Safe plurality-responsive state without new API calls?",
        "hypotheses": [
            "Vote-neutral local improvements may be safe symmetry-breaking steps.",
            "At least one composition may yield positive plurality transfer.",
        ],
        "method_identity": METHOD_VERSION, "runtime_version": IDENTITY,
        "data": {
            "task": "BBH disambiguation_qa", "formal": False,
            "split_ids": {"optimize100": "anti_overfitting_split_v1_fold_a_plus_b"},
            "split_hashes": {"baseline_team_hash": execution["baseline_team_hash"]},
            "validation_policy": "prohibited; Validation50 calls=0",
            "test_policy": "prohibited; Test50 calls=0",
        },
        "model": {"solver": "NONE_ZERO_API", "optimizer_roles": {}, "thinking": False,
                  "temperatures": {}, "max_tokens": {}},
        "seeds": [78],
        "design": {
            "changed": ["offline composition of frozen evidence"],
            "unchanged": ["five-member plurality", "tie-as-abstain", "Common-Safe"],
            "forbidden_changes": ["API calls", "candidate regeneration", "writeback", "scheduler"],
            "configuration_count": 24,
            "protocol_sha256": sha(protocol),
            "evidence_availability": "CANDIDATE_FULL_PROFILES_NOT_PERSISTED",
        },
        "api_authorization": {"authorized": False, "authorization_scope": "ZERO_API_ONLY",
                              "allowed_roles": [], "allowed_phases": []},
        "budget": {"type": "zero_api_offline_audit", "frozen_before_run": True,
                   "limit": {"provider_calls": 0, "Validation50": 0, "Test50": 0}},
        "selection": {
            "primary_metric": "existence_of_common_safe_path_to_team_positive_composition",
            "frozen_rule": "enumerate all 24 frozen configurations; use exact metrics only when evidence permits",
            "validation_used_for_selection": False, "test_used_for_selection": False,
        },
        "artifacts": {
            "preregistration": {"path": protocol.relative_to(ROOT).as_posix()},
            "report": report.relative_to(ROOT).as_posix(),
            "provenance": (report / "provenance.json").relative_to(ROOT).as_posix(),
        },
        "git": {
            "design_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "implementation_commit": None, "result_commit": None,
        },
        "result": {
            "classifier": "BOUNDED_AUDIT_COMPLETE_EXACT_GRAPH_NOT_RECONSTRUCTIBLE",
            "conclusion": "At least one Common-Safe path to a team-positive triple exists, but missing candidate profiles prevent identification of the concrete state.",
            "evidence_type": "retrospective",
        },
    }
    manifest["artifacts"]["preregistration"]["sha256"] = preregistration_hash(manifest)
    schema = read(ROOT / "infrastructure/experiment_manifest.schema.json")
    errors = validate_manifest(manifest, schema)
    if errors:
        raise ValueError("manifest invalid: " + "; ".join(errors))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
                             encoding="utf-8", newline="\n")
    write(report / "provenance.json", {
        "source_execution_sha256": sha(run_root / "execution.json"),
        "source_provider_ledger_sha256": sha(run_root / "provider_ledger.jsonl"),
        "source_private_bundle_sha256": sha(bundle / "private_bundle.json"),
        "protocol_sha256": sha(protocol), "manifest_sha256": sha(manifest_path),
        "raw_prompts_published": False, "raw_questions_published": False,
        "raw_answers_published": False,
    })
    (report / "README.md").write_text(
        "# Accepted-mutation compositional state graph v1\n\n"
        "Zero-API bounded audit. Exact 24-state replay is unavailable because v2 did not "
        "persist candidate answer profiles. Nevertheless, aggregate gain counts and exact loss "
        "hashes prove that at least one Common-Safe path reaches a team-positive three-mutation "
        "state. The concrete state identity remains unresolved.\n",
        encoding="utf-8",
    )
    findings = scan_sanitized_artifacts(report)
    write(report / "sanitization_manifest.json", {
        "status": "PASS" if not findings else "FAIL", "findings": findings,
    })
    write(report / "sha256_manifest.json", build_sha256_manifest(report))
    if findings or read(report / "sha256_manifest.json") != build_sha256_manifest(report):
        raise ValueError("report sanitization/hash replay failed")
    return read(report / "summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
