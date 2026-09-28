"""Read-only, independently reconciled audit of the one Seed81 V4 attempt3 run.

The audit reads SHA-pinned private artifacts but emits counts, identities, and
hashes only. It never calls a provider or alters the frozen run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.governance.production_execution import _expected_bundle  # noqa: E402
from multi_dataset_diverse_rl.governance.startup_identity import (  # noqa: E402
    canonical_sha256, read_bundle, validate_startup_bundle,
)
from multi_dataset_diverse_rl.governance.v4_attempt3_contract import (  # noqa: E402
    SAMPLE_UNIT, SAMPLE_TARGET, MAX_OPPORTUNITIES, PROPOSAL_CEILING,
    MAX_PROPOSALS_PER_OPPORTUNITY, SUCCESSFUL_PROVIDER_CEILING,
    TRANSPORT_ATTEMPT_CEILING,
)
from multi_dataset_diverse_rl.local_optimizers.gepa_runtime import verify_frozen_gepa  # noqa: E402
from multi_dataset_diverse_rl.team_search.execution_runtime import ledger_summary  # noqa: E402
from scripts.audit_online_transfer_diagnostic import audit as ordinary_audit  # noqa: E402


ATTEMPT = "gepa_layer2_local_to_team_transfer_diagnostic_v4_seed81_attempt3"
SOURCE = "66e1762ad7ea8d85f71de3b604d0e04fb8b9c7d0"
PROTOCOL_SHA = "a9a0d72715029ac78e3ab46533f57faa2c0adf2f699c862becc91027de35c20f"
ALLOWED_PHASES = {
    "initialization", "local_optimizer_solver_eval", "local_optimizer_reflection",
    "team_minibatch_eval", "team_full_eval", "diagnostic_full_eval",
    "team_shadow_eval",
}


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("artifact must be an object")
    return value


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def audit(attempt_root: Path) -> dict:
    attempt_root = attempt_root.resolve()
    run = attempt_root / "run"
    prep = attempt_root / "prep_a"
    freeze_path = attempt_root / "raw_evidence_freeze.json"
    freeze = _read(freeze_path)
    if freeze["attempt_id"] != ATTEMPT or freeze["execution_source_sha"] != SOURCE:
        raise AssertionError("raw freeze identity mismatch")
    listed = {row["path"] for row in freeze["files"]}
    actual = {p.relative_to(attempt_root).as_posix() for p in attempt_root.rglob("*")
              if p.is_file() and p != freeze_path}
    if listed != actual or len(listed) != freeze["artifact_count"]:
        raise AssertionError("raw artifact inventory changed")
    for row in freeze["files"]:
        path = attempt_root / row["path"]
        if path.is_symlink() or path.stat().st_size != row["size_bytes"] or _sha(path) != row["sha256"]:
            raise AssertionError("raw artifact hash mismatch")

    manifest, protocol = _read(prep / "manifest.json"), _read(prep / "protocol.json")
    bundle = read_bundle(prep / "startup_identity")
    expected = _expected_bundle(root=ROOT, manifest=manifest, protocol=protocol,
                                execution_source_sha=SOURCE, prep=prep)
    validate_startup_bundle(stored=bundle, expected=expected,
                            require_authorized=True, phase="diagnostic",
                            roles=("solver", "reflection"))
    scientific = bundle["scientific_identity"]
    consumed = _read(prep / "authorization_consumed.json")
    if (manifest["attempt_id"] != ATTEMPT
            or manifest["execution"]["execution_source_sha"] != SOURCE
            or scientific["payload"]["protocol_sha256"] != PROTOCOL_SHA
            or canonical_sha256(protocol) != PROTOCOL_SHA
            or consumed["attempt_id"] != ATTEMPT
            or consumed["run_identity_sha256"] != bundle["run_identity"]["run_identity_sha256"]
            or consumed["status"] != "CONSUMED"
            or bundle["authorization"]["authorization_scope"] != ATTEMPT
            or manifest["runtime"]["seed"] != 81
            or manifest["runtime"]["provider_profile"] != "lwj"
            or manifest["models"] != {
                "solver": {"model": "qwen3-8b", "thinking": False},
                "reflection": {"model": "qwen3.7-flash"},
            }
            or manifest["dependency"] != {"gepa": verify_frozen_gepa()}
            or manifest["access"] != {"validation50_calls": 0, "test50_calls": 0}
            or protocol["sample_unit"] != SAMPLE_UNIT
            or protocol["returned_candidate_target"] != SAMPLE_TARGET
            or protocol["max_opportunities"] != MAX_OPPORTUNITIES
            or protocol["reflection_proposal_ceiling"] != PROPOSAL_CEILING
            or protocol["local_metric_call_budget_per_opportunity"] != 36
            or protocol["successful_provider_ceiling"] != SUCCESSFUL_PROVIDER_CEILING
            or protocol["transport_attempt_ceiling"] != TRANSPORT_ATTEMPT_CEILING):
        raise AssertionError("frozen protocol or authorization mismatch")
    subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{SOURCE}^{{commit}}"],
                   check=True, capture_output=True)
    source_paths = [row["path"] for row in scientific["payload"]["source_files"]]
    changed = subprocess.check_output(
        ["git", "-C", str(ROOT), "diff", "--name-only", SOURCE, "--", *source_paths],
        text=True, encoding="utf-8",
    ).strip()
    if changed:
        raise AssertionError("execution source changed since frozen commit")

    summary, lifecycle = _read(run / "execution_summary.json"), _read(run / "run_lifecycle.json")
    rows = _jsonl(run / "ledger.jsonl")
    usage = ledger_summary(run / "ledger.jsonl")
    baseline = ordinary_audit(run)
    if baseline["gate"] != "PASS":
        raise AssertionError("ordinary diagnostic evidence audit failed")
    if (lifecycle["status"] != "EXECUTION_COMPLETE"
            or summary["attempt_id"] != ATTEMPT
            or summary["run_identity_sha256"] != bundle["run_identity"]["run_identity_sha256"]
            or summary["ledger"] != usage
            or lifecycle["provider_attempts"] != usage["provider_attempts"]
            or lifecycle["provider_successes"] != usage["successful_provider_calls"]
            or lifecycle["provider_failures"] != usage["failed_provider_attempts"]
            or usage["successful_provider_calls"] > SUCCESSFUL_PROVIDER_CEILING
            or usage["provider_attempts"] > TRANSPORT_ATTEMPT_CEILING
            or usage["failed_provider_attempts"] != 0
            or {row["phase"] for row in rows} - ALLOWED_PHASES
            or any(row.get("postprocess_failed") for row in rows)
            or summary["validation50_calls"] != 0 or summary["test50_calls"] != 0):
        raise AssertionError("lifecycle, ledger, budget, or split-access mismatch")

    boundaries = summary["local_boundary_trace"]
    candidate_rows = summary["candidate_diagnostics"]
    if (len(boundaries) != summary["opportunities"]
            or len(candidate_rows) != summary["returned_candidate_count"]
            or summary["internal_accepted_mutations"] != summary["accepted_mutations"]
            or summary["returned_candidate_count"] != 1
            or summary["reflection_proposals"] != 10
            or summary["opportunities"] != 3
            or summary["target_status"] != "TARGET_NOT_REACHED"
            or summary["stop_reason"] != "REFLECTION_PROPOSAL_PREOPPORTUNITY_GUARD"
            or summary["reflection_proposals"] + MAX_PROPOSALS_PER_OPPORTUNITY <= PROPOSAL_CEILING):
        raise AssertionError("sample or prospective stopping mismatch")
    proposal_counts = []
    accepted_counts = []
    for index, boundary in enumerate(boundaries):
        target = summary["evidence_view_trace"][index]["target_member"]
        task = f"seed81_update{index}_member{target}_layer2_evidence"
        lineage = _jsonl(run / "local_gepa" / f"{task}.lineage.jsonl")
        proposals = [row for row in lineage if row["event_type"] == "proposal_end"]
        accepted = [row for row in lineage if row["event_type"] == "candidate_accepted"]
        proposal_counts.append(len(proposals))
        accepted_counts.append(len(accepted))
        if ([row["candidate_index"] for row in accepted] != boundary["accepted_event_indices"]
                or [row["iteration"] for row in accepted] != boundary["accepted_event_iterations"]
                or len(accepted) != boundary["internal_accepted_mutations"]
                or len(boundary["returned_candidate_ids"]) != boundary["returned_candidate_count"]):
            raise AssertionError("GEPA lineage and boundary mismatch")
        local_root = run / "local_gepa" / task
        with (local_root / "gepa_state.bin").open("rb") as handle:
            state = pickle.load(handle)  # SHA-verified local artifact from this attempt.
        materialized = json.loads((local_root / "candidates.json").read_text(encoding="utf-8"))
        if state["program_candidates"] != materialized:
            raise AssertionError("GEPA state and materialized candidates differ")
        frontier = sorted({int(value)
                           for values in state["program_at_pareto_front_valset"].values()
                           for value in values})
        if frontier != boundary["frontier_candidate_indices"]:
            raise AssertionError("GEPA frontier mismatch")
        for candidate_index, candidate_id in zip(
            boundary["returned_candidate_indices"], boundary["returned_candidate_ids"], strict=True,
        ):
            prompt = materialized[candidate_index]["decision_procedure"]
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            row = next(row for row in candidate_rows if row["candidate_id"] == candidate_id)
            if (candidate_id != f"gepa:{candidate_index}:{digest[:12]}"
                    or row["candidate_hash"] != digest
                    or row["update_index"] != index
                    or float(row["local_acceptance_delta"]) <= 0
                    or row["team_minibatch"]["passed"] != (not row["full"]["diagnostic_only"])):
                raise AssertionError("returned candidate/mandatory Full mismatch")
    if (sum(proposal_counts) != summary["reflection_proposals"]
            or sum(accepted_counts) != summary["internal_accepted_mutations"]
            or any(sum(proposal_counts[:index]) + MAX_PROPOSALS_PER_OPPORTUNITY > PROPOSAL_CEILING
                   for index in range(len(proposal_counts)))
            or sum(proposal_counts) + MAX_PROPOSALS_PER_OPPORTUNITY <= PROPOSAL_CEILING):
        raise AssertionError("proposal pre-guard arithmetic mismatch")
    if (summary["commits"] != 0
            or any(trace["parent_team_hash"] != trace["successor_team_hash"]
                   for trace in summary["evidence_view_trace"])
            or any(row["committed"] for row in candidate_rows)
            or any(row["ordinary_shadow"] != "NOT_REACHED" for row in candidate_rows)):
        raise AssertionError("unexpected admission, Shadow, or commit")

    candidate = candidate_rows[0]
    efficacy = "NOT_EVALUABLE" if summary["returned_candidate_count"] < SAMPLE_TARGET else "EVALUABLE"
    return {
        "audit_gate": "PASS",
        "scientific_validity": "VALID",
        "efficacy": efficacy,
        "efficacy_reason": "prospective_five_candidate_target_not_reached" if efficacy == "NOT_EVALUABLE" else None,
        "attempt_id": ATTEMPT,
        "execution_source_sha": SOURCE,
        "protocol_sha256": PROTOCOL_SHA,
        "raw_evidence_freeze_sha256": _sha(freeze_path),
        "raw_artifact_count": freeze["artifact_count"],
        "preregistration_sha256": scientific["preregistration_sha256"],
        "run_identity_sha256": bundle["run_identity"]["run_identity_sha256"],
        "lifecycle": lifecycle["status"],
        "opportunities": summary["opportunities"],
        "proposal_counts_by_opportunity": proposal_counts,
        "reflection_proposals": summary["reflection_proposals"],
        "internal_accepted_mutations": summary["internal_accepted_mutations"],
        "returned_candidate_count": summary["returned_candidate_count"],
        "target_status": summary["target_status"],
        "stop_reason": summary["stop_reason"],
        "candidate_observation": {
            "candidate_id": candidate["candidate_id"],
            "local_acceptance_delta": candidate["local_acceptance_delta"],
            "team_minibatch_promoted": candidate["team_minibatch"]["passed"],
            "full_diagnostic_only": candidate["full"]["diagnostic_only"],
            "full_vote_delta": candidate["full"]["vote_delta"],
            "ordinary_common_safe": candidate["ordinary_common_safe"],
            "ordinary_shadow": candidate["ordinary_shadow"],
            "committed": candidate["committed"],
        },
        "commits": summary["commits"],
        "provider_attempts": usage["provider_attempts"],
        "successful_provider_calls": usage["successful_provider_calls"],
        "failed_provider_attempts": usage["failed_provider_attempts"],
        "cache_hits": usage["cache_hits"],
        "validation50_calls": 0,
        "test50_calls": 0,
        "historical_attempt2_calls_pooled": False,
        "rerun_authorized": False,
        "formal_v3_authorized": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.attempt_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True, indent=2))
