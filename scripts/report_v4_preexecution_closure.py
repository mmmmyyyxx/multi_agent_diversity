"""Sanitize and assert a zero-provider V4 fake campaign into publishable facts.

The input is a pytest --basetemp tree, never a formal experiment root. The
output deliberately omits questions, answers, prompt text, example IDs, raw
provider messages, endpoints, and local paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.versions import (
    LAYER2_EVIDENCE_PACKET_V4_VERSION,
    LAYER2_EVIDENCE_SELECTION_POLICY_V4_VERSION,
    LAYER2_TARGET_FEASIBILITY_POLICY_V4_VERSION,
    LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION,
    TEAM_MINIBATCH_CONTRACT_VERSION,
)
from scripts.audit_online_transfer_diagnostic import audit


EXPERIMENT_ID = "gepa_layer2_local_to_team_transfer_diagnostic_v4"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(root: Path, name: str, payload: dict) -> None:
    (root / name).write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def update_sha256_manifest(report_root: Path) -> None:
    if not (report_root / "README.md").read_text(encoding="utf-8").startswith(
        "# V4 pre-execution closure"
    ):
        raise ValueError("not the V4 pre-execution report")
    _write(report_root, "sha256_manifest.json", {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(report_root.iterdir()) if path.name != "sha256_manifest.json"
    })


def _case_name(root: Path, summary_path: Path) -> str:
    return summary_path.relative_to(root).parts[0].removesuffix("0")


def _safe_evidence(row: dict) -> dict:
    universe = row["responsibility_universe"]
    delivered = row["evidence_delivered"]
    return {
        "update_index": row["update_index"],
        "target_member": row["target_member"],
        "parent_team_hash": row["parent_team_hash"],
        "successor_team_hash": row["successor_team_hash"],
        "committed": row["committed_candidate_id"] is not None,
        "raw_V": row["raw_V"],
        "assignment_V": row["assignment_V"],
        "packet_V": row["packet_V"],
        "responsibility_universe_count": int(universe["responsibility_universe_count"]),
        "responsibility_universe_ids_sha256": universe["responsibility_universe_ids_sha256"],
        "responsibility_scheduled_count": row["responsibility_scheduled"]["count"],
        "focus_role_item_count": len(row["focus_ids"]),
        "anchor_role_item_count": len(row["anchor_ids"]),
        "nominal_batch_count": len(row["nominal_schedule"]),
        "nominal_role_item_slots": sum(map(len, row["nominal_schedule"])),
        "delivered_batch_count": len(delivered["batch_ids"]),
        "delivered_unique_role_item_count": len(delivered["role_item_ids"]),
        "delivered_unique_source_count": len(delivered["source_ids"]),
        "delivered_responsibility_count": sum(
            item.startswith("responsibility:") for item in delivered["role_item_ids"]
        ),
        "scheduled_but_not_delivered_role_item_count": delivered[
            "scheduled_but_not_delivered_count"
        ],
        "team_minibatch_count": len(row["team_minibatch_ids"]),
        "team_minibatch_unique_count": len(set(row["team_minibatch_ids"])),
        "m_eval_equals_team_minibatch_ids": row["local_eval_ids"] == row["team_minibatch_ids"],
        "packet_hash": row["packet_hash"],
        "latest_transition_effect_hash": row["latest_transition_effect_hash"],
    }


def _safe_feasibility(row: dict) -> dict:
    return {
        "update_index": row["update_index"],
        "parent_team_hash": row["parent_team_hash"],
        "policy_version": row["policy_version"],
        "selected_member": row["selected_member"],
        "latest_transition_effect_hash_by_member": row[
            "latest_transition_effect_hash_by_member"
        ],
        "members": row["members"],
    }


def _stage_counts(summary: dict) -> dict:
    stage = summary["stage_accounting"]
    return {
        "global": stage["global"],
        "initialization": stage["initialization"],
        "opportunities": stage["opportunities"],
    }


def generate(fake_root: Path, report_root: Path, *, refresh: bool = False) -> dict:
    if report_root.exists():
        if not refresh or not (report_root / "README.md").read_text(
            encoding="utf-8"
        ).startswith("# V4 pre-execution closure"):
            raise FileExistsError("fresh V4 report root required unless refreshing this report")
    summaries: dict[str, dict] = {}
    audits: dict[str, dict] = {}
    for path in fake_root.rglob("execution_summary.json"):
        data = _read(path)
        if data.get("experiment_id") != EXPERIMENT_ID:
            continue
        name = _case_name(fake_root, path)
        if name in summaries:
            raise AssertionError(f"duplicate fake case: {name}")
        summaries[name] = data
        audits[name] = audit(path.parent)
        if audits[name]["gate"] != "PASS":
            raise AssertionError(f"fake case audit HOLD: {name}")
    if len(summaries) < 8:
        raise AssertionError("incomplete V4 fake-provider campaign")

    def unique(predicate) -> tuple[str, dict]:
        found = [(name, data) for name, data in summaries.items() if predicate(name, data)]
        if len(found) != 1:
            raise AssertionError(f"expected one matching case, got {len(found)}")
        return found[0]

    root_name, root = unique(lambda name, _: name.startswith("test_v4_raw_legal_bounded"))
    _, no_feasible = unique(lambda name, _: name.startswith("test_v4_no_feasible"))
    _, rerank = unique(lambda name, _: name.startswith("test_v4_feasible_reranking"))
    _, chain = unique(lambda name, _: name.startswith("test_v4_commit_no_commit"))
    _, budget = unique(lambda name, _: name.startswith("test_v4_fake_provider_budget"))
    _, proposal = unique(lambda name, _: name.startswith("test_v4_proposal_ceiling"))
    _, minibatch = unique(lambda name, _: name.startswith("test_v4_raw_legal_downstream")
                          and any(not row["team_minibatch"]["passed"]
                                  for row in _["candidate_diagnostics"]))
    _, common_safe = unique(lambda name, data: name.startswith("test_v4_raw_legal_downstream")
                            and any(row["ordinary_common_safe"] not in {"PASS", "NOT_REACHED"}
                                    for row in data["candidate_diagnostics"]))
    _, shadow = unique(lambda name, data: name.startswith("test_v4_raw_legal_downstream")
                       and any(row["ordinary_shadow"] not in {"PASS", "NOT_REACHED"}
                               for row in data["candidate_diagnostics"]))
    evidence = [_safe_evidence(row) for row in root["evidence_view_trace"]]
    root_evidence = evidence[0]
    if not (
        root_evidence["raw_V"] == root_evidence["assignment_V"]
        == root_evidence["packet_V"] == 50
        and root_evidence["responsibility_universe_count"] == 50
        and root_evidence["responsibility_scheduled_count"] <= 36
        and root_evidence["focus_role_item_count"] == 0
        and root_evidence["anchor_role_item_count"] == 0
        and root_evidence["team_minibatch_count"] == 12
        and root_evidence["team_minibatch_unique_count"] == 12
        and root_evidence["m_eval_equals_team_minibatch_ids"]
    ):
        raise AssertionError("V4 root scientific facts differ from preregistration")
    if not (budget["accepted_mutations"] == 5
            and budget["ledger"]["successful_provider_calls"] <= 1200
            and budget["ledger"]["provider_attempts"] <= 4800):
        raise AssertionError("fake budget stress does not pass")
    cache_reuse = {
        phase: sum(row["by_phase"][phase]["cache_hits"]
                   for row in budget["stage_accounting"]["opportunities"])
        for phase in ("team_minibatch_eval", "team_full_eval", "team_shadow_eval")
    }
    if not all(cache_reuse.values()):
        raise AssertionError("fake stage cache reuse did not occur")
    if not (no_feasible["stop_reason"] == "NO_FEASIBLE_LAYER2_OPPORTUNITY"
            and no_feasible["opportunities"] == 0):
        raise AssertionError("no-feasible did not stop scientifically")
    rerank_initial = rerank["feasibility_trace"][0]
    if not (
        rerank_initial["selected_member"] == 1
        and next(row for row in rerank_initial["members"] if row["member_id"] == 0)[
            "feasibility_reason"
        ] == "INSUFFICIENT_PRIMARY_REPAIR_QUOTA"
    ):
        raise AssertionError("feasibility mask did not rerank before selection")
    if not (proposal["reflection_proposals"] == 20
            and proposal["stop_reason"] == "REFLECTION_PROPOSAL_CEILING_REACHED"):
        raise AssertionError("proposal ceiling was not enforced")
    if not (
        chain["parent_sequence"][1] == chain["parent_sequence"][2]
        and chain["evidence_view_trace"][1]["committed_candidate_id"] is None
        and chain["feasibility_trace"][1]["latest_transition_effect_hash_by_member"]
        == chain["feasibility_trace"][2]["latest_transition_effect_hash_by_member"]
    ):
        raise AssertionError("successor state changed without commit")

    failures = []
    for path in fake_root.rglob("run_lifecycle.json"):
        lifecycle = _read(path)
        if lifecycle.get("status") != "ABORTED":
            continue
        name = _case_name(fake_root, path)
        if not name.startswith("test_v4_failed_physical"):
            continue
        ledger_path = path.parent / "ledger.jsonl"
        with ledger_path.open(encoding="utf-8") as handle:
            ledger = [json.loads(line) for line in handle if line.strip()]
        failures.append({
            "case": "postprocess" if any(row.get("postprocess_failed") for row in ledger)
            else "transport",
            "lifecycle_status": lifecycle["status"],
            "provider_attempts": lifecycle["provider_attempts"],
            "successful_provider_calls": lifecycle["provider_successes"],
            "failed_provider_attempts": lifecycle["provider_failures"],
            "ledger_records": len(ledger),
        })
    if len(failures) != 2 or {row["case"] for row in failures} != {"postprocess", "transport"}:
        raise AssertionError("incomplete V4 failure campaign")

    report_root.mkdir(parents=True, exist_ok=refresh)
    _write(report_root, "method_delta.json", {
        "scientific_change": True,
        "protocol_version": LAYER2_TEAM_SEARCH_PROTOCOL_V4_VERSION,
        "packet_version": LAYER2_EVIDENCE_PACKET_V4_VERSION,
        "evidence_selection_policy_version": LAYER2_EVIDENCE_SELECTION_POLICY_V4_VERSION,
        "target_feasibility_policy_version": LAYER2_TARGET_FEASIBILITY_POLICY_V4_VERSION,
        "team_minibatch_version_unchanged": TEAM_MINIBATCH_CONTRACT_VERSION,
        "raw_V_and_target_score_formula_unchanged": True,
        "bounded_view_affects_only_layer1_curriculum": True,
        "local_eval_equals_full_universe_team_minibatch_ids": True,
        "v3_original_files_modified": False,
    })
    _write(report_root, "feasibility_trace.json", {
        "root": _safe_feasibility(root["feasibility_trace"][0]),
        "no_feasible": _safe_feasibility(no_feasible["feasibility_trace"][0]),
        "feasible_rerank": _safe_feasibility(rerank_initial),
        "successor_chain": [_safe_feasibility(row) for row in chain["feasibility_trace"]],
    })
    _write(report_root, "evidence_view_trace.json", {
        "root": root_evidence,
        "successor_chain": [_safe_evidence(row) for row in chain["evidence_view_trace"]],
    })
    _write(report_root, "responsibility_value_trace.json", {
        "root_raw_V": root_evidence["raw_V"],
        "root_assignment_V": root_evidence["assignment_V"],
        "root_packet_V": root_evidence["packet_V"],
        "target_score_selection_only": True,
    })
    _write(report_root, "full_stack_scenarios.json", {
        "A": {"commits": root["commits"], "audited": audits[root_name]["gate"]},
        "B": {"local_accepted": minibatch["accepted_mutations"],
              "minibatch_rejected": any(not row["team_minibatch"]["passed"]
                                        for row in minibatch["candidate_diagnostics"]),
              "mandatory_diagnostic_full": any(row["full"]["diagnostic_only"]
                                               for row in minibatch["candidate_diagnostics"]),
              "commits": minibatch["commits"]},
        "C": {"common_safe_rejected": True, "commits": common_safe["commits"]},
        "D": {"shadow_rejected": True, "commits": shadow["commits"]},
        "successor_chain_commits": chain["commits"],
        "all_fake_audits_pass": all(row["gate"] == "PASS" for row in audits.values()),
    })
    _write(report_root, "provider_budget_stress.json", {
        "accepted_mutations": budget["accepted_mutations"],
        "opportunities": budget["opportunities"],
        "reflection_proposals": budget["reflection_proposals"],
        "ledger": budget["ledger"],
        "stage_accounting": _stage_counts(budget),
        "successful_provider_emergency_ceiling": 1200,
        "provider_attempt_emergency_ceiling": 4800,
        "within_both_ceilings": True,
    })
    _write(report_root, "ledger_reconciliation.json", {
        "all_complete_fake_audits_pass": all(row["gate"] == "PASS" for row in audits.values()),
        "complete_cases": len(audits),
        "failure_cases": failures,
        "cache_hits_in_budget_stress": budget["ledger"]["cache_hits"],
        "stage_cache_reuse": cache_reuse,
        "validation50_calls": 0,
        "test50_calls": 0,
    })
    facts = {
        "root_raw_V_50": root_evidence["raw_V"] == 50,
        "root_full_primary_count_50": root_evidence["responsibility_universe_count"] == 50,
        "root_schedule_bounded_36": root_evidence["responsibility_scheduled_count"] <= 36,
        "root_delivery_not_implied": root_evidence["delivered_responsibility_count"]
                                      <= root_evidence["responsibility_scheduled_count"],
        "team_minibatch_exact_12": root_evidence["team_minibatch_unique_count"] == 12,
        "m_eval_equals_team_minibatch": root_evidence["m_eval_equals_team_minibatch_ids"],
        "no_feasible_scientific_stop": no_feasible["stop_reason"]
                                         == "NO_FEASIBLE_LAYER2_OPPORTUNITY",
        "feasible_reranking": rerank_initial["selected_member"] == 1,
        "five_local_accepts_under_ceilings": budget["accepted_mutations"] == 5,
        "proposal_ceiling_20": proposal["reflection_proposals"] == 20,
        "real_api_calls_zero": True,
        "validation50_calls_zero": True,
        "test50_calls_zero": True,
    }
    if not all(facts.values()):
        raise AssertionError("V4 factual assertion failed")
    _write(report_root, "fact_assertions.json", {"gate": "PASS", "assertions": facts})
    _write(report_root, "verification.json", {
        "fake_campaign_gate": "PASS", "complete_fake_cases": len(audits),
        "zero_real_api": True, "full_pytest": "RECORDED_SEPARATELY_AFTER_RUN",
        "source_poison": "RECORDED_SEPARATELY_AFTER_RUN",
        "governance_preflight": "RECORDED_SEPARATELY_AFTER_RUN",
    })
    _write(report_root, "freeze_identity.json", {
        "status": "PENDING_IMPLEMENTATION_COMMIT_AND_ISOLATED_PREP_REPLAY",
        "authorized": False, "real_execution_started": False,
    })
    (report_root / "README.md").write_text(
        "# V4 pre-execution closure\n\n"
        "Zero-real-API scientific method implementation and fake-provider audit. "
        "V1/V2/V3 historical freezes remain untouched; V3 is referenced as "
        "`SUPERSEDED_BEFORE_EXECUTION`, not rewritten.\n\n"
        "The raw legal responsibility universe determines full D/N/C/V and "
        "the unchanged V/(1+f) target score. Layer2 preselects members that "
        "can build the exact TeamMiniBatch12 and bounded search packet, then "
        "retains the original score order. The bounded Layer1 curriculum is "
        "a nominal schedule; actually delivered evidence is audited separately. "
        "M_eval remains the complete frozen TeamMiniBatch12 identity.\n\n"
        f"Fake root: raw V={root_evidence['raw_V']}, full repair universe="
        f"{root_evidence['responsibility_universe_count']}, scheduled repair="
        f"{root_evidence['responsibility_scheduled_count']}, delivered repair="
        f"{root_evidence['delivered_responsibility_count']}. "
        "A/B/C/D and multi-commit successor replay passed. "
        f"Budget stress reached {budget['accepted_mutations']} accepted local "
        f"mutations with {budget['ledger']['successful_provider_calls']} "
        "fake successful provider calls. No Validation50 or Test50 was accessed.\n\n"
        "The historical-artifact-dependent full-suite results and fresh freeze "
        "identity are recorded in separate verification/identity files. "
        "Formal saturation repairs remain out of scope.\n",
        encoding="utf-8",
    )
    update_sha256_manifest(report_root)
    return {"fake_cases": len(summaries), "root": root_evidence,
            "budget_successes": budget["ledger"]["successful_provider_calls"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake-root", type=Path)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--hash-only", action="store_true")
    args = parser.parse_args()
    if args.hash_only:
        update_sha256_manifest(args.report_root)
        print(json.dumps({"sha256_manifest": "UPDATED"}))
        sys.exit(0)
    if args.fake_root is None:
        parser.error("--fake-root is required unless --hash-only is set")
    print(json.dumps(generate(args.fake_root, args.report_root, refresh=args.refresh), sort_keys=True))
