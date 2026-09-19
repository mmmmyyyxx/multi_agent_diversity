"""Reproducible zero-API inventory and preparation package, never a run freeze."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from multi_dataset_diverse_rl.governance.artifacts import build_sha256_manifest, scan_sanitized_artifacts
from multi_dataset_diverse_rl.governance.manifest import preregistration_hash
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import GEPAOptimizerConfig
from multi_dataset_diverse_rl.local_optimizers.gepa_runtime import verify_frozen_gepa
from multi_dataset_diverse_rl.local_optimizers.proposal_telemetry import EVALUATION_FIELDS, cost_envelope, text_hash, evaluation_schema
from multi_dataset_diverse_rl.versions import METHOD_VERSION, LOCAL_GEPA_PROPOSAL_TELEMETRY_VERSION

IDENTITY = "local_gepa_acceptance_rate_pilot_v1"
REPORT = ROOT / "reports/local_gepa_acceptance_rate_pilot_v1_prep_20260918"
EXPERIMENT = ROOT / "experiments" / IDENTITY
MANIFEST = ROOT / "experiments/manifests" / (IDENTITY + ".yaml")
CANARY = ROOT / "reports/level_b_gepa_real_canary_v2_stagefix2_transportfix1_authorized1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    return text_hash(json.dumps(value, sort_keys=True, separators=(",", ":")))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parent_catalog():
    """Inventory all accessible current canary/Seed78 native roots, plus old registry.

    We never select candidates by mutation outcomes, scores, or later success.
    Incomplete records remain visible with null identities, never fabricated IDs.
    """
    # Execution attempts may live in registered detached worktrees. Discover
    # them read-only; public source locators use commit-qualified logical names.
    worktrees = subprocess.check_output(["git", "worktree", "list", "--porcelain"], cwd=ROOT, text=True)
    checkouts = [(ROOT, "workspace")]
    for block in worktrees.strip().split("\n\n"):
        fields = dict(line.split(" ", 1) for line in block.splitlines() if " " in line)
        checkout = Path(fields["worktree"])
        if checkout.resolve() != ROOT.resolve():
            checkouts.append((checkout, "worktree_" + fields["HEAD"]))
    roots = []
    for checkout, location in checkouts:
        paths = set((checkout / "runs").glob("level_b_gepa_real_canary*/local_gepa"))
        paths.update((checkout / "runs").glob("seed78_primary_responsibility_ab*/*/local_gepa"))
        roots.extend((path, checkout, location) for path in sorted(paths))
    entries, sources = [], {}
    for root, checkout, location in roots:
        def locator(path):
            return location + "/" + path.relative_to(checkout).as_posix()
        for candidate_path in sorted(root.glob("*/candidates.json")):
            match = re.fullmatch(r"seed(\d+)_update(\d+)_member(\d+)", candidate_path.parent.name)
            if not match:
                continue
            seed, update, member = map(int, match.groups())
            source = location + "/" + root.parent.relative_to(checkout / "runs").as_posix()
            candidates = read(candidate_path)
            original = candidates[0]
            prompt = original.get("decision_procedure", original.get("system_prompt"))
            sources[locator(candidate_path)] = file_hash(candidate_path)
            validation = []
            for output in sorted(candidate_path.parent.glob("generated_best_outputs_valset/task_*/iter_0_prog_0.json"),
                                 key=lambda p: int(p.parent.name.split("_")[-1])):
                value = read(output)
                validation.append(value["example_id"])
                sources[locator(output)] = file_hash(output)
            entries.append({"parent_task_id": source.replace("/", "_") + "_" + candidate_path.parent.name,
                            "source_run": source, "source_seed": seed, "update_index": update,
                            "target_member": member, "primary_responsibility_lane": None,
                            "parent_prompt_hash": text_hash(prompt),
                            "search_example_identity_hash": None,
                            "local_validation_identity_hash": digest(validation) if validation else None,
                            "optimization_context_hash": None,
                            "eligible": False,
                            "reasons": ["complete_ordered_search_evidence_and_tags_not_persisted",
                                        "frozen_optimization_context_not_persisted"]})
    registry = ROOT / "runs/responsibility_guided_gepa_fixed_parent_pilot_v1_prep_20260909_retry4/private_registry.json"
    if registry.exists():
        sources[registry.relative_to(ROOT).as_posix()] = file_hash(registry)
        for case in read(registry)["cases"]:
            # These are frozen historical analytical parents, but their source
            # uses a different/undeclared Solver contract and task construction.
            meta_path = ROOT / f"runs/vote_aligned_confirmatory_seed76_77_v1/seed{case['source_seed']}/P1_SHADOW_VOTE_ALIGNED_GENERIC/run_meta.json"
            meta = read(meta_path)
            sources[meta_path.relative_to(ROOT).as_posix()] = file_hash(meta_path)
            contract = meta["config"].get("solver_contract_id")
            entries.append({"parent_task_id": case["case_id"], "source_run": "vote_aligned_confirmatory_seed76_77_v1",
                            "source_seed": case["source_seed"], "update_index": case["source_update_index"],
                            "target_member": case["target_member"], "primary_responsibility_lane": case["responsibility_type"],
                            "parent_prompt_hash": text_hash(case["parent_prompts"][case["target_member"]]),
                            "search_example_identity_hash": None, "local_validation_identity_hash": None,
                            "optimization_context_hash": None, "source_solver_contract_id": contract,
                            "eligible": False,
                            "reasons": ["current_solver_contract_provenance_not_verified",
                                        "historical_task_definition_is_not_current_LocalOptimizationTask"]})
    return {"status": "PARENT_CATALOG_INSUFFICIENT", "verified_eligible_parent_count": 0,
            "inventory_count": len(entries), "entries": entries,
            "scope": "native canary and Seed78 roots in workspace and all registered execution worktrees; frozen seed76/77 analytical registry",
            "limitation": "Zero verified complete compatible tasks; this is not proof that reconstruction from additional private evidence is impossible.",
            "selection_uses_proposal_outcomes": False, "source_file_hashes": sources}


PROTOCOL_TEXT = """# Local-GEPA acceptance-rate pilot v1

Status: preparation for review; PARENT_CATALOG_INSUFFICIENT; no API authorization.
This is independent of both historical canary v2 and the authorized historical
level_b_gepa_local_acceptance_rate_pilot_v1. Neither authorization transfers.

The sole question is how often official Level-B GEPA produces a strict sampled
local improvement under the current responsibility-conditioned LocalOptimizationTask.
Canonical runtime remains member_aware_peer_state_v15 / checkpoint 25.

## Frozen measurement definitions

Primary strict_improvement_acceptance_rate is official GEPA accepted mutations
among contract-valid proposals changed from the exact selected parent that reach
empirical Solver evaluation, divided by those proposal events. Report numerator,
denominator, rate and Wilson 95% interval (z=1.959963984540054); zero denominator
is NA. Duplicate changed proposal events remain in the denominator; unchanged
events do not. Cache reuse is empirical evidence and counts as Solver reached;
provider_called remains a separate row flag. An acceptance outside this eligible
set must be disclosed in the funnel, not hidden. Pooled Wilson intervals are
descriptive binomial summaries; adaptively generated proposals and parents from
shared team states are not independent statistical replicates.

For the exact selected GEPA parent and ordered sampled minibatch, newly_fixed
counts 0->1 and newly_broken counts 1->0. delta_local_count=fixed-broken;
delta_local_rate=delta_local_count/batch size. Check sums against official strict
aggregate comparison; telemetry does not implement acceptance. Scores are binary.
Candidate indices, hashes and lineage follow evolved parents, never assume root.
Official ListDataLoader integer IDs are resolved through the ordered search set.

Each adapter batch records only evaluation_sequence_id, candidate_hash,
example_ids, binary_scores, provider_called flags, capture_traces, evidence_group
and reasoning_lane. Lifecycle events link these batches to parent/proposal hashes.
No texts, labels, raw traces, feedback or reflection input are exported. Native
GEPA state/logs remain private runtime artifacts and are never copied to reports.

token_edit_similarity_v1 casefolds Unicode text, extracts ordered ASCII
[a-z0-9]+ tokens, and uses difflib.SequenceMatcher(None,parent,proposal,
autojunk=False).ratio(). Persist only score, parent/proposal token counts and
absolute length delta; two empty sequences yield 1. Text exists only transiently.

Report fixed/broken counts separately for responsibility, coalition and
preservation. preservation_loss_count=preservation_newly_broken; no preservation
rows means NA (JSON null), never zero. No additional evaluations are introduced.
Failure pattern is (evidence_group,reasoning_lane) from the existing adapter
allowlist, only where selected-parent score is zero. Dominant ties resolve
lexicographically; share=max pattern count / parent_failure_count. Zero failures
means NA. Aggregate these occurrence counts over proposal-producing minibatches
per frozen parent task; report skip iterations separately. Also report pairwise
example-ID set Jaccard, fraction of proposal batches sharing a failure ID with
earlier proposal batches (first batch included in denominator), and unique failure
IDs. Repeated example occurrences within a minibatch count in scores/patterns;
sets are used only for overlap/reuse diagnostics.

## Capacity, stopping and budget

Conditional recommendation: four distinct compatible parent tasks, eight actual
on_proposal_end events each (32 total). Current eligible count is zero; no parent
selection or experimental capacity is frozen as runnable. Manifest records the
proposed 8-event quota and 16 skipped-iteration allowance explicitly, not as code
defaults. Compare 2x8, 4x8, 6x8. For V=12,m=3,N=8,S=16:
seed V + N*(2m+V) =156 metric calls, plus S*m=48 skip allowance, ceiling=205.
One call of headroom makes proposal stopping occur before metric stopping on the
bounded normal path. Official max_metric_calls remains set; a pre-batch guard
also prevents GEPA boundary overshoot. Metric calls count example evaluations,
not cache misses or provider attempts. At most N reflection logical calls.
Transport attempts are separately bounded by the existing contract retry cap.
No currency estimate is asserted without a frozen price schedule; token cost
depends on input sizes, with existing 1800 output-token caps unchanged.

Pinned MaxCandidateProposalsStopper counts loop iterations, including perfect
skips. Use project ExactProposalStopper through official stop_callbacks, observing
proposal events. Stop after N events or S skipped iterations; retain strict
skip_perfect_score=True. Perfect-forever paths cannot guarantee N under finite
budget. Any shortfall is PARENT_PROPOSAL_QUOTA_INCOMPLETE, with observed attempts,
shortfall and actual primary denominator. Do not replace missing attempts, resume,
change budgets, or declare a completed pilot estimate when any parent is incomplete.
Provider/protocol/integrity/persistence failures abort; no efficacy-based stopping.

## Parent governance

Eligibility requires an existing legitimate frozen task or exact zero-API
reconstruction with verified current Solver contract, ordered Optimize-only
search/validation rows, metadata, prompt, context and source provenance. Identical
task content is deduplicated across source updates. Historical compatible states
may be reused only with their exact problem definition; missing fields cannot be
inferred from sampled fragments. No prior GEPA outcome is an eligibility criterion.
Selection, once coverage exists, greedily maximizes the number of previously
unseen source seed/state pairs, then target members, then primary lanes; ties use
ascending (source seed, update index, target member, parent_task_id). Freeze exact
selected identities before search. Do not count renamed duplicates as independent
parents. At present selected_parents=[], PARENT_CATALOG_INSUFFICIENT.

## Isolation and fidelity

Backend is LEVEL_B_API_COMPATIBLE_ADAPTATION. Official pinned search core,
Pareto, selection, acceptance, reflection template/dataset, decision_procedure
contract, scoring, unit weights, skip-perfect, minibatch=3, models and splits stay
unchanged. No custom proposer or adapter propose_new_texts. No Layer-2 scheduler,
realizability, TeamMiniBatch, full-team evaluation, Common-Safe, Shadow or team
write-back is invoked by the pilot harness. local_optimizer_validation consists
only of Optimize evidence; heldout_Validation50=0 and Test50=0.

Report per-parent and pooled funnel: opportunities requested, attempts observed,
changed, unchanged, duplicate, contract invalid, Solver reached, strict-positive,
strict-equal, strict-negative sampled deltas, GEPA accepted and accepted-candidate
full local validation evaluated. Also parent-level >=1 accepted rate, accepted
generations and candidate lineage depths. Correlations are descriptive only.

## Preregistered interpretations

Materially positive acceptance supports only
LOCAL_GEPA_CAN_PRODUCE_STRICT_IMPROVEMENTS_UNDER_EXPANDED_SEARCH, never team transfer.
Near-zero acceptance with mostly zero deltas suggests LOW_BEHAVIORAL_EFFECT /
INEFFECTIVE_MUTATION; frequent fixes and breaks suggests REPAIR_PRESERVATION_TRADEOFF.
High text similarity and recurring failure patterns suggests
SEARCH_EXPLORATION_CONCENTRATION; diverse proposals with near-zero deltas suggests
LOCAL_TASK_OR_FEEDBACK_SIGNAL_LIMITATION. These qualitative diagnostic hypotheses
have no automatic thresholds or causal claims. Do not retrospectively invent a
confirmatory threshold. Follow-up changes need separate analysis/preregistration.

## Handoff

Real API calls=0; heldout_Validation50 calls=0; Test50 calls=0; formal pilot root
absent; API authorization=false; READY_TO_RUN=false. Complete a compatible parent
catalog, freeze exact selection and source commit, then obtain separate explicit
authorization before any real execution. Current runner provides a tested local
parent harness; its CLI fails closed until an executable frozen handoff exists.
Historical manifests remain immutable pre-run objects; existing run_lifecycle and
published canary audit/summary are completion evidence. Do not rewrite them.
"""


def prepare(*, junit: Path | None = None, checks: Path | None = None):
    catalog = parent_catalog()
    config = asdict(GEPAOptimizerConfig())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    REPORT.mkdir(parents=True, exist_ok=True)
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT / "PROTOCOL.md").write_text(PROTOCOL_TEXT, encoding="utf-8")
    protocol_hash = file_hash(EXPERIMENT / "PROTOCOL.md")
    historical = yaml.safe_load((ROOT / "experiments/manifests/level_b_gepa_real_canary_v2.yaml").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "experiment_manifest_v1", "experiment_id": IDENTITY,
        "title": "Local GEPA acceptance-rate preparation", "status": "DRAFT", "legacy_index": False,
        "lifecycle_history": [{"status": "DRAFT", "timestamp": "2026-09-18T00:00:00+08:00"}],
        "lineage": {"parents": ["level_b_gepa_real_canary_v2"], "derives_from": "level_b_gepa_real_canary_v2"},
        "scientific_question": "How often does official GEPA yield strict local improvements, and what paired behaviors explain them?",
        "hypotheses": ["Expanded local search may produce strict improvements; team transfer remains unevaluated."],
        "method_identity": METHOD_VERSION, "runtime_version": IDENTITY,
        "data": {**historical["data"], "formal": False, "validation_policy": "prohibited; heldout_Validation50 calls=0", "test_policy": "prohibited; Test50 calls=0"},
        "model": historical["model"], "seeds": [],
        "design": {"changed": ["observational proposal telemetry", "actual proposal event stopper"],
                   "unchanged": ["official GEPA search semantics", "local task definition", "models", "splits"],
                   "forbidden_changes": ["reflection template", "scheduler", "persistent realizability", "TeamMiniBatch", "Common-Safe", "Shadow", "team write-back"],
                   "attempt_id": IDENTITY + "_attempt1", "formal_run_root": "runs/" + IDENTITY + "_attempt1",
                   "selected_parents": [], "parent_catalog_status": catalog["status"],
                   "parent_catalog_identity": digest(catalog), "recommended_parent_count_if_eligible": 4,
                   "proposal_quota_per_parent": 8, "skipped_iteration_allowance": 16,
                   "selection_rule": "greedy_unseen_seed_state_then_member_then_lane_lexicographic_v1",
                   "official_gepa": config, "protocol_sha256": protocol_hash},
        "api_authorization": {"authorized": False, "authorization_scope": "pending_explicit_authorization",
                              "allowed_roles": [], "allowed_phases": []},
        "budget": {"type": "proposed_event_quota_with_hard_metric_ceiling", "frozen_before_run": False,
                   "limit": {"proposals_per_parent": 8, "max_metric_calls_per_parent": 205,
                             "skip_allowance": 16, "parent_count": None, "heldout_Validation50": 0, "Test50": 0}},
        "selection": {"primary_metric": "strict_improvement_acceptance_rate", "frozen_rule": "official accepted / contract-valid changed empirically evaluated proposal events; descriptive Wilson 95 percent interval",
                      "validation_used_for_selection": False, "test_used_for_selection": False},
        "artifacts": {"preregistration": {"path": (EXPERIMENT / "PROTOCOL.md").relative_to(ROOT).as_posix()},
                      "report": REPORT.relative_to(ROOT).as_posix(),
                      "provenance": (REPORT / "provenance.json").relative_to(ROOT).as_posix()},
        "git": {"design_commit": head, "implementation_commit": None, "result_commit": None},
        "result": {"classifier": "PARENT_CATALOG_INSUFFICIENT", "conclusion": "Preparation only; no local effectiveness estimate.", "evidence_type": "not_yet_available"},
    }
    manifest["artifacts"]["preregistration"]["sha256"] = preregistration_hash(manifest)
    MANIFEST.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    handoff = {"schema_version": "sol_luna_experiment_handoff_v1", "experiment_id": IDENTITY,
               "source": {"base_commit": head, "repository_commit": None, "tracked_worktree_status": "preparation_changes_not_committed"},
               "protocol": {"name": IDENTITY, "version": 1, "protocol_sha256": protocol_hash,
                            "manifest_path": MANIFEST.relative_to(ROOT).as_posix(), "manifest_sha256": file_hash(MANIFEST),
                            "preregistration_path": (EXPERIMENT / "PROTOCOL.md").relative_to(ROOT).as_posix(),
                            "preregistration_sha256": preregistration_hash(manifest)},
               "data": {"split_identity": historical["data"]["split_ids"], "selected_parent_ids": []},
               "models": historical["model"], "execution": {"exact_runner_command": "python scripts/run_local_gepa_acceptance_rate_pilot.py --execute",
                   "expected_output_directory": manifest["design"]["formal_run_root"], "seeds": [], "opportunity_budget": None,
                   "early_stop_rule": "N actual events; S skips; hard metric ceiling", "allowed_retries": "existing transport only; no experiment resume"},
               "authorization": {"api_scope": "none", "validation_access_policy": "prohibited", "test_access_policy": "prohibited"},
               "fail_closed_conditions": ["missing compatible frozen parents", "missing execution commit", "hash mismatch", "missing explicit authorization", "provider or ledger or split failure"],
               "READY_TO_RUN": False}
    (EXPERIMENT / "EXPERIMENT_HANDOFF.md").write_text("# Preparation handoff — NOT EXECUTABLE\n\nNull execution fields explicitly mark the blocked freeze.\n\n```yaml\n" + yaml.safe_dump(handoff, sort_keys=False) + "```\n", encoding="utf-8")
    summary = read(CANARY / "summary.json")
    write(REPORT / "current_evidence_summary.json", {"empirical_path": summary["classifier"],
          "local_effectiveness": "LOCAL_EFFECTIVENESS_UNRESOLVED", "team_transfer": "TEAM_TRANSFER_NOT_EVALUATED",
          "accepted_mutations": summary["accepted_mutations"], "empirically_evaluated_proposals": summary["proposer_diagnostics"]["solver_reached"],
          "source": CANARY.relative_to(ROOT).as_posix(), "historical_summary_sha256": file_hash(CANARY / "summary.json"),
          "bookkeeping_decision": "preserve frozen historical manifests; existing run_lifecycle plus published audit and summary are completion records"})
    write(REPORT / "parent_catalog.json", catalog)
    write(REPORT / "parent_selection_audit.json", {"status": catalog["status"], "selected_parent_ids": [],
          "verified_eligible_count": 0, "rule": manifest["design"]["selection_rule"], "used_prior_proposal_outcomes": False})
    write(REPORT / "proposal_telemetry_schema.json", {"schema_version": LOCAL_GEPA_PROPOSAL_TELEMETRY_VERSION,
          "evaluation_batch_schema": evaluation_schema(),
          "evaluation_fields": sorted(EVALUATION_FIELDS), "strict_row_allowlist": True,
          "missing_values": "JSON null", "edit_similarity": "token_edit_similarity_v1",
          "example_id_mapping": "ListDataLoader positional IDs resolved to ordered search examples",
          "patterns": ["evidence_group", "reasoning_lane"], "pattern_filter": "selected parent binary score == 0"})
    write(REPORT / "proposal_count_stopper_audit.json", {"official_builtin_counts_iterations": True,
          "integration": "official stop_callbacks structural StopperProtocol", "counter": "actual on_proposal_end",
          "normal_exact_quota": True, "perfect_forever_guarantee": False,
          "shortfall_status": "PARENT_PROPOSAL_QUOTA_INCOMPLETE", "skip_allowance": 16,
          "metric_guard": "pre-batch hard ceiling; no partial batch evaluation"})
    envelope = cost_envelope(8, 12, 3, 16)
    from infrastructure.common_solver_contract_v1.contract import CONTRACT_SPEC
    options = [{"parents": count, "proposals_per_parent": 8, "total_proposals": count * 8,
                "per_parent": envelope, "total_metric_ceiling": count * envelope["metric_ceiling"],
                "total_reflection_logical_calls": count * 8,
                "solver_transport_attempt_upper_bound": count * envelope["metric_ceiling"] * CONTRACT_SPEC.transport_attempt_cap,
                "solver_output_token_upper_bound": count * envelope["metric_ceiling"] * 1800,
                "reflection_output_token_upper_bound": count * 8 * 1800,
                "coverage_feasible_now": False} for count in (2, 4, 6)]
    write(REPORT / "budget_capacity_options.json", {"options": options, "conditional_recommendation": "4x8",
          "current_recommendation": "do not execute; resolve parent provenance first", "currency_cost": None,
          "input_token_cost": "not computable until compatible exact parent tasks are available",
          "output_bounds_exclude_retried_or_unreported_failed_attempt_usage": True})
    tests = {"status": "NOT_RECORDED"}
    if junit:
        root = ET.parse(junit).getroot()
        suites = list(root.iter("testsuite"))
        tests = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
        tests["status"] = "PASS" if tests["failures"] == tests["errors"] == 0 else "FAIL"
        tests["junit_sha256"] = file_hash(junit)
    check_results = read(checks) if checks else {"status": "NOT_RECORDED"}
    write(REPORT / "test_summary.json", {"pytest": tests, "checks": check_results})
    write(REPORT / "fake_provider_metric_audit.json", {"evidence": "tests/test_local_gepa_acceptance_rate_preparation.py",
          "suite_status": check_results.get("focused_tests", {}).get("status", "NOT_RECORDED"),
          "observed_smokes": check_results.get("observed_smokes", {}),
          "cases": ["fix", "break", "fix_and_break", "equal", "preservation_only_break", "evolved_parent", "duplicate", "contract_invalid", "perfect_skip", "exact_proposal_stopping", "metric_shortfall", "perfect_forever_shortfall", "row_alignment", "cached_empirical_evidence", "Wilson", "ordered_token_similarity", "full_validation_budget_interrupt", "callback_persistence_failure"],
          "real_provider_calls": 0})
    write(REPORT / "level_b_fidelity_audit.json", {"engine": verify_frozen_gepa(), "config": config,
          "fidelity_level": config["optimizer_fidelity_level"], "search_core_changed": False,
          "custom_candidate_proposer": False, "adapter_propose_new_texts": False,
          "supported_seams": ["callbacks", "adapter observation", "stop_callbacks"], "two_layer_search_invoked": False})
    isolation = {"real_api_calls": 0, "heldout_Validation50_calls": 0, "Test50_calls": 0,
                 "formal_pilot_run_root_absent": not (ROOT / manifest["design"]["formal_run_root"]).exists(),
                 "api_authorized": False, "team_minibatch_calls": 0, "full_team_calls": 0,
                 "common_safe_calls": 0, "shadow_calls": 0, "team_commits": 0}
    write(REPORT / "validation_test_isolation.json", isolation)
    write(REPORT / "protocol_identity.json", {"experiment_id": IDENTITY, "attempt_id": manifest["design"]["attempt_id"],
          "protocol_sha256": protocol_hash, "manifest_sha256": file_hash(MANIFEST),
          "preregistration_sha256": preregistration_hash(manifest), "state": "DRAFT_PARENT_CATALOG_INSUFFICIENT", "READY_TO_RUN": False})
    source_paths = [*sorted((ROOT / "multi_dataset_diverse_rl/local_optimizers").glob("*.py")),
                    ROOT / "multi_dataset_diverse_rl/versions.py",
                    ROOT / "scripts/run_local_gepa_acceptance_rate_pilot.py", Path(__file__),
                    ROOT / "tests/test_local_gepa_acceptance_rate_preparation.py", MANIFEST,
                    EXPERIMENT / "PROTOCOL.md", EXPERIMENT / "EXPERIMENT_HANDOFF.md"]
    write(REPORT / "provenance.json", {"base_commit": head, "evidence_type": "zero_api_preparation",
          "source_files": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in source_paths},
          "canary_files": {p.name: file_hash(p) for p in CANARY.iterdir() if p.is_file()},
          "historical_evidence_modified": False})
    write(REPORT / "fact_assertions.json", {**isolation, "official_core_unchanged": True,
          "catalog_coverage_sufficient": False, "execution_freeze_complete": False,
          "measurement_implementation_complete": check_results.get("focused_tests", {}).get("status") == "PASS",
          "no_new_test_failures": check_results.get("no_new_test_failures", False)})
    (REPORT / "README.md").write_text(f"""# Local-GEPA acceptance-rate pilot preparation

**PARENT_CATALOG_INSUFFICIENT. API authorization=false; READY_TO_RUN=false.**

1. The Level-B empirical path is confirmed by immutable transportfix1 evidence.
2. Its 0/3 accepted proposals do not resolve local effectiveness or team transfer.
3. Catalog inventories {catalog['inventory_count']} records; **0 complete compatible parent tasks are verified**.
   Native artifacts lack the complete ordered search problem/context; six older
   analytical snapshots lack verified current Solver-contract provenance. This is
   an evidence limit, not proof that additional private reconstruction is impossible.
4. Recommend **4x8 conditionally**, not as a runnable freeze. 2x8/4x8/6x8 cost options are included.
5. Official-engine fake-provider tests exercise exact event quotas for normal,
   accepted, rejected and mixed-perfect paths. Perfect-forever or safety termination
   produces an explicit shortfall; an unconditional finite-budget guarantee is impossible.
6. Per-parent normal bound is 156 metric evaluations, 204 including 16 skips;
   proposed hard ceiling is 205. Pre-batch guard prevents overshoot.
7. Ordered adapter score vectors and callback links reconstruct exact fixed/broken counts.
8. Preservation loss uses existing sampled rows; absent preservation is NA.
9. SequenceMatcher token similarity exports only numeric summaries; no texts.
10. Concentration uses allowlisted group/lane metadata of failed parent rows only.
11. The parent harness invokes only Layer 1; all team evaluation and write-back stages are absent.
12. Real API, heldout_Validation50 and Test50 calls are exactly zero; no formal run root exists.
13. Pinned official GEPA source integrity is verified; search core and problem definition are unchanged.
14. A separate **draft** protocol, manifest and blocked handoff are reviewable.
    An executable preregistration is **not ready**: parent identities and execution
    source commit remain unfrozen. The CLI intentionally rejects execution. The
    tested local-parent harness is ready for integration once those inputs exist.

Historical authorized 3x4 pilot and canary v2 remain untouched. The repository
convention separates immutable preregistration from runtime lifecycle/completion
evidence; published canary audit/summary already provide that record.

See test_summary.json for actual validation results, parent_catalog.json for
source identities and explicit exclusions, and protocol_identity.json for hashes.
No scientific efficacy estimate or causal interpretation was produced.

Validation: focused suite {check_results.get('focused_tests', {}).get('status', 'NOT_RECORDED')}.
Full repository tests: {tests.get('tests', 0)} collected, {tests.get('failures', 0)} failures,
{tests.get('errors', 0)} errors. New failures: {check_results.get('new_failure_count', 'NOT_RECORDED')}.
The existing historical v16 cache-coverage test also fails with HEAD versions of
the modified runtime modules; its historical inputs were not repaired or rewritten.
An initial unscoped pytest invocation collected archived tests under runs and
hit import-name collisions; the full repository suite is explicitly tests/.
""", encoding="utf-8")
    findings = scan_sanitized_artifacts(REPORT)
    write(REPORT / "sanitization_manifest.json", {"status": "PASS" if not findings else "FAIL", "findings": findings})
    write(REPORT / "sha256_manifest.json", build_sha256_manifest(REPORT))
    if findings:
        raise ValueError("preparation sanitization failed")
    return {"status": catalog["status"], "inventory_count": catalog["inventory_count"], "verified_eligible_count": 0, "api_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--checks", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(junit=args.junit, checks=args.checks), indent=2))
