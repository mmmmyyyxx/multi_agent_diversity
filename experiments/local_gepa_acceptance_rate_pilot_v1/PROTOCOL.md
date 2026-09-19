# Local-GEPA acceptance-rate pilot v1

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
