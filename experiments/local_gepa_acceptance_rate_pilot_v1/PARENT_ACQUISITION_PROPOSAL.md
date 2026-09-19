# Parent evidence acquisition proposal — scope change pending

The user has authorized API calls for the Local-GEPA pilot and instructed us to
continue. That authorization is recorded separately and does not need repeating.
The existing preparation report and its manifest hashes remain unchanged.

The remaining scientific scope issue is original task section 14:

> No API calls may be used merely to create additional parents in this task.

Current inventory has no verified complete compatible LocalOptimizationTask.
Native canary artifacts do not retain the full ordered search evidence/context.
The six older analytical snapshots do not establish the current Solver-contract
provenance. Additional Common Solver replay artifacts are explicitly held-out
validation evidence and are excluded from parent construction and selection.

## Concrete proposed extension

Create an independent `local_gepa_parent_snapshot_acquisition_v1` phase before
the acceptance pilot. It performs only Optimize100 baseline acquisition using
the unchanged shared P0 prompt from the frozen Seed78 canary, the frozen
COMMON_SOLVER_CONTRACT_V1 request and parser, qwen3-8b, temperature zero, thinking
disabled, and solver_max_tokens=1800. No reflection, optimization, held-out
evaluation, candidate acceptance or team write-back occurs in this phase.

Freeze seed 78 and one initial team before calls. First verify all five P0 prompt
hashes are identical. Evaluate the 100 Optimize examples once per exact request
and reuse each realization across the five identical members. This is at most
500 member/example logical evaluations, 100 distinct successful provider
requests and 400 provider attempts under the existing four-attempt transport
cap. Successful-response output is bounded by 180,000 tokens; input-token and
failed-attempt costs must be included in the final execution budget. Credentials
and endpoints are never written to publishable artifacts.

Persist complete baseline profiles and request evidence privately. Apply the
unchanged responsibility calculation and current LocalTaskBuilder offline to
construct all five member tasks. Validate Optimize-only membership, ordered
search/validation rows, group/lane tags, prompt/context bytes and hashes, and the
existing local-validation quotas. This construction does not evaluate
TeamMiniBatch or call any team acceptance stage.

Deduplicate task identities and apply the preregistered deterministic parent
selection rule without generating or observing any pilot proposals. If fewer
than four compatible distinct tasks exist, stop; do not acquire additional
states or seeds opportunistically. Otherwise freeze exactly four identities
before beginning the separately authorized Layer-1 pilot.

These four tasks share one source team state; they are not independent team-state
replicates. The report must disclose that coverage limitation. The proposed
pilot remains 4x8 actual proposal attempts, at most 205 metric evaluations per
parent (820 total), 32 reflection logical calls, and the existing transport
retry policy. Acquisition costs are separately reported, never hidden in the
pilot acceptance denominator or local-optimization cost.

## Required implementation and freeze

Before any API call, implement and test the acquisition command and the real
provider binding for the current parent harness; freeze source, split/request
identities, private snapshot schema, exact budgets, failure rules, attempt root
and handoff. Provider, hash, ledger, parser/protocol or split-governance failures
stop and preserve evidence. No automatic experiment retry or resume is allowed.

This proposal is reviewable but is not an executable handoff. Its sole unresolved
scope decision is whether to permit the new Optimize-only parent-acquisition
phase despite the original section 14 restriction. An existing complete frozen
task directory remains a zero-acquisition alternative.

Heldout_Validation50 and Test50 calls remain zero in both phases. Historical
canary runs, reports and preregistration are not modified.
